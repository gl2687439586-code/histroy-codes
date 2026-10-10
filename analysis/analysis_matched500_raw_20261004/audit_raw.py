"""Offline matched500 raw audit. No generation, network, or source mutations."""
import gzip
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import string
import sys
import unicodedata

import numpy as np

BASE = Path(__file__).resolve().parent
WORK = BASE.parent
INPUT, OUT = BASE / 'input', BASE / 'results'
SUPPORT = BASE / 'support'
CODE = SUPPORT / 'code'
DATA = SUPPORT / 'data'
COMMON = [str(x) for x in range(100, 601, 100)] + ['FULL']
sys.path.insert(0, str(BASE / 'vendor'))
from tokenizers import Tokenizer
TOKENIZER_PATH = SUPPORT / 'tokenizer.json'
TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


prompts = load_module('raw_audit_prompts', CODE / 'prompts.py')
validate = load_module('raw_export_validate', SUPPORT / 'export_hotpot_raw.py')
paired = load_module('matched_metric_methods', SUPPORT / 'audit_matched_metric_transfers.py')


def sha(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def canonical(x):
    return sha(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')))


def rows(path):
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def normalize(s):
    s = ''.join(c for c in s.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())


def scores(prediction, golds):
    p, gs = normalize(prediction), [normalize(g) for g in golds]
    em = float(p in gs)
    acc = float(any(g and g in p for g in gs))
    f1 = 0.0
    for g in gs:
        overlap = sum((Counter(p.split()) & Counter(g.split())).values())
        if overlap:
            f1 = max(f1, 2.0 * overlap / (len(p.split()) + len(g.split())))
    return {'answer_em': em, 'answer_f1': f1, 'answer_acc': acc}


def budget(row):
    return 'FULL' if row['exposure_mode'] == 'full_per_search' else str(row['root_exposure_tokens'])


def finish(a):
    r = a['finish_reason']
    return r.get('type') if isinstance(r, dict) else r


def hist(counter, key, value):
    counter[key][str(value)] += 1


def docs_hash(ds):
    return canonical([sha(d['contents']) for d in ds])


def ids_hash(text):
    return canonical(TOKENIZER.encode(text, add_special_tokens=False).ids)


def token_check(a, checks, warnings):
    for field, text in [('input_tokens', a['prompt']), ('output_tokens', a['output'])]:
        n = len(TOKENIZER.encode(text, add_special_tokens=False).ids)
        checks['reference_tokenizer_' + field + '_checked'] += 1
        warnings['reference_tokenizer_' + field + '_mismatches'] += int(n != a[field])


def audit_run(label, labels, questions):
    run = INPUT / label
    verified, vroots, vpairs = validate.inspect_run(run, 500)
    roots = {r['sample_id']: r for r in rows(run / 'roots/roots.merged.jsonl.gz')}
    supplied = {(r['sample_id'], r['condition_id']): r for r in rows(run / 'analysis/scored_records.jsonl.gz')}
    assert set(roots) == set(labels) == set(questions)
    checks, warnings, settings, examples = Counter(), Counter(), defaultdict(Counter), defaultdict(list)
    compact, seen = {}, set()
    for sid, r in roots.items():
        assert r['question'] == questions[sid]['question']
        for a in r['action_attempts']:
            token_check(a, checks, warnings)
            assert a['prompt'] == prompts.initial_search_prompt(r['question']), (sid, 'root template mismatch')
            checks['root_prompts_rebuilt_exactly'] += 1
            hist(settings, 'root_max_new_tokens', a['max_new_tokens'])
        for key in ['temperature', 'thinking_control', 'top_k', 'model_path', 'model_url']:
            hist(settings, 'root_' + key, r[key])
        assert r['gold_labels_loaded'] is False
    for row in rows(run / 'trajectories/trajectories.merged.jsonl.gz'):
        sid, cid, b = row['sample_id'], row['condition_id'], budget(row)
        key = (sid, cid)
        assert key not in seen and key in supplied
        seen.add(key)
        score = supplied[key]
        root = roots[sid]
        assert row['question'] == root['question']
        assert row['root_query'] == root['query']
        assert row['gold_labels_loaded'] is False
        rs, events = row['search_rounds'], row['decision_events']
        assert row['search_calls'] == len(rs) == len(events)
        assert 1 <= len(rs) <= row['max_searches'] == 4
        assert rs[0]['query'] == root['query']
        assert docs_hash(rs[0]['evidence']['retrieved_docs']) == docs_hash(root['documents'])
        assert row['usage']['format_repair_calls'] == 0
        for k in ['temperature', 'thinking_control', 'top_k_per_search', 'max_searches', 'schema_version']:
            hist(settings, k, row[k])
        attempts, evmeta, evidence_meta = [], [], []
        prefix_flags, boundary_versions = [], []
        for j, event in enumerate(events):
            completed, retained = event['searches_completed_before_decision'], event['history_rounds_retained']
            assert completed == j + 1
            assert event['context_compacted'] == (retained < completed)
            selected = rs[:completed][-retained:]
            expected = prompts.forced_answer_prompt(row['question'], selected) if event['forced_answer'] else prompts.decision_prompt(row['question'], selected, 4 - completed)
            assert event['forced_answer'] == (completed == 4)
            for a in event['attempts']:
                token_check(a, checks, warnings)
                assert a['prompt'] == expected, (label, sid, b, j, 'prompt mismatch')
                checks['continuation_prompts_rebuilt_exactly'] += 1
                hist(settings, 'continuation_max_new_tokens', a['max_new_tokens'])
                hist(settings, 'stop_boundary_repair', a['stop_boundary_repair'])
                if a['stop_boundary_repair'] == 'appended_full_stop':
                    assert a['output'] == a['raw_output'] + a['finish_reason']['matched']
                elif a['stop_boundary_repair'] == 'none':
                    assert a['output'] == a['raw_output']
                else:
                    suffix = a['output'][len(a['raw_output']):]
                    assert a['output'].startswith(a['raw_output']) and suffix
                attempts.append(a)
            last = event['attempts'][-1]
            if event['action'] in ['search', 'answer']:
                action_matches = list(re.finditer(r'<%s>\s*(.*?)\s*</%s>' % (event['action'], event['action']), last['output'], re.I | re.S))
                if not any(' '.join(m.group(1).split()) == event['payload'] for m in action_matches):
                    warnings['parsed_action_without_matching_closed_tag'] += 1
                matching = [m for m in action_matches if ' '.join(m.group(1).split()) == event['payload']]
                if matching and last['output'].rfind('<think>', 0, matching[-1].start()) > last['output'].rfind('</think>', 0, matching[-1].start()):
                    warnings['action_tag_inside_unclosed_think'] += 1
                    warnings[event['action'] + '_tag_inside_unclosed_think'] += 1
                    if len(examples['action_tag_inside_unclosed_think']) < 4:
                        examples['action_tag_inside_unclosed_think'].append({'sample_id': sid, 'budget': b, 'action': event['action'], 'raw_tail': last['raw_output'][-1500:]})
            evmeta.append({'prompt_hash': sha(last['prompt']), 'prompt_nfc_hash': sha(unicodedata.normalize('NFC', last['prompt'])),
                           'input_ids_hash': ids_hash(last['prompt']),
                           'output_hash': sha(last['raw_output']), 'action': event['action'], 'payload': event['payload'],
                           'finish': finish(last), 'input_tokens': last['input_tokens'], 'output_tokens': last['output_tokens']})
            if j < len(events) - 1:
                assert event['action'] == 'search' and event['payload'] == rs[j + 1]['query']
            else:
                assert (event['action'] == 'answer') == (row['answer_status'] == 'valid')
                if row['answer_status'] == 'valid':
                    assert event['payload'] == row['final_answer']
        for rnd in rs:
            ev = rnd['evidence']
            for f, hf in [('evidence_text', 'evidence_text_sha256'), ('visible_context_text', 'visible_context_sha256')]:
                assert sha(ev[f]) == ev[hf]
            text = ev.get('context_separator', '\n\n').join(d['contents'] for d in ev['retrieved_docs'])
            assert sha(text) == ev['full_context_sha256']
            assert len(ev['retrieved_docs']) == 5
            actual = ev['actual_exposed_token_count']
            full_ids = TOKENIZER.encode(text, add_special_tokens=False).ids
            visible_ids = TOKENIZER.encode(ev['visible_context_text'], add_special_tokens=False).ids
            checks['reference_evidence_token_prefix_checks'] += 1
            warnings['reference_evidence_not_exact_token_prefix'] += int(visible_ids != full_ids[:actual])
            warnings['reference_evidence_token_count_mismatches'] += int(len(visible_ids) != actual)
            assert actual == ev['visible_evidence_tokens'] == ev['visible_context_reencoded_token_count']
            assert actual <= ev['full_context_token_count'] and actual <= ev['requested_token_cap']
            strict_prefix = text.startswith(ev['visible_context_text'])
            nfc_prefix = unicodedata.normalize('NFC', text).startswith(unicodedata.normalize('NFC', ev['visible_context_text']))
            prefix_flags.append(not strict_prefix)
            version = ev.get('prefix_boundary_policy', 'not_recorded_legacy')
            boundary_versions.append(version)
            hist(settings, 'evidence_boundary_policy', version)
            checks['evidence_rounds'] += 1
            warnings['visible_not_exact_original_text_prefix_rounds'] += int(not strict_prefix)
            warnings['visible_not_even_nfc_original_text_prefix_rounds'] += int(not nfc_prefix)
            warnings['rounds_with_boundary_backoff'] += int(ev.get('roundtrip_boundary_backoff_tokens', 0) > 0)
            if not strict_prefix and len(examples['nonliteral_prefix']) < 3:
                examples['nonliteral_prefix'].append({'sample_id': sid, 'budget': b, 'search_round': rnd['search_round'],
                    'full_original_head': text[:700], 'visible_head': ev['visible_context_text'][:700], 'nfc_prefix': nfc_prefix})
            evidence_meta.append({'query': rnd['query'], 'docs_hash': docs_hash(ev['retrieved_docs']),
                'visible_hash': sha(ev['visible_context_text']), 'evidence_hash': sha(ev['evidence_text']),
                'boundary_policy': version, 'actual_tokens': actual})
        assert sum(r['evidence']['visible_evidence_tokens'] for r in rs) == row['visible_evidence_tokens_total']
        u = row['usage']
        for which in ['input', 'output']:
            continuations = sum(a[which + '_tokens'] for a in attempts)
            ru = sum(a[which + '_tokens'] for a in root['action_attempts'])
            assert u['continuation_llm_' + which + '_tokens'] == continuations
            assert u['shared_root_llm_' + which + '_tokens'] == ru
            assert u['standalone_llm_' + which + '_tokens'] == continuations + ru
            assert score['standalone_llm_' + which + '_tokens'] == continuations + ru
        assert score['search_calls'] == row['search_calls']
        assert score['visible_evidence_tokens_total'] == row['visible_evidence_tokens_total']
        raw_scores = scores(row['final_answer'], labels[sid]['gold_answers'])
        calculated = {k: 0.0 if row['answer_status'] == 'local_fallback' else v for k, v in raw_scores.items()}
        for field, value in calculated.items():
            assert abs(float(score[field]) - value) < 1e-12, (label, sid, b, field, score[field], value)
        checks['quality_scores_recomputed_exactly'] += 1
        checks['usage_sums_checked'] += 1
        compact[(sid, b)] = {'sample_id': sid, 'budget': b, 'created_at': row['created_at'],
            'final_answer': row['final_answer'], 'answer_status': row['answer_status'],
            'scores': calculated, 'input_tokens': u['standalone_llm_input_tokens'],
            'output_tokens': u['standalone_llm_output_tokens'],
            'total_tokens': u['standalone_llm_input_tokens'] + u['standalone_llm_output_tokens'],
            'search_calls': row['search_calls'], 'visible_tokens': row['visible_evidence_tokens_total'],
            'any_length': any(finish(a) == 'length' for a in attempts),
            'root_length': any(finish(a) == 'length' for a in root['action_attempts']),
            'compacted': row['context_compactions'] > 0, 'events': evmeta, 'evidence': evidence_meta,
            'nonliteral_prefix': any(prefix_flags), 'boundary_policies': sorted(set(boundary_versions))}
    assert seen == set(supplied)
    for sid, r in roots.items():
        expected_conditions = {c['condition_id'] for c in r['exposure_conditions']}
        assert {cid for s, cid in seen if s == sid} == expected_conditions
    return roots, compact, {'verified': verified, 'checks': dict(checks), 'warnings': dict(warnings),
                            'settings': {k: dict(v) for k, v in settings.items()}, 'examples': dict(examples)}


def mean(rs):
    return {'n': len(rs), **{m: float(np.mean([r['scores'][m] for r in rs])) for m in ['answer_em', 'answer_f1', 'answer_acc']},
            **{'mean_' + k: float(np.mean([r[k] for r in rs])) for k in ['input_tokens', 'output_tokens', 'total_tokens', 'search_calls', 'visible_tokens']},
            'fallbacks': sum(r['answer_status'] != 'valid' for r in rs), 'length_terminated_trajectories': sum(r['any_length'] for r in rs)}


def select(group, sid, choices, metric):
    return min(choices, key=lambda b: (-group[sid, b]['scores'][metric], group[sid, b]['total_tokens'], (1, 0) if b == 'FULL' else (0, int(b))))


def root_comparison(old, new):
    byid = {}
    for sid, a in old.items():
        b = new[sid]
        byid[sid] = {'sample_id': sid, 'question': a['question'], 'old_query': a['query'], 'new_query': b['query'],
            'prompt_same': a['action_attempts'][0]['prompt'] == b['action_attempts'][0]['prompt'],
            'raw_output_same': a['action_attempts'][0]['raw_output'] == b['action_attempts'][0]['raw_output'],
            'query_same': a['query'] == b['query'], 'docs_same': docs_hash(a['documents']) == docs_hash(b['documents']),
            'old_length': any(finish(x) == 'length' for x in a['action_attempts']),
            'new_length': any(finish(x) == 'length' for x in b['action_attempts']),
            'old_output_tokens': sum(x['output_tokens'] for x in a['action_attempts']),
            'new_output_tokens': sum(x['output_tokens'] for x in b['action_attempts'])}
    return byid


def pair_detail(a, b, root):
    ea, eb = a['events'], b['events']
    first_divergence = 'all_events_equal'
    first_round = None
    for j, (x, y) in enumerate(zip(ea, eb)):
        if x['prompt_hash'] != y['prompt_hash']:
            first_divergence, first_round = 'different_prompt', j + 1
            break
        if x['output_hash'] != y['output_hash']:
            first_divergence, first_round = 'same_prompt_different_raw_output', j + 1
            break
    if first_divergence == 'all_events_equal' and len(ea) != len(eb):
        first_divergence, first_round = 'different_event_count', min(len(ea), len(eb)) + 1
    return {'sample_id': a['sample_id'], 'budget': a['budget'],
            'first_prompt_same': ea[0]['prompt_hash'] == eb[0]['prompt_hash'],
            'first_prompt_same_nfc': ea[0]['prompt_nfc_hash'] == eb[0]['prompt_nfc_hash'],
            'first_input_ids_same_reference_tokenizer': ea[0]['input_ids_hash'] == eb[0]['input_ids_hash'],
            'first_output_same': ea[0]['output_hash'] == eb[0]['output_hash'],
            'first_evidence_same': a['evidence'][0]['evidence_hash'] == b['evidence'][0]['evidence_hash'],
            'first_divergence': first_divergence, 'first_divergence_round': first_round,
            'root_query_same': root['query_same'], 'root_docs_same': root['docs_same'],
            'old_length': a['any_length'], 'new_length': b['any_length'],
            'either_root_length': root['old_length'] or root['new_length'],
            'old_fallback': a['answer_status'] != 'valid', 'new_fallback': b['answer_status'] != 'valid',
            'either_compacted': a['compacted'] or b['compacted'],
            'old_nonliteral_prefix': a['nonliteral_prefix'], 'new_nonliteral_prefix': b['nonliteral_prefix'],
            'boundary_policy_changed': a['boundary_policies'] != b['boundary_policies'],
            'old_answer': a['final_answer'], 'new_answer': b['final_answer']}


def metric_audit(old, new, roots, metric, choices):
    details, oldrs, transferrs, newrs, fullrs = [], [], [], [], []
    for sid in sorted(roots):
        opts = choices[sid]
        bo, bn = select(old, sid, opts, metric), select(new, sid, opts, metric)
        a, b, n, f = old[sid, bo], new[sid, bo], new[sid, bn], new[sid, 'FULL']
        d = pair_detail(a, b, roots[sid])
        d.update(old_budget=bo, new_budget=bn, old_quality=a['scores'][metric], transfer_quality=b['scores'][metric],
                 new_oracle_quality=n['scores'][metric], new_full_quality=f['scores'][metric],
                 old_correct=a['scores'][metric] == 1, transfer_correct=b['scores'][metric] == 1,
                 new_oracle_correct=n['scores'][metric] == 1, full_correct=f['scores'][metric] == 1,
                 old_total_tokens=a['total_tokens'], transfer_total_tokens=b['total_tokens'],
                 new_oracle_total_tokens=n['total_tokens'], full_total_tokens=f['total_tokens'])
        details.append(d)
        oldrs.append(a); transferrs.append(b); newrs.append(n); fullrs.append(f)
    losses = [d for d in details if d['old_correct'] and not d['transfer_correct']]
    grouped = {}
    for flag in ['first_prompt_same', 'root_query_same', 'root_docs_same', 'either_root_length', 'new_length',
                 'first_input_ids_same_reference_tokenizer', 'new_fallback', 'boundary_policy_changed', 'old_nonliteral_prefix', 'either_compacted']:
        grouped[flag] = {}
        for val in [False, True]:
            subset = [r for r in details if r[flag] == val]
            before = sum(r['old_correct'] for r in subset)
            loss = sum(r['old_correct'] and not r['transfer_correct'] for r in subset)
            grouped[flag][str(val)] = {'n': len(subset), 'old_correct': before, 'losses': loss,
                                      'loss_rate_given_old_correct': loss / before if before else None}
    delta = [d['transfer_quality'] - d['new_full_quality'] for d in details]
    result = {'metric': metric, 'binary_correct_definition': 'selected/evaluated metric equals 1; F1==1 is not necessarily EM==1',
        'n': len(details), 'old_oracle': mean(oldrs), 'new_at_old_budget': mean(transferrs),
        'new_oracle': mean(newrs), 'new_FULL': mean(fullrs),
        'old_to_new': paired.binary_matrix([d['old_correct'] for d in details], [d['transfer_correct'] for d in details]),
        'full_to_transfer': paired.binary_matrix([d['full_correct'] for d in details], [d['transfer_correct'] for d in details]),
        'transfer_minus_FULL': {'quality_delta': float(np.mean(delta)), 'quality_delta_ci95': paired.paired_ci(delta)[0],
            'mean_total_token_delta': float(np.mean([d['transfer_total_tokens'] - d['full_total_tokens'] for d in details])),
            'total_token_delta_ci95': paired.paired_ci([d['transfer_total_tokens'] - d['full_total_tokens'] for d in details])[0],
            'savings_fraction': 1 - np.mean([d['transfer_total_tokens'] for d in details]) / np.mean([d['full_total_tokens'] for d in details])},
        'lost_old_successes': len(losses), 'losses_with_new_other_budget_correct': sum(d['new_oracle_correct'] for d in losses),
        'losses_without_new_correct_budget': sum(not d['new_oracle_correct'] for d in losses),
        'losses_first_divergence': dict(Counter((d['first_divergence'] + '_round' + str(d['first_divergence_round'])) for d in losses)),
        'loss_flags_nonexclusive': {flag: sum(d[flag] for d in losses) for flag in grouped},
        'groups': grouped, 'selected_budget_same': sum(d['old_budget'] == d['new_budget'] for d in details)}
    return result, details


def main():
    OUT.mkdir(exist_ok=True)
    config = json.loads((INPUT / 'source_metadata/new500/FULL7405_CONFIG.json').read_text(encoding='utf-8'))
    assert validate.digest_file(CODE / 'prompts.py') == config['package_hashes']['prompts.py']
    labels = {r['sample_id']: r for r in rows(DATA / 'gold_labels.sealed.jsonl')}
    questions = {r['sample_id']: r for r in rows(DATA / 'questions.no_labels.jsonl')}
    oldroots, old, oldreport = audit_run('old500', labels, questions)
    newroots, new, newreport = audit_run('new500', labels, questions)
    roots = root_comparison(oldroots, newroots)
    common = sorted(set(old) & set(new))
    pairs = [pair_detail(old[k], new[k], roots[k[0]]) for k in common]
    root_summary = {key: sum(bool(r[key]) for r in roots.values()) for key in
                    ['prompt_same', 'raw_output_same', 'query_same', 'docs_same', 'old_length', 'new_length']}
    root_summary['same_query_different_documents'] = sum(r['query_same'] and not r['docs_same'] for r in roots.values())
    pair_summary = {'common_pairs': len(pairs), **{key: sum(p[key] for p in pairs) for key in
                    ['first_prompt_same', 'first_prompt_same_nfc', 'first_input_ids_same_reference_tokenizer', 'first_output_same', 'first_evidence_same', 'boundary_policy_changed']},
                    'same_prompt_different_first_output': sum(p['first_prompt_same'] and not p['first_output_same'] for p in pairs),
                    'same_reference_input_ids_different_first_output': sum(p['first_input_ids_same_reference_tokenizer'] and not p['first_output_same'] for p in pairs),
                    'same_query_docs_different_first_prompt': sum(p['root_query_same'] and p['root_docs_same'] and not p['first_prompt_same'] for p in pairs)}
    summary = {'archive_sha256': 'b76014d79f8282204ef8ac35acc22ef70d15ed30be1f963f3143c577f0133f25',
               'old500': oldreport, 'new500': newreport, 'root_comparison': root_summary,
               'common_pair_comparison': pair_summary,
               'gold_source': {'path': str(DATA / 'gold_labels.sealed.jsonl'), 'sha256': validate.digest_file(DATA / 'gold_labels.sealed.jsonl'),
                               'n': len(labels), 'question_text_alignment': True},
               'reference_tokenizer': {'path': str(TOKENIZER_PATH), 'sha256': validate.digest_file(TOKENIZER_PATH),
                    'normalizer': json.loads(TOKENIZER_PATH.read_text(encoding='utf-8'))['normalizer'],
                    'caveat': 'Public Qwen3 tokenizer retained from previous independent audit, not archived server tokenizer. Matching all counts does not prove byte-identical server tokenizer.'},
               'definitions': {'primary_grid': COMMON, 'F1': 'legacy token-overlap F1 recomputed from original final_answer; not LLM judge',
                               'ACC': 'normalized gold answer substring in prediction, not semantic judge accuracy',
                               'invalid_rule': 'local_fallback=0 in all primary metrics',
                               'oracle_tie': 'maximize SAME evaluation metric, then minimize standalone input+output tokens, then ascending budget',
                               'token_cost': 'saved tokenizer counts summed and cross-checked with attempts; root cost counted once per selected trajectory, not exploration cost',
                               'causality': 'observational comparison; subgroup flags overlap; two different fresh runs are not controlled server replay'},
               'primary': {}, 'per_question_common_grid': {}}
    for metric in ['answer_em', 'answer_acc', 'answer_f1']:
        choices = {sid: COMMON for sid in roots}
        assert all((sid, b) in old and (sid, b) in new for sid in roots for b in COMMON)
        result, details = metric_audit(old, new, roots, metric, choices)
        summary['primary'][metric] = result
        save('per_question_' + metric + '.json', details)
        choices = {sid: [b for s, b in common if s == sid] for sid in roots}
        result2, _ = metric_audit(old, new, roots, metric, choices)
        summary['per_question_common_grid'][metric] = result2
    summary['fixed_common_budgets'] = {b: {'old': mean([old[sid, b] for sid in roots]),
                                           'new': mean([new[sid, b] for sid in roots])} for b in COMMON}
    save('root_pairs.json', list(roots.values()))
    save('common_trajectory_pairs.json', pairs)
    save('AUDIT_SUMMARY.json', summary)
    # Choose illustrative cases deterministically, not as a representative sample.
    details = json.loads((OUT / 'per_question_answer_em.json').read_text(encoding='utf-8'))
    categories = {'same_prompt_loss': lambda d: d['old_correct'] and not d['transfer_correct'] and d['first_prompt_same'],
                  'different_prompt_loss': lambda d: d['old_correct'] and not d['transfer_correct'] and not d['first_prompt_same'],
                  'truncation_loss': lambda d: d['old_correct'] and not d['transfer_correct'] and d['new_length'],
                  'same_query_docs_prompt_changed': lambda d: d['root_query_same'] and d['root_docs_same'] and not d['first_prompt_same']}
    chosen = {}
    for name, pred in categories.items():
        candidates = [d for d in details if pred(d)]
        for d in candidates[:3]:
            chosen[(d['sample_id'], d['old_budget'])] = {'category': name, 'details': d}
    for label in ['old500', 'new500']:
        for row in rows(INPUT / label / 'trajectories/trajectories.merged.jsonl.gz'):
            key = row['sample_id'], budget(row)
            if key in chosen:
                chosen[key][label] = row
    for (sid, b), item in chosen.items():
        item['old_root'] = oldroots[sid]
        item['new_root'] = newroots[sid]
        item['gold_answers'] = labels[sid]['gold_answers']
    save('RAW_CASES.json', list(chosen.values()))
    print(json.dumps({'root_comparison': root_summary, 'pairs': pair_summary,
        'old_checks': oldreport['checks'], 'new_checks': newreport['checks'],
        'old_warnings': oldreport['warnings'], 'new_warnings': newreport['warnings'],
        'primary_EM': summary['primary']['answer_em'],
        'primary_ACC': summary['primary']['answer_acc']['new_at_old_budget'],
        'primary_F1': summary['primary']['answer_f1']['new_at_old_budget']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
