"""Does the budget-success region coincide with annotated evidence visibility?

No model calls. Gold labels diagnose observed paths, never select a deployable
budget. Common B100..B600/FULL only; FULL is not a numeric neighbor.
"""
from __future__ import annotations

import collections
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
SOURCE = WORK / 'audit_hotpot_full7405_v433_20261001/input/analysis'
LABELS = ['100', '200', '300', '400', '500', '600', 'FULL']


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def relation(a, b):
    """Relation of the second visible fact set to the first."""
    new, lost = b & ~a, a & ~b
    if not new and not lost:
        return 'equal_nonempty' if a else 'equal_empty'
    if new and not lost:
        return 'superset'
    if lost and not new:
        return 'subset'
    return 'mixed'


def mean(x):
    return float(np.mean(x)) if len(x) else None


data = collections.defaultdict(dict)
for r in load(SOURCE / 'scored_records.jsonl.gz'):
    b = 'FULL' if r['exposure_mode'].startswith('full') else str(r['budget'])
    if b in LABELS:
        assert b not in data[r['sample_id']]
        data[r['sample_id']][b] = r
ids = sorted(data)
assert len(ids) == 7405 and all(len(d) == 7 for d in data.values())
index = {s: i for i, s in enumerate(ids)}
rr = [[data[s][b] for b in LABELS] for s in ids]
n = len(ids)
em = np.array([[bool(r['answer_em']) for r in rs] for rs in rr])
acc = np.array([[bool(r['answer_acc']) for r in rs] for rs in rr])
cost = np.array([[r['standalone_llm_input_tokens']+r['standalone_llm_output_tokens'] for r in rs] for rs in rr], float)
clean = np.array([not any(r['local_fallback'] for r in rs) for rs in rr])
assert not any(r.get('context_compactions', 0) for rs in rr for r in rs), 'Cumulative exposure is not a retention certificate when history is compacted.'
types = np.array([rs[0]['type'] for rs in rr])
queries = [[tuple(s['query'] for s in r['per_search_round_evidence']) for r in rs] for rs in rr]
counts = np.zeros(n, int)
questions = {}
masks = {mode: np.zeros((n, 7), np.int64) for mode in ['any_match', 'verbatim_only']}
roots = {mode: np.zeros((n, 7), np.int64) for mode in masks}
root_fact_details = {}
position_count = 0
for r in load(SOURCE / 'evidence_positions/supporting_evidence_positions.jsonl.gz'):
    b = r['condition_label']
    if b not in LABELS:
        continue
    sid, j = r['sample_id'], LABELS.index(b)
    i = index[sid]
    counts[i] = r['supporting_fact_count']
    questions[sid] = r['question']
    position_count += 1
    for fact in r['supporting_facts']:
        bit = 1 << fact['supporting_fact_index']
        for o in fact['occurrences']:
            if b == 'FULL' and o['search_round'] == 1:
                root_fact_details.setdefault(sid, []).append({
                    'fact_index': fact['supporting_fact_index'], 'title': fact['gold_title'],
                    'gold_sentence_id': fact['gold_sentence_id'], 'gold_sentence': fact['gold_sentence'],
                    'rank': o['rank_within_search'], 'match_mode': o['match_mode'],
                    'start': o['search_context_token_span']['token_start_0based'],
                    'end_exclusive': o['search_context_token_span']['token_end_exclusive_0based']})
            if o['visibility_at_this_condition'] != 'fully_visible':
                continue
            for mode in masks:
                if mode == 'verbatim_only' and o['match_mode'] != 'verbatim':
                    continue
                masks[mode][i, j] |= bit
                if o['search_round'] == 1:
                    roots[mode][i, j] |= bit
assert position_count == n*7
assert (counts > 0).all(), 'Evidence completeness must not be vacuously true for unlabelled questions.'
allbits = np.array([(1 << int(k))-1 for k in counts], np.int64)
q_same = np.array([[qq[j] == qq[j+1] for j in range(5)] for qq in queries])
rescued = ~em[:, 6] & em[:, :6].any(axis=1)

result = {'integrity': {'questions': n, 'common_grid_records': position_count,
                        'clean_all7_questions': int(clean.sum()), 'EM_rescues_vs_FULL': int(rescued.sum()),
                        'records_with_context_compaction': 0},
          'definitions': {'evidence': 'Hotpot gold supporting sentences mapped to retrieved Wikipedia passages and tokenizer spans',
                          'any_match': 'fully visible occurrence under any exported matching mode',
                          'verbatim_only': 'fully visible occurrence with match_mode=verbatim; conservative sensitivity check',
                          'set_relation': 'higher-budget cumulative visible fact set relative to lower-budget set',
                          'complete': 'every annotated supporting sentence has a fully visible matched occurrence',
                          'limits': ['Matched absence is not semantic evidence absence: corpus/sentence-version mismatch and alternative evidence exist.',
                                     'Visibility is not actual model use, and all annotated sentences are not necessarily required by a pretrained model.',
                                     'Dynamic queries mean different budgets can retrieve different packets after the fixed root.',
                                     'Observed associations do not establish a causal explanation.',
                                     'Only the five numeric adjacent pairs are used; FULL is separate.']},
          'pair_partitions': [], 'rescue_diagnostics': [], 'annotation_complete_error_counts': [],
          'lexical_acc_diagnostics': [],
          'root_onset_diagnostics': [], 'cases': []}

for mode, visible in masks.items():
    root = roots[mode]
    assert not ((root[:, :5] & ~root[:, 1:6]) != 0).any(), 'Fixed root evidence must be nested.'
    assert not (((visible[:, :5] & ~visible[:, 1:6]) != 0) & q_same).any(), 'Same raw query plan but gold visibility loss.'
    relations = np.array([[relation(int(v[j]), int(v[j+1])) for j in range(5)] for v in visible])
    new = (visible[:, 1:6] & ~visible[:, :5]) != 0
    root_new = (root[:, 1:6] & ~root[:, :5]) != 0
    harm = em[:, :5] & ~em[:, 1:6]
    gain = ~em[:, :5] & em[:, 1:6]
    for cohort in ['all', 'clean_all7', 'bridge', 'comparison']:
        qm = np.ones(n, bool) if cohort == 'all' else clean if cohort == 'clean_all7' else types == cohort
        pm = np.broadcast_to(qm[:, None], (n, 5))
        for rel in ['equal_empty', 'equal_nonempty', 'superset', 'subset', 'mixed']:
            pick = pm & (relations == rel)
            result['pair_partitions'].append({'match_mode': mode, 'cohort': cohort, 'relation': rel,
                'pairs': int(pick.sum()), 'wrong_to_correct': int((pick & gain).sum()),
                'correct_to_wrong': int((pick & harm).sum()),
                'same_raw_query_plan_pairs': int((pick & q_same).sum()),
                'same_raw_query_plan_gains': int((pick & q_same & gain).sum()),
                'same_raw_query_plan_harms': int((pick & q_same & harm).sum()),
                'mean_total_tokens_lower': mean(cost[:, :5][pick]),
                'mean_total_tokens_higher': mean(cost[:, 1:6][pick])})
        result['pair_partitions'].append({'match_mode': mode, 'cohort': cohort, 'relation': 'gain_summary',
            'gains': int((gain & pm).sum()), 'gains_with_new_root_support': int((gain & pm & root_new).sum()),
            'gains_with_new_cumulative_support': int((gain & pm & new).sum()),
            'gains_with_unchanged_nonempty_support': int((gain & pm & (relations == 'equal_nonempty')).sum()),
            'harms': int((harm & pm).sum()),
            'harms_with_no_support_loss': int((harm & pm & np.isin(relations, ['equal_empty','equal_nonempty','superset'])).sum())})
        rescue_ids = np.where(qm & rescued)[0]
        extra, equal, equal_nonempty, subset, complete, complete_both, zero = 0, 0, 0, 0, 0, 0, 0
        for i in rescue_ids:
            js = np.where(em[i, :6])[0]
            f = int(visible[i, 6])
            extra += any(int(visible[i, j]) & ~f for j in js)
            equal += any(int(visible[i, j]) == f for j in js)
            equal_nonempty += f != 0 and any(int(visible[i, j]) == f for j in js)
            subset += all((int(visible[i, j]) & ~f) == 0 for j in js)
            complete += any(visible[i, j] == allbits[i] for j in js)
            complete_both += visible[i, 6] == allbits[i] and any(visible[i, j] == allbits[i] for j in js)
            zero += all(visible[i, j] == 0 for j in js)
        result['rescue_diagnostics'].append({'match_mode': mode, 'cohort': cohort, 'rescued_questions': len(rescue_ids),
            'has_correct_budget_with_support_not_visible_in_FULL': int(extra),
            'all_correct_budget_support_sets_are_subsets_of_FULL': int(subset),
            'has_correct_budget_with_identical_support_set_to_FULL': int(equal),
            'has_correct_budget_with_identical_nonempty_support_set_to_FULL': int(equal_nonempty),
            'has_correct_budget_with_all_annotated_support_visible': int(complete),
            'has_correct_budget_and_FULL_with_all_annotated_support_visible': int(complete_both),
            'no_matched_support_visible_in_any_correct_budget': int(zero)})
        assert extra + subset == len(rescue_ids), 'Rescue partition must be exhaustive and disjoint.'
        complete_mask = visible == allbits[:, None]
        result['annotation_complete_error_counts'].append({'match_mode': mode, 'cohort': cohort,
            'questions_with_any_complete_annotation_visibility': int((qm & complete_mask.any(axis=1)).sum()),
            'questions_with_wrong_answer_despite_complete_annotation_visibility': int((qm & (complete_mask & ~em).any(axis=1)).sum()),
            'numeric_harms_annotated_support_complete_before_and_after': int((harm & pm & complete_mask[:, :5] & complete_mask[:, 1:6]).sum())})
        acc_gain = ~acc[:, :5] & acc[:, 1:6]
        acc_harm = acc[:, :5] & ~acc[:, 1:6]
        result['lexical_acc_diagnostics'].append({'match_mode': mode, 'cohort': cohort,
            'metric_limit': 'answer_acc is lexical containment, NOT an LLM judge or semantic verification',
            'gains': int((acc_gain & pm).sum()),
            'gains_with_new_cumulative_support': int((acc_gain & pm & new).sum()),
            'harms': int((acc_harm & pm).sum()),
            'harms_with_no_support_loss': int((acc_harm & pm & np.isin(relations, ['equal_empty','equal_nonempty','superset'])).sum()),
            'numeric_harms_annotated_support_complete_before_and_after': int((acc_harm & pm & complete_mask[:, :5] & complete_mask[:, 1:6]).sum()),
            'same_plan_numeric_harms_annotated_support_complete_before_and_after': int((acc_harm & pm & q_same & complete_mask[:, :5] & complete_mask[:, 1:6]).sum())})

    root_available_all = root[:, 6] == allbits
    monotonic_success = ~(em[:, :5] & ~em[:, 1:6]).any(axis=1)
    result['root_onset_diagnostics'].append({'match_mode': mode,
        'questions_with_all_gold_sentences_matched_in_frozen_root': int(root_available_all.sum()),
        'among_root_all_available_nonmonotonic_numeric_success_questions': int((root_available_all & ~monotonic_success).sum()),
        'root_visible_fact_loss_adjacent_pairs': 0})

# Cases prioritize single-search exact plans, no fallback, and all support visible.
visible = masks['any_match']
case_candidates = []
for i, sid in enumerate(ids):
    if not clean[i]:
        continue
    for j in range(5):
        same = q_same[i, j]
        if not same or not em[i, j] != em[i, j+1]:
            continue
        before, after = int(visible[i, j]), int(visible[i, j+1])
        rel = relation(before, after)
        if not em[i, j] and em[i, j+1] and rel == 'superset':
            kind = 'new_support_and_success_same_observed_search_plan'
        elif em[i, j] and not em[i, j+1] and before == after == int(allbits[i]):
            if acc[i, j] and not acc[i, j+1]:
                kind = 'complete_support_retained_with_EM_and_lexical_ACC_loss'
            else:
                kind = 'complete_support_retained_but_EM_becomes_wrong_same_observed_search_plan'
        elif em[i, j] and not em[i, j+1] and before == after and before:
            kind = 'nonempty_support_unchanged_but_answer_becomes_wrong_same_observed_search_plan'
        else:
            continue
        verbatim_complete = masks['verbatim_only'][i, j] == masks['verbatim_only'][i, j+1] == allbits[i]
        priority = (int(len(queries[i][j]) != 1), -int(verbatim_complete), -int(before == after == int(allbits[i])), sid, j)
        case_candidates.append((priority, kind, i, j))
for kind in ['new_support_and_success_same_observed_search_plan',
             'complete_support_retained_with_EM_and_lexical_ACC_loss',
             'complete_support_retained_but_EM_becomes_wrong_same_observed_search_plan',
             'nonempty_support_unchanged_but_answer_becomes_wrong_same_observed_search_plan']:
    selected = [c for c in sorted(case_candidates) if c[1] == kind][:3]
    for _, _, i, j in selected:
        sid = ids[i]
        result['cases'].append({'category': kind, 'sample_id': sid, 'question': questions[sid],
            'type': types[i], 'budgets': [LABELS[j], LABELS[j+1]],
            'EM': [bool(em[i, j]), bool(em[i, j+1])],
            'query_plan': list(queries[i][j]), 'all_gold_fact_count': int(counts[i]),
            'visible_fact_indices': [[k for k in range(counts[i]) if int(visible[i, x]) & (1 << k)] for x in [j,j+1]],
            'all_seven_EM': em[i].astype(int).tolist(), 'all_seven_total_tokens': cost[i].tolist(),
            'all_seven_lexical_ACC': acc[i].astype(int).tolist(),
            'root_fact_locations': root_fact_details.get(sid, []),
            'raw_answer_and_thought': 'NOT IN REVIEW ARCHIVE; do not invent a reasoning narrative'})

# Retain the mapped sentence locations for both sides of each selected case.
case_lookup = collections.defaultdict(list)
for c in result['cases']:
    c['pair_fact_locations'] = {}
    for b in c['budgets']:
        case_lookup[(c['sample_id'], b)].append(c)
for p in load(SOURCE / 'evidence_positions/supporting_evidence_positions.jsonl.gz'):
    for c in case_lookup.get((p['sample_id'], p['condition_label']), []):
        c['pair_fact_locations'][p['condition_label']] = p['supporting_facts']

out = HERE / 'evidence_window_audit.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
print(json.dumps({k: result[k] for k in ['integrity','rescue_diagnostics','annotation_complete_error_counts','root_onset_diagnostics']}, ensure_ascii=False, indent=2))
print('Saved:', out)
