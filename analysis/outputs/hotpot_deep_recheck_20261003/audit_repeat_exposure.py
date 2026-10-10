"""Audit at-event unread support and repeated-query cost, without inference.

Gold supporting facts are diagnostic labels only. No switched-policy quality
is estimated. Raw-query equality, not fuzzy normalization, defines repeats.
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
BUDGETS = ['100', '200', '300', '400', '500', '600', 'FULL']
STAGE = WORK / 'work/hotpot_stage_export_review_20260928T141101Z/analysis/scored_records.jsonl.gz'


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def label(r):
    return 'FULL' if r['exposure_mode'].startswith('full') else str(r['budget'])


def first_repeat(r):
    seen = {}
    for s in r['per_search_round_evidence']:
        query, number = s['query'], s['search_round']
        if query in seen:
            return seen[query], number
        seen[query] = number
    return None


def ci(x, seed=20261003):
    x = np.asarray(x, float)
    if not len(x):
        return None
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(30):
        indices = rng.integers(len(x), size=(100, len(x)))
        samples.extend(x[indices].mean(axis=1))
    return np.quantile(samples, [.025, .975]).tolist()


data = collections.defaultdict(dict)
for r in rows(SOURCE / 'scored_records.jsonl.gz'):
    b = label(r)
    if b in BUDGETS:
        data[r['sample_id']][b] = r
assert len(data) == 7405 and all(len(d) == 7 for d in data.values())
stage_ids = {r['sample_id'] for r in rows(STAGE)} & set(data)
assert 4000 <= len(stage_ids) <= 4300
clean = {sid for sid, d in data.items() if not any(r['local_fallback'] for r in d.values())}

repeat_keys = {(sid, b): first_repeat(r) for sid, d in data.items() for b, r in d.items()}
result = {
    'integrity': {'questions': 7405, 'paired_conditions': 7,
                  'previous_stage_questions': len(stage_ids), 'later_questions': 7405-len(stage_ids)},
    'definitions': {
        'repeat': 'first byte-identical Python raw query string repeat; not normalized-query repeat',
        'unread_support': 'previous cached packet has a matched gold support sentence beyond its exposed prefix that is not fully visible in ANY preceding round',
        'gold_usage': 'diagnostic only; runtime rule uses exact query/cache fingerprint/exposed spans, never gold',
        'packet_identity': 'matched support occurrence signatures are a proxy; full packet hash is absent from the review archive',
        'two_times_prefix': 'simulate expanding the original cached packet cap from B to 2B at the first repeat; no new trajectory outcome inferred',
        'cohort': 'earlier stage vs later completed questions is a retrospective replication, not a prospective untouched holdout',
    },
    'static_condition_contrasts': [], 'at_first_repeat': {}, 'cases': [],
}

for a, b in [('100', '600'), ('500', 'FULL')]:
    for cohort in ['all', 'clean_all7', 'earlier_stage', 'later_completion', 'bridge', 'comparison']:
        ids = [sid for sid, d in data.items() if repeat_keys[(sid, a)] and
               (cohort != 'clean_all7' or sid in clean) and
               (cohort != 'earlier_stage' or sid in stage_ids) and
               (cohort != 'later_completion' or sid not in stage_ids) and
               (cohort not in ['bridge', 'comparison'] or d[a]['type'] == cohort)]
        aa, bb = [data[s][a] for s in ids], [data[s][b] for s in ids]
        def arr(rr, field):
            return np.array([r[field] for r in rr], float)
        ca = arr(aa, 'standalone_llm_input_tokens') + arr(aa, 'standalone_llm_output_tokens')
        cb = arr(bb, 'standalone_llm_input_tokens') + arr(bb, 'standalone_llm_output_tokens')
        ea, eb = arr(aa, 'answer_em'), arr(bb, 'answer_em')
        ac, bc = arr(aa, 'answer_acc'), arr(bb, 'answer_acc')
        em_ci = ci(100*(eb-ea))
        result['static_condition_contrasts'].append({
            'from': a, 'to': b, 'cohort': cohort, 'n': len(ids),
            'EM_from': float(ea.mean()), 'EM_to': float(eb.mean()),
            'EM_delta_pp': float(100*(eb-ea).mean()), 'EM_delta_95CI_pp': em_ci,
            'ACC_from': float(ac.mean()), 'ACC_to': float(bc.mean()),
            'ACC_delta_pp': float(100*(bc-ac).mean()),
            'mean_total_tokens_from': float(ca.mean()), 'mean_total_tokens_to': float(cb.mean()),
            'mean_token_delta': float((cb-ca).mean()), 'token_delta_95CI': ci(cb-ca),
            'search_calls_from': float(arr(aa, 'search_calls').mean()),
            'search_calls_to': float(arr(bb, 'search_calls').mean()),
            'warning': 'subset selected by behavior in the FROM condition; not the effect of switching at that event',
        })

counts = {b: collections.Counter() for b in BUDGETS}
for r in rows(SOURCE / 'evidence_positions/supporting_evidence_positions.jsonl.gz'):
    b = r['condition_label']
    if b not in BUDGETS:
        continue
    sid = r['sample_id']
    event = repeat_keys[(sid, b)]
    if not event:
        continue
    old, new = event
    c = counts[b]
    c['repeat_questions'] += 1
    c[f'first_repeat_round_{new}'] += 1
    packet = data[sid][b]['per_search_round_evidence'][old-1]
    cap, full = packet['actual_exposed_tokens'], packet['full_context_tokens']
    if cap < full:
        c['has_unread_packet_tokens'] += 1
        c['unread_packet_tokens_sum'] += full-cap
    previously_visible, hidden_packet, can_2b, can_full = set(), set(), set(), set()
    old_signatures, repeat_signatures = set(), set()
    for fact in r['supporting_facts']:
        idx = fact['supporting_fact_index']
        for o in fact['occurrences']:
            rr = o['search_round']
            end = o['search_context_token_span']['token_end_exclusive_0based']
            if rr < new and end <= o['search_context_exposed_tokens']:
                previously_visible.add(idx)
            signature = (idx, o['document_id'], o['rank_within_search'],
                         o['search_context_token_span']['token_start_0based'], end)
            if rr == old:
                old_signatures.add(signature)
                if end > cap:
                    hidden_packet.add(idx)
                    can_full.add(idx)
                    if b != 'FULL' and end <= min(full, 2*int(b)):
                        can_2b.add(idx)
            if rr == new:
                repeat_signatures.add(signature)
    hidden_packet -= previously_visible
    can_2b -= previously_visible
    can_full -= previously_visible
    total = r['supporting_fact_count']
    c['matched_packet_signatures_identical'] += int(old_signatures == repeat_signatures)
    c['previously_visible_gold_facts_sum'] += len(previously_visible)
    c['gold_facts_sum'] += total
    if hidden_packet:
        c['has_unread_new_matched_support'] += 1
        c['unread_new_matched_support_count_sum'] += len(hidden_packet)
        c['unread_new_matched_support_wrong_final_EM'] += int(not data[sid][b]['answer_em'])
    if can_2b:
        c['two_times_prefix_adds_matched_support'] += 1
    if can_full:
        c['full_cached_packet_adds_matched_support'] += 1
    if len(previously_visible) < total:
        c['not_all_support_visible_before_repeat'] += 1
        if len(previously_visible | can_2b) == total:
            c['two_times_prefix_closes_support'] += 1
        if len(previously_visible | can_full) == total:
            c['full_cached_packet_closes_support'] += 1
    if hidden_packet and len(result['cases']) < 20:
        result['cases'].append({
            'sample_id': sid, 'budget': b, 'question': r['question'],
            'query': packet['query'], 'prior_round': old, 'first_repeat_round': new,
            'old_visible_tokens': cap, 'cached_packet_full_tokens': full,
            'support_visible_before_trigger': sorted(previously_visible),
            'support_unread_in_repeated_packet': sorted(hidden_packet),
            'support_available_with_2B': sorted(can_2b),
            'original_final_EM': data[sid][b]['answer_em'],
            'counterfactual_quality': 'NOT OBSERVED',
        })
for b, c in counts.items():
    assert c['repeat_questions'] == sum(bool(repeat_keys[(s, b)]) for s in data)
    n = c['repeat_questions']
    result['at_first_repeat'][b] = dict(c)
    result['at_first_repeat'][b]['share_all_questions'] = n/7405
    if n:
        result['at_first_repeat'][b]['has_unread_new_matched_support_rate_among_repeats'] = c['has_unread_new_matched_support']/n
        result['at_first_repeat'][b]['two_times_prefix_support_closure_rate_among_repeats'] = c['two_times_prefix_closes_support']/n

# A parser-only change is separately counted, never mislabeled as policy gain.
result['recovered_vs_strict'] = []
for b in BUDGETS:
    rr = [d[b] for d in data.values()]
    result['recovered_vs_strict'].append({
        'budget': b, 'EM': float(np.mean([r['answer_em'] for r in rr])),
        'recovered_EM': float(np.mean([r['recovered_answer_em'] for r in rr])),
        'recovered_EM_delta_pp': float(100*np.mean([r['recovered_answer_em']-r['answer_em'] for r in rr])),
        'warning': 'parser/evaluation sensitivity, not improved reasoning',
    })

path = HERE / 'repeat_exposure_audit.json'
path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
print(json.dumps({k: result[k] for k in ['integrity', 'static_condition_contrasts', 'at_first_repeat', 'recovered_vs_strict']}, ensure_ascii=False, indent=2))
