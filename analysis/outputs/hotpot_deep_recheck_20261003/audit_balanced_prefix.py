"""Support-position audit of rank-prefix vs equal per-document prefixes.

This is not an answer-quality experiment. Equal caps ignore concatenation
boundary re-tokenization/header overhead; an actual implementation must count
the final assembled prompt and back off to budget. No gold controls the rule.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
SOURCE = WORK / 'audit_hotpot_full7405_v433_20261001/input/analysis'


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def ci(x):
    rng = np.random.default_rng(20261003)
    x = np.asarray(x, float)
    means = []
    for _ in range(20):
        idx = rng.integers(len(x), size=(100, len(x)))
        means.extend(x[idx].mean(axis=1))
    return np.quantile(means, [.025, .975]).tolist()


data = {}
for r in rows(SOURCE / 'scored_records.jsonl.gz'):
    if r['exposure_mode'].startswith('full'):
        data[r['sample_id']] = r
position = list(rows(SOURCE / 'evidence_positions/full_endpoint_positions_one_per_question.jsonl.gz'))
assert len(data) == len(position) == 7405
results = []
for b in [100, 200, 300, 400, 500, 600]:
    values = []
    for r in position:
        sequential, balanced, available = set(), set(), set()
        for f in r['supporting_facts']:
            fact = f['supporting_fact_index']
            for o in f['occurrences']:
                if o['search_round'] != 1:
                    continue
                available.add(fact)
                if o['search_context_token_span']['token_end_exclusive_0based'] <= b:
                    sequential.add(fact)
                if o['document_token_span']['token_end_exclusive_0based'] <= b//5:
                    balanced.add(fact)
        total = r['supporting_fact_count']
        values.append({
            'sample_id': r['sample_id'], 'type': data[r['sample_id']]['type'],
            'seq_recall': len(sequential)/total, 'balanced_recall': len(balanced)/total,
            'seq_all': int(len(sequential) == total), 'balanced_all': int(len(balanced) == total),
            'seq_any': int(bool(sequential)), 'balanced_any': int(bool(balanced)),
            'gain_facts': len(balanced-sequential), 'lost_facts': len(sequential-balanced),
            'root_available_recall': len(available)/total,
        })
    for typ in ['all', 'bridge', 'comparison']:
        rr = [r for r in values if typ == 'all' or r['type'] == typ]
        def avg(k):
            return float(np.mean([r[k] for r in rr]))
        results.append({
            'budget': b, 'per_document_cap': b//5, 'cohort': typ, 'n': len(rr),
            'sequential_root_support_recall': avg('seq_recall'),
            'balanced_root_support_recall': avg('balanced_recall'),
            'support_recall_delta_pp': 100*(avg('balanced_recall')-avg('seq_recall')),
            'support_recall_delta_95CI_pp': ci([100*(r['balanced_recall']-r['seq_recall']) for r in rr]),
            'sequential_root_all_support_rate': avg('seq_all'),
            'balanced_root_all_support_rate': avg('balanced_all'),
            'sequential_root_any_support_rate': avg('seq_any'),
            'balanced_root_any_support_rate': avg('balanced_any'),
            'questions_gaining_some_support': sum(r['gain_facts'] > 0 for r in rr),
            'questions_losing_some_support': sum(r['lost_facts'] > 0 for r in rr),
        })
result = {
    'scope': 'root cached Top-5 only, full cohort, gold used only as diagnostic',
    'budget_precision': 'counterfactual per-document token caps only; exact final-prompt tokenizer overhead not available',
    'quality_effect': 'NOT OBSERVED; support coverage does not imply answer improvement',
    'rows': results,
}
(HERE / 'balanced_prefix_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
