"""Paired B500/FULL contrasts by task type and observed search plan.

Official question type is descriptive, not a deployable feature. Same observed
query sequence is not a randomized controlled-plan intervention. All subgroup
results are exploratory and require independent confirmation.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
BASE = WORK / 'audit_hotpot_full7405_v433_20261001/input/analysis'


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


data = {}
for r in load(BASE / 'scored_records.jsonl.gz'):
    b = 'FULL' if r['exposure_mode'].startswith('full') else str(r['budget'])
    if b in ['500', 'FULL']:
        data.setdefault(r['sample_id'], {})[b] = r
stage = {r['sample_id'] for r in load(WORK / 'work/hotpot_stage_export_review_20260928T141101Z/analysis/scored_records.jsonl.gz')}


def metrics(ids):
    rr = [data[s] for s in ids]
    n = len(rr)
    if not n:
        return {'n': 0}
    def a(b, key):
        return np.array([d[b][key] for d in rr], float)
    de = a('500', 'answer_em')-a('FULL', 'answer_em')
    da = a('500', 'answer_acc')-a('FULL', 'answer_acc')
    di = a('500', 'standalone_llm_input_tokens')-a('FULL', 'standalone_llm_input_tokens')
    do = a('500', 'standalone_llm_output_tokens')-a('FULL', 'standalone_llm_output_tokens')
    ca = a('500', 'standalone_llm_input_tokens')+a('500', 'standalone_llm_output_tokens')
    cb = a('FULL', 'standalone_llm_input_tokens')+a('FULL', 'standalone_llm_output_tokens')
    rng = np.random.default_rng(20261003)
    boot = []
    for _ in range(30):
        idx = rng.integers(n, size=(100, n))
        boot.extend(np.column_stack([de[idx].mean(axis=1)*100, da[idx].mean(axis=1)*100,
                                     (di+do)[idx].mean(axis=1)]))
    bounds = np.quantile(np.asarray(boot), [.025, .975], axis=0)
    wins, harms = int(sum(de == 1)), int(sum(de == -1))
    return {'n': n, 'EM_B500': float(a('500', 'answer_em').mean()),
            'EM_FULL': float(a('FULL', 'answer_em').mean()),
            'EM_delta_pp': float(de.mean()*100), 'EM_delta_95CI_pp': bounds[:, 0].tolist(),
            'ACC_delta_pp': float(da.mean()*100), 'ACC_delta_95CI_pp': bounds[:, 1].tolist(),
            'mean_tokens_B500': float(ca.mean()), 'mean_tokens_FULL': float(cb.mean()),
            'token_reduction_rate': float(1-ca.mean()/cb.mean()),
            'input_token_delta': float(di.mean()), 'output_token_delta': float(do.mean()),
            'total_token_delta_95CI': bounds[:, 2].tolist(),
            'EM_B500_correct_FULL_wrong': wins, 'EM_B500_wrong_FULL_correct': harms,
            'mcnemar_exact_p': float(binomtest(wins, wins+harms, .5).pvalue) if wins+harms else 1.,
            'search_delta': float((a('500', 'search_calls')-a('FULL', 'search_calls')).mean()),
            'visible_strict_support_recall_delta': float((a('500', 'support_sentence_strict_recall')-a('FULL', 'support_sentence_strict_recall')).mean())}


results = []
for typ in ['bridge', 'comparison']:
    for group in ['all', 'clean_pair', 'same_query_sequence', 'changed_query_sequence',
                  'same_query_sequence_clean_pair', 'same_query_sequence_clean_same_support_recall',
                  'same_query_sequence_clean_more_FULL_support', 'earlier_stage', 'later_completion', 'test']:
        ids = []
        for sid, d in data.items():
            if d['FULL']['type'] != typ:
                continue
            same = tuple(s['query'] for s in d['500']['per_search_round_evidence']) == tuple(s['query'] for s in d['FULL']['per_search_round_evidence'])
            clean = not any(r['local_fallback'] for r in d.values())
            ht = int(hashlib.sha256(('deep-recheck-v1:'+sid).encode()).hexdigest()[:8], 16) % 100
            if group == 'clean_pair' and not clean:
                continue
            if group.startswith('same_query_sequence') and not same:
                continue
            if group.startswith('same_query_sequence_clean') and not clean:
                continue
            if group == 'changed_query_sequence' and same:
                continue
            if group == 'same_query_sequence_clean_same_support_recall' and d['500']['support_sentence_strict_recall'] != d['FULL']['support_sentence_strict_recall']:
                continue
            if group == 'same_query_sequence_clean_more_FULL_support' and not d['500']['support_sentence_strict_recall'] < d['FULL']['support_sentence_strict_recall']:
                continue
            if group == 'earlier_stage' and sid not in stage:
                continue
            if group == 'later_completion' and sid in stage:
                continue
            if group == 'test' and ht < 80:
                continue
            ids.append(sid)
        results.append({'type': typ, 'group': group, **metrics(ids)})
payload = {'contrast': 'B500 minus FULL',
           'limits': ['Official question type used for subgroup description only.',
                      'Same observed search plan is post-treatment, not a controlled causal effect.',
                      'Single Qwen3-8B run; no independent-model replication.',
                      'ACC is lexical answer containment, not LLM-judge accuracy.',
                      'Unadjusted exploratory confidence intervals and p values; no multiple-testing correction.'],
           'results': results}
(HERE / 'type_plan_audit.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
print(json.dumps(payload, indent=2, ensure_ascii=False))
