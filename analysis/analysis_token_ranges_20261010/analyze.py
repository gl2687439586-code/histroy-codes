"""Observed per-question token cost ranges by exposure budget; no inference."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent
WORK = BASE.parent
SOURCE = WORK / 'audit_hotpot_full7405_v433_20261001/input/analysis/scored_records.jsonl.gz'


def quantile(values, p):
    x = sorted(values)
    position = (len(x) - 1) * p
    left = int(position)
    right = min(left + 1, len(x) - 1)
    return x[left] + (position - left) * (x[right] - x[left])


def describe(values):
    assert values and all(isinstance(x, (int, float)) and x >= 0 for x in values)
    return {'min': min(values), 'p10': quantile(values, .1), 'p25': quantile(values, .25),
            'median': statistics.median(values), 'mean': statistics.mean(values),
            'p75': quantile(values, .75), 'p90': quantile(values, .9), 'p95': quantile(values, .95),
            'max': max(values)}


def main():
    grouped = defaultdict(list)
    keys = set()
    root_usage = {}
    with gzip.open(SOURCE, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            condition = str(row['condition_group'])
            key = row['sample_id'], condition
            assert key not in keys
            keys.add(key)
            inp = int(row['standalone_llm_input_tokens'])
            out = int(row['standalone_llm_output_tokens'])
            root = (inp - int(row['continuation_llm_input_tokens']), out - int(row['continuation_llm_output_tokens']))
            assert all(x >= 0 for x in root)
            if row['sample_id'] in root_usage:
                assert root_usage[row['sample_id']] == root
            root_usage[row['sample_id']] = root
            item = {'id': row['sample_id'], 'input_tokens': inp, 'output_tokens': out,
                    'total_tokens': inp + out, 'evidence_tokens': int(row['visible_evidence_tokens_total']),
                    'search_calls': int(row['search_calls']), 'fallback': bool(row['local_fallback']),
                    'em': row['answer_em'], 'f1': row['answer_f1'], 'acc': row['answer_acc']}
            assert item['search_calls'] in (1, 2, 3, 4)
            assert len(row['per_search_round_evidence']) == item['search_calls']
            grouped[condition].append(item)
    assert len(keys) == 61754 and len(root_usage) == 7405
    result = []
    fields = ['total_tokens', 'input_tokens', 'output_tokens', 'evidence_tokens', 'search_calls']
    for condition in sorted(grouped, key=lambda x: float('inf') if x == 'FULL' else int(x)):
        rows = grouped[condition]
        rec = {'budget': condition, 'n': len(rows), 'coverage': len(rows) / 7405,
               **{field: describe([r[field] for r in rows]) for field in fields},
               'format_fallbacks_included': sum(r['fallback'] for r in rows),
               'min_cost_sample_id': min(rows, key=lambda r: r['total_tokens'])['id'],
               'max_cost_sample_id': max(rows, key=lambda r: r['total_tokens'])['id'],
               'quality': {field: statistics.mean(r[field] for r in rows) for field in ['em','f1','acc']},
               'by_search_calls': []}
        for calls in sorted({r['search_calls'] for r in rows}):
            subset = [r for r in rows if r['search_calls'] == calls]
            rec['by_search_calls'].append({'search_calls': calls, 'n': len(subset),
                                          **{field: describe([r[field] for r in subset]) for field in fields if field != 'search_calls'},
                                          'format_fallbacks_included': sum(r['fallback'] for r in subset)})
        assert sum(x['n'] for x in rec['by_search_calls']) == rec['n']
        for field in fields:
            if field == 'search_calls':
                continue
            weighted = sum(x['n'] * x[field]['mean'] for x in rec['by_search_calls']) / rec['n']
            assert abs(weighted - rec[field]['mean']) < 1e-8
        result.append(rec)
    # Reconcile means and physical successful usage with the earlier independent audit.
    source_table = json.loads((WORK/'audit_hotpot_full7405_v433_20261001/input/analysis/token_budget/summary.json').read_text(encoding='utf-8'))
    physical = sum(sum(x) for x in root_usage.values())
    physical += sum(r['total_tokens'] - sum(root_usage[r['id']]) for rows in grouped.values() for r in rows)
    assert physical == source_table['physical_successful_recorded_input_output_tokens'] == 204278770
    report = {'source': str(SOURCE), 'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'questions': 7405, 'records': 61754,
              'definition': 'standalone_llm_input_tokens + standalone_llm_output_tokens; root included once per deployed trajectory; repeated history input counted; evidence already included in input; observed ranges, not configured limits',
              'physical_successful_tokens_reconciled': physical,
              'all_recorded_trajectories_included_even_format_fallback': True,
              'conditions': result}
    (BASE/'token_ranges.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['HotpotQA v4.3.3 全量7405题：按每轮证据暴露预算统计每题实际token开销',
             '数据：61754条已记录轨迹；包括格式fallback轨迹的实际开销，不只统计答对题。',
             '总tokens=LLM input+output，独立运行一条轨迹时首Search root计一次。',
             '证据tokens已包含在输入中，不能再次相加；区间为本次观测最小/最大，不是硬预算上限。',
             'P10–P90为中心80%经验分位区间（不是置信区间）；对小样本高预算档应谨慎解释。',
             '共同7405题：100、200、300、400、500、600、FULL；更大数值档题集不同。',
             '', '预算 | 题数 | 最小–最大总tokens | P10–P90 | 中位数 | 平均总tokens | 平均输入 | 平均输出 | 平均累计证据']
    for r in result:
        t = r['total_tokens']
        lines.append(f"{r['budget']:>4} | {r['n']:>4} | {t['min']:>6,}–{t['max']:<6,} | {t['p10']:,.1f}–{t['p90']:,.1f} | {t['median']:,.1f} | {t['mean']:,.1f} | {r['input_tokens']['mean']:,.1f} | {r['output_tokens']['mean']:,.1f} | {r['evidence_tokens']['mean']:,.1f}")
    (BASE/'REPORT_ZH.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    step_lines = ['HotpotQA v4.3.3：预算档位 × 实际Search步数的每题完整轨迹token成本',
                  'step = 实际Search次数（包括首Search）；最终回答不额外计一个Search step。',
                  '数值是该组完整轨迹从首query生成至最终回答的输入+输出总tokens，不是某一步增量。',
                  '包括格式fallback记录；没有按正确与否筛选。root每条独立轨迹计一次，证据不重复加价。',
                  '最小–最大为本次观测范围；P10–P90为经验分位区间，不是置信区间或预算硬上限。',
                  'B100–B600和FULL有同样7405题，但按Search次数分组后，各组题目不再保证相同。',
                  'B700以上条件覆盖题目较少；无观测的组合不列出，不代表开销为零。',
                  '', '预算 | Search步数 | 记录数 | 最小–最大总tokens | 平均总tokens | 中位数 | P10–P90 | 平均输入 | 平均输出 | 平均累计证据']
    for r in result:
        for group in r['by_search_calls']:
            t = group['total_tokens']
            step_lines.append(f"{r['budget']:>4} | {group['search_calls']} | {group['n']:>4} | {t['min']:>6,}–{t['max']:<6,} | {t['mean']:,.1f} | {t['median']:,.1f} | {t['p10']:,.1f}–{t['p90']:,.1f} | {group['input_tokens']['mean']:,.1f} | {group['output_tokens']['mean']:,.1f} | {group['evidence_tokens']['mean']:,.1f}")
    (BASE/'BUDGET_BY_STEP_ZH.txt').write_text('\n'.join(step_lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    print('CHECKS PASSED: unique keys, shared-root accounting, 7405 questions, 61754 records, physical usage reconciliation.')
    print('\nBUDGET BY STEP (all conditions):')
    print('\n'.join(step_lines))


if __name__ == '__main__':
    main()
