"""Merge exact-match positives with explicit semantic reviews; never infer negatives.

No model calls. Unreviewed non-EM answers remain unknown. Original files are read-only.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent
WORK = BASE.parent
SCORED = WORK / 'audit_hotpot_full7405_v433_20261001/input/analysis/scored_records.jsonl.gz'
ELIGIBLE = BASE / 'semantic_acc_pending/eligible_trajectory_map.jsonl.gz'
REVIEW = BASE / 'codex_direct_semantic/batch_001_labelled_trajectories.json'
OUT = BASE / 'em_plus_semantic'


def read_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def key(row):
    return row['sample_id'], row['condition_id']


def stats(rows):
    if not rows:
        return dict(n=0, mean_total_tokens=None, min_total_tokens=None, max_total_tokens=None)
    costs = [r['total_tokens'] for r in rows]
    assert len({r['sample_id'] for r in rows}) == len(rows)
    return dict(n=len(rows), mean_total_tokens=statistics.mean(costs),
                min_total_tokens=min(costs), max_total_tokens=max(costs))


def aggregate(rows, scope):
    groups = defaultdict(list)
    for row in rows:
        groups[row['budget'], row['search_calls']].append(row)
    budgets = sorted({r['budget'] for r in rows}, key=lambda b: float('inf') if b == 'FULL' else int(b))
    tables = {0: [], 1: []}
    coverage = []
    for budget in budgets:
        for step in range(1, 5):
            group = groups.get((budget, step), [])
            if not group:
                continue
            for label in (0, 1):
                tables[label].append(dict(budget=budget, search_steps=step,
                                          **stats([r for r in group if r['combined_acc'] == label])))
            count = Counter(r['combined_acc'] for r in group)
            coverage.append(dict(budget=budget, search_steps=step, eligible=len(group),
                                 acc1=count[1], acc0=count[0], pending=count[None]))
            assert count[1] + count[0] + count[None] == len(group)
    counts = Counter(r['combined_acc'] for r in rows)
    summary = dict(scope=scope, records=len(rows), unique_questions=len({r['sample_id'] for r in rows}),
                   acc1=counts[1], acc0=counts[0], pending=counts[None],
                   exact_match_positives=sum(r['answer_em'] == 1 for r in rows),
                   semantic_additional_positives=sum(r['answer_em'] != 1 and r['semantic_correct'] == 1 for r in rows),
                   sources=dict(Counter(r['label_source'] for r in rows)),
                   tables=tables, coverage=coverage)
    assert counts[1] == summary['exact_match_positives'] + summary['semantic_additional_positives']
    return summary


def format_table(rows):
    lines = ['| 每轮证据预算 | 搜索 step 数 | 题数 | 平均总 tokens | 总 tokens 范围：最小–最大 |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        mean = f"{row['mean_total_tokens']:,.1f}" if row['n'] else '—'
        interval = f"{row['min_total_tokens']:,}–{row['max_total_tokens']:,}" if row['n'] else '—'
        lines.append(f"| {row['budget']} | {row['search_steps']} | {row['n']:,} | {mean} | {interval} |")
    return lines


def write_report(summary, filename):
    lines = ['EM=1与直接语义评审联合分组：'+summary['scope'],
             'ACC=1：原始评分answer_em=1，或Codex逐条语义评审为正确。',
             'ACC=0：answer_em不为1，且Codex逐条语义评审明确为错误。',
             '未评审/待裁决且EM不为1的记录不放入任何二元表，不计作错误。',
             '未使用原来的词法answer_acc字段。以下不是全量语义ACC完成结果。',
             '排除任何root或续写调用触及8192生成上限/以length结束的轨迹；不是按整条轨迹总tokens>=8192排除。',
             '总tokens=独立完整轨迹输入+输出，root计一次，包含重复历史输入；证据不重复相加。',
             'step=实际Search次数，包含首Search；题数指该预算×step×标签组内独立题数，不可跨预算相加为独立问题数。',
             '0题组的均值和范围为未定义，不是零成本。',
             f"本范围{summary['records']:,}条、{summary['unique_questions']:,}题；ACC=1：{summary['acc1']:,}；ACC=0：{summary['acc0']:,}；未评审/待裁决：{summary['pending']:,}。",
             f"ACC=1来源：EM=1共{summary['exact_match_positives']:,}条，额外语义正确{summary['semantic_additional_positives']:,}条。",
             '全量部分标注表中正例和负例的标签覆盖不同，不可据此比较总体正确/错误组的平均成本或计算全量语义ACC。',
             '', '表1：ACC=1', '']
    lines += format_table(summary['tables'][1])
    lines += ['', '表2：ACC=0', '']
    lines += format_table(summary['tables'][0])
    lines += ['', '逐分组标签覆盖：预算 | step | 保留数 | ACC1 | ACC0 | 未评审/待裁决']
    for row in summary['coverage']:
        lines.append(' | '.join(str(row[k]) for k in ('budget', 'search_steps', 'eligible', 'acc1', 'acc0', 'pending')))
    (OUT/filename).write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    scored = {key(r): r for r in read_gz(SCORED)}
    assert len(scored) == 61754
    reviewed_rows = json.loads(REVIEW.read_text(encoding='utf-8'))
    reviewed = {key(r): r for r in reviewed_rows}
    assert len(reviewed) == len(reviewed_rows) == 420
    judged = {}
    for row in reviewed_rows:
        jid = row['judge_item_id']
        decision = (row['semantic_correct'], row['reason_zh'])
        if jid in judged:
            assert judged[jid] == decision
        judged[jid] = decision
    rows = []
    conflicts = []
    for row in read_gz(ELIGIBLE):
        source = scored[key(row)]
        assert not row['root_cap_hit'] and not row['continuation_cap_hit']
        assert row['total_tokens'] == source['standalone_llm_input_tokens'] + source['standalone_llm_output_tokens']
        assert row['budget'] == str(source['condition_group'])
        assert row['search_calls'] == source['search_calls']
        em = source['answer_em']
        assert em in (0, 1)
        review = judged.get(row['judge_item_id'])
        semantic = review[0] if review else None
        if em == 1 and semantic == 0:
            conflicts.append(dict(sample_id=row['sample_id'], condition_id=row['condition_id'],
                                  reason=review[1]))
        if em == 1:
            label, origin = 1, 'exact_match'
        elif semantic is not None:
            label, origin = semantic, 'direct_semantic'
        else:
            label, origin = None, 'needs_adjudication' if review else 'not_reviewed'
        rows.append({**row, 'answer_em': em, 'semantic_correct': semantic,
                     'combined_acc': label, 'label_source': origin,
                     'semantic_reason_zh': review[1] if review else None})
    assert len(rows) == 58868 and len({key(r) for r in rows}) == len(rows)
    if conflicts:
        raise RuntimeError('EM-positive / semantic-negative conflict requires explicit review: '+str(conflicts))
    full = aggregate(rows, 'full7405_currently_labelled_only')
    subset = aggregate([r for r in rows if key(r) in reviewed], 'reviewed50_question_subset')
    for name, summary in [('full_partial', full), ('reviewed50', subset)]:
        (OUT/(name+'_summary.json')).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        write_report(summary, name+'_TABLES_ZH.txt')
    with gzip.open(OUT/'combined_label_ledger.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    manifest = dict(rule='EM==1 OR direct_semantic==1; negative only if EM!=1 AND direct_semantic==0; all other records unknown',
                    input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (SCORED, ELIGIBLE, REVIEW)},
                    full_counts={k:v for k,v in full.items() if k not in ('tables', 'coverage')},
                    reviewed50_counts={k:v for k,v in subset.items() if k not in ('tables', 'coverage')},
                    conflicts=conflicts, new_semantic_reviews=0, new_external_calls=0,
                    all_semantic_labels_complete=False)
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    for label in (1, 0):
        print('\nACC='+str(label)+' full currently labelled, B100-B600')
        print('\n'.join(format_table([r for r in full['tables'][label] if r['budget'] in ('100','200','300','400','500','600')])))


if __name__ == '__main__':
    main()
