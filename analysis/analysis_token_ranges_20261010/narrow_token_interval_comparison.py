"""Compare 80/70/60/50% shortest intervals without altering audited labels."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

from tight_token_intervals import read_gz, shortest_interval, format_tables, MAIN_BUDGETS, self_test

BASE = Path(__file__).resolve().parent
OUT = BASE / 'narrow_token_intervals'
PCTS = (80, 70, 60, 50)


def comparison_tables(tables, main_only=False):
    lines = []
    for label in ('1', '0'):
        lines += [f'ACC={label}', '',
                  '| 每轮证据预算 | Search steps | 题数 | 最窄80%区间 | 最窄70%区间 | 最窄60%区间 | 最窄50%区间 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for row in tables[label]:
            if main_only and row['budget'] not in MAIN_BUDGETS:
                continue
            intervals = []
            for p in PCTS:
                interval = row['intervals'][str(p)]
                intervals.append(f"{interval['lower']:,}–{interval['upper']:,}" if interval else '—')
            lines.append(f"| {row['budget']} | {row['search_steps']} | {row['n']:,} | " + ' | '.join(intervals) + ' |')
        lines += ['']
    return lines


def main():
    self_test()
    OUT.mkdir(exist_ok=True)
    source = BASE / 'full_semantic_review/final_label_ledger.jsonl.gz'
    previous = json.loads((BASE / 'tight_token_intervals/summary.json').read_text(encoding='utf-8'))
    groups = defaultdict(list)
    rows = list(read_gz(source))
    assert len(rows) == 58868
    assert len({(r['sample_id'], r['condition_id']) for r in rows}) == 58868
    for row in rows:
        assert row['combined_acc'] in (0, 1)
        assert not row['root_cap_hit'] and not row['continuation_cap_hit']
        groups[str(row['combined_acc']), row['budget'], row['search_calls']].append(row)
    tables = {'1': [], '0': []}
    for label in ('1', '0'):
        for old in previous['tables'][label]:
            group = groups[label, old['budget'], old['search_steps']]
            assert len(group) == old['n'] == len({r['sample_id'] for r in group})
            intervals = {str(p): shortest_interval([r['total_tokens'] for r in group], p) for p in PCTS}
            assert intervals['80'] == old['intervals']['80']
            assert intervals['50'] == old['intervals']['50']
            if group:
                assert intervals['80']['width'] >= intervals['70']['width'] >= intervals['60']['width'] >= intervals['50']['width']
                for p in PCTS:
                    interval = intervals[str(p)]
                    interval['width_reduction_vs_80'] = (1 - interval['width']/intervals['80']['width']) if intervals['80']['width'] else None
            tables[label].append({**old, 'intervals': intervals})
    assert sum(r['n'] for r in tables['1']) == 34888
    assert sum(r['n'] for r in tables['0']) == 23980
    main_reductions = {}
    for label in ('1', '0'):
        selected = [r for r in tables[label] if r['budget'] in MAIN_BUDGETS]
        main_reductions[label] = {
            str(p): {
                'groups': len(selected),
                'min_group_width_reduction': min(r['intervals'][str(p)]['width_reduction_vs_80'] for r in selected),
                'max_group_width_reduction': max(r['intervals'][str(p)]['width_reduction_vs_80'] for r in selected),
                'unweighted_mean_group_width_reduction': sum(r['intervals'][str(p)]['width_reduction_vs_80'] for r in selected)/len(selected)
            } for p in (70, 60, 50)
        }
    result = dict(records=58868, acc1=34888, acc0=23980,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  coverage_percentages=PCTS, original_80_percent_results_unchanged=True,
                  minimum_width_and_coverage_checks_passed=True,
                  labels_changed=0, additional_exclusions=0,
                  main_groups_width_reductions=main_reductions, tables=tables)
    (OUT/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    intro = [
        'HotpotQA最窄token区间：80% / 70% / 60% / 50%覆盖率对比', '',
        '全量58,868条，ACC=1 34,888条，ACC=0 23,980条；保持上轮原金标语义评审和触顶剔除口径。',
        '在每个预算×Search steps×ACC组内，寻找覆盖至少ceil(n*p/100)题的最小宽度闭区间。',
        '端点相同的token值不拆分，所以实际覆盖率可能高于目标；实际覆盖题数见各覆盖率独立表与summary.json。',
        '等宽时优先覆盖题数更多的区间，仍并列则取下界更小者。',
        '这不是等尾分位数区间、均值置信区间或未来题目的覆盖保证。区间可以偏向分布一侧。',
        '覆盖率降低时，最窄区间的位置可以移动，不保证50%区间嵌套在60%或80%区间内；只保证最小宽度不增加。',
        '80%、70%、60%、50%分别描述该组至少八成、七成、六成、半数题目；未落入区间的轨迹没有从统计中删除。',
        '总tokens仍为独立轨迹输入+输出；ACC和实际Search次数是运行后属性，不可直接作运行前截断规则。',
        '低样本量组需谨慎，特别是n<30；B700–2700的题目集合更小，不能当作同题总体对比。', ''
    ]
    for main_only in (True, False):
        suffix = 'MAIN' if main_only else 'ALL'
        (OUT/f'COMPARISON_{suffix}_ZH.txt').write_text('\n'.join(intro + comparison_tables(tables, main_only))+'\n', encoding='utf-8')
    for p in (70, 60, 50):
        (OUT/f'ACC_TABLES_SHORTEST{p}_ALL_ZH.txt').write_text('\n'.join(intro + format_tables(tables, p))+'\n', encoding='utf-8')
    paths = sorted(OUT.glob('*.txt')) + [OUT/'summary.json', Path(__file__).resolve(), BASE/'tight_token_intervals.py']
    manifest = {str(p.relative_to(BASE)).replace('\\', '/'): {
        'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    paths.append(OUT/'MANIFEST.json')
    archive = BASE/'hotpot_semantic_acc_narrow_intervals_50_60_70_20261010.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            bundle.write(path, path.relative_to(BASE))
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
    print(json.dumps(dict(passed=True, records=58868, archive=str(archive), main_groups_width_reductions=main_reductions), indent=2))
    print('\n'.join(comparison_tables(tables, True)))


if __name__ == '__main__':
    main()
