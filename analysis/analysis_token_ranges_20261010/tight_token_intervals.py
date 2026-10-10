"""Shortest empirical token intervals, preserving the completed semantic audit.

No model calls, resampling, label changes, or removal of additional trajectories.
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import statistics
import zipfile

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'full_semantic_review'
OUT = BASE / 'tight_token_intervals'
PERCENTAGES = (50, 80, 90, 95)
MAIN_BUDGETS = {'100', '200', '300', '400', '500', '600', 'FULL'}


def read_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def shortest_interval(values, percent):
    """Minimize width subject to >= ceil(percent*n/100) observations.

    Ties prefer more covered observations, then the smaller lower endpoint.
    Endpoints are inclusive and equal-valued tokens are never split.
    """
    x = sorted(values)
    n = len(x)
    if not n:
        return None
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in x)
    k = (percent * n + 99) // 100
    best = None
    for i in range(n - k + 1):
        lo, hi = x[i], x[i + k - 1]
        covered = bisect_right(x, hi) - bisect_left(x, lo)
        candidate = (hi - lo, -covered, lo, hi)
        if best is None or candidate < best:
            best = candidate
    width, negative_covered, lo, hi = best
    covered = -negative_covered
    assert covered >= k
    # Independent two-pointer check: no shorter integer-width interval covers k.
    if width > 0:
        right = 0
        for left in range(n):
            while right < n and x[right] - x[left] < width:
                right += 1
            assert right - left < k
    actual = sum(lo <= v <= hi for v in values)
    assert actual == covered
    full_width = x[-1] - x[0]
    return dict(target_percent=percent, minimum_required_n=k,
                lower=lo, upper=hi, width=width, covered_n=covered,
                actual_coverage=covered/n, below_n=bisect_left(x, lo),
                above_n=n-bisect_right(x, hi),
                width_fraction_of_full=width/full_width if full_width else None,
                low_n_warning=n < 30)


def self_test():
    rng = random.Random(20261010)
    assert shortest_interval([], 80) is None
    assert shortest_interval([5], 80)['covered_n'] == 1
    assert shortest_interval([5] * 10, 80)['covered_n'] == 10
    for _ in range(200):
        x = [rng.randrange(25) for _ in range(rng.randrange(1, 30))]
        for percent in PERCENTAGES:
            result = shortest_interval(x, percent)
            k = (percent*len(x)+99)//100
            candidates = []
            for lo in set(x):
                for hi in set(x):
                    if hi < lo:
                        continue
                    covered = sum(lo <= v <= hi for v in x)
                    if covered >= k:
                        candidates.append((hi-lo, -covered, lo, hi))
            assert (result['width'], -result['covered_n'], result['lower'],
                    result['upper']) == min(candidates)


def format_tables(tables, percent, main_only=False):
    lines = []
    for label in (1, 0):
        lines += [f'表{1 if label else 2}：ACC={label}', '',
                  f'| 每轮证据预算 | 搜索 step 数 | 题数 | 平均总 tokens | 总 tokens 范围：最小–最大 | 覆盖至少{percent}%题目的最窄区间（实际覆盖题数） |',
                  '|---|---:|---:|---:|---:|---:|']
        for row in tables[str(label)]:
            if main_only and row['budget'] not in MAIN_BUDGETS:
                continue
            if row['n']:
                mean = f"{row['mean_total_tokens']:,.1f}"
                full = f"{row['min_total_tokens']:,}–{row['max_total_tokens']:,}"
                interval = row['intervals'][str(percent)]
                tight = f"{interval['lower']:,}–{interval['upper']:,}（{interval['covered_n']:,}/{row['n']:,}）"
            else:
                mean = full = tight = '—'
            lines.append(f"| {row['budget']} | {row['search_steps']} | {row['n']:,} | {mean} | {full} | {tight} |")
        lines += ['']
    return lines


def main():
    self_test()
    OUT.mkdir(exist_ok=True)
    source = SOURCE/'final_label_ledger.jsonl.gz'
    rows = list(read_gz(source))
    old = json.loads((SOURCE/'full_summary.json').read_text(encoding='utf-8'))
    assert len(rows) == 58868
    assert len({(r['sample_id'], r['condition_id']) for r in rows}) == len(rows)
    groups = defaultdict(list)
    for row in rows:
        assert row['combined_acc'] in (0, 1)
        assert not row['root_cap_hit'] and not row['continuation_cap_hit']
        groups[row['budget'], row['search_calls'], row['combined_acc']].append(row)
    tables = {'1': [], '0': []}
    for label in (1, 0):
        for prior in old['tables'][str(label)]:
            group = groups[prior['budget'], prior['search_steps'], label]
            assert len(group) == len({r['sample_id'] for r in group}) == prior['n']
            costs = [r['total_tokens'] for r in group]
            if costs:
                assert statistics.mean(costs) == prior['mean_total_tokens']
                assert min(costs) == prior['min_total_tokens']
                assert max(costs) == prior['max_total_tokens']
            tables[str(label)].append({**prior, 'intervals': {
                str(p): shortest_interval(costs, p) for p in PERCENTAGES}})
    for label, expected in (('1', 34888), ('0', 23980)):
        assert sum(r['n'] for r in tables[label]) == expected
    result = dict(records=len(rows), acc1=34888, acc0=23980,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  definition='Shortest inclusive empirical interval covering at least ceil(n*p/100) observations; ties: more observations, then lower endpoint.',
                  selected_main_coverage_percent=80, alternative_coverage_percentages=list(PERCENTAGES),
                  validation='All intervals pass exact coverage and independent minimum-width checks; randomized exhaustive small-data tests passed.',
                  predictive_guarantee=False, labels_changed=0, additional_exclusions=0,
                  tables=tables)
    (OUT/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    introduction = [
        'HotpotQA：ACC分组 × 证据预算 × Search步骤的最窄token覆盖区间', '',
        '保留上次审计全部58,868条轨迹：ACC=1 34,888；ACC=0 23,980。标签和排除规则不变。',
        '严格对照原金标：EM=1或Codex语义评审正确记ACC=1，其余已评审答案记ACC=0。语义评审不是人工专家金标。',
        '总tokens=独立轨迹LLM输入+输出；root计一次、重复历史输入计入；不是单次生成长度。',
        '最小宽度与最大覆盖无法同时无约束优化，因此主表固定至少80%覆盖，再最小化区间宽度。另提供50%、90%、95%用于敏感性比较。',
        '算法：排序x，k=ceil(p*n)，遍历[x[i],x[i+k-1]]，最小化上界减下界；等宽时优先实际覆盖更多题，再取下界较小者。',
        '端点包含在区间内；相同token值不可拆分，因此实际覆盖可能超过目标比例。附实际覆盖题数。',
        '这不是等尾10%–90%分位数区间，也不是均值的置信区间，而是样本内最集中区间；不保证未来题目具有同样覆盖。',
        '少于30题的分组只能描述现有样本；全相同token时区间宽度为0。区间外轨迹未删除，也没有重新计算均值。',
        '同一道题可以跨预算重复；B700–2700只覆盖更小题目子集。即使B100–600和FULL，剔除触顶后也可能留下不同题目。',
        '正确/错误分组及Search次数都是运行后属性，不能把该区间直接当作运行前的预算预测器或硬截断阈值。', ''
    ]
    for percent in (80, 90):
        for main_only in (False, True):
            suffix = '_MAIN' if main_only else '_ALL'
            path = OUT/f'ACC_TABLES_SHORTEST{percent}{suffix}_ZH.txt'
            path.write_text('\n'.join(introduction+format_tables(tables, percent, main_only))+'\n', encoding='utf-8')
    files = [Path(__file__).resolve()] + sorted(OUT.glob('*.txt')) + [OUT/'summary.json']
    manifest = {str(p.relative_to(BASE)).replace('\\', '/'):
                dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    files.append(OUT/'MANIFEST.json')
    archive = BASE/'hotpot_semantic_acc_tight_token_intervals_20261010.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as stream:
        for path in files:
            stream.write(path, path.relative_to(BASE))
    with zipfile.ZipFile(archive) as stream:
        assert stream.testzip() is None
    print(json.dumps(dict(passed=True, records=len(rows), cells_per_table=len(tables['1']),
                          archive=str(archive), source_hash=result['source_sha256']), indent=2))
    print('\n'.join(format_tables(tables, 80, True)))


if __name__ == '__main__':
    main()
