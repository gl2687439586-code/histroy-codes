"""Finalize previously computed READ-signal audits; no model/API calls.

Only stored outcomes and out-of-fold decisions are required for the extra
contrasts and figures. This does not select a new policy on held-out outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest


DEFAULT = Path(__file__).resolve().parent / 'readscore1942_prior_analysis'


def bootstrap(delta, seed=20260930, reps=10000):
    rng = np.random.default_rng(seed)
    delta = np.asarray(delta)
    estimates = []
    for start in range(0, reps, 250):
        draws = rng.integers(0, len(delta), size=(min(250, reps-start), len(delta)))
        estimates.extend(delta[draws].mean(axis=1))
    return np.quantile(estimates, [.025, .975]).tolist()


def contrast(z, selected, reference, subset=None):
    idx = np.arange(len(z['ids'])) if subset is None else np.asarray(subset)
    s = np.asarray(selected)[idx]
    r = np.full(len(idx), reference, int) if np.isscalar(reference) else np.asarray(reference)[idx]
    out = {'n': len(idx)}
    for key in ['acc', 'em', 'f1', 'tokens']:
        value, ref = z[key][idx, s], z[key][idx, r]
        delta = value-ref
        out[key] = {'mean': float(value.mean()), 'reference': float(ref.mean()),
                    'delta': float(delta.mean()), 'ci95': bootstrap(delta)}
    acc, ref = z['acc'][idx, s], z['acc'][idx, r]
    gain = int(((acc == 1) & (ref == 0)).sum())
    harm = int(((acc == 0) & (ref == 1)).sum())
    out.update(gain=gain, harm=harm,
               mcnemar_p=float(binomtest(gain, gain+harm, .5).pvalue) if gain+harm else 1.)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis-dir', type=Path, default=DEFAULT)
    args = ap.parse_args()
    out = args.analysis_dir
    load = lambda name: json.loads((out/name).read_text(encoding='utf-8'))
    d, main_cv, conditional = (load(name) for name in
                               ['descriptive.json', 'nested_cv_20260930.json', 'conditional_policy.json'])
    z = np.load(out/'outcomes.npz')
    first = np.load(out/'cv_predictions_20260930.npz')
    cp = np.load(out/'conditional_predictions.npz')
    assert np.array_equal(z['ids'], first['ids']) and np.array_equal(z['ids'], cp['ids'])
    n = len(z['ids'])
    policies = {'fixed_600': np.full(n, 5, int),
                'read_semantic_neighbor': first['read_semantic_neighbor'],
                'read_geometry': first['read_geometry'],
                'nested_family_selector': first['nested_family_selector'],
                'conditional_with_read': cp['conditional_with_read'],
                'conditional_without_read': cp['conditional_without_read']}
    extra = {'protocol': 'paired contrasts from frozen OOF decisions; no refitting/selection',
             'fixed_600_vs_full': contrast(z, policies['fixed_600'], 6),
             'vs_fixed_600': {k: contrast(z, v, 5) for k, v in policies.items() if k != 'fixed_600'}}
    regime = np.where(z['acc'].max(1) == 0, 'always_wrong',
                      np.where(z['acc'].min(1) == 1, 'always_correct', 'sensitive'))
    extra['regime_effects'] = {
        name: {group: contrast(z, ch, 6, np.flatnonzero(regime == group))
               for group in ['always_wrong', 'always_correct', 'sensitive']}
        for name, ch in policies.items()}
    # Count READ calls only where the trained semantic route actually requires it.
    # A non-READ route or constant FULL fallback does not need teacher scoring.
    read_mask = np.zeros(n, bool)
    for fold in conditional['fold_selections']:
        for group in ['bridge', 'comparison']:
            choice = fold['routes'][group+'_with_read']
            if choice.get('name', '').startswith('read_'):
                read_mask |= ((cp['fold_id'] == fold['fold']) &
                              ((cp['pred_type'] >= .5) == (group == 'comparison')))
    extra['online_cost_bookkeeping'] = {
        'warning': 'GPT and Qwen token units are not identical; use real money/latency for deployment costs.',
        'teacher_model': 'openai/gpt-5.6-sol',
        'mean_teacher_tokens_all_questions': float(z['score_tokens'].mean()),
        'best_neighbor_cached_qwen_tokens': main_cv['policies']['read_semantic_neighbor']['tokens']['mean'],
        'best_neighbor_fresh_all_teacher_plus_qwen_tokens': float(main_cv['policies']['read_semantic_neighbor']['tokens']['mean'] + z['score_tokens'].mean()),
        'conditional_teacher_needed_n': int(read_mask.sum()),
        'conditional_teacher_needed_fraction': float(read_mask.mean()),
        'conditional_mean_teacher_tokens_only_needed': float((z['score_tokens']*read_mask).mean()),
        'conditional_fresh_selective_teacher_plus_qwen_tokens': float(conditional['policies']['conditional_with_read']['tokens']['mean'] + (z['score_tokens']*read_mask).mean()),
        'total_teacher_cost_usd_already_recorded': d['total_score_cost_usd'],
    }
    meta = [json.loads(line) for line in (out/'joined_questions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    assert [x['sample_id'] for x in meta] == list(z['ids'])
    extra['root_gold_coverage_descriptive_only'] = {}
    for group in ['all', 'bridge', 'comparison']:
        extra['root_gold_coverage_descriptive_only'][group] = {}
        for score in [1, 2, 3]:
            ix = np.flatnonzero((z['score'] == score) & ((z['types'] == group) if group != 'all' else True))
            extra['root_gold_coverage_descriptive_only'][group][score] = {
                'n': len(ix),
                'mean_root_gold_title_recall': float(np.mean([meta[i]['root_title_recall'] for i in ix])),
                'mean_root_gold_sentence_fuzzy_recall': float(np.mean([meta[i]['root_fuzzy_recall'] for i in ix])),
            }
    (out/'final_contrasts.json').write_text(json.dumps(extra, ensure_ascii=False, indent=2), encoding='utf-8')

    # Research figure: difficulty association, conditional budget effects, and cost.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.labelcolor': '#263648',
                         'text.color': '#263648', 'axes.edgecolor': '#a2adba',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.7), layout='constrained')
    colors = ['#17898b', '#5374bd', '#e18b45']
    budget_labels = d['budgets']
    for score, color in zip([1, 2, 3], colors):
        g = d['read_groups'][str(score)]
        label = f"READ {'3+' if score == 3 else score} (n={g['n']:,})"
        for ax, metric in zip(axes[0], ['acc', 'tokens']):
            values = [g['curves'][b][metric]['mean']*(100 if metric == 'acc' else 1) for b in budget_labels]
            ax.plot(np.arange(7), values, marker='o', lw=2.1, ms=5, color=color, label=label)
            ax.set_xticks(np.arange(7), budget_labels)
            ax.grid(axis='y', color='#e7ebf0', lw=.7)
            ax.set_xlabel('Per-search evidence exposure condition')
    axes[0, 0].set_title('A  READ demand is not monotonic answerability', loc='left', pad=12)
    axes[0, 0].set_ylabel('Answer ACC (%)')
    axes[0, 0].legend(frameon=False, loc='lower right', fontsize=9)
    axes[0, 1].set_title('B  Higher READ demand predicts higher compute', loc='left', pad=12)
    axes[0, 1].set_ylabel('Mean Qwen input + output tokens / question')

    names = ['Fixed B600', 'READ + semantic neighbors', 'READ + retrieval geometry',
             'Type-routed, with READ', 'Type-routed, without READ']
    vals = [extra['fixed_600_vs_full'], main_cv['policies']['read_semantic_neighbor'],
            main_cv['policies']['read_geometry'], conditional['policies']['conditional_with_read'],
            conditional['policies']['conditional_without_read']]
    ax = axes[1, 0]
    for i, (name, v) in enumerate(zip(names, vals)):
        delta = v['acc']['delta']*100
        lo, hi = np.asarray(v['acc']['ci95'])*100
        color = '#596b80' if i in [0, 4] else '#17898b'
        ax.errorbar(delta, i, xerr=[[delta-lo], [hi-delta]], fmt='o', color=color,
                    capsize=4, ms=6, lw=1.8)
    ax.axvline(0, color='#99a6b4', ls='--', lw=1)
    ax.set_yticks(np.arange(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel('ACC change vs FULL (percentage points; paired 95% CI)')
    ax.set_title('C  No demonstrated ACC improvement', loc='left', pad=12)
    ax.grid(axis='x', color='#e7ebf0', lw=.7)

    ax = axes[1, 1]
    cost = extra['online_cost_bookkeeping']
    qwen = [d['full']['tokens']['mean'], d['fixed']['600']['tokens']['mean'],
            main_cv['policies']['read_semantic_neighbor']['tokens']['mean'],
            conditional['policies']['conditional_with_read']['tokens']['mean']]
    overhead = [0, 0, cost['mean_teacher_tokens_all_questions'],
                cost['conditional_mean_teacher_tokens_only_needed']]
    x = np.arange(4)
    ax.bar(x, qwen, color='#5374bd', width=.6, label='Qwen generation: input + output')
    ax.bar(x, overhead, bottom=qwen, color='#e18b45', width=.6, label='Fresh teacher READ scoring')
    for i, (value, over) in enumerate(zip(qwen, overhead)):
        ax.text(i, value+over+75, f'{value+over:,.0f}', ha='center', fontsize=10)
    ax.set_xticks(x, ['FULL', 'Fixed\nB600', 'READ +\nneighbors', 'Selective\nREAD route'])
    ax.set_ylim(0, 6900)
    ax.set_title('D  Fresh scoring erases small token savings', loc='left', pad=12)
    ax.set_ylabel('Token-count bookkeeping / question*')
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    ax.grid(axis='y', color='#e7ebf0', lw=.7)
    ax.set_axisbelow(True)
    fig.suptitle('READ-demand priors: useful difficulty information, weak budget utility',
                 fontsize=17, fontweight='bold')
    fig.supxlabel('1,942 matched HotpotQA questions | Frozen out-of-fold routes | Exploratory, not a confirmatory holdout\n'
                  '* Teacher and Qwen token units differ. Cached-score routing excludes teacher cost; bars add fresh scoring.',
                  fontsize=9, color='#667689')
    fig.savefig(out/'read_signal_audit.png', dpi=220)
    fig.savefig(out/'read_signal_audit.svg')
    plt.close(fig)
    print(json.dumps({'fixed_600_vs_full': extra['fixed_600_vs_full'],
                      'vs_fixed_600': extra['vs_fixed_600'],
                      'online_cost_bookkeeping': cost,
                      'root_gold_coverage_descriptive_only': extra['root_gold_coverage_descriptive_only']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
