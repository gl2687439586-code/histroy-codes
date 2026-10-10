"""Stratified READ-demand audit with unconstrained and inverse budget mappings.

Regimes are retrospective diagnostics, not deployable features. Mapping tests
are fitted on outer training questions only, without knowing a test regime.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import readscore1942_prior_analysis as base


OUT = Path(__file__).resolve().parent/'readscore1942_regime_audit'


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return {'rho': None, 'p': None, 'reason': 'constant/insufficient variation'}
    r = spearmanr(x, y)
    return {'rho': float(r.statistic), 'p': float(r.pvalue)}


def grouped_summary(data, ix, reg, choices):
    n = len(data['ids'])
    a = data['acc'][ix]
    out = {'n': len(ix), 'score_counts': dict(Counter(data['score'][ix].astype(int).tolist())),
           'type_counts': dict(Counter(data['types'][ix])),
           'full_mean_tokens': float(data['tokens'][ix, 6].mean()),
           'full_mean_searches': float(data['searches'][ix, 6].mean()),
           'root_gold_title_recall': float(np.mean([data['meta'][i]['root_title_recall'] for i in ix])),
           'root_gold_fuzzy_sentence_recall': float(np.mean([data['meta'][i]['root_fuzzy_recall'] for i in ix])),
           'curves': {b: base.evaluate(data, np.full(n, j), ix)
                      for j, b in enumerate(base.BUDGETS)},
           'budget_distributions': {}, 'correlations': {}, 'score_groups': {}}
    for key, ch in choices.items():
        if key in ['minimum_success', 'minimum_stable_success', 'lowest_cost_correct', 'quality_then_cost'] and reg == 'always_wrong':
            continue
        valid = ix[ch[ix] >= 0]
        out['budget_distributions'][key] = {
            'n_defined': len(valid), 'undefined': len(ix)-len(valid),
            'distribution': dict(Counter(base.BUDGETS[j] for j in ch[valid])),
            'mean_total_tokens': float(data['tokens'][valid, ch[valid]].mean()) if len(valid) else None,
            'mean_selected_budget_index': float(ch[valid].mean()) if len(valid) else None,
        }
        if len(valid):
            out['correlations'][key] = correlation(data['score'][valid], ch[valid])
    out['correlations']['full_total_tokens'] = correlation(data['score'][ix], data['tokens'][ix, 6])
    if reg != 'always_wrong':
        out['correctness_oracles'] = {
            key: base.evaluate(data, ch, ix) for key, ch in choices.items()
            if key in ['lowest_cost_correct', 'quality_then_cost']}
    out['cost_oracle_without_quality_constraint'] = base.evaluate(data, choices['lowest_cost_any'], ix)
    out['cost_oracle_preserving_full_acc_em_f1'] = base.evaluate(data, choices['lowest_cost_preserving_full'], ix)
    if reg == 'sensitive':
        full_correct = a[:, 6] == 1
        flips = np.sum(a[:, 1:] != a[:, :-1], axis=1)
        out['flip_diagnostics'] = {
            'full_correct': int(full_correct.sum()), 'full_wrong': int((~full_correct).sum()),
            'correct_b100_but_full_wrong': int(((a[:, 0] == 1) & ~full_correct).sum()),
            'some_reduced_budget_correct_but_full_wrong': int(((a[:, :6].max(1) == 1) & ~full_correct).sum()),
            'has_correct_to_wrong_adjacent_transition': int(((a[:, :-1] == 1) & (a[:, 1:] == 0)).any(1).sum()),
            'mean_adjacent_flips': float(flips.mean()),
            'at_least_two_flips': int((flips >= 2).sum()),
            'success_by_b200': int((a[:, :2].max(1) == 1).sum()),
            'success_by_b300': int((a[:, :3].max(1) == 1).sum()),
            'lowest_cost_correct_budget_at_most_b200': int((choices['lowest_cost_correct'][ix] <= 1).sum()),
        }
    for score in [1, 2, 3]:
        si = ix[data['score'][ix] == score]
        if not len(si):
            continue
        full = data['acc'][si, 6]
        g = {'n': len(si), 'full_mean_tokens': float(data['tokens'][si, 6].mean()),
             'full_mean_acc': float(full.mean()), 'curves': {}, 'distributions': {}}
        for j, b in enumerate(base.BUDGETS):
            m = base.evaluate(data, np.full(n, j), si)
            g['curves'][b] = {k: m[k] for k in ['acc', 'em', 'f1', 'tokens', 'searches']}
            g['curves'][b].update(gain=m['gain'], harm=m['harm'])
        for key in ['minimum_success', 'lowest_cost_correct', 'minimum_stable_success', 'quality_then_cost']:
            valid = si[choices[key][si] >= 0]
            g['distributions'][key] = {
                'n_defined': len(valid), 'undefined': len(si)-len(valid),
                'budget_counts': dict(Counter(base.BUDGETS[j] for j in choices[key][valid])),
            }
        if reg == 'sensitive':
            sa = data['acc'][si]
            g['small_budget_examples'] = {
                'b100_correct': int(sa[:, 0].sum()),
                'success_by_b200': int((sa[:, :2].max(1) == 1).sum()),
                'lowest_cost_correct_budget_at_most_b200': int((choices['lowest_cost_correct'][si] <= 1).sum()),
                'full_wrong_but_some_reduced_budget_correct': int((sa[:, 6] == 0).sum()),
            }
        out['score_groups'][score] = g
    return out


def mapping_cv(data, seed):
    """Explicitly compare ascending, descending, and arbitrary READ mappings.

    Seven budgets for each of the three READ groups: 343 unconstrained maps.
    Direction is optional, never built into the main learned models. Mapping
    selection uses only training-fold outcomes; FULL fallback is always allowed.
    """
    n = len(data['ids']); idx = np.arange(n)
    mappings = np.asarray(list(itertools.product(range(7), repeat=3)))
    group = data['score'].astype(int)-1
    routes = mappings[:, group]
    masks = {
        'unconstrained': np.ones(len(mappings), bool),
        'nondecreasing': (mappings[:, 0] <= mappings[:, 1]) & (mappings[:, 1] <= mappings[:, 2]),
        'nonincreasing': (mappings[:, 0] >= mappings[:, 1]) & (mappings[:, 1] >= mappings[:, 2]),
    }
    labels = np.char.add(data['types'], data['score'].astype(int).astype(str))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    chosen = {key: np.full(n, 6, int) for key in masks}; folds = []
    for fold, (train, test) in enumerate(cv.split(idx, labels)):
        a = (data['acc'][train[None, :], routes[:, train]]-data['acc'][train, 6]).mean(1)
        f = (data['f1'][train[None, :], routes[:, train]]-data['f1'][train, 6]).mean(1)
        t = (data['tokens'][train[None, :], routes[:, train]]-data['tokens'][train, 6]).mean(1)
        eligible = (a >= 0) & (f >= 0) & (t <= -100)
        rec = {'fold': fold, 'choices': {}}
        for name, mask in masks.items():
            candidates = np.flatnonzero(mask & eligible)
            if len(candidates):
                j = max(candidates, key=lambda i: (a[i]+.2*f[i], -t[i]))
                chosen[name][test] = routes[j, test]
                rec['choices'][name] = {'mapping_score1_score2_score3': [base.BUDGETS[k] for k in mappings[j]],
                                       'train_acc_delta': float(a[j]), 'train_f1_delta': float(f[j]),
                                       'train_tokens_delta': float(t[j])}
            else:
                rec['choices'][name] = {'fallback': 'FULL'}
        folds.append(rec)
    report = {'seed': seed, 'mapping_count': {k: int(v.sum()) for k, v in masks.items()},
              'protocol': '5-fold crossfit; direction/mapping selected on training only; no test-regime feature',
              'fold_choices': folds, 'policies': {k: base.evaluate(data, v, ci=True) for k, v in chosen.items()}}
    return report, chosen


def main():
    data = base.load_data(); n = len(data['ids']); idx = np.arange(n); a = data['acc']
    regimes = np.where(a.max(1) == 0, 'always_wrong', np.where(a.min(1) == 1, 'always_correct', 'sensitive'))
    em_regimes = np.where(data['em'].max(1) == 0, 'always_wrong', np.where(data['em'].min(1) == 1, 'always_correct', 'sensitive'))
    any_success = a.max(1) == 1
    minimum_success = np.where(any_success, a.argmax(1), -1)
    stable_eligible = np.flip(np.cumprod(np.flip(a, axis=1), axis=1), axis=1)
    stable_defined = stable_eligible.max(1) == 1
    minimum_stable = np.where(stable_defined, stable_eligible.argmax(1), -1)
    lowest_correct = np.where(any_success, np.argmin(np.where(a == 1, data['tokens'], np.inf), axis=1), -1)
    quality = np.asarray([max(range(7), key=lambda j: (a[i, j], data['f1'][i, j], -data['tokens'][i, j])) for i in idx])
    quality = np.where(any_success, quality, -1)
    keep_full = ((a >= a[:, [6]]) & (data['em'] >= data['em'][:, [6]]) & (data['f1'] >= data['f1'][:, [6]]))
    choices = {'minimum_success': minimum_success, 'minimum_stable_success': minimum_stable,
               'lowest_cost_correct': lowest_correct, 'quality_then_cost': quality,
               'lowest_cost_any': data['tokens'].argmin(1),
               'lowest_cost_preserving_full': np.argmin(np.where(keep_full, data['tokens'], np.inf), axis=1)}
    report = {
        'n': n, 'budgets': base.BUDGETS,
        'regime_definition': 'all seven strict lexical answer_acc; not identical to EM=1',
        'warning': 'regimes and all oracles are hindsight-only; subgroup conditioning may introduce selection/collider effects',
        'regime_counts_acc': dict(Counter(regimes)), 'regime_counts_em': dict(Counter(em_regimes)),
        'prior_model_direction': 'no monotonic restriction; all seven budgets compete independently',
        'definitions': {
            'minimum_success': 'first correct exposure condition; undefined for always-wrong',
            'lowest_cost_correct': 'minimum full-trajectory LLM input+output cost among correct conditions; no F1 tie priority',
            'quality_then_cost': 'ACC, then F1, then lower total tokens; undefined for always-wrong in this correction',
            'minimum_stable_success': 'first budget after which every later condition is correct; undefined if FULL wrong',
            'lowest_cost_any': 'cost-only diagnostic; not a successful-answer oracle for always-wrong',
        }, 'groups': {}, 'paired_reduced_budget_vs_full': {}, 'mapping_cv': {}, 'high_read_pair_cases': {},
        'strict_em_robustness': {}}
    for group in ['always_correct', 'always_wrong', 'sensitive']:
        ix = np.flatnonzero(regimes == group)
        report['groups'][group] = grouped_summary(data, ix, group, choices)
        report['groups'][group]['by_type'] = {
            ty: grouped_summary(data, ix[data['types'][ix] == ty], group, choices)
            for ty in ['bridge', 'comparison']}
    em = data['em']
    em_first = np.where(em.max(1) == 1, em.argmax(1), -1)
    em_lowest = np.where(em.max(1) == 1, np.argmin(np.where(em == 1, data['tokens'], np.inf), axis=1), -1)
    for group in ['always_correct', 'always_wrong', 'sensitive']:
        ix = np.flatnonzero(em_regimes == group)
        v = {'n': len(ix), 'curves': {
            b: {k: float(data[k][ix, j].mean()) for k in ['em', 'acc', 'f1', 'tokens', 'searches']}
            for j, b in enumerate(base.BUDGETS)}}
        if group != 'always_wrong':
            v['score_vs_min_success'] = correlation(data['score'][ix], em_first[ix])
            v['score_vs_lowest_cost_success_budget'] = correlation(data['score'][ix], em_lowest[ix])
        if group == 'sensitive':
            hi = ix[data['score'][ix] >= 3]
            v['high_read'] = {
                'n': len(hi), 'full_wrong_some_reduced_correct': int((em[hi, 6] == 0).sum()),
                'b100_correct_full_wrong': int(((em[hi, 0] == 1) & (em[hi, 6] == 0)).sum()),
                'b100_wrong_full_correct': int(((em[hi, 0] == 0) & (em[hi, 6] == 1)).sum()),
            }
        report['strict_em_robustness'][group] = v
    high_flip = (data['score'] >= 3) & (regimes == 'sensitive')
    for key, mask in {
        'b100_correct_full_wrong': high_flip & (a[:, 0] == 1) & (a[:, 6] == 0),
        'b100_wrong_full_correct': high_flip & (a[:, 0] == 0) & (a[:, 6] == 1),
        'both_correct': high_flip & (a[:, 0] == 1) & (a[:, 6] == 1),
        'both_wrong': high_flip & (a[:, 0] == 0) & (a[:, 6] == 0),
    }.items():
        ix = np.flatnonzero(mask)
        report['high_read_pair_cases'][key] = {
            'n': len(ix), 'warning': 'selected by hindsight outcomes; descriptive mechanism cases only',
            'b100': {m: float(data[m][ix, 0].mean()) for m in ['tokens', 'input_tokens', 'output_tokens', 'searches', 'visible_tokens']},
            'full': {m: float(data[m][ix, 6].mean()) for m in ['tokens', 'input_tokens', 'output_tokens', 'searches', 'visible_tokens']},
            'mean_root_gold_title_recall': float(np.mean([data['meta'][i]['root_title_recall'] for i in ix])),
            'sample_ids': data['ids'][ix].tolist(),
        }
    # Re-evaluate the old pooled Oracle correlation only where success exists.
    original = np.asarray([max(range(7), key=lambda j: (a[i, j], data['f1'][i, j], -data['tokens'][i, j])) for i in idx])
    report['original_pooled_oracle_correlation'] = correlation(data['score'], original)
    # Frozen first-stage learned policy is disaggregated, not refitted on gold regimes.
    old = np.load(base.OUT/'cv_predictions_20260930.npz')
    assert np.array_equal(old['ids'], data['ids'])
    report['frozen_read_semantic_neighbor_by_regime'] = {
        group: base.evaluate(data, old['read_semantic_neighbor'], np.flatnonzero(regimes == group), ci=True)
        for group in ['always_correct', 'always_wrong', 'sensitive']}
    for b in [1, 4, 5]:
        for group in ['always_correct', 'always_wrong', 'sensitive']:
            ix = np.flatnonzero(regimes == group)
            report['paired_reduced_budget_vs_full'][f'{group}_{base.BUDGETS[b]}'] = base.evaluate(data, np.full(n, b), ix, ci=True)
    OUT.mkdir(parents=True, exist_ok=True)
    all_predictions = {'ids': data['ids'], 'regime': regimes, **choices}
    for seed in [20260930, 20261004]:
        cr, pred = mapping_cv(data, seed)
        report['mapping_cv'][seed] = cr
        all_predictions.update({f'{key}_{seed}': val for key, val in pred.items()})
    np.savez_compressed(OUT/'regime_decisions.npz', **all_predictions)
    (OUT/'regime_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (OUT/'question_audit.jsonl').open('w', encoding='utf-8') as handle:
        for i, meta in enumerate(data['meta']):
            row = dict(meta, regime=regimes[i], correctness_curve=a[i].astype(int).tolist(),
                       tokens_curve=data['tokens'][i].astype(int).tolist(),
                       f1_curve=data['f1'][i].tolist(), search_calls_curve=data['searches'][i].astype(int).tolist())
            row.update({key: None if ch[i] < 0 else base.BUDGETS[ch[i]] for key, ch in choices.items()})
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    print(json.dumps({
        'regimes': report['regime_counts_acc'], 'em_regimes': report['regime_counts_em'],
        'group_correlations': {key: val['correlations'] for key, val in report['groups'].items()},
        'flip': report['groups']['sensitive']['flip_diagnostics'],
        'high_read_flip': report['groups']['sensitive']['score_groups'][3],
        'mapping_cv': {seed: {key: {m: val[m] for m in ['acc','f1','tokens']} for key, val in record['policies'].items()}
                       for seed, record in report['mapping_cv'].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
