"""Literal question-only READ-group leave-one-question-out budget policy.

No API calls. Current question outcomes and total costs are excluded from
both the same-score and global reference means before selecting its budget.
Image Loose quality is interpreted as the existing lexical answer_acc.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


WORK = Path(__file__).resolve().parent
INPUT = WORK/'readscore1942_prior_analysis'
OUT = WORK/'readscore1942_picture_loo'
BUDGETS = ['100', '200', '300', '400', '500', '600', 'FULL']


def load_data():
    z = np.load(INPUT/'outcomes.npz')
    data = {k: z[k] for k in z.files}
    meta = [json.loads(line) for line in (INPUT/'joined_questions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    assert np.array_equal(data['ids'], [m['sample_id'] for m in meta])
    data['raw_score'] = np.asarray([m['read_score'] for m in meta], int)
    data['meta'] = meta
    return data


def predict_loo(data, weight=None, smoothing=.1, grouping='exact', pseudo_count=None, global_only=False):
    n = len(data['ids'])
    weight = np.ones(n) if weight is None else np.asarray(weight, float)
    values = np.stack([data[k] for k in ['acc', 'em', 'f1', 'tokens']], axis=2)
    score = data['raw_score'] if grouping == 'exact' else np.minimum(3, data['raw_score'])
    total = np.einsum('i,ijk->jk', weight, values)
    other_n = weight.sum()-weight
    assert (other_n > 0).all()
    global_mean = (total[None, :, :]-weight[:, None, None]*values)/other_n[:, None, None]
    if global_only:
        return global_mean
    pred = np.empty_like(values)
    for group in np.unique(score):
        ix = np.flatnonzero(score == group)
        count = weight[ix].sum()-weight[ix]
        sums = np.einsum('i,ijk->jk', weight[ix], values[ix])
        numer = sums[None, :, :]-weight[ix, None, None]*values[ix]
        mean = numer/np.maximum(count[:, None, None], 1.)
        mix = np.full(len(ix), smoothing) if pseudo_count is None else pseudo_count/(count+pseudo_count)
        mix = np.where(count > 0, mix, 1.)
        pred[ix] = (1-mix[:, None, None])*mean+mix[:, None, None]*global_mean[ix]
    return pred


def choose(pred, objective='acc'):
    metric = {'acc': 0, 'em': 1, 'f1': 2}[objective]
    primary = pred[:, :, metric]
    possible = np.isclose(primary, primary.max(1, keepdims=True), rtol=0, atol=1e-12)
    # Plain ACC and F1 policies use cost as their only tie-break. The secondary
    # lexicographic version is reported separately, never silently substituted.
    if objective == 'acc_then_f1':
        raise ValueError('Use choose_acc_f1 for the explicitly labelled variant')
    return np.argmin(np.where(possible, pred[:, :, 3], np.inf), axis=1)


def choose_acc_f1(pred):
    possible = np.isclose(pred[:, :, 0], pred[:, :, 0].max(1, keepdims=True), rtol=0, atol=1e-12)
    f1 = np.where(possible, pred[:, :, 2], -np.inf)
    possible &= np.isclose(f1, f1.max(1, keepdims=True), rtol=0, atol=1e-12)
    return np.argmin(np.where(possible, pred[:, :, 3], np.inf), axis=1)


def bootstrap_delta(delta, reps=10000, seed=20260930):
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, reps, 250):
        ix = rng.integers(0, len(delta), size=(min(250, reps-start), len(delta)))
        out.extend(delta[ix].mean(1))
    return np.quantile(out, [.025, .975]).tolist()


def evaluate(data, chosen, index=None, reference=6, ci=False):
    ix = np.arange(len(chosen)) if index is None else np.asarray(index)
    ch = chosen[ix]
    output = {'n': len(ix), 'budgets': dict(Counter(BUDGETS[i] for i in ch)), 'reference_budget': BUDGETS[reference]}
    for name in ['acc', 'em', 'f1', 'tokens']:
        selected = data[name][ix, ch]; ref = data[name][ix, reference]
        delta = selected-ref
        output[name] = {'mean': float(selected.mean()), 'reference': float(ref.mean()), 'delta': float(delta.mean())}
        if ci:
            output[name]['ci95_fixed_oof_decisions'] = bootstrap_delta(delta)
    selected = data['acc'][ix, ch]; ref = data['acc'][ix, reference]
    gain = int(((selected == 1) & (ref == 0)).sum())
    harm = int(((selected == 0) & (ref == 1)).sum())
    output.update(gain=gain, harm=harm,
                  mcnemar_p=float(binomtest(gain, gain+harm, .5).pvalue) if gain+harm else 1.)
    return output


def refit_bootstrap(data, reps=2000, seed=20260930, objective='acc'):
    """Bootstrap questions and refit the entire pictured LOO mapping.

    All bootstrap copies of the held-out original question are removed from
    its reference set. Removing only one copy would leak its own outcomes.
    """
    rng = np.random.default_rng(seed); n = len(data['ids']); idx = np.arange(n)
    estimates = {k: [] for k in ['acc', 'em', 'f1', 'tokens']}
    budget_hist = []
    for _ in range(reps):
        w = rng.multinomial(n, np.full(n, 1/n)).astype(float)
        pred = predict_loo(data, w, smoothing=.1)
        ch = choose(pred, objective)
        for metric in estimates:
            estimates[metric].append(float((w*(data[metric][idx, ch]-data[metric][:, 6])).sum()/n))
        budget_hist.append(np.bincount(ch, weights=w, minlength=7)/n)
    return {'reps': reps, 'objective': objective,
            'protocol': 'question bootstrap with complete LOO refit; exclude all copies of held-out question',
            'ci95_delta': {k: np.quantile(v, [.025, .975]).tolist() for k, v in estimates.items()},
            'replicate_mean_delta': {k: float(np.mean(v)) for k, v in estimates.items()},
            'budget_fraction_ci95': {b: np.quantile(np.asarray(budget_hist)[:, j], [.025, .975]).tolist() for j, b in enumerate(BUDGETS)}}


def leakage_test(data):
    # An arbitrarily changed target question must receive identical predictions
    # for all budgets, including the predicted cost used for tie-breaking.
    reference = predict_loo(data)
    seen = []
    for group in np.unique(data['raw_score']):
        i = int(np.flatnonzero(data['raw_score'] == group)[0])
        altered = dict(data)
        for name in ['acc', 'em', 'f1', 'tokens']:
            altered[name] = data[name].copy()
            altered[name][i] = np.arange(7)*17.+1024.
        probe = predict_loo(altered)
        assert np.allclose(probe[i], reference[i], atol=1e-9, rtol=1e-10)
        assert choose(probe)[i] == choose(reference)[i]
        seen.append({'score_group': int(group), 'heldout_id': str(data['ids'][i]), 'passed': True})
    return seen


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--bootstrap-reps', type=int, default=2000)
    args = ap.parse_args(); data = load_data(); n = len(data['ids']); idx = np.arange(n)
    policies = {}; preds = {}
    for smoothing in [0., .05, .1, .2]:
        key = f'image_acc_globalmix_{smoothing:.2f}'
        pred = predict_loo(data, smoothing=smoothing)
        policies[key] = choose(pred); preds[key] = pred
    main_key = 'image_acc_globalmix_0.10'
    policies['image_f1_globalmix_0.10'] = choose(preds[main_key], 'f1')
    policies['image_em_globalmix_0.10'] = choose(preds[main_key], 'em')
    policies['acc_then_f1_globalmix_0.10'] = choose_acc_f1(preds[main_key])
    policies['merged3plus_acc_globalmix_0.10'] = choose(predict_loo(data, grouping='merged'))
    policies['pseudo50_acc'] = choose(predict_loo(data, pseudo_count=50))
    policies['global_only_loo_acc'] = choose(predict_loo(data, global_only=True))
    regimes = np.where(data['acc'].max(1) == 0, 'always_wrong',
                       np.where(data['acc'].min(1) == 1, 'always_correct', 'sensitive'))
    report = {'n': n, 'budgets': BUDGETS,
              'protocol': 'leave one original question out of both same-score and global statistics',
              'image_adaptation': 'remaining 1941 questions, not 99; available 100..600 plus FULL, not a nonexistent B700',
              'primary_policy': main_key,
              'primary_smoothing': '.90 same exact READ score mean + .10 global mean; both exclude current question',
              'primary_objective': 'argmax predicted lexical ACC (Loose); exact ties use predicted lower full-trajectory LLM tokens',
              'evaluation_warning': 'retrospective partial sample; LOO training sets overlap; not a new independent benchmark experiment',
              'leakage_tests': leakage_test(data),
              'full': evaluate(data, np.full(n, 6)), 'fixed_600': evaluate(data, np.full(n, 5)), 'policies': {}}
    for key, chosen in policies.items():
        r = evaluate(data, chosen, ci=True)
        r['vs_fixed_600'] = evaluate(data, chosen, reference=5, ci=True)
        r['by_type'] = {t: evaluate(data, chosen, np.flatnonzero(data['types'] == t)) for t in ['bridge', 'comparison']}
        r['by_regime'] = {g: evaluate(data, chosen, np.flatnonzero(regimes == g)) for g in ['always_correct', 'always_wrong', 'sensitive']}
        r['by_score'] = {str(s): evaluate(data, chosen, np.flatnonzero(data['raw_score'] == s)) for s in np.unique(data['raw_score'])}
        r['fresh_score_token_count_bookkeeping'] = {
            'mean_teacher_tokens': float(data['score_tokens'].mean()),
            'mean_teacher_plus_qwen_tokens': float(r['tokens']['mean']+data['score_tokens'].mean()),
            'delta_vs_full': float(r['tokens']['delta']+data['score_tokens'].mean()),
            'warning': 'Different GPT/Qwen tokenizers and costs; token sum is not money/FLOP equivalence.'}
        report['policies'][key] = r
    report['primary_refit_bootstrap'] = refit_bootstrap(data, args.bootstrap_reps)
    report['f1_variant_refit_bootstrap'] = refit_bootstrap(data, args.bootstrap_reps, objective='f1')
    read2 = np.flatnonzero(data['raw_score'] == 2)
    b600 = data['acc'][read2, 5]; full = data['acc'][read2, 6]
    report['read2_near_tie_diagnostic'] = {
        'n': len(read2), 'correct_b600': int(b600.sum()), 'correct_full': int(full.sum()),
        'b600_vs_full_gain': int(((b600 == 1) & (full == 0)).sum()),
        'b600_vs_full_harm': int(((b600 == 0) & (full == 1)).sum()),
        'interpretation': 'One-correct-answer mean advantage is unstable under LOO and global mixing; not own-outcome leakage.'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'loo_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    np.savez_compressed(OUT/'loo_decisions.npz', ids=data['ids'], predicted_primary_curve=preds[main_key], **policies)
    with (OUT/'question_decisions.jsonl').open('w', encoding='utf-8') as handle:
        for i, meta in enumerate(data['meta']):
            j = policies[main_key][i]
            same = int((data['raw_score'] == data['raw_score'][i]).sum())-1
            record = {'sample_id': meta['sample_id'], 'question': meta['question'], 'read_score': int(data['raw_score'][i]),
                      'same_score_reference_n': same, 'global_reference_n': n-1,
                      'selected_budget': BUDGETS[j], 'predicted_acc_curve': preds[main_key][i, :, 0].tolist(),
                      'predicted_cost_curve': preds[main_key][i, :, 3].tolist(),
                      'actual_selected_acc': float(data['acc'][i, j]), 'actual_full_acc': float(data['acc'][i, 6]),
                      'actual_selected_f1': float(data['f1'][i, j]), 'actual_full_f1': float(data['f1'][i, 6]),
                      'actual_selected_tokens': float(data['tokens'][i, j]), 'actual_full_tokens': float(data['tokens'][i, 6]),
                      'f1_variant_budget': BUDGETS[policies['image_f1_globalmix_0.10'][i]],
                      'hindsight_regime_for_evaluation_only': str(regimes[i])}
            handle.write(json.dumps(record, ensure_ascii=False)+'\n')
    print(json.dumps({'primary': report['policies'][main_key],
                      'sensitivity': {k: {m: v[m] for m in ['acc', 'f1', 'tokens']} for k, v in report['policies'].items()},
                      'primary_refit_bootstrap': report['primary_refit_bootstrap']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
