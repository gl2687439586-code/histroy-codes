"""Matched-budget permutation controls for OOF prior policies."""
from __future__ import annotations
import json
import numpy as np
import readscore1942_prior_analysis as base


def main():
    d=base.load_data(); n=len(d['ids']); ix=np.arange(n)
    main=np.load(base.OUT/'cv_predictions_20260930.npz',allow_pickle=True)
    conditional=np.load(base.OUT/'conditional_predictions.npz',allow_pickle=True)
    chosen={'read_semantic_neighbor':main['read_semantic_neighbor'],'nested_family_selector':main['nested_family_selector'],'conditional_with_read':conditional['conditional_with_read'],'conditional_without_read':conditional['conditional_without_read']}
    report={}; rng=np.random.default_rng(base.SEED+30)
    for name,ch in chosen.items():
        observed=float(d['acc'][ix,ch].mean()); ff=float(d['f1'][ix,ch].mean())
        result={}
        for strat in ['all','within_official_type']:
            groups=[ix] if strat=='all' else [np.flatnonzero(d['types']==t) for t in ['bridge','comparison']]
            acc=[]; f1=[]
            for _ in range(5000):
                perm=np.copy(ch)
                for g in groups: perm[g]=rng.permutation(ch[g])
                acc.append(float(d['acc'][ix,perm].mean())); f1.append(float(d['f1'][ix,perm].mean()))
            result[strat]={'observed_acc':observed,'random_mean_acc':float(np.mean(acc)),'matched_acc_advantage':observed-float(np.mean(acc)),'permutation_p_acc':float((1+np.sum(np.asarray(acc)>=observed))/(1+len(acc))),'observed_f1':ff,'random_mean_f1':float(np.mean(f1)),'matched_f1_advantage':ff-float(np.mean(f1)),'permutation_p_f1':float((1+np.sum(np.asarray(f1)>=ff))/(1+len(f1)))}
        report[name]=result
    (base.OUT/'budget_matching_permutation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
