"""Supplement: unchanged prior 20 features/models; question-disjoint temporal test."""
from pathlib import Path
import sys,importlib.util,json
from itertools import combinations
import numpy as np
import analyze as five

P=Path(__file__).resolve().parent.parent/'analysis_raw_prior_signals_20261004'
spec=importlib.util.spec_from_file_location('prior_analysis_features',P/'analyze.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
# The archived local module imports its feature helper as 'analyze'. Supply it only during import.
saved=sys.modules.get('analyze');sys.modules['analyze']=a
spec=importlib.util.spec_from_file_location('prior_success_methods',P/'success_set_prediction.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
sys.modules['analyze']=saved
METHODS=['constant','title_groups','all_prior','shared_difficulty']
SEEDS=[20261004,20261005,20261006]

def main():
    ids,rr,roots,maps,Y,C,I,O,E,S,F,RC=five.load();yy=Y[:,:,:,0]
    ff=[[a.root_features(rt[sid]) for sid in ids] for rt in roots]
    X=np.array([[[d[f] for f in a.FEATURE_NAMES] for d in run] for run in ff])
    summaries=[]
    for seed in SEEDS:
        folds=np.array_split(np.random.default_rng(seed).permutation(500),5)
        predictions={m:np.zeros((3,500,7)) for m in METHODS}
        p_any={m:np.zeros((3,500)) for m in METHODS}
        tc=np.zeros((500,7));best2=np.zeros((500,2),int)
        for test in folds:
            train=np.setdiff1d(np.arange(500),test)
            assert not set(train)&set(test)
            x=X[:2,train].reshape(-1,20);y=yy[:2,train].reshape(-1,7)
            tc[test]=C[:2,train].mean((0,1))
            combs=list(combinations(range(7),2))
            bb=min(combs,key=lambda z:(-float(yy[:2,train][:,:,list(z)].max(2).mean()),
                float((C[:2,train][:,:,list(z)].sum(2)-RC[:2,train]).mean()),z))
            best2[test]=bb
            for m in METHODS:
                model=s.fit_model(m,x,y)
                for t in range(3):
                    p,pa=s.predict(model,X[t+2,test]);predictions[m][t,test]=p;p_any[m][t,test]=pa
        vecs={}
        for m in METHODS:
            ps=predictions[m];selected=[];within=[];top2hit=[];top2cost=[];basehit=[];basecost=[]
            for t in range(3):
                order=s.topk(ps[t],tc,2);selected.append(order[:,0])
                wa=s.within_auc(yy[t+2],ps[t]);within.append(wa)
                q=np.arange(500)[:,None]
                top2hit.append(yy[t+2,q,order].max(1));top2cost.append(C[t+2,q,order].sum(1)-RC[t+2])
                basehit.append(yy[t+2,q,best2].max(1));basecost.append(C[t+2,q,best2].sum(1)-RC[t+2])
            st,values=five.selected_stats(Y,C,I,O,E,S,F,[2,3,4],np.array(selected),ci=seed==SEEDS[0])
            item={'seed':seed,'method':m,'top1':st,'any_success_auc_by_target':[s.auc(yy[t+2].max(1),p_any[m][t]) for t in range(3)],
                'within_question_auc':float(np.nanmean(within)),
                'budget_brier':float(((ps-yy[2:])**2).mean()),
                'top2_oracle_hit':float(np.mean(top2hit)),'top2_bundle_tokens':float(np.mean(top2cost)),
                'fixed_pair_oracle_hit':float(np.mean(basehit)),'fixed_pair_bundle_tokens':float(np.mean(basecost))}
            if seed==SEEDS[0]:
                item['top2_vs_fixed_pair_hit']=five.mean_ci(np.mean(top2hit,0)-np.mean(basehit,0))
                item['top2_vs_fixed_pair_tokens']=five.mean_ci(np.mean(top2cost,0)-np.mean(basecost,0))
                item['top1_vs_FULL']={k:five.mean_ci(values[k]-(Y[2:,:,6,five.M.index(k)].mean(0) if k in five.M else C[2:,:,6].mean(0))) for k in five.M+['total_tokens']}
            vecs[m]=values
            summaries.append(item)
        if seed==SEEDS[0]:
            for it in summaries[-len(METHODS):]:
                it['top1_vs_constant']={k:five.mean_ci(vecs[it['method']][k]-vecs['constant'][k]) for k in five.M+['total_tokens']}
        print('Question-disjoint predictive test complete:',seed,flush=True)
    five.save('question_disjoint_prediction.json',{'design':{'train_runs':['old500','new500'],'test_runs':['repeat_1','repeat_2','repeat_3'],
        'folds':5,'train_unique_questions_per_fold':400,'heldout_unique_questions_per_fold':100,
        'all_runs_of_test_ids_excluded_from_training':True,'feature_names':a.FEATURE_NAMES,'L2':s.L2,
        'hyperparameters_unchanged_from_prior_analysis':True,'seeds':SEEDS,'no_new_model_calls':True},'results':summaries})

if __name__=='__main__':main()
