"""Secondary four-source-runs -> heldout execution, still question-disjoint."""
import numpy as np
import evaluate as e

def main():
    data,X,texts=e.load();Y,C=data['Y'],data['C']
    idx=np.random.default_rng(20261009).integers(0,500,(10000,500),dtype=np.int16)
    full=e.vectors(data,np.full((5,500),6),range(5));results=[];foldlogs=[]
    baselines=[]
    for j,b in enumerate(e.B):
        v=e.vectors(data,np.full((5,500),j),range(5))
        baselines.append({'method':'fixed_'+b,**e.meanstats(v),'vs_FULL':e.compare(v,full,idx,True,range(5))})
    for seed in e.SEEDS:
        pred={m:np.full((5,500),-1,int) for m in e.METHODS+['train_fixed']}
        folds=np.array_split(np.random.default_rng(seed).permutation(500),5)
        for r in range(5):
            sources=[s for s in range(5) if s!=r]
            for f,test in enumerate(folds):
                train=np.setdiff1d(np.arange(500),test)
                p=e.fit_and_predict(train,test,[r],X,texts,Y,C,e.METHODS,source_runs=sources)
                for m,choices in p.items():pred[m][r,test]=choices[0]
                foldlogs.append({'seed':seed,'target_run':r,'source_runs':sources,'fold':f,
                    'train_ids':[data['ids'][i] for i in train],'test_ids':[data['ids'][i] for i in test]})
        assert all(np.all(p>=0) for p in pred.values())
        np.savez_compressed(e.BASE/f'choices_loo_{seed}.npz',**pred)
        fixed=e.vectors(data,pred['train_fixed'],range(5))
        for m,choices in pred.items():
            v=e.vectors(data,choices,range(5))
            results.append({'method':m,'seed':seed,**e.meanstats(v),
                'vs_FULL':e.compare(v,full,idx,seed==e.SEEDS[0],range(5)),
                'vs_train_fixed':e.compare(v,fixed,idx,seed==e.SEEDS[0],range(5)),
                'per_run':[{k:float(a[r].mean()) for k,a in v.items()} for r in range(5)]})
        print('LOO completed seed',seed,flush=True)
    e.save('secondary_loo_results.json',{'design':'retrospective four-to-one execution, question-disjoint, exploratory secondary',
        'baselines':baselines,'policies':results})
    e.save('secondary_loo_folds.json',foldlogs)
    for x in baselines+[r for r in results if r['seed']==e.SEEDS[0]]:
        print(x['method'],round(x['EM'],4),round(x['ACC'],4),round(x['F1'],4),round(x['total_tokens'],1))

if __name__=='__main__':main()
