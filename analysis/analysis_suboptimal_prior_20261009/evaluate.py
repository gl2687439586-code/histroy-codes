"""Offline, question-disjoint and cross-execution one-budget routing. NumPy only."""
from pathlib import Path
from collections import Counter
import gzip, hashlib, importlib.util, json, math, re, sys, time
import numpy as np

BASE = Path(__file__).resolve().parent
WORK = BASE.parent
B = ['100','200','300','400','500','600','FULL']
RUNS = ['old500','new500','repeat_1','repeat_2','repeat_3']
SEEDS = [20261009,20261010,20261011]
PRIORDIR = WORK/'analysis_raw_prior_signals_20261004'
PATHS = [WORK/'analysis_matched500_raw_20261004/input'/r if i<2 else
         WORK/'analysis_five_runs_20261005/input'/r for i,r in enumerate(RUNS)]
METHODS = ['question_soft','generation_soft','retrieval_soft','all20_soft',
           'all20_advantage','text_all20_advantage','all20_cost01','all20_cost02']
NESTED = ['train_fixed','all20_soft','all20_advantage','text_all20_advantage',
          'all20_cost01','all20_cost02']

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

a=module('features_prior',PRIORDIR/'analyze.py')
previous=sys.modules.get('analyze');sys.modules['analyze']=a
s=module('success_models',PRIORDIR/'success_set_prediction.py')
if previous is None: del sys.modules['analyze']
else: sys.modules['analyze']=previous

def save(name,obj):
    (BASE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]

def keypaths(x,prefix=''):
    if isinstance(x,dict):
        for k,v in x.items():
            p=prefix+'.'+k
            yield p
            yield from keypaths(v,p)
    elif isinstance(x,list):
        for v in x: yield from keypaths(v,prefix+'[]')

def textwords(root):
    # No final answers or labels; separate question and current first query vocabulary.
    return ['q_'+w for w in a.words(root['question']) if w not in a.STOP]+[
        's_'+w for w in a.words(root['query']) if w not in a.STOP]

def load():
    data=dict(np.load(WORK/'analysis_five_runs_20261005/results/matched_arrays.npz'))
    ids=data['ids'].tolist()
    roots=[{r['sample_id']:r for r in rows(p/'roots/roots.merged.jsonl.gz')} for p in PATHS]
    for r in roots: assert set(r)==set(ids)
    cache=BASE/'features.npz'
    source_hashes={str(p/'roots/roots.merged.jsonl.gz'):hashlib.sha256((p/'roots/roots.merged.jsonl.gz').read_bytes()).hexdigest() for p in PATHS}
    if cache.exists():
        saved=np.load(cache);assert saved['ids'].tolist()==ids
        assert json.loads((BASE/'feature_source_hashes.json').read_text())==source_hashes
        X=saved['X']
    else:
        feature_rows=[[a.root_features(rt[sid]) for sid in ids] for rt in roots]
        X=np.array([[[record[f] for f in a.FEATURE_NAMES] for record in run] for run in feature_rows])
        np.savez_compressed(cache,X=X,ids=data['ids'])
        save('feature_source_hashes.json',source_hashes)
    assert X.shape==(5,500,20) and np.isfinite(X).all()
    texts=[[textwords(r[sid]) for sid in ids] for r in roots]
    inventory=[]
    for rr,rt in zip(RUNS,roots):
        interesting=Counter()
        for record in rt.values():
            for p in set(keypaths(record)):
                if re.search(r'logprob|logit|entropy|gradient|hidden_state|consensus',p,re.I): interesting[p]+=1
        inventory.append({'run':rr,'root_records':len(rt),'matching_field_counts':dict(interesting)})
    wb=WORK/'outputs/whitebox_gate100_qalignfix_audit_20261002/input/gate100_v1_20261002_qalignfix/features/per_question'
    whitebox_ids={p.stem for p in wb.glob('*.json')}
    save('availability.json',{'root_field_inventory':inventory,'whitebox_gate100_ids':len(whitebox_ids),
        'whitebox_overlap_with_matched500':len(whitebox_ids&set(ids)),
        'whitebox_note':'Separate outcome-enriched diagnostic sample; not five-run entropy coverage. No merge into primary analysis.',
        'consensus_note':'Repeated final answers exist, but are post-execution; unavailable before choosing one continuation.',
        'generation_length_note':'Root output/thinking/query lengths are available; chosen future answer length is excluded.',
        'gradients_note':'No neural gradient computation performed.', 'features':a.FEATURE_NAMES})
    return data, X, texts

def scale_fit(x):
    lo,hi=np.quantile(x,[.01,.99],axis=0);x=np.clip(x,lo,hi)
    mean=x.mean(0);sd=x.std(0);sd[sd<1e-8]=1
    return (lo,hi,mean,sd)

def scale(x,params):
    lo,hi,mu,sd=params
    return (np.clip(x,lo,hi)-mu)/sd

def ridge(x,y,l2=.1):
    mu=x.mean(0);ym=y.mean(0);xc=x-mu
    w=np.linalg.solve(xc.T@xc/len(x)+l2*np.eye(x.shape[1]),xc.T@(y-ym)/len(x))
    return mu,ym,w

def ridge_pred(x,fit):
    mu,ym,w=fit;return (x-mu)@w+ym

def vocab_fit(texts):
    df=Counter(w for t in texts for w in set(t))
    vocab=sorted([w for w,n in df.items() if n>=4],key=lambda w:(-df[w],w))[:128]
    return vocab,np.array([math.log((len(texts)+1)/(df[w]+1))+1 for w in vocab])

def tfidf(texts,fit):
    vocab,idf=fit;ix={w:j for j,w in enumerate(vocab)};out=np.zeros((len(texts),len(vocab)))
    for i,t in enumerate(texts):
        for w,n in Counter(t).items():
            if w in ix: out[i,ix[w]]=1+math.log(n)
    out*=idf
    return out/np.maximum(np.linalg.norm(out,axis=1,keepdims=True),1e-12)

def pick_quality(quality,cost):
    if cost.ndim==1:cost=np.tile(cost,(len(quality),1))
    return np.array([np.lexsort((np.arange(7),c,-q))[0] for q,c in zip(quality,cost)])

def fit_and_predict(train,test,targ_runs,X,texts,Y,C,methods,source_runs=(0,1)):
    assert not set(train)&set(test)
    source_runs=list(source_runs);nr=len(source_runs)
    x=X[source_runs][:,train].reshape(-1,20)
    target=Y[:,:,:,0][source_runs][:,train].mean(0)
    y=np.tile(target,(nr,1))
    assert y.shape==(nr*len(train),7) and x.shape==(nr*len(train),20)
    cost=C[source_runs][:,train].mean(0);global_cost=cost.mean(0)
    meanquality=target.mean(0)
    fixed=int(np.lexsort((np.arange(7),global_cost,-meanquality))[0])
    tt=[texts[r][i] for r in source_runs for i in train]
    xt=np.concatenate([X[r,test] for r in targ_runs])
    pred={ 'train_fixed':np.full((len(targ_runs),len(test)),fixed,int)}
    allsc=scale_fit(x);xx=scale(x,allsc);xxt=scale(xt,allsc)
    fitc=ridge(xx,np.tile(np.log1p(cost),(nr,1)))
    logtrain=ridge_pred(xx,fitc)
    smear=np.maximum(cost.mean(0),1)/np.maximum(np.expm1(logtrain).mean(0),1)
    pcost=np.clip(np.expm1(np.clip(ridge_pred(xxt,fitc),0,15))*smear,100,100000)
    if any(m.startswith('all20_') or m=='shared_difficulty' for m in methods):
        w=s.fit_logistic(xx,y);pall=s.sigmoid(np.column_stack([np.ones(len(xxt)),xxt])@w)
    for m in methods:
        if m=='train_fixed':continue
        if m in ['question_soft','generation_soft','retrieval_soft']:
            ix={'question_soft':range(3),'generation_soft':range(3,11),'retrieval_soft':range(11,20)}[m]
            sc=scale_fit(x[:,ix]);xf=scale(x[:,ix],sc);xfte=scale(xt[:,ix],sc)
            w=s.fit_logistic(xf,y)
            p=s.sigmoid(np.column_stack([np.ones(len(xfte)),xfte])@w)
            choice=pick_quality(p,global_cost)
        elif m=='all20_soft':choice=pick_quality(pall,global_cost)
        elif m in ['all20_cost01','all20_cost02']:
            lam=.01 if m.endswith('01') else .02
            choice=pick_quality(pall-lam*pcost/1000,pcost)
        elif m=='shared_difficulty':
            w=s.fit_logistic(xx,y.mean(1))
            z=np.column_stack([np.ones(len(xxt)),xxt])@w
            offsets=s.logit((y.sum(0)+.5)/(len(y)+1))
            choice=pick_quality(s.sigmoid(z+offsets),global_cost)
            assert np.all(choice==fixed)
        elif m in ['all20_advantage','text_all20_advantage']:
            xf,xfte=xx,xxt
            if m.startswith('text'):
                vf=vocab_fit(tt)
                te=[texts[r][i] for r in targ_runs for i in test]
                xf=np.column_stack([xx,tfidf(tt,vf)*4])
                xfte=np.column_stack([xxt,tfidf(te,vf)*4])
            f=ridge(xf,y-y[:,6,None]);p=ridge_pred(xfte,f)
            assert np.max(np.abs(p[:,6]))<1e-10
            choice=pick_quality(p,global_cost)
        else:raise ValueError(m)
        pred[m]=choice.reshape(len(targ_runs),len(test))
    return pred

def crossfit(data,X,texts):
    Y,C=data['Y'],data['C'];allpred={};foldlogs=[]
    for seed in SEEDS:
        start=time.perf_counter()
        folds=np.array_split(np.random.default_rng(seed).permutation(500),5)
        pred={m:np.full((3,500),-1,int) for m in METHODS+['train_fixed','shared_difficulty','nested_selector']}
        foldid=np.full(500,-1,int)
        for f,test in enumerate(folds):
            train=np.setdiff1d(np.arange(500),test);foldid[test]=f
            p=fit_and_predict(train,test,[2,3,4],X,texts,Y,C,METHODS+['shared_difficulty'])
            for m,v in p.items():pred[m][:,test]=v
            # Inner group crossfit uses source executions only, never target run labels.
            ifolds=np.array_split(np.random.default_rng(seed+f+100).permutation(train),3)
            ev={m:[] for m in NESTED}
            for itest in ifolds:
                itrain=np.setdiff1d(train,itest)
                ip=fit_and_predict(itrain,itest,[0,1],X,texts,Y,C,NESTED)
                for m in NESTED:
                    for r in range(2):
                        ev[m].append(np.column_stack([Y[r,itest,ip[m][r],0],C[r,itest,ip[m][r]]]))
            scores={m:np.concatenate(v).mean(0).tolist() for m,v in ev.items()}
            chosen=min(NESTED,key=lambda m:(-scores[m][0],scores[m][1],NESTED.index(m)))
            pred['nested_selector'][:,test]=pred[chosen][:,test]
            foldlogs.append({'seed':seed,'fold':f,'training_ids':[data['ids'][i] for i in train],
                'test_ids':[data['ids'][i] for i in test],'inner_scores_em_cost':scores,'inner_selected':chosen})
            print('fold complete',seed,f,'nested:',chosen,flush=True)
        assert all(np.all(p>=0) for p in pred.values())
        np.savez_compressed(BASE/f'choices_{seed}.npz',foldid=foldid,**pred)
        allpred[seed]=(pred,foldid)
        print('seed complete',seed,'seconds',round(time.perf_counter()-start,2),flush=True)
    save('folds.json',foldlogs)
    return allpred

def vectors(data,pick,target_runs=(2,3,4)):
    q=np.arange(500)
    out={}
    for name,ar in [('EM',data['Y'][:,:,:,0]),('F1',data['Y'][:,:,:,1]),('ACC',data['Y'][:,:,:,2]),
        ('input_tokens',data['I']),('output_tokens',data['O']),('total_tokens',data['C']),
        ('search_calls',data['S']),('visible_evidence_tokens',data['E']),('fallback_rate',data['F'])]:
        out[name]=np.array([ar[r,q,pick[t]] for t,r in enumerate(target_runs)])
    return out

def meanstats(v):return {k:float(a.mean()) for k,a in v.items()}

def compare(v,b,idx,with_ci,target_runs=(2,3,4)):
    out={}
    for k in ['EM','ACC','F1','total_tokens']:
        d=(v[k]-b[k]).mean(0)
        out[k]={'delta':float(d.mean())}
        if with_ci:out[k]['ci95']=np.quantile(d[idx].mean(1),[.025,.975]).tolist()
    out['token_saving_fraction']=1-float(v['total_tokens'].mean()/b['total_tokens'].mean())
    out['per_execution_rescue_harm']=[{'run':RUNS[rr],
        'rescue':int(((v['EM'][r]==1)&(b['EM'][r]==0)).sum()),
        'harm':int(((v['EM'][r]==0)&(b['EM'][r]==1)).sum()),
        'mcnemar_exact_p':a.mcnemar(int(((v['EM'][r]==1)&(b['EM'][r]==0)).sum()),int(((v['EM'][r]==0)&(b['EM'][r]==1)).sum()))} for r,rr in enumerate(target_runs)] if np.isin(b['EM'],[0,1]).all() else None
    return out

def matched_mix(data,choices,foldid):
    out={k:np.zeros((3,500)) for k in ['EM','ACC','F1','total_tokens']}
    for r in range(3):
        for f in range(5):
            ix=np.flatnonzero(foldid==f);prob=np.bincount(choices[r,ix],minlength=7)/len(ix)
            for k,ar in [('EM',data['Y'][:,:,:,0]),('ACC',data['Y'][:,:,:,2]),('F1',data['Y'][:,:,:,1]),('total_tokens',data['C'])]:
                out[k][r,ix]=ar[r+2,ix]@prob
    return out

def evaluate(data,allpred):
    idx=np.random.default_rng(20261009).integers(0,500,(10000,500),dtype=np.int16)
    full=vectors(data,np.full((3,500),6))
    results=[];baselines=[]
    for j,b in enumerate(B):
        v=vectors(data,np.full((3,500),j))
        baselines.append({'method':'fixed_'+b,**meanstats(v),'vs_FULL':compare(v,full,idx,True),
            'per_run':[{k:float(x[r].mean()) for k,x in v.items()} for r in range(3)]})
    for seed,(pred,foldid) in allpred.items():
        fixed=vectors(data,pred['train_fixed'])
        for m,choice in pred.items():
            v=vectors(data,choice);mix=matched_mix(data,choice,foldid)
            results.append({'seed':seed,'method':m,**meanstats(v),
                'budget_counts':dict(Counter(B[j] for j in choice.ravel())),
                'per_run':[{k:float(x[r].mean()) for k,x in v.items()} for r in range(3)],
                'vs_FULL':compare(v,full,idx,seed==SEEDS[0]),
                'vs_train_fixed':compare(v,fixed,idx,seed==SEEDS[0]),
                'vs_random_same_budget_mix':compare(v,mix,idx,seed==SEEDS[0])})
    # Explicitly privileged: known SAME question's old outcomes.
    em=data['Y'][:2,:,:,0].mean(0);tc=data['C'][:2].mean(0)
    choice=pick_quality(em,tc);pv=vectors(data,np.tile(choice,(3,1)))
    privileged={'label':'same_question_old2_success_frequency_NOT_unseen_prior',**meanstats(pv),'vs_FULL':compare(pv,full,idx,True)}
    yy=data['Y'][:2,:,:,0].astype(bool);both=yy.all(axis=0).any(1);eligible=yy.any(2).all(0)
    overlap={'both_executions_have_success':int(eligible.sum()),'common_success_budget':int(both.sum()),
             'conditional_overlap':float(both.sum()/eligible.sum()),'all500_common_fraction':float(both.mean())}
    save('results.json',{'primary_seed':SEEDS[0],'baselines':baselines,'policies':results,'privileged_diagnostic':privileged,'overlap_935':overlap})
    with gzip.open(BASE/'selected_records_primary.jsonl.gz','wt',encoding='utf-8') as f:
        pred,folds=allpred[SEEDS[0]]
        for m,pick in pred.items():
            v=vectors(data,pick)
            for r in range(3):
                for i,sid in enumerate(data['ids']):
                    f.write(json.dumps({'method':m,'run':RUNS[r+2],'sample_id':sid,'outer_fold':int(folds[i]),
                        'budget':B[pick[r,i]],**{k:float(ar[r,i]) for k,ar in v.items()}},ensure_ascii=False)+'\n')
    print('PRIMARY RESULTS',flush=True)
    for x in baselines+[r for r in results if r['seed']==SEEDS[0]]+[privileged]:
        print(x.get('method',x.get('label')),round(x['EM'],4),round(x['ACC'],4),round(x['F1'],4),round(x['total_tokens'],1),flush=True)

def main():
    start=time.perf_counter();data,X,texts=load()
    if all((BASE/f'choices_{seed}.npz').exists() for seed in SEEDS):
        allpred={}
        for seed in SEEDS:
            p=dict(np.load(BASE/f'choices_{seed}.npz'));fold=p.pop('foldid');allpred[seed]=(p,fold)
    else: allpred=crossfit(data,X,texts)
    evaluate(data,allpred)
    save('execution.json',{'elapsed_seconds':time.perf_counter()-start,'new_llm_calls':0,'remote_mutations':0,
        'numpy':np.__version__,'question_count':500,'target_question_executions':1500,
        'training_executions':['old500','new500'],'test_executions':RUNS[2:],
        'primary_metric':'EM','ACC':'normalized contains, not judge','cost':'selected standalone input+output',
        'exploratory_reuse_of_previously_inspected_data':True})

if __name__=='__main__':main()
