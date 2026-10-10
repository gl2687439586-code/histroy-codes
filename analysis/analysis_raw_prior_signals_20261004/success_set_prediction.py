"""Cross-run question-disjoint multilabel success-set prediction."""
from itertools import combinations
import json
import numpy as np
import analyze as a

DEST=a.BASE/'success_set_results'
L2=.03
MODEL_FEATURES={
    'question_only':a.FEATURE_NAMES[:3],
    'root_generation':a.FEATURE_NAMES[3:11],
    'retrieved_lexical':a.FEATURE_NAMES[11:],
    'all_prior':a.FEATURE_NAMES,
    'shared_difficulty':a.FEATURE_NAMES,
}


def save(name,obj):
    DEST.mkdir(exist_ok=True)
    (DEST/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-35,35)))
def logit(x):
    x=np.clip(x,1e-5,1-1e-5)
    return np.log(x/(1-x))


def fit_logistic(x,y):
    """Average BCE + L2/2 squared weights; intercept unpenalized."""
    if y.ndim==1:y=y[:,None]
    xx=np.column_stack([np.ones(len(x)),x]);d=xx.shape[1]
    w=np.zeros((d,y.shape[1]));w[0]=logit((y.sum(0)+.5)/(len(y)+1))
    penalty=np.r_[0.,np.full(d-1,L2)]
    for j in range(y.shape[1]):
        for iteration in range(60):
            p=sigmoid(xx@w[:,j]);v=p*(1-p)
            gradient=xx.T@(p-y[:,j])/len(x)+penalty*w[:,j]
            hessian=(xx.T*v)@xx/len(x)+np.diag(penalty)+np.eye(d)*1e-9
            step=np.linalg.solve(hessian,gradient)
            old=np.mean(np.logaddexp(0,xx@w[:,j])-y[:,j]*(xx@w[:,j]))+np.sum(penalty*w[:,j]**2)/2
            scale=1.
            for _ in range(20):
                candidate=w[:,j]-scale*step;z=xx@candidate
                loss=np.mean(np.logaddexp(0,z)-y[:,j]*z)+np.sum(penalty*candidate**2)/2
                if loss<=old+1e-12:break
                scale*=.5
            w[:,j]=candidate
            if np.max(np.abs(scale*step))<1e-7:break
    return w


def fit_model(method,train_x,y):
    any_y=y.max(1)
    # Mild smoothing for constant rates, and to avoid infinite intercepts.
    rate=(y.sum(0)+.5)/(len(y)+1)
    any_rate=(any_y.sum()+.5)/(len(y)+1)
    if method=='constant':return {'method':method,'rate':rate,'any_rate':any_rate}
    if method=='title_groups':
        labels=title_groups(train_x);params={}
        for g in range(3):
            mask=labels==g
            params[g]=((y[mask].sum(0)+20*rate)/(mask.sum()+20),
                       (any_y[mask].sum()+20*any_rate)/(mask.sum()+20))
        return {'method':method,'params':params}
    ix=[a.FEATURE_NAMES.index(k) for k in MODEL_FEATURES[method]]
    x=train_x[:,ix]
    low,high=np.quantile(x,[.01,.99],axis=0);x=np.clip(x,low,high)
    mean=x.mean(0);sd=x.std(0);sd[sd<1e-8]=1
    x=(x-mean)/sd
    target=np.column_stack([y.mean(1),any_y]) if method=='shared_difficulty' else np.column_stack([y,any_y])
    w=fit_logistic(x,target)
    return {'method':method,'ix':ix,'low':low,'high':high,'mean':mean,'sd':sd,'w':w,
            'budget_offsets':logit(rate)-logit(rate.mean())}


def title_groups(x):
    d=x[:,a.FEATURE_NAMES.index('unique_title_fraction')]
    return np.where(d<=.4,0,np.where(d<1,1,2))


def predict(model,x):
    method=model['method']
    if method=='constant':return np.tile(model['rate'],(len(x),1)),np.full(len(x),model['any_rate'])
    if method=='title_groups':
        v=[model['params'][int(g)] for g in title_groups(x)]
        return np.array([t[0] for t in v]),np.array([t[1] for t in v])
    xx=(np.clip(x[:,model['ix']],model['low'],model['high'])-model['mean'])/model['sd']
    z=np.column_stack([np.ones(len(x)),xx])@model['w']
    if method=='shared_difficulty':return sigmoid(z[:,0,None]+model['budget_offsets']),sigmoid(z[:,1])
    return sigmoid(z[:,:7]),sigmoid(z[:,7])


def auc(y,p):
    y=np.asarray(y);p=np.asarray(p);n1=int(y.sum());n0=len(y)-n1
    if not n1 or not n0:return None
    u,iv,c=np.unique(p,return_inverse=True,return_counts=True)
    ranks=(np.cumsum(c)-(c-1)/2)[iv]
    return float((ranks@y-n1*(n1+1)/2)/(n1*n0))


def within_auc(y,p):
    out=np.full(len(y),np.nan)
    for i,(yi,pi) in enumerate(zip(y,p)):
        pos=pi[yi==1];neg=pi[yi==0]
        if len(pos) and len(neg):
            d=pos[:,None]-neg
            out[i]=np.mean((d>0)+.5*(d==0))
    return out


def topk(prob,cost,k):
    # Ties use source training expected token cost, never target realized cost.
    return np.array([sorted(range(7),key=lambda j:(-p[j],c[j],j))[:k] for p,c in zip(prob,cost)],int)


def selection_summary(run,y,actions):
    ar=run['arrays'];ix=np.arange(len(y))[:,None]
    yy=y[ix,actions];k=actions.shape[1]
    root_cost=ar['total_tokens'][:,0]-ar['continuation_tokens'][:,0]
    cost=ar['continuation_tokens'][ix,actions].sum(1)+root_cost
    vectors={'hit':yy.max(1),'precision':yy.mean(1),'bundle_tokens':cost,
             'success_budget_recall':yy.sum(1)/np.maximum(y.sum(1),1)}
    m={'k':k,'candidate_hit_rate':float(vectors['hit'].mean()),
       'candidate_hit_given_any_success':float(vectors['hit'][y.sum(1)>0].mean()),
       'mean_set_precision':float(vectors['precision'].mean()),
       'mean_recall_given_any_success':float(vectors['success_budget_recall'][y.sum(1)>0].mean()),
       'mean_bundle_total_tokens':float(cost.mean())}
    if k==1:
        m['single_answer_metrics']=a.means(a.selected(ar,actions[:,0]))
    return m,vectors


def threshold_summary(y,p):
    pred=p>=.5
    intersection=(pred*y).sum(1);union=np.logical_or(pred,y).sum(1)
    sz=pred.sum(1);goldsz=y.sum(1)
    jacc=np.divide(intersection,union,out=np.ones(len(y)),where=union>0)
    return {'threshold':.5,'mean_set_size':float(sz.mean()),'empty_set_rate':float((sz==0).mean()),
        'exact_set_match':float(np.all(pred==y,axis=1).mean()),
        'jaccard_empty_empty_is_one':float(jacc.mean()),
        'jaccard_given_nonempty_gold':float(jacc[goldsz>0].mean()),
        'micro_precision':float(intersection.sum()/max(1,sz.sum())),
        'micro_recall':float(intersection.sum()/max(1,goldsz.sum()))}


def auc_delta_ci(y,p,q,seed=20261004):
    rng=np.random.default_rng(seed);v=[]
    for _ in range(2000):
        ix=rng.integers(0,len(y),len(y));ap=auc(y[ix],p[ix]);aq=auc(y[ix],q[ix])
        if ap is not None and aq is not None:v.append(ap-aq)
    return np.quantile(v,[.025,.975]).tolist()


def paired(v,w):
    keys=list(v);ds=np.column_stack([v[k]-w[k] for k in keys]);bounds=a.ci(ds)
    return {k:{'delta':float(ds[:,j].mean()),'ci95':bounds[j]} for j,k in enumerate(keys)}


def best_fixed_subset(train_y,train_cost,train_cont,k):
    root=train_cost[:,0]-train_cont[:,0]
    opts=list(combinations(range(7),k))
    return min(opts,key=lambda ix:(-train_y[:,ix].max(1).mean(),
                                  (train_cont[:,ix].sum(1)+root).mean(),ix))


def run_direction(source,target,ids):
    n=len(ids);y=target['arrays']['answer_em'];out=[];pred_records=[]
    models=['constant','title_groups',*MODEL_FEATURES]
    for seed in a.SEEDS:
        folds=np.array_split(np.random.default_rng(seed).permutation(n),5)
        predictions={m:(np.zeros((n,7)),np.zeros(n)) for m in models}
        train_costs=np.zeros((n,7));bestsets={k:np.zeros((n,k),int) for k in [1,2,3]}
        summaries=[]
        for fold,test in enumerate(folds):
            train=np.setdiff1d(np.arange(n),test)
            assert not set(np.array(ids)[train])&set(np.array(ids)[test])
            tr_y=source['arrays']['answer_em'][train];tr_c=source['arrays']['total_tokens'][train]
            tr_cont=source['arrays']['continuation_tokens'][train]
            train_costs[test]=tr_c.mean(0)
            for k in bestsets:
                bestsets[k][test]=best_fixed_subset(tr_y,tr_c,tr_cont,k)
            summaries.append({'fold':fold,'best_fixed_sets':{k:[a.BUDGETS[j] for j in bestsets[k][test[0]]] for k in bestsets}})
            for name in models:
                model=fit_model(name,source['x'][train],tr_y)
                p,pa=predict(model,target['x'][test])
                predictions[name][0][test]=p;predictions[name][1][test]=pa
        baselines={k:selection_summary(target,y,acts) for k,acts in bestsets.items()}
        fold_v={}
        for name,(p,pa) in predictions.items():
            assert np.isfinite(p).all() and np.all((p>0)&(p<1))
            wa=within_auc(y,p)
            item={'source':source['label'],'target':target['label'],'seed':seed,'model':name,
                  'any_success_auc':auc(y.max(1),pa),'any_success_brier':float(np.mean((pa-y.max(1))**2)),
                  'budget_brier':float(np.mean((p-y)**2)),'mean_per_budget_auc':float(np.mean([auc(y[:,j],p[:,j]) for j in range(7)])),
                  'within_question_budget_auc':float(np.nanmean(wa)), 'mixed_success_question_count':int(np.isfinite(wa).sum()),
                  'threshold_set':threshold_summary(y,p),'topk':{},'fold_fixed_set_rules':summaries}
            vs={}
            for k in [1,2,3]:
                acts=topk(p,train_costs,k);m,v=selection_summary(target,y,acts)
                item['topk'][str(k)]=m
                item['topk'][str(k)]['training_selected_fixed_set_baseline']=baselines[k][0]
                if seed==a.SEEDS[0]:item['topk'][str(k)]['vs_fixed_set']=paired(v,baselines[k][1])
                vs[k]=v
                if seed==a.SEEDS[0] and k==2:
                    pred_records.extend({'source':source['label'],'target':target['label'],'model':name,'sample_id':sid,
                        'probabilities':pi.tolist(),'p_any':float(pai),'top2':[a.BUDGETS[j] for j in aa],
                        'true_success_set':[a.BUDGETS[j] for j in np.flatnonzero(yi)]}
                        for sid,pi,pai,aa,yi in zip(ids,p,pa,acts,y))
            fold_v[name]={'wa':wa,'brier':np.mean((p-y)**2,axis=1),'topk':vs}
            out.append(item)
        if seed==a.SEEDS[0]:
            for item in out[-len(models):]:
                name=item['model'];p,pa=predictions[name]
                item['vs_constant']={'any_success_auc_delta':item['any_success_auc']-auc(y.max(1),predictions['constant'][1]),
                                    'any_success_auc_delta_ci95':auc_delta_ci(y.max(1),pa,predictions['constant'][1])}
                for baseline in ['constant','shared_difficulty']:
                    valid=np.isfinite(fold_v[name]['wa'])
                    delta=fold_v[name]['wa'][valid]-fold_v[baseline]['wa'][valid]
                    item['vs_'+baseline]=dict(item.get('vs_'+baseline,{}),
                        within_question_auc_delta=float(delta.mean()),within_question_auc_delta_ci95=a.ci(delta)[0],
                        brier_delta=float((fold_v[name]['brier']-fold_v[baseline]['brier']).mean()),
                        brier_delta_ci95=a.ci(fold_v[name]['brier']-fold_v[baseline]['brier'])[0],
                        top2=paired(fold_v[name]['topk'][2],fold_v[baseline]['topk'][2]))
        print('success sets',source['label'],'->',target['label'],seed,flush=True)
    return out,pred_records


def tests():
    assert auc(np.array([0,1]),np.array([0,1]))==1
    assert auc(np.array([0,1]),np.array([1,1]))==.5
    assert auc(np.array([0,0]),np.array([0,1])) is None
    yy=np.array([[0,1,0,0,0,0,0],[1,1,1,1,1,1,1]])
    wa=within_auc(yy,np.array([[0,1,0,0,0,0,0],[1,1,1,1,1,1,1]]))
    assert wa[0]==1 and np.isnan(wa[1])
    rng=np.random.default_rng(1);x=rng.normal(size=(200,2));y=(x[:,0]>0).astype(float)
    w=fit_logistic(x,y);p=sigmoid(np.column_stack([np.ones(200),x])@w)[:,0]
    assert auc(y,p)>.99
    assert list(topk(np.ones((1,7)),np.array([[7,6,5,4,3,2,1]]),2)[0])==[6,5]
    print('SET PREDICTION TESTS PASSED',flush=True)


def main():
    tests();ids,old=a.load_run('old500');_,new=a.load_run('new500',ids)
    old_res,old_pred=run_direction(old,new,ids)
    save('OLD_TO_NEW.json',old_res)
    new_res,new_pred=run_direction(new,old,ids)
    save('ALL_MODELS.json',old_res+new_res);save('PREDICTIONS_PRIMARY_SEED.json',old_pred+new_pred)
    save('MANIFEST.json',{'samples':len(ids),'budgets':a.BUDGETS,'seeds':a.SEEDS,'models':7,'runs':2,
        'primary_seed':a.SEEDS[0],'L2':L2,'new_model_calls':0,'target':'observed per-budget EM success set in target run',
        'cost_definition':'sum selected budget continuations plus root once',
        'warning':'topK hit is an oracle-within-candidate-set diagnostic, not deployable final answer accuracy',
        'source_sha256':'b76014d79f8282204ef8ac35acc22ef70d15ed30be1f963f3143c577f0133f25'})
    assert len(old_res+new_res)==42
    print('ALL SET RESULTS SAVED',flush=True)


if __name__=='__main__':main()
