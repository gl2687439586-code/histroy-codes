"""Mechanism, low-cost distillation, and score-routing robustness checks."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import readscore1942_prior_analysis as base


def student_predict(data,train,test):
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=18000,sublinear_tf=True)
    tr=vec.fit_transform(data['q'][train]); te=vec.transform(data['q'][test])
    clf=LogisticRegression(C=1,max_iter=1500,solver='lbfgs',random_state=base.SEED)
    clf.fit(tr,data['score'][train].astype(int)); p=clf.predict_proba(te)
    return p@clf.classes_,clf.classes_[p.argmax(1)]


def unconstrained_mapping(data,train,test,key,metric='acc',alpha=50):
    # A straightforward training-only group policy, no post-hoc test tuning.
    glob=np.column_stack([data['acc'][train].mean(0),data['f1'][train].mean(0),data['tokens'][train].mean(0)])
    chosen=[]
    for k in key[test]:
        members=train[key[train]==k]
        table=np.column_stack([data['acc'][members].sum(0),data['f1'][members].sum(0),data['tokens'][members].sum(0)])
        table=(table+alpha*glob)/(len(members)+alpha)
        if metric=='acc': j=max(range(7),key=lambda j:(table[j,0],table[j,1],-table[j,2]))
        else: j=max(range(7),key=lambda j:(table[j,1],-table[j,2]))
        chosen.append(j)
    return np.asarray(chosen)


def main():
    data=base.load_data(); n=len(data['ids']); idx=np.arange(n)
    regime=np.where(data['acc'].max(1)==0,'always_wrong',np.where(data['acc'].min(1)==1,'always_correct','sensitive'))
    oracle=np.asarray([max(range(7),key=lambda j:(data['acc'][i,j],data['f1'][i,j],-data['tokens'][i,j])) for i in idx])
    report={'by_type_score':{},'correlations_by_type':{},'distillation':{},'simple_crossfit':{}}
    for ty in ['bridge','comparison']:
        ti=np.flatnonzero(data['types']==ty)
        report['correlations_by_type'][ty]={}
        for label,y in [('oracle_budget',oracle),('any_success',data['acc'].max(1)),('full_acc',data['acc'][:,6]),('full_tokens',data['tokens'][:,6])]:
            r=spearmanr(data['score'][ti],y[ti]); report['correlations_by_type'][ty][label]={'rho':float(r.statistic),'p':float(r.pvalue)}
        for score in [1,2,3]:
            ix=np.flatnonzero((data['types']==ty)&(data['score']==score))
            report['by_type_score'][f'{ty}_score{score}']={'n':len(ix),'regimes':dict(Counter(regime[ix])),'curves':{b:base.evaluate(data,np.full(n,j),ix) for j,b in enumerate(base.BUDGETS)} if len(ix) else {}}
    cv=list(StratifiedKFold(5,shuffle=True,random_state=base.SEED).split(idx,data['types']))
    ch={name:np.zeros(n,int) for name in ['read_group_acc','read_group_f1','read_operator_acc','student_read_group_acc']}
    pred_score=np.zeros(n); pred_class=np.zeros(n,int)
    key=data['score'].astype(int).astype(str)
    opkey=np.char.add(np.char.add(key,'|'),data['operators'])
    for fold,(train,test) in enumerate(cv):
        p,c=student_predict(data,train,test); pred_score[test]=p; pred_class[test]=c
        for name,keys,metric in [('read_group_acc',key,'acc'),('read_group_f1',key,'f1'),('read_operator_acc',opkey,'acc')]:
            ch[name][test]=unconstrained_mapping(data,train,test,keys,metric)
        # Within training data, obtain predicted scores without using each
        # training question's own score label. Thus routing sees the same noisy
        # student-score representation during fitting and test deployment.
        train_class=np.zeros(n,int); train_class[test]=c
        inner=StratifiedKFold(3,shuffle=True,random_state=base.SEED+fold+10)
        for tr0,va0 in inner.split(train,data['types'][train]):
            _,tc=student_predict(data,train[tr0],train[va0]); train_class[train[va0]]=tc
        skey=train_class.astype(str)
        ch['student_read_group_acc'][test]=unconstrained_mapping(data,train,test,skey,'acc')
        print('student outer',fold+1,flush=True)
    report['distillation']={'score_exact_accuracy':float(accuracy_score(data['score'].astype(int),pred_class)),'score_mean_absolute_error':float(np.abs(data['score']-pred_score).mean()),'score_spearman':float(spearmanr(data['score'],pred_score).statistic),'incremental_generation_calls':0}
    for name,chosen in ch.items():
        report['simple_crossfit'][name]=base.evaluate(data,chosen,ci=True)
        report['simple_crossfit'][name]['by_type']={ty:base.evaluate(data,chosen,np.flatnonzero(data['types']==ty),ci=True) for ty in ['bridge','comparison']}
        if not name.startswith('student'): report['simple_crossfit'][name]['online_cost']=base.evaluate(data,chosen,ci=True,charge=True)
    # Test information supplied by READ beyond a matched no-READ model.
    ys={'sensitive':(regime=='sensitive').astype(int),'any_success':data['acc'].max(1).astype(int),'full_correct':data['acc'][:,6].astype(int)}
    report['detectors']={}
    for kind in ['score_conf','geometry','lowcost','read_all']:
        probs={k:np.zeros(n) for k in ys}
        for train,test in cv:
            xtr,xte=base.matrices(data,train,test,kind,'ridge')
            for name,y in ys.items():
                clf=LogisticRegression(C=.1,max_iter=1500,solver='liblinear').fit(xtr,y[train])
                probs[name][test]=clf.predict_proba(xte)[:,1]
        report['detectors'][kind]={name:float(roc_auc_score(ys[name],p)) for name,p in probs.items()}
    base.OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(base.OUT/'simple_predictions.npz',ids=data['ids'],pred_score=pred_score,pred_class=pred_class,**ch)
    (base.OUT/'deeper_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SIMPLE POLICIES',json.dumps({k:{m:round(v[m]['delta'],5) for m in ['acc','f1','tokens']} for k,v in report['simple_crossfit'].items()},indent=2),flush=True)
    print('DETECTORS',json.dumps(report['detectors'],indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
