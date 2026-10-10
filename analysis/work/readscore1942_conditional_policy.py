"""Separate semantic routing for bridge/comparison, without test gold labels.

The routing classifier is fitted on training question text. Budget-model and
operating-point selection is performed on inner out-of-fold data separately
for the two predicted semantic groups. This is an exploratory ablation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import readscore1942_prior_analysis as base


FAMILIES=['question_structure','structure_geometry','read_structure','read_geometry','read_structure_geometry','read_semantic_neighbor','semantic_neighbor_without_read']


def type_predict(data,train,test):
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=12000,sublinear_tf=True)
    xtr=vec.fit_transform(data['q'][train]); xte=vec.transform(data['q'][test])
    clf=LogisticRegression(C=2,max_iter=1500,solver='liblinear',random_state=base.SEED)
    clf.fit(xtr,(data['types'][train]=='comparison').astype(int))
    return clf.predict_proba(xte)[:,1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=20261002)
    ap.add_argument('--suffix',default='')
    args=ap.parse_args()
    data=base.load_data(); n=len(data['ids']); idx=np.arange(n); seed=args.seed
    cv=StratifiedKFold(5,shuffle=True,random_state=seed)
    pred_type=np.zeros(n); fold_ids=np.full(n,-1,int)
    chosen={name:np.full(n,6,int) for name in ['conditional_with_read','conditional_without_read','comparison_only_with_read']}
    selections=[]
    for fold,(train,test) in enumerate(cv.split(idx,data['types'])):
        fold_ids[test]=fold
        tp=type_predict(data,train,test); pred_type[test]=tp
        inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+10).split(train,data['types'][train]))
        train_type=np.zeros(len(train))
        for tr0,va0 in inner: train_type[va0]=type_predict(data,train[tr0],train[va0])
        cache={}
        for name in FAMILIES:
            kind,method,params=base.FAMILIES[name]
            for param in params:
                oof=np.zeros((len(train),18))
                for tr0,va0 in inner: oof[va0]=base.predict(data,train[tr0],train[va0],kind,method,param)
                cache[(name,param)]={'inner':oof}
        fold_record={'fold':fold,'routes':{}}
        for group in ['bridge','comparison']:
            pos=np.flatnonzero((train_type>=.5)==(group=='comparison'))
            tepos=np.flatnonzero((tp>=.5)==(group=='comparison'))
            for setting in ['with_read','without_read']:
                candidates=[]
                allowed=FAMILIES if setting=='with_read' else [x for x in FAMILIES if not x.startswith('read_')]
                for (name,param),pr in cache.items():
                    if name not in allowed: continue
                    for op in base.OPERATING:
                        ch=base.choose(pr['inner'][pos],op); a,f,t=base.quick(data,train[pos],ch)
                        if a>=0 and f>=0 and t<=-100: candidates.append((a+.2*f,-t,name,param,op,a,f,t))
                if candidates:
                    best=max(candidates,key=lambda x:(x[0],x[1])); _,_,name,param,op,a,f,t=best
                    pr=cache[(name,param)]
                    if 'test' not in pr:
                        kind,method,_=base.FAMILIES[name]; pr['test']=base.predict(data,train,test,kind,method,param)
                    ch=base.choose(pr['test'][tepos],op)
                    chosen['conditional_'+setting][test[tepos]]=ch
                    if group=='comparison' and setting=='with_read': chosen['comparison_only_with_read'][test[tepos]]=ch
                    decision={'name':name,'param':param,'operating':op,'inner_acc_delta':a,'inner_f1_delta':f,'inner_tokens_delta':t,'inner_n':len(pos),'test_n':len(tepos)}
                else: decision={'fallback':'FULL','inner_n':len(pos),'test_n':len(tepos)}
                fold_record['routes'][group+'_'+setting]=decision
                print('fold',fold+1,group,setting,decision,flush=True)
        selections.append(fold_record)
    report={'protocol':'5 outer / 3 inner; question-text type classifier; no held-out official type as feature','seed':seed,'type_auroc':float(roc_auc_score(data['types']=='comparison',pred_type)),'fold_selections':selections,'policies':{}}
    for name,ch in chosen.items():
        report['policies'][name]=base.evaluate(data,ch,ci=True)
        report['policies'][name]['by_type']={ty:base.evaluate(data,ch,np.flatnonzero(data['types']==ty),ci=True) for ty in ['bridge','comparison']}
    # The two arms are paired on exactly the same test questions and CV folds.
    with_ch,without_ch=chosen['conditional_with_read'],chosen['conditional_without_read']
    report['read_added_value']={}
    for metric in ['acc','em','f1','tokens']:
        d=data[metric][idx,with_ch]-data[metric][idx,without_ch]
        report['read_added_value'][metric]={'delta':float(d.mean()),'ci95':base.bootstrap(d)}
    np.savez_compressed(base.OUT/f'conditional_predictions{args.suffix}.npz',ids=data['ids'],fold_id=fold_ids,pred_type=pred_type,**chosen)
    (base.OUT/f'conditional_policy{args.suffix}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({name:{m:v[m] for m in ['acc','f1','tokens']} for name,v in report['policies'].items()},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
