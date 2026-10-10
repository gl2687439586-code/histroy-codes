"""Train on rated questions, transfer a zero-generation router to unrated ones.

The additional 2,267 questions are disjoint from the READ-score training set.
They have been inspected in earlier research, so this is a transfer audit,
not a final untouched confirmation set.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import readscore1942_prior_analysis as base


def signal_predict(data,train,test,target):
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=18000,sublinear_tf=True)
    tr=vec.fit_transform(data['q'][train]); te=vec.transform(data['q'][test])
    model=Ridge(alpha=10,solver='lsqr').fit(tr,target[train])
    p=model.predict(te)
    p[:,0]=np.clip(p[:,0],1,3); p[:,1]=np.clip(p[:,1],0,1)
    return p


def main():
    ratings=list(base.rows(base.SCORE_ROOT/'results/read_scores_partial.jsonl'))
    qs={x['_id']:x['question'] for x in json.load(base.DATASET.open(encoding='utf-8'))}
    by={}
    for r in base.rows(base.STAGE/'analysis/scored_records.jsonl.gz'): by.setdefault(r['sample_id'],set()).add(r['condition_label'])
    rated={r['question_id'] for r in ratings}
    rest=sorted(s for s,b in by.items() if all(x in b for x in base.BUDGETS) and s not in rated)
    dummy=[{'question_id':s,'query':qs[s],'read_basis':'','read_score':2,'read_confidence':.9,'usage':{'total_tokens':0}} for s in rest]
    data=base.load_data(ratings+dummy); n=len(data['ids'])
    train=np.flatnonzero(np.isin(data['ids'],list(rated))); test=np.flatnonzero(np.isin(data['ids'],rest))
    assert len(train)==1942 and len(test)==2267
    targets=np.column_stack([data['score'],data['confidence']])
    pred_signal=np.zeros((n,2)); cv=list(StratifiedKFold(5,shuffle=True,random_state=base.SEED).split(train,data['types'][train]))
    for tr0,va0 in cv: pred_signal[train[va0]]=signal_predict(data,train[tr0],train[va0],targets)
    pred_signal[test]=signal_predict(data,train,test,targets)
    student=dict(data); student['score']=pred_signal[:,0]; student['confidence']=pred_signal[:,1]
    families=['semantic_neighbor_without_read','structure_geometry','read_geometry','read_structure_geometry','read_semantic_neighbor']
    policies={}; configs={}
    for name in families:
        kind,method,params=base.FAMILIES[name]; candidates=[]
        for param in params:
            pr=np.zeros((len(train),18))
            for tr0,va0 in cv: pr[va0]=base.predict(student,train[tr0],train[va0],kind,method,param)
            for op in base.OPERATING:
                ch=base.choose(pr,op); a,f,t=base.quick(data,train,ch)
                if a>=0 and f>=0 and t<=-100: candidates.append((a+.2*f,-t,param,op,a,f,t))
        if candidates:
            best=max(candidates,key=lambda x:(x[0],x[1])); _,_,param,op,a,f,t=best
            pr=base.predict(student,train,test,kind,method,param); test_ch=base.choose(pr,op)
            configs[name]={'param':param,'operating':op,'source_acc_delta':a,'source_f1_delta':f,'source_token_delta':t,'source_score':a+.2*f}
        else:
            test_ch=np.full(len(test),6); configs[name]={'fallback':'FULL','source_score':0}
        ch=np.full(n,6); ch[test]=test_ch; policies[name]=ch
        print(name,configs[name],flush=True)
    selected=max(configs,key=lambda k:configs[k]['source_score'])
    policies['source_selected_family']=policies[selected]
    report={'source_n':len(train),'transfer_n':len(test),'warning':'Disjoint from READ-score training; previously inspected under other analyses, exploratory transfer only. Student features fitted before outcome-router inner folds, so source operating-point scores are exploratory; transfer outcomes never used in fitting or selection.','new_generation_calls':0,'student_target':'query-only READ score and confidence; TF-IDF Ridge','selected_on_source':selected,'configs':configs,'policies':{}}
    for name,ch in policies.items():
        report['policies'][name]=base.evaluate(data,ch,test,ci=True)
        report['policies'][name]['by_type']={ty:base.evaluate(data,ch,test[data['types'][test]==ty],ci=True) for ty in ['bridge','comparison']}
    (base.OUT/'transfer_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(base.OUT/'transfer_predictions.npz',ids=data['ids'],train=train,test=test,predicted_read=pred_signal,**policies)
    print(json.dumps({k:{m:v[m]['delta'] for m in ['acc','f1','tokens']} for k,v in report['policies'].items()},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
