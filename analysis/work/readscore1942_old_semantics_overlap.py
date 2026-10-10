"""Supplementary READ + old breadth/depth semantic scores, 153 questions.

Only whitelisted original prior fields are imported from the old joined CSV;
none of that CSV's oracle, outcome, evidence-hit or future-query fields enter
the features. Outcomes come exclusively from the current stage snapshot.
"""
from __future__ import annotations

import csv
import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import readscore1942_prior_analysis as base


def main():
    data=base.load_data()
    with (base.ROOT/'evidence_need_oracle_association_20260902/joined_query_scores_oracle.csv').open(encoding='utf-8-sig') as f:
        old={r['sample_id']:{k:float(r[k]) for k in ['breadth_score','depth_score','confidence']} for r in csv.DictReader(f)}
    ix=np.asarray([i for i,s in enumerate(data['ids']) if s in old]); n=len(ix)
    semantic=np.asarray([[old[s]['breadth_score'],old[s]['depth_score'],old[s]['breadth_score']-old[s]['depth_score'],old[s]['confidence']] for s in data['ids'][ix]])
    read=base.numeric(data,'score_conf')[ix]; low=base.numeric(data,'lowcost')[ix]
    features={'read_only':read,'old_breadth_depth':semantic,'read_breadth_depth':np.column_stack([read,semantic]),'read_lowcost':np.column_stack([read,low]),'read_breadth_depth_lowcost':np.column_stack([read,semantic,low,semantic*data['score'][ix,None]])}
    cv=StratifiedKFold(3,shuffle=True,random_state=20261004)
    y=base.target(data)[ix]; policies={k:np.full(len(data['ids']),6) for k in features}; selected=[]
    for fold,(tr,te) in enumerate(cv.split(np.arange(n),data['types'][ix])):
        inner=list(StratifiedKFold(2,shuffle=True,random_state=base.SEED+fold).split(tr,data['types'][ix[tr]]))
        for name,x in features.items():
            candidates=[]
            for alpha in [50,500]:
                pred=np.zeros((len(tr),18))
                for tr0,va0 in inner:
                    sc=StandardScaler().fit(x[tr[tr0]])
                    m=Ridge(alpha=alpha).fit(sc.transform(x[tr[tr0]]),y[tr[tr0]])
                    pred[va0]=m.predict(sc.transform(x[tr[va0]]))
                for op in base.OPERATING:
                    ch=base.choose(pred,op); a,f,t=base.quick(data,ix[tr],ch)
                    if a>=0 and f>=0 and t<=-100: candidates.append((a+.2*f,-t,alpha,op))
            if candidates:
                _,_,alpha,op=max(candidates,key=lambda x:(x[0],x[1]))
                sc=StandardScaler().fit(x[tr]); m=Ridge(alpha=alpha).fit(sc.transform(x[tr]),y[tr]); ch=base.choose(m.predict(sc.transform(x[te])),op)
                policies[name][ix[te]]=ch
                selected.append({'fold':fold,'name':name,'alpha':alpha,'op':op})
            else: selected.append({'fold':fold,'name':name,'fallback':'FULL'})
    report={'n':n,'warning':'Small 153-question overlap. Exploratory only; old scorer extraction cost is not available here and is excluded.','feature_whitelist':['breadth_score','depth_score','confidence'],'fold_selections':selected,'policies':{k:base.evaluate(data,ch,ix,ci=True) for k,ch in policies.items()}}
    (base.OUT/'old_semantics_overlap.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:{m:v[m]['delta'] for m in ['acc','f1','tokens']} for k,v in report['policies'].items()},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
