"""Exploratory offline feasibility check only. No new LLM calls, no source changes.

Fixed design: old2 train, new3 test; group 5-fold by question; seeds fixed from
prior audits. One-level decision rules; first budget 200 or 500; utility
EM - lambda * additional tokens/1000, lambda in 0,.02,.05. No grid winner
reported as an independently confirmed result.
"""
import re,json
from collections import Counter
import numpy as np
import analyze as a

FEATURES=['input_tokens','output_tokens','search_calls','any_length','invalid_format',
          'answer_word_count','last_generation_output_tokens','max_generation_output_tokens',
          'repeated_query_fraction','answer_uncertainty','query_word_count']

def feat(r):
    ans=r['final_answer'];qs=[x['query'].strip().lower() for x in r['evidence']]
    outs=[x['output_tokens'] for x in r['events']]
    return [r['input_tokens'],r['output_tokens'],r['search_calls'],r['any_length'],r['answer_status']!='valid',
        len(ans.split()),outs[-1],max(outs),1-len(set(qs))/len(qs),
        int(bool(re.search(r'\b(?:unknown|unclear|insufficient|cannot|not enough|unable|not provided)\b',ans.lower()))),
        sum(len(q.split()) for q in qs)]

def action(u):
    # action0=Stop, all others=replace first answer with a fresh archived budget run.
    means=u.mean(0);return int(np.argmax(means)),float(means.max())

def fit(x,u,qids):
    ac,val=action(u);best={'feature':None,'left':ac,'right':ac,'value':val,'threshold':None}
    for j in range(x.shape[1]):
        for t in np.unique(np.quantile(x[:,j],[.2,.4,.6,.8])):
            left=x[:,j]<=t
            if len(np.unique(qids[left]))<60 or len(np.unique(qids[~left]))<60:continue
            aa,va=action(u[left]);bb,vb=action(u[~left]);v=(left.sum()*va+(~left).sum()*vb)/len(x)
            if v>best['value']+1e-12:best={'feature':j,'threshold':float(t),'left':aa,'right':bb,'value':v}
    return best

def predict(m,x):
    if m['feature'] is None:return np.full(len(x),m['left'],int)
    return np.where(x[:,m['feature']]<=m['threshold'],m['left'],m['right'])

def main():
    ids,rr,roots,maps,Y,C,I,O,E,S,F,RC=a.load();q=np.arange(500);yy=Y[:,:,:,0]
    cm=[{(r['sample_id'],r['budget']):r for r in a.rows(a.OUT/(name+'_compact.jsonl.gz'))} for name in a.RUNS]
    X=np.array([[[feat(mp[sid,b]) for b in a.B] for sid in ids] for mp in cm],float)
    allout=[]
    for base in [1,4]:
        dest=[None]+[b for b in range(7) if b!=base]
        rewards=np.zeros((5,500,7));extra=np.zeros((5,500,7))
        for j,b in enumerate(dest[1:],1):
            rewards[:,:,j]=yy[:,:,b]-yy[:,:,base];extra[:,:,j]=C[:,:,b]-RC
        for lam in [0.,.02,.05]:
            for seed in [20261004,20261005,20261006]:
                actions=np.zeros((3,500),int);fixed=np.zeros_like(actions);models=[]
                for test in np.array_split(np.random.default_rng(seed).permutation(500),5):
                    train=np.setdiff1d(q,test);u=(rewards[:2,train]-lam*extra[:2,train]/1000).reshape(-1,7)
                    model=fit(X[:2,train,base].reshape(-1,len(FEATURES)),u,np.tile(train,2));models.append(model)
                    for t in range(3):actions[t,test]=predict(model,X[t+2,test,base]);fixed[t,test]=action(u)[0]
                selected=np.stack([np.array([base if j==0 else dest[j] for j in acts]) for acts in actions])
                metrics=np.stack([Y[t+2,q,selected[t]] for t in range(3)])
                costs=np.stack([C[t+2,:,base]+extra[t+2,q,actions[t]] for t in range(3)])
                rec={'base':a.B[base],'lambda':lam,'seed':seed,'switch_fraction':float((actions>0).mean()),
                    **{m:float(metrics[:,:,j].mean()) for j,m in enumerate(a.M)},'total_tokens':float(costs.mean()),
                    'models':[{**m,'feature_name':FEATURES[m['feature']] if m['feature'] is not None else None,
                        'left_budget':'STOP' if m['left']==0 else a.B[dest[m['left']]],
                        'right_budget':'STOP' if m['right']==0 else a.B[dest[m['right']]]} for m in models]}
                if seed==20261004:
                    rec['vs_FULL']={'EM':a.mean_ci(metrics[:,:,0].mean(0)-yy[2:,:,6].mean(0)),
                                    'tokens':a.mean_ci(costs.mean(0)-C[2:,:,6].mean(0))}
                    rec['vs_fixed500']={'EM':a.mean_ci(metrics[:,:,0].mean(0)-yy[2:,:,4].mean(0)),
                                        'tokens':a.mean_ci(costs.mean(0)-C[2:,:,4].mean(0))}
                    fx=np.stack([np.array([base if j==0 else dest[j] for j in acts]) for acts in fixed])
                    fq=np.stack([yy[t+2,q,fx[t]] for t in range(3)])
                    rec['vs_global_action_EM']=a.mean_ci(metrics[:,:,0].mean(0)-fq.mean(0))
                allout.append(rec)
    a.save('postexecution_feasibility.json',{'feature_names':FEATURES,'design':'EXPLORATORY OFFLINE REPLAY, old2 training, new3 test, question-disjoint 5fold; sunk first cost included; shared frozen root reused; final answer replaced after switch. No same-root same-budget repeat control exists in these logs. All settings retained, no heldout grid tuning.',
        'results':allout})
    for x in allout:print(x['base'],x['lambda'],x['seed'],f"EM={x['EM']:.4f} F1={x['F1']:.4f} ACC={x['ACC']:.4f} tokens={x['total_tokens']:.1f} switch={x['switch_fraction']:.3f}")

if __name__=='__main__':main()
