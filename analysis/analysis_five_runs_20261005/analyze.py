"""Five-run, matched-question budget and success-set audit. Offline, no inference."""
from pathlib import Path
from itertools import combinations
from collections import Counter
import gzip, hashlib, json, math
import numpy as np

BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'analysis_matched500_raw_20261004/input'
RUNS=['old500','new500','repeat_1','repeat_2','repeat_3']
PATHS=[OLD/r if j<2 else BASE/'input'/r for j,r in enumerate(RUNS)]
B=['100','200','300','400','500','600','FULL']
M=['EM','F1','ACC']
FIELDS=['answer_em','answer_f1','answer_acc']
OUT=BASE/'results'
IDX=np.random.default_rng(20261005).integers(0,500,(10000,500),dtype=np.int16)

def rows(p):
    with (gzip.open if str(p).endswith('.gz') else open)(p,'rt',encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]

def save(name,x):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def interval(x):
    """Input one value per question, after averaging target runs if needed."""
    x=np.asarray(x,float)
    assert len(x)==500
    return [float(z) for z in np.quantile(x[IDX].mean(1),[.025,.975])]

def mean_ci(x):return {'mean':float(np.mean(x)),'ci95_question_cluster':interval(x)}

def paired_p(a,b):
    gain=int(np.sum((a==1)&(b==0)));harm=int(np.sum((a==0)&(b==1)))
    n=gain+harm
    p=min(1.,2*sum(math.comb(n,j) for j in range(min(gain,harm)+1))/2**n) if n else 1.
    return {'gain':gain,'harm':harm,'mcnemar_exact_p':p}

def load():
    rr=[rows(p/'analysis/scored_records.jsonl.gz') for p in PATHS]
    roots=[{x['sample_id']:x for x in rows(p/'roots/roots.merged.jsonl.gz')} for p in PATHS]
    ids=sorted(roots[0]);assert len(ids)==500
    for rt in roots:assert set(rt)==set(ids)
    maps=[{(x['sample_id'],str(x['condition_group'])):x for x in rs} for rs in rr]
    for mp,rs in zip(maps,rr):assert len(mp)==len(rs)
    Y=np.zeros((5,500,7,3));cost=np.zeros((5,500,7));inp=cost.copy();out=cost.copy();ev=cost.copy();sr=cost.copy();bad=cost.copy()
    rootcost=np.zeros((5,500))
    for r in range(5):
        for i,sid in enumerate(ids):
            for k in ['question','top_k','temperature','thinking_control']:
                assert roots[r][sid][k]==roots[0][sid][k]
            for j,b in enumerate(B):
                x=maps[r][sid,b];Y[r,i,j]=[x[f] for f in FIELDS]
                inp[r,i,j]=x['standalone_llm_input_tokens'];out[r,i,j]=x['standalone_llm_output_tokens']
                cost[r,i,j]=inp[r,i,j]+out[r,i,j]
                rc=cost[r,i,j]-x['continuation_llm_input_tokens']-x['continuation_llm_output_tokens']
                if j:assert rootcost[r,i]==rc
                rootcost[r,i]=rc
                ev[r,i,j]=x['visible_evidence_tokens_total'];sr[r,i,j]=x['search_calls'];bad[r,i,j]=x['local_fallback']
                if x['local_fallback']:assert not any(x[f] for f in FIELDS)
    return ids,rr,roots,maps,Y,cost,inp,out,ev,sr,bad,rootcost

def rank_budgets(y,c):
    # y,c [question,budget]. Higher mean quality, then LOWER training cost only.
    return np.stack([np.lexsort((np.arange(7),c[i],-y[i])) for i in range(len(y))])

def selected_stats(Y,cost,inp,out,ev,sr,bad,targets,selection,ci=False):
    q=np.arange(500)
    if selection.ndim==1:selection=np.tile(selection,(len(targets),1))
    vals={m:np.stack([Y[t,q,selection[z],mi] for z,t in enumerate(targets)]).mean(0) for mi,m in enumerate(M)}
    for name,arr in [('input_tokens',inp),('output_tokens',out),('total_tokens',cost),('evidence_tokens',ev),('search_calls',sr),('fallback_rate',bad)]:
        vals[name]=np.stack([arr[t,q,selection[z]] for z,t in enumerate(targets)]).mean(0)
    res={k:float(v.mean()) for k,v in vals.items()}
    if ci:res['ci95_question_cluster']={k:interval(v) for k,v in vals.items() if k in M+['total_tokens']}
    return res,vals

def main():
    ids,rr,roots,maps,Y,C,I,O,E,S,F,RC=load();Q=np.arange(500);em=Y[:,:,:,0];acc=Y[:,:,:,2]
    fixed=[];native=[]
    for r in range(5):
        rec={'run':RUNS[r],'records':len(rr[r]),'fixed':{},'oracles':{}}
        for j,b in enumerate(B):rec['fixed'][b]=selected_stats(Y,C,I,O,E,S,F,[r],np.full(500,j))[0]
        for mi,m in enumerate(M):
            pick=rank_budgets(Y[r,:,:,mi],C[r])[:,0]
            st,vs=selected_stats(Y,C,I,O,E,S,F,[r],pick)
            full=Y[r,:,6,mi];best=int(np.argmax(Y[r,:,:,mi].mean(0)))
            st.update(selection_objective=m,budget_counts=dict(Counter(B[j] for j in pick)),
                delta_vs_FULL=mean_ci(vs[m]-full),delta_vs_best_fixed=mean_ci(vs[m]-Y[r,:,best,mi]),best_fixed_budget=B[best])
            rec['oracles'][m]=st
        fixed.append(rec)
        native.append({'run':RUNS[r], 'native_per_question_variable_grid_oracle':{
            m:float(np.mean([max(x[FIELDS[mi]] for (s,b),x in maps[r].items() if s==sid) for sid in ids])) for mi,m in enumerate(M)}})
    save('per_run_metrics.json',fixed);save('native_grid_sensitivity.json',native)
    sets={}
    for name,yy in [('EM',em),('ACC',acc)]:
        freq=yy.sum(0);anyrun=yy.max(2);allrun=anyrun.min(0)>0;ever=anyrun.max(0)>0
        sets[name]={'ever_any_success':int(ever.sum()),'any_success_in_every_run':int(allrun.sum()),
            'same_budget_success_in_every_run':int((freq.max(1)==5).sum()),
            'success_exists_in_run_count_histogram':dict(Counter(map(str,anyrun.sum(0).astype(int)))),
            'stable_core_question_counts_at_least_runs':{str(k):int((freq.max(1)>=k).sum()) for k in range(1,6)},
            'cell_success_frequency_histogram':dict(Counter(map(str,freq.astype(int).ravel()))),
            'mean_success_set_size_by_run':yy.sum(2).mean(1).tolist(),
            'intersection_size_histogram':dict(Counter(map(str,(freq==5).sum(1))))}
        pairs=[]
        for a,b in combinations(range(5),2):
            aa=yy[a].astype(bool);bb=yy[b].astype(bool);ni=(aa&bb).sum(1);nu=(aa|bb).sum(1)
            both=(aa.any(1)&bb.any(1));eligible=nu>0
            wa=rank_budgets(yy[a],C[a])[:,0];wb=rank_budgets(yy[b],C[b])[:,0]
            pairs.append({'a':RUNS[a],'b':RUNS[b],'both_have_success':int(both.sum()),'either_have_success':int(eligible.sum()),
                'overlapping_success_sets':int((ni>0).sum()),'disjoint_despite_both_nonempty':int((both&(ni==0)).sum()),
                'jaccard_nonempty_union':float((ni[eligible]/nu[eligible]).mean()),
                'jaccard_both_nonempty':float((ni[both]/nu[both]).mean()),
                'exact_set_agreement_all':float((aa==bb).all(1).mean()),
                'exact_set_agreement_nonempty_union':float((aa[eligible]==bb[eligible]).all(1).mean()),
                'winner_agreement_all':float((wa==wb).mean()),
                'winner_agreement_both_nonempty':float((wa[both]==wb[both]).mean()),
                'observed_overlap_mean_both':float(ni[both].mean()),
                'uniform_same_cardinality_expected_overlap_both':float((aa.sum(1)[both]*bb.sum(1)[both]/7).mean())})
        sets[name]['pairs']=pairs
    save('success_sets.json',sets)
    # Chronological transfer, untouched R3-R5 test outcomes (selection R1/R2 only).
    transfers=[]
    for mi,m in enumerate(M):
        train=Y[:2,:,:,mi].mean(0);tc=C[:2].mean(0)
        historical=rank_budgets(train,tc)[:,0]
        policies={'FULL':np.full(500,6),'fixed500':np.full(500,4),
            'fixed600':np.full(500,5),
            'best_global_fixed_on_old2':np.full(500,np.lexsort((np.arange(7),tc.mean(0),-train.mean(0)))[0]),
            'old1_winner':rank_budgets(Y[0,:,:,mi],C[0])[:,0],
            'old2_winner':rank_budgets(Y[1,:,:,mi],C[1])[:,0],
            'old2_mean_per_question':historical}
        for name,pick in policies.items():
            aggregate,vals=selected_stats(Y,C,I,O,E,S,F,[2,3,4],pick,ci=True)
            record={'objective':m,'policy':name,'aggregate':aggregate,'budget_counts':dict(Counter(B[j] for j in pick)),'per_test_run':[]}
            for bas,bp in [('FULL',np.full(500,6)),('fixed500',np.full(500,4)),('best_global_fixed_on_old2',policies['best_global_fixed_on_old2'])]:
                _,bv=selected_stats(Y,C,I,O,E,S,F,[2,3,4],bp)
                record['vs_'+bas]={k:mean_ci(vals[k]-bv[k]) for k in M+['total_tokens']}
            for t in [2,3,4]:
                st,_=selected_stats(Y,C,I,O,E,S,F,[t],pick)
                st.update(run=RUNS[t])
                if m in ['EM','ACC']:st['vs_FULL_binary']=paired_p(Y[t,Q,pick,mi],Y[t,:,6,mi])
                record['per_test_run'].append(st)
            transfers.append(record)
    save('temporal_transfer.json',transfers)
    # Compare full successful set to equal-size controls and fixed-K shortlist upper bounds.
    set_transfer={}
    for name,yy in [('EM',em),('ACC',acc)]:
        old_freq=yy[:2].sum(0);tc=C[:2].mean(0)
        methods={}
        masks={'old_intersection':old_freq==2,'old_union':old_freq>0}
        for k in [1,2,3]:
            order=rank_budgets(old_freq,tc);mask=np.zeros((500,7),bool)
            mask[Q[:,None],order[:,:k]]=True;masks['old_ranked_top'+str(k)]=mask
            candidates=list(combinations(range(7),k))
            def global_objective(bs):return (-float(yy[:2][:,:,list(bs)].max(2).mean()),float((C[:2][:,:,list(bs)].sum(2)-(k-1)*RC[:2]).mean()),bs)
            best=min(candidates,key=global_objective)
            mask=np.zeros((500,7),bool);mask[:,list(best)]=True;masks['global_fixed_top'+str(k)]=mask
        for method,mask in masks.items():
            cardinality=mask.sum(1);eligible=cardinality>0
            hits=(yy[2:]*mask).max(2)
            costs=np.stack([(C[t]*mask).sum(1)-np.maximum(0,cardinality-1)*RC[t]+(cardinality==0)*RC[t] for t in [2,3,4]])
            intersections=(yy[2:]*mask).sum(2)
            precision=np.divide(intersections,cardinality[None,:],out=np.zeros_like(intersections),where=cardinality[None,:]>0)
            newsize=yy[2:].sum(2)
            recall=np.divide(intersections,newsize,out=np.zeros_like(intersections),where=newsize>0)
            methods[method]={'candidate_mean_size':float(cardinality.mean()),'candidate_nonempty_questions':int(eligible.sum()),
                'candidate_counts_by_budget':{b:int(mask[:,j].sum()) for j,b in enumerate(B)},
                'hit_is_hindsight_oracle_not_executable_accuracy':True,
                'oracle_hit':mean_ci(hits.mean(0)),'mean_bundle_tokens':mean_ci(costs.mean(0)),
                'precision_given_nonempty':float(precision[:,eligible].mean()),
                'recall_given_test_success_exists':float(recall[newsize>0].mean()),
                'all3_have_a_hit_questions':int((hits.min(0)>0).sum()),
                'same_old_candidate_works_all3_questions':int(((yy[2:].min(0)>0)&mask).any(1).sum()),
                'per_test_run_oracle_hit':hits.mean(1).tolist()}
        methods['ranked_vs_fixed']={}
        for k in [1,2,3]:
            a=masks['old_ranked_top'+str(k)];b=masks['global_fixed_top'+str(k)]
            va=(yy[2:]*a).max(2).mean(0);vb=(yy[2:]*b).max(2).mean(0)
            methods['ranked_vs_fixed'][str(k)]=mean_ci(va-vb)
        set_transfer[name]=methods
    save('set_transfer.json',set_transfer)
    # LOO and number-of-repeats curves, always exclude target run.
    curves={};loo=[]
    for name,yy in [('EM',em),('ACC',acc)]:
        metrics=[]
        for ntrain in [1,2,3,4]:
            points=[];per_question=[];per_cost=[]
            for t in range(5):
                for train in combinations([r for r in range(5) if r!=t],ntrain):
                    pick=rank_budgets(yy[list(train)].mean(0),C[list(train)].mean(0))[:,0]
                    success=yy[t,Q,pick];cost=C[t,Q,pick]
                    per_question.append(success);per_cost.append(cost)
                    if ntrain==4:
                        mi=0 if name=='EM' else 2
                        st,_=selected_stats(Y,C,I,O,E,S,F,[t],pick)
                        loo.append({'objective':name,'test_run':RUNS[t],**st,'vs_FULL':mean_ci(success-yy[t,:,6]),'selection_counts':dict(Counter(B[j] for j in pick))})
            metrics.append({'n_training_runs':ntrain,'number_of_train_test_combinations':len(per_question),
                'heldout_'+name:mean_ci(np.mean(per_question,axis=0)),
                'total_tokens':mean_ci(np.mean(per_cost,axis=0))})
        curves[name]=metrics
    save('learning_curve.json',curves);save('leave_one_run_out.json',loo)
    upper={}
    for name,yy in [('EM',em),('ACC',acc)]:
        upper[name]={}
        for runset,label in [(list(range(5)),'all5'),([1,2,3,4],'exclude_legacy_R1'),([2,3,4],'new3_only')]:
            freq=yy[runset].mean(0);pick=rank_budgets(freq,C[runset].mean(0))[:,0]
            val=yy[runset][:,Q,pick].mean(0)
            anytime=yy[runset].max(2).mean(0)
            upper[name][label]={'persistent_one_budget_hindsight':mean_ci(val),'per_run_hindsight_oracle':mean_ci(anytime),
                'adaptation_gap':mean_ci(anytime-val),'total_tokens':float(C[runset][:,Q,pick].mean()),
                'same_budget_correct_all_runs':int((yy[runset].min(0).max(1)>0).sum()),
                'any_budget_correct_each_run':int((yy[runset].max(2).min(0)>0).sum())}
    save('hindsight_bounds.json',upper)
    # Match roots, and stratify old-two policy gains by root agreement / failures.
    root_pairs=[]
    for a,b in combinations(range(5),2):
        root_pairs.append({'a':RUNS[a],'b':RUNS[b],
            'same_root_query':sum(roots[a][sid]['query']==roots[b][sid]['query'] for sid in ids),
            'same_root_raw_output':sum(roots[a][sid]['action_attempts'][0]['raw_output']==roots[b][sid]['action_attempts'][0]['raw_output'] for sid in ids),
            'same_root_prompt':sum(roots[a][sid]['action_attempts'][0]['prompt']==roots[b][sid]['action_attempts'][0]['prompt'] for sid in ids)})
    save('root_pairs.json',root_pairs)
    perquestion=[]
    for i,sid in enumerate(ids):
        perquestion.append({'sample_id':sid,'question':roots[0][sid]['question'],
            'success_sets':{RUNS[r]:[B[j] for j in range(7) if em[r,i,j]] for r in range(5)},
            'all5_frequency_by_budget':{b:int(em[:,i,j].sum()) for j,b in enumerate(B)},
            'old2_policy_budget':B[rank_budgets(em[:2].mean(0),C[:2].mean(0))[i,0]],
            'root_queries':{RUNS[r]:roots[r][sid]['query'] for r in range(5)}})
    save('per_question_sets.json',perquestion)
    np.savez_compressed(OUT/'matched_arrays.npz',Y=Y,C=C,I=I,O=O,E=E,S=S,F=F,RC=RC,ids=np.array(ids))
    save('analysis_metadata.json',{'runs':RUNS,'budgets':B,'n':500,'bootstrap':10000,'seed':20261005,
        'records_all_native':sum(len(x) for x in rr),'records_common_primary':17500,
        'metric_warning':'ACC is normalized contains, not LLM Judge; strict fallback=0',
        'cost_definition':'standalone input+output tokens; root once; no counterfactual enumeration cost included',
        'source_raw_audit_required':True})
    print('Statistics completed; temporal EM old2 policy:')
    print(json.dumps(next(x for x in transfers if x['objective']=='EM' and x['policy']=='old2_mean_per_question'),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
