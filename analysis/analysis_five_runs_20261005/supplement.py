"""Audit sensitivity to tie-breaking, protocol flags and topic/difficulty effects."""
import json,hashlib
import numpy as np
import analyze as a

def main():
    ids,rr,roots,maps,Y,C,I,O,E,S,F,RC=a.load();Q=np.arange(500);yy=Y[:,:,:,0]
    summaries=[];masks={};bad_answers=[]
    for r,label in enumerate(a.RUNS):
        cs=a.rows(a.OUT/(label+'_compact.jsonl.gz'));sm={};mask=np.zeros((500,7),bool)
        rowmap={x['sample_id']:i for i,x in enumerate(cs)}
        strict_counts={'native_records':len(cs),'fallback':0,'any_length':0,'root_length':0,'compacted':0,'unclosed_think_answer':0}
        for x in cs:
            strict_counts['fallback']+=x['answer_status']!='valid'
            for k in ['any_length','root_length','compacted']:strict_counts[k]+=bool(x[k])
        # Inspect original output to avoid treating an accepted Search tag as an Answer failure.
        for x in a.rows(a.PATHS[r]/'trajectories/trajectories.merged.jsonl.gz'):
            last=x['decision_events'][-1];out=last['attempts'][-1]['output'];pos=out.rfind('<answer>')
            suspicious=last['action']=='answer' and pos>=0 and out.rfind('<think>',0,pos)>out.rfind('</think>',0,pos)
            strict_counts['unclosed_think_answer']+=suspicious
            if suspicious:
                b='FULL' if x['exposure_mode']=='full_per_search' else str(x['root_exposure_tokens'])
                bad_answers.append({'run':label,'sample_id':x['sample_id'],'budget':b,'answer':x['final_answer']})
                if b in a.B:mask[ids.index(x['sample_id']),a.B.index(b)]=True
        for b in a.B:
            xs=[x for x in cs if x['budget']==b]
            sm[b]={'fallback':sum(x['answer_status']!='valid' for x in xs),'length':sum(x['any_length'] for x in xs),'compacted':sum(x['compacted'] for x in xs)}
        summaries.append({'run':label,**strict_counts,'primary_by_budget':sm});masks[r]=mask
    a.save('raw_protocol_summary.json',summaries);a.save('unclosed_think_answer_cases.json',bad_answers)
    # One question with a stable successful budget is not necessarily budget-sensitive.
    persistent=yy.min(0);ever=yy.max(0)
    qu={'all35_success_questions':int(yy.min((0,2)).sum()),'all35_failure_questions':int((yy.max((0,2))==0).sum()),
        'mixed_over_runs_or_budgets_questions':int(((yy.max((0,2))>0)&(yy.min((0,2))==0)).sum()),
        'has_both_stable_success_and_stable_failure_budget':int(((persistent.max(1)==1)&(ever.min(1)==0)).sum()),
        'five_run_same_budget_success_but_not_all35_success':int(((persistent.max(1)==1)&(persistent.min(1)==0)).sum())}
    freq=yy[:2].sum(0);out=[]
    for f in [0,1,2]:
        mask=freq==f;total=int(mask.sum());success=float((yy[2:]*mask).sum())
        out.append({'old_success_count':f,'question_budget_cells':total,'new3_cell_success_rate':success/(3*total)})
    qu['historical_success_frequency_calibration']=out
    eligible=((freq==2).any(1)&(freq==0).any(1))
    pos=np.array([yy[2:,i,freq[i]==2].mean() for i in np.flatnonzero(eligible)])
    neg=np.array([yy[2:,i,freq[i]==0].mean() for i in np.flatnonzero(eligible)])
    draws=np.random.default_rng(20261005).integers(0,len(pos),(10000,len(pos)))
    qu['within_question_old_consistent_success_vs_failure']={'eligible_questions':int(eligible.sum()),
        'old_two_success_budget_new_success':float(pos.mean()),'old_two_failure_budget_new_success':float(neg.mean()),
        'paired_difference':float((pos-neg).mean()),'ci95':np.quantile((pos-neg)[draws].mean(1),[.025,.975]).tolist(),
        'note':'Eligibility uses old runs only; equal weight per eligible question, new3 averaged.'}
    a.save('nontrivial_budget_structure.json',qu)
    train=yy[:2].mean(0);tc=C[:2].mean(0)
    choices={'old2_quality_then_cost':a.rank_budgets(train,tc)[:,0],
        'old2_quality_then_nominal_budget':a.rank_budgets(train,np.tile(np.arange(7),(500,1)))[:,0],
        'old2_cost_only':a.rank_budgets(np.zeros((500,7)),tc)[:,0]}
    sens=[]
    for name,pick in choices.items():
        st,vs=a.selected_stats(Y,C,I,O,E,S,F,[2,3,4],pick)
        out={'policy':name,'original':st,'vs_fixed600':{k:a.mean_ci(vs[k]-(Y[2:,:,5,a.M.index(k)].mean(0) if k in a.M else C[2:,:,5].mean(0))) for k in a.M+['total_tokens']}}
        out['vs_old2_cost_only']={}
        _,cv=a.selected_stats(Y,C,I,O,E,S,F,[2,3,4],choices['old2_cost_only'])
        out['vs_old2_cost_only']={k:a.mean_ci(vs[k]-cv[k]) for k in a.M+['total_tokens']}
        altered=Y.copy()
        for r in range(5):altered[r][masks[r]]=0
        out['suspect_think_answers_zeroed']=a.selected_stats(altered,C,I,O,E,S,F,[2,3,4],pick)[0]
        # Evaluable strata reported without pretending they are random subsets.
        strata=[]
        for t in [2,3,4]:
            same=np.array([roots[t][sid]['query']==roots[0][sid]['query'] or roots[t][sid]['query']==roots[1][sid]['query'] for sid in ids])
            for label,mask in [('query_matches_either_old',same),('query_differs_from_both_old',~same)]:
                strata.append({'run':a.RUNS[t],'stratum':label,'n':int(mask.sum()),
                    'selected_EM':float(yy[t,Q,pick][mask].mean()),'FULL_EM':float(yy[t,:,6][mask].mean()),
                    'selected_tokens':float(C[t,Q,pick][mask].mean()),'FULL_tokens':float(C[t,:,6][mask].mean())})
        out['root_strata']=strata;sens.append(out)
    a.save('transfer_sensitivity.json',sens)
    # Source config comparisons and actual file hashes.
    new=json.loads((a.BASE/'input/BATCH_CONFIG.json').read_text(encoding='utf-8'))
    old=json.loads((a.OLD/'source_metadata/new500/FULL7405_CONFIG.json').read_text(encoding='utf-8'))
    compare={'protocol_differences':{k:[old['protocol'].get(k),new['protocol'].get(k)] for k in set(old['protocol'])|set(new['protocol']) if old['protocol'].get(k)!=new['protocol'].get(k)},
        'shared_package_hash_differences':{k:[v,new['package_hashes'][k]] for k,v in old['package_hashes'].items() if k in new['package_hashes'] and new['package_hashes'][k]!=v},
        'model_config_hash_matches':old['model_config_sha256']==new['model_config_sha256'],
        'retriever_config_hash_matches':old['retriever_config_sha256']==new['retriever_config_sha256'],
        'library_versions_match':old['core_library_versions']==new['core_library_versions'],'dataset_hash_checks':[]}
    for r in ['repeat_1','repeat_2','repeat_3']:
        for p,k in [('dataset/questions.no_labels.jsonl','questions_sha256'),('dataset/gold_labels.sealed.jsonl','labels_sha256')]:
            val=hashlib.sha256((a.BASE/'input'/r/p).read_bytes()).hexdigest()
            assert val==new[k]
            compare['dataset_hash_checks'].append({'run':r,'file':p,'matches_config':True})
    a.save('configuration_comparison.json',compare)
    print('Supplement audit complete',json.dumps(qu,ensure_ascii=False))

if __name__=='__main__':main()
