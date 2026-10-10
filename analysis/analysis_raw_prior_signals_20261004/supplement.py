"""Post-discovery diagnostics: explicitly exploratory, not policy validation."""
from collections import Counter
import json
import numpy as np
import analyze as a


def main():
    feature_rows=json.loads((a.OUT/'FEATURES.json').read_text(encoding='utf-8'))['rows']
    feat={(r['run'],r['sample_id']):r for r in feature_rows}
    result={'note':'Post-discovery descriptions. Gold support used only for outcome audit, never routing features.', 'profiles':{},'same_question_profile_replication':{}}
    all_groups={};all_scored={}
    for run in ['old500','new500']:
        p=a.RAW/'input'/run
        scored={(r['sample_id'],r['condition_group']):r for r in a.rows(p/'analysis/scored_records.jsonl.gz') if r['condition_group'] in a.BUDGETS}
        all_scored[run]=scored
        ids=sorted({k[0] for k in scored})
        groups={
            'all':ids,
            'one_or_two_unique_titles':[s for s in ids if feat[run,s]['unique_title_fraction']<=.4],
            'five_unique_titles':[s for s in ids if feat[run,s]['unique_title_fraction']==1],
            'top1_query_coverage_le_0.4':[s for s in ids if feat[run,s]['query_top1_coverage']<=.4],
            'top1_query_coverage_ge_0.8':[s for s in ids if feat[run,s]['query_top1_coverage']>=.8],
        }
        all_groups[run]=groups
        profiles={}
        for name,ss in groups.items():
            budgets={}
            for b in a.BUDGETS:
                rr=[scored[s,b] for s in ss]
                fields=['answer_em','answer_f1','answer_acc','standalone_llm_input_tokens','standalone_llm_output_tokens',
                        'continuation_llm_input_tokens','continuation_llm_output_tokens','search_calls','local_fallback',
                        'root_retrieved_title_recall','cumulative_retrieved_title_recall','support_sentence_strict_recall']
                m={k:float(np.mean([r[k] for r in rr])) for k in fields}
                m['total_tokens']=m['standalone_llm_input_tokens']+m['standalone_llm_output_tokens']
                m['continuation_tokens']=m['continuation_llm_input_tokens']+m['continuation_llm_output_tokens']
                budgets[b]=m
            profiles[name]={'n':len(ss),'budgets':budgets}
        result['profiles'][run]=profiles
        # Confirm that root length correlations are not only root cap artifacts.
        clean=[s for s in ids if not feat[run,s]['root_length_or_fallback']]
        result.setdefault('root_length_without_root_caps',{})[run]={
            'n':len(clean),
            'rho_full_em':a.corr([feat[run,s]['root_output_tokens'] for s in clean],[scored[s,'FULL']['answer_em'] for s in clean]),
            'rho_full_continuation_cost':a.corr([feat[run,s]['root_output_tokens'] for s in clean],
                [scored[s,'FULL']['continuation_llm_input_tokens']+scored[s,'FULL']['continuation_llm_output_tokens'] for s in clean])}
    for g in ['one_or_two_unique_titles','five_unique_titles']:
        ss=sorted(set(all_groups['old500'][g])&set(all_groups['new500'][g]))
        rr={'n':len(ss)}
        for run in ['old500','new500']:
            v=[all_scored[run][s,'FULL'] for s in ss]
            rr[run]={'em':float(np.mean([r['answer_em'] for r in v])),
                     'total_tokens':float(np.mean([r['standalone_llm_input_tokens']+r['standalone_llm_output_tokens'] for r in v])),
                     'input_tokens':float(np.mean([r['standalone_llm_input_tokens'] for r in v])),
                     'output_tokens':float(np.mean([r['standalone_llm_output_tokens'] for r in v])),
                     'continuation_tokens':float(np.mean([r['continuation_llm_input_tokens']+r['continuation_llm_output_tokens'] for r in v]))}
        result['same_question_profile_replication'][g]=rr
    policies=json.loads((a.OUT/'CROSS_RUN_POLICIES.json').read_text(encoding='utf-8'))
    summaries=[]
    for src in ['old500','new500']:
        for lam in [0,.02]:
            for family in a.FAMILIES:
                rs=[r for r in policies if r['source']==src and r['lambda']==lam and r['family']==family]
                summaries.append({'source':src,'lambda':lam,'family':family,
                                  'em_range':[min(r['means']['answer_em'] for r in rs),max(r['means']['answer_em'] for r in rs)],
                                  'total_tokens_range':[min(r['means']['total_tokens'] for r in rs),max(r['means']['total_tokens'] for r in rs)],
                                  'beats_fixed500_em_seeds':sum(r['comparisons']['500']['answer_em']['delta']>0 for r in rs),
                                  'primary_seed':rs[0]})
    result['routing_seed_summary']=summaries
    a.save('EXPLORATORY_SUPPLEMENT.json',result)
    print('Post-discovery diagnostics saved.')
    print(json.dumps(result['same_question_profile_replication'],ensure_ascii=False))


if __name__=='__main__':main()
