"""Audit all recorded generation calls and prepare semantic-judge inputs. No inference."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

BASE=Path(__file__).resolve().parent
WORK=BASE.parent
RAW=WORK/'analysis_hotpot_raw_20261004/input/full7405'
OUT=BASE/'semantic_acc_pending'
SOURCE_GOLD=WORK/'work/hotpot_signal_search_v2/hotpot_dev_distractor_v1.json'


def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as handle:
        for line in handle:
            if line.strip():yield json.loads(line)


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def key_hash(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def cap_check(attempt):
    reason=attempt['finish_reason']
    finish=reason.get('type') if isinstance(reason,dict) else reason
    count=int(attempt['output_tokens'])
    cap=int(attempt['max_new_tokens'])
    assert cap>0 and count>=0
    return {'exclude': finish=='length' or count>=cap or count>=8192,
            'finish_type':finish,'output_tokens':count,'max_new_tokens':cap,
            'length_finish':finish=='length','reached_requested_limit':count>=cap,
            'reached_8192':count>=8192}


def main():
    OUT.mkdir(exist_ok=True)
    scored_path=RAW/'analysis/scored_records.jsonl.gz'
    trajectory_path=RAW/'trajectories/trajectories.merged.jsonl.gz'
    assert sha_file(scored_path)=='954e70acbdc1c720a2c0d0b3ed6c91484320bd78e87e5e26608a599c8db93ff1'
    assert sha_file(trajectory_path)=='86da77f331bcd14ca24988bff640b703fa05f93a7c93ea4af9265a9e391d6e09'
    roots={r['sample_id']:r for r in rows(RAW/'roots/roots.merged.jsonl.gz')}
    scores={(r['sample_id'],r['condition_id']):r for r in rows(scored_path)}
    gold_rows=json.loads(SOURCE_GOLD.read_text(encoding='utf-8'))
    gold={r['_id']:r for r in gold_rows}
    assert len(roots)==7405 and len(scores)==61754 and set(roots)<=set(gold)
    counts=Counter();caps=Counter();root_flags={};calls=Counter()
    for sid,root in roots.items():
        assert ' '.join(root['question'].split())==' '.join(gold[sid]['question'].split())
        counts['gold_question_whitespace_only_differences']+=root['question']!=gold[sid]['question']
        checks=[cap_check(a) for a in root['action_attempts']]
        root_flags[sid]=checks
        counts['roots_with_cap_or_length']+=any(a['exclude'] for a in checks)
        for a in checks:
            calls['root_calls']+=1
            calls['root_length_calls']+=a['length_finish']
            calls['root_reached_8192_calls']+=a['reached_8192']
            caps[str(a['max_new_tokens'])]+=1
    groups=defaultdict(lambda:Counter())
    kept_costs=defaultdict(list)
    seen=set();dedup={};scored_fields=set();raw_fields=set()
    with gzip.open(OUT/'exclusion_ledger.jsonl.gz','wt',encoding='utf-8') as excluded, \
         gzip.open(OUT/'eligible_trajectory_map.jsonl.gz','wt',encoding='utf-8') as eligible:
        for row in rows(trajectory_path):
            sid,cid=row['sample_id'],row['condition_id'];key=(sid,cid)
            assert key not in seen and key in scores
            seen.add(key);s=scores[key];root=roots[sid]
            assert row['root_checkpoint_sha256']==root['root_checkpoint_sha256']
            assert row['question']==root['question']
            assert row['search_calls']==s['search_calls']==len(row['decision_events'])
            for f in ['standalone_llm_input_tokens','standalone_llm_output_tokens']:
                assert row['usage'][f]==s[f]
            checks=[]
            for event in row['decision_events']:
                for attempt in event['attempts']:
                    a=cap_check(attempt);checks.append(a)
                    calls['continuation_calls']+=1
                    calls['continuation_length_calls']+=a['length_finish']
                    calls['continuation_reached_8192_calls']+=a['reached_8192']
                    caps[str(a['max_new_tokens'])]+=1
            root_hit=any(a['exclude'] for a in root_flags[sid])
            continuation_hit=any(a['exclude'] for a in checks)
            total=s['standalone_llm_input_tokens']+s['standalone_llm_output_tokens']
            group=(str(s['condition_group']),s['search_calls'])
            groups[group]['original']+=1
            counts['records']+=1
            scored_fields.update(s);raw_fields.update(row)
            item={'sample_id':sid,'condition_id':cid,'budget':group[0],'search_calls':group[1],
                  'total_tokens':total,'root_cap_hit':root_hit,'continuation_cap_hit':continuation_hit,
                  'answer_status':row['answer_status']}
            if root_hit or continuation_hit:
                counts['excluded']+=1;groups[group]['excluded']+=1
                counts['root_only_excluded']+=root_hit and not continuation_hit
                counts['continuation_only_excluded']+=continuation_hit and not root_hit
                counts['both_root_and_continuation_excluded']+=root_hit and continuation_hit
                item['root_cap_calls']=[a for a in root_flags[sid] if a['exclude']]
                item['continuation_cap_calls']=[a for a in checks if a['exclude']]
                excluded.write(json.dumps(item,ensure_ascii=False)+'\n')
            else:
                counts['kept']+=1;groups[group]['kept']+=1;kept_costs[group].append(total)
                counts['kept_total_tokens_above_8192']+=total>8192
                counts['kept_local_fallback']+=row['answer_status']=='local_fallback'
                # Freeze final-answer identity. No correctness inferred from answer_acc or EM.
                judge_input={'question':row['question'],'reference_answer':gold[sid]['answer'],
                             'candidate_answer':row['final_answer']}
                judge_id=key_hash(judge_input)
                dedup.setdefault(judge_id,{'judge_item_id':judge_id,'sample_id':sid,**judge_input})
                item['judge_item_id']=judge_id
                item['semantic_correct']=None
                eligible.write(json.dumps(item,ensure_ascii=False)+'\n')
            if counts['records']%10000==0:
                print(f"Audited {counts['records']}/61754; kept={counts['kept']} excluded={counts['excluded']}",flush=True)
    assert seen==set(scores)
    assert counts['kept']+counts['excluded']==61754
    with gzip.open(OUT/'semantic_judge_inputs_UNSCORED.jsonl.gz','wt',encoding='utf-8') as target:
        for judge_id in sorted(dedup):target.write(json.dumps(dedup[judge_id],ensure_ascii=False)+'\n')
    by_group=[]
    for (b,steps),c in sorted(groups.items(),key=lambda x:(float('inf') if x[0][0]=='FULL' else int(x[0][0]),x[0][1])):
        values=kept_costs[(b,steps)]
        by_group.append({'budget':b,'search_calls':steps,**dict(c),
                         'unlabelled_kept_mean_tokens':statistics.mean(values) if values else None,
                         'unlabelled_kept_min_tokens':min(values) if values else None,
                         'unlabelled_kept_max_tokens':max(values) if values else None})
    report={'status':'CAP_FILTER_COMPLETE_SEMANTIC_JUDGE_PENDING',
            'exclusion_rule':'Any root or continuation generation finishes length OR reaches its requested max_new_tokens OR outputs >=8192. Not episode total tokens.',
            'source_raw':str(trajectory_path),'source_raw_sha256':sha_file(trajectory_path),
            'source_gold':str(SOURCE_GOLD),'source_gold_sha256':sha_file(SOURCE_GOLD),
            'semantic_labels_found_in_supplied_full7405_records':False,
            'existing_answer_acc_definition':'normalized gold substring containment; NOT semantic accuracy',
            'counts':dict(counts),'calls':dict(calls),'requested_generation_caps':dict(caps),
            'unique_semantic_judge_inputs':len(dedup),'new_model_calls':0,
            'quality_split_performed':False,'by_budget_step':by_group,
            'scored_record_fields':sorted(scored_fields),'raw_record_fields':sorted(raw_fields)}
    (OUT/'AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'README_ZH.txt').write_text('\n'.join([
        '语义ACC正确/错误分层：生成上限筛选已完成，语义Judge尚未执行。',
        '现有answer_acc是词法包含指标，不能替代语义ACC。所有semantic_correct目前为null。',
        '任何首Search或后续调用finish_reason=length、触及请求生成上限或output_tokens>=8192，排除整条轨迹。',
        '达到8192只是可观测的触顶代理，不足以证明语义上的过度思考；累计轨迹tokens>8192不直接排除。',
        f"原记录{counts['records']}；排除{counts['excluded']}；保留{counts['kept']}。",
        f"精确去重后需要{len(dedup)}个question/reference/candidate语义评分；尚无模型调用。",
        'semantic_judge_inputs_UNSCORED.jsonl.gz：冻结的问题、gold、最终候选答案；仅供评分，不能用于生成。',
        'eligible_trajectory_map.jsonl.gz：每条保留轨迹映射到Judge输入及成本，正确性为null。',
        'exclusion_ledger.jsonl.gz：每条被排除轨迹的root/continuation触顶原因。',
        'AUDIT.json：各预算×step保留/排除数量、未分正确错误的成本及输入哈希。',
        '下一步需确认Judge模型、端点/本地评估方式与语义评分规则；不能直接把答案字符串命中当作Judge结果。',
        '过滤后结果仅描述未触顶子集。正式报告应同时保留全体轨迹结果和排除率，避免隐瞒长推理/失败成本。',
    ])+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','counts','calls','requested_generation_caps','unique_semantic_judge_inputs','new_model_calls']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
