"""Audit full semantic review coverage before emitting ACC=1 / ACC=0 tables."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics
from combine_em_semantic import aggregate, format_table, read_gz, key

BASE = Path(__file__).resolve().parent
WORK = BASE.parent
OUT = BASE/'full_semantic_review'


def write_json(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    initial_path=BASE/'codex_direct_semantic/batch_001_labelled_trajectories.json'
    initial=json.loads(initial_path.read_text(encoding='utf-8'))
    reviews={}
    for r in initial:
        reviews[r['judge_item_id']] = dict(semantic_correct=r['semantic_correct'],
                                         reason_zh=r['reason_zh'], note=r.get('review_flag',''),
                                         reviewer='Codex main direct review batch_001')
    compiled_path=OUT/'compiled_labels.json'
    if compiled_path.exists():
        for r in json.loads(compiled_path.read_text(encoding='utf-8')):
            jid=r['judge_item_id']
            if jid in reviews and reviews[jid]['semantic_correct'] is not None:
                assert reviews[jid]['semantic_correct']==r['semantic_correct']
            reviews[jid]=r
    adjudication_path=OUT/'adjudications.json'
    adjudications=[]
    by_item_path=OUT/'adjudications_by_item.json'
    if by_item_path.exists():
        adoc=json.loads(by_item_path.read_text(encoding='utf-8'))
        queue=json.loads((OUT/'queue.json').read_text(encoding='utf-8'))
        lookup={a['i']:a['judge_item_id'] for c in queue['chunks'] for q in c['questions'] for a in q['answers']}
        for item,decision,reason,flag in adoc['decisions']:
            adjudications.append(dict(judge_item_id=lookup[item],semantic_correct=decision,
                                       reason_zh=reason, note=flag, queue_item_id=item, reviewer=adoc['reviewer']))
    if adjudication_path.exists():
        adjudications+=json.loads(adjudication_path.read_text(encoding='utf-8'))['decisions']
    if adjudications:
        assert len({r['judge_item_id'] for r in adjudications})==len(adjudications)
        for r in adjudications:
            assert r['judge_item_id'] in reviews
            assert r['semantic_correct'] in (0,1) and r['reason_zh']
            r={**r,'original_review':reviews[r['judge_item_id']]}
            reviews[r['judge_item_id']]=r
    strict_path=OUT/'strict_gold_overrides.json'
    strict_overrides=[]
    if strict_path.exists():
        strict=json.loads(strict_path.read_text(encoding='utf-8'))
        for item,value,reason in strict['by_item']:
            strict_overrides.append((lookup[item],value,reason))
        strict_overrides+=strict['by_judge_item']
        assert len({r[0] for r in strict_overrides})==len(strict_overrides)
        for jid,value,reason in strict_overrides:
            assert jid in reviews and value in (0,1) and reason
            reviews[jid]=dict(judge_item_id=jid,semantic_correct=value,reason_zh=reason,
                             note='original_gold_policy; reference_defect_or_nonunique_question',
                             reviewer=strict['reviewer'],previous_context_review=reviews[jid])
    scored_path=WORK/'audit_hotpot_full7405_v433_20261001/input/analysis/scored_records.jsonl.gz'
    scored={key(r):r for r in read_gz(scored_path)}
    eligible_path=BASE/'semantic_acc_pending/eligible_trajectory_map.jsonl.gz'
    inputs={r['judge_item_id']:r for r in read_gz(BASE/'semantic_acc_pending/semantic_judge_inputs_UNSCORED.jsonl.gz')}
    combined=[]
    conflict=[]
    pending={}
    for r in read_gz(eligible_path):
        source=scored[key(r)]
        assert r['total_tokens']==source['standalone_llm_input_tokens']+source['standalone_llm_output_tokens']
        assert not r['root_cap_hit'] and not r['continuation_cap_hit']
        assert r['budget']==str(source['condition_group']) and r['search_calls']==source['search_calls']
        em=source['answer_em']
        review=reviews.get(r['judge_item_id'])
        sem=review['semantic_correct'] if review else None
        if em==1:
            acc=1; origin='exact_match'
            if sem==0:conflict.append(key(r))
        elif sem is not None:
            acc=sem; origin='direct_semantic'
        else:
            acc=None; origin='needs_adjudication' if review else 'not_reviewed'
            pending[r['judge_item_id']]={**inputs[r['judge_item_id']], 'review':review}
        combined.append({**r,'answer_em':em,'semantic_correct':sem,'combined_acc':acc,
                         'label_source':origin,'review':review})
    assert len(combined)==58868 and len({key(r) for r in combined})==58868
    assert len(scored)==61754
    summary=aggregate(combined, 'full7405_semantic_review')
    summary['all_labels_complete']=summary['pending']==0
    summary['independent_human_gold']=False
    summary['evaluation_standard']='EM positive OR Codex semantic match against ORIGINAL frozen gold. User confirmed: reference defects are logged separately and never repaired in primary ACC. No unreviewed candidate is defaulted to incorrect.'
    status=dict(all_kept_records_labelled=summary['pending']==0,
                original_records=len(scored), excluded_cap_or_length=61754-len(combined),
                kept_records=len(combined), questions_after_exclusion=summary['unique_questions'],
                acc1=summary['acc1'],acc0=summary['acc0'],pending_trajectories=summary['pending'],
                pending_unique_items=len(pending),pending_reasons=dict(Counter(r['label_source'] for r in combined if r['combined_acc'] is None)),
                em_positive=summary['exact_match_positives'],semantic_additional_positive=summary['semantic_additional_positives'],
                direct_review_unique_items=len(reviews),adjudicated_unique_items=len(adjudications),
                original_gold_policy_overrides=len(strict_overrides),
                em_semantic_conflicts=conflict,new_external_API_calls=0)
    assert status['acc1']+status['acc0']+status['pending_trajectories']==status['kept_records']
    write_json('COMPLETION_AUDIT.json',status)
    write_json('pending_items.json',list(pending.values()))
    write_json('full_summary.json',summary)
    # Export the actual question/gold/answer and final judgement for every unique
    # candidate; the scripts only join explicit manual labels, never infer them.
    by_candidate=defaultdict(list)
    for row in combined:by_candidate[row['judge_item_id']].append(row)
    candidate_reviews=[]
    for jid,rows in by_candidate.items():
        labels={r['combined_acc'] for r in rows}
        assert len(labels)==1, (jid,labels)
        candidate_reviews.append({**inputs[jid],
                                  'combined_acc':next(iter(labels)),
                                  'answer_em':max(r['answer_em'] for r in rows),
                                  'represented_trajectories':len(rows),
                                  'review':reviews.get(jid)})
    with gzip.open(OUT/'candidate_reviews.jsonl.gz','wt',encoding='utf-8') as stream:
        for row in candidate_reviews:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    defect_markers=('reference_','reference_defect','reference_error','reference_type','reference_scope',
                    'reference_subject','reference_counting','reference_geographic',
                    'reference_physical','reference_frequency','reference_event',
                    'reference_one_sided','question_entity_error','question_time_direction',
                    'question_relation','nonunique','金标错误','gold_typo')
    def review_notes(review):
        if not isinstance(review,dict):return ''
        return ' '.join([str(review.get('note','')),
                         review_notes(review.get('original_review')),
                         review_notes(review.get('previous_context_review'))]).lower()
    defects=[]
    for row in candidate_reviews:
        review=row.get('review') or {}
        audit_text=review_notes(review)
        if any(m in audit_text for m in defect_markers):
            defects.append(row)
    write_json('REFERENCE_SUSPECTS.json',dict(
        description='Review-flagged reference/question/nonunique-answer concerns, NOT verified corrected gold. Primary ACC remains original-gold relative. This list is not guaranteed exhaustive.',
        candidate_count=len(defects),question_count=len({r['sample_id'] for r in defects}),
        represented_trajectories=sum(r['represented_trajectories'] for r in defects),items=defects))
    boundaries=[r for r in candidate_reviews if any(marker in review_notes(r.get('review') or {})
                for marker in ('unverified','ambigu','nonunique','reference_'))]
    write_json('REVIEW_BOUNDARIES.json',dict(
        description='Boundary/ambiguity flags, including conservative unresolved-alias or conversion decisions. These may contain model-judge errors; they are not a human-gold correction list. Primary scores are kept unchanged.',
        candidate_count=len(boundaries),question_count=len({r['sample_id'] for r in boundaries}),
        represented_trajectories=sum(r['represented_trajectories'] for r in boundaries),items=boundaries))
    with gzip.open(OUT/'final_label_ledger.jsonl.gz','wt',encoding='utf-8') as stream:
        for r in combined:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    lines=['HotpotQA全量语义ACC与token成本审计',
           '已撤下全量EM正例与小样本语义负例混合的对照表；该表不可用于总体成本对比。',
           f"原始61,754条；剔除触顶/length 2,886条；保留{len(combined):,}条。",
           f"正确{summary['acc1']:,}；错误{summary['acc0']:,}；未评审或待裁决{summary['pending']:,}。",
           '完整性约束：ACC1 + ACC0 + pending = 58,868；EM不为1不能自动判错。']
    if summary['pending'] or conflict:
        lines += ['状态：未完成，暂不发布全量二元分组表。']
    else:
        lines += ['状态：全部保留轨迹已有二元标签；语义评审为模型判断，不是人工专家金标。',
                  '正确定义：EM=1或对照原金标语义正确；错误定义：EM不为1且对照原金标语义错误。',
                  '任何root/续写调用触及8192上限或length结束即剔除，不按整条总tokens阈值剔除。',
                  '总tokens=完整独立轨迹LLM输入+输出，root计一次、重复历史输入计入、证据不二次加价。',
                  'step=Search次数（包括root）。每组题数独立，但同一道题可在多个预算中重复出现。',
                  'B100–600与FULL原始覆盖同一批7,405题；剔除触顶后，各条件保留题目可能不同。B700–2700只覆盖更小子集，不能直接当作同题总体比较。',
                  '均值和范围均以本组全部已保留轨迹计算；0题表示该分组无该类轨迹，非零成本。',
                  '金标缺陷/歧义项保留评审理由，不把置信度当作正确性。',
                  '按用户确认，疑似金标错误不修正主指标；REFERENCE_SUSPECTS.json单列疑点。',
                  '', '表一 ACC=1', '']
        lines += format_table(summary['tables'][1])
        lines += ['', '表二 ACC=0', '']+format_table(summary['tables'][0])
        (OUT/'FINAL_ACC_TABLES_ZH.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'PROGRESS_ZH.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
