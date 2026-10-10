"""Prepare blinded direct-Codex review batches and aggregate explicit human-readable decisions.

This program NEVER classifies answers. Only reviewed labels in a separate JSON file
are joined to trajectories. Unreviewed items remain unlabelled.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

BASE=Path(__file__).resolve().parent
PENDING=BASE/'semantic_acc_pending'
OUT=BASE/'codex_direct_semantic'


def read_gz(name):
    with gzip.open(PENDING/name,'rt',encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(name, value):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def prepare(args):
    inputs={r['judge_item_id']:r for r in read_gz('semantic_judge_inputs_UNSCORED.jsonl.gz')}
    mapping=read_gz('eligible_trajectory_map.jsonl.gz')
    by_question=defaultdict(set)
    for row in mapping:by_question[row['sample_id']].add(row['judge_item_id'])
    order=sorted(by_question,key=lambda sid:hashlib.sha256(('semantic-direct-20261010:'+sid).encode()).hexdigest())
    chosen=order[args.start:args.start+args.questions]
    batch=[]
    for i,sid in enumerate(chosen,1):
        candidates=[]
        for j,item_id in enumerate(sorted(by_question[sid]),1):
            item=inputs[item_id]
            candidates.append({'alias':f'Q{i:03d}A{j:02d}','judge_item_id':item_id,'candidate_answer':item['candidate_answer']})
        base=inputs[next(iter(by_question[sid]))]
        batch.append({'question_alias':f'Q{i:03d}','sample_id':sid,'question':base['question'],
                      'reference_answer':base['reference_answer'],'candidates':candidates})
    doc={'batch':args.batch,'selection':'SHA256(question ID + fixed seed), not selected by outcomes, budget or costs',
         'start':args.start,'question_count':len(batch),'questions':batch}
    file=OUT/(args.batch+'_inputs.json')
    if file.exists():assert json.loads(file.read_text(encoding='utf-8'))==doc
    else:save(file.name,doc)
    print(json.dumps({'batch':args.batch,'questions':len(batch),'candidate_items':sum(len(q['candidates']) for q in batch)},ensure_ascii=False))


def show(args):
    batch=json.loads((OUT/(args.batch+'_inputs.json')).read_text(encoding='utf-8'))
    for q in batch['questions'][args.start:args.start+args.questions]:
        print(f"\n{q['question_alias']} Q: {q['question']}\nREFERENCE: {q['reference_answer']}")
        for a in q['candidates']:print(a['alias']+': '+json.dumps(a['candidate_answer'],ensure_ascii=False))


def summarize(args):
    batch=json.loads((OUT/(args.batch+'_inputs.json')).read_text(encoding='utf-8'))
    labels=json.loads((OUT/(args.batch+'_labels.json')).read_text(encoding='utf-8'))
    candidates={a['alias']:a for q in batch['questions'] for a in q['candidates']}
    assert set(labels['decisions'])==set(candidates)
    by_item={}
    for alias,decision in labels['decisions'].items():
        assert decision['semantic_correct'] in [0,1,None] and decision['reason_zh']
        item_id=candidates[alias]['judge_item_id']
        if item_id in by_item:assert by_item[item_id]['semantic_correct']==decision['semantic_correct']
        by_item[item_id]={**decision,'alias':alias}
    mapping=read_gz('eligible_trajectory_map.jsonl.gz')
    selected_ids={q['sample_id'] for q in batch['questions']}
    groups=defaultdict(list);applied=[]
    for row in mapping:
        if row['sample_id'] not in selected_ids:continue
        decision=by_item[row['judge_item_id']]
        if decision['semantic_correct'] is not None:
            group=(row['budget'],row['search_calls'],decision['semantic_correct'])
            groups[group].append(row['total_tokens'])
        applied.append({**row,**decision})
    out=[]
    for (budget,step,correct),costs in sorted(groups.items(),key=lambda x:(float('inf') if x[0][0]=='FULL' else int(x[0][0]),x[0][1],-x[0][2])):
        out.append({'budget':budget,'search_steps':step,'semantic_correct':correct,'n':len(costs),
                    'mean_total_tokens':statistics.mean(costs),'median_total_tokens':statistics.median(costs),
                    'min_total_tokens':min(costs),'max_total_tokens':max(costs)})
    summary={'status':'PARTIAL_DIRECT_REVIEW_NOT_FULL7405','question_count':len(selected_ids),
             'reviewed_unique_candidates':len(by_item),
             'binary_labelled_unique_candidates':sum(v['semantic_correct'] is not None for v in by_item.values()),
             'needs_adjudication_unique_candidates':sum(v['semantic_correct'] is None for v in by_item.values()),
             'unreviewed_unique_candidates_total':len({r['judge_item_id'] for r in mapping})-len(by_item),
             'trajectory_count':len(applied),'total_eligible_trajectories':len(mapping),
             'correct_trajectories':sum(r['semantic_correct']==1 for r in applied),
             'incorrect_trajectories':sum(r['semantic_correct']==0 for r in applied),
             'needs_adjudication_trajectories':sum(r['semantic_correct'] is None for r in applied),
             'groups':out,'new_external_model_calls':0,
             'inputs_sha256':hashlib.sha256((OUT/(args.batch+'_inputs.json')).read_bytes()).hexdigest(),
             'labels_sha256':hashlib.sha256((OUT/(args.batch+'_labels.json')).read_bytes()).hexdigest(),
             'judge':'Current Codex assistant read question, reference and complete candidate text; deterministic script only aggregates explicit labels.'}
    assert sum(g['n'] for g in out)+summary['needs_adjudication_trajectories']==len(applied)
    assert len({(r['sample_id'],r['condition_id']) for r in applied})==len(applied)
    save(args.batch+'_summary.json',summary)
    save(args.batch+'_labelled_trajectories.json',applied)
    report=['Codex直接语义评审：阶段性结果，不是全量7405题结果。',
            '逐条读取问题、gold、候选答案；不使用字符串包含/EM代替语义标签。',
            f"本批{len(selected_ids)}题；{len(by_item)}个不同候选；覆盖{len(applied)}条未触顶轨迹。",
            f"明确二元判定{summary['binary_labelled_unique_candidates']}个候选；待裁决{summary['needs_adjudication_unique_candidates']}个候选。",
            f"覆盖轨迹：正确{summary['correct_trajectories']}，错误{summary['incorrect_trajectories']}，待裁决{summary['needs_adjudication_trajectories']}。待裁决不计作错误，不进入下面的二元分组。",
            '先前已排除任何root或续写调用触及长度上限的轨迹；这里的成本仍为完整输入+输出。',
            'step仅指实际Search次数，包含首次Search。预算是每次搜索后的证据暴露档位，不是整条轨迹总token开销。',
            '审阅输入隐藏预算/搜索步数/成本，判定后才回填这些字段。',
            '相同问题下完全相同答案复用一次标签，不把未读过的不同答案自动判定。',
            '模型评审仍可能出错；保留理由与原答案供复核，不能称为人工专家金标。',
            '', '预算 | Search步数 | 语义正确 | 记录数 | 平均总tokens | 中位数 | 最小–最大']
    for r in out:
        report.append(f"{r['budget']} | {r['search_steps']} | {'正确' if r['semantic_correct'] else '错误'} | {r['n']} | {r['mean_total_tokens']:,.1f} | {r['median_total_tokens']:,.1f} | {r['min_total_tokens']:,}–{r['max_total_tokens']:,}")
    (OUT/(args.batch+'_REPORT_ZH.txt')).write_text('\n'.join(report)+'\n',encoding='utf-8')
    decisions_report=['Codex直接语义评审逐项记录（首批，非全量）',
                      '下列标签由当前对话助手逐条阅读作出；脚本只排版和聚合，不负责判断。',
                      '标准答案或粒度存在疑点时标为待裁决，不用词法评分补齐。',
                      '评审者仍可能误判，保留问题、原答案、理由用于复核。', '']
    for q in batch['questions']:
        decisions_report.extend([q['question_alias']+' | '+q['sample_id'],
                                 '问题：'+q['question'],'参考答案：'+q['reference_answer']])
        for a in q['candidates']:
            decision=labels['decisions'][a['alias']]
            verdict={1:'正确',0:'错误',None:'待裁决'}[decision['semantic_correct']]
            decisions_report.extend([a['alias']+'：'+a['candidate_answer'],
                                     '判断：'+verdict+'；理由：'+decision['reason_zh']])
        decisions_report.append('')
    (OUT/(args.batch+'_DECISIONS_ZH.txt')).write_text('\n'.join(decisions_report)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='groups'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','show','summarize'])
    parser.add_argument('--batch',default='batch_001')
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--questions',type=int,default=50)
    args=parser.parse_args()
    globals()[args.mode](args)
