"""Blinded manual semantic-review queue. This utility never predicts a label."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'full_semantic_review'
QUEUE = OUT / 'queue.json'
REASONS = {
    'E': '答案与参考答案语义等价（包括别名、简称、数字写法或非矛盾的释义）。',
    'W': '回答的实体、数值、关系或肯否结论与参考答案不符。',
    'N': '未回答所问事实，表示未知、证据不足或无法确定。',
    'I': '只回答部分关键内容，或答案在所问命题上自相矛盾。',
    'A': '金标、问题含义或可接受语义粒度存在疑点，需主评审复核。',
}


def read_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(l) for l in stream if l.strip()]


def prepare():
    ledger = read_gz(BASE/'em_plus_semantic/combined_label_ledger.jsonl.gz')
    pending = {r['judge_item_id'] for r in ledger if r['combined_acc'] is None}
    inputs = read_gz(BASE/'semantic_acc_pending/semantic_judge_inputs_UNSCORED.jsonl.gz')
    groups = defaultdict(list)
    for item in inputs:
        if item['judge_item_id'] in pending:
            groups[item['sample_id']].append(item)
    ordered = sorted(groups, key=lambda s: hashlib.sha256(('full-semantic-20261010:'+s).encode()).hexdigest())
    lanes = [[], [], []]
    counts = [0, 0, 0]
    uid = 0
    for sid in ordered:
        lane = min(range(3), key=lambda i: counts[i])
        items = sorted(groups[sid], key=lambda r: r['judge_item_id'])
        q = {k: items[0][k] for k in ('sample_id', 'question', 'reference_answer')}
        q['answers'] = []
        for item in items:
            uid += 1
            q['answers'].append(dict(i=uid, judge_item_id=item['judge_item_id'], answer=item['candidate_answer']))
        lanes[lane].append(q)
        counts[lane] += len(items)
    chunks = []
    for lane, questions in enumerate(lanes):
        chunk, size, number = [], 0, 0
        for q in questions:
            if chunk and size+len(q['answers']) > 120:
                chunks.append(dict(lane=lane, chunk=number, questions=chunk))
                chunk, size, number = [], 0, number+1
            chunk.append(q)
            size += len(q['answers'])
        if chunk:
            chunks.append(dict(lane=lane, chunk=number, questions=chunk))
    doc = dict(queue_version=1, items=uid, questions=len(groups), lane_items=counts, chunks=chunks)
    OUT.mkdir(exist_ok=True)
    (OUT/'labels').mkdir(exist_ok=True)
    if QUEUE.exists():
        assert json.loads(QUEUE.read_text(encoding='utf-8')) == doc
    else:
        QUEUE.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in doc.items() if k!='chunks'}))
    print('chunks per lane:',dict(Counter(c['lane'] for c in chunks)))


def get_chunk(lane, number):
    queue = json.loads(QUEUE.read_text(encoding='utf-8'))
    return next(c for c in queue['chunks'] if c['lane']==lane and c['chunk']==number)


def show(args):
    chunk = get_chunk(args.lane, args.chunk)
    print('LANE',args.lane,'CHUNK',args.chunk)
    for q in chunk['questions']:
        print('\nQ:',q['question'],'\nGOLD:',q['reference_answer'])
        for a in q['answers']:
            print(str(a['i'])+' '+json.dumps(a['answer'], ensure_ascii=False))


def context(args):
    queue = json.loads(QUEUE.read_text(encoding='utf-8'))
    q = next(q for c in queue['chunks'] for q in c['questions'] if any(a['i']==args.item for a in q['answers']))
    gold = json.loads((BASE.parent/'work/hotpot_signal_search_v2/hotpot_dev_distractor_v1.json').read_text(encoding='utf-8'))
    item = next(r for r in gold if r['_id']==q['sample_id'])
    print(json.dumps(q, ensure_ascii=False, indent=2))
    pages = dict(item['context'])
    for title, index in item['supporting_facts']:
        print(title,'[',index,']:',pages[title][index])


def progress(args):
    queue = json.loads(QUEUE.read_text(encoding='utf-8'))
    chunks = {(c['lane'], c['chunk']):c for c in queue['chunks']}
    expected = {a['i']: (c['lane'],c['chunk'],a['judge_item_id']) for c in queue['chunks'] for q in c['questions'] for a in q['answers']}
    seen = {}; counts = Counter(); done = Counter()
    for path in sorted((OUT/'labels').glob('lane*_chunk*.json')):
        doc = json.loads(path.read_text(encoding='utf-8'))
        chunk = chunks[doc['lane'], doc['chunk']]
        required = {a['i'] for q in chunk['questions'] for a in q['answers']}
        actual = [x[0] for x in doc['labels']]
        assert len(actual)==len(set(actual)) and set(actual)==required, str(path)
        for label in doc['labels']:
            i, decision, reason = label[:3]
            assert i not in seen and decision in (0,1,None) and reason in REASONS, (path,label)
            assert (decision==1 and reason=='E') or (decision==0 and reason in ('W','N','I')) or (decision is None and reason=='A')
            seen[i] = dict(judge_item_id=expected[i][2], semantic_correct=decision, reason_code=reason,
                           reason_zh=REASONS[reason], note=label[3] if len(label)>3 else '', reviewer=doc['reviewer'],
                           source=str(path), queue_item_id=i)
            counts[(doc['lane'],str(decision))] += 1
        done[doc['lane']] += 1
    status = dict(expected_items=queue['items'], reviewed_items=len(seen),
                  pending_items=queue['items']-len(seen), chunks_complete=dict(done),
                  labels={str(k):v for k,v in counts.items()},
                  needs_adjudication=sum(v['semantic_correct'] is None for v in seen.values()),
                  complete=len(seen)==queue['items'])
    print(json.dumps(status, ensure_ascii=False, indent=2))
    (OUT/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'compiled_labels.json').write_text(json.dumps(list(seen.values()),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','show','context','progress'])
    p.add_argument('--lane',type=int,default=0)
    p.add_argument('--chunk',type=int,default=0)
    p.add_argument('--item',type=int)
    args=p.parse_args()
    if args.mode=='prepare':prepare()
    else:globals()[args.mode](args)
