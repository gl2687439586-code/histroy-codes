"""Read-only source validation; extracts regular archive files into a new analysis folder."""
from pathlib import Path, PurePosixPath
import hashlib, importlib.util, json, sys, tarfile, gzip

BASE=Path(__file__).resolve().parent
WORK=BASE.parent
ARCHIVE=Path('C:/Users/RESEARCH_USER/Desktop/过程结果/repeat500_20261004_review_20261005T061210Z.tar.gz')
DEST=BASE/'input'
OUT=BASE/'results'

def save(name,obj):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    members=[];seen=set()
    with tarfile.open(ARCHIVE,'r:gz') as tar:
        for m in tar.getmembers():
            p=PurePosixPath(m.name)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in m.name and ':' not in m.name
            assert m.name not in seen;seen.add(m.name)
            assert m.isfile() or m.isdir(), ('nonregular member',m.name)
            target=DEST.joinpath(*p.parts)
            assert target.resolve().is_relative_to(DEST.resolve())
            if m.isdir():target.mkdir(parents=True,exist_ok=True);continue
            raw=tar.extractfile(m).read()
            if target.exists():assert target.read_bytes()==raw
            else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            members.append({'path':m.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    save('archive_integrity.json',{'archive':str(ARCHIVE),'bytes':ARCHIVE.stat().st_size,
        'sha256':hashlib.file_digest(ARCHIVE.open('rb'),'sha256').hexdigest(),'files':members})
    spec=importlib.util.spec_from_file_location('five_raw_audit',WORK/'analysis_matched500_raw_20261004/audit_raw.py')
    audit=importlib.util.module_from_spec(spec);sys.modules[spec.name]=audit;spec.loader.exec_module(audit)
    labels={r['sample_id']:r for r in audit.rows(audit.DATA/'gold_labels.sealed.jsonl')}
    questions={r['sample_id']:r for r in audit.rows(audit.DATA/'questions.no_labels.jsonl')}
    cfg=json.loads((DEST/'BATCH_CONFIG.json').read_text(encoding='utf-8'))
    for label in ['old500','new500','repeat_1','repeat_2','repeat_3']:
        if label.startswith('repeat'):
            audit.INPUT=DEST
            ll={r['sample_id']:r for r in audit.rows(DEST/label/'dataset/gold_labels.sealed.jsonl')}
            qq={r['sample_id']:r for r in audit.rows(DEST/label/'dataset/questions.no_labels.jsonl')}
            assert set(ll)==set(labels)==set(qq)
            for sid in labels:
                assert ll[sid]['gold_answers']==labels[sid]['gold_answers'],sid
                assert qq[sid]['question']==questions[sid]['question'],sid
        else:audit.INPUT=audit.BASE/'input'
        cache=OUT/(label+'_compact.jsonl.gz')
        if cache.exists() and (OUT/(label+'_raw_audit.json')).exists():
            print('Already audited',label,flush=True);continue
        print('Auditing raw prompts, scores, evidence, costs:',label,flush=True)
        roots,compact,report=audit.audit_run(label,labels,questions)
        save(label+'_raw_audit.json',report)
        with gzip.open(cache,'wt',encoding='utf-8') as f:
            for key in sorted(compact):f.write(json.dumps(compact[key],ensure_ascii=False)+'\n')
        print(label,'verified:',len(roots),'roots,',len(compact),'trajectories; warnings=',report['warnings'],flush=True)
    print('All five raw audits completed.',flush=True)

if __name__=='__main__':main()
