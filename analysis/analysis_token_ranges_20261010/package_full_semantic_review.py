"""Package completed explicit semantic labels and both full token tables."""
import hashlib
import json
from pathlib import Path
import zipfile

BASE=Path(__file__).resolve().parent
OUT=BASE/'full_semantic_review'


def main():
    status=json.loads((OUT/'COMPLETION_AUDIT.json').read_text(encoding='utf-8'))
    assert status['all_kept_records_labelled']
    assert status['pending_unique_items']==status['pending_trajectories']==0
    assert not status['em_semantic_conflicts']
    assert status['acc1']+status['acc0']==58868
    queue_status=json.loads((OUT/'STATUS.json').read_text(encoding='utf-8'))
    assert queue_status['reviewed_items']==queue_status['expected_items']==17749
    files=[OUT/n for n in (
        'FINAL_ACC_TABLES_ZH.txt','EVALUATION_PROTOCOL_ZH.txt','COMPLETION_AUDIT.json',
        'full_summary.json','final_label_ledger.jsonl.gz','candidate_reviews.jsonl.gz',
        'REFERENCE_SUSPECTS.json','REVIEW_BOUNDARIES.json','queue.json','compiled_labels.json','STATUS.json',
        'adjudications_by_item.json','adjudications.json','strict_gold_overrides.json',
        'SECOND_REVIEW_QA.json','SECOND_REVIEW_QA_LATE.json')]
    files+=sorted((OUT/'labels').glob('lane*_chunk*.json'))
    files += [BASE/'full_semantic_review.py',BASE/'finalize_full_semantic_review.py',
              BASE/'combine_em_semantic.py',BASE/'verify_semantic_export.py',Path(__file__).resolve(),
              BASE/'semantic_acc_pending/AUDIT.json',
              BASE/'semantic_acc_pending/exclusion_ledger.jsonl.gz',
              BASE/'semantic_acc_pending/eligible_trajectory_map.jsonl.gz',
              BASE/'semantic_acc_pending/semantic_judge_inputs_UNSCORED.jsonl.gz',
              BASE/'codex_direct_semantic/batch_001_labelled_trajectories.json']
    manifest={
        'scope':'Full kept HotpotQA v4.3.3 trajectories, original-gold-relative model semantic review',
        'counts':status,
        'review_type':'Codex direct reading plus user-authorized parallel review; not human expert gold',
        'source_raw_included':False,
        'source_provenance':'Original raw/gold hashes are in semantic_acc_pending/AUDIT.json. Original experiment was not rerun.',
        'files':{str(p.relative_to(BASE)).replace('\\','/'):
                 dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}}
    (OUT/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    files.append(OUT/'PACKAGE_MANIFEST.json')
    archive=BASE/'hotpot_full_semantic_acc_token_tables_20261010.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as package:
        for p in files:package.write(p,str(p.relative_to(BASE)))
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert len(package.namelist())==len(files)
    print(json.dumps(dict(archive=str(archive),files=len(files),bytes=archive.stat().st_size,
                          sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
