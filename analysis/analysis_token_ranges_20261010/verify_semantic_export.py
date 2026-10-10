"""Read-only verification/reaggregation of exported semantic ACC token tables."""
import hashlib
import json
from pathlib import Path
from collections import Counter
from combine_em_semantic import aggregate, read_gz, key

BASE=Path(__file__).resolve().parent
OUT=BASE/'full_semantic_review'


def main():
    manifest=json.loads((OUT/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    for name,info in manifest['files'].items():
        data=(BASE/name).read_bytes()
        assert len(data)==info['bytes']
        assert hashlib.sha256(data).hexdigest()==info['sha256'],name
    rows=list(read_gz(OUT/'final_label_ledger.jsonl.gz'))
    assert len(rows)==len({key(r) for r in rows})==58868
    counts=Counter(r['combined_acc'] for r in rows)
    assert set(counts)=={0,1} and sum(counts.values())==58868
    for row in rows:
        assert not row['root_cap_hit'] and not row['continuation_cap_hit']
        if row['answer_em']==1:assert row['combined_acc']==1
        else:assert row['review']['semantic_correct']==row['combined_acc']
    candidates={r['judge_item_id']:r for r in read_gz(OUT/'candidate_reviews.jsonl.gz')}
    assert sum(r['represented_trajectories'] for r in candidates.values())==58868
    for row in rows:assert candidates[row['judge_item_id']]['combined_acc']==row['combined_acc']
    result=aggregate(rows,'export_read_only_verification')
    saved=json.loads((OUT/'full_summary.json').read_text(encoding='utf-8'))
    assert json.loads(json.dumps(result['tables']))==saved['tables']
    assert result['coverage']==saved['coverage']
    assert result['pending']==0
    print(json.dumps(dict(passed=True,records=len(rows),acc1=counts[1],acc0=counts[0],
                          unique_candidates=len(candidates),table_cells=len(saved['coverage']),
                          independently_recomputed_token_statistics=True),indent=2))


if __name__=='__main__':main()
