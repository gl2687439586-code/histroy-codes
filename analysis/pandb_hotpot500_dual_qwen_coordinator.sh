#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?}"
: "${PYTHON_BIN:?}"
: "${RUN_DIR:?}"
: "${QUESTIONS:?}"
: "${LABELS:?}"
: "${PLANS:?}"
: "${QWEN_MODEL:?}"
: "${RETRIEVER_URL:?}"
: "${GENERATION_MAX_TOKENS:?}"
: "${MAX_SEARCHES:?}"
: "${TOP_K:?}"
: "${NUM_SHARDS:?}"

export PYTHONPATH="$REPO_ROOT/scripts/experiments${PYTHONPATH:+:$PYTHONPATH}"

run_shard_pipeline() {
  local shard="$1" port="$2" scheduler out log
  for scheduler in polynomial cosine; do
    out="$RUN_DIR/trajectories/$scheduler/shards/shard_$shard"
    log="$RUN_DIR/tmux_logs/${scheduler}_shard_${shard}.log"
    mkdir -p "$out"
    echo "[$(date -Is)] starting $scheduler shard=$shard port=$port" | tee -a "$log"
    "$PYTHON_BIN" -m plan_and_budget_hotpot_agent_v2_soft.run_trajectories \
      --questions "$QUESTIONS" \
      --plans "$PLANS" \
      --output-dir "$out" \
      --scheduler "$scheduler" \
      --model-url "http://127.0.0.1:$port" \
      --model-name Qwen3-8B \
      --expected-model-fragment Qwen3-8B \
      --retriever-url "$RETRIEVER_URL" \
      --tokenizer-path "$QWEN_MODEL" \
      --max-searches "$MAX_SEARCHES" \
      --top-k "$TOP_K" \
      --generation-max-tokens "$GENERATION_MAX_TOKENS" \
      --rollout 0 \
      --limit 500 \
      --shard-id "$shard" \
      --num-shards "$NUM_SHARDS" \
      >>"$log" 2>&1
    echo "[$(date -Is)] completed $scheduler shard=$shard" | tee -a "$log"
  done
}

run_shard_pipeline 0 9103 &
pid0=$!
run_shard_pipeline 1 9104 &
pid1=$!

set +e
wait "$pid0"; rc0=$?
wait "$pid1"; rc1=$?
set -e
if [[ "$rc0" != 0 || "$rc1" != 0 ]]; then
  echo "Shard pipeline failed: shard0=$rc0 shard1=$rc1" >&2
  exit 1
fi

"$PYTHON_BIN" - "$RUN_DIR" "$QUESTIONS" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1])
questions_path = Path(sys.argv[2])
questions = [json.loads(x) for x in questions_path.open(encoding="utf-8") if x.strip()]
order = [str(x["sample_id"]) for x in questions]
assert len(order) == 500 and len(set(order)) == 500

for scheduler in ("polynomial", "cosine"):
    base = run_dir / "trajectories" / scheduler
    by_id = {}
    shard_manifests = []
    for shard in range(2):
        shard_dir = base / "shards" / f"shard_{shard}"
        manifest = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
        shard_manifests.append(manifest)
        assert manifest["complete"] is True, (scheduler, shard, manifest)
        assert manifest["expected_records"] == 250, (scheduler, shard, manifest)
        for line in (shard_dir / "records.jsonl").open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            assert row["scheduler"] == scheduler
            assert row["rollout"] == 0
            assert sample_id not in by_id, (scheduler, sample_id)
            by_id[sample_id] = row
    assert set(by_id) == set(order), (scheduler, len(by_id), len(set(order) - set(by_id)))
    output = base / "records.jsonl"
    tmp = output.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for sample_id in order:
            f.write(json.dumps(by_id[sample_id], ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    merged_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scheduler": scheduler,
        "rollout": 0,
        "num_shards": 2,
        "expected_records": 500,
        "records": len(by_id),
        "unique_sample_ids": len(by_id),
        "complete": True,
        "records_sha256": digest,
        "shard_manifests": shard_manifests,
        "output": str(output),
    }
    (base / "manifest.json").write_text(
        json.dumps(merged_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"MERGED {scheduler}: 500 records sha256={digest}")
PY

"$PYTHON_BIN" -m plan_and_budget_hotpot_agent_v2_soft.analyze \
  --run-dir "$RUN_DIR" \
  --labels "$LABELS" \
  --plans "$PLANS" \
  --expected 500

echo "[$(date -Is)] FULL 500 DUAL-QWEN EXPERIMENT COMPLETED"
