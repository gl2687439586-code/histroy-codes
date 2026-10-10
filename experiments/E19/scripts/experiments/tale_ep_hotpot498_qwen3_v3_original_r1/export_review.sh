#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:?Set RUN_DIR to a completed TALE-EP run}"
OUTPUT="${OUTPUT:-${RUN_DIR%/}_review_$(date +%Y%m%d_%H%M%S).tar.gz}"
required=(
  "$RUN_DIR/COMPLETED"
  "$RUN_DIR/dataset/dataset_manifest.json"
  "$RUN_DIR/estimates/estimates.merged.jsonl"
  "$RUN_DIR/estimates/estimates.merged.jsonl.manifest.json"
  "$RUN_DIR/estimates/shards/shard_0.physical_requests.jsonl"
  "$RUN_DIR/estimates/shards/shard_1.physical_requests.jsonl"
  "$RUN_DIR/trajectories/trajectories.merged.jsonl"
  "$RUN_DIR/trajectories/trajectories.merged.jsonl.manifest.json"
  "$RUN_DIR/trajectories/shards/shard_0.physical_requests.jsonl"
  "$RUN_DIR/trajectories/shards/shard_1.physical_requests.jsonl"
  "$RUN_DIR/analysis/summary.json"
  "$RUN_DIR/analysis/AUDIT.json"
  "$RUN_DIR/analysis/REPORT_ZH.md"
)
for file in "${required[@]}"; do
  [[ -s "$file" ]] || { echo "Missing required file: $file" >&2; exit 1; }
done
python - "$RUN_DIR/analysis/AUDIT.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value.get("passed") is True, value.get("errors")
print("AUDIT PASSED")
PY
parent="$(dirname "$RUN_DIR")"
name="$(basename "$RUN_DIR")"
tar -C "$parent" -czf "$OUTPUT" "$name"
sha256sum "$OUTPUT" | tee "$OUTPUT.sha256"
du -h "$OUTPUT"
echo "Review bundle: $OUTPUT"
