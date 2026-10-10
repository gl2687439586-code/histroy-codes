#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?Usage: export_review.sh RUN_DIR [OUTPUT_TAR_GZ]}"
OUTPUT="${2:-${RUN_DIR%/}_review_$(date +%Y%m%d_%H%M%S).tar.gz}"
PYTHON_BIN="${PYTHON_BIN:-/home/researcher/smartsearch_rebuild/envs/src-core/bin/python}"
required=(
  "$RUN_DIR/dataset/dataset_manifest.json"
  "$RUN_DIR/PLAN.json"
  "$RUN_DIR/roots/roots.jsonl"
  "$RUN_DIR/roots/manifest.json"
  "$RUN_DIR/generations/generations.merged.jsonl"
  "$RUN_DIR/generations/generations.merged.jsonl.manifest.json"
  "$RUN_DIR/analysis/summary.json"
  "$RUN_DIR/analysis/per_record_metrics.jsonl"
  "$RUN_DIR/analysis/REPORT_ZH.md"
  "$RUN_DIR/FINAL_AUDIT.json"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "Missing required artifact: $path" >&2; exit 1; }
done
"$PYTHON_BIN" - "$RUN_DIR/FINAL_AUDIT.json" <<'PY'
import json, sys
from pathlib import Path
x = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert x["passed"], x
PY
parent="$(dirname "$RUN_DIR")"; base="$(basename "$RUN_DIR")"
tar -C "$parent" -czf "$OUTPUT" \
  "$base/dataset/dataset_manifest.json" "$base/PLAN.json" \
  "$base/roots/roots.jsonl" "$base/roots/manifest.json" \
  "$base/generations/generations.merged.jsonl" \
  "$base/generations/generations.merged.jsonl.manifest.json" \
  "$base/generations/shards" "$base/analysis" "$base/FINAL_AUDIT.json" \
  "$base/tmux_logs"
sha256sum "$OUTPUT" >"$OUTPUT.sha256"
echo "Review archive: $OUTPUT"
echo "SHA256 file:    $OUTPUT.sha256"
