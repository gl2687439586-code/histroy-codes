#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:?Usage: finalize_probe5.sh RUN_DIR [OUTPUT_TAR_GZ]}"
PYTHON_BIN="${PYTHON_BIN:-/home/researcher/smartsearch_rebuild/envs/src-core/bin/python}"
OUTPUT="${2:-${RUN_DIR%/}_review_$(date +%Y%m%d_%H%M%S).tar.gz}"

# Rebuild only the five-item dataset view and plan. This does not contact the
# retriever or model services and preserves already generated roots/rollouts.
env RUN_DIR="$RUN_DIR" LIMIT=5 PYTHON_BIN="$PYTHON_BIN" \
  bash "$SCRIPT_DIR/run.sh" plan

"$PYTHON_BIN" "$SCRIPT_DIR/merge_results.py" \
  --questions "$RUN_DIR/dataset/questions.no_labels.jsonl" \
  --roots "$RUN_DIR/roots/roots.jsonl" \
  --shards_dir "$RUN_DIR/generations/shards" \
  --rollouts 2 --num_shards 2 \
  --output "$RUN_DIR/generations/generations.merged.jsonl"

env RUN_DIR="$RUN_DIR" LIMIT=5 PYTHON_BIN="$PYTHON_BIN" \
  bash "$SCRIPT_DIR/run.sh" analyze
env RUN_DIR="$RUN_DIR" LIMIT=5 PYTHON_BIN="$PYTHON_BIN" \
  bash "$SCRIPT_DIR/run.sh" audit

"$PYTHON_BIN" "$SCRIPT_DIR/gate_audit_report.py" --run_dir "$RUN_DIR" \
  | tee "$RUN_DIR/analysis/GATE_INITIAL_AUDIT.console.json"

PYTHON_BIN="$PYTHON_BIN" bash "$SCRIPT_DIR/export_review.sh" "$RUN_DIR" "$OUTPUT"
echo "FIVE-QUESTION GATE FINALIZATION PASSED"
