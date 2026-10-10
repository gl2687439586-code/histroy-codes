#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:?Set RUN_DIR}"
EXPECTED_RECORDS="${EXPECTED_RECORDS:-500}"
PYTHON_BIN="${PYTHON_BIN:-/home/researcher/smartsearch_rebuild/envs/src-core/bin/python}"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT:-$RUN_DIR/plan_and_budget_hotpot_review_$STAMP.tar.gz}"

required=(
  "$RUN_DIR/plans/plans.jsonl"
  "$RUN_DIR/plans/manifest.json"
  "$RUN_DIR/trajectories/polynomial/records.jsonl"
  "$RUN_DIR/trajectories/polynomial/manifest.json"
  "$RUN_DIR/trajectories/cosine/records.jsonl"
  "$RUN_DIR/trajectories/cosine/manifest.json"
  "$RUN_DIR/analysis/summary.json"
  "$RUN_DIR/analysis/REPORT_ZH.md"
)
for file in "${required[@]}"; do
  [[ -s "$file" ]] || { echo "MISSING $file" >&2; exit 1; }
  echo "OK $file"
done

"$PYTHON_BIN" - "$RUN_DIR" "$EXPECTED_RECORDS" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
expected = int(sys.argv[2])
for scheduler in ("polynomial", "cosine"):
    rows = [json.loads(x) for x in (run / "trajectories" / scheduler / "records.jsonl").open(encoding="utf-8") if x.strip()]
    assert len(rows) == expected, (scheduler, len(rows), expected)
    assert len({x["sample_id"] for x in rows}) == expected
summary = json.loads((run / "analysis" / "summary.json").read_text(encoding="utf-8"))
assert summary["passed"] is True
assert summary["records_per_scheduler"] == expected
print("EXPORT INTEGRITY PASSED")
PY

tar -C "$RUN_DIR" -czf "$OUT" plans trajectories analysis tmux_logs
sha256sum "$OUT" >"$OUT.sha256"
echo "Review archive: $OUT"
echo "SHA256: $OUT.sha256"
