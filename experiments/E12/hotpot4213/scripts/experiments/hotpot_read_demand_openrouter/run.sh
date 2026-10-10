#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/researcher/smartsearch_rebuild/SmartSearch
PYTHON_BIN=${PYTHON_BIN:-/home/researcher/smartsearch_rebuild/envs/src-core/bin/python}
EXP="$ROOT/scripts/experiments/hotpot_read_demand_openrouter"
INPUT=${INPUT:-$ROOT/hotpot_stage_export_4213_queries.jsonl}
RUN_DIR=${RUN_DIR:-/home/researcher/smartsearch_rebuild/outputs/query_prior/hotpot_stage4213_openrouter_gpt56sol_read_v1}
EXPECTED_SAMPLES=${EXPECTED_SAMPLES:-4213}

MODE=${1:-run}

common=(
  --input "$INPUT"
  --output-dir "$RUN_DIR"
  --expected-samples "$EXPECTED_SAMPLES"
  --model openai/gpt-5.6-sol
  --workers "${WORKERS:-1}"
  --timeout "${TIMEOUT:-180}"
  --max-attempts "${MAX_ATTEMPTS:-1}"
  --reasoning-effort "${REASONING_EFFORT:-medium}"
  --max-tokens "${MAX_TOKENS:-2048}"
  --seed "${SEED:-2026}"
)

case "$MODE" in
  self-test)
    exec "$PYTHON_BIN" "$EXP/self_test.py"
    ;;
  plan)
    exec "$PYTHON_BIN" "$EXP/score_hotpot_read_openrouter.py" "${common[@]}" --plan
    ;;
  preflight)
    exec "$PYTHON_BIN" "$EXP/score_hotpot_read_openrouter.py" "${common[@]}" --preflight-only
    ;;
  run)
    exec "$PYTHON_BIN" "$EXP/score_hotpot_read_openrouter.py" "${common[@]}"
    ;;
  validate)
    exec "$PYTHON_BIN" "$EXP/score_hotpot_read_openrouter.py" "${common[@]}" --validate-only
    ;;
  analyze)
    exec "$PYTHON_BIN" "$EXP/analyze_read_scores.py" \
      --input "$RUN_DIR/results/read_scores_combined.jsonl" \
      --output-dir "$RUN_DIR/analysis"
    ;;
  audit)
    exec "$PYTHON_BIN" "$EXP/audit_final.py" \
      --input "$INPUT" \
      --run-dir "$RUN_DIR" \
      --expected-samples "$EXPECTED_SAMPLES" \
      --expected-model openai/gpt-5.6-sol
    ;;
  run-until-complete)
    recovery_rounds=${RECOVERY_ROUNDS:-40}
    recovery_sleep=${RECOVERY_SLEEP_SECONDS:-30}
    mkdir -p "$RUN_DIR"
    for ((round=1; round<=recovery_rounds; round++)); do
      echo "[$(date -Is)] recovery round $round/$recovery_rounds"
      set +e
      "$PYTHON_BIN" "$EXP/score_hotpot_read_openrouter.py" "${common[@]}"
      run_rc=$?
      set -e

      progress=$(
        "$PYTHON_BIN" - "$RUN_DIR/manifest.json" "$EXPECTED_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file():
    print(f"0 {expected}")
else:
    data = json.loads(path.read_text(encoding="utf-8"))
    completed = int(data.get("completed_logical_ratings", 0))
    missing = len(data.get("missing_tasks", []))
    print(f"{completed} {missing}")
PY
      )
      read -r completed missing <<<"$progress"
      echo "[$(date -Is)] run_rc=$run_rc completed=$completed/$EXPECTED_SAMPLES missing=$missing"

      if [[ "$completed" -eq "$EXPECTED_SAMPLES" && "$missing" -eq 0 ]]; then
        "$PYTHON_BIN" "$EXP/analyze_read_scores.py" \
          --input "$RUN_DIR/results/read_scores_combined.jsonl" \
          --output-dir "$RUN_DIR/analysis"
        "$PYTHON_BIN" "$EXP/audit_final.py" \
          --input "$INPUT" \
          --run-dir "$RUN_DIR" \
          --expected-samples "$EXPECTED_SAMPLES" \
          --expected-model openai/gpt-5.6-sol
        echo "[$(date -Is)] all $EXPECTED_SAMPLES READ ratings completed"
        exit 0
      fi

      if [[ "$round" -lt "$recovery_rounds" ]]; then
        echo "[$(date -Is)] sleeping ${recovery_sleep}s before retrying only missing tasks"
        sleep "$recovery_sleep"
      fi
    done
    "$PYTHON_BIN" "$EXP/audit_final.py" \
      --input "$INPUT" \
      --run-dir "$RUN_DIR" \
      --expected-samples "$EXPECTED_SAMPLES" \
      --expected-model openai/gpt-5.6-sol || true
    echo "Stopped after $recovery_rounds rounds before reaching $EXPECTED_SAMPLES completed ratings" >&2
    exit 1
    ;;
  status)
    "$PYTHON_BIN" - "$RUN_DIR" "$EXPECTED_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
expected = int(sys.argv[2])
manifest = run / "manifest.json"
if manifest.is_file():
    data = json.loads(manifest.read_text(encoding="utf-8"))
    completed = int(data.get("completed_logical_ratings", 0))
    missing = len(data.get("missing_tasks", []))
    physical = int(data.get("generation_requests_logged_across_runs", 0))
    print(json.dumps({
        "run_dir": str(run),
        "completed": completed,
        "expected": expected,
        "missing": missing,
        "physical_requests": physical,
        "completion_rate": completed / expected,
    }, ensure_ascii=False, indent=2))
else:
    print(json.dumps({"run_dir": str(run), "completed": 0, "expected": expected, "missing": expected}, indent=2))
PY
    ;;
  *)
    echo "Usage: $0 {self-test|plan|preflight|run|run-until-complete|validate|analyze|audit|status}" >&2
    exit 2
    ;;
esac
