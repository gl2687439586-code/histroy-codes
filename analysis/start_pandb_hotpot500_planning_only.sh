#!/usr/bin/env bash
set -euo pipefail

REBUILD_ROOT=/home/researcher/smartsearch_rebuild
REPO_ROOT="$REBUILD_ROOT/SmartSearch"
EXP="$REPO_ROOT/scripts/experiments/plan_and_budget_hotpot_agent_v2_soft"
CORE_PY="$REBUILD_ROOT/envs/src-core/bin/python"
MODEL_PY="$REBUILD_ROOT/envs/src-model/bin/python"
PLANNER_MODEL="$REBUILD_ROOT/models/Llama-3.1-8B-Instruct"

QUERIES="$REPO_ROOT/hotpot_random500_queries.jsonl"
GOLD_LABELS="$REBUILD_ROOT/data/hotpotqa/prepared/random500_seed2026/gold_labels.sealed.jsonl"
PREPARED_DIR="$REBUILD_ROOT/data/plan_and_budget_hotpot_agent_v2_soft/hotpot_random500_seed2026"
RUN_DIR="$REBUILD_ROOT/outputs/plan_and_budget_hotpot_agent_v2_soft/hotpot500_qwen3_llama31_soft_v2_full_v1"
SERVICE_RUN_DIR="$REBUILD_ROOT/outputs/plan_and_budget_hotpot_agent_v2_soft/planner_only_g2_v1"

PLANNER_GPU=2
PLANNER_PORT=9103
PLANNER_CONTEXT_LENGTH=16384
PLANNER_MEM_FRACTION=0.68
MODEL_SESSION=pandb_planner_g2_hotpot500
PLAN_SESSION=pandb_hotpot500_plan_only_v1

mkdir -p "$SERVICE_RUN_DIR/tmux_logs" "$RUN_DIR/tmux_logs"
cd "$REPO_ROOT"

[[ -x "$CORE_PY" ]] || { echo "Missing Python: $CORE_PY" >&2; exit 1; }
[[ -x "$MODEL_PY" ]] || { echo "Missing model Python: $MODEL_PY" >&2; exit 1; }
[[ -s "$PLANNER_MODEL/config.json" ]] || { echo "Missing planner model: $PLANNER_MODEL" >&2; exit 1; }
[[ -s "$QUERIES" ]] || { echo "Missing queries: $QUERIES" >&2; exit 1; }
[[ -s "$GOLD_LABELS" ]] || { echo "Missing labels: $GOLD_LABELS" >&2; exit 1; }
[[ -f "$EXP/orchestrate.sh" ]] || { echo "Missing experiment: $EXP" >&2; exit 1; }

query_count=$("$CORE_PY" - "$QUERIES" <<'PY'
import json, sys
from pathlib import Path

ids = []
for line in Path(sys.argv[1]).open(encoding="utf-8"):
    if not line.strip():
        continue
    row = json.loads(line)
    sample_id = str(
        row.get(
            "query_id",
            row.get("question_id", row.get("sample_id", row.get("_id", row.get("id", "")))),
        )
    )
    if not sample_id:
        raise SystemExit("query row missing ID")
    ids.append(sample_id)
if len(ids) != len(set(ids)):
    raise SystemExit("query file contains duplicate IDs")
print(len(ids))
PY
)
[[ "$query_count" == 500 ]] || {
  echo "Expected 500 unique queries, found $query_count" >&2
  exit 1
}

model_path_on_port() {
  curl -fsS --max-time 5 "http://127.0.0.1:$PLANNER_PORT/get_model_info" 2>/dev/null \
    | "$CORE_PY" -c 'import json,sys; print(json.load(sys.stdin).get("model_path", ""))' \
    2>/dev/null || true
}

existing_model=$(model_path_on_port)
if [[ "$existing_model" == "$PLANNER_MODEL" ]]; then
  echo "Reusing verified Llama planner on GPU2 port $PLANNER_PORT"
else
  pids=$(fuser -n tcp "$PLANNER_PORT" 2>/dev/null || true)
  for pid in $pids; do
    cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
    if [[ "$cmd" != *"sglang.launch_server"* ]]; then
      echo "Refusing to stop unknown process on port $PLANNER_PORT: PID=$pid CMD=$cmd" >&2
      exit 1
    fi
    echo "Stopping old SGLang service on port $PLANNER_PORT: PID=$pid"
    kill -TERM "$pid" 2>/dev/null || true
  done

  for _ in $(seq 1 30); do
    [[ -z "$(fuser -n tcp "$PLANNER_PORT" 2>/dev/null || true)" ]] && break
    sleep 1
  done
  [[ -z "$(fuser -n tcp "$PLANNER_PORT" 2>/dev/null || true)" ]] || {
    echo "Port $PLANNER_PORT did not become free" >&2
    exit 1
  }

  tmux kill-session -t "$MODEL_SESSION" 2>/dev/null || true
  tmux new-session -d -s "$MODEL_SESSION" \
    "cd '$REPO_ROOT' && env \
CUDA_VISIBLE_DEVICES='$PLANNER_GPU' \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
'$MODEL_PY' -u -m sglang.launch_server \
--served-model-name Llama-3.1-8B-Instruct \
--model-path '$PLANNER_MODEL' \
--tp 1 --dp 1 \
--context-length '$PLANNER_CONTEXT_LENGTH' \
--dtype bfloat16 \
--host 127.0.0.1 \
--port '$PLANNER_PORT' \
--trust-remote-code \
--attention-backend triton \
--sampling-backend pytorch \
--disable-cuda-graph \
--disable-overlap \
--disable-radix-cache \
--chunked-prefill-size 4096 \
--max-running-requests 1 \
--mem-fraction-static '$PLANNER_MEM_FRACTION' \
>>'$SERVICE_RUN_DIR/tmux_logs/planner_g2.log' 2>&1"

  echo "Started Llama planner on GPU2; waiting for health endpoint"
  ready=0
  for _ in $(seq 1 400); do
    if curl -fsS --max-time 5 "http://127.0.0.1:$PLANNER_PORT/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 3
  done
  if [[ "$ready" != 1 ]]; then
    tail -n 200 "$SERVICE_RUN_DIR/tmux_logs/planner_g2.log" >&2 || true
    exit 1
  fi

  existing_model=$(model_path_on_port)
  [[ "$existing_model" == "$PLANNER_MODEL" ]] || {
    echo "Planner identity mismatch: $existing_model" >&2
    exit 1
  }
fi

if tmux has-session -t "$PLAN_SESSION" 2>/dev/null; then
  echo "Planning session already exists: $PLAN_SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$PLAN_SESSION" \
  "cd '$REPO_ROOT' && env \
REBUILD_ROOT='$REBUILD_ROOT' \
REPO_ROOT='$REPO_ROOT' \
EXP_DIR='$EXP' \
PYTHON_BIN='$CORE_PY' \
RUN_DIR='$RUN_DIR' \
QUERIES='$QUERIES' \
GOLD_LABELS='$GOLD_LABELS' \
PREPARED_DIR='$PREPARED_DIR' \
PLANNER_URL='http://127.0.0.1:$PLANNER_PORT' \
bash '$EXP/orchestrate.sh' plan \
>>'$RUN_DIR/tmux_logs/planning.log' 2>&1"

echo "Started 500-question planning only"
echo "Planner service: $MODEL_SESSION (GPU2, port $PLANNER_PORT)"
echo "Planning task:   $PLAN_SESSION"
echo "Run directory:   $RUN_DIR"
echo "Follow: tail -f '$RUN_DIR/tmux_logs/planning.log'"
echo "Count:  watch -n 20 \"wc -l '$RUN_DIR/plans/plans.jsonl' 2>/dev/null || true\""
