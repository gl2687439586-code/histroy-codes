#!/usr/bin/env bash
set -euo pipefail

REBUILD_ROOT=/home/researcher/smartsearch_rebuild
REPO_ROOT="$REBUILD_ROOT/SmartSearch"
EXP="$REPO_ROOT/scripts/experiments/plan_and_budget_hotpot_agent_v2_soft"
CORE_PY="$REBUILD_ROOT/envs/src-core/bin/python"
MODEL_PY="$REBUILD_ROOT/envs/src-model/bin/python"
QWEN_MODEL="$REBUILD_ROOT/models/Qwen3-8B"
PREPARED_DIR="$REBUILD_ROOT/data/plan_and_budget_hotpot_agent_v2_soft/hotpot_random500_seed2026"
RUN_DIR="$REBUILD_ROOT/outputs/plan_and_budget_hotpot_agent_v2_soft/hotpot500_qwen3_llama31_soft_v2_full_v1"

QUESTIONS="$PREPARED_DIR/questions.no_labels.jsonl"
LABELS="$PREPARED_DIR/gold_labels.sealed.jsonl"
PLANS="$RUN_DIR/plans/plans.jsonl"
RETRIEVER_URL=http://127.0.0.1:6008

CONTEXT_LENGTH=40960
MEM_FRACTION=0.80
GENERATION_MAX_TOKENS=12288
MAX_SEARCHES=4
TOP_K=5
NUM_SHARDS=2

MODEL_SESSION_G2=pandb_qwen_g2_hotpot500
MODEL_SESSION_G3=pandb_qwen_g3_hotpot500
COORDINATOR_SESSION=pandb_hotpot500_dualqwen_v1
COORDINATOR_SCRIPT="$REPO_ROOT/pandb_hotpot500_dual_qwen_coordinator.sh"
SERVICE_LOG_DIR="$RUN_DIR/tmux_logs/dual_qwen_services"
mkdir -p "$SERVICE_LOG_DIR" "$RUN_DIR/tmux_logs"
cd "$REPO_ROOT"

[[ -x "$CORE_PY" ]] || { echo "Missing Python: $CORE_PY" >&2; exit 1; }
[[ -x "$MODEL_PY" ]] || { echo "Missing model Python: $MODEL_PY" >&2; exit 1; }
[[ -s "$QWEN_MODEL/config.json" ]] || { echo "Missing Qwen model: $QWEN_MODEL" >&2; exit 1; }
[[ -s "$QUESTIONS" ]] || { echo "Missing questions: $QUESTIONS" >&2; exit 1; }
[[ -s "$LABELS" ]] || { echo "Missing labels: $LABELS" >&2; exit 1; }
[[ -s "$PLANS" ]] || { echo "Missing plans: $PLANS" >&2; exit 1; }
[[ -f "$COORDINATOR_SCRIPT" ]] || {
  echo "Missing coordinator script: $COORDINATOR_SCRIPT" >&2
  exit 1
}

"$CORE_PY" - "$QUESTIONS" "$PLANS" "$RUN_DIR/plans/manifest.json" <<'PY'
import json, sys
from pathlib import Path

questions_path, plans_path, manifest_path = map(Path, sys.argv[1:])
questions = [json.loads(x) for x in questions_path.open(encoding="utf-8") if x.strip()]
plans = [json.loads(x) for x in plans_path.open(encoding="utf-8") if x.strip()]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
question_ids = [str(x["sample_id"]) for x in questions]
plan_ids = [str(x["sample_id"]) for x in plans]
assert len(question_ids) == 500, len(question_ids)
assert len(set(question_ids)) == 500
assert len(plan_ids) == 500, len(plan_ids)
assert len(set(plan_ids)) == 500
assert set(question_ids) == set(plan_ids)
assert manifest.get("complete") is True
assert manifest.get("complete_plans") == 500
print("PLANNING INTEGRITY PASSED: 500/500")
PY

retriever_health=$(curl -fsS --max-time 10 "$RETRIEVER_URL/health") || {
  echo "Retriever unavailable on port 6008" >&2
  exit 1
}
"$CORE_PY" - "$retriever_health" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
assert payload.get("status") == "healthy", payload
assert payload.get("retrievers", {}).get("available") == 1, payload
print("FlashRAG retriever identity passed")
PY

retrieved_count=$(curl -fsS --max-time 300 \
  -X POST "$RETRIEVER_URL/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"Albert Einstein birthplace","top_n":5,"return_score":false}' \
  | "$CORE_PY" -c 'import json,sys; print(len(json.load(sys.stdin)))')
[[ "$retrieved_count" == 5 ]] || {
  echo "Retriever Top-5 probe returned $retrieved_count" >&2
  exit 1
}

stop_known_sglang_port() {
  local port="$1" pids pid cmd
  pids=$(fuser -n tcp "$port" 2>/dev/null || true)
  [[ -z "$pids" ]] && return 0
  for pid in $pids; do
    cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
    if [[ "$cmd" != *"sglang.launch_server"* ]]; then
      echo "Refusing to stop unknown process on port $port: PID=$pid CMD=$cmd" >&2
      exit 1
    fi
  done
  echo "Stopping old SGLang service on port $port: $pids"
  kill -TERM $pids 2>/dev/null || true
  for _ in $(seq 1 40); do
    [[ -z "$(fuser -n tcp "$port" 2>/dev/null || true)" ]] && return 0
    sleep 1
  done
  echo "Port $port did not become free" >&2
  exit 1
}

start_qwen() {
  local gpu="$1" port="$2" session="$3" log="$4"
  stop_known_sglang_port "$port"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "cd '$REPO_ROOT' && env \
CUDA_VISIBLE_DEVICES='$gpu' \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
'$MODEL_PY' -u -m sglang.launch_server \
--served-model-name Qwen3-8B \
--model-path '$QWEN_MODEL' \
--tp 1 --dp 1 \
--context-length '$CONTEXT_LENGTH' \
--dtype bfloat16 \
--host 127.0.0.1 \
--port '$port' \
--trust-remote-code \
--attention-backend triton \
--sampling-backend pytorch \
--disable-cuda-graph \
--disable-overlap \
--disable-radix-cache \
--chunked-prefill-size 4096 \
--max-running-requests 1 \
--mem-fraction-static '$MEM_FRACTION' \
>>'$log' 2>&1"
  echo "Started Qwen3-8B: GPU$gpu port=$port session=$session"
}

wait_qwen() {
  local port="$1" log="$2" ready=0
  for _ in $(seq 1 400); do
    if curl -fsS --max-time 5 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 3
  done
  if [[ "$ready" != 1 ]]; then
    tail -n 200 "$log" >&2 || true
    exit 1
  fi
  local model_path
  model_path=$(curl -fsS --max-time 5 "http://127.0.0.1:$port/get_model_info" \
    | "$CORE_PY" -c 'import json,sys; print(json.load(sys.stdin).get("model_path", ""))')
  [[ "$model_path" == "$QWEN_MODEL" ]] || {
    echo "Wrong model on port $port: $model_path" >&2
    exit 1
  }
}

# Port 9103 currently hosts the planner. Stop it and restart both reasoner replicas
# with identical serving settings.
start_qwen 2 9103 "$MODEL_SESSION_G2" "$SERVICE_LOG_DIR/qwen_g2.log"
start_qwen 3 9104 "$MODEL_SESSION_G3" "$SERVICE_LOG_DIR/qwen_g3.log"
wait_qwen 9103 "$SERVICE_LOG_DIR/qwen_g2.log"
wait_qwen 9104 "$SERVICE_LOG_DIR/qwen_g3.log"

echo "Both Qwen endpoints are healthy"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

if tmux has-session -t "$COORDINATOR_SESSION" 2>/dev/null; then
  echo "Coordinator session already exists: $COORDINATOR_SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$COORDINATOR_SESSION" \
  "cd '$REPO_ROOT' && env \
REBUILD_ROOT='$REBUILD_ROOT' \
REPO_ROOT='$REPO_ROOT' \
EXP_DIR='$EXP' \
PYTHON_BIN='$CORE_PY' \
RUN_DIR='$RUN_DIR' \
PREPARED_DIR='$PREPARED_DIR' \
QUESTIONS='$QUESTIONS' \
LABELS='$LABELS' \
PLANS='$PLANS' \
QWEN_MODEL='$QWEN_MODEL' \
RETRIEVER_URL='$RETRIEVER_URL' \
GENERATION_MAX_TOKENS='$GENERATION_MAX_TOKENS' \
MAX_SEARCHES='$MAX_SEARCHES' \
TOP_K='$TOP_K' \
NUM_SHARDS='$NUM_SHARDS' \
bash '$COORDINATOR_SCRIPT' >>'$RUN_DIR/tmux_logs/dual_qwen_coordinator.log' 2>&1"

echo "Started dual-Qwen 500-question inference"
echo "GPU2 endpoint: http://127.0.0.1:9103"
echo "GPU3 endpoint: http://127.0.0.1:9104"
echo "Coordinator:  $COORDINATOR_SESSION"
echo "Run directory: $RUN_DIR"
echo "Follow: tail -f '$RUN_DIR/tmux_logs/dual_qwen_coordinator.log'"
