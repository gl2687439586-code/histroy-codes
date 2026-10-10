#!/usr/bin/env bash
set -euo pipefail

REBUILD_ROOT="${REBUILD_ROOT:-/home/researcher/smartsearch_rebuild}"
REPO_ROOT="${REPO_ROOT:-$REBUILD_ROOT/SmartSearch}"
EXP_DIR="${EXP_DIR:-$REPO_ROOT/scripts/experiments/plan_and_budget_hotpot_agent_v2_soft}"
PYTHON_BIN="${PYTHON_BIN:-$REBUILD_ROOT/envs/src-core/bin/python}"
QUERIES="${QUERIES:-$REPO_ROOT/hotpot_random500_queries.jsonl}"
PREPARED_DIR="${PREPARED_DIR:-$REBUILD_ROOT/data/plan_and_budget_hotpot_agent_v2_soft/hotpot_random500_seed2026}"
RUN_DIR="${RUN_DIR:-$REBUILD_ROOT/outputs/plan_and_budget_hotpot_agent_v2_soft/hotpot500_qwen3_llama31_soft_v2}"
PLANNER_MODEL_PATH="${PLANNER_MODEL_PATH:-$REBUILD_ROOT/models/Llama-3.1-8B-Instruct}"
REASONER_MODEL_PATH="${REASONER_MODEL_PATH:-$REBUILD_ROOT/models/Qwen3-8B}"
PLANNER_URL="${PLANNER_URL:-http://127.0.0.1:9103}"
MODEL_URL="${MODEL_URL:-http://127.0.0.1:9104}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:6008}"
MAX_SEARCHES_PER_SUBQUESTION="${MAX_SEARCHES_PER_SUBQUESTION:-4}"
N_ROLLOUTS="${N_ROLLOUTS:-1}"
REASONER_GENERATION_MAX_TOKENS="${REASONER_GENERATION_MAX_TOKENS:-12288}"

export PYTHONPATH="$REPO_ROOT/scripts/experiments${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_DIR/tmux_logs" "$PREPARED_DIR"

prepare() {
  local gold_args=()
  if [[ -n "${GOLD_LABELS:-}" ]]; then
    gold_args=(--gold "$GOLD_LABELS")
  fi
  "$PYTHON_BIN" -m plan_and_budget_hotpot_agent_v2_soft.prepare_hotpot \
    --queries "$QUERIES" \
    "${gold_args[@]}" \
    --output-dir "$PREPARED_DIR" \
    --expected 500
}

plan() {
  local limit="${1:-500}"
  "$PYTHON_BIN" -m plan_and_budget_hotpot_agent_v2_soft.plan_questions \
    --questions "$PREPARED_DIR/questions.no_labels.jsonl" \
    --output-dir "$RUN_DIR/plans" \
    --planner-url "$PLANNER_URL" \
    --planner-model Llama-3.1-8B-Instruct \
    --expected-model-fragment Llama-3.1-8B-Instruct \
    --limit "$limit" \
    --max-attempts 3 \
    --max-tokens 2048
}

run_scheduler() {
  local scheduler="$1" limit="$2"
  "$PYTHON_BIN" -m plan_and_budget_hotpot_agent_v2_soft.run_trajectories \
    --questions "$PREPARED_DIR/questions.no_labels.jsonl" \
    --plans "$RUN_DIR/plans/plans.jsonl" \
    --output-dir "$RUN_DIR/trajectories/$scheduler" \
    --scheduler "$scheduler" \
    --model-url "$MODEL_URL" \
    --model-name Qwen3-8B \
    --expected-model-fragment Qwen3-8B \
    --retriever-url "$RETRIEVER_URL" \
    --tokenizer-path "$REASONER_MODEL_PATH" \
    --max-searches "$MAX_SEARCHES_PER_SUBQUESTION" \
    --top-k 5 \
    --generation-max-tokens "$REASONER_GENERATION_MAX_TOKENS" \
    --rollout 0 \
    --limit "$limit"
}

analyze() {
  local expected="${1:-500}"
  "$PYTHON_BIN" -m plan_and_budget_hotpot_agent_v2_soft.analyze \
    --run-dir "$RUN_DIR" \
    --labels "$PREPARED_DIR/gold_labels.sealed.jsonl" \
    --plans "$RUN_DIR/plans/plans.jsonl" \
    --expected "$expected"
}

[[ "$N_ROLLOUTS" == "1" ]] || { echo "This protocol is locked to N_ROLLOUTS=1" >&2; exit 1; }
[[ "$MAX_SEARCHES_PER_SUBQUESTION" == "4" ]] || { echo "This protocol is locked to 4 searches/subquestion" >&2; exit 1; }

case "${1:-}" in
  self-test)
    "$PYTHON_BIN" -m plan_and_budget_hotpot_agent_v2_soft.self_test
    ;;
  prepare) prepare ;;
  plan) prepare; plan 500 ;;
  gate5)
    prepare
    plan 5
    run_scheduler polynomial 5
    run_scheduler cosine 5
    analyze 5
    ;;
  full)
    prepare
    plan 500
    run_scheduler polynomial 500
    run_scheduler cosine 500
    analyze 500
    ;;
  analyze) analyze "${EXPECTED_RECORDS:-500}" ;;
  *) echo "Usage: $0 {self-test|prepare|plan|gate5|full|analyze}" >&2; exit 2 ;;
esac
