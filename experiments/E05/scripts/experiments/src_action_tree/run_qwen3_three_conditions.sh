#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/home/researcher/models/Qwen3-8B}"
MODEL_URL="${MODEL_URL:-http://127.0.0.1:9001}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:6008}"
SANDBOX_URL="${SANDBOX_URL:-http://127.0.0.1:6006}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-/home/researcher/smartsearch_workspace/outputs/full_bamboogle_100w_timeout120/bamboogle_2026_06_25_15_04_SmartSearch-3B_100w_timeout120}"
INTERMEDIATE_DATA="${INTERMEDIATE_DATA:-${SOURCE_RUN_DIR}/intermediate_data.json}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/researcher/smartsearch_workspace/outputs/src_action_tree/qwen3_only_three_conditions_v6}"
PYTHON_BIN="${PYTHON_BIN:-python}"

LIMIT="${LIMIT:-0}"
SAMPLE_IDS="${SAMPLE_IDS:-}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_STEPS="${MAX_STEPS:-6}"
MIN_HORIZON="${MIN_HORIZON:-3}"
MAX_HORIZON="${MAX_HORIZON:-6}"
SEARCH_MAX_NEW_TOKENS="${SEARCH_MAX_NEW_TOKENS:-512}"
READ_MAX_NEW_TOKENS="${READ_MAX_NEW_TOKENS:-768}"
ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS:-2048}"
ANSWER_REPAIR_MAX_NEW_TOKENS="${ANSWER_REPAIR_MAX_NEW_TOKENS:-96}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"

BLANK_DIR="${OUTPUT_BASE}/01_blank"
ROOT_DIR="${OUTPUT_BASE}/roots"
ROOT_FILE="${ROOT_DIR}/qwen3_first_search_checkpoints.jsonl"
TREE_DIR="${OUTPUT_BASE}/02_src_tree"
NEUTRAL_DIR="${OUTPUT_BASE}/03_neutral"
ANALYSIS_DIR="${OUTPUT_BASE}/joint_analysis"
PRIMARY_TREE_FILE="${ANALYSIS_DIR}/terminal_trajectories_natural_roots.jsonl"

guard_qwen_only() {
  local normalized="${MODEL_PATH,,}"
  [[ "${normalized}" == */qwen3-8b ]] || {
    echo "MODEL_PATH must end in Qwen3-8B: ${MODEL_PATH}" >&2
    exit 2
  }
  [[ "${normalized}" != *smartsearch-3b* ]] || {
    echo "SmartSearch-3B is forbidden in the Qwen3-only protocol." >&2
    exit 2
  }
  [[ "${MODEL_URL}" != *:9000* ]] || {
    echo "Port 9000 is forbidden; use the Qwen3 server on 9001." >&2
    exit 2
  }
}

natural_args=(
  --intermediate_data "${INTERMEDIATE_DATA}"
  --model_url "${MODEL_URL}"
  --model_path "${MODEL_PATH}"
  --retriever_url "${RETRIEVER_URL}"
  --sandbox_url "${SANDBOX_URL}"
  --retriever_mode direct
  --temperature "${TEMPERATURE}"
  --max_steps "${MAX_STEPS}"
  --max_new_tokens "${ANSWER_MAX_NEW_TOKENS}"
  --answer_max_new_tokens "${ANSWER_MAX_NEW_TOKENS}"
  --context_length "${CONTEXT_LENGTH}"
  --limit "${LIMIT}"
  --sample_ids "${SAMPLE_IDS}"
  --resume
  --retry_failures
)

run_blank() {
  mkdir -p "${BLANK_DIR}"
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/run_qwen3_natural.py" \
    --condition blank --output_dir "${BLANK_DIR}" "${natural_args[@]}"
}

prepare_roots() {
  mkdir -p "${ROOT_DIR}"
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/prepare_qwen3_roots.py" \
    --blank "${BLANK_DIR}/trajectories.jsonl" \
    --output "${ROOT_FILE}" \
    --model_url "${MODEL_URL}" \
    --model_path "${MODEL_PATH}" \
    --retriever_url "${RETRIEVER_URL}" \
    --sandbox_url "${SANDBOX_URL}" \
    --retriever_mode direct \
    --top_k 5 \
    --temperature "${TEMPERATURE}" \
    --max_new_tokens "${SEARCH_MAX_NEW_TOKENS}" \
    --resume
}

run_tree() {
  mkdir -p "${TREE_DIR}"
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/run_action_tree.py" \
    --intermediate_data "${ROOT_FILE}" \
    --output_dir "${TREE_DIR}" \
    --model_url "${MODEL_URL}" \
    --model_path "${MODEL_PATH}" \
    --retriever_url "${RETRIEVER_URL}" \
    --sandbox_url "${SANDBOX_URL}" \
    --retriever_mode direct \
    --checkpoint_format qwen3 \
    --prompt_protocol qwen3_matched \
    --horizon_mode original_searches \
    --min_horizon "${MIN_HORIZON}" \
    --max_horizon "${MAX_HORIZON}" \
    --fixed_horizon "${MIN_HORIZON}" \
    --zero_search_horizon "${MIN_HORIZON}" \
    --limit "${LIMIT}" \
    --sample_ids "${SAMPLE_IDS}" \
    --temperature "${TEMPERATURE}" \
    --n_rollouts 1 \
    --max_workers 1 \
    --action_validation flexible \
    --action_attempts 1 \
    --search_max_new_tokens "${SEARCH_MAX_NEW_TOKENS}" \
    --read_max_new_tokens "${READ_MAX_NEW_TOKENS}" \
    --answer_max_new_tokens "${ANSWER_MAX_NEW_TOKENS}" \
    --answer_repair_max_new_tokens "${ANSWER_REPAIR_MAX_NEW_TOKENS}" \
    --context_length "${CONTEXT_LENGTH}" \
    --resume \
    --skip_previous_failures
}

run_neutral() {
  mkdir -p "${NEUTRAL_DIR}"
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/run_qwen3_natural.py" \
    --condition neutral \
    --roots "${ROOT_FILE}" \
    --output_dir "${NEUTRAL_DIR}" \
    "${natural_args[@]}"
}

analyze_all() {
  mkdir -p "${ANALYSIS_DIR}"
  if [[ -s "${BLANK_DIR}/failures.jsonl" || -s "${TREE_DIR}/failures.jsonl" || -s "${NEUTRAL_DIR}/failures.jsonl" ]]; then
    echo "WARNING: one or more stages recorded failures. The audit will report matched completed-case counts." >&2
  fi
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/audit_qwen3_protocol.py" \
    --blank "${BLANK_DIR}/trajectories.jsonl" \
    --roots "${ROOT_FILE}" \
    --tree "${TREE_DIR}/terminal_trajectories.jsonl" \
    --neutral "${NEUTRAL_DIR}/trajectories.jsonl" \
    --output "${ANALYSIS_DIR}/protocol_audit.json" \
    --blank_load_report "${BLANK_DIR}/load_report.json" \
    --require_complete

  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/filter_tree_roots.py" \
    --input "${TREE_DIR}/terminal_trajectories.jsonl" \
    --output "${PRIMARY_TREE_FILE}" \
    --root_origin qwen_natural_root

  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/verify_outputs.py" \
    --input "${TREE_DIR}/terminal_trajectories.jsonl" \
    --report "${TREE_DIR}/verification_report.json"

  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/analyze_action_tree.py" \
    --input "${PRIMARY_TREE_FILE}" \
    --output_dir "${TREE_DIR}/analysis_primary_natural_roots" \
    --quality_metric f1 \
    --oracle_value auto \
    --pareto_cost_metric total_cost_tokens \
    --token_budget_metric total_cost_tokens \
    --token_budgets auto \
    --doc_budget_metric retrieved_docs_nominal \
    --doc_budgets 5,10,15,20,25,30 \
    --sensitivity_metrics f1,em,acc,normalized_exact,normalized_contains \
    --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
    --bootstrap_seed 2026

  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/analyze_action_tree.py" \
    --input "${TREE_DIR}/terminal_trajectories.jsonl" \
    --output_dir "${TREE_DIR}/analysis_sensitivity_all_roots" \
    --quality_metric f1 \
    --oracle_value auto \
    --pareto_cost_metric total_cost_tokens \
    --token_budget_metric total_cost_tokens \
    --token_budgets auto \
    --doc_budget_metric retrieved_docs_nominal \
    --doc_budgets 5,10,15,20,25,30 \
    --sensitivity_metrics f1,em,acc,normalized_exact,normalized_contains \
    --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
    --bootstrap_seed 2026

  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/analyze_qwen3_conditions.py" \
    --blank "${BLANK_DIR}/trajectories.jsonl" \
    --roots "${ROOT_FILE}" \
    --tree "${TREE_DIR}/terminal_trajectories.jsonl" \
    --neutral "${NEUTRAL_DIR}/trajectories.jsonl" \
    --output_dir "${ANALYSIS_DIR}" \
    --quality_metric f1 \
    --bootstrap_samples "${BOOTSTRAP_SAMPLES}"
}

dry_run() {
  mkdir -p "${OUTPUT_BASE}/dry_run"
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/run_qwen3_natural.py" \
    --condition blank \
    --intermediate_data "${INTERMEDIATE_DATA}" \
    --output_dir "${OUTPUT_BASE}/dry_run/blank" \
    --model_url "${MODEL_URL}" \
    --model_path "${MODEL_PATH}" \
    --retriever_url "${RETRIEVER_URL}" \
    --limit "${LIMIT}" \
    --sample_ids "${SAMPLE_IDS}" \
    --dry_run
  echo "Dry run passed. No network calls were made."
}

guard_qwen_only
MODE="${1:-help}"
case "${MODE}" in
  dry-run) dry_run ;;
  blank) run_blank ;;
  roots) prepare_roots ;;
  tree) run_tree ;;
  neutral) run_neutral ;;
  analyze) analyze_all ;;
  all)
    run_blank
    prepare_roots
    run_tree
    run_neutral
    analyze_all
    ;;
  *)
    echo "Usage: bash run_qwen3_three_conditions.sh {dry-run|blank|roots|tree|neutral|analyze|all}" >&2
    exit 2
    ;;
esac
