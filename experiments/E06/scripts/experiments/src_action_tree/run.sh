#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-/home/researcher/smartsearch_workspace/outputs/full_bamboogle_100w_timeout120/bamboogle_2026_06_25_15_04_SmartSearch-3B_100w_timeout120}"
INTERMEDIATE_DATA="${INTERMEDIATE_DATA:-${SOURCE_RUN_DIR}/intermediate_data.json}"
TRACE_JSONL="${TRACE_JSONL:-${SOURCE_RUN_DIR}/human_trace_report_v2/trace_steps_full_valid.jsonl}"

# Compatibility with the alternative absolute path mentioned in older notes.
if [[ ! -f "${TRACE_JSONL}" && -f "/human_trace_report_v2/trace_steps_full_valid.jsonl" ]]; then
  TRACE_JSONL="/human_trace_report_v2/trace_steps_full_valid.jsonl"
fi

MODEL_PATH="${MODEL_PATH:-/home/researcher/smartsearch_workspace/models/SmartSearch-3B}"
MODEL_URL="${MODEL_URL:-http://127.0.0.1:9000}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:6008}"
SANDBOX_URL="${SANDBOX_URL:-http://127.0.0.1:6006}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/researcher/smartsearch_workspace/outputs/src_action_tree}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TEMPERATURE="${TEMPERATURE:-0.0}"
N_ROLLOUTS="${N_ROLLOUTS:-1}"
MAX_WORKERS="${MAX_WORKERS:-1}"
SAMPLE_IDS="${SAMPLE_IDS:-}"
HORIZON_MODE="${HORIZON_MODE:-original_searches}"
MIN_HORIZON="${MIN_HORIZON:-3}"
MAX_HORIZON="${MAX_HORIZON:-6}"
FIXED_HORIZON="${FIXED_HORIZON:-3}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-random}"
SAMPLE_SEED="${SAMPLE_SEED:-2026}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
ACTION_ATTEMPTS="${ACTION_ATTEMPTS:-4}"
ACTION_VALIDATION="${ACTION_VALIDATION:-flexible}"
SEARCH_MAX_NEW_TOKENS="${SEARCH_MAX_NEW_TOKENS:-512}"
READ_MAX_NEW_TOKENS="${READ_MAX_NEW_TOKENS:-768}"
ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS:-1024}"
QUALITY_METRIC="${QUALITY_METRIC:-f1}"
TOKEN_BUDGETS="${TOKEN_BUDGETS:-auto}"
DOC_BUDGETS="${DOC_BUDGETS:-5,10,15,20,25,30}"
SENSITIVITY_METRICS="${SENSITIVITY_METRICS:-f1,em,acc,normalized_contains}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
RETRIEVER_MODE="${RETRIEVER_MODE:-direct}"
FIXED_DOC_TOKENS="${FIXED_DOC_TOKENS:-0}"

MODE="${1:-dry-run}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${OUTPUT_BASE}/${MODE}_${STAMP}}"

common_args=(
  --intermediate_data "${INTERMEDIATE_DATA}"
  --trace_jsonl "${TRACE_JSONL}"
  --output_dir "${RUN_DIR}"
  --model_url "${MODEL_URL}"
  --retriever_url "${RETRIEVER_URL}"
  --sandbox_url "${SANDBOX_URL}"
  --model_path "${MODEL_PATH}"
  --api_mode sglang
  --retriever_mode "${RETRIEVER_MODE}"
  --top_k 5
  --context_length "${CONTEXT_LENGTH}"
  --temperature "${TEMPERATURE}"
  --n_rollouts "${N_ROLLOUTS}"
  --max_workers "${MAX_WORKERS}"
  --sample_ids "${SAMPLE_IDS}"
  --horizon_mode "${HORIZON_MODE}"
  --min_horizon "${MIN_HORIZON}"
  --max_horizon "${MAX_HORIZON}"
  --fixed_horizon "${FIXED_HORIZON}"
  --sample_strategy "${SAMPLE_STRATEGY}"
  --sample_seed "${SAMPLE_SEED}"
  --action_attempts "${ACTION_ATTEMPTS}"
  --action_validation "${ACTION_VALIDATION}"
  --search_max_new_tokens "${SEARCH_MAX_NEW_TOKENS}"
  --read_max_new_tokens "${READ_MAX_NEW_TOKENS}"
  --answer_max_new_tokens "${ANSWER_MAX_NEW_TOKENS}"
  --model_timeout 300
  --retriever_timeout 300
  --fixed_doc_tokens "${FIXED_DOC_TOKENS}"
)

analyze_run() {
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/verify_outputs.py" \
    --input "${RUN_DIR}/terminal_trajectories.jsonl" \
    --report "${RUN_DIR}/verification_report.json"

  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/analyze_action_tree.py" \
    --input "${RUN_DIR}/terminal_trajectories.jsonl" \
    --output_dir "${RUN_DIR}/analysis" \
    --quality_metric "${QUALITY_METRIC}" \
    --oracle_value auto \
    --pareto_cost_metric total_cost_tokens \
    --token_budget_metric total_cost_tokens \
    --token_budgets "${TOKEN_BUDGETS}" \
    --doc_budget_metric retrieved_docs_nominal \
    --doc_budgets "${DOC_BUDGETS}" \
    --sensitivity_metrics "${SENSITIVITY_METRICS}" \
    --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
    --bootstrap_seed 2026
}

run_limit() {
  local limit="$1"
  mkdir -p "${RUN_DIR}"
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/run_action_tree.py" \
    "${common_args[@]}" \
    --limit "${limit}" \
    --resume
  analyze_run
  echo "Completed. Results: ${RUN_DIR}"
  echo "Main report: ${RUN_DIR}/analysis/report.md"
}

case "${MODE}" in
  self-test)
    "${PYTHON_BIN}" -u "${SCRIPT_DIR}/self_test.py"
    ;;
  check-services)
    "${PYTHON_BIN}" -u "${SCRIPT_DIR}/probe_services.py" \
      --model_url "${MODEL_URL}" \
      --retriever_url "${RETRIEVER_URL}" \
      --sandbox_url "${SANDBOX_URL}" \
      --model_path "${MODEL_PATH}" \
      --timeout 300
    ;;
  dry-run)
    mkdir -p "${RUN_DIR}"
    "${PYTHON_BIN}" -u "${SCRIPT_DIR}/run_action_tree.py" \
      "${common_args[@]}" \
      --limit 5 \
      --dry_run
    echo "Dry-run report: ${RUN_DIR}/dry_run_report.json"
    ;;
  dry-run20)
    mkdir -p "${RUN_DIR}"
    "${PYTHON_BIN}" -u "${SCRIPT_DIR}/run_action_tree.py" \
      "${common_args[@]}" \
      --limit 20 \
      --dry_run
    echo "Dry-run report: ${RUN_DIR}/dry_run_report.json"
    ;;
  dry-run-full)
    mkdir -p "${RUN_DIR}"
    "${PYTHON_BIN}" -u "${SCRIPT_DIR}/run_action_tree.py" \
      "${common_args[@]}" \
      --limit 0 \
      --dry_run
    echo "Full-dataset dry-run report: ${RUN_DIR}/dry_run_report.json"
    ;;
  probe5)
    run_limit 5
    ;;
  probe20)
    run_limit 20
    ;;
  full)
    run_limit 0
    ;;
  analyze)
    if [[ ! -f "${RUN_DIR}/terminal_trajectories.jsonl" ]]; then
      echo "RUN_DIR must point to an existing experiment for analyze mode." >&2
      exit 2
    fi
    analyze_run
    ;;
  *)
    echo "Usage: bash run.sh {self-test|check-services|dry-run|dry-run20|dry-run-full|probe5|probe20|full|analyze}" >&2
    exit 2
    ;;
esac
