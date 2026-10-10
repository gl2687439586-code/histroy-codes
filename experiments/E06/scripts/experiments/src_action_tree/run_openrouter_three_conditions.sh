#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-openai/gpt-5.6-sol}"
MODEL_URL="${MODEL_URL:-https://openrouter.ai/api/v1}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/researcher/smartsearch_rebuild/models/Qwen3-8B}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:6008}"
SANDBOX_URL="${SANDBOX_URL:-http://127.0.0.1:6006}"
INTERMEDIATE_DATA="${INTERMEDIATE_DATA:-/home/researcher/smartsearch_rebuild/data/intermediate_data.json}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/researcher/smartsearch_rebuild/outputs/src_action_tree/openrouter_gpt56sol_v6}"
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
# Per Python stage (Blank, forced roots, tree, Neutral), not a whole-run cap.
# A finite default prevents an accidental unbounded API bill; raise explicitly
# and resume if the formal full tree legitimately needs more.
MAX_USD="${MAX_USD:-25}"
OPENROUTER_PROVIDER="${OPENROUTER_PROVIDER:-openai}"
OPENROUTER_ALLOW_FALLBACKS="${OPENROUTER_ALLOW_FALLBACKS:-0}"
OPENROUTER_REASONING_EFFORT="${OPENROUTER_REASONING_EFFORT:-medium}"
export OPENROUTER_PROVIDER OPENROUTER_ALLOW_FALLBACKS OPENROUTER_REASONING_EFFORT

BLANK_DIR="${OUTPUT_BASE}/01_blank"
ROOT_DIR="${OUTPUT_BASE}/roots"
ROOT_FILE="${ROOT_DIR}/openrouter_first_search_checkpoints.jsonl"
TREE_DIR="${OUTPUT_BASE}/02_src_tree"
NEUTRAL_DIR="${OUTPUT_BASE}/03_neutral"
ANALYSIS_DIR="${OUTPUT_BASE}/joint_analysis"
PRIMARY_TREE_FILE="${ANALYSIS_DIR}/terminal_trajectories_natural_roots.jsonl"

guard_openrouter_only() {
  [[ "${MODEL_PATH}" == "openai/gpt-5.6-sol" ]] || {
    echo "MODEL_PATH must be exactly openai/gpt-5.6-sol: ${MODEL_PATH}" >&2
    exit 2
  }
  [[ "${MODEL_URL%/}" == "https://openrouter.ai/api/v1" ]] || {
    echo "MODEL_URL must be https://openrouter.ai/api/v1: ${MODEL_URL}" >&2
    exit 2
  }
  if [[ "${MODE:-}" =~ ^(blank|roots|tree|neutral|all)$ ]]; then
    [[ -n "${OPENROUTER_API_KEY:-}" ]] || {
      echo "OPENROUTER_API_KEY is not set." >&2
      exit 2
    }
  fi
  [[ -f "${INTERMEDIATE_DATA}" ]] || { echo "Missing ${INTERMEDIATE_DATA}" >&2; exit 2; }
  [[ -d "${TOKENIZER_PATH}" ]] || { echo "Missing ${TOKENIZER_PATH}" >&2; exit 2; }
}

natural_args=(
  --intermediate_data "${INTERMEDIATE_DATA}"
  --model_url "${MODEL_URL}"
  --model_path "${MODEL_PATH}"
  --tokenizer_path "${TOKENIZER_PATH}"
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
  --max_usd "${MAX_USD}"
)

run_blank() {
  mkdir -p "${BLANK_DIR}"
  OPENROUTER_STAGE=blank OPENROUTER_BILLING_LEDGER="${BLANK_DIR}/api_billing_ledger.jsonl" \
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/run_openrouter_natural.py" \
    --condition blank --output_dir "${BLANK_DIR}" "${natural_args[@]}"
}

prepare_roots() {
  mkdir -p "${ROOT_DIR}"
  [[ -s "${BLANK_DIR}/trajectories.jsonl" ]] || {
    echo "Blank stage is incomplete: missing or empty ${BLANK_DIR}/trajectories.jsonl" >&2
    echo "Inspect ${BLANK_DIR}/run_status.json and ${BLANK_DIR}/failures.jsonl before resuming." >&2
    return 1
  }
  OPENROUTER_STAGE=roots OPENROUTER_BILLING_LEDGER="${ROOT_DIR}/api_billing_ledger.jsonl" \
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/prepare_openrouter_roots.py" \
    --blank "${BLANK_DIR}/trajectories.jsonl" \
    --output "${ROOT_FILE}" \
    --model_url "${MODEL_URL}" \
    --model_path "${MODEL_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --retriever_url "${RETRIEVER_URL}" \
    --sandbox_url "${SANDBOX_URL}" \
    --retriever_mode direct \
    --top_k 5 \
    --temperature "${TEMPERATURE}" \
    --max_new_tokens "${SEARCH_MAX_NEW_TOKENS}" \
    --max_usd "${MAX_USD}" \
    --resume
}

run_tree() {
  mkdir -p "${TREE_DIR}"
  [[ -s "${ROOT_FILE}" ]] || {
    echo "Root stage is incomplete: missing or empty ${ROOT_FILE}" >&2
    return 1
  }
  OPENROUTER_STAGE=tree OPENROUTER_BILLING_LEDGER="${TREE_DIR}/api_billing_ledger.jsonl" \
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/run_action_tree.py" \
    --intermediate_data "${ROOT_FILE}" \
    --output_dir "${TREE_DIR}" \
    --model_url "${MODEL_URL}" \
    --model_path "${MODEL_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --api_mode openrouter \
    --retriever_url "${RETRIEVER_URL}" \
    --sandbox_url "${SANDBOX_URL}" \
    --retriever_mode direct \
    --checkpoint_format openrouter \
    --prompt_protocol openrouter_matched \
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
    --max_usd "${MAX_USD}" \
    --resume \
    --retry_failures
}

run_neutral() {
  mkdir -p "${NEUTRAL_DIR}"
  [[ -s "${ROOT_FILE}" ]] || {
    echo "Root stage is incomplete: missing or empty ${ROOT_FILE}" >&2
    return 1
  }
  OPENROUTER_STAGE=neutral OPENROUTER_BILLING_LEDGER="${NEUTRAL_DIR}/api_billing_ledger.jsonl" \
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/run_openrouter_natural.py" \
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
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/audit_openrouter_protocol.py" \
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
    --root_origin openrouter_natural_root

  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/verify_outputs.py" \
    --input "${TREE_DIR}/terminal_trajectories.jsonl" \
    --report "${TREE_DIR}/verification_report.json"

  if [[ -s "${PRIMARY_TREE_FILE}" ]]; then
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
      --input "${PRIMARY_TREE_FILE}" \
      --output_dir "${TREE_DIR}/analysis_api_dollars_primary_natural_roots" \
      --quality_metric f1 \
      --oracle_value auto \
      --pareto_cost_metric api_cost_usd \
      --token_budget_metric total_cost_tokens \
      --token_budgets auto \
      --doc_budget_metric retrieved_docs_nominal \
      --doc_budgets 5,10,15,20,25,30 \
      --sensitivity_metrics f1,em,acc,normalized_exact,normalized_contains \
      --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
      --bootstrap_seed 2026
  else
    echo "Primary natural-root analysis unavailable: no natural GPT-5.6 Search roots were observed." >&2
    echo "Continuing with the explicitly labeled all-root/forced-root sensitivity analysis." >&2
  fi

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

  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/analyze_openrouter_conditions.py" \
    --blank "${BLANK_DIR}/trajectories.jsonl" \
    --roots "${ROOT_FILE}" \
    --tree "${TREE_DIR}/terminal_trajectories.jsonl" \
    --neutral "${NEUTRAL_DIR}/trajectories.jsonl" \
    --output_dir "${ANALYSIS_DIR}" \
    --quality_metric f1 \
    --bootstrap_samples "${BOOTSTRAP_SAMPLES}"

  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/summarize_openrouter_billing.py" \
    --input "${BLANK_DIR}/api_billing_ledger.jsonl" \
    --input "${ROOT_DIR}/api_billing_ledger.jsonl" \
    --input "${TREE_DIR}/api_billing_ledger.jsonl" \
    --input "${NEUTRAL_DIR}/api_billing_ledger.jsonl" \
    --output "${ANALYSIS_DIR}/openrouter_billing_summary.json"
}

dry_run() {
  mkdir -p "${OUTPUT_BASE}/dry_run"
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/run_openrouter_natural.py" \
    --condition blank \
    --intermediate_data "${INTERMEDIATE_DATA}" \
    --output_dir "${OUTPUT_BASE}/dry_run/blank" \
    --model_url "${MODEL_URL}" \
    --model_path "${MODEL_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --retriever_url "${RETRIEVER_URL}" \
    --limit "${LIMIT}" \
    --sample_ids "${SAMPLE_IDS}" \
    --dry_run
  echo "Dry run passed. No network calls were made."
}

MODE="${1:-help}"
guard_openrouter_only
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
    echo "Usage: bash run_openrouter_three_conditions.sh {dry-run|blank|roots|tree|neutral|analyze|all}" >&2
    exit 2
    ;;
esac
