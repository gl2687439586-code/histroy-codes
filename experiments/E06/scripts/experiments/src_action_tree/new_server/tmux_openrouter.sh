#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

MODEL_URL="${OPENROUTER_MODEL_URL:-https://openrouter.ai/api/v1}"
MODEL_PATH="${OPENROUTER_MODEL_ID:-openai/gpt-5.6-sol}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${WORK_ROOT}/models/Qwen3-8B}"
PROBE_OUTPUT_BASE="${OPENROUTER_PROBE_OUTPUT_BASE:-${OUTPUT_ROOT}/openrouter_gpt56sol_probe5_v6}"
FULL_OUTPUT_BASE="${OPENROUTER_FULL_OUTPUT_BASE:-${OUTPUT_ROOT}/openrouter_gpt56sol_bamboogle_v6}"
MAX_USD="${MAX_USD:-25}"
OPENROUTER_PROVIDER="${OPENROUTER_PROVIDER:-openai}"
OPENROUTER_ALLOW_FALLBACKS="${OPENROUTER_ALLOW_FALLBACKS:-0}"
OPENROUTER_REASONING_EFFORT="${OPENROUTER_REASONING_EFFORT:-medium}"
AUTO_RESUME_TRANSIENT="${AUTO_RESUME_TRANSIENT:-1}"
AUTO_RESUME_MAX_ATTEMPTS="${AUTO_RESUME_MAX_ATTEMPTS:-20}"
AUTO_RESUME_BASE_DELAY_SECONDS="${AUTO_RESUME_BASE_DELAY_SECONDS:-15}"
AUTO_RESUME_MAX_DELAY_SECONDS="${AUTO_RESUME_MAX_DELAY_SECONDS:-120}"
export OPENROUTER_PROVIDER OPENROUTER_ALLOW_FALLBACKS OPENROUTER_REASONING_EFFORT

require_dir "${CORE_ENV_PREFIX}"
require_dir "${TOKENIZER_PATH}"
require_file "${RETRIEVER_CONFIG}"
require_file "${INTERMEDIATE_DATA}"
require_file "${INDEX_PATH}"

require_api_key() {
  [[ -n "${OPENROUTER_API_KEY:-}" ]] || {
    echo "OPENROUTER_API_KEY is not set. Export it in this shell first." >&2
    exit 2
  }
}

LOG_DIR="${OUTPUT_ROOT}/tmux_logs_openrouter"
STATUS_DIR="${OUTPUT_ROOT}/tmux_status_openrouter"
mkdir -p "${LOG_DIR}" "${STATUS_DIR}"

RETRIEVER_SESSION="${RETRIEVER_SESSION:-src_or_retriever}"
SANDBOX_SESSION="${SANDBOX_SESSION:-src_or_sandbox}"
PROBE_SESSION="${PROBE_SESSION:-src_or_probe5}"
EXPERIMENT_SESSION="${EXPERIMENT_SESSION:-src_or_experiment}"

has_session() { tmux has-session -t "$1" 2>/dev/null; }
port_open() { (echo >/dev/tcp/127.0.0.1/"$1") >/dev/null 2>&1; }
status_file_for() { printf '%s/%s.exit\n' "${STATUS_DIR}" "$1"; }
log_file_for() { printf '%s/%s.log\n' "${LOG_DIR}" "$1"; }

start_managed() {
  local session="$1" label="$2"
  shift 2
  local status_file log_file command_string
  status_file="$(status_file_for "${session}")"
  log_file="$(log_file_for "${session}")"
  if has_session "${session}"; then
    if [[ -f "${status_file}" ]]; then
      tmux kill-session -t "${session}"
    else
      echo "${label} session is already active: ${session}" >&2
      return 2
    fi
  fi
  rm -f "${status_file}" "${status_file}.tmp."* 2>/dev/null || true
  printf -v command_string '%q ' \
    "${SCRIPT_DIR}/session_wrapper.sh" "${label}" "${log_file}" "${status_file}" "$@"
  tmux new-session -d -s "${session}" "${command_string}"
  echo "Started ${label}: tmux ${session}; log ${log_file}"
}

wait_http() {
  local url="$1" label="$2" session="$3" elapsed=0 status_file
  status_file="$(status_file_for "${session}")"
  until curl -fsS --max-time 5 "${url}" >/dev/null 2>&1; do
    if [[ -f "${status_file}" ]]; then
      echo "${label} exited with code $(cat "${status_file}")." >&2
      tail -n 80 "$(log_file_for "${session}")" >&2 || true
      return 1
    fi
    (( elapsed < WAIT_SECONDS )) || {
      echo "Timed out after ${WAIT_SECONDS}s waiting for ${label}." >&2
      return 1
    }
    sleep 10
    elapsed=$((elapsed + 10))
  done
  echo "${label} ready: ${url}"
}

start_retriever() {
  if curl -fsS --max-time 5 "${RETRIEVER_URL}/health" >/dev/null 2>&1; then
    echo "Retriever already ready: ${RETRIEVER_URL}/health"
    return
  fi
  if port_open "${RETRIEVER_PORT}"; then
    echo "Port ${RETRIEVER_PORT} is occupied but Retriever health failed." >&2
    exit 2
  fi
  start_managed "${RETRIEVER_SESSION}" "CPU-FAISS retriever" \
    env SRCAT_VMEM_GIB="${RETRIEVER_VMEM_GIB}" \
      CUDA_VISIBLE_DEVICES="${RETRIEVER_GPU}" PYTHONPATH="${REPO_ROOT}/src" \
      HF_HOME="${HF_HOME}" HF_HUB_CACHE="${HF_HUB_CACHE}" \
      HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" HF_DATASETS_IN_MEMORY_MAX_SIZE=0 \
      TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
      TMPDIR="${TMPDIR}" MALLOC_ARENA_MAX=2 TOKENIZERS_PARALLELISM=false \
      OMP_NUM_THREADS="${RETRIEVER_THREADS}" MKL_NUM_THREADS="${RETRIEVER_THREADS}" \
      OPENBLAS_NUM_THREADS="${RETRIEVER_THREADS}" NUMEXPR_NUM_THREADS="${RETRIEVER_THREADS}" \
    "${CORE_ENV_PREFIX}/bin/python" -u "${REPO_ROOT}/scripts/serving/retriever_serving.py" \
      --config "${RETRIEVER_CONFIG}" --num_retriever 1 --port "${RETRIEVER_PORT}"
  wait_http "${RETRIEVER_URL}/health" Retriever "${RETRIEVER_SESSION}"
}

start_sandbox() {
  if curl -fsS --max-time 5 "${SANDBOX_URL}/docs" >/dev/null 2>&1; then
    echo "Sandbox already ready: ${SANDBOX_URL}/docs"
    return
  fi
  if port_open "${SANDBOX_PORT}"; then
    echo "Port ${SANDBOX_PORT} is occupied but Sandbox health failed." >&2
    exit 2
  fi
  start_managed "${SANDBOX_SESSION}" Sandbox \
    env PYTHONPATH="${REPO_ROOT}/src" \
    "${CORE_ENV_PREFIX}/bin/python" -u "${REPO_ROOT}/scripts/serving/sandbox.py" \
      --port "${SANDBOX_PORT}"
  wait_http "${SANDBOX_URL}/docs" Sandbox "${SANDBOX_SESSION}"
}

services() {
  require_api_key
  start_retriever
  start_sandbox
  OPENROUTER_STAGE=service_probe \
  OPENROUTER_BILLING_LEDGER="${OUTPUT_ROOT}/openrouter_service_probe_billing.jsonl" \
  "${CORE_ENV_PREFIX}/bin/python" -u "${SRC_ACTION_TREE_DIR}/probe_openrouter.py" \
    --model_url "${MODEL_URL}" --model "${MODEL_PATH}" \
    --retriever_url "${RETRIEVER_URL}" --sandbox_url "${SANDBOX_URL}" --timeout 300
}

start_experiment() {
  require_api_key
  local session="$1" label="$2" output_base="$3" limit="$4"
  # Existing tmux servers do not reliably inherit custom variables from the
  # invoking shell.  Install the secret only long enough to create the pane,
  # then remove it from tmux's global environment.
  tmux set-environment -g OPENROUTER_API_KEY "${OPENROUTER_API_KEY}"
  if [[ -n "${OPENROUTER_HTTP_REFERER:-}" ]]; then
    tmux set-environment -g OPENROUTER_HTTP_REFERER "${OPENROUTER_HTTP_REFERER}"
  fi
  set +e
  start_managed "${session}" "${label}" \
    env PATH="${CORE_ENV_PREFIX}/bin:${PATH}" PYTHON_BIN="${CORE_ENV_PREFIX}/bin/python" \
      PYTHONPATH="${REPO_ROOT}/src" HF_HOME="${HF_HOME}" HF_HUB_CACHE="${HF_HUB_CACHE}" \
      HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
      XDG_CACHE_HOME="${XDG_CACHE_HOME}" TMPDIR="${TMPDIR}" \
      OUTPUT_BASE="${output_base}" INTERMEDIATE_DATA="${INTERMEDIATE_DATA}" \
      MODEL_PATH="${MODEL_PATH}" MODEL_URL="${MODEL_URL}" TOKENIZER_PATH="${TOKENIZER_PATH}" \
      RETRIEVER_URL="${RETRIEVER_URL}" SANDBOX_URL="${SANDBOX_URL}" \
      LIMIT="${limit}" MIN_HORIZON="${MIN_HORIZON}" MAX_HORIZON="${MAX_HORIZON}" \
      TEMPERATURE="${TEMPERATURE}" CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
      SEARCH_MAX_NEW_TOKENS="${SEARCH_MAX_NEW_TOKENS}" READ_MAX_NEW_TOKENS="${READ_MAX_NEW_TOKENS}" \
      ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS}" \
      ANSWER_REPAIR_MAX_NEW_TOKENS="${ANSWER_REPAIR_MAX_NEW_TOKENS}" MAX_USD="${MAX_USD}" \
      OPENROUTER_PROVIDER="${OPENROUTER_PROVIDER}" \
      OPENROUTER_ALLOW_FALLBACKS="${OPENROUTER_ALLOW_FALLBACKS}" \
      OPENROUTER_REASONING_EFFORT="${OPENROUTER_REASONING_EFFORT}" \
      AUTO_RESUME_TRANSIENT="${AUTO_RESUME_TRANSIENT}" \
      AUTO_RESUME_MAX_ATTEMPTS="${AUTO_RESUME_MAX_ATTEMPTS}" \
      AUTO_RESUME_BASE_DELAY_SECONDS="${AUTO_RESUME_BASE_DELAY_SECONDS}" \
      AUTO_RESUME_MAX_DELAY_SECONDS="${AUTO_RESUME_MAX_DELAY_SECONDS}" \
      RESOURCE_CHECK_INTERVAL_SECONDS="${RESOURCE_CHECK_INTERVAL_SECONDS}" \
      RESOURCE_PROGRESS_EVERY="${RESOURCE_PROGRESS_EVERY}" \
      RESOURCE_MIN_AVAILABLE_RAM_GIB="${RESOURCE_MIN_AVAILABLE_RAM_GIB}" \
      RESOURCE_MAX_SWAP_USED_GIB="${RESOURCE_MAX_SWAP_USED_GIB}" \
      RESOURCE_MIN_FREE_DISK_GIB="${RESOURCE_MIN_FREE_DISK_GIB}" \
      RESOURCE_MAX_GPU_MEMORY_PERCENT="${RESOURCE_MAX_GPU_MEMORY_PERCENT}" \
      RESOURCE_STOP_ON_CRITICAL="${RESOURCE_STOP_ON_CRITICAL}" \
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    bash "${SCRIPT_DIR}/experiment_openrouter_with_monitor.sh" "${output_base}" all
  local start_status=$?
  set -e
  tmux set-environment -gu OPENROUTER_API_KEY || true
  tmux set-environment -gu OPENROUTER_HTTP_REFERER || true
  return "${start_status}"
}

probe5() {
  services
  start_experiment "${PROBE_SESSION}" "OpenRouter GPT-5.6-Sol five-question gate" "${PROBE_OUTPUT_BASE}" 5
  echo "Follow: tail -f $(log_file_for "${PROBE_SESSION}")"
}

probe_passed() {
  local audit="${PROBE_OUTPUT_BASE}/joint_analysis/protocol_audit.json"
  [[ -f "${audit}" ]] || return 1
  "${CORE_ENV_PREFIX}/bin/python" - "${audit}" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("passed") is True else 1)
PY
}

full() {
  services
  if [[ "${SKIP_PROBE_GATE:-0}" != 1 ]] && ! probe_passed; then
    echo "Refusing full run until the GPT-5.6-Sol five-question audit passes:" >&2
    echo "  ${PROBE_OUTPUT_BASE}/joint_analysis/protocol_audit.json" >&2
    exit 2
  fi
  start_experiment "${EXPERIMENT_SESSION}" "OpenRouter GPT-5.6-Sol full Bamboogle v6" "${FULL_OUTPUT_BASE}" 0
  echo "Follow: tail -f $(log_file_for "${EXPERIMENT_SESSION}")"
}

status() {
  echo "=== tmux ==="
  tmux ls 2>/dev/null || true
  echo "=== endpoints ==="
  curl -fsS --max-time 3 "${RETRIEVER_URL}/health" >/dev/null && echo "retriever: healthy" || echo "retriever: unavailable"
  curl -fsS --max-time 3 "${SANDBOX_URL}/docs" >/dev/null && echo "sandbox: healthy" || echo "sandbox: unavailable"
  echo "OpenRouter model: ${MODEL_PATH} (remote API; run services for a billed live probe)"
  echo "=== GPU/RAM/disk ==="
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  free -h
  df -h "${WORK_ROOT}" /
  echo "=== full log tail ==="
  tail -n 50 "$(log_file_for "${EXPERIMENT_SESSION}")" 2>/dev/null || true
}

resources() {
  local output_base="${1:-${FULL_OUTPUT_BASE}}"
  "${CORE_ENV_PREFIX}/bin/python" -u "${SCRIPT_DIR}/resource_monitor.py" \
    --output-base "${output_base}" --work-root "${WORK_ROOT}" \
    --interval-seconds "${RESOURCE_CHECK_INTERVAL_SECONDS}" \
    --progress-every "${RESOURCE_PROGRESS_EVERY}" \
    --min-available-ram-gib "${RESOURCE_MIN_AVAILABLE_RAM_GIB}" \
    --max-swap-used-gib "${RESOURCE_MAX_SWAP_USED_GIB}" \
    --min-free-disk-gib "${RESOURCE_MIN_FREE_DISK_GIB}" \
    --max-gpu-memory-percent "${RESOURCE_MAX_GPU_MEMORY_PERCENT}" --once
}

case "${1:-help}" in
  services) services ;;
  probe5) probe5 ;;
  full|resume-full) full ;;
  status) status ;;
  resources) resources "${2:-}" ;;
  attach) tmux attach -t "${2:-${EXPERIMENT_SESSION}}" ;;
  *)
    echo "Usage: bash tmux_openrouter.sh {services|probe5|full|resume-full|status|resources [output]|attach [session]}" >&2
    exit 2
    ;;
esac
