#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CONDA_SH="${CONDA_SH:-/home/researcher/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-smartsearch}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"

MODEL_PATH="${MODEL_PATH:-/home/researcher/models/Qwen3-8B}"
MODEL_HOST="${MODEL_HOST:-127.0.0.1}"
MODEL_PORT="${MODEL_PORT:-9001}"
MODEL_URL="${MODEL_URL:-http://${MODEL_HOST}:${MODEL_PORT}}"
MODEL_CONTEXT_LENGTH="${MODEL_CONTEXT_LENGTH:-32768}"
MODEL_MEM_FRACTION="${MODEL_MEM_FRACTION:-0.85}"

RETRIEVER_HOST="${RETRIEVER_HOST:-127.0.0.1}"
RETRIEVER_PORT="${RETRIEVER_PORT:-6008}"
RETRIEVER_URL="${RETRIEVER_URL:-http://${RETRIEVER_HOST}:${RETRIEVER_PORT}}"
RETRIEVER_CONFIG="${RETRIEVER_CONFIG:-${REPO_ROOT}/scripts/serving/retriever_config_src_action_tree.yaml}"

SANDBOX_HOST="${SANDBOX_HOST:-127.0.0.1}"
SANDBOX_PORT="${SANDBOX_PORT:-6006}"
SANDBOX_URL="${SANDBOX_URL:-http://${SANDBOX_HOST}:${SANDBOX_PORT}}"

OUTPUT_BASE="${OUTPUT_BASE:-/home/researcher/smartsearch_workspace/outputs/src_action_tree}"
PLAN_DIR="${PLAN_DIR:-${OUTPUT_BASE}/qwen3_full_adaptive_h_v4_plan}"
RUN_DIR="${RUN_DIR:-${OUTPUT_BASE}/qwen3_full_adaptive_h_v4_answer2048}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/tmux_logs}"

RETRIEVER_TMUX="${RETRIEVER_TMUX:-srcat_retriever}"
SANDBOX_TMUX="${SANDBOX_TMUX:-srcat_sandbox}"
MODEL_TMUX="${MODEL_TMUX:-srcat_model}"
EXPERIMENT_TMUX="${EXPERIMENT_TMUX:-srcat_full}"

SERVICE_WAIT_SECONDS="${SERVICE_WAIT_SECONDS:-1800}"
ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS:-2048}"
MIN_HORIZON="${MIN_HORIZON:-3}"
MAX_HORIZON="${MAX_HORIZON:-6}"
ACTION_VALIDATION="${ACTION_VALIDATION:-flexible}"
ACTION_ATTEMPTS="${ACTION_ATTEMPTS:-1}"

activate_prefix="source '${CONDA_SH}' && conda activate '${CONDA_ENV}' && cd '${REPO_ROOT}' && export PYTHONUNBUFFERED=1"

require_prerequisites() {
  command -v tmux >/dev/null 2>&1 || {
    echo "tmux is not installed or not on PATH." >&2
    exit 2
  }
  [[ -f "${CONDA_SH}" ]] || {
    echo "Conda activation script not found: ${CONDA_SH}" >&2
    exit 2
  }
  [[ -f "${RETRIEVER_CONFIG}" ]] || {
    echo "Retriever config not found: ${RETRIEVER_CONFIG}" >&2
    exit 2
  }
  [[ -d "${MODEL_PATH}" ]] || {
    echo "Qwen model path not found: ${MODEL_PATH}" >&2
    exit 2
  }
}

tmux_exists() {
  tmux has-session -t "${1}" 2>/dev/null
}

start_tmux() {
  local name="$1"
  local command="$2"
  if tmux_exists "${name}"; then
    echo "tmux session already exists: ${name}" >&2
    return 1
  fi
  tmux new-session -d -s "${name}" bash -l
  # Send the command literally, then close the session when the process ends.
  tmux send-keys -t "${name}" -l \
    "${command}; task_status=\$?; echo '[src-action-tree] process exited with status' \${task_status}; exit \${task_status}"
  tmux send-keys -t "${name}" Enter
  echo "Started tmux session: ${name}"
}

wait_http() {
  local url="$1"
  local label="$2"
  local session="${3:-}"
  local elapsed=0
  while ! curl -fsS --max-time 5 "${url}" >/dev/null 2>&1; do
    if [[ -n "${session}" ]] && ! tmux_exists "${session}"; then
      echo "${label} tmux session exited before becoming ready: ${session}" >&2
      echo "Inspect: ${LOG_DIR}" >&2
      return 1
    fi
    if (( elapsed >= SERVICE_WAIT_SECONDS )); then
      echo "Timed out waiting for ${label}: ${url}" >&2
      return 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  echo "${label} is ready: ${url}"
}

wait_tcp() {
  local host="$1"
  local port="$2"
  local label="$3"
  local session="${4:-}"
  local elapsed=0
  while ! (echo >/dev/tcp/"${host}"/"${port}") >/dev/null 2>&1; do
    if [[ -n "${session}" ]] && ! tmux_exists "${session}"; then
      echo "${label} tmux session exited before becoming ready: ${session}" >&2
      echo "Inspect: ${LOG_DIR}" >&2
      return 1
    fi
    if (( elapsed >= SERVICE_WAIT_SECONDS )); then
      echo "Timed out waiting for ${label}: ${host}:${port}" >&2
      return 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  echo "${label} is ready: ${host}:${port}"
}

refuse_unmanaged_port() {
  local host="$1"
  local port="$2"
  local session="$3"
  local label="$4"
  if (echo >/dev/tcp/"${host}"/"${port}") >/dev/null 2>&1 && ! tmux_exists "${session}"; then
    echo "${label} port ${host}:${port} is already occupied outside tmux ${session}." >&2
    echo "Stop or reuse that process before requesting a four-tmux launch; no process was killed." >&2
    return 1
  fi
}

start_retriever() {
  if tmux_exists "${RETRIEVER_TMUX}"; then
    echo "Reusing tmux session: ${RETRIEVER_TMUX}"
  else
    refuse_unmanaged_port "${RETRIEVER_HOST}" "${RETRIEVER_PORT}" "${RETRIEVER_TMUX}" "Retriever"
    start_tmux "${RETRIEVER_TMUX}" \
      "${activate_prefix} && set -o pipefail; CUDA_VISIBLE_DEVICES='${GPU_ID}' '${PYTHON_BIN}' -u scripts/serving/retriever_serving.py --config '${RETRIEVER_CONFIG}' --num_retriever 1 --port '${RETRIEVER_PORT}' 2>&1 | tee -a '${LOG_DIR}/retriever.log'"
  fi
  wait_http "${RETRIEVER_URL}/health" "Retriever" "${RETRIEVER_TMUX}"
}

start_sandbox() {
  if tmux_exists "${SANDBOX_TMUX}"; then
    echo "Reusing tmux session: ${SANDBOX_TMUX}"
  else
    refuse_unmanaged_port "${SANDBOX_HOST}" "${SANDBOX_PORT}" "${SANDBOX_TMUX}" "Sandbox"
    start_tmux "${SANDBOX_TMUX}" \
      "${activate_prefix} && set -o pipefail; '${PYTHON_BIN}' -u scripts/serving/sandbox.py --port '${SANDBOX_PORT}' 2>&1 | tee -a '${LOG_DIR}/sandbox.log'"
  fi
  wait_tcp "${SANDBOX_HOST}" "${SANDBOX_PORT}" "Sandbox" "${SANDBOX_TMUX}"
}

start_model() {
  if tmux_exists "${MODEL_TMUX}"; then
    echo "Reusing tmux session: ${MODEL_TMUX}"
  else
    refuse_unmanaged_port "${MODEL_HOST}" "${MODEL_PORT}" "${MODEL_TMUX}" "Model"
    start_tmux "${MODEL_TMUX}" \
      "${activate_prefix} && set -o pipefail; CUDA_VISIBLE_DEVICES='${GPU_ID}' '${PYTHON_BIN}' -u -m sglang.launch_server --served-model-name Qwen3-8B --model-path '${MODEL_PATH}' --tp 1 --dp 1 --context-length '${MODEL_CONTEXT_LENGTH}' --enable-metrics --dtype bfloat16 --host 0.0.0.0 --port '${MODEL_PORT}' --trust-remote-code --disable-overlap --disable-radix-cache --mem-fraction-static '${MODEL_MEM_FRACTION}' 2>&1 | tee -a '${LOG_DIR}/model.log'"
  fi
  wait_http "${MODEL_URL}/health" "Qwen model" "${MODEL_TMUX}"
}

check_all_services() {
  bash -lc "${activate_prefix} && '${PYTHON_BIN}' -u '${SCRIPT_DIR}/probe_services.py' --model_url '${MODEL_URL}' --retriever_url '${RETRIEVER_URL}' --sandbox_url '${SANDBOX_URL}' --model_path '${MODEL_PATH}' --timeout 300"
}

start_services() {
  require_prerequisites
  mkdir -p "${LOG_DIR}"
  # On one GPU, load the smaller Retriever first, then reserve Qwen memory.
  start_retriever
  start_sandbox
  start_model
  check_all_services
}

plan_full() {
  require_prerequisites
  mkdir -p "${PLAN_DIR}"
  bash -lc "${activate_prefix} && RUN_DIR='${PLAN_DIR}' MODEL_URL='${MODEL_URL}' MODEL_PATH='${MODEL_PATH}' RETRIEVER_URL='${RETRIEVER_URL}' SANDBOX_URL='${SANDBOX_URL}' HORIZON_MODE=original_searches MIN_HORIZON='${MIN_HORIZON}' MAX_HORIZON='${MAX_HORIZON}' ANSWER_MAX_NEW_TOKENS='${ANSWER_MAX_NEW_TOKENS}' CONTEXT_LENGTH='${MODEL_CONTEXT_LENGTH}' bash '${SCRIPT_DIR}/run.sh' dry-run-full"
}

start_experiment() {
  require_prerequisites
  mkdir -p "${LOG_DIR}"
  check_all_services
  if tmux_exists "${EXPERIMENT_TMUX}"; then
    echo "Experiment tmux session already exists: ${EXPERIMENT_TMUX}" >&2
    return 1
  fi
  start_tmux "${EXPERIMENT_TMUX}" \
    "${activate_prefix} && set -o pipefail; RUN_DIR='${RUN_DIR}' MODEL_URL='${MODEL_URL}' MODEL_PATH='${MODEL_PATH}' RETRIEVER_URL='${RETRIEVER_URL}' SANDBOX_URL='${SANDBOX_URL}' HORIZON_MODE=original_searches MIN_HORIZON='${MIN_HORIZON}' MAX_HORIZON='${MAX_HORIZON}' ACTION_VALIDATION='${ACTION_VALIDATION}' ACTION_ATTEMPTS='${ACTION_ATTEMPTS}' ANSWER_MAX_NEW_TOKENS='${ANSWER_MAX_NEW_TOKENS}' CONTEXT_LENGTH='${MODEL_CONTEXT_LENGTH}' TEMPERATURE=0.0 N_ROLLOUTS=1 MAX_WORKERS=1 QUALITY_METRIC=f1 BOOTSTRAP_SAMPLES=10000 bash scripts/experiments/src_action_tree/run.sh full 2>&1 | tee -a '${LOG_DIR}/experiment.log'"
  echo "Attach with: tmux attach -t ${EXPERIMENT_TMUX}"
  echo "Result directory: ${RUN_DIR}"
}

show_status() {
  echo "Expected tmux sessions:"
  printf '  %s\n' "${RETRIEVER_TMUX}" "${SANDBOX_TMUX}" "${MODEL_TMUX}" "${EXPERIMENT_TMUX}"
  echo
  tmux ls || true
  echo
  curl -fsS --max-time 3 "${RETRIEVER_URL}/health" || true
  echo
  curl -fsS --max-time 3 "${MODEL_URL}/health" || true
  echo
  if [[ -f "${RUN_DIR}/run_status.json" ]]; then
    "${PYTHON_BIN}" -m json.tool "${RUN_DIR}/run_status.json"
  else
    echo "No final run_status.json yet: ${RUN_DIR}/run_status.json"
  fi
}

attach_tmux() {
  local target="${1:-experiment}"
  case "${target}" in
    retriever) tmux attach -t "${RETRIEVER_TMUX}" ;;
    sandbox) tmux attach -t "${SANDBOX_TMUX}" ;;
    model) tmux attach -t "${MODEL_TMUX}" ;;
    experiment|full) tmux attach -t "${EXPERIMENT_TMUX}" ;;
    *) echo "Unknown tmux alias: ${target}" >&2; exit 2 ;;
  esac
}

MODE="${1:-help}"
case "${MODE}" in
  plan) plan_full ;;
  services) start_services ;;
  experiment) start_experiment ;;
  all)
    start_services
    start_experiment
    ;;
  status) show_status ;;
  attach) attach_tmux "${2:-experiment}" ;;
  *)
    cat <<EOF
Usage: bash tmux_full.sh {plan|services|experiment|all|status|attach [retriever|sandbox|model|experiment]}

Recommended:
  bash tmux_full.sh plan       # no model/retriever calls; inspect scale
  bash tmux_full.sh all        # start three services, wait, then fourth experiment tmux
  bash tmux_full.sh status
  bash tmux_full.sh attach experiment

This script never kills an existing process or tmux session.
EOF
    ;;
esac
