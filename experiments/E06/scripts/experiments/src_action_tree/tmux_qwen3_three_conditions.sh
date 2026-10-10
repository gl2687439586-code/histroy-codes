#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/researcher/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-smartsearch}"
MODEL_PATH="${MODEL_PATH:-/home/researcher/models/Qwen3-8B}"
MODEL_PORT="${MODEL_PORT:-9001}"
MODEL_URL="${MODEL_URL:-http://127.0.0.1:${MODEL_PORT}}"
RETRIEVER_PORT="${RETRIEVER_PORT:-6008}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:${RETRIEVER_PORT}}"
SANDBOX_PORT="${SANDBOX_PORT:-6006}"
SANDBOX_URL="${SANDBOX_URL:-http://127.0.0.1:${SANDBOX_PORT}}"
RETRIEVER_CONFIG="${RETRIEVER_CONFIG:-${REPO_ROOT}/scripts/serving/retriever_config_src_action_tree.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/researcher/smartsearch_workspace/outputs/src_action_tree/qwen3_only_three_conditions_v6}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-/home/researcher/smartsearch_workspace/outputs/full_bamboogle_100w_timeout120/bamboogle_2026_06_25_15_04_SmartSearch-3B_100w_timeout120}"
INTERMEDIATE_DATA="${INTERMEDIATE_DATA:-${SOURCE_RUN_DIR}/intermediate_data.json}"
LIMIT="${LIMIT:-0}"
SAMPLE_IDS="${SAMPLE_IDS:-}"
MIN_HORIZON="${MIN_HORIZON:-3}"
MAX_HORIZON="${MAX_HORIZON:-6}"
LOG_DIR="${OUTPUT_BASE}/tmux_logs"
GPU_ID="${GPU_ID:-0}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
WAIT_SECONDS="${WAIT_SECONDS:-1800}"

RETRIEVER_TMUX="${RETRIEVER_TMUX:-q3src_retriever}"
SANDBOX_TMUX="${SANDBOX_TMUX:-q3src_sandbox}"
MODEL_TMUX="${MODEL_TMUX:-q3src_model}"
EXPERIMENT_TMUX="${EXPERIMENT_TMUX:-q3src_experiment}"

activate="source '${CONDA_SH}' && conda activate '${CONDA_ENV}' && cd '${REPO_ROOT}' && export PYTHONUNBUFFERED=1"

exists() { tmux has-session -t "$1" 2>/dev/null; }
start() {
  local name="$1" command="$2"
  tmux new-session -d -s "${name}" bash -l
  tmux send-keys -t "${name}" -l "${command}; status=\$?; echo '[qwen3-src] exit' \${status}; exit \${status}"
  tmux send-keys -t "${name}" Enter
  echo "Started tmux: ${name}"
}
refuse_unmanaged_port() {
  local port="$1" session="$2" label="$3"
  if (echo >/dev/tcp/127.0.0.1/"${port}") >/dev/null 2>&1 && ! exists "${session}"; then
    echo "${label} port ${port} is occupied outside expected tmux ${session}; refusing ambiguous reuse." >&2
    exit 2
  fi
}
wait_http() {
  local url="$1" label="$2" session="$3" elapsed=0
  until curl -fsS --max-time 5 "${url}" >/dev/null 2>&1; do
    exists "${session}" || { echo "${label} session exited" >&2; return 1; }
    (( elapsed < WAIT_SECONDS )) || { echo "Timed out waiting for ${label}" >&2; return 1; }
    sleep 10; elapsed=$((elapsed+10))
  done
  echo "${label} ready: ${url}"
}

services() {
  mkdir -p "${LOG_DIR}"
  refuse_unmanaged_port "${RETRIEVER_PORT}" "${RETRIEVER_TMUX}" Retriever
  if ! exists "${RETRIEVER_TMUX}"; then
    start "${RETRIEVER_TMUX}" "${activate} && set -o pipefail; CUDA_VISIBLE_DEVICES='${GPU_ID}' python -u scripts/serving/retriever_serving.py --config '${RETRIEVER_CONFIG}' --num_retriever 1 --port '${RETRIEVER_PORT}' 2>&1 | tee -a '${LOG_DIR}/retriever.log'"
  fi
  wait_http "${RETRIEVER_URL}/health" Retriever "${RETRIEVER_TMUX}"
  refuse_unmanaged_port "${SANDBOX_PORT}" "${SANDBOX_TMUX}" Sandbox
  if ! exists "${SANDBOX_TMUX}"; then
    start "${SANDBOX_TMUX}" "${activate} && set -o pipefail; python -u scripts/serving/sandbox.py --port '${SANDBOX_PORT}' 2>&1 | tee -a '${LOG_DIR}/sandbox.log'"
  fi
  wait_http "${SANDBOX_URL}/docs" Sandbox "${SANDBOX_TMUX}"
  refuse_unmanaged_port "${MODEL_PORT}" "${MODEL_TMUX}" Qwen3
  if ! exists "${MODEL_TMUX}"; then
    start "${MODEL_TMUX}" "${activate} && set -o pipefail; CUDA_VISIBLE_DEVICES='${GPU_ID}' python -u -m sglang.launch_server --served-model-name Qwen3-8B --model-path '${MODEL_PATH}' --tp 1 --dp 1 --context-length '${CONTEXT_LENGTH}' --enable-metrics --dtype bfloat16 --host 0.0.0.0 --port '${MODEL_PORT}' --trust-remote-code --disable-overlap --disable-radix-cache --mem-fraction-static '${MEM_FRACTION}' 2>&1 | tee -a '${LOG_DIR}/model.log'"
  fi
  wait_http "${MODEL_URL}/health" Qwen3 "${MODEL_TMUX}"
  python "${SCRIPT_DIR}/probe_services.py" --model_url "${MODEL_URL}" --model_path "${MODEL_PATH}" --retriever_url "${RETRIEVER_URL}" --sandbox_url "${SANDBOX_URL}" --timeout 300
}

experiment() {
  exists "${EXPERIMENT_TMUX}" && { echo "Experiment session exists: ${EXPERIMENT_TMUX}" >&2; exit 2; }
  mkdir -p "${LOG_DIR}"
  start "${EXPERIMENT_TMUX}" "${activate} && set -o pipefail; OUTPUT_BASE='${OUTPUT_BASE}' SOURCE_RUN_DIR='${SOURCE_RUN_DIR}' INTERMEDIATE_DATA='${INTERMEDIATE_DATA}' LIMIT='${LIMIT}' SAMPLE_IDS='${SAMPLE_IDS}' MIN_HORIZON='${MIN_HORIZON}' MAX_HORIZON='${MAX_HORIZON}' MODEL_PATH='${MODEL_PATH}' MODEL_URL='${MODEL_URL}' RETRIEVER_URL='${RETRIEVER_URL}' SANDBOX_URL='${SANDBOX_URL}' SEARCH_MAX_NEW_TOKENS='${SEARCH_MAX_NEW_TOKENS:-512}' READ_MAX_NEW_TOKENS='${READ_MAX_NEW_TOKENS:-768}' ANSWER_MAX_NEW_TOKENS='${ANSWER_MAX_NEW_TOKENS:-2048}' ANSWER_REPAIR_MAX_NEW_TOKENS='${ANSWER_REPAIR_MAX_NEW_TOKENS:-96}' CONTEXT_LENGTH='${CONTEXT_LENGTH}' bash '${SCRIPT_DIR}/run_qwen3_three_conditions.sh' all 2>&1 | tee -a '${LOG_DIR}/experiment.log'"
  echo "Attach: tmux attach -t ${EXPERIMENT_TMUX}"
  echo "Log: ${LOG_DIR}/experiment.log"
}

status() {
  tmux ls || true
  echo "Log tail:"
  tail -n 40 "${LOG_DIR}/experiment.log" 2>/dev/null || true
}

case "${1:-help}" in
  services) services ;;
  experiment) experiment ;;
  all) services; experiment ;;
  status) status ;;
  attach) tmux attach -t "${2:-${EXPERIMENT_TMUX}}" ;;
  *) echo "Usage: bash tmux_qwen3_three_conditions.sh {services|experiment|all|status|attach [session]}" >&2; exit 2 ;;
esac
