#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

require_dir "${CORE_ENV_PREFIX}"
require_dir "${MODEL_ENV_PREFIX}"
require_file "${RETRIEVER_CONFIG}"
require_file "${INTERMEDIATE_DATA}"
require_file "${INDEX_PATH}"

LOG_DIR="${OUTPUT_ROOT}/tmux_logs_rebuild"
STATUS_DIR="${OUTPUT_ROOT}/tmux_status_rebuild"
mkdir -p "${LOG_DIR}" "${STATUS_DIR}"

MODEL_SESSION="${MODEL_SESSION:-src_rebuild_model}"
RETRIEVER_SESSION="${RETRIEVER_SESSION:-src_rebuild_retriever}"
SANDBOX_SESSION="${SANDBOX_SESSION:-src_rebuild_sandbox}"
PROBE_SESSION="${PROBE_SESSION:-src_rebuild_probe5}"
EXPERIMENT_SESSION="${EXPERIMENT_SESSION:-src_rebuild_experiment}"

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
      echo "${label} exited with code $(cat "${status_file}"). Last log lines:" >&2
      tail -n 80 "$(log_file_for "${session}")" >&2 || true
      return 1
    fi
    (( elapsed < WAIT_SECONDS )) || {
      echo "Timed out after ${WAIT_SECONDS}s waiting for ${label}." >&2
      tail -n 80 "$(log_file_for "${session}")" >&2 || true
      return 1
    }
    sleep 10
    elapsed=$((elapsed + 10))
  done
  echo "${label} ready: ${url}"
}

refuse_unmanaged_port() {
  local port="$1" session="$2" label="$3"
  if port_open "${port}" && ! has_session "${session}"; then
    echo "${label} port ${port} is occupied outside managed tmux session ${session}." >&2
    exit 2
  fi
}

start_model() {
  refuse_unmanaged_port "${MODEL_PORT}" "${MODEL_SESSION}" Model
  if ! port_open "${MODEL_PORT}"; then
    if ! has_session "${MODEL_SESSION}" || [[ -f "$(status_file_for "${MODEL_SESSION}")" ]]; then
      start_managed "${MODEL_SESSION}" "Qwen3-8B model" \
        env \
          CUDA_VISIBLE_DEVICES="${MODEL_GPUS}" \
          PATH="${MODEL_ENV_PREFIX}/bin:${PATH}" \
          HF_HOME="${HF_HOME}" HF_HUB_CACHE="${HF_HUB_CACHE}" \
          TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
          XDG_CACHE_HOME="${XDG_CACHE_HOME}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
          TMPDIR="${TMPDIR}" TOKENIZERS_PARALLELISM=false \
        "${MODEL_ENV_PREFIX}/bin/python" -u -m sglang.launch_server \
          --served-model-name Qwen3-8B \
          --model-path "${MODEL_PATH}" \
          --tp "${MODEL_TP}" --dp 1 \
          --context-length "${CONTEXT_LENGTH}" \
          --dtype bfloat16 \
          --host 0.0.0.0 --port "${MODEL_PORT}" \
          --trust-remote-code --enable-metrics \
          --attention-backend triton --sampling-backend pytorch \
          --disable-custom-all-reduce \
          --disable-overlap --disable-radix-cache --disable-cuda-graph \
          --mem-fraction-static "${MODEL_MEM_FRACTION}" \
          --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
          --max-running-requests "${MAX_RUNNING_REQUESTS}"
    fi
  fi
  wait_http "${MODEL_URL}/health" Model "${MODEL_SESSION}"
}

start_retriever() {
  refuse_unmanaged_port "${RETRIEVER_PORT}" "${RETRIEVER_SESSION}" Retriever
  if ! port_open "${RETRIEVER_PORT}"; then
    if ! has_session "${RETRIEVER_SESSION}" || [[ -f "$(status_file_for "${RETRIEVER_SESSION}")" ]]; then
      start_managed "${RETRIEVER_SESSION}" "CPU-FAISS retriever" \
        env \
          SRCAT_VMEM_GIB="${RETRIEVER_VMEM_GIB}" \
          CUDA_VISIBLE_DEVICES="${RETRIEVER_GPU}" \
          PYTHONPATH="${REPO_ROOT}/src" \
          HF_HOME="${HF_HOME}" HF_HUB_CACHE="${HF_HUB_CACHE}" \
          HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" HF_DATASETS_IN_MEMORY_MAX_SIZE=0 \
          TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
          XDG_CACHE_HOME="${XDG_CACHE_HOME}" TMPDIR="${TMPDIR}" \
          MALLOC_ARENA_MAX=2 TOKENIZERS_PARALLELISM=false \
          OMP_NUM_THREADS="${RETRIEVER_THREADS}" MKL_NUM_THREADS="${RETRIEVER_THREADS}" \
          OPENBLAS_NUM_THREADS="${RETRIEVER_THREADS}" NUMEXPR_NUM_THREADS="${RETRIEVER_THREADS}" \
        "${CORE_ENV_PREFIX}/bin/python" -u "${REPO_ROOT}/scripts/serving/retriever_serving.py" \
          --config "${RETRIEVER_CONFIG}" --num_retriever 1 --port "${RETRIEVER_PORT}"
    fi
  fi
  wait_http "${RETRIEVER_URL}/health" Retriever "${RETRIEVER_SESSION}"
}

start_sandbox() {
  refuse_unmanaged_port "${SANDBOX_PORT}" "${SANDBOX_SESSION}" Sandbox
  if ! port_open "${SANDBOX_PORT}"; then
    if ! has_session "${SANDBOX_SESSION}" || [[ -f "$(status_file_for "${SANDBOX_SESSION}")" ]]; then
      start_managed "${SANDBOX_SESSION}" "Sandbox" \
        env PYTHONPATH="${REPO_ROOT}/src" \
        "${CORE_ENV_PREFIX}/bin/python" -u "${REPO_ROOT}/scripts/serving/sandbox.py" \
          --port "${SANDBOX_PORT}"
    fi
  fi
  wait_http "${SANDBOX_URL}/docs" Sandbox "${SANDBOX_SESSION}"
}

services() {
  # Start the GPU model before the large CPU index so a retriever failure cannot
  # obscure model-start diagnostics. They use disjoint physical GPUs.
  start_model
  start_retriever
  start_sandbox
  "${CORE_ENV_PREFIX}/bin/python" "${SRC_ACTION_TREE_DIR}/probe_services.py" \
    --model_url "${MODEL_URL}" --model_path "${MODEL_PATH}" \
    --retriever_url "${RETRIEVER_URL}" --sandbox_url "${SANDBOX_URL}" --timeout 300
}

start_experiment() {
  local session="$1" label="$2" output_base="$3" limit="$4"
  start_managed "${session}" "${label}" \
    env \
      PATH="${CORE_ENV_PREFIX}/bin:${PATH}" \
      PYTHON_BIN="${CORE_ENV_PREFIX}/bin/python" \
      PYTHONPATH="${REPO_ROOT}/src" \
      HF_HOME="${HF_HOME}" HF_HUB_CACHE="${HF_HUB_CACHE}" \
      HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
      XDG_CACHE_HOME="${XDG_CACHE_HOME}" TMPDIR="${TMPDIR}" \
      OUTPUT_BASE="${output_base}" INTERMEDIATE_DATA="${INTERMEDIATE_DATA}" \
      MODEL_PATH="${MODEL_PATH}" MODEL_URL="${MODEL_URL}" \
      RETRIEVER_URL="${RETRIEVER_URL}" SANDBOX_URL="${SANDBOX_URL}" \
      LIMIT="${limit}" MIN_HORIZON="${MIN_HORIZON}" MAX_HORIZON="${MAX_HORIZON}" \
      TEMPERATURE="${TEMPERATURE}" CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
      SEARCH_MAX_NEW_TOKENS="${SEARCH_MAX_NEW_TOKENS}" \
      READ_MAX_NEW_TOKENS="${READ_MAX_NEW_TOKENS}" \
      ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS}" \
      ANSWER_REPAIR_MAX_NEW_TOKENS="${ANSWER_REPAIR_MAX_NEW_TOKENS}" \
      RESOURCE_CHECK_INTERVAL_SECONDS="${RESOURCE_CHECK_INTERVAL_SECONDS}" \
      RESOURCE_PROGRESS_EVERY="${RESOURCE_PROGRESS_EVERY}" \
      RESOURCE_MIN_AVAILABLE_RAM_GIB="${RESOURCE_MIN_AVAILABLE_RAM_GIB}" \
      RESOURCE_MAX_SWAP_USED_GIB="${RESOURCE_MAX_SWAP_USED_GIB}" \
      RESOURCE_MIN_FREE_DISK_GIB="${RESOURCE_MIN_FREE_DISK_GIB}" \
      RESOURCE_MAX_GPU_MEMORY_PERCENT="${RESOURCE_MAX_GPU_MEMORY_PERCENT}" \
      RESOURCE_STOP_ON_CRITICAL="${RESOURCE_STOP_ON_CRITICAL}" \
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      TOKENIZERS_PARALLELISM=false \
    bash "${SCRIPT_DIR}/experiment_with_monitor.sh" "${output_base}" all
}

probe5() {
  services
  start_experiment "${PROBE_SESSION}" "Qwen3-only five-question gate" "${PROBE_OUTPUT_BASE}" 5
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
    echo "Refusing full run: the five-question protocol audit has not passed at:" >&2
    echo "  ${PROBE_OUTPUT_BASE}/joint_analysis/protocol_audit.json" >&2
    echo "Run '$0 probe5', wait for completion, and inspect status. Override only with SKIP_PROBE_GATE=1." >&2
    exit 2
  fi
  # The completed probe pane is retained only for diagnostics. Remove that
  # completed wrapper before the formal run so the requested layout is exactly
  # four sessions: model, retriever, sandbox, experiment.
  if has_session "${PROBE_SESSION}" && [[ -f "$(status_file_for "${PROBE_SESSION}")" ]]; then
    tmux kill-session -t "${PROBE_SESSION}"
  fi
  start_experiment "${EXPERIMENT_SESSION}" "Qwen3-only full Bamboogle v6" "${FULL_OUTPUT_BASE}" 0
  echo "Follow: tail -f $(log_file_for "${EXPERIMENT_SESSION}")"
}

status() {
  echo "=== tmux ==="
  tmux ls 2>/dev/null || true
  echo "=== endpoints ==="
  for pair in "model ${MODEL_URL}/health" "retriever ${RETRIEVER_URL}/health" "sandbox ${SANDBOX_URL}/docs"; do
    read -r name url <<<"${pair}"
    if curl -fsS --max-time 3 "${url}" >/dev/null 2>&1; then echo "${name}: healthy"; else echo "${name}: unavailable"; fi
  done
  echo "=== managed exits ==="
  for session in "${MODEL_SESSION}" "${RETRIEVER_SESSION}" "${SANDBOX_SESSION}" "${PROBE_SESSION}" "${EXPERIMENT_SESSION}"; do
    file="$(status_file_for "${session}")"
    [[ -f "${file}" ]] && echo "${session}: exit $(cat "${file}")"
  done
  echo "=== GPU ==="
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  echo "=== RAM/disk ==="
  free -h
  df -h "${WORK_ROOT}" /
  echo "=== resource reminder flags ==="
  for output_base in "${PROBE_OUTPUT_BASE}" "${FULL_OUTPUT_BASE}"; do
    warning_file="${output_base}/RESOURCE_WARNING.flag"
    monitor_file="${output_base}/resource_monitor.jsonl"
    if [[ -f "${warning_file}" ]]; then
      echo "WARNING FLAG: ${warning_file}"
      tail -n 1 "${warning_file}"
    elif [[ -f "${monitor_file}" ]]; then
      echo "resource monitor OK: ${monitor_file}"
      tail -n 1 "${monitor_file}"
    fi
  done
  echo "=== full log tail ==="
  tail -n 50 "$(log_file_for "${EXPERIMENT_SESSION}")" 2>/dev/null || true
}

resources() {
  local output_base="${1:-${FULL_OUTPUT_BASE}}"
  local monitor_args=(
    --output-base "${output_base}"
    --work-root "${WORK_ROOT}"
    --interval-seconds "${RESOURCE_CHECK_INTERVAL_SECONDS}"
    --progress-every "${RESOURCE_PROGRESS_EVERY}"
    --min-available-ram-gib "${RESOURCE_MIN_AVAILABLE_RAM_GIB}"
    --max-swap-used-gib "${RESOURCE_MAX_SWAP_USED_GIB}"
    --min-free-disk-gib "${RESOURCE_MIN_FREE_DISK_GIB}"
    --max-gpu-memory-percent "${RESOURCE_MAX_GPU_MEMORY_PERCENT}"
    --once
  )
  "${CORE_ENV_PREFIX}/bin/python" -u "${SCRIPT_DIR}/resource_monitor.py" "${monitor_args[@]}"
}

case "${1:-help}" in
  services) services ;;
  probe5) probe5 ;;
  full|resume-full) full ;;
  status) status ;;
  resources) resources "${2:-}" ;;
  attach) tmux attach -t "${2:-${EXPERIMENT_SESSION}}" ;;
  *)
    echo "Usage: bash tmux_safe.sh {services|probe5|full|resume-full|status|resources [output_base]|attach [session]}" >&2
    exit 2
    ;;
esac
