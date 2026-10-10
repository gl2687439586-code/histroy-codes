#!/usr/bin/env bash
set -euo pipefail

NEW_SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ACTION_TREE_DIR="$(cd "${NEW_SERVER_DIR}/.." && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SRC_ACTION_TREE_DIR}/../../.." && pwd)"

if [[ -f "${NEW_SERVER_DIR}/server.env" ]]; then
  # shellcheck disable=SC1091
  source "${NEW_SERVER_DIR}/server.env"
else
  # shellcheck disable=SC1091
  source "${NEW_SERVER_DIR}/server.env.example"
fi

# When the bundle is unpacked into a clone at a non-default path, prefer that
# actual clone unless REPO_ROOT was explicitly exported by the caller.
if [[ -z "${REPO_ROOT_EXPLICIT:-}" && -d "${DEFAULT_REPO_ROOT}/scripts/serving" ]]; then
  REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
fi

if [[ -f "${NEW_SERVER_DIR}/resolved.env" ]]; then
  # shellcheck disable=SC1091
  source "${NEW_SERVER_DIR}/resolved.env"
fi

# Backward-compatible defaults when an older server.env is reused.
RESOURCE_CHECK_INTERVAL_SECONDS="${RESOURCE_CHECK_INTERVAL_SECONDS:-300}"
RESOURCE_PROGRESS_EVERY="${RESOURCE_PROGRESS_EVERY:-10}"
RESOURCE_MIN_AVAILABLE_RAM_GIB="${RESOURCE_MIN_AVAILABLE_RAM_GIB:-64}"
RESOURCE_MAX_SWAP_USED_GIB="${RESOURCE_MAX_SWAP_USED_GIB:-16}"
RESOURCE_MIN_FREE_DISK_GIB="${RESOURCE_MIN_FREE_DISK_GIB:-150}"
RESOURCE_MAX_GPU_MEMORY_PERCENT="${RESOURCE_MAX_GPU_MEMORY_PERCENT:-95}"
RESOURCE_STOP_ON_CRITICAL="${RESOURCE_STOP_ON_CRITICAL:-0}"

export HF_HOME="${WORK_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export MODELSCOPE_CACHE="${WORK_ROOT}/cache/modelscope"
export XDG_CACHE_HOME="${WORK_ROOT}/cache/xdg"
export PIP_CACHE_DIR="${WORK_ROOT}/cache/pip"
export CONDA_PKGS_DIRS="${WORK_ROOT}/cache/conda/pkgs"
export CONDA_ENVS_PATH="${WORK_ROOT}/envs"
export TRITON_CACHE_DIR="${WORK_ROOT}/cache/triton"
export FLASHINFER_WORKSPACE_DIR="${WORK_ROOT}/cache/flashinfer"
export TMPDIR="${WORK_ROOT}/tmp"
export HF_ENDPOINT
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_URL="${MODEL_URL:-http://127.0.0.1:${MODEL_PORT}}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:${RETRIEVER_PORT}}"
SANDBOX_URL="${SANDBOX_URL:-http://127.0.0.1:${SANDBOX_PORT}}"
RETRIEVER_CONFIG="${RETRIEVER_CONFIG:-${REPO_ROOT}/scripts/serving/retriever_config_src_action_tree.yaml}"

mkdir -p "${WORK_ROOT}" "${HF_HOME}" "${HF_DATASETS_CACHE}" \
  "${MODELSCOPE_CACHE}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" \
  "${CONDA_PKGS_DIRS}" "${TRITON_CACHE_DIR}" \
  "${FLASHINFER_WORKSPACE_DIR}" "${TMPDIR}" "${OUTPUT_ROOT}"

core_python() { "${CORE_ENV_PREFIX}/bin/python" "$@"; }
model_python() { "${MODEL_ENV_PREFIX}/bin/python" "$@"; }

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing required directory: $1" >&2; exit 2; }
}
