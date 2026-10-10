#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

output_base="${1:?output base is required}"
mode="${2:-all}"
mkdir -p "${output_base}"

monitor_args=(
  --output-base "${output_base}"
  --work-root "${WORK_ROOT}"
  --parent-pid "$$"
  --interval-seconds "${RESOURCE_CHECK_INTERVAL_SECONDS}"
  --progress-every "${RESOURCE_PROGRESS_EVERY}"
  --min-available-ram-gib "${RESOURCE_MIN_AVAILABLE_RAM_GIB}"
  --max-swap-used-gib "${RESOURCE_MAX_SWAP_USED_GIB}"
  --min-free-disk-gib "${RESOURCE_MIN_FREE_DISK_GIB}"
  --max-gpu-memory-percent "${RESOURCE_MAX_GPU_MEMORY_PERCENT}"
)
if [[ "${RESOURCE_STOP_ON_CRITICAL}" == 1 ]]; then
  monitor_args+=(--stop-on-critical)
fi

"${CORE_ENV_PREFIX}/bin/python" -u "${SCRIPT_DIR}/resource_monitor.py" "${monitor_args[@]}" &
monitor_pid=$!
cleanup_monitor() {
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
}
trap cleanup_monitor EXIT INT TERM

set +e
bash "${SRC_ACTION_TREE_DIR}/run_qwen3_three_conditions.sh" "${mode}"
experiment_status=$?
set -e
cleanup_monitor
trap - EXIT INT TERM
exit "${experiment_status}"
