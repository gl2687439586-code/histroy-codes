#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

output_base="${1:?output base is required}"
mode="${2:-all}"
mkdir -p "${output_base}"
# The child runner reads OUTPUT_BASE from the environment.  Export the
# positional value so foreground invocation and tmux invocation have exactly
# the same output-path semantics.
export OUTPUT_BASE="${output_base}"

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

AUTO_RESUME_TRANSIENT="${AUTO_RESUME_TRANSIENT:-1}"
AUTO_RESUME_MAX_ATTEMPTS="${AUTO_RESUME_MAX_ATTEMPTS:-20}"
AUTO_RESUME_BASE_DELAY_SECONDS="${AUTO_RESUME_BASE_DELAY_SECONDS:-15}"
AUTO_RESUME_MAX_DELAY_SECONDS="${AUTO_RESUME_MAX_DELAY_SECONDS:-120}"
retry_dir="${output_base}/auto_resume"
mkdir -p "${retry_dir}"

attempt=1
experiment_status=1
while true; do
  attempt_log="${retry_dir}/attempt_$(printf '%03d' "${attempt}").log"
  echo "[$(date -Is)] pipeline attempt ${attempt}/${AUTO_RESUME_MAX_ATTEMPTS} starting"
  set +e
  bash "${SRC_ACTION_TREE_DIR}/run_openrouter_three_conditions.sh" "${mode}" \
    2>&1 | tee "${attempt_log}"
  experiment_status=${PIPESTATUS[0]}
  set -e

  if [[ "${experiment_status}" == 0 ]]; then
    echo "[$(date -Is)] pipeline completed on attempt ${attempt}"
    break
  fi
  if [[ "${AUTO_RESUME_TRANSIENT}" != 1 ]]; then
    echo "[$(date -Is)] automatic transient resume is disabled" >&2
    break
  fi
  if (( attempt >= AUTO_RESUME_MAX_ATTEMPTS )); then
    echo "[$(date -Is)] automatic resume limit reached (${AUTO_RESUME_MAX_ATTEMPTS})" >&2
    break
  fi
  if ! grep -Eqi \
    'SSLError|UNEXPECTED_EOF|EOF occurred|Connection reset|RemoteDisconnected|ConnectTimeout|ReadTimeout|ConnectionError|Temporary failure in name resolution|NameResolutionError|HTTP (408|425|429|500|502|503|504)' \
    "${attempt_log}"; then
    echo "[$(date -Is)] non-transient failure detected; refusing automatic retry" >&2
    break
  fi

  delay=$((AUTO_RESUME_BASE_DELAY_SECONDS * (2 ** (attempt - 1))))
  if (( delay > AUTO_RESUME_MAX_DELAY_SECONDS )); then
    delay="${AUTO_RESUME_MAX_DELAY_SECONDS}"
  fi
  echo "[$(date -Is)] transient network failure detected; resuming safely in ${delay}s"
  sleep "${delay}"
  attempt=$((attempt + 1))
done
cleanup_monitor
trap - EXIT INT TERM
exit "${experiment_status}"
