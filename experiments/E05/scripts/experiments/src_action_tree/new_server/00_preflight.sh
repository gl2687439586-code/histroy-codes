#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

failures=0
warn() { echo "[WARN] $*"; }
fail() { echo "[FAIL] $*" >&2; failures=$((failures + 1)); }
pass() { echo "[ OK ] $*"; }

echo "Work root: ${WORK_ROOT}"
for command_name in git curl wget unzip tmux nvidia-smi df awk sed sha256sum; do
  if command -v "${command_name}" >/dev/null 2>&1; then
    pass "command ${command_name}"
  else
    fail "missing command ${command_name} (Ubuntu package may be required)"
  fi
done

if [[ "${WORK_ROOT}" != /home/${USER}/* ]]; then
  fail "WORK_ROOT must stay under /home/${USER} on this configuration: ${WORK_ROOT}"
elif [[ -w "${WORK_ROOT}" ]]; then
  pass "WORK_ROOT is writable"
else
  fail "WORK_ROOT is not writable: ${WORK_ROOT}"
fi

disk_kib=$(df -Pk "${WORK_ROOT}" | awk 'NR==2 {print $4}')
disk_gib=$((disk_kib / 1024 / 1024))
if (( disk_gib >= 250 )); then
  pass "data filesystem free space ${disk_gib}GiB (minimum 250GiB)"
else
  fail "only ${disk_gib}GiB free under WORK_ROOT; model/index/extraction need at least 250GiB"
fi

root_kib=$(df -Pk / | awk 'NR==2 {print $4}')
root_gib=$((root_kib / 1024 / 1024))
if (( root_gib < 20 )); then
  warn "root filesystem has only ${root_gib}GiB free; this is acceptable only when /home is a separate filesystem and every cache is redirected below WORK_ROOT"
else
  pass "root filesystem free space ${root_gib}GiB"
fi

home_device=$(df -P /home | awk 'NR==2 {print $1}')
root_device=$(df -P / | awk 'NR==2 {print $1}')
if [[ "${home_device}" == "${root_device}" ]]; then
  fail "/home and / are on the same filesystem (${home_device}); the root filesystem is too small for this experiment"
else
  pass "/home is separate from / (${home_device} versus ${root_device})"
fi

if mountpoint -q /boot 2>/dev/null; then
  boot_kib=$(df -Pk /boot | awk 'NR==2 {print $4}')
  boot_mib=$((boot_kib / 1024))
  if (( boot_mib < 1024 )); then
    warn "/boot has only ${boot_mib}MiB free. Do not update the kernel, NVIDIA driver, or CUDA toolkit; ask the administrator to remove obsolete kernels first."
  fi
fi

ram_kib=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
ram_gib=$((ram_kib / 1024 / 1024))
if (( ram_gib >= 384 )); then
  pass "RAM ${ram_gib}GiB"
else
  fail "RAM ${ram_gib}GiB is below the safe 384GiB threshold for the full CPU FAISS index"
fi

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( gpu_count >= 4 )); then
  pass "GPU count ${gpu_count}"
else
  fail "need four visible GPUs for the recommended safe allocation; found ${gpu_count}"
fi

while IFS=',' read -r gpu_index total_mib used_mib gpu_name; do
  gpu_index="${gpu_index// /}"
  total_mib="${total_mib// /}"
  used_mib="${used_mib// /}"
  if (( total_mib < 23000 )); then
    fail "GPU ${gpu_index} has only ${total_mib}MiB: ${gpu_name}"
  elif (( gpu_index <= 2 && used_mib > 1024 )); then
    fail "GPU ${gpu_index} already uses ${used_mib}MiB; GPUs 0,1,2 must be free before service launch"
  else
    pass "GPU ${gpu_index}: ${gpu_name}, ${used_mib}/${total_mib}MiB"
  fi
done < <(nvidia-smi --query-gpu=index,memory.total,memory.used,name --format=csv,noheader,nounits)

for port in "${SANDBOX_PORT}" "${RETRIEVER_PORT}" "${MODEL_PORT}"; do
  if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :${port}" | grep -q LISTEN; then
    fail "port ${port} is already listening"
  else
    pass "port ${port} is free"
  fi
done

if (( failures > 0 )); then
  echo "PRE-FLIGHT FAILED: ${failures} blocking issue(s)." >&2
  exit 2
fi
echo "PRE-FLIGHT PASSED. No model, index, or service was loaded."
