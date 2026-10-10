#!/usr/bin/env bash
set -uo pipefail

label="$1"
log_file="$2"
status_file="$3"
shift 3
mkdir -p "$(dirname "${log_file}")" "$(dirname "${status_file}")"

echo "[$(date -Is)] ${label} starting" | tee -a "${log_file}"
if [[ -z "${SRCAT_VMEM_GIB:-}" ]]; then
  for argument in "$@"; do
    case "${argument}" in
      SRCAT_VMEM_GIB=*) SRCAT_VMEM_GIB="${argument#SRCAT_VMEM_GIB=}"; break ;;
    esac
  done
fi
if [[ -n "${SRCAT_VMEM_GIB:-}" ]]; then
  limit_kib=$((SRCAT_VMEM_GIB * 1024 * 1024))
  ulimit -v "${limit_kib}"
  echo "[$(date -Is)] virtual-memory ceiling=${SRCAT_VMEM_GIB}GiB" | tee -a "${log_file}"
fi

set +e
"$@" 2>&1 | tee -a "${log_file}"
exit_code=${PIPESTATUS[0]}
set -e

temporary="${status_file}.tmp.$$"
printf '%s\n' "${exit_code}" > "${temporary}"
mv -f "${temporary}" "${status_file}"
echo "[$(date -Is)] ${label} exited with code ${exit_code}" | tee -a "${log_file}"
echo "The tmux pane is intentionally kept alive for inspection."
while true; do sleep 3600; done
