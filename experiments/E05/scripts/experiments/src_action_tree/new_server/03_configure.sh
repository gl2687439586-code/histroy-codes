#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

require_dir "${REPO_ROOT}"
require_dir "${MODEL_PATH}"
require_dir "${RETRIEVER_MODEL_PATH}"
require_file "${CORPUS_PATH}"
require_file "${INTERMEDIATE_DATA}"

mapfile -t preferred < <(find "${INDEX_DIR}" -type f -name 'e5_flat_inner.index' -print)
if (( ${#preferred[@]} == 1 )); then
  INDEX_PATH="${preferred[0]}"
else
  mapfile -t all_indices < <(find "${INDEX_DIR}" -type f -name '*.index' -print)
  if (( ${#all_indices[@]} != 1 )); then
    printf 'Could not select one index. Candidates:\n%s\n' "${all_indices[*]:-none}" >&2
    exit 2
  fi
  INDEX_PATH="${all_indices[0]}"
fi

"${CORE_ENV_PREFIX}/bin/python" "${SCRIPT_DIR}/write_retriever_config.py" \
  --model "${RETRIEVER_MODEL_PATH}" \
  --index "${INDEX_PATH}" \
  --corpus "${CORPUS_PATH}" \
  --output "${RETRIEVER_CONFIG}"

{
  printf 'INDEX_PATH=%q\n' "${INDEX_PATH}"
  printf 'RETRIEVER_CONFIG=%q\n' "${RETRIEVER_CONFIG}"
} > "${SCRIPT_DIR}/resolved.env"

echo "Resolved index: ${INDEX_PATH}"
if ! du -h "${INDEX_PATH}" "${CORPUS_PATH}" "${MODEL_PATH}" | tail -n 10; then
  echo "[WARN] Size display reported an error; required files were already resolved and will be validated in the next step." >&2
fi
echo "CONFIGURATION PASSED"
