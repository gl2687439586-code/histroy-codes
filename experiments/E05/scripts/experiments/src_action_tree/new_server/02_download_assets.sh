#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"
require_dir "${CORE_ENV_PREFIX}"

mkdir -p "${WORK_ROOT}/models" "${RETRIEVAL_ROOT}" "${DATA_ROOT}/bamboogle"

download_hf_repo() {
  local repo_id="$1" destination="$2"
  if [[ -s "${destination}/config.json" ]]; then
    echo "Verifying/resuming existing download: ${destination}"
  else
    echo "Downloading ${repo_id} -> ${destination}"
  fi
  # hf download is cache-aware: rerunning verifies the complete repository and
  # downloads only missing/incomplete files.  A config.json alone is not a
  # sufficient completeness check after an interrupted multi-shard download.
  "${CORE_ENV_PREFIX}/bin/hf" download "${repo_id}" \
    --local-dir "${destination}" --max-workers 4
}

download_hf_repo "Qwen/Qwen3-8B" "${MODEL_PATH}"
download_hf_repo "intfloat/e5-base-v2" "${RETRIEVER_MODEL_PATH}"

download_file() {
  local url="$1" destination="$2"
  local curl_args=(-fL --retry 8 --retry-delay 10 -C -)
  mkdir -p "$(dirname "${destination}")"
  echo "Downloading/resuming ${destination}"
  # --retry-all-errors was added in curl 7.71.  Older Ubuntu installations do
  # not recognize it, while the remaining retry and resume options are enough.
  if curl --help all 2>/dev/null | grep -q -- '--retry-all-errors'; then
    curl_args+=(--retry-all-errors)
  fi
  curl "${curl_args[@]}" "${url}" -o "${destination}"
}

download_file \
  "https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/resolve/master/retrieval_corpus/wiki18_100w.jsonl" \
  "${CORPUS_PATH}"
download_file \
  "https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/resolve/master/retrieval_corpus/wiki18_100w_e5_index.zip" \
  "${INDEX_ARCHIVE}"
download_file \
  "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/bamboogle/test.jsonl" \
  "${BAMBOOGLE_JSONL}"

corpus_bytes=$(stat -c %s "${CORPUS_PATH}")
archive_bytes=$(stat -c %s "${INDEX_ARCHIVE}")
(( corpus_bytes > 10000000000 )) || { echo "Corpus is unexpectedly small: ${corpus_bytes} bytes" >&2; exit 2; }
(( archive_bytes > 10000000000 )) || { echo "Index archive is unexpectedly small: ${archive_bytes} bytes" >&2; exit 2; }

if ! find "${INDEX_DIR}" -type f -name '*.index' -print -quit 2>/dev/null | grep -q .; then
  mkdir -p "${INDEX_DIR}"
  echo "Extracting the prebuilt E5 index. This may take several minutes."
  unzip -q -n "${INDEX_ARCHIVE}" -d "${INDEX_DIR}"
else
  echo "Index is already extracted under ${INDEX_DIR}"
fi

prepare_args=(
  --dataset_jsonl "${BAMBOOGLE_JSONL}"
  --output "${INTERMEDIATE_DATA}"
)
if [[ -n "${PAPER_INTERMEDIATE_DATA}" ]]; then
  prepare_args+=(--paper_intermediate "${PAPER_INTERMEDIATE_DATA}")
fi
"${CORE_ENV_PREFIX}/bin/python" "${SCRIPT_DIR}/prepare_bamboogle_input.py" "${prepare_args[@]}"

echo "ASSET DOWNLOAD PASSED"
