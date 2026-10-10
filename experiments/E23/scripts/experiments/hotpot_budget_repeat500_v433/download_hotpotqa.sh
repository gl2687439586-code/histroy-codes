#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/researcher/smartsearch_rebuild/data/hotpotqa}"
PYTHON_BIN="${PYTHON_BIN:-/home/researcher/smartsearch_rebuild/envs/src-core/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$DATA_DIR/hotpot_dev_distractor_v1.json"
PARQUET="$DATA_DIR/hotpot_dev_distractor_validation.parquet"
EXPECTED_PARQUET_SHA256="c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
MIRROR_URL="${HOTPOTQA_MIRROR_URL:-https://hf-mirror.com/datasets/hotpotqa/hotpot_qa/resolve/refs%2Fconvert%2Fparquet/distractor/validation/0000.parquet}"
HF_URL="${HOTPOTQA_HF_URL:-https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/refs%2Fconvert%2Fparquet/distractor/validation/0000.parquet}"
OFFICIAL_URL="${HOTPOTQA_URL:-http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json}"
mkdir -p "$DATA_DIR"

download_parquet() {
  local url="$1" label="$2"
  echo "Trying $label: $url"
  rm -f "$PARQUET.part"
  if ! curl -fL --retry 4 --retry-delay 3 --connect-timeout 15 --max-time 1200 \
      -o "$PARQUET.part" "$url"; then
    rm -f "$PARQUET.part"
    return 1
  fi
  actual="$(sha256sum "$PARQUET.part" | awk '{print $1}')"
  if [[ "$actual" != "$EXPECTED_PARQUET_SHA256" ]]; then
    echo "Rejecting $label: SHA256=$actual, expected=$EXPECTED_PARQUET_SHA256" >&2
    rm -f "$PARQUET.part"
    return 1
  fi
  mv "$PARQUET.part" "$PARQUET"
  return 0
}

parquet_to_json() {
  "$PYTHON_BIN" - "$PARQUET" "$SOURCE" <<'PY'
import json
import os
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
except Exception as exc:
    raise SystemExit(
        "pyarrow is required in src-core to convert the mirror parquet: " + str(exc)
    )

source = Path(sys.argv[1])
target = Path(sys.argv[2])
rows = pq.read_table(source).to_pylist()
if len(rows) != 7405:
    raise SystemExit(f"Expected 7405 validation rows, got {len(rows)}")
required = {"id", "question", "answer", "type", "level", "supporting_facts", "context"}
missing = required - set(rows[0])
if missing:
    raise SystemExit(f"Mirror parquet is missing required fields: {sorted(missing)}")
temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(rows, handle, ensure_ascii=False)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
print({"converted_rows": len(rows), "source": str(source), "output": str(target)})
PY
}

if [[ ! -s "$SOURCE" ]]; then
  if [[ -s "$PARQUET" ]]; then
    actual="$(sha256sum "$PARQUET" | awk '{print $1}')"
    if [[ "$actual" != "$EXPECTED_PARQUET_SHA256" ]]; then
      echo "Existing parquet has wrong SHA256; removing it." >&2
      rm -f "$PARQUET"
    fi
  fi

  if [[ ! -s "$PARQUET" ]]; then
    download_parquet "$MIRROR_URL" "HF-Mirror China parquet" \
      || download_parquet "$HF_URL" "Hugging Face parquet" \
      || true
  fi

  if [[ -s "$PARQUET" ]]; then
    parquet_to_json
  else
    echo "Parquet mirrors unavailable; falling back to the official JSON URL."
    curl -fL --retry 6 --retry-delay 5 --connect-timeout 30 --max-time 1800 \
      -o "$SOURCE.part" "$OFFICIAL_URL"
    mv "$SOURCE.part" "$SOURCE"
  fi
fi

"$PYTHON_BIN" "$SCRIPT_DIR/prepare_hotpotqa.py" \
  --input "$SOURCE" --output_dir "$DATA_DIR/prepared" --expected_rows 7405

sha256sum "$SOURCE" "$DATA_DIR/prepared/questions.no_labels.jsonl" \
  "$DATA_DIR/prepared/gold_labels.sealed.jsonl"
