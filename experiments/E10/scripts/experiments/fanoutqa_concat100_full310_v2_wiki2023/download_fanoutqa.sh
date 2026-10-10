#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/researcher/smartsearch_rebuild/envs/src-core/bin/python}"
DATA_DIR="${DATA_DIR:-/home/researcher/smartsearch_rebuild/data/fanoutqa}"
RAW_DIR="$DATA_DIR/raw"
RAW_FILE="$RAW_DIR/fanout-final-dev.json"
mkdir -p "$RAW_DIR" "$DATA_DIR/prepared"

validate() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
from pathlib import Path
x = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert isinstance(x, list) and len(x) == 310, len(x) if isinstance(x, list) else type(x)
assert all(row.get("id") and row.get("question") and "answer" in row for row in x)
print("FanOutQA dev validation passed:", len(x))
PY
}

copy_from_installed_package() {
  "$PYTHON_BIN" - "$RAW_FILE" <<'PY'
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

try:
    import fanoutqa
except Exception as exc:
    raise SystemExit(f"fanoutqa package unavailable: {exc}")

version = importlib.metadata.version("fanoutqa")
major, minor, *_ = [int(part) for part in version.split(".") if part.isdigit()]
if (major, minor) < (1, 3):
    raise SystemExit(f"fanoutqa>=1.3.0 required; installed={version}")
source = Path(fanoutqa.__file__).resolve().parent / "data" / "fanout-final-dev.json"
data = json.loads(source.read_text(encoding="utf-8"))
if not isinstance(data, list) or len(data) != 310:
    raise SystemExit(f"Unexpected bundled dev file: {source}, rows={len(data)}")
destination = Path(sys.argv[1])
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, destination)
print(f"Copied official bundled FanOutQA dev {version}: {source} -> {destination}")
PY
}

if [[ ! -s "$RAW_FILE" ]] || ! validate "$RAW_FILE" >/dev/null 2>&1; then
  echo "Trying the official FanOutQA >=1.3.0 package already installed from domestic PyPI."
  copy_from_installed_package || true
fi

if [[ ! -s "$RAW_FILE" ]] || ! validate "$RAW_FILE" >/dev/null 2>&1; then
  rm -f "$RAW_FILE.tmp"
  urls=(
    "https://ghfast.top/https://raw.githubusercontent.com/zhudotexe/fanoutqa/main/fanoutqa/data/fanout-final-dev.json"
    "https://hf-mirror.com/datasets/zhuexe/fanoutqa/resolve/main/fanout-final-dev.json"
    "https://huggingface.co/datasets/zhuexe/fanoutqa/resolve/main/fanout-final-dev.json?download=true"
    "https://raw.githubusercontent.com/zhudotexe/fanoutqa/main/fanoutqa/data/fanout-final-dev.json"
  )
  downloaded=0
  for url in "${urls[@]}"; do
    echo "Trying: $url"
    if curl -fL --connect-timeout 30 --max-time 600 --retry 3 \
      --retry-delay 2 "$url" -o "$RAW_FILE.tmp" && \
      validate "$RAW_FILE.tmp" >/dev/null 2>&1; then
      mv "$RAW_FILE.tmp" "$RAW_FILE"
      downloaded=1
      break
    fi
    rm -f "$RAW_FILE.tmp"
  done
  [[ "$downloaded" == "1" ]] || {
    echo "Unable to download corrected FanOutQA dev. Manually place it at: $RAW_FILE" >&2
    exit 2
  }
fi

validate "$RAW_FILE"
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_fanoutqa.py" \
  --input "$RAW_FILE" --output_dir "$DATA_DIR/prepared" --expected_count 310
