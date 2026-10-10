#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/home/researcher/smartsearch_rebuild}"
ASSET_DIR="${WIKI2023_ASSET_DIR:-$WORK_ROOT/retrieval/fanoutqa_wiki2023}"
ZIM_URL="${WIKI2023_ZIM_URL:-https://datasets.mechanus.zhu.codes/fanoutqa/wikipedia_en_all_nopic_2023-09.zim}"
ZIM_PATH="${WIKI2023_ZIM_PATH:-$ASSET_DIR/wikipedia_en_all_nopic_2023-09.zim}"
KIWIX_VERSION="${KIWIX_VERSION:-3.8.2}"
KIWIX_ARCHIVE="$ASSET_DIR/kiwix-tools_linux-x86_64-${KIWIX_VERSION}.tar.gz"
KIWIX_DIR="$ASSET_DIR/kiwix-tools"
KIWIX_URL="${KIWIX_URL:-https://download.kiwix.org/release/kiwix-tools/kiwix-tools_linux-x86_64-${KIWIX_VERSION}.tar.gz}"
MIN_FREE_GIB="${MIN_FREE_GIB:-80}"

mkdir -p "$ASSET_DIR"

available_bytes="$(df -PB1 "$ASSET_DIR" | awk 'NR==2 {print $4}')"
required_bytes="$((MIN_FREE_GIB * 1024 * 1024 * 1024))"
if (( available_bytes < required_bytes )); then
  echo "At least ${MIN_FREE_GIB}GiB free is required under $ASSET_DIR" >&2
  df -h "$ASSET_DIR" >&2
  exit 2
fi

download_resume() {
  local url="$1" destination="$2" partial="${2}.part"
  if [[ -s "$destination" ]]; then
    echo "Already present: $destination"
    return 0
  fi
  echo "Downloading with resume: $url"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c --continue=true --max-connection-per-server=8 --split=8 \
      --min-split-size=16M --file-allocation=none \
      --dir="$(dirname "$partial")" --out="$(basename "$partial")" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --continue --timeout=60 --read-timeout=60 --tries=0 \
      --output-document="$partial" "$url"
  else
    curl -fL --connect-timeout 60 --retry 20 --retry-delay 10 \
      --retry-all-errors -C - "$url" -o "$partial"
  fi
  mv "$partial" "$destination"
}

download_resume "$ZIM_URL" "$ZIM_PATH"
zim_bytes="$(stat -c %s "$ZIM_PATH")"
(( zim_bytes > 10000000000 )) || {
  echo "ZIM is unexpectedly small: $zim_bytes bytes ($ZIM_PATH)" >&2
  exit 2
}

if [[ ! -x "$KIWIX_DIR/kiwix-serve" ]]; then
  download_resume "$KIWIX_URL" "$KIWIX_ARCHIVE"
  temporary="$ASSET_DIR/kiwix-tools.extracting"
  mkdir -p "$temporary"
  tar -xzf "$KIWIX_ARCHIVE" -C "$temporary"
  extracted="$temporary/kiwix-tools_linux-x86_64-${KIWIX_VERSION}"
  [[ -x "$extracted/kiwix-serve" ]] || {
    echo "kiwix-serve was not found after extraction" >&2
    exit 2
  }
  mv "$extracted" "$KIWIX_DIR"
fi

"$KIWIX_DIR/kiwix-serve" --version
echo "WIKI2023 ASSETS READY"
echo "ZIM=$ZIM_PATH"
echo "KIWIX_SERVE=$KIWIX_DIR/kiwix-serve"
du -h "$ZIM_PATH" "$KIWIX_DIR/kiwix-serve"
