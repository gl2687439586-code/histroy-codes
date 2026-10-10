#!/usr/bin/env bash
set -euo pipefail

# Reuse the already-audited Polynomial snapshot implementation, changing only
# the frozen scheduler namespace.  The source script itself is never modified.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${POLYNOMIAL_EXPORT_SCRIPT:-$SCRIPT_DIR/export_audit_pandb_polynomial500.sh}"

[[ -s "$SOURCE" ]] || {
  echo "Missing source snapshot script: $SOURCE" >&2
  exit 1
}

TEMP_SCRIPT="$(mktemp "${TMPDIR:-/tmp}/pandb_cosine500_export.XXXXXX.sh")"
trap 'rm -f "$TEMP_SCRIPT"' EXIT

sed \
  -e 's/POLY_DIR/COS_DIR/g' \
  -e 's/POLYNOMIAL/COSINE/g' \
  -e 's/Polynomial/Cosine/g' \
  -e 's/polynomial/cosine/g' \
  -e 's/Cosine only; read-only snapshot while Cosine continued running/Cosine only; read-only snapshot after completion; Polynomial excluded/g' \
  -e 's/本报告只审计已经冻结的 Cosine；生成时未加载金标。正在运行的 Cosine 未被读取、停止或修改。/本报告只审计已经冻结的 Cosine；生成时未加载金标。Polynomial 未被重新打包或修改。/g' \
  "$SOURCE" > "$TEMP_SCRIPT"

chmod 700 "$TEMP_SCRIPT"
bash -n "$TEMP_SCRIPT"
exec bash "$TEMP_SCRIPT"
