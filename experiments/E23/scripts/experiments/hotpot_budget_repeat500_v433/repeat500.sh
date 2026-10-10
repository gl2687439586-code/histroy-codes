#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/home/researcher/smartsearch_rebuild}"
CORE_PYTHON="$WORKSPACE_ROOT/envs/src-core/bin/python"
[[ -x "$CORE_PYTHON" ]] || { echo "Missing Python 3: $CORE_PYTHON" >&2; exit 2; }
if [[ "$#" -eq 0 ]]; then set -- status; fi
exec "$CORE_PYTHON" -u "$SCRIPT_DIR/repeat500.py" "$@"
