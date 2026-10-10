#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/home/researcher/smartsearch_rebuild}"
CORE_PYTHON="$WORKSPACE_ROOT/envs/src-core/bin/python"
if [[ ! -x "$CORE_PYTHON" ]]; then
  echo "Missing Python 3 environment: $CORE_PYTHON" >&2
  exit 2
fi
# New terminals do not need inherited shell variables or an active conda prompt.
if [[ "$#" -eq 0 ]]; then set -- status; fi
exec "$CORE_PYTHON" -u "$SCRIPT_DIR/full7405.py" "$@"
