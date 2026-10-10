#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Compatibility wrapper: this v4.3 package runs only on physical GPU2 and GPU3." >&2
exec bash "$SCRIPT_DIR/tmux_2gpu.sh" "$@"
