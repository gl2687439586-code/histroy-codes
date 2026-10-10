#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

require_file "${RETRIEVER_CONFIG}"
require_file "${INTERMEDIATE_DATA}"
require_file "${MODEL_PATH}/config.json"
require_file "${RETRIEVER_MODEL_PATH}/config.json"
require_file "${INDEX_PATH}"

echo "Static model/data checks (the 60GB FAISS index is not loaded here)."
"${CORE_ENV_PREFIX}/bin/python" - "${MODEL_PATH}" "${INTERMEDIATE_DATA}" "${RETRIEVER_CONFIG}" <<'PY'
import json, sys, yaml
from pathlib import Path
model_path, data_path, config_path = map(Path, sys.argv[1:])
model = json.loads((model_path / "config.json").read_text())
assert model.get("model_type") == "qwen3", model.get("model_type")
assert model.get("architectures") == ["Qwen3ForCausalLM"], model.get("architectures")
rows = json.loads(data_path.read_text())
assert len(rows) == 125, len(rows)
assert len({str(row["id"]) for row in rows}) == 125
config = yaml.safe_load(config_path.read_text())
assert config["faiss_gpu"] is False
assert config["gpu_id"] is None
print({"model_type": model["model_type"], "samples": len(rows), "faiss_gpu": config["faiss_gpu"]})
PY

"${CORE_ENV_PREFIX}/bin/python" "${SRC_ACTION_TREE_DIR}/self_test_qwen3_protocol.py"

dry_dir="${OUTPUT_ROOT}/install_dry_run"
LIMIT=5 \
OUTPUT_BASE="${dry_dir}" \
MODEL_PATH="${MODEL_PATH}" \
MODEL_URL="${MODEL_URL}" \
RETRIEVER_URL="${RETRIEVER_URL}" \
SANDBOX_URL="${SANDBOX_URL}" \
INTERMEDIATE_DATA="${INTERMEDIATE_DATA}" \
PYTHON_BIN="${CORE_ENV_PREFIX}/bin/python" \
"${SRC_ACTION_TREE_DIR}/run_qwen3_three_conditions.sh" dry-run

nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
df -h "${WORK_ROOT}" /
echo "INSTALL VALIDATION PASSED. No service or FAISS index was loaded."
