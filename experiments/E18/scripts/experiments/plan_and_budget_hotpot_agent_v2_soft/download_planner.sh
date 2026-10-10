#!/usr/bin/env bash
set -euo pipefail

REBUILD_ROOT="${REBUILD_ROOT:-/home/researcher/smartsearch_rebuild}"
MODEL_DIR="${PLANNER_MODEL_PATH:-$REBUILD_ROOT/models/Llama-3.1-8B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-$REBUILD_ROOT/envs/src-core/bin/python}"
MODEL_ID="${PLANNER_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export the token that has accepted the Meta Llama license." >&2
  exit 1
fi

mkdir -p "$MODEL_DIR"

"$PYTHON_BIN" - "$MODEL_ID" "$MODEL_DIR" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

model_id, model_dir = sys.argv[1:]
snapshot_download(
    repo_id=model_id,
    local_dir=model_dir,
    token=os.environ["HF_TOKEN"],
    resume_download=True,
)
print(f"Planner model ready: {model_dir}")
PY

"$PYTHON_BIN" - "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
weight_bytes = sum(p.stat().st_size for p in root.glob("*.safetensors"))
print({
    "model_type": config.get("model_type"),
    "architectures": config.get("architectures"),
    "max_position_embeddings": config.get("max_position_embeddings"),
    "weight_gib": round(weight_bytes / 1024**3, 2),
})
assert config.get("model_type") == "llama"
assert any("LlamaForCausalLM" in x for x in config.get("architectures", []))
assert weight_bytes > 10 * 1024**3
print("STRICT LLAMA-3.1-8B-INSTRUCT DOWNLOAD CHECK PASSED")
PY

