#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

mkdir -p "${WORK_ROOT}/envs" "${WORK_ROOT}/downloads" "$(dirname "${CONDA_ROOT}")"

find_conda_sh() {
  local candidate
  for candidate in \
    "${CONDA_ROOT}/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    [[ -f "${candidate}" ]] && { echo "${candidate}"; return 0; }
  done
  return 1
}

if ! CONDA_SH=$(find_conda_sh); then
  installer="${WORK_ROOT}/downloads/Miniconda3-latest-Linux-x86_64.sh"
  echo "No conda installation found; installing Miniconda under ${CONDA_ROOT}."
  curl -fL --retry 5 --retry-delay 5 -C - \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o "${installer}"
  bash "${installer}" -b -p "${CONDA_ROOT}"
  CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
fi
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda config --set auto_activate_base false >/dev/null 2>&1 || true

create_env() {
  local prefix="$1"
  if [[ -x "${prefix}/bin/python" ]]; then
    echo "Environment already exists: ${prefix}"
  else
    conda create -y -p "${prefix}" python=3.11 pip
  fi
}
create_env "${CORE_ENV_PREFIX}"
create_env "${MODEL_ENV_PREFIX}"

echo "Installing the CPU-FAISS/core environment..."
conda install -y -p "${CORE_ENV_PREFIX}" -c pytorch \
  "faiss-cpu=1.8.0" "numpy=1.26.4"
"${CORE_ENV_PREFIX}/bin/python" -m pip install --upgrade pip setuptools wheel
"${CORE_ENV_PREFIX}/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"
"${CORE_ENV_PREFIX}/bin/python" -m pip install \
  "accelerate==1.9.0" "datasets==4.0.0" "transformers==4.51.1" \
  sentence-transformers pandas tqdm pyyaml requests openai jsonlines \
  fastapi uvicorn pydantic omegaconf hydra-core bm25s rank-bm25 \
  jieba langid rouge rouge-chinese termcolor modelscope huggingface-hub \
  scipy scikit-learn psutil

echo "Installing the isolated SGLang model environment..."
"${MODEL_ENV_PREFIX}/bin/python" -m pip install --upgrade pip setuptools wheel
"${MODEL_ENV_PREFIX}/bin/python" -m pip install "sglang[all]==0.4.6.post4"
"${MODEL_ENV_PREFIX}/bin/python" -m pip install --upgrade --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"
"${MODEL_ENV_PREFIX}/bin/python" -m pip install --upgrade \
  "transformers==4.51.1" "sglang==0.4.6.post4" \
  "ninja==1.11.1.4" requests
# sglang[all] leaves these runtime dependencies open-ended, so a fresh install
# in 2026 can combine the 2025 SGLang/Transformers stack with incompatible new
# releases.  Pin the contemporaneous pair after all SGLang dependency solving,
# without dependencies, so pip cannot upgrade torch or transformers again.
"${MODEL_ENV_PREFIX}/bin/python" -m pip install --force-reinstall --no-deps \
  "torchao==0.10.0" "compressed-tensors==0.9.4"

echo "Checking pinned runtime versions..."
"${CORE_ENV_PREFIX}/bin/python" - <<'PY'
import faiss, torch, transformers, datasets
print("core torch", torch.__version__, "cuda", torch.version.cuda)
print("core cuda available", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("faiss", getattr(faiss, "__version__", "unknown"))
print("transformers", transformers.__version__, "datasets", datasets.__version__)
assert torch.__version__.startswith("2.6.0")
assert transformers.__version__ == "4.51.1"
PY
"${MODEL_ENV_PREFIX}/bin/python" - <<'PY'
from importlib.metadata import version
import sglang, torch, torchao, transformers
from sglang.srt.layers.quantization import QUANTIZATION_METHODS
print("sglang", sglang.__version__)
print("model torch", torch.__version__, "cuda", torch.version.cuda)
print("torchao", version("torchao"))
print("compressed-tensors", version("compressed-tensors"))
print("transformers", transformers.__version__)
print("quantization registry", len(QUANTIZATION_METHODS))
assert sglang.__version__ == "0.4.6.post4"
assert torch.__version__.startswith("2.6.0")
assert version("torchao") == "0.10.0"
assert version("compressed-tensors") == "0.9.4"
assert transformers.__version__ == "4.51.1"
PY
"${MODEL_ENV_PREFIX}/bin/ninja" --version

printf 'CONDA_SH=%q\n' "${CONDA_SH}" > "${SCRIPT_DIR}/conda_resolved.env"
echo "ENVIRONMENT SETUP PASSED"
