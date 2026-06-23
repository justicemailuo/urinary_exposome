#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-vllm-qwen}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
MINIFORGE_DIR="${HOME}/miniforge3"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is not available inside WSL. Install/update NVIDIA Windows driver with WSL CUDA support first." >&2
  exit 1
fi

nvidia-smi

if [ ! -x "${MINIFORGE_DIR}/bin/conda" ]; then
  echo "Installing Miniforge into ${MINIFORGE_DIR} ..."
  installer="/tmp/Miniforge3-Linux-x86_64.sh"
  curl -L -o "${installer}" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "${installer}" -b -p "${MINIFORGE_DIR}"
fi

source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip wheel setuptools
python -m pip install vllm

python - <<'PY'
import torch
import vllm
print("torch", torch.__version__)
print("torch_cuda_available", torch.cuda.is_available())
print("torch_cuda_version", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("vllm", vllm.__version__)
PY

echo "vLLM environment is ready: conda activate ${ENV_NAME}"
