#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-vllm-qwen}"
MINIFORGE_DIR="${HOME}/miniforge3"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
QUANTIZATION="${QUANTIZATION:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Avoid FlashInfer sampler JIT on WSL/CUDA 13, which can fail when CUDA
# headers and bundled compiler headers disagree. PyTorch-native sampling is
# slower but much more robust for this single-GPU RAG service.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if [[ -z "${QUANTIZATION}" && "${MODEL}" == *AWQ* ]]; then
  QUANTIZATION="awq"
  export VLLM_USE_TRITON_AWQ="${VLLM_USE_TRITON_AWQ:-1}"
fi

QUANTIZATION_ARGS=()
if [[ -n "${QUANTIZATION}" ]]; then
  QUANTIZATION_ARGS=(--quantization "${QUANTIZATION}")
fi

source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

CUDA_PIP_HOME="${CONDA_PREFIX}/lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/nvidia/cu13"
if [ -x "${CUDA_PIP_HOME}/bin/nvcc" ]; then
  export CUDA_HOME="${CUDA_HOME:-${CUDA_PIP_HOME}}"
  export CUDACXX="${CUDACXX:-${CUDA_PIP_HOME}/bin/nvcc}"
  export PATH="${CUDA_PIP_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_PIP_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi

python -m vllm.entrypoints.openai.api_server \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  "${QUANTIZATION_ARGS[@]}" \
  --dtype auto \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.65}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --enforce-eager \
  --trust-remote-code
