#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME="${VLLM_MODEL_NAME:-Qwen/Qwen3.5-4B}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"

echo "Starting vLLM service..."
echo "Model: ${MODEL_NAME}"
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "Max model length: ${MAX_MODEL_LEN}"
echo "Reasoning parser: ${REASONING_PARSER}"

vllm serve "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --reasoning-parser "${REASONING_PARSER}"