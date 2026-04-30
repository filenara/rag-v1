#!/usr/bin/env bash

set -euo pipefail

EMBED_HOST="${EMBED_HOST:-127.0.0.1}"
EMBED_PORT="${EMBED_PORT:-8081}"

RERANK_HOST="${RERANK_HOST:-127.0.0.1}"
RERANK_PORT="${RERANK_PORT:-8082}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

echo "Starting embedding service on ${EMBED_HOST}:${EMBED_PORT}"
python -m uvicorn services.embed_service:app \
  --host "${EMBED_HOST}" \
  --port "${EMBED_PORT}" \
  > "${LOG_DIR}/embed.log" 2>&1 &

EMBED_PID=$!
echo "Embedding service PID: ${EMBED_PID}"

echo "Starting reranker service on ${RERANK_HOST}:${RERANK_PORT}"
python -m uvicorn services.rerank_service:app \
  --host "${RERANK_HOST}" \
  --port "${RERANK_PORT}" \
  > "${LOG_DIR}/rerank.log" 2>&1 &

RERANK_PID=$!
echo "Reranker service PID: ${RERANK_PID}"

echo ""
echo "Services started."
echo "Embedding log: ${LOG_DIR}/embed.log"
echo "Reranker log: ${LOG_DIR}/rerank.log"
echo ""
echo "Health checks:"
echo "curl http://127.0.0.1:${EMBED_PORT}/health"
echo "curl http://127.0.0.1:${RERANK_PORT}/health"
echo ""
echo "To stop:"
echo "kill ${EMBED_PID} ${RERANK_PID}"

wait