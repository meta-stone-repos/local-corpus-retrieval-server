#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="/home/xww/workspace/local-corpus-retrieval-server"
# CORPUS_NAME="xww-local"
CORPUS_NAME="xww-enwikibooks"
BM25_PORT="8090"
DENSE_PORT="8091"
GPU_ID="7"

cd "$PROJECT_DIR"
source .venv/bin/activate
[[ ! -f search_env.sh ]] || source search_env.sh

cleanup() {
  trap - EXIT INT TERM
  kill "${BM25_PID:-}" "${DENSE_PID:-}" 2>/dev/null || true
  wait "${BM25_PID:-}" "${DENSE_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python searcher/mcp_server.py \
  --searcher-type bm25 \
  --corpus "$CORPUS_NAME" \
  --get-document \
  --bm25-workers 16 \
  --transport sse \
  --port "$BM25_PORT" &
BM25_PID=$!

CUDA_VISIBLE_DEVICES="$GPU_ID" python searcher/mcp_server.py \
  --searcher-type faiss \
  --corpus "$CORPUS_NAME" \
  --get-document \
  --dense-max-batch-size 16 \
  --dense-max-wait-ms 1 \
  --dense-max-queue-size 64 \
  --transport sse \
  --port "$DENSE_PORT" &
DENSE_PID=$!

wait -n "$BM25_PID" "$DENSE_PID"