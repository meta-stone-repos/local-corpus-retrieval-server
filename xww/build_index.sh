#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="/home/xww/workspace/local-corpus-retrieval-server"
INDEX_DIR="/home/xww/workspace/data/local-corpus-retrieval-server/indexes"
CONFIG="$PROJECT_DIR/configs/corpora/xww-enwikibooks.yaml"
export CUDA_VISIBLE_DEVICES=6

cd "$PROJECT_DIR"
source .venv/bin/activate

mkdir -p "$INDEX_DIR"
if [[ -e indexes && ! -L indexes ]]; then
  echo "错误: $PROJECT_DIR/indexes 已存在且不是软链接" >&2
  exit 1
fi
ln -sfn "$INDEX_DIR" indexes

python scripts_build_index/run_corpus_pipeline.py --config "$CONFIG"
