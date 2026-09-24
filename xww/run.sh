#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${CONFIG:-$PROJECT_DIR/configs/services.xww-enwikibooks.yaml}"
# CONFIG="${CONFIG:-$PROJECT_DIR/configs/services.xww-browscomp-plus.yaml}"

cd "$PROJECT_DIR"
source .venv/bin/activate
[[ ! -f search_env.sh ]] || source search_env.sh

case "${1:-}" in
  --status|--stop)
    exec python3 searcher/serve.py --config "$CONFIG" "$1"
    ;;
esac

python3 searcher/serve.py --config "$CONFIG" --services bm25 --check
exec python3 searcher/serve.py --config "$CONFIG" --services bm25
