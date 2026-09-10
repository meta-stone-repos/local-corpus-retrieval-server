#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xww/workspace/local-corpus-retrieval-server}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-${1:-}}"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"

die() {
  echo "错误: $*" >&2
  exit 1
}

command -v uv >/dev/null 2>&1 || die "未找到 uv。请先执行: curl -LsSf https://astral.sh/uv/install.sh | sh"
command -v java >/dev/null 2>&1 || die "未找到 Java。Pyserini/Lucene 需要 Java 21。"
[[ -x "$PYTHON_BIN" ]] || die "Python 不存在或不可执行: $PYTHON_BIN"
[[ -f "$PROJECT_DIR/pyproject.toml" ]] || die "项目目录无效: $PROJECT_DIR"
[[ -f "$PROJECT_DIR/uv.lock" ]] || die "未找到 uv.lock: $PROJECT_DIR/uv.lock"

java_major="$(
  java -version 2>&1 |
    awk -F '[\".]' '/version/ { print ($2 == "1" ? $3 : $2); exit }'
)"
[[ "$java_major" =~ ^[0-9]+$ ]] || die "无法识别 Java 版本。"
((java_major >= 21)) || die "Java 版本过低（当前 $java_major），需要 Java 21 或更高版本。"

cd "$PROJECT_DIR"

if [[ -x "$VENV_DIR/bin/python" ]]; then
  echo "==> 复用虚拟环境: $VENV_DIR"
else
  echo "==> 创建虚拟环境: $VENV_DIR"
  uv venv --python "$PYTHON_BIN" "$VENV_DIR"
fi

echo "==> 安装普通依赖（排除锁文件中的 CUDA 12.6 GPU 栈）"
UV_PROJECT_ENVIRONMENT="$VENV_DIR" UV_LINK_MODE=copy \
  uv sync --frozen \
  --no-install-package torch \
  --no-install-package torchvision \
  --no-install-package torchaudio \
  --no-install-package nvidia-cublas-cu12 \
  --no-install-package nvidia-cuda-cupti-cu12 \
  --no-install-package nvidia-cuda-nvrtc-cu12 \
  --no-install-package nvidia-cuda-runtime-cu12 \
  --no-install-package nvidia-cudnn-cu12 \
  --no-install-package nvidia-cufft-cu12 \
  --no-install-package nvidia-cufile-cu12 \
  --no-install-package nvidia-curand-cu12 \
  --no-install-package nvidia-cusolver-cu12 \
  --no-install-package nvidia-cusparse-cu12 \
  --no-install-package nvidia-cusparselt-cu12 \
  --no-install-package nvidia-nccl-cu12 \
  --no-install-package nvidia-nvjitlink-cu12 \
  --no-install-package nvidia-nvtx-cu12

echo "==> 安装 README 验证过的 PyTorch CUDA 12.8"
uv pip install \
  --python "$VENV_DIR/bin/python" \
  --index-url "${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}" \
  "torch==${TORCH_VERSION:-2.7.0+cu128}" \
  "torchvision==${TORCHVISION_VERSION:-0.22.0+cu128}" \
  "torchaudio==${TORCHAUDIO_VERSION:-2.7.0+cu128}"

echo "==> 安装 ModelScope 和 SOCKS 支持"
uv pip install \
  --python "$VENV_DIR/bin/python" \
  modelscope \
  "httpx[socks]"

if [[ -n "$FLASH_ATTN_WHEEL" ]]; then
  [[ -f "$FLASH_ATTN_WHEEL" ]] || die "FlashAttention wheel 不存在: $FLASH_ATTN_WHEEL"
  flash_attn_source="$FLASH_ATTN_WHEEL"
else
  flash_attn_source="$FLASH_ATTN_URL"
fi

echo "==> 安装 FlashAttention 2.8.3.post1"
uv pip install \
  --python "$VENV_DIR/bin/python" \
  --no-deps \
  "$flash_attn_source"

echo "==> 检查依赖"
uv pip check --python "$VENV_DIR/bin/python"

echo "环境安装完成。"
echo "激活命令: source \"$VENV_DIR/bin/activate\""
