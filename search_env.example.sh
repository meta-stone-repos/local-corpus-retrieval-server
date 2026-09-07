#!/usr/bin/env bash

# Copy this file to search_env.sh and customize it locally:
#   cp search_env.example.sh search_env.sh

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

# Optional Hugging Face endpoint:
# export HF_ENDPOINT="https://huggingface.co"

# Optional deployment proxy:
# export HTTPS_PROXY="http://proxy-host:proxy-port"
# export HTTP_PROXY="$HTTPS_PROXY"
# export NO_PROXY="localhost,127.0.0.1"
