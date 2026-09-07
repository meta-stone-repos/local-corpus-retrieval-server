# 本地语料检索服务器

[English](README.md) | [简体中文](README.zh-CN.md)

这是一个面向固定本地语料库的自托管搜索引擎。项目将稀疏检索、稠密检索、文档读取、语料导入、索引构建和 MCP 服务封装为一套可部署系统，主要用于让 Agent 和应用检索私有语料、精选数据集或需要可复现结果的数据，而不依赖实时互联网搜索引擎。

本服务基于 [BrowseComp-Plus](https://texttron.github.io/BrowseComp-Plus/) 的检索代码开发，但它是一个独立的工程项目，并不是 BrowseComp-Plus 基准本身。项目仍然兼容 BrowseComp-Plus 的数据集、预构建索引和评测工具，可将其作为可选的数据源与复现任务。

BrowseComp-Plus 上游资源：[数据集](https://huggingface.co/datasets/Tevatron/browsecomp-plus) · [排行榜](https://huggingface.co/spaces/Tevatron/BrowseComp-Plus) · [论文](https://arxiv.org/pdf/2508.06600) · [项目主页](https://texttron.github.io/BrowseComp-Plus/)

本地检索服务器提供：

- 基于 Pyserini/Lucene 的 BM25 检索；
- 基于 FAISS Flat 的稠密精确检索；
- 面向大规模语料的 FAISS IVF-Flat 近似检索；
- 相互独立、由 manifest 管理的本地 corpus；
- 用于受控全文读取的 SQLite 文档库；
- 为 BM25 和稠密检索提供一致工具接口的 MCP 服务；
- 可恢复的语料标准化与索引构建流程；
- 面向 Wikimedia Enterprise HTML 归档的 YAML 流水线。

## 目录

- [检索系统概览](#检索系统概览)
- [快速开始](#快速开始)
- [环境安装](#环境安装)
- [数据集与预构建索引](#数据集与预构建索引)
- [启动检索服务](#启动检索服务)
- [构建本地语料库](#构建本地语料库)
- [Wikimedia Enterprise YAML 流水线](#wikimedia-enterprise-yaml-流水线)
- [构建产物与磁盘空间](#构建产物与磁盘空间)
- [评测](#评测)
- [相关文档](#相关文档)
- [上游致谢与引用](#上游致谢与引用)

## 检索系统概览

| 组件 | 实现 | 典型用途 |
| --- | --- | --- |
| 稀疏检索 | Pyserini/Lucene BM25 | 关键词检索和 CPU 服务 |
| 稠密精确检索 | 基于 Tevatron pickle shard 的 FAISS Flat | 中小型语料的精确向量检索 |
| 稠密 ANN 检索 | FAISS IVF-Flat | 精确搜索成本过高的大型语料 |
| 文档库 | 只读 SQLite | 查询全文、URL、标题、语言和字符数 |
| Corpus manifest | `corpora/<name>/manifest.json` | 保证构建时和查询时的参数一致 |
| MCP 接口 | FastMCP | 提供统一的 `search` 和 `read_document` 工具 |

每个本地 corpus 都是相互独立的。它的标准化文档、文档库、稀疏索引、稠密索引和 manifest 不会与其他 corpus 合并。

## 快速开始

下面使用 BrowseComp-Plus 上游提供的 BM25 和 Qwen3-Embedding-0.6B 索引启动服务，可以在不重新构建语料索引的情况下验证安装是否正确。如果要直接索引自己的数据，请跳到[构建本地语料库](#构建本地语料库)。开始前请先完成[环境安装](#环境安装)。

### 1. 激活环境

```bash
REPO_DIR=/path/to/local-repository
cd "$REPO_DIR"
source .venv/bin/activate

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
```

### 2. 下载预构建索引

```bash
bash scripts_build_index/download_indexes.sh
```

索引将下载到 `indexes/`。下载脚本包含上游 BrowseComp-Plus 的 BM25 以及多个 Qwen3-Embedding 索引。

### 3. 生成本地文档库

预构建稠密索引包含向量和文档 ID，但 `read_document` 需要本地标准语料和 SQLite 文档库：

```bash
python scripts_preprocess/materialize_browsecomp_corpus.py \
  --cache-dir "$HF_HOME" \
  --output-dir corpora/browsecomp
```

只有在确定要替换现有 `corpora/browsecomp` 目录时才使用 `--overwrite`。

### 4. 启动 BM25

```bash
BM25_PORT=8080

python searcher/mcp_server.py \
  --searcher-type bm25 \
  --index-path indexes/bm25 \
  --document-store corpora/browsecomp/documents.sqlite \
  --get-document \
  --bm25-workers 16 \
  --snippet-max-tokens 512 \
  --transport sse \
  --port "$BM25_PORT"
```

### 5. 启动稠密检索

在另一个终端中执行：

```bash
source .venv/bin/activate

GPU_ID=0
DENSE_PORT=8081

CUDA_VISIBLE_DEVICES="$GPU_ID" \
python searcher/mcp_server.py \
  --searcher-type faiss \
  --index-path "indexes/qwen3-embedding-0.6b/corpus.shard*.pkl" \
  --model-name Qwen/Qwen3-Embedding-0.6B \
  --pooling eos \
  --normalize \
  --document-store corpora/browsecomp/documents.sqlite \
  --get-document \
  --dense-max-batch-size 16 \
  --dense-max-wait-ms 1 \
  --dense-max-queue-size 64 \
  --snippet-max-tokens 512 \
  --transport sse \
  --port "$DENSE_PORT"
```

本地 MCP 地址为：

```text
BM25:  http://127.0.0.1:8080/mcp
Dense: http://127.0.0.1:8081/mcp
```

只有当客户端位于其他机器、并且已经安全配置网络访问时，才需要将 `127.0.0.1` 替换为服务器主机名。

两个服务提供相同的工具：

```text
search(query)
read_document(docid, chunk_index=0)
```

`--get-document` 是保留的旧 CLI 参数名，实际向客户端注册的工具名是 `read_document`。每次调用返回一个最多 8,192 个字符的受控分块。

## 环境安装

本项目的环境比普通 Python 项目更复杂，因为 PyTorch、CUDA、FlashAttention、FAISS 和 GPU 架构之间必须相互兼容。

### 已验证的 Blackwell 环境

以下软件栈已经过验证：

| 组件 | 已验证版本 |
| --- | --- |
| 操作系统/容器 | Ubuntu 22.04 |
| Python | 3.10.12 |
| CUDA Toolkit | 12.8 |
| GPU | NVIDIA RTX 6000D，Blackwell `sm_120` |
| PyTorch | 2.7.0+cu128 |
| TorchVision | 0.22.0+cu128 |
| TorchAudio | 2.7.0+cu128 |
| FlashAttention | 2.8.3.post1 |
| vLLM | 0.9.0.1 |
| FAISS | 1.11.0 |
| Transformers | 4.53.2 |
| Datasets | 4.0.0 |
| FastMCP | 2.9.2 |

其他硬件也可以运行，但应当选择该机器支持的 PyTorch、CUDA 和 FlashAttention 组合。

### 1. 进入运行环境

如果使用已有 Docker 容器：

```bash
CONTAINER_NAME=my-container

docker start "$CONTAINER_NAME"
docker exec -it "$CONTAINER_NAME" bash
REPO_DIR=/path/to/local-repository
cd "$REPO_DIR"
```

Docker 并非必需。在兼容的 Linux 主机上，也可以直接进入克隆后的仓库。

### 2. 安装 `uv` 和 Java

项目使用 `uv` 和 Python 3.10。Pyserini/Lucene 还需要 Java 21。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

请使用主机上的包管理器安装 Java。Ubuntu 示例：

```bash
sudo apt update
sudo apt install -y openjdk-21-jdk
```

确认安装：

```bash
java -version
```

### 3. 创建虚拟环境

```bash
uv venv --python /usr/bin/python3 .venv
source .venv/bin/activate
python -V
```

已验证环境会输出 `Python 3.10.12`。

### 4. 安装普通依赖，但暂不安装默认 GPU 栈

当前 `uv.lock` 会解析到 PyPI 默认的 PyTorch/CUDA 12.6 依赖栈，而已验证的 Blackwell 环境使用 PyTorch CUDA 12.8。因此应先安装普通依赖，并排除 GPU 栈：

```bash
UV_LINK_MODE=copy \
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
```

### 5. 安装 PyTorch CUDA 12.8

```bash
uv pip install \
  --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.7.0+cu128' \
  'torchvision==0.22.0+cu128' \
  'torchaudio==2.7.0+cu128'
```

### 6. 安装 FlashAttention

FlashAttention wheel 必须与 Python、PyTorch、CUDA、操作系统和 GPU 架构匹配。已验证环境的安装方式为：

```bash
FLASH_ATTN_WHEEL=/path/to/compatible-flash-attention.whl

uv pip install \
  --python .venv/bin/python \
  --no-deps \
  "$FLASH_ATTN_WHEEL"
```

已验证 wheel 提供 `flash-attn==2.8.3.post1`，适用于 CPython 3.10、PyTorch 2.7.0+cu128 和 Blackwell `sm_120`。

不要从其他机器复制 `.venv`。Python 虚拟环境包含与机器相关的二进制文件和解释器绝对路径，应当根据仓库中的依赖元数据重新创建。

### 7. 配置 Hugging Face 缓存

缓存应放在 `.venv` 之外的持久化位置：

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
```

可以将 `search_env.example.sh` 复制为已被忽略的 `search_env.sh`，再按具体部署进行调整。不要提交凭据或机器相关的秘密配置。

如果环境需要代理，请使用部署侧的实际值为当前 shell 配置：

```bash
export HTTPS_PROXY="http://<proxy-host>:<proxy-port>"
export HTTP_PROXY="$HTTPS_PROXY"
export NO_PROXY="localhost,127.0.0.1"
```

### 8. 验证环境

检查模块导入和依赖一致性：

```bash
python - <<'PY'
modules = [
    "torch",
    "flash_attn",
    "faiss",
    "transformers",
    "datasets",
    "tevatron",
    "fastmcp",
    "pyserini",
    "vllm",
]

for name in modules:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name:15s} OK   {version}")
    except Exception as exc:
        print(f"{name:15s} FAIL {type(exc).__name__}: {exc}")
PY

uv pip check --python .venv/bin/python
```

在 GPU 上验证 CUDA 和 FlashAttention：

```bash
GPU_ID=0

CUDA_VISIBLE_DEVICES="$GPU_ID" python - <<'PY'
import torch
from flash_attn import flash_attn_func

print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))

x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
_ = x @ x
torch.cuda.synchronize()
print("CUDA matmul: OK")

q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
out = flash_attn_func(q, k, v)
torch.cuda.synchronize()
print("FlashAttention:", tuple(out.shape), "OK")
PY
```

在 lockfile 能直接描述 CUDA 12.8 覆盖关系之前，不要使用裸 `uv sync` 或 `uv run` 改动已验证的 Blackwell 环境，应重新执行上述两阶段安装。

误删环境或离线恢复 `uv` 缓存属于故障处理流程，不应混入正常安装步骤。

## 数据集与预构建索引

### 可选：下载上游基准数据

下载并解密基准查询、答案和相关性标注：

```bash
python scripts_build_index/decrypt_dataset.py \
  --output data/browsecomp_plus_decrypted.jsonl \
  --generate-tsv topics-qrels/queries.tsv
```

下载时可能需要 Hugging Face 身份验证。建议使用 `HF_TOKEN` 环境变量或 `huggingface-cli login`，不要把 token 直接写进 shell history。

也可以单独加载公开 corpus：

```python
from datasets import load_dataset

corpus = load_dataset(
    "Tevatron/browsecomp-plus-corpus",
    split="train",
)
```

### 下载索引

```bash
bash scripts_build_index/download_indexes.sh
```

脚本会从 `Tevatron/browsecomp-plus-indexes` 下载可用的 BM25 和 Qwen3-Embedding 索引，并保存到 `indexes/`。

## 启动检索服务

服务支持两种配置方式。

### Manifest 模式：本地构建 corpus 的推荐方式

每个由 `build_corpus.py` 构建的 corpus 都会在 `corpora/<name>/manifest.json` 中保存配置。Manifest 会记录索引路径、模型、pooling、归一化、query prefix、文档库、索引类型和 ANN 参数。

启动 BM25：

```bash
CORPUS_NAME=my-corpus
PORT=8080

python searcher/mcp_server.py \
  --searcher-type bm25 \
  --corpus "$CORPUS_NAME" \
  --get-document \
  --bm25-workers 16 \
  --transport sse \
  --port "$PORT"
```

启动稠密检索：

```bash
CORPUS_NAME=my-corpus
GPU_ID=0
PORT=8081

CUDA_VISIBLE_DEVICES="$GPU_ID" \
python searcher/mcp_server.py \
  --searcher-type faiss \
  --corpus "$CORPUS_NAME" \
  --get-document \
  --dense-max-batch-size 16 \
  --dense-max-wait-ms 1 \
  --dense-max-queue-size 64 \
  --transport sse \
  --port "$PORT"
```

除非有意覆盖 manifest 支持的服务参数，否则不要在命令行重复设置模型或索引信息。查询编码参数必须与文档索引一致。

### 显式路径模式：旧索引和高级用法

BM25 模板：

```bash
BM25_INDEX_DIR=indexes/bm25
DOCUMENT_STORE=corpora/browsecomp/documents.sqlite
PORT=8080

python searcher/mcp_server.py \
  --searcher-type bm25 \
  --index-path "$BM25_INDEX_DIR" \
  --document-store "$DOCUMENT_STORE" \
  --get-document \
  --bm25-workers 16 \
  --transport sse \
  --port "$PORT"
```

Dense Flat 模板：

```bash
MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
POOLING=eos
DENSE_INDEX_GLOB='indexes/qwen3-embedding-0.6b/corpus.shard*.pkl'
DOCUMENT_STORE=corpora/browsecomp/documents.sqlite
GPU_ID=0
PORT=8081

CUDA_VISIBLE_DEVICES="$GPU_ID" \
python searcher/mcp_server.py \
  --searcher-type faiss \
  --index-type flat \
  --index-path "$DENSE_INDEX_GLOB" \
  --model-name "$MODEL_NAME" \
  --pooling "$POOLING" \
  --normalize \
  --document-store "$DOCUMENT_STORE" \
  --get-document \
  --dense-max-batch-size 16 \
  --dense-max-wait-ms 1 \
  --dense-max-queue-size 64 \
  --transport sse \
  --port "$PORT"
```

IVF-Flat 模板：

```bash
MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
POOLING=eos
DENSE_INDEX_DIR=indexes/corpora/my-corpus/qwen3-embedding-0.6b
DOCUMENT_STORE=corpora/my-corpus/documents.sqlite
GPU_ID=0
PORT=8081

CUDA_VISIBLE_DEVICES="$GPU_ID" \
python searcher/mcp_server.py \
  --searcher-type faiss \
  --index-type ivf-flat \
  --index-path "$DENSE_INDEX_DIR/index.ivf-flat.faiss" \
  --lookup-path "$DENSE_INDEX_DIR/index.lookup.pkl" \
  --nprobe 32 \
  --model-name "$MODEL_NAME" \
  --pooling "$POOLING" \
  --normalize \
  --document-store "$DOCUMENT_STORE" \
  --get-document \
  --transport sse \
  --port "$PORT"
```

### 推荐服务参数

以下参数在已验证部署中表现良好，可以作为起点，但并不保证适用于所有硬件。

| 范围 | 参数 | 推荐值 | 代码默认值 |
| --- | --- | ---: | ---: |
| BM25 | `--bm25-workers` | 16 | 1 |
| BM25 | 客户端请求并发数 | 8 | 不适用 |
| Dense | `--dense-max-batch-size` | 16 | 16 |
| Dense | `--dense-max-wait-ms` | 1 ms | 5 ms |
| Dense | `--dense-max-queue-size` | 64 | 64 |
| Dense | 客户端请求并发数 | 32 | 不适用 |
| MCP | `--transport` | `sse` | `sse` |
| Search | `--k` | 5 | 5 |
| Search | `--snippet-max-tokens` | 512 | 512 |
| Read | 分块大小 | 8,192 字符 | 8,192 字符 |

在已验证硬件和 BrowseComp corpus 上测得的回归参考值约为：

```text
BM25:  ~43 QPS, p50 ~176 ms, p95 ~251 ms
Dense: ~35–36 QPS, p50 ~862 ms, p95 ~1.1 s
```

实际结果取决于语料、CPU、GPU、索引类型、客户端会话数和网络路径。使用可比测试时，如果 BM25 明显低于约 38 QPS，或稠密检索明显低于约 30 QPS，应进一步排查性能回退。

### Transport 与 MCP 地址

支持以下 transport：

```text
stdio
sse
streamable-http
```

HTTP 服务路径固定为 `/mcp`，例如：

```text
http://127.0.0.1:<port>/mcp
```

在当前依赖组合下，SSE 是经过测试的推荐方案；兼容客户端仍可使用 Streamable HTTP。

## 构建本地语料库

普通 JSONL、DPR Wiki18 TSV 和已经标准化的 corpus 使用 `scripts_build_index/build_corpus.py`。Wikimedia Enterprise HTML tar 使用专门的 [YAML 流水线](#wikimedia-enterprise-yaml-流水线)。

### 输出目录结构

```text
corpora/<name>/
├── corpus.jsonl
├── documents.sqlite
└── manifest.json

indexes/corpora/<name>/
├── bm25/
└── <embedding-model>/
    ├── corpus.shard00_of_01.pkl
    ├── index.ivf-flat.faiss
    └── index.lookup.pkl
```

最后两个文件只会出现在 IVF-Flat 索引中。

### 普通 JSONL

输入文件要求每行包含一个 JSON 对象：

```json
{"docid": "doc-001", "text": "文档正文"}
{"docid": "doc-002", "text": "另一篇文档正文"}
```

字段名可以映射，例如：

```json
{"id": "doc-001", "title": "文档标题", "content": "文档正文"}
```

要求与处理行为：

- 文档 ID 必须存在，并且全局唯一；
- 配置的正文字段必须存在；
- 标准化后为空的正文会跳过；
- 每一个非空行都必须是合法 JSON；
- 如果配置标题，会按照 `title + "\n" + text` 拼接；
- 普通 JSONL 入口只保留 ID、标准化正文和可选标题；
- 如果需要保留 URL 或其他 metadata，应编写自定义 materializer。

构建 BM25 和 Dense Flat：

```bash
GPU_IDS=0

python scripts_build_index/build_corpus.py \
  --name my-corpus \
  --input data/my-corpus/corpus.jsonl \
  --format jsonl \
  --id-field docid \
  --text-field text \
  --build bm25,dense \
  --bm25-threads 16 \
  --dense-model Qwen/Qwen3-Embedding-0.6B \
  --pooling eos \
  --passage-max-len 4096 \
  --query-max-len 8192 \
  --batch-size 32 \
  --dtype fp16 \
  --gpus "$GPU_IDS" \
  --dense-shards 1 \
  --dense-index-type flat \
  --normalize
```

输入字段为 `id`、`content` 和 `title` 时：

```bash
python scripts_build_index/build_corpus.py \
  --name my-corpus \
  --input data/my-corpus/documents.jsonl \
  --format jsonl \
  --id-field id \
  --text-field content \
  --title-field title \
  --build bm25,dense \
  --dense-model Qwen/Qwen3-Embedding-0.6B \
  --pooling eos \
  --gpus 0 \
  --dense-shards 1
```

只构建 BM25：

```bash
python scripts_build_index/build_corpus.py \
  --name my-corpus \
  --input data/my-corpus/corpus.jsonl \
  --format jsonl \
  --id-field docid \
  --text-field text \
  --build bm25
```

只进行格式标准化并创建 SQLite 文档库，不构建索引：

```bash
python scripts_build_index/build_corpus.py \
  --name my-corpus \
  --input data/my-corpus/corpus.jsonl \
  --format jsonl \
  --id-field docid \
  --text-field text \
  --build none
```

### DPR Wiki18 TSV

输入文件必须包含表头，并具有以下制表符分隔字段：

```text
id<TAB>text<TAB>title
```

先构建一个包含 10,000 篇文档的 canary：

```bash
python scripts_build_index/build_corpus.py \
  --name wiki18-canary \
  --input data/wiki18/psgs_w100.tsv.gz \
  --format dpr-wiki18 \
  --limit 10000 \
  --build bm25,dense \
  --passage-max-len 512 \
  --pooling eos \
  --gpus 0 \
  --dense-shards 1
```

验证 canary 后，使用另一个名称构建完整语料：

```bash
GPU_IDS=0,1,2,3

python scripts_build_index/build_corpus.py \
  --name wiki18 \
  --input data/wiki18/psgs_w100.tsv.gz \
  --format dpr-wiki18 \
  --build bm25,dense \
  --passage-max-len 512 \
  --pooling eos \
  --gpus "$GPU_IDS" \
  --dense-shards 4
```

### 构建 IVF-Flat ANN

Flat 是精确检索，直接读取 embedding pickle shards。IVF-Flat 会额外生成近似索引，更适合超大语料。

一次完成标准化、BM25、稠密向量和 IVF-Flat：

```bash
GPU_IDS=0,1,2,3

python scripts_build_index/build_corpus.py \
  --name my-large-corpus \
  --input data/my-large-corpus/corpus.jsonl \
  --format jsonl \
  --id-field docid \
  --text-field text \
  --build bm25,dense,ann \
  --dense-model Qwen/Qwen3-Embedding-0.6B \
  --pooling eos \
  --gpus "$GPU_IDS" \
  --dense-shards 4 \
  --dense-index-type ivf-flat \
  --ivf-nlist 16384 \
  --ivf-nprobe 32 \
  --ivf-train-size 819200 \
  --ivf-niter 25 \
  --ivf-backend gpu \
  --ivf-gpus "$GPU_IDS" \
  --ivf-gpu-batch-size 8192 \
  --ivf-gpu-dtype float32
```

用于稠密编码的 GPU（`--gpus`）与用于 ANN 构建的 GPU（`--ivf-gpus`）相互独立。

中小型语料建议先使用 `--dense-index-type flat`。大型语料可以从 `nprobe=32` 开始，并根据召回率和延迟进行调整。

### 从中断处恢复

```bash
python scripts_build_index/build_corpus.py \
  --name my-corpus \
  --build bm25,dense \
  --resume \
  --gpus 0 \
  --dense-shards 1
```

`--resume` 可以：

- 复用已经标准化的 corpus 和文档库；
- manifest 显示 BM25 完成时跳过该阶段；
- 复用完整的 dense shard；
- 补建缺失 shard；
- 除非命令行显式覆盖，否则从 manifest 继承模型、pooling、dtype、归一化和相关配置。

`--overwrite` 会删除已有 corpus 目录和索引树并重新构建。它不能与 `--resume` 同时使用，只有确定要替换已有内容时才应使用。

### 已清洗的自定义 Corpus

如果外部 materializer 已经生成：

```text
corpora/<name>/corpus.jsonl
corpora/<name>/documents.sqlite
```

可以不再传入原始文件，直接构建索引：

```bash
python scripts_build_index/build_corpus.py \
  --name my-cleaned-corpus \
  --build bm25,dense \
  --resume \
  --dense-model Qwen/Qwen3-Embedding-0.6B \
  --pooling eos \
  --gpus 0 \
  --dense-shards 1
```

这种方式适用于必须保留 URL、标题、来源、语言、字符数或其他源数据 metadata 的自定义 materializer。

### Pooling、归一化与模型兼容性

查询编码器必须使用与文档编码相同的模型、pooling、归一化、dtype 和 query prefix。BrowseComp-Plus 上游预构建 Qwen3-Embedding 索引和已经验证的 Wiki18 manifest 均使用 `pooling=eos` 和归一化向量。

不要把其他模型示例中的配置当作通用默认值。只要条件允许，应始终以 corpus manifest 为准。

## Wikimedia Enterprise YAML 流水线

该 YAML 流水线专门用于 Wikimedia Enterprise HTML `.json.tar.gz` 归档。普通 JSONL 和 DPR TSV 应直接使用 `build_corpus.py`。

流水线依次完成：

```text
发现或下载归档
→ 固化输入 manifest
→ 清洗并去重记录
→ 写入 corpus.jsonl 和 documents.sqlite
→ 构建 BM25
→ 编码稠密向量
→ 可选构建 IVF-Flat
→ 写入 manifest.json
```

从仓库根目录运行：

```bash
CONFIG=configs/corpora/wikimedia-enterprise.yaml

python scripts_build_index/run_corpus_pipeline.py \
  --config "$CONFIG"
```

### 完整配置示例

```yaml
config_version: 1

corpus:
  name: wikimedia-enterprise-20250320

source:
  type: web
  url: https://dumps.wikimedia.org/other/enterprise_html/runs/20250320/
  path: null
  recursive: true
  archive_pattern: "*-ENTERPRISE-HTML.json.tar.gz"
  include_files: []
  exclude_files: []
  download_dir: data/wikimedia-enterprise-20250320/downloads
  keep_downloads: true
  user_agent: "LocalCorpusRetrievalServer/1.0"

cleaning:
  min_chars: 1
  batch_size: 1000
  progress_every: 10000
  on_error: fail

build:
  targets:
    - bm25
    - dense
    - ann

  bm25:
    threads: 16

  dense:
    model: Qwen/Qwen3-Embedding-0.6B
    pooling: eos
    passage_max_len: 4096
    query_max_len: 8192
    batch_size: 32
    normalize: true
    dtype: fp16
    gpus: "0,1,2,3"
    shards: 4
    index_type: ivf-flat
    query_prefix: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

  ann:
    nlist: 16384
    nprobe: 32
    train_size: 819200
    seed: 42
    threads: 0
    niter: 25
    backend: gpu
    gpus: "0,1,2,3"
    gpu_batch_size: 8192
    gpu_dtype: float32

runtime:
  dry_run: false
  resume: true
  overwrite: false
  keep_build_files: false
```

GPU 列表必须加引号，确保 YAML 解析器向构建命令传递逗号分隔的字符串。

### 使用本地归档

如果归档已经存在于本地，请替换 source 配置：

```yaml
source:
  type: local
  url: null
  path: data/wikimedia-enterprise
  recursive: true
  archive_pattern: "*-ENTERPRISE-HTML.json.tar.gz"
  include_files: []
  exclude_files: []
  download_dir: null
  keep_downloads: true
  user_agent: "LocalCorpusRetrievalServer/1.0"
```

相对路径以仓库根目录为基准解析。

### 选择归档文件

筛选顺序如下：

```text
archive_pattern → include_files → exclude_files
```

这些字段使用区分大小写的 glob，而不是正则表达式。`exclude_files` 的优先级高于 `include_files`。

示例：

```yaml
# 只包含 namespace 0。
include_files:
  - "*-NS0-*"
exclude_files: []
```

```yaml
# 包含所有匹配归档，但排除模板和分类。
include_files: []
exclude_files:
  - "*-NS10-*"
  - "*-NS14-*"
```

### 重要 YAML 字段

| 字段 | 含义 | 默认值或建议 |
| --- | --- | --- |
| `corpus.name` | 输出 corpus 名称 | 必填 |
| `source.type` | `web` 或 `local` | 必填 |
| `source.archive_pattern` | 第一阶段归档筛选 | 默认 `*.tar.gz`；生产环境应使用更严格的模式 |
| `source.include_files` | 可选的允许列表 glob | `[]` |
| `source.exclude_files` | 可选的排除列表 glob | `[]` |
| `cleaning.min_chars` | 最短文本字符数 | `1` |
| `cleaning.batch_size` | SQLite 写入批次大小 | `1000` |
| `cleaning.on_error` | `fail` 或 `skip` | 首次验证建议使用 `fail` |
| `build.targets` | `bm25`、`dense` 和/或 `ann` | `bm25,dense` |
| `build.bm25.threads` | Lucene 索引线程数 | `16` |
| `build.dense.shards` | Embedding shard 数量 | 通常与 GPU 数量一致 |
| `build.dense.index_type` | `flat` 或 `ivf-flat` | 小型语料使用 `flat` |
| `build.ann.nprobe` | 每次查询的 IVF list 数量 | 建议从 `32` 开始 |
| `runtime.dry_run` | 只打印命令而不执行 | `false` |
| `runtime.keep_build_files` | 保留临时 Pyserini 输入 | `false` |

### 当前流水线限制

- `source.keep_downloads` 是预留字段；目前设为 `false` 也不会删除已下载归档；
- `runtime.overwrite` 尚未传递到全部三个阶段，应保持为 `false`；
- 流水线会使用 `--resume` 调用索引构建器，可恢复执行是预期运行方式；
- Web loader 只解析一级目录列表，不会递归抓取网页中的子目录链接。

## 构建产物与磁盘空间

| 产物 | 在线检索是否需要 | 重建是否需要 | 删除建议 |
| --- | --- | --- | --- |
| `corpora/<name>/documents.sqlite` | 文档读取和 metadata 需要 | 需要 | 保留 |
| `corpora/<name>/manifest.json` | `--corpus` 模式需要 | 需要 | 保留 |
| `indexes/corpora/<name>/bm25/` | BM25 需要 | 最终产物 | 保留 |
| `index.ivf-flat.faiss` | IVF-Flat 需要 | 最终产物 | 保留 |
| `index.lookup.pkl` | IVF-Flat 需要 | 最终产物 | 保留 |
| Flat 模式的 `corpus.shard*.pkl` | 需要 | 最终 Flat 索引 | 保留 |
| IVF-Flat 模式的 `corpus.shard*.pkl` | 不需要 | 可避免重建 ANN 时重新编码 | 验证完成后再归档或删除 |
| `.build/pyserini/docs.jsonl` | 不需要 | 可由 `corpus.jsonl` 重新生成 | BM25 成功后可安全删除 |
| 下载的源归档 | 不需要 | 可用于重新清洗 | 根据恢复需求归档或删除 |

`--keep-build-files` 会保留临时 Pyserini 输入。未设置该参数时，BM25 构建成功后会自动删除 corpus 的 `.build` 目录。

删除 IVF-Flat 使用过的 embedding shard 不会影响在线检索，但以后重建 ANN 时必须重新编码整个 corpus。如果删除 shard，还应同步更新 manifest 中的 `embeddings_built` 状态。

生成的 corpus、索引、模型、缓存、虚拟环境和运行日志不应提交到 Git。应该提交源代码、配置模板、`pyproject.toml`、`uv.lock` 和文档。

## 评测

### 可选：复现上游 BrowseComp-Plus 结果

`docs/` 下提供使用 OpenAI、Anthropic、Gemini、Qwen、OSS、GLM、Search-R1 和 Tongyi Agent 的说明。

### 评测 Deep-Research 运行结果

将每个查询对应的 JSON 文件放入 `runs/my-model/` 等运行目录。每个文件至少应包含：

```json
{
  "query_id": "query-id",
  "tool_call_counts": {"search": 3},
  "status": "completed",
  "retrieved_docids": ["doc-001"],
  "result": [
    {
      "type": "output_text",
      "output": "Final answer"
    }
  ]
}
```

执行评测：

```bash
python scripts_evaluation/evaluate_run.py \
  --input_dir runs/my-model
```

评测器使用 Qwen3-32B 作为 judge。详细信息参见 `docs/llm_as_judge.md`，并可通过 `--tensor_parallel_size` 配置 GPU 并行度。

### 评测纯检索结果

将结果保存为标准 TREC run 文件，然后分别针对 evidence 和 gold 标注评测：

```bash
TREC_RUN=runs/my-retriever.trec

python -m pyserini.eval.trec_eval \
  -c \
  -m recall.5,100,1000 \
  -m ndcg_cut.10 \
  topics-qrels/qrel_evidence.txt \
  "$TREC_RUN"

python -m pyserini.eval.trec_eval \
  -c \
  -m recall.5,100,1000 \
  -m ndcg_cut.10 \
  topics-qrels/qrel_golds.txt \
  "$TREC_RUN"
```

Evidence 文档指回答查询时需要检索到的文档；Gold 文档不仅是回答所必需的，而且在语义上包含最终答案。

### 轨迹数据

下载已经发布的 baseline trajectory：

```bash
bash scripts_build_index/download_run_files.sh
```

解密后的 trajectory 位于 `data/decrypted_run_files/`。

## 相关文档

- [OpenAI API](docs/openai.md)
- [Gemini API](docs/gemini.md)
- [Anthropic API](docs/anthropic.md)
- [GLM API](docs/glm.md)
- [OSS](docs/oss.md)
- [Qwen](docs/qwen.md)
- [Search-R1](docs/search-r1.md)
- [Tongyi DeepResearch](docs/tongyi.md)
- [自定义检索器](docs/custom_retriever.md)
- [相互独立的本地 corpus](docs/local_corpora.md)
- [LLM-as-a-judge 评测](docs/llm_as_judge.md)
- [NetMind MCP](docs/netmind_mcp.md)

## 上游致谢与引用

本服务基于并扩展了 BrowseComp-Plus 发布的检索与评测组件。本文档描述的本地 corpus 流水线、manifest 配置、SQLite 文档库、并发 MCP 服务、可恢复构建和 IVF-Flat 工作流属于当前搜索服务器项目，不应与上游基准本身的功能范围混淆。

如果使用 BrowseComp-Plus 数据集、预构建索引、评测数据或其他上游研究产物，请引用原论文：

```bibtex
@article{chen2025BrowseCompPlus,
  title={BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent},
  author={Chen, Zijian and Ma, Xueguang and Zhuang, Shengyao and Nie, Ping and Zou, Kai and Liu, Andrew and Green, Joshua and Patel, Kshama and Meng, Ruoxi and Su, Mingyi and Sharifymoghaddam, Sahel and Li, Yanxi and Hong, Haoran and Shi, Xinyu and Liu, Xuye and Thakur, Nandan and Zhang, Crystina and Gao, Luyu and Chen, Wenhu and Lin, Jimmy},
  year={2025},
  journal={arXiv preprint arXiv:2508.06600}
}
```

有关本搜索服务器的问题，请使用当前仓库的 issue tracker；有关上游基准的问题，请使用 BrowseComp-Plus 项目公布的联系方式。

本项目依据 [LICENSE](LICENSE) 中的条款发布。
