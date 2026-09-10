# 本地语料检索服务器

[English](README.md) | [简体中文](README.zh-CN.md)

这是一个同时支持固定本地语料和实时网页 API 的自托管检索服务器，整合 BM25、Dense Flat/IVF-Flat、文档读取、语料导入、索引构建和 MCP 服务。本地检索用于私有语料、精选数据集和可复现场景，不依赖网页搜索；可选的智谱和阿里云 IQS 适配器通过同一套 Tool 字段提供实时搜索与按需网页阅读。

本服务基于 [BrowseComp-Plus](https://texttron.github.io/BrowseComp-Plus/) 的检索代码开发，但它是一个独立的工程项目，并不是 BrowseComp-Plus 基准本身。项目仍然兼容 BrowseComp-Plus 的数据集、预构建索引和评测工具，可将其作为可选的数据源与复现任务。

BrowseComp-Plus 上游资源：[数据集](https://huggingface.co/datasets/Tevatron/browsecomp-plus) · [排行榜](https://huggingface.co/spaces/Tevatron/BrowseComp-Plus) · [论文](https://arxiv.org/pdf/2508.06600) · [项目主页](https://texttron.github.io/BrowseComp-Plus/)

本地检索服务器提供：

- 基于 Pyserini/Lucene 的 BM25 检索；
- 基于 FAISS Flat 的稠密精确检索；
- 面向大规模语料的 FAISS IVF-Flat 近似检索；
- 相互独立、由 manifest 管理的本地 corpus；
- 用于受控全文读取的 SQLite 文档库；
- 为离线和在线提供一致 `search` / `read_document` 字段的 MCP 服务；
- 智谱和 IQS 适配器，支持稳定 URL ID、按需阅读及有上限的临时缓存；
- 统一 YAML 服务选择、就绪提示、日志和整组查看/停止；
- 可恢复的语料标准化与索引构建流程；
- 面向 Wikimedia Enterprise HTML 归档的 YAML 流水线。

## 目录

- [检索系统概览](#检索系统概览)
- [快速开始](#快速开始)
- [环境安装](#环境安装)
- [数据集与预构建索引](#数据集与预构建索引)
- [启动检索服务](#启动检索服务)
- [工具输出字段](#工具输出字段)
- [在线 API 与缓存](#在线-api-与缓存)
- [新增 Provider 流程](#新增-provider-流程)
- [测试与验收](#测试与验收)
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
| 在线检索 | 智谱 / 阿里云 IQS | 实时搜索与同服务商网页阅读 |
| 在线缓存 | 进程内映射及正文 LRU/空闲 TTL | 跨连接读取，避免反复抓取 |
| 服务启动器 | `searcher/serve.py` + YAML | 选择服务、检查就绪、查看/停止整组 |
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

### 4. 配置并启动 MCP 服务

按[统一 YAML 示例](#统一-yaml-示例)创建本地配置；已有部署应保留现有 YAML 和凭据。示例使用 BrowseComp-Plus 索引、Dense GPU 6，以及提前准备好的 `/workspace/hf_cache`，请根据机器修改。

```bash
python3 searcher/serve.py --config configs/services.local.yaml --services bm25 dense --check
python3 searcher/serve.py --config configs/services.local.yaml --services bm25 dense
```

等待两个服务都显示 `READY`。BM25 地址为 `http://127.0.0.1:8080/mcp`，Dense 地址为 `http://127.0.0.1:8082/mcp`。不要在另一个终端重复启动同一配置，应使用 `--status` 或 `--stop`。

如果只使用在线搜索，不必构建本地索引：准备智谱/IQS 凭据后选择 `--services online`、`--services iqs` 或 `--services online iqs`。在线服务不依赖本地 corpus，也不需要 snippet tokenizer；snippet 直接按 Unicode 字符截断。详见[在线设置](#在线-api-与缓存)和[测试与验收](#测试与验收)。

离线与在线都提供 `search(query, k=5)` 和 `read_document(docid, chunk_index=0)`，采用[相同的成功结果字段](#工具输出字段)。离线需设置 `get_document: true` 才会注册读取工具；每次读取最多返回 8192 个 Unicode 字符。

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

推荐使用 `searcher/serve.py` 和一个 YAML 文件统一启动 BM25、Dense Flat/IVF-Flat、智谱和 IQS。先配置，再按服务名选择启动；修改配置后重启生效，不支持运行中热切换。以下命令均在运行环境的仓库根目录执行；使用 Docker 时就是容器内部。

### 统一 YAML 示例

首次使用仓库时，从不含凭据的模板创建本地配置：

```bash
cp -n configs/services.example.yaml configs/services.local.yaml
chmod 600 configs/services.local.yaml
```

`cp -n` 不会覆盖已有本地配置。请在本地设置凭据，并调整路径和 GPU 编号。本地配置及其备份已忽略，只提交脱敏后的 example 模板。


**新部署**可以将以下内容保存为 `configs/services.local.yaml`。不要直接覆盖已经使用的本地配置，其中可能包含 key、实际路径和调优参数。这是部署示例，不表示所有仓库版本都自带该文件。请按机器调整缓存、索引路径和 GPU 编号。Dense 示例使用 GPU **6**、BrowseComp-Plus Flat shards 和本地 SQLite 文档库。

```yaml
version: 1

defaults:
  runtime:
    hf_home: /workspace/hf_cache
    hf_offline: true
  args:
    transport: sse
    k: 5
    snippet_max_chars: 2048

services:
  bm25:
    backend: bm25
    args:
      port: 8080
      index_path: indexes/bm25
      document_store: corpora/browsecomp/documents.sqlite
      get_document: true
      bm25_workers: 16

  dense:
    backend: faiss
    runtime:
      cuda_visible_devices: "6"
    args:
      port: 8082
      index_type: flat
      index_path: "indexes/qwen3-embedding-0.6b/corpus.shard*.pkl"
      model_name: Qwen/Qwen3-Embedding-0.6B
      pooling: eos
      normalize: true
      torch_dtype: float16
      document_store: corpora/browsecomp/documents.sqlite
      get_document: true
      dense_max_batch_size: 16
      dense_max_wait_ms: 1
      dense_max_queue_size: 64

  online:
    backend: online
    args:
      host: 127.0.0.1
      port: 8081
      content_cache_idle_ttl_seconds: 1800
      content_cache_max_mib: 256
      mapping_idle_ttl_seconds: 86400
      mapping_max_entries: 100000
      cache_cleanup_interval_seconds: 60
    provider:
      name: zhipu
      api_key: ""
      api_key_env: ZHIPU_API_KEY
      base_url: https://open.bigmodel.cn/api/paas/v4/
      timeout_seconds: 60
      search:
        search_engine: search_std
        search_intent: false
        count: 10  # Fallback for direct Provider calls; MCP passes per-call k.
        search_recency_filter: noLimit
        content_size: medium
      read:
        no_cache: true
        return_format: text
        retain_images: false
        with_images_summary: false
        with_links_summary: false

  iqs:
    backend: online
    args:
      host: 127.0.0.1
      port: 8083
      content_cache_idle_ttl_seconds: 1800
      content_cache_max_mib: 256
      mapping_idle_ttl_seconds: 86400
      mapping_max_entries: 100000
      cache_cleanup_interval_seconds: 60
    provider:
      name: iqs
      api_key: ""
      api_key_env: IQS_API_KEY
      base_url: https://cloud-iqs.aliyuncs.com/
      timeout_seconds: 60
      search:
        engineType: LiteAdvanced
        timeRange: NoLimit
        advancedParams:
          numResults: 10  # Fallback for direct Provider calls; MCP passes per-call k.
      read:
        pageTimeout: 10000
        maxAge: 0
        readability:
          readabilityMode: normal
```

示例在线服务仅监听本机回环地址，便于本地测试。目前离线 HTTP 入口固定监听 `0.0.0.0`，不接受 `args.host`；请通过 Docker 端口发布和防火墙限制访问。这些示例不提供鉴权和 TLS，不要将服务直接暴露到不可信网络。

### 启动、查看和停止

```bash
# 检查配置结构和本地路径；不加载模型，不调用 API。
python3 searcher/serve.py --config configs/services.local.yaml --all --check

# 对同一个配置选择一种启动方式。
python3 searcher/serve.py --config configs/services.local.yaml --services bm25 dense
# 也可以选择：--services online iqs
# 或：--services bm25 iqs
# 或：--all
```

`--all` 会启动文件内的全部服务，因此配置同时包含两家在线服务时，两家的凭据都需要准备好。`--services` 后面用空格分隔名称，不是逗号分隔。示例地址：

如果不选择子集，而是启动配置中的全部四个服务：

```bash
python3 searcher/serve.py --config configs/services.local.yaml --all
```

| 服务名 | 后端/服务商 | 地址 |
| --- | --- | --- |
| `bm25` | BM25 | `http://127.0.0.1:8080/mcp` |
| `dense` | FAISS | `http://127.0.0.1:8082/mcp` |
| `online` | 智谱 | `http://127.0.0.1:8081/mcp` |
| `iqs` | IQS | `http://127.0.0.1:8083/mcp` |

启动器保持在前台，依次显示 `CHECK`、`STARTING`、`WAITING` 和 `READY`。`READY` 表示已经成功建立 MCP 连接并列出预期工具，**不代表** API key、上游搜索/阅读、检索质量或吞吐量已经通过验证。启动等待时限默认为 `--startup-timeout 900`，单位秒。子进程退出或启动失败会停止本次启动的其他子进程；目前没有自动重启策略。

在另一个终端中，使用**相同运行环境/容器和配置路径**：

```bash
python3 searcher/serve.py --config configs/services.local.yaml --status
python3 searcher/serve.py --config configs/services.local.yaml --stop
```

`--status` 查看已记录的进程及状态，不会重新执行端到端 API 健康测试。`--stop` 停止记录中的整组服务，不能指定只停止其中一家；不要与 `--services`、`--all` 或 `--check` 混用。前台 `Ctrl+C` 也会停止本组服务。服务组以解析后的配置文件路径标识，同一配置只能有一个启动器。需要变更启动组合时，先停整组，再以新组合启动。需要独立管理时，使用不同配置文件和互不冲突的端口。此前由 `services.iqs.local.yaml` 启动的 IQS，必须用旧配置路径停止后再从统一配置启动；合并 YAML 不会转移进程归属。

日志位于 `logs/services/<run-id>/<service>.log`，启动器会打印完整路径；状态和锁文件位于 `logs/services/.state/`。整组管理依赖 Linux `/proc`、进程组、文件锁和 PID 身份检查。启动器不是守护进程管理器，不保证关闭终端后仍持续运行，也不会自动接管遗留进程。

### YAML 字段与继承规则

| 字段 | 含义 |
| --- | --- |
| `version` | 必填，目前为 `1` |
| `defaults.runtime`、`defaults.args` | 公共配置；只放所有继承服务都接受的字段 |
| `services.<name>.backend` | `bm25`、`faiss` 或 `online`；Dense ANN 仍然使用 `faiss` |
| `services.<name>.runtime` | 该子进程的环境配置 |
| `services.<name>.args` | 子进程 CLI 参数，采用下划线：`snippet_max_chars` → `--snippet-max-chars` |
| `services.<name>.provider` | 仅在线使用；`name` 选择已注册适配器，其他字段配置该适配器 |

服务配置递归覆盖 defaults。列表和标量整体替换；没有设置的字段仍然继承。重复 YAML key、未知顶层字段、服务字段和 runtime 字段会被拒绝。`args` 只接受标量：`true` 生成开关，`false`/`null` 不传该参数，其他值生成 `--name=value`。不传参数并不等于否定后端自身默认值或 manifest 设置。Provider 的嵌套参数放在 `provider`，不要放在 `args`。

`--config` 本身相对于当前工作目录解析；子进程始终从仓库根目录运行。示例中的相对索引、语料和缓存路径以仓库根目录为基准，而不是以 YAML 所在目录为基准。YAML **不会**展开 `$VARIABLE` 或 `${VARIABLE}`，应使用 `api_key_env` 或实际路径。

| `runtime` 字段 | 行为 |
| --- | --- |
| `hf_home` | 设置 `HF_HOME`、`HF_HUB_CACHE`、`HUGGINGFACE_HUB_CACHE`、`HF_DATASETS_CACHE`，移除继承的 `TRANSFORMERS_CACHE` |
| `hf_offline` | 将 `HF_HUB_OFFLINE` 和 `TRANSFORMERS_OFFLINE` 设置为 `1`/`0` |
| `cuda_visible_devices` | 设置 `CUDA_VISIBLE_DEVICES`；GPU 编号/列表加引号，例如 `"6"`、`"0,1"` |
| `env` | 额外环境变量，值为字符串或整数；专用 runtime 字段优先级更高 |

子进程继承启动器的环境。HF 离线模式只影响模型/tokenizer 下载，**不会**禁止 HTTP 搜索和阅读 API。BM25、Dense 和在线 MCP 的 snippet 现在直接按字符截断，不再加载 snippet tokenizer。Dense 查询编码仍然需要 embedding 模型及其 tokenizer，应提前准备缓存或使用兼容的本地 `model_name` 目录。公共 HF 离线设置可以为 Dense 保留；首次下载模型和索引时不要开启 HF 离线模式。

| 通用 `args` 字段 | 默认值/说明 |
| --- | --- |
| `port` | 统一启动器要求显式配置，所选服务不能重复 |
| `transport` | 默认 `sse`；整组启动也支持 `streamable-http`，不支持 `stdio` |
| `k` | 默认 `5`；Agent 省略 Tool 参数 k 时的默认值，不是服务固定返回上限 |
| `snippet_max_chars` | 每条结果默认最多 `2048` 个 Unicode 字符；`-1` 关闭截断，其他值须为正整数 |
| `get_document` | 仅离线使用；`true` 注册 `read_document`；在线始终注册 |
| `host` | 仅在线使用；在线入口默认 `0.0.0.0`，示例设置为 `127.0.0.1` |

`--check` 不会创建 Provider、检查每个后端的全部 CLI 参数、加载索引/模型、检查端口占用或请求上游 API。因此未知子进程参数以及更深层配置错误仍可能在启动时才发现。不要设置 `args.config`、`args.searcher_type`、`args.hf_token`，分别使用 `provider`、`backend` 和 `HF_TOKEN` 环境变量。

### 离线：Manifest、Flat 与 ANN

统一 YAML 保留原有两种索引配置方式：

- **Manifest 模式**：设置 `args.corpus: my-corpus`，由 `corpora/my-corpus/manifest.json` 提供索引/模型信息。启动器禁止将 `corpus` 与 `index_path`、`model_name`、`pooling`、`normalize`、`torch_dtype`、`lookup_path`、`index_type`、`nprobe`、`task_prefix`、`max_length` 同时使用，以 manifest 为准。BM25 可显式配置 `document_store` 用于读取。snippet 不再从 manifest 推导或加载 tokenizer；Dense 查询编码保持不变。
- **显式模式**：使用完整示例中的路径和模型参数。查询模型、pooling、归一化、dtype、query prefix 必须与构建索引时保持兼容。

| 离线 `args` | 用途 |
| --- | --- |
| `index_path`、`document_store` | BM25 Lucene 目录或 FAISS 输入；用于文档读取的 SQLite 库 |
| `bm25_workers` | BM25 工作线程，默认 `1`，部署示例 `16` |
| `index_type` | FAISS `flat`（默认）或 `ivf-flat` |
| `lookup_path`、`nprobe` | IVF-Flat docid 映射及探测列表数；`nprobe` 默认 `32` |
| `model_name`、`pooling`、`normalize`、`torch_dtype` | 查询编码参数，与已有索引匹配 |
| `task_prefix`、`max_length` | 查询指令及 token 长度；FAISS 默认长度 `8192` |
| `corpus_path`、`dataset_name` | FAISS 的替代文档来源；建议使用本地 `document_store`，避免加载远端数据集 |
| `dense_max_batch_size` | 每批最多请求数，默认 `16` |
| `dense_max_wait_ms` | 默认 `5` ms，部署示例 `1` ms |
| `dense_max_queue_size` | 等待队列容量，默认 `64` |

ANN 是 Dense 的索引类型，不是额外的在线 Provider。使用已构建 IVF-Flat 索引时，将 Dense 服务中的 Flat 索引字段替换为：

```yaml
index_type: ivf-flat
index_path: indexes/corpora/my-corpus/qwen3-embedding-0.6b/index.ivf-flat.faiss
lookup_path: indexes/corpora/my-corpus/qwen3-embedding-0.6b/index.lookup.pkl
nprobe: 32
```

还需要使用对应 corpus 的文档库和匹配的模型参数。`nlist` 等训练参数属于索引构建，不属于服务启动参数。不要向 Flat 检索传入 IVF 专用路径。

原有直接启动入口仍然保留，用于调试和离线 `stdio` 客户端：

```bash
python3 searcher/mcp_server.py --help --searcher-type bm25
python3 searcher/mcp_server.py --help --searcher-type faiss
python3 searcher/online_mcp_server.py --help
```

直接运行在线入口时，`--config` 接收形如 `{"provider":"iqs","providers":{"iqs":{...}}}` 的 JSON，**不是**服务 YAML。统一启动器自动将 YAML 转为权限为 `600` 的临时 JSON；日常使用推荐统一启动器。

已验证 BrowseComp 部署的历史离线性能参考：BM25 约 43 QPS（p50 约 176 ms、p95 约 251 ms），Dense 约 35–36 QPS（p50 约 862 ms、p95 约 1.1 s）。这些不是本次新测结果，也不是在线 API 的吞吐量承诺；只应在相同语料、硬件和并发条件下比较。

## 工具输出字段

### `search(query: str, k: int = 5)`

返回最多 `k` 条结果的 JSON 数组：

```json
[
  {"docid": "web_<sha256-of-url>", "score": null, "snippet": "受长度限制的预览文本。"}
]
```

| 字段 | 类型/说明 |
| --- | --- |
| `docid` | 字符串；离线为语料 ID，在线为 URL 生成的 ID |
| `score` | 数值或 `null`；在线目前固定返回 `null`，即使上游 API 提供评分也不透出 |
| `snippet` | 预览字符串，按 Unicode 字符数截断，不依赖 snippet tokenizer |

Search **不返回** `metadata`、URL/title/source/language、`char_count` 或分块数。每条结果直接执行 `text[:snippet_max_chars]`（离线文档正文）或 `snippet[:snippet_max_chars]`（在线 API 摘要），按 Python Unicode 码点计数，不是 UTF-8 字节、token、词或自然段。原先的 8192 字符预截断和 tokenizer 编解码两步都已移除；短文本原样保留，`-1` 返回完整来源字符串。这不改变独立的 `read_document` 8192 字符分块。

Agent 能看到大于 0 的整数参数 `k`；schema 默认值在启动时取自 YAML `args.k`，示例为 5。例如 `{"query":"history","k":7}` 请求最多 7 条，省略 k 则使用默认值。BM25 将 k 传给 Lucene，Dense 将其传给 FAISS；同一 Dense batch 中不同 k 的请求按最大 k 检索，再各自裁剪到所请求数量。在线把 k 传给 Provider，并将公共结果限制在 k 条以内。可用结果不足时允许少于 k；IQS 拒绝 1–50 以外的值，其他服务商仍受自身规则限制。

迁移方式：将 `snippet_max_tokens: 512` 替换为 `snippet_max_chars: 2048`，删除 defaults 和各服务覆盖配置中的 `snippet_tokenizer`。旧 CLI 参数改为 `--snippet-max-chars`，旧 MCP snippet 参数不再接受。2048 字符只是原 512 tokens 的粗略替代，不是跨语言等价换算。修改配置后重启；客户端可能需要重新连接或刷新 Tool schema。独立的 `search_r1_server.py` 不在此次 MCP 改动范围内。

### `read_document(docid: str, chunk_index: int = 0)`

下面是字段和长度完全一致的最后一块示例：

```json
{
  "docid": "doc-example",
  "chunk_index": 0,
  "metadata": {"char_count": 5, "chunk_size_chars": 8192, "chunk_count": 1},
  "char_start": 0,
  "char_end": 5,
  "has_more": false,
  "next_chunk_index": null,
  "text": "Hello"
}
```

- 分块索引从 0 开始。`char_start` 包含起点，`char_end` 不包含终点；按 Python Unicode 码点计算，满足 `len(text) == char_end - char_start`，不是 UTF-8 字节、token 或自然段边界。
- MCP 分块大小固定为 `8192`，不是 YAML 可调参数。`chunk_count = max(1, ceil(char_count / 8192))`。
- 当 `has_more` 为 true 时按 `next_chunk_index` 继续读取；最后一块的 `next_chunk_index` 为 `null`。
- Read 的 `metadata` 仅包含 `char_count`、`chunk_size_chars`、`chunk_count`。字数描述当前可用文档文本；在线时是 Provider 返回的正文，而不是独立验证过的网页完整副本。
- 未知/过期在线 docid、非法/越界 chunk 会报错。离线基础读取器在文档不存在时可能返回 `null`；不同后端的错误返回尚未完全统一，客户端应同时处理工具错误和缺失结果。

旧客户端迁移：调用工具 `read_document`，而不是 `get_document`；`get_document: true` 只是离线注册工具的旧参数名。把长度和分块信息的读取从 search 移到 read，允许 `score: null`，不要再依赖已删除的公共 URL/title/source/language 字段。内部 metadata 和文档库列并没有因此删除。

## 在线 API 与缓存

### 凭据与服务商配置

现有适配器将搜索与阅读绑定到同一家服务商。`provider.name` 是注册表名称，不是任意 Python 导入路径。非空 `provider.api_key` 优先；否则读取 `api_key_env` 指定的环境变量。没有 `${...}` 替换，在线入口也不会自动加载 `.env` 文件。

| 通用 `provider` 字段 | 含义 |
| --- | --- |
| `name` | 注册的适配器，目前为 `zhipu` 或 `iqs` |
| `api_key`、`api_key_env` | 直接保存的凭据，或保存凭据的环境变量名称 |
| `base_url` | 可信的服务商 API 基础地址；默认值见完整示例 |
| `timeout_seconds` | HTTP 客户端超时，默认 `60` 秒；IQS 还会校验其与网页超时的关系 |
| `search`、`read` | 服务商特有请求选项，未知选项名会被拒绝 |

下面将 key 写入现有 YAML，不回显、不把 key 放入命令历史。提示时输入服务名 `iqs` 或 `online`：

```bash
python3 - <<'PY'
import getpass
from pathlib import Path
import yaml
from searcher.serve import UniqueLoader

path = Path("configs/services.local.yaml")
config = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
# stdin contains this script; use the terminal explicitly for interactive input.
with open("/dev/tty", "r+") as terminal:
    terminal.write("Service name (iqs or online): ")
    terminal.flush()
    name = terminal.readline().strip()
if name not in {"iqs", "online"} or name not in config["services"]:
    raise SystemExit("Unknown service; unchanged")
key = getpass.getpass("API key (hidden): ").strip()
if not key:
    raise SystemExit("Empty key; unchanged")
config["services"][name]["provider"]["api_key"] = key
path.chmod(0o600)
path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
print("Saved. Restart the service group to apply.")
PY
```

该命令会重写 YAML 排版和注释。本地配置、备份、临时凭据文件以及运行日志不要进入版本控制或共享产物；应检查排除规则，不要默认认为已经被忽略。公开文档只放脱敏示例，不要在问题报告中提供 key。API 地址应使用可信 HTTPS 端点，尤其不要随意修改 `base_url`。

| 适配器 | 搜索 | 阅读 | 参数差异 |
| --- | --- | --- | --- |
| `zhipu` | `web_search` → `search_result` | `reader` → `reader_result.content` | `search_engine`、`search_intent`、`count`、`search_recency_filter`、`content_size` |
| `iqs` | `search/unified` → `pageItems` | `readpage/basic` → `data.text` | `engineType`、`timeRange`、可选 `category`、`advancedParams` |

智谱 `search_recency_filter` 可使用 `oneDay`、`oneWeek`、`oneMonth`、`oneYear`、`noLimit`。阅读参数为 `no_cache`、`return_format`（必须为 `text`）、`retain_images`、`with_images_summary`、`with_links_summary`。

IQS `timeRange` 使用 `OneDay`、`OneWeek`、`OneMonth`、`OneYear`、`NoLimit`，注意大小写与智谱不同。适配器接受 `Generic`、`GenericAdvanced`、`LiteAdvanced`、`Deep` 引擎。`advancedParams` 支持 `numResults`（1–50；所选引擎可能有自己的返回规则）、`startPublishedDate`、`endPublishedDate`（`YYYY-MM-DD`）。搜索请求关闭 `mainText`、`markdownText`、`richMainBody`、`summary`，只使用 snippet；请求 `rerankScore` 用于上游排序，但不将其透出为公共 `score`。

IQS 阅读目前**只接入 Basic**。YAML 可配置 `pageTimeout`（0–100000 毫秒）、`maxAge`（非负秒数）、`readability.readabilityMode`（`none`、`normal`、`article`）、`excludeAllImages`、`excludeAllLinks`、`excludedTags`。`formats: [text]` 固定在适配器中，不可配置。`timeout_seconds` 必须大于 `(pageTimeout + 6000) / 1000`。目标网页的 `data.statusCode` 非成功值或正文为空时返回错误，不缓存成正常文档。搜索接口的长文本字段不用于替代 lazy read。

MCP 调用时会用本次请求的 k（省略则用 YAML `args.k`）覆盖智谱 `search.count` 和 IQS `advancedParams.numResults`。Provider 配置中的数量只为不传 k 的直接适配器调用保留默认值，不再将 MCP 请求固定或限制为 10 条。上游限制、过滤或空结果可能导致不足 k 条。智谱 `read.no_cache: true` 和 IQS `read.maxAge: 0` 关闭的是服务商侧缓存，**不是**本服务缓存。网页访问限制、动态渲染或服务商提取行为仍可能导致可用正文不完整。

官方参考：[智谱 Web Search](https://open.bigmodel.cn/api/paas/v4/web_search)（API 端点）、[IQS HTTP 快速入门](https://help.aliyun.com/zh/document_detail/2870227.html)、[IQS 搜索参数](https://help.aliyun.com/zh/document_detail/2883041.html)、[IQS Basic 阅读](https://help.aliyun.com/zh/document_detail/2983380.html)。

### 生命周期与容量限制

```text
search(query, k) → Provider.search(query, k) → 注册 URL → 稳定 docid + snippet
read_document → 查 docid → 缓存正文或 Provider.read(URL)
              → 计算字符数 → 返回指定的 8192 字符分块
```

ID 为 `web_` 加**原始 URL 字符串**的 SHA-256 十六进制摘要，不做 URL 规范化。同一内容的不同 URL 可能生成不同 ID。只注册真正返回给 Agent 的搜索结果；相同 URL 会生成相同 ID，但读取时仍要求该服务进程内存在有效映射。

| 在线 `args` 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| `content_cache_idle_ttl_seconds` | `1800` | 正文空闲过期时间；成功读取缓存后刷新访问时间 |
| `content_cache_max_mib` | `256` | 缓存正文 UTF-8 字节总量，单位 MiB；超出时按 LRU 淘汰 |
| `mapping_idle_ttl_seconds` | `86400` | docid → URL 映射的空闲过期时间 |
| `mapping_max_entries` | `100000` | 映射条目上限；淘汰最久未使用项 |
| `cache_cleanup_interval_seconds` | `60` | 后台清理间隔；请求路径也会清理过期项 |

上述值都必须大于 0，容量和条目数字段为整数。建议映射 TTL 明显长于正文 TTL。搜索注册和对已知 ID 的读取尝试会刷新映射访问时间。淘汰正文仍保留映射；淘汰映射则同时删除其正文缓存。映射不存在时需重新搜索。单篇正文超过全部缓存容量会直接报错，不会截断正文以塞入缓存。

字节预算**不是**进程 RSS 上限：Python 对象、metadata、映射和正在抓取的响应都有额外开销。同一进程内并发读取同一个未缓存 ID 会共享一次抓取；缓存没有对不同 ID 实施全局请求限速。失败的抓取不缓存，后续请求可以重新尝试；适配器本身没有自动重试/退避循环。

缓存作用域是单个服务进程，跨 MCP 连接共享，不是按连接或 rollout 隔离。当前没有 rollout 结束通知、持久化、跨 worker 共享或租户隔离。客户端断开不会释放缓存，重启进程会清空正文和映射。正文过期重新抓取后，网页可能变化，旧 chunk 编号可以对应不同内容；当前有意不提供版本固定。

## 新增 Provider 流程

新服务商需要写一小段适配代码；已经注册的服务商只需修改 YAML 并重启。不能只靠字段名替换来描述任意鉴权、嵌套响应和错误语义。

1. 新建 `searcher/providers/<vendor>.py`，类的构造函数接收服务商配置字典。检查允许的字段和值，读取凭据，持有一个可复用的异步 HTTP 客户端。
2. 实现下面的内部规范接口，包括每次调用的 k。将 k 映射为服务商数量参数，构造新的请求字典（嵌套参数也要复制），不要在并发调用中修改共享配置。显式校验服务商允许的范围，不要忽略 k。`read` 返回提取后的正文，而不是搜索 snippet。不要求上游端点一定叫 Reader，但正文淘汰后仍须能合理地重新读取。如果某服务商只在搜索中附带正文，需要明确设计重新获取策略，不能静默把摘要当全文返回。
3. 在 `searcher/providers/registry.py` 导入并注册，例如在原有条目之外增加 `PROVIDERS["vendor"] = VendorProvider`。无具体需要时不修改 `__init__.py` 和 `online_provider.py` 兼容入口。
4. 添加服务，设置 `backend: online`、空闲端口和 `provider.name: vendor`。服务商特殊设置放在 `provider.search`/`provider.read`，公共缓存和 snippet 设置继续放在 `args`。目前时间过滤是启动配置，不是 Agent `search` 工具的新参数。
5. 执行适配器 mock 测试、下方公共缓存测试，以及使用真实 key 的 MCP 搜索/阅读验收。修改代码或配置后重启。公共 Tool 和启动器不应为每家服务商增加特殊分支。

```python
class VendorProvider:
    def __init__(self, config: dict): ...

    async def search(self, query: str, k: int) -> list[dict]:
        # Pass k to the upstream request without mutating shared options.
        # Required per item: url (absolute HTTP(S)), snippet (string).
        # Optional: title, source, language, publish_date.
        ...

    async def read(self, url: str) -> dict:
        # Required: text (nonempty string).
        # Optional: url, title, source, language.
        ...

    async def close(self) -> None:
        # Close owned HTTP clients/resources.
        ...
```

HTTP/网络/API 失败、响应格式损坏、目标网页受限或正文缺失都应报错。确实没有搜索结果时返回 `[]`，不要将 API 失败伪装为空结果成功。避免阻塞式 I/O，不记录凭据或完整响应。HTTP 错误处理、超时/限流策略、服务商默认参数、提取限制以及语言/标题映射归适配器负责；ID、公共 `score: null`、snippet 截断、缓存和分块由公共层负责。

适配器 mock 至少检查 URL/方法/鉴权/请求体、嵌套响应映射、空搜索、缺失或非法字段、非 2xx 响应、服务商错误码、Reader 目标页失败、长 Unicode 正文和客户端资源关闭。可以使用 `httpx.MockTransport`，不需要 key、不产生费用。公共缓存测试不能代替服务商响应适配测试。

## 测试与验收

### 1. 静态和启动检查

```bash
python3 -m py_compile searcher/serve.py searcher/tools.py searcher/searchers/base.py \
  searcher/online_mcp_server.py searcher/online_service.py \
  searcher/providers/base.py searcher/providers/registry.py \
  searcher/providers/zhipu.py searcher/providers/iqs.py
python3 searcher/serve.py --config configs/services.local.yaml --all --check
```

启动所选服务组，等待 `READY`。这些检查不会发起付费搜索/阅读请求；下面的真实 API 测试会请求上游，结果具有实时性，不能保证固定不变。

### 2. 搜索、跨连接阅读与分块边界

测试显式使用 `SSETransport` 和 `/mcp`。`/sse` 不是本服务地址；对 SSE 服务直接使用 `Client(url)` 可能误选 Streamable HTTP，产生 HTTP 405。如果服务明确以 `streamable-http` 启动，则改用 `Client(URL)`。

在仓库根目录保存并执行以下代码，会打印完整结果并检查字段。修改 `MCP_URL` 可测试其他服务；`EXPECT_ONLINE=0` 允许离线数值 score，`REQUESTED_K` 指定本次请求条数，`SNIPPET_MAX_CHARS` 应与服务端字符上限一致。

```bash
cat > /tmp/retrieval_mcp_smoke.py <<'PY'
import asyncio
import json
import os

from fastmcp import Client
from fastmcp.client.transports import SSETransport

URL = os.environ.get("MCP_URL", "http://127.0.0.1:8083/mcp")
QUERY = os.environ.get("TEST_QUERY", "Context Folding LLM Agent")
ONLINE = os.environ.get("EXPECT_ONLINE", "1") == "1"
K = int(os.environ.get("REQUESTED_K", "7"))
SNIPPET_MAX_CHARS = int(os.environ.get("SNIPPET_MAX_CHARS", "2048"))


def decode(result):
    if getattr(result, "is_error", False) or getattr(result, "isError", False):
        raise RuntimeError(str(result))
    blocks = result.content if hasattr(result, "content") else result
    for block in blocks:
        if getattr(block, "type", None) == "text":
            try:
                return json.loads(block.text)
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"No JSON text in result: {result}")


async def call(name, arguments):
    async with Client(SSETransport(URL)) as client:
        return decode(await client.call_tool(name, arguments))


def check_chunk(result, docid, index):
    assert set(result) == {
        "docid", "chunk_index", "metadata", "char_start", "char_end",
        "has_more", "next_chunk_index", "text",
    }, result
    assert result["docid"] == docid and result["chunk_index"] == index
    metadata = result["metadata"]
    assert set(metadata) == {"char_count", "chunk_size_chars", "chunk_count"}
    total = metadata["char_count"]
    count = metadata["chunk_count"]
    assert type(total) is int and total >= 0
    assert metadata["chunk_size_chars"] == 8192
    assert count == max(1, (total + 8191) // 8192)
    assert result["char_start"] == index * 8192
    assert result["char_end"] == min((index + 1) * 8192, total)
    assert len(result["text"]) == result["char_end"] - result["char_start"]
    assert result["has_more"] is (index + 1 < count)
    assert result["next_chunk_index"] == (index + 1 if index + 1 < count else None)


async def main():
    rows = await call("search", {"query": QUERY, "k": K})
    print("=== search ===")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    assert isinstance(rows, list) and 0 < len(rows) <= K
    for row in rows:
        assert set(row) == {"docid", "score", "snippet"}
        assert isinstance(row["docid"], str) and isinstance(row["snippet"], str)
        if SNIPPET_MAX_CHARS != -1:
            assert len(row["snippet"]) <= SNIPPET_MAX_CHARS
        assert row["score"] is None or type(row["score"]) in (int, float)
        if ONLINE:
            assert row["score"] is None

    docid = rows[0]["docid"]
    first = await call("read_document", {"docid": docid, "chunk_index": 0})
    check_chunk(first, docid, 0)
    print("=== read_document ===")
    print(json.dumps(first, ensure_ascii=False, indent=2))
    if first["has_more"]:
        second = await call("read_document", {"docid": docid, "chunk_index": 1})
        check_chunk(second, docid, 1)
        assert first["char_end"] == second["char_start"]
        assert first["metadata"] == second["metadata"]
        print("PASS: cross-connection chunk boundary")
    else:
        print("SKIP: multi-chunk boundary (single-chunk document)")

    last_index = first["metadata"]["chunk_count"] - 1
    last = await call("read_document", {"docid": docid, "chunk_index": last_index})
    check_chunk(last, docid, last_index)
    assert last["has_more"] is False and last["next_chunk_index"] is None
    print("PASS: search/read fields, cross-connection reading, final chunk")


asyncio.run(main())
PY

python3 -m py_compile /tmp/retrieval_mcp_smoke.py
MCP_URL=http://127.0.0.1:8083/mcp EXPECT_ONLINE=1 REQUESTED_K=7 SNIPPET_MAX_CHARS=2048 \
  python3 /tmp/retrieval_mcp_smoke.py
```

每次读取都会新建 MCP 连接。如果文档只有一块，测试会明确跳过多块边界检查；需要换更长的文档/查询来覆盖该项。断言假设两次读取间缓存没有过期。

### 3. 在线缓存与失败验收

真实 TTL 测试请使用**临时测试服务/配置**，设置 `content_cache_idle_ttl_seconds: 10`，映射 TTL 保持更长，例如 `300`，然后重启。搜索一次保存 docid，读取第 0 块，立即重复读取，至少空闲等待 12 秒，再**不重新搜索，直接读取原 docid**。核对服务日志依次出现 `Reader fetch`、`Reader cache hit` 和新的 `Reader fetch`。重新抓取后不要断言正文一定完全一致。测试后恢复生产 TTL；只修改 YAML 不会改变已经运行的进程。

真实 single-flight 测试：对刚搜索得到且正文尚未缓存的 ID，使用三个独立连接并发读取。预期本轮文本相同，日志只新增一次 `Reader fetch`。顺序请求或已经预热的缓存不能证明并发抓取合并有效。

下面的确定性测试覆盖 lazy read、并发抓取合并、TTL、LRU 淘汰、映射过期/容量、失败后重试、超大正文和非法分块，不联网、不下载 HF 文件、不产生 API 费用。它直接测试 `OnlineService`，仅为缓存测试关闭字符截断；通过调整缓存时间戳模拟过期，不实际等待 TTL。

```bash
cat > /tmp/test_retrieval_online_cache.py <<'PY'
import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "searcher"))
from online_service import OnlineService


class FakeProvider:
    def __init__(self):
        self.calls = []
        self.fail_once = False
        self.closed = False
        self.body = "正文🙂" * 4000

    async def search(self, query, k):
        return [{"url": f"https://example.com/{query}", "snippet": "preview"}]

    async def read(self, url):
        self.calls.append(url)
        await asyncio.sleep(0)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("simulated upstream failure")
        return {"text": self.body, "url": url}

    async def close(self):
        self.closed = True


class CacheTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.provider = FakeProvider()
        self.service = OnlineService(
            self.provider, snippet_max_chars=-1,
            content_ttl=10, mapping_ttl=300,
        )

    async def asyncTearDown(self):
        await self.service.close()
        self.assertTrue(self.provider.closed)

    async def docid(self, query):
        return (await self.service.search(query))[0]["docid"]

    async def test_lazy_stable_id_single_flight_and_chunks(self):
        docid = await self.docid("a")
        self.assertEqual(await self.docid("a"), docid)
        self.assertEqual(self.provider.calls, [])
        results = await asyncio.gather(*[
            self.service.read_document(docid, 0) for _ in range(3)
        ])
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])
        self.assertEqual(len(self.provider.calls), 1)
        first = results[0]
        second = await self.service.read_document(docid, 1)
        self.assertEqual(first["char_end"], second["char_start"])
        self.assertEqual(first["text"] + second["text"], self.provider.body)
        self.assertEqual(second["metadata"]["char_count"], 12000)
        self.assertFalse(second["has_more"])
        self.assertIsNone(second["next_chunk_index"])
        self.assertEqual(len(self.provider.calls), 1)

    async def test_body_idle_expiry_keeps_mapping(self):
        docid = await self.docid("a")
        await self.service.read_document(docid)
        self.service.contents[docid]["last_access"] -= 11
        await self.service.read_document(docid)
        self.assertEqual(len(self.provider.calls), 2)
        self.assertIn(docid, self.service.documents)

    async def test_lru_body_eviction(self):
        self.provider.body = "abc"
        self.service.max_bytes = 6
        a, b, c = [await self.docid(q) for q in ("a", "b", "c")]
        await self.service.read_document(a)
        await self.service.read_document(b)
        await self.service.read_document(a)
        await self.service.read_document(c)
        self.assertIn(a, self.service.contents)
        self.assertNotIn(b, self.service.contents)
        self.assertIn(b, self.service.documents)
        await self.service.read_document(b)
        self.assertEqual(len(self.provider.calls), 4)

    async def test_mapping_expiry_drops_body(self):
        docid = await self.docid("a")
        await self.service.read_document(docid)
        self.service.documents[docid]["last_access"] -= 301
        with self.assertRaises(ValueError):
            await self.service.read_document(docid)
        self.assertNotIn(docid, self.service.contents)
        self.assertEqual(await self.docid("a"), docid)
        await self.service.read_document(docid)
        self.assertEqual(len(self.provider.calls), 2)

    async def test_mapping_capacity(self):
        self.service.mapping_limit = 1
        a = await self.docid("a")
        await self.service.read_document(a)
        await self.docid("b")
        self.assertNotIn(a, self.service.documents)
        self.assertNotIn(a, self.service.contents)

    async def test_failure_not_cached_later_retry_succeeds(self):
        docid = await self.docid("a")
        self.provider.fail_once = True
        with self.assertRaises(RuntimeError):
            await self.service.read_document(docid)
        self.assertNotIn(docid, self.service.contents)
        self.assertNotIn(docid, self.service.inflight)
        await self.service.read_document(docid)
        self.assertEqual(len(self.provider.calls), 2)

    async def test_bad_requests_do_not_break_service(self):
        docid = await self.docid("a")
        for target, index in (("web_missing", 0), (docid, -1), (docid, 2)):
            with self.assertRaises(ValueError):
                await self.service.read_document(target, index)
        result = await self.service.read_document(docid, 0)
        self.assertEqual(result["chunk_index"], 0)

    async def test_oversized_document_not_cached(self):
        self.service.max_bytes = 2
        docid = await self.docid("a")
        with self.assertRaises(RuntimeError):
            await self.service.read_document(docid)
        self.assertNotIn(docid, self.service.contents)


if __name__ == "__main__":
    unittest.main()
PY

python3 -m py_compile /tmp/test_retrieval_online_cache.py
python3 -m unittest discover -s /tmp -p test_retrieval_online_cache.py -v
```

### 验收记录与边界

开发部署已经反馈通过在线字段检查，智谱跨连接/分块/TTL/并发/错误检查，LRU/映射/失败的 mock 测试，以及 IQS 真实搜索后阅读。这些属于历史开发验收，不代表新安装或所有服务商引擎都自动通过；本次 k/字符截断改动，用户另已确认四个服务均暴露默认值为 5、要求大于 0 的整数 k，请求 k=2、k=7 分别返回 2 条、7 条，且所有 snippet 不超过 2048 字符。这确认了搜索核心改动，但不能据此证明不同 k 的 Dense 请求恰好进入同一 GPU batch，也不代表全部边界和性能已覆盖。本次文档同步由助手直接更新，未重新执行实机测试。每次部署应记录日期、依赖版本、脱敏配置、后端/服务商、测试结果和抓取次数，不要记录 API key。

根据部署需求，还应补充服务商限流/鉴权/网络失败、持续负载与内存开销、动态/受限网页、多服务组和客户端兼容性检查。`READY` 和冒烟测试不能代替检索质量评测。修改公共 Tool 后重新执行离线冒烟测试；修改适配器后重新执行对应服务商 mock。

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

| 产物 | 本地索引服务是否需要 | 重建是否需要 | 删除建议 |
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

删除 IVF-Flat 使用过的 embedding shard 不会影响该本地 ANN 索引的服务，但以后重建 ANN 时必须重新编码整个 corpus。如果删除 shard，还应同步更新 manifest 中的 `embeddings_built` 状态。

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
