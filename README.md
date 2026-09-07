# Local Corpus Retrieval Server

[English](README.md) | [简体中文](README.zh-CN.md)

This project is a self-hosted search engine for fixed local corpora. It packages sparse retrieval, dense retrieval, document reading, corpus ingestion, index construction, and MCP serving into one deployable system. Its main purpose is to let agents and applications search private, curated, or reproducible datasets without depending on a live web search engine.

The server was developed from the retrieval code in [BrowseComp-Plus](https://texttron.github.io/BrowseComp-Plus/), but it is a separate engineering project rather than the BrowseComp-Plus benchmark itself. BrowseComp-Plus datasets, indexes, and evaluation tools remain supported as an optional integration and reproducibility workload.

Upstream BrowseComp-Plus resources: [dataset](https://huggingface.co/datasets/Tevatron/browsecomp-plus) · [leaderboard](https://huggingface.co/spaces/Tevatron/BrowseComp-Plus) · [paper](https://arxiv.org/pdf/2508.06600) · [project page](https://texttron.github.io/BrowseComp-Plus/)

The local retrieval server provides:

- BM25 retrieval through Pyserini/Lucene;
- dense exact retrieval through FAISS Flat indexes;
- large-scale approximate retrieval through FAISS IVF-Flat;
- independent, manifest-backed local corpora;
- a SQLite document store for bounded full-document reading;
- MCP servers exposing the same tools for BM25 and dense backends;
- resumable corpus normalization and index construction;
- a YAML pipeline for Wikimedia Enterprise HTML archives.

## Contents

- [Retrieval stack at a glance](#retrieval-stack-at-a-glance)
- [Quick start](#quick-start)
- [Environment setup](#environment-setup)
- [Dataset and pre-built indexes](#dataset-and-pre-built-indexes)
- [Running retrieval servers](#running-retrieval-servers)
- [Building a local corpus](#building-a-local-corpus)
- [Wikimedia Enterprise YAML pipeline](#wikimedia-enterprise-yaml-pipeline)
- [Build artifacts and disk usage](#build-artifacts-and-disk-usage)
- [Evaluation](#evaluation)
- [Documentation](#documentation)
- [Upstream acknowledgement and citation](#upstream-acknowledgement-and-citation)

## Retrieval stack at a glance

| Component | Implementation | Typical use |
| --- | --- | --- |
| Sparse retrieval | Pyserini/Lucene BM25 | Keyword retrieval and CPU serving |
| Dense exact retrieval | FAISS Flat over Tevatron pickle shards | Small and medium corpora, exact vector search |
| Dense ANN retrieval | FAISS IVF-Flat | Large corpora where exact search is too expensive |
| Document store | Read-only SQLite | Full text, URL, title, language, and character-count lookup |
| Corpus manifest | `corpora/<name>/manifest.json` | Keeps build-time and query-time settings consistent |
| MCP interface | FastMCP | Uniform `search` and `read_document` tools |

Each local corpus is independent. Its normalized documents, document store, sparse index, dense index, and manifest are never merged with another corpus.

## Quick start

This smoke test starts the server with the upstream BrowseComp-Plus BM25 and Qwen3-Embedding-0.6B indexes. It verifies an installation without requiring a new corpus build. To index your own data instead, go directly to [Building a local corpus](#building-a-local-corpus). Complete the [environment setup](#environment-setup) first.

### 1. Activate the environment

```bash
REPO_DIR=/path/to/local-repository
cd "$REPO_DIR"
source .venv/bin/activate

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
```

### 2. Download the pre-built indexes

```bash
bash scripts_build_index/download_indexes.sh
```

This downloads the upstream BrowseComp-Plus BM25 and Qwen3-Embedding indexes under `indexes/`.

### 3. Materialize the local document store

The pre-built dense index contains vectors and document IDs, but document reading needs a local corpus and SQLite store:

```bash
python scripts_preprocess/materialize_browsecomp_corpus.py \
  --cache-dir "$HF_HOME" \
  --output-dir corpora/browsecomp
```

Use `--overwrite` only when intentionally replacing an existing `corpora/browsecomp` directory.

### 4. Start BM25

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

### 5. Start dense retrieval

Run this in another shell:

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

The local MCP endpoints are:

```text
BM25:  http://127.0.0.1:8080/mcp
Dense: http://127.0.0.1:8081/mcp
```

Replace `127.0.0.1` with the server hostname only when the client runs on another machine and network access has been configured safely.

Both servers expose the same tools:

```text
search(query)
read_document(docid, chunk_index=0)
```

`--get-document` is a legacy CLI flag name. The tool registered for clients is `read_document`, which returns one bounded 8,192-character chunk at a time.

## Environment setup

The environment is more specialized than a typical Python installation because PyTorch, CUDA, FlashAttention, FAISS, and the GPU architecture must be compatible.

### Validated Blackwell environment

The following stack has been validated:

| Component | Validated version |
| --- | --- |
| OS/container | Ubuntu 22.04 |
| Python | 3.10.12 |
| CUDA Toolkit | 12.8 |
| GPU | NVIDIA RTX 6000D, Blackwell `sm_120` |
| PyTorch | 2.7.0+cu128 |
| TorchVision | 0.22.0+cu128 |
| TorchAudio | 2.7.0+cu128 |
| FlashAttention | 2.8.3.post1 |
| vLLM | 0.9.0.1 |
| FAISS | 1.11.0 |
| Transformers | 4.53.2 |
| Datasets | 4.0.0 |
| FastMCP | 2.9.2 |

Other hardware can work, but select a PyTorch/CUDA/FlashAttention combination supported by that machine.

### 1. Enter the runtime environment

When using an existing Docker container:

```bash
CONTAINER_NAME=my-container

docker start "$CONTAINER_NAME"
docker exec -it "$CONTAINER_NAME" bash
REPO_DIR=/path/to/local-repository
cd "$REPO_DIR"
```

Docker is optional. On a compatible Linux host, enter the cloned repository directly.

### 2. Install `uv` and Java

The project uses `uv` and Python 3.10. Pyserini/Lucene also requires Java 21.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install Java with the package manager available on the host. For Ubuntu:

```bash
sudo apt update
sudo apt install -y openjdk-21-jdk
```

Confirm the installation:

```bash
java -version
```

### 3. Create the virtual environment

```bash
uv venv --python /usr/bin/python3 .venv
source .venv/bin/activate
python -V
```

The validated environment reports `Python 3.10.12`.

### 4. Install ordinary dependencies without the default GPU stack

The current `uv.lock` resolves the default PyPI PyTorch/CUDA 12.6 dependency stack, while the validated Blackwell environment uses PyTorch CUDA 12.8. For that environment, install ordinary dependencies first and exclude the GPU stack:

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

### 5. Install PyTorch CUDA 12.8

```bash
uv pip install \
  --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.7.0+cu128' \
  'torchvision==0.22.0+cu128' \
  'torchaudio==2.7.0+cu128'
```

### 6. Install FlashAttention

Use a wheel built for the selected Python, PyTorch, CUDA, operating system, and GPU architecture. For the validated environment:

```bash
FLASH_ATTN_WHEEL=/path/to/compatible-flash-attention.whl

uv pip install \
  --python .venv/bin/python \
  --no-deps \
  "$FLASH_ATTN_WHEEL"
```

The validated wheel provides `flash-attn==2.8.3.post1` for CPython 3.10, PyTorch 2.7.0+cu128, and Blackwell `sm_120`.

Do not copy `.venv` from another machine. Python virtual environments contain machine-specific binaries and absolute interpreter paths and should be recreated from the repository metadata.

### 7. Configure Hugging Face caches

Use a persistent location outside `.venv`:

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
```

Copy `search_env.example.sh` to the ignored `search_env.sh` and customize it for the deployment. Do not commit credentials or machine-specific secrets.

If a proxy is required, configure it for the current shell using deployment-specific values:

```bash
export HTTPS_PROXY="http://<proxy-host>:<proxy-port>"
export HTTP_PROXY="$HTTPS_PROXY"
export NO_PROXY="localhost,127.0.0.1"
```

### 8. Validate the environment

Check imports and package consistency:

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

Validate CUDA and FlashAttention on a GPU:

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

Until the lockfile is updated to model the CUDA 12.8 override directly, do not use a bare `uv sync` or `uv run` to mutate the validated Blackwell environment. Repeat the two-stage installation above instead.

Recovery from a deleted environment or an offline `uv` cache is a troubleshooting procedure, not part of the normal installation flow.

## Dataset and pre-built indexes

### Optional: download upstream benchmark data

Download and decrypt the benchmark queries, answers, and relevance judgments:

```bash
python scripts_build_index/decrypt_dataset.py \
  --output data/browsecomp_plus_decrypted.jsonl \
  --generate-tsv topics-qrels/queries.tsv
```

Authentication with Hugging Face may be required. Prefer the `HF_TOKEN` environment variable or `huggingface-cli login` instead of placing a token in shell history.

The public corpus can be loaded independently:

```python
from datasets import load_dataset

corpus = load_dataset(
    "Tevatron/browsecomp-plus-corpus",
    split="train",
)
```

### Download indexes

```bash
bash scripts_build_index/download_indexes.sh
```

The script downloads the available BM25 and Qwen3-Embedding indexes from `Tevatron/browsecomp-plus-indexes` into `indexes/`.

## Running retrieval servers

There are two configuration modes.

### Manifest mode: recommended for locally built corpora

Every corpus built with `build_corpus.py` has a manifest under `corpora/<name>/manifest.json`. The manifest records the index path, model, pooling mode, normalization, query prefix, document store, index type, and ANN settings.

Start BM25:

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

Start dense retrieval:

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

Do not repeat model or index settings on the command line unless intentionally overriding a manifest-supported server option. Query-time embedding settings must match the document index.

### Explicit-path mode: legacy indexes and advanced use

BM25 template:

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

Dense Flat template:

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

IVF-Flat template:

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

### Recommended serving parameters

These values performed well in the validated deployment. They are starting points, not hardware-independent guarantees.

| Area | Parameter | Recommended value | Code default |
| --- | --- | ---: | ---: |
| BM25 | `--bm25-workers` | 16 | 1 |
| BM25 | Client request concurrency | 8 | N/A |
| Dense | `--dense-max-batch-size` | 16 | 16 |
| Dense | `--dense-max-wait-ms` | 1 ms | 5 ms |
| Dense | `--dense-max-queue-size` | 64 | 64 |
| Dense | Client request concurrency | 32 | N/A |
| MCP | `--transport` | `sse` | `sse` |
| Search | `--k` | 5 | 5 |
| Search | `--snippet-max-tokens` | 512 | 512 |
| Read | Chunk size | 8,192 characters | 8,192 characters |

Observed regression reference points on the validated hardware and BrowseComp corpus were approximately:

```text
BM25:  ~43 QPS, p50 ~176 ms, p95 ~251 ms
Dense: ~35–36 QPS, p50 ~862 ms, p95 ~1.1 s
```

Results depend on the corpus, CPU, GPU, index type, client sessions, and network path. Investigate a regression when comparable tests fall substantially below approximately 38 QPS for BM25 or 30 QPS for dense retrieval.

### Transport and MCP endpoint

Supported transports are:

```text
stdio
sse
streamable-http
```

The HTTP path configured by this server is `/mcp`. For example:

```text
http://127.0.0.1:<port>/mcp
```

SSE is the tested recommendation for this dependency set. Streamable HTTP remains available for compatible clients.

## Building a local corpus

Use `scripts_build_index/build_corpus.py` for ordinary JSONL files, DPR Wiki18 TSV files, and already-normalized corpora. The Wikimedia Enterprise HTML tar format has a dedicated [YAML pipeline](#wikimedia-enterprise-yaml-pipeline).

### Output layout

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

The final two files are present only for IVF-Flat indexes.

### Generic JSONL

The input must contain one JSON object per line:

```json
{"docid": "doc-001", "text": "Document body"}
{"docid": "doc-002", "text": "Another document"}
```

Fields can be mapped. For example:

```json
{"id": "doc-001", "title": "Document title", "content": "Document body"}
```

Requirements and behavior:

- document IDs are required and must be globally unique;
- the configured text field is required;
- blank normalized text is skipped;
- every non-empty line must be valid JSON;
- a configured title is prepended as `title + "\n" + text`;
- the generic JSONL path retains the ID, normalized text, and optional title only;
- use a custom materializer when URL or additional metadata must be preserved.

Build BM25 and Dense Flat:

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

For `id`, `content`, and `title` fields:

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

Only BM25:

```bash
python scripts_build_index/build_corpus.py \
  --name my-corpus \
  --input data/my-corpus/corpus.jsonl \
  --format jsonl \
  --id-field docid \
  --text-field text \
  --build bm25
```

Normalize and create the SQLite store without building an index:

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

The input must contain a header and these tab-separated columns:

```text
id<TAB>text<TAB>title
```

Build a 10,000-document canary first:

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

After validating the canary, build the full corpus under a different name:

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

### Build IVF-Flat ANN

Flat search is exact and reads the embedding pickle shards directly. IVF-Flat builds a separate approximate index and is more appropriate for very large corpora.

Build normalization, BM25, dense embeddings, and IVF-Flat in one invocation:

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

Dense encoding GPUs (`--gpus`) and ANN-construction GPUs (`--ivf-gpus`) are configured independently.

For smaller corpora, start with `--dense-index-type flat`. For large corpora, start with `nprobe=32` and measure recall and latency before tuning it.

### Resume an interrupted build

```bash
python scripts_build_index/build_corpus.py \
  --name my-corpus \
  --build bm25,dense \
  --resume \
  --gpus 0 \
  --dense-shards 1
```

`--resume` can:

- reuse an existing normalized corpus and document store;
- skip BM25 when its manifest state is complete;
- reuse complete dense shards;
- generate missing shards;
- inherit model, pooling, dtype, normalization, and related settings from the manifest unless explicitly overridden.

`--overwrite` deletes the existing corpus directory and index tree before rebuilding. It cannot be combined with `--resume` and should be used only when replacement is intentional.

### Already-cleaned custom corpora

If an external materializer has already created:

```text
corpora/<name>/corpus.jsonl
corpora/<name>/documents.sqlite
```

build indexes without passing the original input again:

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

This path is appropriate when a custom materializer must preserve URL, title, source, language, character count, or other source-specific metadata.

### Pooling, normalization, and model compatibility

The query encoder must use the same model, pooling mode, normalization behavior, dtype, and query prefix used for document encoding. The upstream pre-built Qwen3-Embedding indexes and the validated Wiki18 manifest use `pooling=eos` and normalized embeddings.

Do not treat a setting seen in another model's example as a universal default. Use the corpus manifest as the source of truth whenever possible.

## Wikimedia Enterprise YAML pipeline

The YAML pipeline is specific to Wikimedia Enterprise HTML `.json.tar.gz` archives. Ordinary JSONL and DPR TSV files should use `build_corpus.py` directly.

The pipeline performs:

```text
discover or download archives
→ freeze an input manifest
→ clean and deduplicate records
→ write corpus.jsonl and documents.sqlite
→ build BM25
→ encode dense embeddings
→ optionally build IVF-Flat
→ write manifest.json
```

Run it from the repository root:

```bash
CONFIG=configs/corpora/wikimedia-enterprise.yaml

python scripts_build_index/run_corpus_pipeline.py \
  --config "$CONFIG"
```

### Complete example

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

Keep GPU lists quoted so the YAML parser passes a comma-separated string to the build command.

### Local archive source

For archives already present on disk, replace the source block:

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

Relative paths are resolved from the repository root.

### Archive selection

Selection is applied in this order:

```text
archive_pattern → include_files → exclude_files
```

The fields use case-sensitive glob patterns, not regular expressions. `exclude_files` takes precedence over `include_files`.

Examples:

```yaml
# Include namespace 0 only.
include_files:
  - "*-NS0-*"
exclude_files: []
```

```yaml
# Include all matched archives except templates and categories.
include_files: []
exclude_files:
  - "*-NS10-*"
  - "*-NS14-*"
```

### Important YAML fields

| Field | Meaning | Default or recommendation |
| --- | --- | --- |
| `corpus.name` | Output corpus name | Required |
| `source.type` | `web` or `local` | Required |
| `source.archive_pattern` | First-stage archive filter | `*.tar.gz`; use a stricter pattern in production |
| `source.include_files` | Optional allowlist globs | `[]` |
| `source.exclude_files` | Optional denylist globs | `[]` |
| `cleaning.min_chars` | Minimum text length | `1` |
| `cleaning.batch_size` | SQLite write batch size | `1000` |
| `cleaning.on_error` | `fail` or `skip` | Use `fail` for initial validation |
| `build.targets` | `bm25`, `dense`, and/or `ann` | `bm25,dense` |
| `build.bm25.threads` | Lucene indexing threads | `16` |
| `build.dense.shards` | Number of embedding shards | Usually the GPU count |
| `build.dense.index_type` | `flat` or `ivf-flat` | `flat` for smaller corpora |
| `build.ann.nprobe` | IVF lists searched per query | Start with `32` |
| `runtime.dry_run` | Print commands without executing | `false` |
| `runtime.keep_build_files` | Retain temporary Pyserini input | `false` |

### Current pipeline limitations

- `source.keep_downloads` is reserved but does not currently delete downloaded archives when set to `false`.
- `runtime.overwrite` is not propagated through all three stages and should remain `false`.
- the pipeline invokes the index builder with `--resume`; resumable operation is the intended mode.
- the web loader parses one directory listing and does not recursively crawl linked web subdirectories.

## Build artifacts and disk usage

| Artifact | Needed for serving | Needed for rebuilding | Deletion guidance |
| --- | --- | --- | --- |
| `corpora/<name>/documents.sqlite` | Yes, for document reading and metadata | Yes | Keep |
| `corpora/<name>/manifest.json` | Yes in `--corpus` mode | Yes | Keep |
| `indexes/corpora/<name>/bm25/` | Yes for BM25 | Final artifact | Keep |
| `index.ivf-flat.faiss` | Yes for IVF-Flat | Final artifact | Keep |
| `index.lookup.pkl` | Yes for IVF-Flat | Final artifact | Keep |
| `corpus.shard*.pkl` with Flat | Yes | Final Flat index | Keep |
| `corpus.shard*.pkl` with IVF-Flat | No | Avoids re-encoding when rebuilding ANN | Archive or delete only after validation |
| `.build/pyserini/docs.jsonl` | No | Regenerated from `corpus.jsonl` | Safe to delete after BM25 succeeds |
| Downloaded source archives | No | Allows cleaning to be repeated | Archive or delete according to recovery needs |

`--keep-build-files` retains the temporary Pyserini input. Without that flag, successful BM25 construction removes the corpus `.build` directory automatically.

Deleting IVF-Flat embedding shards does not affect online retrieval, but a future ANN rebuild will require re-encoding the corpus. If shards are removed, the manifest's `embeddings_built` state should be updated accordingly.

Generated corpora, indexes, models, caches, virtual environments, and runtime logs should not be committed to Git. Commit source code, configuration templates, `pyproject.toml`, `uv.lock`, and documentation instead.

## Evaluation

### Optional: reproduce upstream BrowseComp-Plus results

Guides for agents using OpenAI, Anthropic, Gemini, Qwen, OSS, GLM, Search-R1, and Tongyi are available under `docs/`.

### Evaluate Deep-Research runs

Place one JSON file per query under a run directory such as `runs/my-model/`. Each file must contain at least:

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

Run the evaluator:

```bash
python scripts_evaluation/evaluate_run.py \
  --input_dir runs/my-model
```

The evaluator uses Qwen3-32B as a judge. See `docs/llm_as_judge.md` for details and use `--tensor_parallel_size` to configure its GPU parallelism.

### Evaluate retrieval-only results

Format results as a standard TREC run file, then evaluate against evidence and gold judgments:

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

Evidence documents are labeled as needed to answer the query. Gold documents are needed to answer the query and semantically contain the final answer.

### Trajectory data

Download released baseline trajectories with:

```bash
bash scripts_build_index/download_run_files.sh
```

The decrypted trajectories are written under `data/decrypted_run_files/`.

## Documentation

- [OpenAI API](docs/openai.md)
- [Gemini API](docs/gemini.md)
- [Anthropic API](docs/anthropic.md)
- [GLM API](docs/glm.md)
- [OSS](docs/oss.md)
- [Qwen](docs/qwen.md)
- [Search-R1](docs/search-r1.md)
- [Tongyi DeepResearch](docs/tongyi.md)
- [Custom retrievers](docs/custom_retriever.md)
- [Local independent corpora](docs/local_corpora.md)
- [LLM-as-a-judge evaluation](docs/llm_as_judge.md)
- [NetMind MCP](docs/netmind_mcp.md)

## Upstream acknowledgement and citation

This server is based on and extends the retrieval and evaluation components released with BrowseComp-Plus. The local corpus pipeline, manifest-based configuration, SQLite document store, concurrent MCP serving, resumable builds, and IVF-Flat workflow documented here belong to this server project and should not be confused with the scope of the upstream benchmark.

When using the BrowseComp-Plus dataset, pre-built indexes, evaluation data, or other upstream research artifacts, cite the original paper:

```bibtex
@article{chen2025BrowseCompPlus,
  title={BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent},
  author={Chen, Zijian and Ma, Xueguang and Zhuang, Shengyao and Nie, Ping and Zou, Kai and Liu, Andrew and Green, Joshua and Patel, Kshama and Meng, Ruoxi and Su, Mingyi and Sharifymoghaddam, Sahel and Li, Yanxi and Hong, Haoran and Shi, Xinyu and Liu, Xuye and Thakur, Nandan and Zhang, Crystina and Gao, Luyu and Chen, Wenhu and Lin, Jimmy},
  year={2025},
  journal={arXiv preprint arXiv:2508.06600}
}
```

For questions about this server, use the issue tracker of this repository. For questions specifically about the upstream benchmark, use the contact information published by the BrowseComp-Plus project.

This project is distributed under the terms in [LICENSE](LICENSE).
