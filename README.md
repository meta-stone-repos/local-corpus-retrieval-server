# Local Corpus Retrieval Server

[English](README.md) | [简体中文](README.zh-CN.md)

This project is a self-hosted retrieval server for both fixed local corpora and live web APIs. It combines BM25, Dense Flat/IVF-Flat, document reading, corpus ingestion, index construction, and MCP serving. Local retrieval supports private, curated, and reproducible datasets without a web search dependency; optional Zhipu and Alibaba Cloud IQS adapters provide live search and lazy webpage reading through the same Tool contract.

The server was developed from the retrieval code in [BrowseComp-Plus](https://texttron.github.io/BrowseComp-Plus/), but it is a separate engineering project rather than the BrowseComp-Plus benchmark itself. BrowseComp-Plus datasets, indexes, and evaluation tools remain supported as an optional integration and reproducibility workload.

Upstream BrowseComp-Plus resources: [dataset](https://huggingface.co/datasets/Tevatron/browsecomp-plus) · [leaderboard](https://huggingface.co/spaces/Tevatron/BrowseComp-Plus) · [paper](https://arxiv.org/pdf/2508.06600) · [project page](https://texttron.github.io/BrowseComp-Plus/)

The local retrieval server provides:

- BM25 retrieval through Pyserini/Lucene;
- dense exact retrieval through FAISS Flat indexes;
- large-scale approximate retrieval through FAISS IVF-Flat;
- independent, manifest-backed local corpora;
- a SQLite document store for bounded full-document reading;
- uniform `search` / `read_document` tools for offline and online retrieval;
- Zhipu and IQS adapters with stable URL IDs, lazy reading, and bounded transient caches;
- unified YAML service selection, readiness reporting, logs, and group stop/status;
- resumable corpus normalization and index construction;
- a YAML pipeline for Wikimedia Enterprise HTML archives.

## Contents

- [Retrieval stack at a glance](#retrieval-stack-at-a-glance)
- [Quick start](#quick-start)
- [Environment setup](#environment-setup)
- [Dataset and pre-built indexes](#dataset-and-pre-built-indexes)
- [Running retrieval servers](#running-retrieval-servers)
- [Tool output contract](#tool-output-contract)
- [Online APIs and cache](#online-apis-and-cache)
- [Adding a Provider](#adding-a-provider)
- [Acceptance tests](#acceptance-tests)
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
| Online retrieval | Zhipu / Alibaba Cloud IQS | Live search and same-vendor webpage reading |
| Online cache | Process-local mapping and body LRU/idle TTL | Cross-connection reads without repeated fetching |
| Service launcher | `searcher/serve.py` + YAML | Select services, check readiness, inspect/stop a group |
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

### 4. Configure and start MCP services

Create a local YAML using the [unified example](#unified-yaml-example). For an existing deployment, keep your current YAML and credentials. The example uses BrowseComp-Plus indexes, Dense GPU 6, and a pre-populated `/workspace/hf_cache`; adjust these to your machine.

```bash
python3 searcher/serve.py --config configs/services.local.yaml --services bm25 dense --check
python3 searcher/serve.py --config configs/services.local.yaml --services bm25 dense
```

Wait for both `READY` messages. BM25 uses `http://127.0.0.1:8080/mcp`; Dense uses `http://127.0.0.1:8082/mcp`. Do not start the same configuration a second time in another terminal. Use `--status` or `--stop` there instead.

To use online search without building local indexes, configure Zhipu/IQS credentials and start `--services online`, `--services iqs`, or `--services online iqs`. Online services do not require a local corpus or a snippet tokenizer; snippets are sliced by Unicode characters. See [online settings](#online-apis-and-cache) and [acceptance tests](#acceptance-tests).

Both offline and online services expose `search(query, k=5)` and `read_document(docid, chunk_index=0)` with the [same successful-result schema](#tool-output-contract). Offline services need `get_document: true` to enable reading. Each read returns at most 8192 Unicode characters.

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

Use `searcher/serve.py` and one YAML file to launch BM25, Dense Flat/IVF-Flat, Zhipu, and IQS. Configure first, then select services by name; changes take effect after restart, not through hot reload. Commands below run from the repository root inside the runtime environment (inside Docker when using Docker).

### Unified YAML example

For a fresh checkout, create the local configuration from the credential-free template:

```bash
cp -n configs/services.example.yaml configs/services.local.yaml
chmod 600 configs/services.local.yaml
```

`cp -n` preserves an existing local configuration. Configure credentials locally and adjust paths/GPU IDs. Local configuration files and their backups are ignored; commit only the sanitized example.


Save the following as `configs/services.local.yaml` for a **new deployment**. Do not overwrite an existing local configuration: it may contain credentials and tuned paths. This is a deployment example, not a claim that this file is shipped with every checkout. Adjust the cache/index paths and GPU ID to your machine. The Dense example uses GPU **6**, BrowseComp-Plus Flat shards, and the local SQLite document store.

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

The online entries use loopback binding for local tests. The current offline HTTP entry point binds `0.0.0.0` and does not accept `args.host`; restrict Docker port publishing/firewall access as appropriate. These examples do not add authentication or TLS. Do not expose the services directly to untrusted networks.

### Start, inspect, and stop

```bash
# Validate configuration structure and local paths; no model loading/API calls.
python3 searcher/serve.py --config configs/services.local.yaml --all --check

# Choose ONE launch command for this configuration.
python3 searcher/serve.py --config configs/services.local.yaml --services bm25 dense
# Or: --services online iqs
# Or: --services bm25 iqs
# Or: --all
```

`--all` includes every service in the file, so both online API credentials must be available if both online entries are present. `--services` takes space-separated names, not a comma-separated string. Example endpoints:

To launch all four configured services instead of a subset:

```bash
python3 searcher/serve.py --config configs/services.local.yaml --all
```

| Service name | Backend/provider | Endpoint |
| --- | --- | --- |
| `bm25` | BM25 | `http://127.0.0.1:8080/mcp` |
| `dense` | FAISS | `http://127.0.0.1:8082/mcp` |
| `online` | Zhipu | `http://127.0.0.1:8081/mcp` |
| `iqs` | IQS | `http://127.0.0.1:8083/mcp` |

The launcher stays in the foreground and prints `CHECK`, `STARTING`, `WAITING`, and `READY`. `READY` means an MCP connection succeeded and the expected tools were listed; it does **not** validate API credentials, upstream search/read behavior, retrieval quality, or throughput. `--startup-timeout 900` is the default startup deadline in seconds. A child exit or startup failure stops the other children in that launch group; there is no automatic restart policy.

In another terminal, using the **same runtime/container and configuration path**:

```bash
python3 searcher/serve.py --config configs/services.local.yaml --status
python3 searcher/serve.py --config configs/services.local.yaml --stop
```

`--status` reports tracked process/state information, not a new end-to-end API health test. `--stop` stops the whole tracked group, not one selected service; do not combine it with `--services`, `--all`, or `--check`. `Ctrl+C` also stops the foreground group. Groups are identified by the resolved configuration path, with one launcher per configuration. To change the selected services, stop that group and start the new selection. To manage groups independently, use separate configuration files with non-conflicting ports. A service previously started from `services.iqs.local.yaml` must be stopped using that old path before launching IQS from the unified file; editing/merging YAML does not transfer process ownership.

Logs are written to `logs/services/<run-id>/<service>.log`; the launcher prints exact paths. Process state/locks are under `logs/services/.state/`. Group management relies on Linux `/proc`, process groups, file locks, and PID identity checks. The launcher is not a daemon/service manager and does not promise survival after terminal closure or automatically adopt orphan processes.

### YAML fields and inheritance

| Field | Meaning |
| --- | --- |
| `version` | Required; currently `1` |
| `defaults.runtime`, `defaults.args` | Shared settings; keep only fields accepted by all inheriting services |
| `services.<name>.backend` | `bm25`, `faiss`, or `online`; Dense ANN still uses `faiss` |
| `services.<name>.runtime` | Environment configuration for that child process |
| `services.<name>.args` | Worker CLI arguments, using underscore names: `snippet_max_chars` → `--snippet-max-chars` |
| `services.<name>.provider` | Online only; `name` selects a registered adapter; other fields configure that adapter |

Service mappings recursively override defaults. Lists/scalars replace inherited values; absent fields continue to inherit. Duplicate YAML keys and unknown top-level/service/runtime fields are rejected. `args` accepts scalar values only: `true` emits a flag, `false`/`null` omits it, other values emit `--name=value`. Omitting a flag does not negate a backend's own default or a manifest setting. Nested provider options belong under `provider`, not `args`.

`--config` itself is resolved from the current working directory. Workers run from the repository root; relative index, corpus, and cache paths in this example are rooted there, not at the YAML directory. YAML does **not** expand `$VARIABLE` or `${VARIABLE}` strings. Use `api_key_env` or actual paths instead.

| `runtime` field | Behavior |
| --- | --- |
| `hf_home` | Sets `HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, and `HF_DATASETS_CACHE`; removes inherited `TRANSFORMERS_CACHE` |
| `hf_offline` | Sets `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` to `1`/`0` |
| `cuda_visible_devices` | Sets `CUDA_VISIBLE_DEVICES`; quote GPU lists, e.g. `"6"` or `"0,1"` |
| `env` | Additional environment variables; values are strings or integers; named runtime settings take precedence |

The child inherits the launcher's environment. HF offline mode affects model/tokenizer downloads, **not** the HTTP search/reader API. Snippet generation in BM25, Dense, and online MCP now uses character slicing and loads no snippet tokenizer. Dense still requires its embedding model and query-encoding tokenizer; prepare their caches beforehand or use a compatible local `model_name` directory. HF offline settings can remain in shared configuration for Dense. Do not enable HF offline mode while first downloading models/indexes.

| Common `args` field | Default/meaning |
| --- | --- |
| `port` | Required by the grouped launcher; must be unique within selected services |
| `transport` | `sse` by default; grouped serving also supports `streamable-http`, not `stdio` |
| `k` | Default `5`; fallback when an Agent omits Tool argument `k`, not a fixed per-service result cap |
| `snippet_max_chars` | Default `2048` Unicode characters per result; `-1` disables truncation; positive integers otherwise |
| `get_document` | Offline only; set `true` to register `read_document`; online always registers it |
| `host` | Online only; worker default `0.0.0.0`, example uses `127.0.0.1` |

`--check` does not instantiate Providers, validate every backend CLI option, load indexes/models, check port availability, or contact upstream APIs. Unknown worker arguments and deeper configuration errors can therefore still fail at startup. Do not set `args.config`, `args.searcher_type`, or `args.hf_token`: use `provider`, `backend`, and the `HF_TOKEN` environment variable respectively.

### Offline: manifests, Flat, and ANN

The YAML launcher preserves both existing index configuration modes:

- **Manifest mode:** set `args.corpus: my-corpus`; `corpora/my-corpus/manifest.json` supplies index/model settings. The launcher rejects combining `corpus` with `index_path`, `model_name`, `pooling`, `normalize`, `torch_dtype`, `lookup_path`, `index_type`, `nprobe`, `task_prefix`, or `max_length`. The manifest remains authoritative. BM25 can use an explicit `document_store` for reading. Snippet generation no longer derives or loads a tokenizer from the manifest; Dense query encoding is unchanged.
- **Explicit mode:** use the paths and model settings shown in the full example. Keep the query model, pooling, normalization, dtype, and query prefix compatible with the index.

| Offline `args` | Use |
| --- | --- |
| `index_path`, `document_store` | BM25 Lucene directory or FAISS input; SQLite store for document access |
| `bm25_workers` | BM25 thread workers; default `1`, deployment example `16` |
| `index_type` | FAISS `flat` (default) or `ivf-flat` |
| `lookup_path`, `nprobe` | IVF-Flat docid mapping and lists probed; `nprobe` default `32` |
| `model_name`, `pooling`, `normalize`, `torch_dtype` | Query embedding settings; match the built index |
| `task_prefix`, `max_length` | Query instruction and token length; FAISS length default `8192` |
| `corpus_path`, `dataset_name` | Alternative FAISS document sources; prefer local `document_store` to avoid dataset loading |
| `dense_max_batch_size` | Default `16` requests per batch |
| `dense_max_wait_ms` | Default `5` ms; deployment example `1` ms |
| `dense_max_queue_size` | Default `64` pending requests |

ANN is a Dense index choice, not another online Provider. For an existing IVF-Flat index, replace the Flat index fields in the Dense service with:

```yaml
index_type: ivf-flat
index_path: indexes/corpora/my-corpus/qwen3-embedding-0.6b/index.ivf-flat.faiss
lookup_path: indexes/corpora/my-corpus/qwen3-embedding-0.6b/index.lookup.pkl
nprobe: 32
```

Also use that corpus's document store and matching model parameters. `nlist` and training parameters belong to index construction, not serving. Do not pass IVF-only paths to Flat search.

Direct worker invocation remains available for debugging and offline `stdio` clients:

```bash
python3 searcher/mcp_server.py --help --searcher-type bm25
python3 searcher/mcp_server.py --help --searcher-type faiss
python3 searcher/online_mcp_server.py --help
```

The direct online worker's `--config` accepts JSON shaped as `{"provider":"iqs","providers":{"iqs":{...}}}`, **not** the service YAML. The launcher translates YAML to a temporary permission-`600` JSON file automatically. Prefer the launcher for normal operation.

Historical offline performance reference points on the validated BrowseComp deployment were BM25 ~43 QPS (p50 ~176 ms, p95 ~251 ms) and Dense ~35–36 QPS (p50 ~862 ms, p95 ~1.1 s). These are not fresh measurements or online API throughput guarantees; compare only equivalent corpus/hardware/concurrency conditions.

## Tool output contract

### `search(query: str, k: int = 5)`

Returns a JSON array with up to `k` entries:

```json
[
  {"docid": "web_<sha256-of-url>", "score": null, "snippet": "A bounded preview."}
]
```

| Field | Type/meaning |
| --- | --- |
| `docid` | String; corpus ID offline, generated URL ID online |
| `score` | Number or `null`; online currently always returns `null`, even if an upstream API supplies a score |
| `snippet` | String preview, sliced by Unicode character count; no snippet tokenizer |

Search does **not** expose `metadata`, URL/title/source/language, `char_count`, or chunk counts. Each result uses `text[:snippet_max_chars]` (offline document text) or `snippet[:snippet_max_chars]` (online API snippet). Length is measured in Python Unicode code points, not UTF-8 bytes, tokens, words, or natural paragraphs. The old 8192-character pre-guard and token encode/decode stages are removed. Short text stays unchanged; `-1` returns the full source string. This does not change the separate 8192-character `read_document` chunk size.

`k` is visible to the Agent as an integer greater than zero. The schema default is taken from YAML `args.k` at startup (5 in the example). A call such as `{"query":"history","k":7}` asks for up to seven results; omitting `k` uses that default. BM25 passes k to Lucene, and Dense passes it to FAISS. Mixed-k Dense requests share a batch using its largest k, then each result list is sliced to its own requested size. Online passes k to the Provider and limits the public list to at most k. Fewer hits may be available; IQS rejects values outside 1–50, and other vendor restrictions still apply.

Migration: replace `snippet_max_tokens: 512` with `snippet_max_chars: 2048` and remove `snippet_tokenizer` from both defaults and per-service overrides. Replace old CLI flags with `--snippet-max-chars`; the old MCP snippet flags are no longer accepted. 2048 characters is only a rough replacement for 512 tokens, not a language-independent equivalence. Restart after configuration changes; clients may need to reconnect/refresh their Tool schemas. The separate `search_r1_server.py` endpoint is outside this MCP change.

### `read_document(docid: str, chunk_index: int = 0)`

The following is a complete, internally consistent final-chunk example:

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

- Indices are zero-based. `char_start` is inclusive and `char_end` exclusive; `len(text) == char_end - char_start` using Python Unicode code points, not UTF-8 bytes, tokens, or paragraph boundaries.
- The MCP chunk size is fixed at `8192`; it is not a YAML tuning field. `chunk_count = max(1, ceil(char_count / 8192))`.
- Follow `next_chunk_index` while `has_more` is true. The last chunk has `next_chunk_index: null`.
- Read `metadata` contains only `char_count`, `chunk_size_chars`, and `chunk_count`. Counts describe the available document text (the Provider-returned text online), not an independently verified full copy of the website.
- Unknown/expired online IDs and invalid/out-of-range chunk indices fail. The offline base reader can return `null` for a missing document; error payloads are not yet identical across all backends. Clients should handle both tool errors and missing results.

Migration from older clients: use `read_document`, not a `get_document` tool; `get_document: true` remains only the offline registration flag. Move length/chunk logic from search results to read results, allow `score: null`, and do not depend on the removed public URL/title/source/language fields. Internal metadata/document-store columns have not been removed.

## Online APIs and cache

### Credentials and provider-specific settings

The bundled adapters bind search and reader capabilities to the same vendor. `provider.name` is a registry key, not an arbitrary import path. A nonempty literal `provider.api_key` takes precedence; otherwise the adapter reads the environment variable named by `api_key_env`. There is no `${...}` substitution and the online worker does not automatically load a `.env` file.

| Common `provider` field | Meaning |
| --- | --- |
| `name` | Registered adapter: currently `zhipu` or `iqs` |
| `api_key`, `api_key_env` | Literal secret or the name of its environment variable |
| `base_url` | Trusted vendor API base; defaults to the corresponding URL in the example |
| `timeout_seconds` | HTTP client timeout, default `60` seconds; IQS also validates it against its page timeout |
| `search`, `read` | Vendor-specific request options; unknown option names are rejected |

To enter a key into the existing YAML without echoing it or placing it in shell history (choose `iqs` or `online` at the prompt):

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

This rewrites YAML formatting/comments. Keep local YAML, backups, temporary credential files, and runtime logs out of version control and shared artifacts; verify exclusion rules rather than assuming they exist. Use sanitized examples in documentation. Do not send your key in issue reports. Use trusted HTTPS API endpoints, especially when changing `base_url`.

| Adapter | Search | Reader | Configuration differences |
| --- | --- | --- | --- |
| `zhipu` | `web_search` → `search_result` | `reader` → `reader_result.content` | `search_engine`, `search_intent`, `count`, `search_recency_filter`, `content_size` |
| `iqs` | `search/unified` → `pageItems` | `readpage/basic` → `data.text` | `engineType`, `timeRange`, optional `category`, `advancedParams` |

Zhipu `search_recency_filter`: `oneDay`, `oneWeek`, `oneMonth`, `oneYear`, `noLimit`. Reader options are `no_cache`, `return_format` (must be `text`), `retain_images`, `with_images_summary`, and `with_links_summary`.

IQS `timeRange`: `OneDay`, `OneWeek`, `OneMonth`, `OneYear`, `NoLimit` (case differs from Zhipu). The adapter accepts `Generic`, `GenericAdvanced`, `LiteAdvanced`, and `Deep` engines. `advancedParams` supports `numResults` (1–50; the selected engine may impose its own behavior), `startPublishedDate`, and `endPublishedDate` (`YYYY-MM-DD`). The request asks for snippets without `mainText`, `markdownText`, `richMainBody`, or `summary`; `rerankScore` is requested for upstream ranking but not exposed as the public `score`.

IQS reading currently supports **Basic only**. Its YAML options are `pageTimeout` (0–100000 milliseconds), `maxAge` (nonnegative seconds), `readability.readabilityMode` (`none`, `normal`, `article`), `excludeAllImages`, `excludeAllLinks`, and `excludedTags`. `formats: [text]` is fixed in the adapter, not configurable. `timeout_seconds` must exceed `(pageTimeout + 6000) / 1000`. Non-success target `data.statusCode` and empty text are errors, not cached documents. Search long-text fields are not used as a substitute for lazy reading.

MCP calls override Zhipu `search.count` and IQS `advancedParams.numResults` with the request's k (or YAML `args.k` when omitted). The configured vendor counts remain only as fallbacks for direct adapter calls that omit k; they no longer limit or fix MCP requests at 10. Upstream caps, filters, and empty results can return fewer hits. `read.no_cache: true` (Zhipu) or `read.maxAge: 0` (IQS) disables the vendor-side cache, **not** this server's cache. The available extracted text may still be incomplete because of access restrictions, dynamic rendering, or the provider's extraction behavior.

Official references: [Zhipu Web Search](https://open.bigmodel.cn/api/paas/v4/web_search) (API endpoint), [IQS HTTP quick start](https://help.aliyun.com/zh/document_detail/2870227.html), [IQS search parameters](https://help.aliyun.com/zh/document_detail/2883041.html), [IQS Basic reader](https://help.aliyun.com/zh/document_detail/2983380.html).

### Lifecycle and limits

```text
search(query, k) → Provider.search(query, k) → URL registration → stable docid + snippet
read_document → docid lookup → cached text OR Provider.read(URL)
              → character counts → requested 8192-character chunk
```

IDs use `web_` plus the SHA-256 hex digest of the **exact URL string**. URL variants are not canonicalized; identical content under different URLs can have different IDs. Only returned search hits are registered. Repeating a URL yields the same ID but still needs a live mapping in that service process.

| Online `args` field | Default | Meaning |
| --- | ---: | --- |
| `content_cache_idle_ttl_seconds` | `1800` | Idle expiry of cached bodies; successful cached reads refresh access time |
| `content_cache_max_mib` | `256` | Sum of cached body UTF-8 bytes, in MiB; LRU eviction when necessary |
| `mapping_idle_ttl_seconds` | `86400` | Idle expiry of docid → URL mappings |
| `mapping_max_entries` | `100000` | Maximum mapping entries; least-recently-used entries evicted |
| `cache_cleanup_interval_seconds` | `60` | Background cleanup interval; requests also purge expired entries |

All values must be positive; capacity/count fields are integers. Keep mapping TTL substantially longer than body TTL. Mapping access is refreshed by search registration and known-ID read attempts. Body eviction leaves its mapping available; mapping eviction also drops its cached body. An unknown mapping requires another search. One body larger than the entire configured capacity is rejected; it is not truncated to fit.

The byte budget is **not** a process RSS cap: Python objects, metadata, mappings, and in-flight responses add memory overhead. Concurrent reads of the same uncached ID share one fetch within a process; different IDs are not globally rate-limited by this cache. Failed fetches are not cached and can be retried by a later request; the adapters do not implement an automatic retry/backoff loop.

Cache scope is one service process, shared across MCP connections, not one connection or rollout. There is no rollout-end signal, persistence, cross-worker sharing, or tenant isolation. Disconnecting a client does not clear the cache; restarting the process clears bodies and mappings. After body expiry/refetch a page can change, so old chunk numbers may refer to different content. Version pinning is intentionally not provided.

## Adding a Provider

New vendors require a small code adapter; existing registered vendors are selected by YAML and restart. Do not try to express arbitrary authentication, nested response parsing, and error semantics as field-name substitutions alone.

1. Add `searcher/providers/<vendor>.py` with a class accepting a provider-specific configuration dictionary. Validate allowed keys and values, resolve credentials, and own one reusable asynchronous HTTP client.
2. Implement the normalized internal contract below, including per-call k. Map it to the vendor's count parameter using a new request dictionary (copy nested options too); do not mutate shared configuration during concurrent calls. Validate vendor-specific bounds rather than silently ignoring k. Return extracted document text, not the search snippet, from `read`. A dedicated upstream endpoint named “reader” is not required, but document retrieval must remain meaningful after body eviction. If a vendor only embeds text in search results, explicitly design the refetch strategy instead of silently returning summaries as full documents.
3. Import/register the class in `searcher/providers/registry.py`, e.g. `PROVIDERS["vendor"] = VendorProvider` alongside the existing entries. Keep `__init__.py` and the `online_provider.py` compatibility facade unless there is a concrete reason to change them.
4. Add a service with `backend: online`, a free port, and `provider.name: vendor`. Vendor-specific options stay under `provider.search`/`provider.read`; shared cache/snippet settings remain under `args`. Time filters currently are startup settings, not extra Agent `search` arguments.
5. Run mock adapter tests, the shared cache tests below, and real MCP search/read acceptance with the vendor's key. Restart after changing code/configuration. Neither the shared tools nor the launcher should need vendor-specific branches.

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

Raise on HTTP/network/API failures, malformed responses, blocked target pages, or missing body text. An actual empty search returns `[]`; do not turn an API failure into an empty success. Avoid blocking I/O and credential/full-response logging. HTTP error handling, timeout/rate-limit policy, vendor defaults, extraction limits, and language/title mapping are adapter responsibilities. The shared layer owns IDs, public `score: null`, snippet truncation, caches, and chunks.

Adapter mock coverage should check exact URL/method/auth/body fields, nested response mapping, empty search, missing/invalid fields, non-2xx responses, vendor error codes, reader target failures, long Unicode text, and client cleanup. `httpx.MockTransport` can exercise these without credentials or charges. Shared cache tests do not replace vendor-specific response tests.

## Acceptance tests

### 1. Static and startup checks

```bash
python3 -m py_compile searcher/serve.py searcher/tools.py searcher/searchers/base.py \
  searcher/online_mcp_server.py searcher/online_service.py \
  searcher/providers/base.py searcher/providers/registry.py \
  searcher/providers/zhipu.py searcher/providers/iqs.py
python3 searcher/serve.py --config configs/services.local.yaml --all --check
```

Launch the chosen group and wait for `READY`. This does not make paid search/read calls. The following real API tests do, and results are live/non-deterministic.

### 2. Search, cross-connection read, and chunk boundaries

The test uses explicit `SSETransport` and `/mcp`. `/sse` is not this server's endpoint; constructing `Client(url)` for an SSE server may incorrectly select Streamable HTTP and produce HTTP 405. For a service deliberately started with `streamable-http`, use `Client(URL)` instead.

Save and run the following from the repository root. It prints complete results and validates the field contract. Change `MCP_URL` to test another service. `EXPECT_ONLINE=0` permits offline numeric scores; `REQUESTED_K` sets the per-call result count and `SNIPPET_MAX_CHARS` must match the server's configured character limit.

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

Each read opens a new MCP connection. For a single-chunk document the test explicitly skips the multi-chunk boundary check; choose a longer document/query to exercise it. The assertions assume the cached document has not expired between reads.

### 3. Online cache and failure acceptance

For real TTL behavior, use a **temporary test service/config**, set `content_cache_idle_ttl_seconds: 10`, keep mapping TTL longer (e.g. `300`), and restart it. Search once, save the docid, read chunk 0, immediately read it again, wait at least 12 seconds without reading, then read the **same ID without another search**. Check the service log for `Reader fetch`, `Reader cache hit`, and a later `Reader fetch`. Do not assert content equality after refetch. Restore production TTL afterwards; changing the YAML alone does not alter a running process.

To check single-flight live, read a freshly searched, uncached ID from three concurrent independent connections. Expect equal text from that fetch and exactly one new `Reader fetch` log entry. Sequential requests or a warm cache do not prove single-flight behavior.

The deterministic test below covers lazy reads, concurrent single-flight, TTL, LRU eviction, mapping expiry/capacity, failure-then-retry, oversized bodies, and bad indices without paid requests, network, or HF downloads. It runs directly against `OnlineService`, with character truncation disabled solely for the cache test. It advances stored timestamps rather than sleeping through real TTLs.

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

### Acceptance record and boundaries

The development deployment has reported successful online field checks, Zhipu cross-connection/chunk/TTL/concurrency/error checks, mock LRU/mapping/failure tests, and real IQS search followed by read. These are historical development results, not a guarantee for a fresh installation or every vendor engine. For the per-call-k/character-snippet change, the user additionally reported all four services exposing integer k with default 5 and a positive constraint, returning exactly 2 and 7 hits for k=2 and k=7, with every snippet at most 2048 characters. This verifies the core search changes; it does not prove that mixed-k Dense requests entered the same GPU batch or cover all boundaries/performance. No live tests were rerun by the assistant for this documentation synchronization. Record the date, dependency versions, non-secret configuration, backend/provider, test results, and fetch counts for each deployment. Never include API keys.

Remaining operational checks depend on deployment needs: vendor throttling/auth/network failures, sustained load and memory overhead, dynamic/blocked pages, multiple service groups, and client compatibility. `READY` and a successful smoke test do not replace retrieval-quality evaluation. Re-run offline smoke tests after shared Tool changes and provider-specific mocks after adapter changes.

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

Deleting IVF-Flat embedding shards does not affect serving that local ANN index, but a future ANN rebuild will require re-encoding the corpus. If shards are removed, the manifest's `embeddings_built` state should be updated accordingly.

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
