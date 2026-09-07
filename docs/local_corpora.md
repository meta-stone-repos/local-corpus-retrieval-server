# Local independent corpora

BrowseComp-Plus can keep multiple corpora side by side. Each corpus has its own
BM25 index, dense index, document store and manifest; indexes are never merged.

## Layout

```text
corpora/<name>/
  corpus.jsonl
  documents.sqlite
  manifest.json

indexes/corpora/<name>/
  bm25/
  qwen3-embedding-0.6b/
    corpus.shard*.pkl
```

`manifest.json` records the model/pooling/normalization settings used to build
the dense index. `mcp_server.py --corpus <name>` reads those settings back so
query-time encoding cannot accidentally use a different configuration.

The SQLite document store is used by local dense corpora for docid -> full-text
lookups. This avoids loading a very large corpus such as Wiki18 into a Python
dictionary at server startup.

## Generic JSONL

```bash
python scripts_build_index/build_corpus.py \
  --name my-corpus \
  --input /path/to/docs.jsonl \
  --format jsonl \
  --id-field id \
  --text-field text \
  --build bm25,dense \
  --gpus 4
```

## DPR Wiki18

The commonly used DPR Wikipedia passage corpus is available as
`psgs_w100.tsv.gz` and contains columns `id`, `text`, and `title`.

Download:

```bash
mkdir -p data/wiki18
wget -c \
  https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz \
  -O data/wiki18/psgs_w100.tsv.gz
```

First build a 10k canary corpus:

```bash
python scripts_build_index/build_corpus.py \
  --name wiki18-canary \
  --input data/wiki18/psgs_w100.tsv.gz \
  --format dpr-wiki18 \
  --limit 10000 \
  --build bm25,dense \
  --passage-max-len 512 \
  --gpus 4
```

Start the two independent services:

```bash
python searcher/mcp_server.py \
  --searcher-type bm25 \
  --corpus wiki18-canary \
  --get-document \
  --port 8090
```

```bash
CUDA_VISIBLE_DEVICES=4 python searcher/mcp_server.py \
  --searcher-type faiss \
  --corpus wiki18-canary \
  --get-document \
  --port 8091
```

After the canary passes, build the full corpus with a new name (recommended) or
rebuild with `--overwrite`. Multiple GPUs can encode independent Tevatron
shards in parallel:

```bash
python scripts_build_index/build_corpus.py \
  --name wiki18 \
  --input data/wiki18/psgs_w100.tsv.gz \
  --format dpr-wiki18 \
  --build bm25,dense \
  --passage-max-len 512 \
  --gpus 0,1,2,3,4,5,6,7 \
  --dense-shards 8
```

## Legacy BrowseComp usage

Existing commands without `--corpus` are unchanged. In particular, the
prebuilt BrowseComp dense index can continue using
`Tevatron/browsecomp-plus-corpus` from Hugging Face.
