#!/usr/bin/env python3
"""Build independent BM25 + dense indexes for a local corpus.

The script normalizes input into Tevatron-compatible JSONL, creates a SQLite
random-access document store, builds a Pyserini Lucene index, encodes dense
Qwen3 embeddings with Tevatron, and writes corpora/<name>/manifest.json.

DPR Wiki18 input (psgs_w100.tsv or .tsv.gz) is supported directly.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build an independent local retrieval corpus")
    parser.add_argument("--name", required=True, help="Corpus name, e.g. wiki18 or wiki18-canary")
    parser.add_argument(
        "--input",
        help=(
            "Input corpus path. Required for a fresh build; optional with "
            "--resume when an existing corpus.jsonl/documents.sqlite can be reused."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "dpr-wiki18"],
        default="jsonl",
        help="Input format (default: jsonl)",
    )
    parser.add_argument("--id-field", default="docid", help="JSONL document-id field")
    parser.add_argument("--text-field", default="text", help="JSONL text field")
    parser.add_argument("--title-field", default=None, help="Optional JSONL title field prepended to text")
    parser.add_argument("--limit", type=int, default=None, help="Only ingest the first N documents (for canary builds)")
    parser.add_argument(
        "--build",
        default="bm25,dense",
        help=(
            "Comma-separated targets: bm25,dense,ann (default: bm25,dense). "
            "'ann' builds IVF-Flat from existing dense embedding shards. "
            "Use 'none' for normalize-only."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing corpus directory/indexes")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse an existing normalized corpus and completed index artifacts. "
            "Missing dense shards are encoded; completed BM25/Dense stages are skipped."
        ),
    )
    parser.add_argument("--keep-build-files", action="store_true", help="Keep temporary Pyserini input files")
    parser.add_argument("--bm25-threads", type=int, default=16)
    parser.add_argument("--dense-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--pooling", default="eos")
    parser.add_argument("--passage-max-len", type=int, default=4096)
    parser.add_argument("--query-max-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPU ids for dense encoding, e.g. 0 or 0,1,2,3",
    )
    parser.add_argument(
        "--dense-shards",
        type=int,
        default=None,
        help="Number of dense encoding shards; defaults to number of GPUs",
    )
    parser.add_argument(
        "--dense-index-type",
        choices=["flat", "ivf-flat"],
        default="flat",
        help=(
            "Dense search index type. 'flat' keeps the current exact-search "
            "pickle shards; 'ivf-flat' additionally builds a trained FAISS "
            "IVF-Flat index for large corpora (default: flat)."
        ),
    )
    parser.add_argument(
        "--ivf-nlist",
        type=int,
        default=0,
        help="Number of IVF coarse lists. 0 chooses an automatic power-of-two value.",
    )
    parser.add_argument(
        "--ivf-nprobe",
        type=int,
        default=32,
        help="Number of IVF lists searched per query (default: 32).",
    )
    parser.add_argument(
        "--ivf-train-size",
        type=int,
        default=0,
        help="Number of vectors used to train IVF centroids. 0 chooses automatically.",
    )
    parser.add_argument(
        "--ivf-seed",
        type=int,
        default=42,
        help="Random seed used when sampling IVF training vectors (default: 42).",
    )
    parser.add_argument(
        "--ivf-threads",
        type=int,
        default=0,
        help=(
            "FAISS/OpenMP threads for IVF training and add. 0 chooses up to 64 "
            "threads from the CPUs available to the process (default: 0)."
        ),
    )
    parser.add_argument(
        "--ivf-niter",
        type=int,
        default=25,
        help="Number of FAISS coarse-clustering iterations (default: 25).",
    )
    parser.add_argument(
        "--ivf-backend",
        choices=["cpu", "gpu"],
        default="cpu",
        help=(
            "Backend for IVF coarse clustering and vector-to-list assignment. "
            "'cpu' uses FAISS CPU; 'gpu' uses PyTorch CUDA across --ivf-gpus "
            "while keeping the final on-disk FAISS index CPU-compatible "
            "(default: cpu)."
        ),
    )
    parser.add_argument(
        "--ivf-gpus",
        default="",
        help=(
            "Comma-separated CUDA-visible GPU ids used by --ivf-backend gpu, "
            "for example 4,5,6,7. The ids are independent of --gpus used for "
            "dense encoding."
        ),
    )
    parser.add_argument(
        "--ivf-gpu-batch-size",
        type=int,
        default=8192,
        help=(
            "Per-GPU assignment batch size for IVF GPU training/add "
            "(default: 8192). Lower this if GPU memory is limited."
        ),
    )
    parser.add_argument(
        "--ivf-gpu-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help=(
            "CUDA matmul dtype used for coarse assignment when --ivf-backend "
            "gpu. Centroid accumulation remains float32 (default: float32)."
        ),
    )
    parser.add_argument("--query-prefix", default=DEFAULT_QUERY_PREFIX)
    args = parser.parse_args()
    args._specified_flags = {
        token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")
    }
    return args


def validate_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("--name may contain only letters, numbers, '.', '_' and '-'")


def _flag_was_set(args, *flags: str) -> bool:
    specified = getattr(args, "_specified_flags", set())
    return any(flag in specified for flag in flags)


def load_existing_manifest(corpus_dir: Path) -> Dict[str, object] | None:
    path = corpus_dir / "manifest.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("name") not in {None, corpus_dir.name}:
        raise ValueError(
            f"Manifest name mismatch in {path}: expected '{corpus_dir.name}', "
            f"found '{manifest.get('name')}'"
        )
    return manifest


def apply_resume_defaults(args, manifest: Dict[str, object] | None) -> None:
    """Reuse build settings recorded in an existing manifest unless CLI overrides them."""

    if not manifest:
        return

    source = manifest.get("source", {}) or {}
    faiss_cfg = (manifest.get("retrievers", {}) or {}).get("faiss", {}) or {}

    def inherit(attr: str, value, *flags: str) -> None:
        if value is not None and not _flag_was_set(args, *flags):
            setattr(args, attr, value)

    inherit("input", source.get("input"), "--input")
    inherit("format", source.get("format"), "--format")
    inherit("limit", source.get("limit"), "--limit")
    inherit("dense_model", faiss_cfg.get("model_name"), "--dense-model")
    inherit("pooling", faiss_cfg.get("pooling"), "--pooling")
    inherit("passage_max_len", faiss_cfg.get("passage_max_len"), "--passage-max-len")
    inherit("query_max_len", faiss_cfg.get("max_length"), "--query-max-len")
    inherit("query_prefix", faiss_cfg.get("task_prefix"), "--query-prefix")

    if not _flag_was_set(args, "--normalize", "--no-normalize") and "normalize" in faiss_cfg:
        args.normalize = bool(faiss_cfg["normalize"])

    if not _flag_was_set(args, "--dtype"):
        dtype = faiss_cfg.get("torch_dtype")
        if dtype == "float16":
            args.dtype = "fp16"
        elif dtype == "bfloat16":
            args.dtype = "bf16"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def normalize_text(title: str | None, text: str) -> str:
    text = (text or "").strip()
    title = (title or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def iter_dpr_wiki18(path: Path) -> Iterator[Dict[str, str]]:
    with open_text(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"id", "text", "title"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"DPR Wiki18 TSV must have columns id, text, title; got {reader.fieldnames}"
            )
        for row in reader:
            yield {
                "docid": str(row["id"]),
                "document_text": normalize_text(row.get("title"), row.get("text", "")),
                "title": row.get("title", ""),
            }


def iter_jsonl(path: Path, id_field: str, text_field: str, title_field: str | None) -> Iterator[Dict[str, str]]:
    with open_text(path) as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if id_field not in row:
                raise ValueError(f"Missing id field '{id_field}' at line {line_no}")
            if text_field not in row:
                raise ValueError(f"Missing text field '{text_field}' at line {line_no}")
            title = row.get(title_field) if title_field else None
            yield {
                "docid": str(row[id_field]),
                "document_text": normalize_text(title, str(row[text_field])),
                **({"title": str(title)} if title is not None else {}),
            }


def iter_input(args) -> Iterator[Dict[str, str]]:
    path = Path(args.input)
    if not path.is_file():
        raise FileNotFoundError(f"Input corpus not found: {path}")
    if args.format == "dpr-wiki18":
        rows = iter_dpr_wiki18(path)
    else:
        rows = iter_jsonl(path, args.id_field, args.text_field, args.title_field)

    for i, row in enumerate(rows):
        if args.limit is not None and i >= args.limit:
            break
        if not row["document_text"]:
            continue
        yield row


def prepare_corpus(args, corpus_dir: Path) -> int:
    corpus_path = corpus_dir / "corpus.jsonl"
    sqlite_path = corpus_dir / "documents.sqlite"

    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("CREATE TABLE documents (docid TEXT PRIMARY KEY, text TEXT NOT NULL)")

    count = 0
    batch = []
    with corpus_path.open("w", encoding="utf-8") as out:
        try:
            for row in iter_input(args):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                batch.append((row["docid"], row["document_text"]))
                count += 1
                if len(batch) >= 10000:
                    conn.executemany("INSERT INTO documents(docid, text) VALUES (?, ?)", batch)
                    conn.commit()
                    batch.clear()
                    print(f"[prepare] {count:,} documents", flush=True)
            if batch:
                conn.executemany("INSERT INTO documents(docid, text) VALUES (?, ?)", batch)
                conn.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError(f"Duplicate docid encountered while building {args.name}: {e}") from e
        finally:
            conn.close()

    if count == 0:
        raise ValueError("No documents were produced from the input corpus")
    print(f"[prepare] completed: {count:,} documents")
    return count


def prepared_corpus_exists(corpus_dir: Path) -> bool:
    return (corpus_dir / "corpus.jsonl").is_file() and (
        corpus_dir / "documents.sqlite"
    ).is_file()


def document_count(corpus_dir: Path) -> int:
    sqlite_path = corpus_dir / "documents.sqlite"
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"Document store not found: {sqlite_path}")
    conn = sqlite3.connect(sqlite_path)
    try:
        row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
    finally:
        conn.close()
    if not row:
        raise ValueError(f"Could not count documents in {sqlite_path}")
    return int(row[0])


def run(cmd: List[str], env=None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def corpus_row_text(row: Dict[str, object]) -> str:
    """Read text from legacy or canonical normalized corpus rows.

    Field priority preserves the existing build_corpus behavior.
    """
    for field in ("document_text", "text", "contents"):
        value = row.get(field)
        if value is not None:
            return str(value)

    raise KeyError(
        "Corpus row must contain one of: "
        "document_text, text, contents"
    )


def build_bm25(args, corpus_dir: Path, index_dir: Path) -> None:
    build_input_dir = corpus_dir / ".build" / "pyserini"
    build_input_dir.mkdir(parents=True, exist_ok=True)
    docs_path = build_input_dir / "docs.jsonl"

    with (corpus_dir / "corpus.jsonl").open("r", encoding="utf-8") as src, docs_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            row = json.loads(line)
            dst.write(
                json.dumps(
                    {
                        "id": str(row["docid"]),
                        "contents": corpus_row_text(row),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(build_input_dir),
            "--index",
            str(index_dir),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            str(args.bm25_threads),
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ]
    )

    if not args.keep_build_files:
        shutil.rmtree(corpus_dir / ".build", ignore_errors=True)


def dense_model_slug(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1].lower()


def dense_command(args, corpus_path: Path, output_path: Path, shard: int, shards: int) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "tevatron.retriever.driver.encode",
        "--model_name_or_path",
        args.dense_model,
        "--dataset_name",
        "json",
        "--dataset_path",
        str(corpus_path),
        "--dataset_split",
        "train",
        "--encode_output_path",
        str(output_path),
        "--passage_max_len",
        str(args.passage_max_len),
        "--pooling",
        args.pooling,
        "--padding_side",
        "left",
        "--passage_prefix",
        "",
        "--per_device_eval_batch_size",
        str(args.batch_size),
        "--dataset_number_of_shards",
        str(shards),
        "--dataset_shard_index",
        str(shard),
    ]
    if args.normalize:
        cmd.append("--normalize")
    cmd.append("--fp16" if args.dtype == "fp16" else "--bf16")
    return cmd


def build_dense(args, corpus_dir: Path, dense_dir: Path) -> None:
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")

    existing_layout = dense_shard_layout(dense_dir)
    if existing_layout is not None:
        existing_total, _ = existing_layout
        if args.dense_shards is not None and args.dense_shards != existing_total:
            raise ValueError(
                f"Existing dense shards use {existing_total} shards, but "
                f"--dense-shards={args.dense_shards}. Remove/rebuild the dense directory "
                "or resume with the existing shard count."
            )
        shards = existing_total
    else:
        shards = args.dense_shards or len(gpus)

    if shards < 1:
        raise ValueError("--dense-shards must be >= 1")

    dense_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = corpus_dir / "corpus.jsonl"

    missing = []
    for shard in range(shards):
        output_path = dense_dir / f"corpus.shard{shard:02d}_of_{shards:02d}.pkl"
        if args.resume and output_path.is_file() and output_path.stat().st_size > 0:
            print(f"[dense] resume: shard {shard + 1}/{shards} already exists -> skip")
        else:
            missing.append(shard)

    if not missing:
        print(f"[dense] resume: all {shards} shards already exist -> skip encoding")
        return

    # Run batches so that each concurrently running shard owns a distinct GPU.
    for batch_start in range(0, len(missing), len(gpus)):
        processes = []
        batch = missing[batch_start : batch_start + len(gpus)]
        for offset, shard in enumerate(batch):
            gpu = gpus[offset]
            output_path = dense_dir / f"corpus.shard{shard:02d}_of_{shards:02d}.pkl"
            cmd = dense_command(args, corpus_path, output_path, shard, shards)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            print(f"[dense] shard {shard + 1}/{shards} -> GPU {gpu}")
            print("+", " ".join(cmd), flush=True)
            processes.append((shard, subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env)))

        failed = []
        for shard, proc in processes:
            code = proc.wait()
            if code != 0:
                failed.append((shard, code))
        if failed:
            raise RuntimeError(f"Dense encoding failed: {failed}")

    complete, total, found = dense_shards_complete(dense_dir, expected_shards=shards)
    if not complete:
        raise RuntimeError(
            f"Dense encoding returned successfully but shard set is incomplete: "
            f"found {found}/{total} in {dense_dir}"
        )


_DENSE_SHARD_RE = re.compile(r"^corpus\.shard(\d+)_of_(\d+)\.pkl$")


def dense_shard_layout(dense_dir: Path) -> Tuple[int, Dict[int, Path]] | None:
    """Return (total_shards, shard_paths) for a consistent Tevatron shard set."""

    if not dense_dir.is_dir():
        return None

    totals = set()
    shards: Dict[int, Path] = {}
    for path in dense_dir.glob("corpus.shard*_of_*.pkl"):
        match = _DENSE_SHARD_RE.match(path.name)
        if not match:
            continue
        shard = int(match.group(1))
        total = int(match.group(2))
        totals.add(total)
        if shard in shards:
            raise ValueError(f"Duplicate dense shard id {shard} in {dense_dir}")
        shards[shard] = path

    if not totals:
        return None
    if len(totals) != 1:
        raise ValueError(f"Mixed dense shard layouts in {dense_dir}: {sorted(totals)}")
    total = totals.pop()
    if any(shard < 0 or shard >= total for shard in shards):
        raise ValueError(f"Invalid dense shard id for {total}-shard layout in {dense_dir}")
    return total, shards


def dense_shards_complete(
    dense_dir: Path, expected_shards: int | None = None
) -> Tuple[bool, int, int]:
    layout = dense_shard_layout(dense_dir)
    if layout is None:
        total = expected_shards or 0
        return False, total, 0
    total, shards = layout
    if expected_shards is not None and total != expected_shards:
        return False, expected_shards, len(shards)
    complete_ids = {
        shard
        for shard, path in shards.items()
        if path.is_file() and path.stat().st_size > 0
    }
    complete = len(complete_ids) == total and complete_ids == set(range(total))
    return complete, total, len(complete_ids)


def _load_dense_shard(
    path: Path, *, stringify_lookup: bool = True
) -> Tuple["np.ndarray", List[str] | object]:
    """Load one Tevatron embedding shard.

    ANN training only needs vectors, so callers can set stringify_lookup=False
    to avoid converting millions of document ids to Python strings during the
    sampling pass. The pickle still contains the lookup object, but we can drop
    it immediately after validation.
    """
    import pickle
    import numpy as np

    with path.open("rb") as f:
        reps, lookup = pickle.load(f)
    reps = np.asarray(reps, dtype=np.float32)
    if reps.ndim != 2:
        raise ValueError(f"Dense shard must be 2D, got {reps.shape}: {path}")
    if reps.shape[0] != len(lookup):
        raise ValueError(
            f"Dense shard/lookup mismatch in {path}: {reps.shape[0]} vs {len(lookup)}"
        )

    if not stringify_lookup:
        return reps, lookup

    # Tevatron normally writes string ids already. Avoid an O(N) Python-level
    # str() pass when that is the case; only normalize non-string lookups.
    if isinstance(lookup, list) and (not lookup or isinstance(lookup[0], str)):
        return reps, lookup
    return reps, [str(x) for x in lookup]


def _auto_nlist(n_vectors: int) -> int:
    """Choose a conservative power-of-two IVF list count.

    Roughly targets 4*sqrt(N), capped so small canaries stay trainable and
    Wiki18-scale corpora land around 16k lists.
    """
    import math

    target = max(64, min(65536, int(4 * math.sqrt(max(1, n_vectors)))))
    power = 1 << max(0, int(round(math.log2(target))))
    return max(1, min(power, n_vectors))


def _elapsed(start: float) -> str:
    seconds = max(0.0, time.monotonic() - start)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s"


def _choose_ivf_threads(args, faiss) -> int:
    available = os.cpu_count() or 1
    faiss_max = int(faiss.omp_get_max_threads())
    if args.ivf_threads > 0:
        threads = int(args.ivf_threads)
    else:
        # 64 is a conservative default on shared hosts. GPU IVF construction
        # only uses these CPU threads for the inexpensive add_core stage.
        threads = min(64, available, faiss_max)
    threads = max(1, min(threads, available, faiss_max))
    faiss.omp_set_num_threads(threads)
    return threads


def _parse_ivf_gpu_ids(spec: str) -> List[int]:
    ids: List[int] = []
    seen = set()
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        gpu = int(raw)
        if gpu < 0:
            raise ValueError("--ivf-gpus entries must be non-negative CUDA ids")
        if gpu not in seen:
            ids.append(gpu)
            seen.add(gpu)
    if not ids:
        raise ValueError(
            "--ivf-backend gpu requires --ivf-gpus, e.g. --ivf-gpus 4,5,6,7"
        )
    return ids


def _validate_torch_gpus(gpu_ids: List[int]):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("--ivf-backend gpu requested but torch.cuda is unavailable")
    n = torch.cuda.device_count()
    bad = [x for x in gpu_ids if x >= n]
    if bad:
        raise ValueError(
            f"Invalid --ivf-gpus {bad}; this process sees CUDA devices 0..{n - 1}"
        )
    return torch


def _torch_compute_dtype(torch, name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _gpu_partial_kmeans_stats(
    torch,
    data,
    centroids_np,
    *,
    device_id: int,
    batch_size: int,
    compute_dtype_name: str,
):
    """Assign one resident training partition and accumulate centroid sums."""
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    compute_dtype = _torch_compute_dtype(torch, compute_dtype_name)
    centroids = torch.as_tensor(centroids_np, device=device, dtype=compute_dtype)
    k, dim = centroids_np.shape
    sums = torch.zeros((k, dim), device=device, dtype=torch.float32)
    counts = torch.zeros(k, device=device, dtype=torch.int64)
    objective = 0.0

    with torch.inference_mode():
        for start in range(0, data.shape[0], batch_size):
            x32 = data[start : start + batch_size]
            x = x32 if compute_dtype == torch.float32 else x32.to(compute_dtype)
            scores = x @ centroids.T
            best, assign = torch.max(scores, dim=1)
            objective += float(best.float().sum().item())
            sums.index_add_(0, assign, x32)
            counts += torch.bincount(assign, minlength=k)
            del x, scores, best, assign

    torch.cuda.synchronize(device)
    return sums.cpu().numpy(), counts.cpu().numpy(), objective


def _torch_multi_gpu_spherical_kmeans(
    train,
    *,
    nlist: int,
    niter: int,
    gpu_ids: List[int],
    batch_size: int,
    compute_dtype_name: str,
    seed: int,
):
    """Spherical k-means using PyTorch CUDA for the expensive assignment step.

    Faiss IVF with inner-product metric uses spherical centroids.  We mirror
    that behavior here, but execute assignments independently on the selected
    GPUs.  Only centroid sums/counts are reduced on CPU between iterations.
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    torch = _validate_torch_gpus(gpu_ids)
    if batch_size < 1:
        raise ValueError("--ivf-gpu-batch-size must be >= 1")

    rng = np.random.default_rng(seed)
    if train.shape[0] < nlist:
        raise ValueError(
            f"IVF training needs at least nlist vectors: {train.shape[0]} < {nlist}"
        )

    # Faiss clustering defaults to uniform random sampled centroids. Keep the
    # same simple/reproducible initialization here.
    init_ids = rng.choice(train.shape[0], size=nlist, replace=False)
    centroids = np.ascontiguousarray(train[init_ids], dtype=np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids /= np.maximum(norms, 1e-12)

    parts_np = [np.ascontiguousarray(x, dtype=np.float32) for x in np.array_split(train, len(gpu_ids))]
    resident = []
    print(
        f"[ann][gpu] placing {train.shape[0]:,} training vectors across GPUs {gpu_ids}",
        flush=True,
    )
    for gpu, part in zip(gpu_ids, parts_np):
        torch.cuda.set_device(gpu)
        tensor = torch.from_numpy(part).to(device=f"cuda:{gpu}", dtype=torch.float32)
        resident.append((gpu, tensor))
        print(
            f"[ann][gpu] cuda:{gpu}: training vectors={part.shape[0]:,}, "
            f"resident={tensor.numel() * tensor.element_size() / (1024 ** 3):.2f} GiB",
            flush=True,
        )
    del parts_np

    try:
        for iteration in range(1, niter + 1):
            iter_start = time.monotonic()
            partials = []
            with ThreadPoolExecutor(max_workers=len(resident)) as pool:
                futures = [
                    pool.submit(
                        _gpu_partial_kmeans_stats,
                        torch,
                        data,
                        centroids,
                        device_id=gpu,
                        batch_size=batch_size,
                        compute_dtype_name=compute_dtype_name,
                    )
                    for gpu, data in resident
                ]
                for future in futures:
                    partials.append(future.result())

            sums = np.zeros_like(centroids, dtype=np.float32)
            counts = np.zeros(nlist, dtype=np.int64)
            objective = 0.0
            for psums, pcounts, pobj in partials:
                sums += psums
                counts += pcounts
                objective += pobj

            empty = np.flatnonzero(counts == 0)
            nonempty = counts > 0
            centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
            if empty.size:
                repl = rng.choice(train.shape[0], size=empty.size, replace=True)
                centroids[empty] = train[repl]

            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            centroids /= np.maximum(norms, 1e-12)
            mean_ip = objective / max(1, train.shape[0])
            print(
                f"[ann][gpu] kmeans iter {iteration}/{niter}: "
                f"mean_ip={mean_ip:.6f}, empty={empty.size:,}, "
                f"elapsed={_elapsed(iter_start)}",
                flush=True,
            )
    finally:
        del resident
        for gpu in gpu_ids:
            with torch.cuda.device(gpu):
                torch.cuda.empty_cache()

    return np.ascontiguousarray(centroids, dtype=np.float32)


def _gpu_assign_worker(
    torch,
    reps,
    centroids_np,
    *,
    device_id: int,
    start: int,
    end: int,
    batch_size: int,
    compute_dtype_name: str,
):
    """Assign a CPU vector slice to IVF lists on one GPU."""
    import numpy as np

    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    compute_dtype = _torch_compute_dtype(torch, compute_dtype_name)
    centroids = torch.as_tensor(centroids_np, device=device, dtype=compute_dtype)
    out = np.empty(end - start, dtype=np.int64)

    with torch.inference_mode():
        cursor = 0
        for pos in range(start, end, batch_size):
            stop = min(end, pos + batch_size)
            # Copy only the current batch, so GPU memory does not scale with a
            # 14+ GiB embedding shard.
            cpu_batch = np.ascontiguousarray(reps[pos:stop], dtype=np.float32)
            x = torch.from_numpy(cpu_batch).to(device=device, dtype=compute_dtype)
            scores = x @ centroids.T
            assign = torch.argmax(scores, dim=1)
            n = stop - pos
            out[cursor : cursor + n] = assign.cpu().numpy().astype(np.int64, copy=False)
            cursor += n
            del cpu_batch, x, scores, assign

    torch.cuda.synchronize(device)
    return start, out


def _torch_multi_gpu_assign(
    reps,
    centroids,
    *,
    gpu_ids: List[int],
    batch_size: int,
    compute_dtype_name: str,
):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    torch = _validate_torch_gpus(gpu_ids)
    n = int(reps.shape[0])
    if n == 0:
        return np.empty(0, dtype=np.int64)

    # Contiguous row ranges keep output ids aligned with the existing lookup.
    bounds = np.linspace(0, n, num=len(gpu_ids) + 1, dtype=np.int64)
    assignments = np.empty(n, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = []
        for i, gpu in enumerate(gpu_ids):
            start, end = int(bounds[i]), int(bounds[i + 1])
            if start == end:
                continue
            futures.append(
                pool.submit(
                    _gpu_assign_worker,
                    torch,
                    reps,
                    centroids,
                    device_id=gpu,
                    start=start,
                    end=end,
                    batch_size=batch_size,
                    compute_dtype_name=compute_dtype_name,
                )
            )
        for future in futures:
            start, local = future.result()
            assignments[start : start + local.shape[0]] = local
    return assignments


def build_ivf_flat_index(
    args, dense_dir: Path, *, expected_total: int | None = None
) -> Dict[str, object]:
    """Build a CPU-readable FAISS IVF-Flat index from Tevatron pickle shards.

    `--ivf-backend cpu` uses the normal FAISS CPU training/add path.
    `--ivf-backend gpu` keeps the final index CPU-compatible but accelerates
    spherical k-means assignment and full-corpus coarse assignment with the
    user-selected PyTorch CUDA devices. This avoids requiring a faiss-gpu wheel.
    """
    import pickle
    import numpy as np
    import faiss

    overall_start = time.monotonic()
    shard_paths = sorted(dense_dir.glob("corpus.shard*.pkl"))
    if not shard_paths:
        raise FileNotFoundError(f"No dense shards found in {dense_dir}")

    total_hint = int(expected_total) if expected_total is not None else 0
    if total_hint <= 0:
        raise ValueError(
            "IVF build requires the normalized corpus document count so it can "
            "sample and train without an extra full embedding-shard scan"
        )

    nlist = int(args.ivf_nlist) if args.ivf_nlist > 0 else _auto_nlist(total_hint)
    nlist = max(1, min(nlist, total_hint))

    auto_train = min(total_hint, max(100_000, 50 * nlist))
    train_size = int(args.ivf_train_size) if args.ivf_train_size > 0 else auto_train
    train_size = max(nlist, min(train_size, total_hint))

    if args.ivf_niter < 1:
        raise ValueError("--ivf-niter must be >= 1")

    threads = _choose_ivf_threads(args, faiss)
    gpu_ids: List[int] = []
    if args.ivf_backend == "gpu":
        gpu_ids = _parse_ivf_gpu_ids(args.ivf_gpus)
        _validate_torch_gpus(gpu_ids)
        if args.ivf_gpu_batch_size < 1:
            raise ValueError("--ivf-gpu-batch-size must be >= 1")

    backend_desc = (
        f"gpu:{','.join(map(str, gpu_ids))}/{args.ivf_gpu_dtype}"
        if gpu_ids
        else f"cpu/{threads}threads"
    )
    print(
        f"[ann] plan: vectors={total_hint:,}, shards={len(shard_paths)}, "
        f"nlist={nlist:,}, nprobe={args.ivf_nprobe}, train_size={train_size:,}, "
        f"niter={args.ivf_niter}, backend={backend_desc}",
        flush=True,
    )

    rng = np.random.default_rng(args.ivf_seed)
    per_shard = max(1, (train_size + len(shard_paths) - 1) // len(shard_paths))
    samples = []
    shard_meta = []
    total = 0
    dim = None

    print("[ann] stage 1/4: sampling IVF training vectors", flush=True)
    for i, path in enumerate(shard_paths, start=1):
        shard_start = time.monotonic()
        print(
            f"[ann] sample shard {i}/{len(shard_paths)}: loading {path.name}",
            flush=True,
        )
        reps, lookup = _load_dense_shard(path, stringify_lookup=False)
        shard_count = int(reps.shape[0])
        shard_dim = int(reps.shape[1])
        if dim is None:
            dim = shard_dim
        elif shard_dim != dim:
            raise ValueError(f"Embedding dimension mismatch in {path}: {shard_dim} vs {dim}")

        take = min(per_shard, shard_count)
        if take == shard_count:
            sample = np.array(reps, dtype=np.float32, copy=True, order="C")
        else:
            ids = rng.choice(shard_count, size=take, replace=False)
            sample = np.ascontiguousarray(reps[ids], dtype=np.float32)
        samples.append(sample)
        shard_meta.append((path, shard_count))
        total += shard_count
        del reps, lookup, sample
        print(
            f"[ann] sample shard {i}/{len(shard_paths)}: "
            f"vectors={shard_count:,}, sampled={take:,}, elapsed={_elapsed(shard_start)}",
            flush=True,
        )

    if total != total_hint:
        raise RuntimeError(
            f"Dense vectors/document-count mismatch: embeddings={total:,}, "
            f"corpus={total_hint:,}"
        )
    if total == 0 or dim is None:
        raise ValueError("Cannot build IVF index from zero vectors")

    concat_start = time.monotonic()
    print(f"[ann] assembling training matrix from {len(samples)} samples...", flush=True)
    train = np.concatenate(samples, axis=0)
    del samples
    if train.shape[0] > train_size:
        ids = rng.choice(train.shape[0], size=train_size, replace=False)
        train = np.ascontiguousarray(train[ids], dtype=np.float32)
    else:
        train = np.ascontiguousarray(train, dtype=np.float32)
    print(
        f"[ann] training matrix ready: shape={train.shape}, "
        f"size={train.nbytes / (1024 ** 3):.2f} GiB, elapsed={_elapsed(concat_start)}",
        flush=True,
    )

    train_start = time.monotonic()
    if args.ivf_backend == "gpu":
        print(
            f"[ann] stage 2/4: GPU spherical k-means "
            f"({nlist:,} centroids, {args.ivf_niter} iterations, GPUs={gpu_ids}, "
            f"batch={args.ivf_gpu_batch_size:,}, dtype={args.ivf_gpu_dtype})",
            flush=True,
        )
        centroids = _torch_multi_gpu_spherical_kmeans(
            train,
            nlist=nlist,
            niter=int(args.ivf_niter),
            gpu_ids=gpu_ids,
            batch_size=int(args.ivf_gpu_batch_size),
            compute_dtype_name=args.ivf_gpu_dtype,
            seed=int(args.ivf_seed),
        )
        quantizer = faiss.IndexFlatIP(dim)
        quantizer.add(centroids)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            raise RuntimeError("GPU-trained IVF quantizer did not initialize a trained CPU index")
        print(
            f"[ann] GPU IVF training complete: elapsed={_elapsed(train_start)}, "
            f"is_trained={bool(index.is_trained)}",
            flush=True,
        )
    else:
        print(
            f"[ann] stage 2/4: training IVF coarse quantizer "
            f"({nlist:,} centroids, {args.ivf_niter} iterations, {threads} threads)",
            flush=True,
        )
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        if hasattr(index, "cp"):
            index.cp.verbose = True
            index.cp.niter = int(args.ivf_niter)
            if hasattr(index.cp, "seed"):
                index.cp.seed = int(args.ivf_seed)
        if hasattr(index, "verbose"):
            index.verbose = True
        print("[ann] FAISS clustering output follows:", flush=True)
        index.train(train)
        print(
            f"[ann] IVF training complete: elapsed={_elapsed(train_start)}, "
            f"is_trained={bool(index.is_trained)}",
            flush=True,
        )
    del train

    index.nprobe = max(1, min(int(args.ivf_nprobe), nlist))
    print("[ann] stage 3/4: adding all dense vectors", flush=True)
    lookup_all: List[str] = []
    for i, (path, shard_count) in enumerate(shard_meta, start=1):
        shard_start = time.monotonic()
        print(f"[ann] add shard {i}/{len(shard_meta)}: loading {path.name}", flush=True)
        reps, lookup = _load_dense_shard(path, stringify_lookup=True)
        reps = np.ascontiguousarray(reps, dtype=np.float32)
        if args.ivf_backend == "gpu":
            assign_start = time.monotonic()
            print(
                f"[ann][gpu] assign shard {i}/{len(shard_meta)} across GPUs {gpu_ids}",
                flush=True,
            )
            assignments = _torch_multi_gpu_assign(
                reps,
                centroids,
                gpu_ids=gpu_ids,
                batch_size=int(args.ivf_gpu_batch_size),
                compute_dtype_name=args.ivf_gpu_dtype,
            )
            assignments = np.ascontiguousarray(assignments, dtype=np.int64)
            if not hasattr(index, "add_core"):
                raise RuntimeError(
                    "This FAISS Python build does not expose IndexIVFFlat.add_core; "
                    "cannot add GPU-preassigned vectors to a CPU IVF index"
                )
            index.add_core(
                reps.shape[0],
                faiss.swig_ptr(reps),
                None,
                faiss.swig_ptr(assignments),
            )
            print(
                f"[ann][gpu] assignment shard {i}/{len(shard_meta)} complete: "
                f"elapsed={_elapsed(assign_start)}",
                flush=True,
            )
            del assignments
        else:
            index.add(reps)
        lookup_all.extend(lookup)
        del reps, lookup
        print(
            f"[ann] add shard {i}/{len(shard_meta)} complete: "
            f"vectors={shard_count:,}, ntotal={int(index.ntotal):,}, "
            f"elapsed={_elapsed(shard_start)}",
            flush=True,
        )

    if int(index.ntotal) != len(lookup_all):
        raise RuntimeError(
            f"ANN index/lookup mismatch after build: {index.ntotal} vs {len(lookup_all)}"
        )
    if int(index.ntotal) != total:
        raise RuntimeError(f"ANN index/vector mismatch after build: {index.ntotal} vs {total}")

    print("[ann] stage 4/4: writing IVF index and lookup", flush=True)
    write_start = time.monotonic()
    index_path = dense_dir / "index.ivf-flat.faiss"
    lookup_path = dense_dir / "index.lookup.pkl"
    faiss.write_index(index, str(index_path))
    print(
        f"[ann] wrote {index_path} ({index_path.stat().st_size / (1024 ** 3):.2f} GiB)",
        flush=True,
    )
    with lookup_path.open("wb") as f:
        pickle.dump(lookup_all, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[ann] wrote {lookup_path} ({lookup_path.stat().st_size / (1024 ** 2):.1f} MiB)",
        flush=True,
    )
    print(
        f"[ann] IVF-Flat build complete: total elapsed={_elapsed(overall_start)}, "
        f"write elapsed={_elapsed(write_start)}",
        flush=True,
    )

    meta: Dict[str, object] = {
        "index_type": "ivf-flat",
        "index_path": index_path,
        "lookup_path": lookup_path,
        "nlist": nlist,
        "nprobe": max(1, min(int(args.ivf_nprobe), nlist)),
        "train_size": train_size,
        "ntotal": total,
        "dimension": dim,
        "niter": int(args.ivf_niter),
        "threads": threads,
        "build_backend": args.ivf_backend,
    }
    if gpu_ids:
        meta["build_gpus"] = gpu_ids
        meta["gpu_batch_size"] = int(args.ivf_gpu_batch_size)
        meta["gpu_dtype"] = args.ivf_gpu_dtype
    return meta


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def write_manifest(
    args,
    corpus_dir: Path,
    count: int,
    bm25_dir: Path,
    dense_dir: Path,
    *,
    bm25_built: bool | None = None,
    dense_built: bool | None = None,
    embeddings_built: bool | None = None,
    dense_index_type: str | None = None,
    ann_meta: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Atomically checkpoint corpus/index state without dropping unknown metadata.

    Keeping unknown keys is intentional: later preprocessing may add document or
    corpus metadata used by get_document, and a resume run must not erase it.
    """

    manifest_path = corpus_dir / "manifest.json"
    manifest = load_existing_manifest(corpus_dir) or {}
    manifest["schema_version"] = max(1, int(manifest.get("schema_version", 1)))
    manifest["name"] = args.name
    manifest["document_count"] = int(count)

    source = dict(manifest.get("source", {}) or {})
    if args.input:
        source["input"] = str(Path(args.input).resolve())
    source.setdefault("input", None)
    source["format"] = args.format
    source["limit"] = args.limit

    input_manifest_path = corpus_dir / "input_manifest.json"
    if input_manifest_path.is_file():
        source["input_manifest"] = rel(input_manifest_path)

    manifest["source"] = source

    corpus_cfg = dict(manifest.get("corpus", {}) or {})
    corpus_cfg["path"] = rel(corpus_dir / "corpus.jsonl")
    corpus_cfg["document_store"] = rel(corpus_dir / "documents.sqlite")
    manifest["corpus"] = corpus_cfg

    retrievers = dict(manifest.get("retrievers", {}) or {})

    bm25_cfg = dict(retrievers.get("bm25", {}) or {})
    bm25_cfg.setdefault("built", False)
    if bm25_built is not None:
        bm25_cfg["built"] = bool(bm25_built)
    bm25_cfg["index_path"] = rel(bm25_dir)
    retrievers["bm25"] = bm25_cfg

    faiss_cfg = dict(retrievers.get("faiss", {}) or {})
    faiss_cfg.setdefault("built", False)
    if dense_built is not None:
        faiss_cfg["built"] = bool(dense_built)
    if embeddings_built is not None:
        faiss_cfg["embeddings_built"] = bool(embeddings_built)

    faiss_cfg["embedding_shards"] = rel(dense_dir / "corpus.shard*.pkl")
    faiss_cfg["document_store"] = rel(corpus_dir / "documents.sqlite")
    faiss_cfg["model_name"] = args.dense_model
    faiss_cfg["normalize"] = args.normalize
    faiss_cfg["pooling"] = args.pooling
    faiss_cfg["torch_dtype"] = "float16" if args.dtype == "fp16" else "bfloat16"
    faiss_cfg["task_prefix"] = args.query_prefix
    faiss_cfg["max_length"] = args.query_max_len
    faiss_cfg["passage_max_len"] = args.passage_max_len

    if ann_meta:
        faiss_cfg["built"] = True
        faiss_cfg["embeddings_built"] = True
        faiss_cfg["index_type"] = str(ann_meta["index_type"])
        faiss_cfg["index_path"] = rel(Path(ann_meta["index_path"]))
        faiss_cfg["lookup_path"] = rel(Path(ann_meta["lookup_path"]))
        for key in ("nlist", "nprobe", "train_size", "ntotal", "dimension", "niter", "threads", "gpu_batch_size"):
            if key in ann_meta:
                faiss_cfg[key] = int(ann_meta[key])
        for key in ("build_backend", "gpu_dtype"):
            if key in ann_meta:
                faiss_cfg[key] = str(ann_meta[key])
        if "build_gpus" in ann_meta:
            faiss_cfg["build_gpus"] = [int(x) for x in ann_meta["build_gpus"]]
    elif dense_built is not None:
        # Only switch the active runtime backend after that backend is complete.
        index_type = dense_index_type or "flat"
        faiss_cfg["index_type"] = index_type
        if index_type == "flat":
            faiss_cfg["index_path"] = rel(dense_dir / "corpus.shard*.pkl")
            for key in ("lookup_path", "nlist", "nprobe", "train_size", "ntotal", "dimension"):
                faiss_cfg.pop(key, None)

    faiss_cfg.setdefault("index_type", "flat")
    faiss_cfg.setdefault("index_path", rel(dense_dir / "corpus.shard*.pkl"))
    retrievers["faiss"] = faiss_cfg
    manifest["retrievers"] = retrievers

    tmp_path = manifest_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, manifest_path)
    print(f"[manifest] checkpoint -> {manifest_path}")
    return manifest


def main():
    args = parse_args()
    validate_name(args.name)

    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    build_targets = {x.strip() for x in args.build.split(",") if x.strip()}
    if build_targets == {"none"}:
        build_targets = set()
    unknown = build_targets - {"bm25", "dense", "ann"}
    if unknown:
        raise ValueError(f"Unknown --build target(s): {sorted(unknown)}")

    if "ann" in build_targets and args.dense_index_type != "ivf-flat":
        raise ValueError("--build ann currently requires --dense-index-type ivf-flat")

    corpus_dir = PROJECT_ROOT / "corpora" / args.name
    index_root = PROJECT_ROOT / "indexes" / "corpora" / args.name

    existing_manifest = load_existing_manifest(corpus_dir)
    if args.resume:
        apply_resume_defaults(args, existing_manifest)

    bm25_dir = index_root / "bm25"
    dense_dir = index_root / dense_model_slug(args.dense_model)
    if args.resume and existing_manifest:
        embedding_shards = (
            (existing_manifest.get("retrievers", {}) or {})
            .get("faiss", {})
            .get("embedding_shards")
        )
        if embedding_shards:
            dense_dir = resolve_project_path(embedding_shards).parent

    if corpus_dir.exists() or index_root.exists():
        if args.overwrite:
            shutil.rmtree(corpus_dir, ignore_errors=True)
            shutil.rmtree(index_root, ignore_errors=True)
            existing_manifest = None
        elif not args.resume:
            raise FileExistsError(
                f"Corpus '{args.name}' already exists. Use --resume to reuse it "
                "or --overwrite to rebuild it."
            )

    corpus_dir.mkdir(parents=True, exist_ok=True)
    index_root.mkdir(parents=True, exist_ok=True)

    corpus_jsonl = corpus_dir / "corpus.jsonl"
    document_store = corpus_dir / "documents.sqlite"
    if args.resume and prepared_corpus_exists(corpus_dir):
        count = document_count(corpus_dir)
        print(f"[prepare] resume: reuse existing corpus ({count:,} documents)")
    else:
        partial_prepare = corpus_jsonl.exists() or document_store.exists()
        if args.resume and partial_prepare:
            raise RuntimeError(
                f"Corpus '{args.name}' has only a partial normalized corpus. "
                "Refusing to guess whether it is complete; use --overwrite to rebuild."
            )
        if not args.input:
            raise ValueError("--input is required when the normalized corpus does not already exist")
        count = prepare_corpus(args, corpus_dir)

    # First checkpoint happens immediately after corpus preparation/recovery.
    manifest = write_manifest(args, corpus_dir, count, bm25_dir, dense_dir)

    bm25_cfg = (manifest.get("retrievers", {}) or {}).get("bm25", {}) or {}
    faiss_cfg = (manifest.get("retrievers", {}) or {}).get("faiss", {}) or {}
    bm25_ready = bool(bm25_cfg.get("built", False))
    dense_ready = bool(faiss_cfg.get("built", False))

    complete_embeddings, embedding_total, embedding_found = dense_shards_complete(dense_dir)
    if complete_embeddings:
        write_manifest(
            args,
            corpus_dir,
            count,
            bm25_dir,
            dense_dir,
            embeddings_built=True,
        )
    elif embedding_found:
        print(
            f"[dense] found partial shard set: {embedding_found}/{embedding_total}; "
            "--resume can fill the missing shards"
        )

    if "bm25" in build_targets:
        if args.resume and bm25_ready:
            print("[bm25] resume: manifest marks BM25 complete -> skip")
        else:
            # A failed Lucene build can leave a non-empty but unusable directory.
            shutil.rmtree(bm25_dir, ignore_errors=True)
            build_bm25(args, corpus_dir, bm25_dir)
            bm25_ready = True
            write_manifest(
                args,
                corpus_dir,
                count,
                bm25_dir,
                dense_dir,
                bm25_built=True,
            )

    if "dense" in build_targets:
        build_dense(args, corpus_dir, dense_dir)
        complete_embeddings, embedding_total, embedding_found = dense_shards_complete(dense_dir)
        if not complete_embeddings:
            raise RuntimeError(
                f"Dense stage incomplete after encoding: {embedding_found}/{embedding_total} shards"
            )
        write_manifest(
            args,
            corpus_dir,
            count,
            bm25_dir,
            dense_dir,
            embeddings_built=True,
        )

        if args.dense_index_type == "flat":
            dense_ready = True
            write_manifest(
                args,
                corpus_dir,
                count,
                bm25_dir,
                dense_dir,
                dense_built=True,
                embeddings_built=True,
                dense_index_type="flat",
            )

    build_ann = "ann" in build_targets or (
        "dense" in build_targets and args.dense_index_type == "ivf-flat"
    )
    if build_ann:
        complete_embeddings, embedding_total, embedding_found = dense_shards_complete(dense_dir)
        if not complete_embeddings:
            raise RuntimeError(
                "Cannot build ANN before dense embeddings are complete: "
                f"{embedding_found}/{embedding_total} shards found in {dense_dir}"
            )

        current_faiss = (
            (load_existing_manifest(corpus_dir) or {}).get("retrievers", {}) or {}
        ).get("faiss", {}) or {}
        ann_files_ready = (
            current_faiss.get("built", False)
            and current_faiss.get("index_type") == "ivf-flat"
            and (dense_dir / "index.ivf-flat.faiss").is_file()
            and (dense_dir / "index.lookup.pkl").is_file()
        )
        if args.resume and ann_files_ready:
            print("[ann] resume: manifest and IVF artifacts are complete -> skip")
            dense_ready = True
        else:
            ann_meta = build_ivf_flat_index(args, dense_dir, expected_total=count)
            dense_ready = True
            write_manifest(
                args,
                corpus_dir,
                count,
                bm25_dir,
                dense_dir,
                dense_built=True,
                embeddings_built=True,
                ann_meta=ann_meta,
            )

    print("\nBuild complete.")
    final_manifest = load_existing_manifest(corpus_dir) or {}
    final_retrievers = final_manifest.get("retrievers", {}) or {}
    if (final_retrievers.get("bm25", {}) or {}).get("built", False):
        print(f"BM25:  python searcher/mcp_server.py --searcher-type bm25 --corpus {args.name} --get-document --port <PORT>")
    if (final_retrievers.get("faiss", {}) or {}).get("built", False):
        print(f"Dense: python searcher/mcp_server.py --searcher-type faiss --corpus {args.name} --get-document --port <PORT>")


if __name__ == "__main__":
    main()
