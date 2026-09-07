#!/usr/bin/env python3
"""Materialize BrowseComp-Plus into a model-independent local corpus.

Canonical per-document JSONL schema::

    {
      "docid": "...",
      "text": "...",
      "url": "...",
      "metadata": {"char_count": 12345}
    }

``char_count`` is exactly Python ``len(text)`` (Unicode code points).  This
script deliberately does not tokenize documents and does not pre-compute any
serving/chunk fields.

A SQLite document store is written alongside the JSONL.  SQLite stores
``char_count`` as an INTEGER column so serving can read it without reparsing a
metadata JSON blob.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from datasets import load_dataset


DEFAULT_DATASET = "Tevatron/browsecomp-plus-corpus"
DEFAULT_OUTPUT_DIR = "corpora/browsecomp"
DEFAULT_STATS_OUTPUT = "outputs/browsecomp_char_stats.json"
CHUNK_SIZES = (4096, 8192, 16384)
HIST_BINS = (
    (0, 2_000, "0-2k"),
    (2_000, 4_000, "2k-4k"),
    (4_000, 8_000, "4k-8k"),
    (8_000, 16_000, "8k-16k"),
    (16_000, 32_000, "16k-32k"),
    (32_000, 64_000, "32k-64k"),
    (64_000, 128_000, "64k-128k"),
    (128_000, None, ">128k"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize BrowseComp-Plus corpus metadata without tokenization."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stats-output", default=DEFAULT_STATS_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing materialized BrowseComp corpus.",
    )
    return parser.parse_args()


def _to_text(value) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _canonical_row(row: Mapping[str, object]) -> dict:
    missing = [key for key in ("docid", "text", "url") if key not in row]
    if missing:
        raise KeyError(f"BrowseComp corpus row is missing required fields: {missing}")

    text = _to_text(row["text"])
    out = {
        "docid": _to_text(row["docid"]),
        "text": text,
        "url": _to_text(row["url"]),
        "metadata": {
            "char_count": len(text),
        },
    }

    # Keep optional model-independent source metadata only when it is actually
    # present in the source corpus.  Do not synthesize any of these fields.
    for key in ("title", "source", "language"):
        if key in row and row[key] is not None:
            out[key] = _to_text(row[key])

    return out


def _create_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        """
        CREATE TABLE documents (
            docid TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            url TEXT NOT NULL,
            char_count INTEGER NOT NULL,
            title TEXT,
            source TEXT,
            language TEXT
        )
        """
    )
    # Schema version 2 = canonical corpus metadata with char_count.
    conn.execute("PRAGMA user_version=2")
    return conn


def _stats(lengths: np.ndarray) -> dict:
    n = int(lengths.size)
    if n == 0:
        raise ValueError("Cannot compute statistics for an empty corpus")

    result = {
        "document_count": n,
        "char_length": {
            "min": int(lengths.min()),
            "mean": float(lengths.mean()),
            "p50": float(np.percentile(lengths, 50)),
            "p75": float(np.percentile(lengths, 75)),
            "p90": float(np.percentile(lengths, 90)),
            "p95": float(np.percentile(lengths, 95)),
            "p99": float(np.percentile(lengths, 99)),
            "max": int(lengths.max()),
        },
        "chunk_stats": {},
        "histogram": {},
    }

    for chunk_size in CHUNK_SIZES:
        counts = (lengths + chunk_size - 1) // chunk_size
        result["chunk_stats"][str(chunk_size)] = {
            "chunk_size_chars": chunk_size,
            "chunk_count_eq_1_ratio": float(np.mean(counts == 1)),
            "chunk_count_le_2_ratio": float(np.mean(counts <= 2)),
            "chunk_count_le_4_ratio": float(np.mean(counts <= 4)),
            "chunk_count_le_8_ratio": float(np.mean(counts <= 8)),
            "chunk_count_gt_8_ratio": float(np.mean(counts > 8)),
            "chunk_count_gt_16_ratio": float(np.mean(counts > 16)),
            "max_chunk_count": int(counts.max()),
            "mean_chunk_count": float(counts.mean()),
        }

    for low, high, label in HIST_BINS:
        if high is None:
            mask = lengths >= low
        else:
            mask = (lengths >= low) & (lengths < high)
        count = int(mask.sum())
        result["histogram"][label] = {
            "count": count,
            "ratio": float(count / n),
        }

    return result


def _print_stats(stats: Mapping[str, object]) -> None:
    c = stats["char_length"]
    print("\n===== CHAR LENGTH =====")
    print(f"documents : {stats['document_count']:,}")
    print(f"min       : {c['min']:,}")
    print(f"mean      : {c['mean']:,.2f}")
    for key in ("p50", "p75", "p90", "p95", "p99"):
        print(f"{key:9s}: {c[key]:,.0f}")
    print(f"max       : {c['max']:,}")

    for chunk_size in CHUNK_SIZES:
        s = stats["chunk_stats"][str(chunk_size)]
        print(f"\n===== CHUNK SIZE = {chunk_size:,} chars =====")
        for key, label in (
            ("chunk_count_eq_1_ratio", "chunk_count = 1 "),
            ("chunk_count_le_2_ratio", "chunk_count <= 2"),
            ("chunk_count_le_4_ratio", "chunk_count <= 4"),
            ("chunk_count_le_8_ratio", "chunk_count <= 8"),
            ("chunk_count_gt_8_ratio", "chunk_count > 8 "),
            ("chunk_count_gt_16_ratio", "chunk_count > 16"),
        ):
            print(f"{label}: {100.0 * s[key]:8.3f}%")
        print(f"max chunk_count : {s['max_chunk_count']:,}")
        print(f"mean chunk_count: {s['mean_chunk_count']:.3f}")

    print("\n===== HISTOGRAM =====")
    for label, item in stats["histogram"].items():
        print(f"{label:10s} {item['count']:8,d} {100.0 * item['ratio']:8.3f}%")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    corpus_path = output_dir / "corpus.jsonl"
    sqlite_path = output_dir / "documents.sqlite"
    schema_path = output_dir / "corpus_schema.json"

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite only if you intentionally want to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] dataset={args.dataset} split={args.split}")
    ds = load_dataset(
        args.dataset,
        split=args.split,
        cache_dir=args.cache_dir,
    )
    print(f"[load] rows={len(ds):,}")
    print(f"[load] columns={ds.column_names}")

    required = {"docid", "text", "url"}
    missing_columns = required.difference(ds.column_names)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required BrowseComp columns: {sorted(missing_columns)}"
        )

    lengths = np.empty(len(ds), dtype=np.int64)
    conn = _create_sqlite(sqlite_path)
    batch = []

    try:
        with corpus_path.open("w", encoding="utf-8") as out:
            for i, source_row in enumerate(ds):
                row = _canonical_row(source_row)
                char_count = int(row["metadata"]["char_count"])
                lengths[i] = char_count

                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                batch.append(
                    (
                        row["docid"],
                        row["text"],
                        row["url"],
                        char_count,
                        row.get("title"),
                        row.get("source"),
                        row.get("language"),
                    )
                )

                if len(batch) >= args.batch_size:
                    conn.executemany(
                        """
                        INSERT INTO documents(
                            docid, text, url, char_count, title, source, language
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
                    conn.commit()
                    batch.clear()
                    print(f"[materialize] {i + 1:,}/{len(ds):,}", flush=True)

            if batch:
                conn.executemany(
                    """
                    INSERT INTO documents(
                        docid, text, url, char_count, title, source, language
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
    finally:
        conn.close()

    schema = {
        "schema_version": 2,
        "document_schema": {
            "required": ["docid", "text", "url", "metadata.char_count"],
            "optional": ["title", "source", "language"],
            "char_count_definition": "len(text)",
            "tokenization": False,
            "precomputed_chunking": False,
        },
        "files": {
            "corpus": str(corpus_path),
            "document_store": str(sqlite_path),
        },
    }
    schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")

    stats = _stats(lengths)
    stats.update(
        {
            "dataset": args.dataset,
            "split": args.split,
            "char_count_definition": "len(text)",
        }
    )
    stats_path = Path(args.stats_output).resolve()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")

    print(f"\n[done] corpus: {corpus_path}")
    print(f"[done] sqlite: {sqlite_path}")
    print(f"[done] schema: {schema_path}")
    print(f"[done] stats:  {stats_path}")
    _print_stats(stats)


if __name__ == "__main__":
    main()
