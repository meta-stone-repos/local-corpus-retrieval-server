import argparse
import math
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset


CHUNK_SIZES = [4096, 8192, 16384]

HIST_BINS = [
    (0, 2_000, "0–2k"),
    (2_000, 4_000, "2k–4k"),
    (4_000, 8_000, "4k–8k"),
    (8_000, 16_000, "8k–16k"),
    (16_000, 32_000, "16k–32k"),
    (32_000, 64_000, "32k–64k"),
    (64_000, 128_000, "64k–128k"),
    (128_000, None, ">128k"),
]


def pct(x):
    return round(float(x) * 100.0, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="Tevatron/browsecomp-plus-corpus",
    )
    parser.add_argument(
        "--split",
        default="train",
    )
    parser.add_argument(
        "--text-field",
        default="text",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
    )
    parser.add_argument(
        "--output",
        default="outputs/browsecomp_char_stats.json",
    )
    args = parser.parse_args()

    print(f"[load] dataset={args.dataset}")
    ds = load_dataset(
        args.dataset,
        split=args.split,
        cache_dir=args.cache_dir,
    )

    print(f"[load] documents={len(ds):,}")
    print(f"[load] columns={ds.column_names}")

    if args.text_field not in ds.column_names:
        raise ValueError(
            f"text field '{args.text_field}' not found; "
            f"available columns: {ds.column_names}"
        )

    lengths = np.empty(len(ds), dtype=np.int64)

    print("[stats] scanning character lengths...")

    for i, row in enumerate(ds):
        text = row[args.text_field]

        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)

        lengths[i] = len(text)

        if (i + 1) % 10_000 == 0:
            print(f"[stats] scanned {i + 1:,}/{len(ds):,}")

    n = len(lengths)

    result = {
        "dataset": args.dataset,
        "split": args.split,
        "text_field": args.text_field,
        "document_count": int(n),
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

        stats = {
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

        result["chunk_stats"][str(chunk_size)] = stats

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

    print()
    print("=" * 72)
    print("CHAR LENGTH")
    print("=" * 72)

    c = result["char_length"]

    print(f"documents : {n:,}")
    print(f"min       : {c['min']:,.0f}")
    print(f"mean      : {c['mean']:,.2f}")
    print(f"p50       : {c['p50']:,.0f}")
    print(f"p75       : {c['p75']:,.0f}")
    print(f"p90       : {c['p90']:,.0f}")
    print(f"p95       : {c['p95']:,.0f}")
    print(f"p99       : {c['p99']:,.0f}")
    print(f"max       : {c['max']:,.0f}")

    for chunk_size in CHUNK_SIZES:
        s = result["chunk_stats"][str(chunk_size)]

        print()
        print("=" * 72)
        print(f"CHUNK SIZE = {chunk_size:,} chars")
        print("=" * 72)

        print(
            f"chunk_count = 1  : "
            f"{pct(s['chunk_count_eq_1_ratio']):8.3f}%"
        )
        print(
            f"chunk_count <= 2 : "
            f"{pct(s['chunk_count_le_2_ratio']):8.3f}%"
        )
        print(
            f"chunk_count <= 4 : "
            f"{pct(s['chunk_count_le_4_ratio']):8.3f}%"
        )
        print(
            f"chunk_count <= 8 : "
            f"{pct(s['chunk_count_le_8_ratio']):8.3f}%"
        )
        print(
            f"chunk_count > 8  : "
            f"{pct(s['chunk_count_gt_8_ratio']):8.3f}%"
        )
        print(
            f"chunk_count > 16 : "
            f"{pct(s['chunk_count_gt_16_ratio']):8.3f}%"
        )
        print(
            f"max chunk_count   : "
            f"{s['max_chunk_count']:,}"
        )
        print(
            f"mean chunk_count  : "
            f"{s['mean_chunk_count']:.3f}"
        )

    print()
    print("=" * 72)
    print("CHAR LENGTH HISTOGRAM")
    print("=" * 72)

    for label, x in result["histogram"].items():
        print(
            f"{label:12s} "
            f"{x['count']:8,d} "
            f"{pct(x['ratio']):8.3f}%"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )

    print()
    print(f"[done] JSON written to {output}")


if __name__ == "__main__":
    main()
