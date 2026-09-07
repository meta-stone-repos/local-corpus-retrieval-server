#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def add_scalar(
    command: list[str],
    config: Mapping[str, Any],
    key: str,
    option: str,
) -> None:
    value = config.get(key)
    if value is not None:
        command.extend([option, str(value)])


def run_stage(
    name: str,
    command: list[str],
    project_root: Path,
    dry_run: bool,
) -> None:
    print(f"\n[{name}]")
    print(shlex.join(command), flush=True)

    if not dry_run:
        subprocess.run(
            command,
            cwd=project_root,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load, clean and index one corpus from a YAML configuration"
        )
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )
    config = require_mapping(config, "config")

    corpus = require_mapping(config.get("corpus"), "corpus")
    build = require_mapping(config.get("build"), "build")
    runtime = require_mapping(
        config.get("runtime", {}),
        "runtime",
    )

    corpus_name = str(corpus.get("name", "")).strip()
    if not corpus_name:
        raise ValueError("corpus.name is required")

    raw_targets = build.get("targets", ["bm25", "dense"])
    if isinstance(raw_targets, str):
        targets = raw_targets
    elif isinstance(raw_targets, list):
        targets = ",".join(str(item) for item in raw_targets)
    else:
        raise ValueError("build.targets must be a string or list")

    dry_run = bool(runtime.get("dry_run", False))

    prepare_command = [
        sys.executable,
        str(
            project_root
            / "scripts_preprocess"
            / "prepare_wikimedia_inputs.py"
        ),
        "--config",
        str(config_path),
    ]

    clean_command = [
        sys.executable,
        str(
            project_root
            / "scripts_preprocess"
            / "materialize_wikimedia_manifest.py"
        ),
        "--config",
        str(config_path),
    ]

    build_command = [
        sys.executable,
        str(
            project_root
            / "scripts_build_index"
            / "build_corpus.py"
        ),
        "--name",
        corpus_name,
        "--build",
        targets,
        "--resume",
    ]

    if bool(runtime.get("keep_build_files", False)):
        build_command.append("--keep-build-files")

    bm25 = require_mapping(build.get("bm25", {}), "build.bm25")
    add_scalar(
        build_command,
        bm25,
        "threads",
        "--bm25-threads",
    )

    dense = require_mapping(
        build.get("dense", {}),
        "build.dense",
    )

    dense_options = {
        "model": "--dense-model",
        "pooling": "--pooling",
        "passage_max_len": "--passage-max-len",
        "query_max_len": "--query-max-len",
        "batch_size": "--batch-size",
        "dtype": "--dtype",
        "gpus": "--gpus",
        "shards": "--dense-shards",
        "index_type": "--dense-index-type",
        "query_prefix": "--query-prefix",
    }

    for key, option in dense_options.items():
        add_scalar(build_command, dense, key, option)

    if "normalize" in dense:
        build_command.append(
            "--normalize"
            if bool(dense["normalize"])
            else "--no-normalize"
        )

    ann = require_mapping(build.get("ann", {}), "build.ann")

    ann_options = {
        "nlist": "--ivf-nlist",
        "nprobe": "--ivf-nprobe",
        "train_size": "--ivf-train-size",
        "seed": "--ivf-seed",
        "threads": "--ivf-threads",
        "niter": "--ivf-niter",
        "backend": "--ivf-backend",
        "gpus": "--ivf-gpus",
        "gpu_batch_size": "--ivf-gpu-batch-size",
        "gpu_dtype": "--ivf-gpu-dtype",
    }

    for key, option in ann_options.items():
        add_scalar(build_command, ann, key, option)

    run_stage(
        "1/3 prepare inputs",
        prepare_command,
        project_root,
        dry_run,
    )
    run_stage(
        "2/3 clean corpus",
        clean_command,
        project_root,
        dry_run,
    )
    run_stage(
        "3/3 build indexes",
        build_command,
        project_root,
        dry_run,
    )

    print(f"\nPipeline completed: {corpus_name}")


if __name__ == "__main__":
    main()
