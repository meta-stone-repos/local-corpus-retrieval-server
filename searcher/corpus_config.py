"""Corpus manifest loading for local, independently indexed corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPORA_ROOT = PROJECT_ROOT / "corpora"


class CorpusConfigError(ValueError):
    """Raised when a corpus manifest is missing or malformed."""


def _resolve_project_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def load_corpus_manifest(name: str) -> Dict[str, Any]:
    manifest_path = CORPORA_ROOT / name / "manifest.json"
    if not manifest_path.is_file():
        raise CorpusConfigError(
            f"Corpus '{name}' has no manifest: {manifest_path}. "
            "Build it first with scripts_build_index/build_corpus.py."
        )

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("name") != name:
        raise CorpusConfigError(
            f"Manifest name mismatch: requested '{name}', found '{manifest.get('name')}'."
        )
    if "retrievers" not in manifest:
        raise CorpusConfigError(f"Manifest has no 'retrievers' section: {manifest_path}")

    manifest["_manifest_path"] = str(manifest_path)
    return manifest


def apply_corpus_manifest(args):
    """Populate searcher-specific CLI args from ``corpora/<name>/manifest.json``.

    When --corpus is used, the manifest is intentionally authoritative for
    retrieval-model/index settings. High-level server options such as --port,
    --k and --get-document remain normal CLI arguments.
    """

    if not getattr(args, "corpus", None):
        return args

    manifest = load_corpus_manifest(args.corpus)
    retrievers = manifest["retrievers"]
    config = retrievers.get(args.searcher_type)
    if not config or not config.get("built", False):
        raise CorpusConfigError(
            f"Corpus '{args.corpus}' does not have a built '{args.searcher_type}' retriever."
        )

    if "index_path" in config:
        args.index_path = _resolve_project_path(config["index_path"])

    if args.searcher_type in {"faiss", "reasonir"}:
        mapping = {
            "model_name": "model_name",
            "normalize": "normalize",
            "pooling": "pooling",
            "torch_dtype": "torch_dtype",
            "task_prefix": "task_prefix",
            "max_length": "max_length",
            "index_type": "index_type",
            "nprobe": "nprobe",
        }
        for manifest_key, arg_name in mapping.items():
            if manifest_key in config:
                setattr(args, arg_name, config[manifest_key])

        lookup_path = config.get("lookup_path")
        if lookup_path:
            args.lookup_path = _resolve_project_path(lookup_path)

        document_store = config.get("document_store") or manifest.get("corpus", {}).get(
            "document_store"
        )
        if document_store:
            args.document_store = _resolve_project_path(document_store)

        corpus_path = manifest.get("corpus", {}).get("path")
        if corpus_path:
            args.corpus_path = _resolve_project_path(corpus_path)

    print(
        f"[CORPUS] Loaded '{args.corpus}' ({args.searcher_type}) from "
        f"{manifest['_manifest_path']}"
    )
    return args
