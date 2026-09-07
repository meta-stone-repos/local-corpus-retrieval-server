#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sqlite3
import tarfile
import time
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup


LOGGER = logging.getLogger("wikimedia-cleaner")

ARCHIVE_RE = re.compile(
    r"^(?P<wiki>.+)-NS(?P<namespace>\d+)-(?P<date>\d{8})-"
    r"ENTERPRISE-HTML\.json\.tar\.gz$"
)

UPSERT_SQL = """
INSERT INTO documents(
    docid,
    text,
    url,
    char_count,
    title,
    source,
    language,
    metadata_json,
    record_rank,
    dump_date
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(docid) DO UPDATE SET
    text = excluded.text,
    url = excluded.url,
    char_count = excluded.char_count,
    title = excluded.title,
    source = excluded.source,
    language = excluded.language,
    metadata_json = excluded.metadata_json,
    record_rank = excluded.record_rank,
    dump_date = excluded.dump_date
WHERE excluded.record_rank >= documents.record_rank
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tar", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument(
        "--on-error",
        choices=["fail", "skip"],
        default="fail",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def nested_identifier(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None

    identifier = value.get("identifier")
    if identifier is None:
        return None

    text = str(identifier).strip()
    return text or None


def timestamp_rank(value: Any) -> float:
    if not value:
        return 0.0

    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError, OverflowError):
        return 0.0


def clean_html(html: str) -> str:
    if not html.strip():
        return ""

    soup = BeautifulSoup(html, "html.parser")

    for node in soup(["script", "style", "noscript", "template"]):
        node.decompose()

    root = soup.body or soup

    sections = [
        section
        for section in root.find_all("section")
        if section.find_parent("section") is None
    ]

    nodes = sections or [root]
    parts: list[str] = []

    for node in nodes:
        text = node.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            parts.append(text)

    return "\n\n".join(parts)


def fallback_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, Mapping):
        for key in ("text", "wikitext", "html"):
            candidate = value.get(key)

            if not isinstance(candidate, str) or not candidate.strip():
                continue

            if key == "html":
                return clean_html(candidate)

            return candidate.strip()

    return str(value).strip()


def license_ids(record: Mapping[str, Any]) -> list[str]:
    result: list[str] = []

    for item in record.get("license") or []:
        if not isinstance(item, Mapping):
            continue

        value = item.get("identifier") or item.get("name")

        if value is not None and str(value).strip():
            result.append(str(value).strip())

    return list(dict.fromkeys(result))


def filename_metadata(path: Path) -> tuple[str, int | None, str | None]:
    match = ARCHIVE_RE.fullmatch(path.name)

    if match is None:
        return "", None, None

    return (
        match.group("wiki"),
        int(match.group("namespace")),
        match.group("date"),
    )


def normalize_record(
    record: Mapping[str, Any],
    *,
    filename_wiki: str,
    filename_namespace: int | None,
    dump_date: str | None,
) -> tuple[Any, ...] | None:
    identifier_value = record.get("identifier")

    if identifier_value is None:
        return None

    identifier = str(identifier_value).strip()

    if not identifier:
        return None

    wiki = nested_identifier(record.get("is_part_of")) or filename_wiki

    if not wiki:
        return None

    docid = f"{wiki}:{identifier}"
    title = str(record.get("name") or "").strip()
    url = str(record.get("url") or "").strip()

    body_value = record.get("article_body")
    body = body_value if isinstance(body_value, Mapping) else {}

    html = body.get("html")
    text = clean_html(html) if isinstance(html, str) else ""
    text_source = "html"

    if not text:
        text = fallback_text(body.get("wikitext"))
        text_source = "wikitext"

    if not text:
        text = fallback_text(record.get("abstract"))
        text_source = "abstract"

    language = nested_identifier(record.get("in_language"))

    namespace_value = nested_identifier(record.get("namespace"))

    try:
        namespace = (
            int(namespace_value)
            if namespace_value is not None
            else filename_namespace
        )
    except ValueError:
        namespace = filename_namespace

    revision_id = nested_identifier(record.get("version"))
    qid = nested_identifier(record.get("main_entity"))

    date_modified = (
        str(record.get("date_modified") or "").strip() or None
    )

    event_value = record.get("event")
    event = event_value if isinstance(event_value, Mapping) else {}

    event_date = (
        str(event.get("date_created") or "").strip() or None
    )

    metadata = {
        "title": title or None,
        "char_count": len(text),
        "source": "wikimedia_enterprise",
        "language": language,
        "wiki": wiki,
        "namespace": namespace,
        "revision_id": revision_id,
        "qid": qid,
        "date_modified": date_modified,
        "event_date": event_date,
        "license_ids": license_ids(record),
        "text_source": text_source,
    }

    return (
        docid,
        text,
        url,
        len(text),
        title or None,
        "wikimedia_enterprise",
        language,
        json.dumps(
            metadata,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        timestamp_rank(date_modified or event_date),
        dump_date,
    )


def create_database(path: Path) -> sqlite3.Connection:
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
            language TEXT,
            metadata_json TEXT NOT NULL,
            record_rank REAL NOT NULL,
            dump_date TEXT
        )
        """
    )

    conn.execute("PRAGMA user_version=3")

    return conn


def flush_batch(
    conn: sqlite3.Connection,
    batch: list[tuple[Any, ...]],
) -> None:
    if not batch:
        return

    conn.executemany(UPSERT_SQL, batch)
    conn.commit()
    batch.clear()


def export_jsonl(
    conn: sqlite3.Connection,
    output_path: Path,
) -> int:
    count = 0

    with output_path.open("w", encoding="utf-8") as output:
        rows = conn.execute(
            """
            SELECT docid, text, url, metadata_json
            FROM documents
            ORDER BY docid
            """
        )

        for docid, text, url, metadata_json in rows:
            output.write(
                json.dumps(
                    {
                        "docid": docid,
                        "text": text,
                        "url": url,
                        "metadata": json.loads(metadata_json),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            count += 1

    return count


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    input_tar = Path(args.input_tar).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_tar.is_file():
        raise FileNotFoundError(input_tar)

    if args.min_chars < 0:
        raise ValueError("--min-chars must be non-negative")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists; use --overwrite to replace it"
            )

        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    temporary_dir = output_dir.parent / (
        f".{output_dir.name}.cleaning-{uuid.uuid4().hex}"
    )
    temporary_dir.mkdir()

    sqlite_path = temporary_dir / "documents.sqlite"
    corpus_path = temporary_dir / "corpus.jsonl"
    report_path = temporary_dir / "cleaning_report.json"

    filename_wiki, filename_namespace, dump_date = (
        filename_metadata(input_tar)
    )

    stats = {
        "input_tar": str(input_tar),
        "compressed_bytes": input_tar.stat().st_size,
        "records_seen": 0,
        "normalized_records": 0,
        "identity_missing": 0,
        "record_errors": 0,
    }

    started = time.time()
    conn = create_database(sqlite_path)
    batch: list[tuple[Any, ...]] = []

    try:
        with tarfile.open(input_tar, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue

                if not member.name.lower().endswith(
                    (".ndjson", ".jsonl", ".json")
                ):
                    continue

                extracted = archive.extractfile(member)

                if extracted is None:
                    continue

                for line_number, raw_line in enumerate(
                    extracted,
                    start=1,
                ):
                    if not raw_line.strip():
                        continue

                    stats["records_seen"] += 1

                    try:
                        record = json.loads(raw_line)

                        if not isinstance(record, Mapping):
                            raise ValueError("record is not a JSON object")

                        normalized = normalize_record(
                            record,
                            filename_wiki=filename_wiki,
                            filename_namespace=filename_namespace,
                            dump_date=dump_date,
                        )
                    except Exception:
                        stats["record_errors"] += 1

                        if args.on_error == "fail":
                            raise RuntimeError(
                                f"Bad record: {member.name}:{line_number}"
                            )

                        if stats["record_errors"] <= 10:
                            LOGGER.exception(
                                "Skipping %s:%d",
                                member.name,
                                line_number,
                            )

                        continue

                    if normalized is None:
                        stats["identity_missing"] += 1
                        continue

                    stats["normalized_records"] += 1
                    batch.append(normalized)

                    if len(batch) >= args.batch_size:
                        flush_batch(conn, batch)

                    if (
                        stats["records_seen"]
                        % args.progress_every
                        == 0
                    ):
                        LOGGER.info(
                            "records_seen=%d",
                            stats["records_seen"],
                        )

        flush_batch(conn, batch)

        unique_before_prune = conn.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]

        # Deduplication happens before pruning. Therefore, if the newest
        # revision is empty, an older non-empty revision is not retained.
        conn.execute(
            "DELETE FROM documents WHERE char_count < ?",
            (args.min_chars,),
        )
        conn.commit()

        document_count = export_jsonl(conn, corpus_path)

        if document_count == 0:
            raise ValueError(
                "No documents remained after cleaning"
            )

        stats.update(
            {
                "unique_before_prune": int(unique_before_prune),
                "duplicates": max(
                    0,
                    stats["normalized_records"]
                    - int(unique_before_prune),
                ),
                "pruned_by_min_chars": (
                    int(unique_before_prune)
                    - document_count
                ),
                "document_count": document_count,
                "dump_date": dump_date,
                "elapsed_seconds": round(
                    time.time() - started,
                    3,
                ),
            }
        )

        report_path.write_text(
            json.dumps(
                stats,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    except Exception:
        LOGGER.error(
            "Cleaning failed; partial output retained at %s",
            temporary_dir,
        )
        raise
    finally:
        conn.close()

    os.replace(temporary_dir, output_dir)

    LOGGER.info(
        "Completed: documents=%d output=%s",
        stats["document_count"],
        output_dir,
    )


if __name__ == "__main__":
    main()
