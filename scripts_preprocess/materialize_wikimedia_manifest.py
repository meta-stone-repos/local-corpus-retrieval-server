#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from materialize_wikimedia_enterprise_tar import (
    create_database,
    export_jsonl,
    filename_metadata,
    flush_batch,
    normalize_record,
)


LOGGER = logging.getLogger("wikimedia-manifest-cleaner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild cleaned artifacts but preserve input_manifest.json",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("YAML root must be a mapping")

    if config.get("config_version") != 1:
        raise ValueError("Only config_version: 1 is supported")

    return config


def open_work_database(path: Path) -> sqlite3.Connection:
    if path.exists():
        conn = sqlite3.connect(path)
    else:
        conn = create_database(path)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_archives (
            logical_name TEXT PRIMARY KEY,
            size INTEGER,
            records_seen INTEGER NOT NULL,
            normalized_records INTEGER NOT NULL,
            identity_missing INTEGER NOT NULL,
            record_errors INTEGER NOT NULL,
            completed_at TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cleaning_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    conn.commit()
    return conn


def state_hash(
    manifest: dict[str, Any],
    cleaning: dict[str, Any],
) -> str:
    relevant = {
        "selection_hash": manifest.get("selection_hash"),
        "min_chars": cleaning.get("min_chars", 1),
        "on_error": cleaning.get("on_error", "fail"),
    }

    payload = json.dumps(
        relevant,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def check_state(
    conn: sqlite3.Connection,
    expected_hash: str,
) -> None:
    row = conn.execute(
        """
        SELECT value
        FROM cleaning_state
        WHERE key = 'state_hash'
        """
    ).fetchone()

    if row is None:
        conn.execute(
            """
            INSERT INTO cleaning_state(key, value)
            VALUES ('state_hash', ?)
            """,
            (expected_hash,),
        )
        conn.commit()
        return

    if row[0] != expected_hash:
        raise ValueError(
            "Cleaning settings or input selection changed. "
            "Use --overwrite to rebuild."
        )


def archive_completed(
    conn: sqlite3.Connection,
    logical_name: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM processed_archives
        WHERE logical_name = ?
        """,
        (logical_name,),
    ).fetchone()

    return row is not None


def mark_archive_completed(
    conn: sqlite3.Connection,
    *,
    logical_name: str,
    size: int,
    stats: dict[str, int],
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO processed_archives(
            logical_name,
            size,
            records_seen,
            normalized_records,
            identity_missing,
            record_errors,
            completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            logical_name,
            size,
            stats["records_seen"],
            stats["normalized_records"],
            stats["identity_missing"],
            stats["record_errors"],
            datetime.now(timezone.utc).isoformat(),
        ),
    )

    conn.commit()


def process_archive(
    conn: sqlite3.Connection,
    archive: dict[str, Any],
    *,
    batch_size: int,
    progress_every: int,
    on_error: str,
) -> dict[str, int]:
    logical_name = str(archive["logical_name"])
    local_path_value = archive.get("local_path")

    if not local_path_value:
        raise ValueError(
            f"Archive has no local_path: {logical_name}"
        )

    path = Path(local_path_value)

    if not path.is_file():
        raise FileNotFoundError(path)

    expected_size = archive.get("size")
    actual_size = path.stat().st_size

    if (
        expected_size is not None
        and actual_size != int(expected_size)
    ):
        raise ValueError(
            f"Archive size mismatch: {path}; "
            f"expected={expected_size}, actual={actual_size}"
        )

    filename_wiki, filename_namespace, dump_date = (
        filename_metadata(path)
    )

    stats = {
        "records_seen": 0,
        "normalized_records": 0,
        "identity_missing": 0,
        "record_errors": 0,
    }

    batch: list[tuple[Any, ...]] = []

    LOGGER.info("Cleaning %s", logical_name)

    with tarfile.open(path, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue

            if not member.name.lower().endswith(
                (".ndjson", ".jsonl", ".json")
            ):
                continue

            extracted = tar.extractfile(member)

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

                    if not isinstance(record, dict):
                        raise ValueError(
                            "record is not a JSON object"
                        )

                    normalized = normalize_record(
                        record,
                        filename_wiki=filename_wiki,
                        filename_namespace=filename_namespace,
                        dump_date=dump_date,
                    )

                except Exception as exc:
                    stats["record_errors"] += 1

                    if on_error == "fail":
                        raise RuntimeError(
                            f"Bad record: "
                            f"{logical_name}/"
                            f"{member.name}:"
                            f"{line_number}"
                        ) from exc

                    if stats["record_errors"] <= 10:
                        LOGGER.exception(
                            "Skipping bad record %s/%s:%d",
                            logical_name,
                            member.name,
                            line_number,
                        )

                    continue

                if normalized is None:
                    stats["identity_missing"] += 1
                    continue

                stats["normalized_records"] += 1
                batch.append(normalized)

                if len(batch) >= batch_size:
                    flush_batch(conn, batch)

                if (
                    stats["records_seen"]
                    % progress_every
                    == 0
                ):
                    LOGGER.info(
                        "%s records_seen=%d",
                        logical_name,
                        stats["records_seen"],
                    )

    flush_batch(conn, batch)

    mark_archive_completed(
        conn,
        logical_name=logical_name,
        size=actual_size,
        stats=stats,
    )

    LOGGER.info(
        "Completed archive=%s records=%d normalized=%d errors=%d",
        logical_name,
        stats["records_seen"],
        stats["normalized_records"],
        stats["record_errors"],
    )

    return stats


def remove_artifact(path: Path) -> None:
    if path.is_file():
        path.unlink()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)

    corpus = config.get("corpus") or {}
    cleaning = config.get("cleaning") or {}

    corpus_name = str(
        corpus.get("name") or ""
    ).strip()

    if not corpus_name:
        raise ValueError("corpus.name is required")

    corpus_dir = (
        project_root
        / "corpora"
        / corpus_name
    )

    input_manifest_path = (
        corpus_dir
        / "input_manifest.json"
    )

    if not input_manifest_path.is_file():
        raise FileNotFoundError(
            f"Input manifest not found: "
            f"{input_manifest_path}"
        )

    manifest = json.loads(
        input_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if manifest.get("corpus") != corpus_name:
        raise ValueError(
            "input_manifest corpus name mismatch"
        )

    archives = manifest.get("archives")

    if not isinstance(archives, list) or not archives:
        raise ValueError(
            "input_manifest has no archives"
        )

    final_db = corpus_dir / "documents.sqlite"
    work_db = corpus_dir / "documents.sqlite.inprogress"
    final_jsonl = corpus_dir / "corpus.jsonl"
    temp_jsonl = corpus_dir / "corpus.jsonl.tmp"
    final_report = corpus_dir / "cleaning_report.json"
    temp_report = corpus_dir / "cleaning_report.json.tmp"

    if args.overwrite:
        for path in (
            final_db,
            work_db,
            final_jsonl,
            temp_jsonl,
            final_report,
            temp_report,
        ):
            remove_artifact(path)

    if (
        final_db.is_file()
        and final_jsonl.is_file()
        and final_report.is_file()
    ):
        LOGGER.info(
            "Corpus cleaning already complete: %s",
            corpus_dir,
        )
        return

    if final_db.exists():
        raise RuntimeError(
            "Found documents.sqlite without complete outputs. "
            "Use --overwrite to rebuild."
        )

    batch_size = int(
        cleaning.get("batch_size", 1000)
    )
    min_chars = int(
        cleaning.get("min_chars", 1)
    )
    progress_every = int(
        cleaning.get("progress_every", 10000)
    )
    on_error = str(
        cleaning.get("on_error") or "fail"
    )

    if batch_size < 1:
        raise ValueError(
            "cleaning.batch_size must be positive"
        )

    if min_chars < 0:
        raise ValueError(
            "cleaning.min_chars must be non-negative"
        )

    if progress_every < 1:
        raise ValueError(
            "cleaning.progress_every must be positive"
        )

    if on_error not in {"fail", "skip"}:
        raise ValueError(
            "cleaning.on_error must be fail or skip"
        )

    conn = open_work_database(work_db)

    try:
        check_state(
            conn,
            state_hash(manifest, cleaning),
        )

        for index, archive in enumerate(
            archives,
            start=1,
        ):
            logical_name = str(
                archive["logical_name"]
            )

            if archive_completed(
                conn,
                logical_name,
            ):
                LOGGER.info(
                    "Skipping completed archive "
                    "%d/%d %s",
                    index,
                    len(archives),
                    logical_name,
                )
                continue

            LOGGER.info(
                "Processing archive %d/%d",
                index,
                len(archives),
            )

            process_archive(
                conn,
                archive,
                batch_size=batch_size,
                progress_every=progress_every,
                on_error=on_error,
            )

        completed_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM processed_archives
            """
        ).fetchone()[0]

        if completed_count != len(archives):
            raise RuntimeError(
                f"Only {completed_count}/{len(archives)} "
                f"archives completed"
            )

        unique_before_prune = int(
            conn.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
        )

        conn.execute(
            """
            DELETE FROM documents
            WHERE char_count < ?
            """,
            (min_chars,),
        )
        conn.commit()

        document_count = export_jsonl(
            conn,
            temp_jsonl,
        )

        if document_count == 0:
            raise ValueError(
                "No documents remained after cleaning"
            )

        archive_rows = conn.execute(
            """
            SELECT
                logical_name,
                size,
                records_seen,
                normalized_records,
                identity_missing,
                record_errors,
                completed_at
            FROM processed_archives
            ORDER BY logical_name
            """
        ).fetchall()

        per_archive = [
            {
                "logical_name": row[0],
                "size": row[1],
                "records_seen": row[2],
                "normalized_records": row[3],
                "identity_missing": row[4],
                "record_errors": row[5],
                "completed_at": row[6],
            }
            for row in archive_rows
        ]

        total_records = sum(
            item["records_seen"]
            for item in per_archive
        )

        total_normalized = sum(
            item["normalized_records"]
            for item in per_archive
        )

        total_missing = sum(
            item["identity_missing"]
            for item in per_archive
        )

        total_errors = sum(
            item["record_errors"]
            for item in per_archive
        )

        report = {
            "manifest_path": str(
                input_manifest_path
            ),
            "archive_count": len(archives),
            "completed_archives": completed_count,
            "records_seen": total_records,
            "normalized_records": total_normalized,
            "identity_missing": total_missing,
            "record_errors": total_errors,
            "unique_before_prune": (
                unique_before_prune
            ),
            "duplicates": max(
                0,
                total_normalized
                - unique_before_prune,
            ),
            "pruned_by_min_chars": (
                unique_before_prune
                - document_count
            ),
            "document_count": document_count,
            "min_chars": min_chars,
            "per_archive": per_archive,
        }

        temp_report.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    except Exception:
        LOGGER.error(
            "Cleaning interrupted. Resume database retained: %s",
            work_db,
        )
        raise
    finally:
        conn.close()

    # documents.sqlite 最后发布；它存在即代表整个清洗阶段完成。
    os.replace(temp_jsonl, final_jsonl)
    os.replace(temp_report, final_report)
    os.replace(work_db, final_db)

    LOGGER.info(
        "Completed corpus=%s archives=%d documents=%d",
        corpus_name,
        len(archives),
        document_count,
    )


if __name__ == "__main__":
    main()
