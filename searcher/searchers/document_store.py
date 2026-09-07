"""Local document stores used by dense retrieval."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional


class SQLiteDocumentStore:
    """Read-only docid -> text lookup without loading an entire corpus into RAM."""

    def __init__(self, path: str):
        self.path = str(Path(path).resolve())
        if not Path(self.path).is_file():
            raise FileNotFoundError(f"SQLite document store not found: {self.path}")

        uri = f"file:{self.path}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._lock = threading.Lock()

    def get(self, docid: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT text FROM documents WHERE docid = ?", (str(docid),)
            ).fetchone()
        return row[0] if row else None

    def get_metadata(self, docid: str) -> Optional[dict]:
        """Return lightweight metadata without reading document text."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT url, char_count, title, source, language
                FROM documents
                WHERE docid = ?
                """,
                (str(docid),),
            ).fetchone()

        if row is None:
            return None

        url, char_count, title, source, language = row

        return {
            "url": url,
            "char_count": (
                int(char_count)
                if char_count is not None
                else None
            ),
            "title": title,
            "source": source,
            "language": language,
        }

    def get_metadata_batch(self, docids: list[str]) -> dict[str, dict]:
        """Return lightweight metadata for multiple docids in one query."""
        if not docids:
            return {}

        # Preserve only unique normalized docids.
        normalized_docids = list(dict.fromkeys(str(docid) for docid in docids))
        placeholders = ",".join("?" for _ in normalized_docids)

        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT docid, url, char_count, title, source, language
                FROM documents
                WHERE docid IN ({placeholders})
                """,
                normalized_docids,
            ).fetchall()

        results = {}

        for row in rows:
            docid, url, char_count, title, source, language = row

            results[str(docid)] = {
                "url": url,
                "char_count": (
                    int(char_count)
                    if char_count is not None
                    else None
                ),
                "title": title,
                "source": source,
                "language": language,
            }

        return results

    def get_document(self, docid: str) -> Optional[dict]:
        """Return full document content and lightweight metadata by docid."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT text, url, char_count, title, source, language
                FROM documents
                WHERE docid = ?
                """,
                (str(docid),),
            ).fetchone()

        if row is None:
            return None

        text, url, char_count, title, source, language = row

        # Backward-compatible fallback for stores without a populated
        # char_count value.
        if char_count is None:
            char_count = len(text)

        return {
            "docid": str(docid),
            "text": text,
            "url": url,
            "metadata": {
                "char_count": int(char_count),
                "title": title,
                "source": source,
                "language": language,
            },
        }

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return int(row[0])

    def close(self) -> None:
        self._conn.close()
