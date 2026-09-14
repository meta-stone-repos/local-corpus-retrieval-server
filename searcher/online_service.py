from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from collections import OrderedDict
from contextlib import suppress

from online_provider import Provider, valid_url
from document_search import DocumentSearch


log = logging.getLogger(__name__)
CHUNK_SIZE = 8192


class OnlineService:
    def __init__(
        self,
        provider: Provider,
        content_ttl: float = 1800,
        max_bytes: int = 256 * 1024 * 1024,
        mapping_ttl: float = 86400,
        mapping_limit: int = 100000,
        cleanup_interval: float = 60,
        snippet_max_chars: int = 2048,
        k: int = 5,
        document_search: DocumentSearch | None = None,
    ):
        for name, value in (
            ("content_ttl", content_ttl),
            ("mapping_ttl", mapping_ttl),
            ("cleanup_interval", cleanup_interval),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if max_bytes <= 0 or mapping_limit <= 0:
            raise ValueError("Cache capacities must be positive")

        if (
            type(snippet_max_chars) is not int
            or (snippet_max_chars != -1 and snippet_max_chars <= 0)
        ):
            raise ValueError("snippet_max_chars must be positive or -1")
        self.snippet_max_chars = snippet_max_chars
        if type(k) is not int or k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.provider = provider
        self.document_search = document_search or DocumentSearch()
        self.content_ttl = content_ttl
        self.max_bytes = max_bytes
        self.mapping_ttl = mapping_ttl
        self.mapping_limit = mapping_limit
        self.cleanup_interval = cleanup_interval

        self.documents = OrderedDict()
        self.contents = OrderedDict()
        self.inflight = {}
        self.cached_bytes = 0
        self.cleaner = None

    async def start(self):
        if self.cleaner is None:
            self.cleaner = asyncio.create_task(self._cleanup_loop())

    def _drop_content(self, docid):
        entry = self.contents.pop(docid, None)
        if entry is not None:
            self.cached_bytes -= entry["bytes"]

    def _drop_document(self, docid):
        self.documents.pop(docid, None)
        self._drop_content(docid)

    def _touch_document(self, docid):
        self.documents[docid]["last_access"] = time.monotonic()
        self.documents.move_to_end(docid)

    def _purge(self):
        now = time.monotonic()

        while self.contents:
            docid = next(iter(self.contents))
            entry = self.contents[docid]
            if now - entry["last_access"] < self.content_ttl:
                break
            self._drop_content(docid)

        while self.documents:
            docid = next(iter(self.documents))
            entry = self.documents[docid]
            if now - entry["last_access"] < self.mapping_ttl:
                break
            self._drop_document(docid)

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(self.cleanup_interval)
            self._purge()

    def _truncate_snippets(self, rows):
        if self.snippet_max_chars == -1:
            return rows
        return [
            {
                **row,
                "snippet": row["snippet"][:self.snippet_max_chars],
            }
            for row in rows
        ]

    async def search(
        self, query: str, k: int | None = None
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("query must not be empty")
        requested_k = self.k if k is None else k
        if type(requested_k) is not int or requested_k <= 0:
            raise ValueError("k must be a positive integer")

        rows = await self.provider.search(query, k=requested_k)
        rows = self._truncate_snippets(rows[:requested_k])
        self._purge()

        urls = {valid_url(row["url"]) for row in rows}
        if len(urls) > self.mapping_limit:
            raise ValueError("Search results exceed mapping capacity")

        results = []
        for row in rows:
            # 保留原 URL，避免规范化误合并不同页面。
            url = valid_url(row["url"])
            docid = "web_" + hashlib.sha256(
                url.encode("utf-8")
            ).hexdigest()

            document = self.documents.get(docid)
            if document is not None and document["url"] != url:
                raise RuntimeError("Document ID collision")

            if document is None:
                document = {"url": url}
                self.documents[docid] = document

            previous = document.get("metadata", {})
            document["metadata"] = {
                "url": url,
                "title": row.get("title") or previous.get("title"),
                "source": row.get("source") or previous.get("source"),
                "language": row.get("language") or previous.get("language"),
            }
            self._touch_document(docid)

            results.append({
                "docid": docid,
                "score": None,
                "snippet": row["snippet"],
            })

        while len(self.documents) > self.mapping_limit:
            self._drop_document(next(iter(self.documents)))

        return results

    async def _fetch(self, docid, url):
        try:
            log.info("Reader fetch: %s", docid)
            result = await self.provider.read(url)
            text = result["text"]
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("Reader returned empty or invalid text")

            size = len(text.encode("utf-8"))
            if size > self.max_bytes:
                raise RuntimeError(
                    "Document exceeds cache capacity; "
                    "increase --content-cache-max-mib"
                )

            document = self.documents.get(docid)
            previous = document.get("metadata", {}) if document else {}
            metadata = {
                "url": result.get("url") or url,
                "title": result.get("title") or previous.get("title"),
                "source": result.get("source") or previous.get("source"),
                "language": result.get("language") or previous.get("language"),
            }

            entry = {
                "metadata": metadata,
                "text": text,
                "char_count": len(text),
                "bytes": size,
                "last_access": time.monotonic(),
            }

            self._purge()
            document = self.documents.get(docid)

            # 映射可能在网络请求期间过期或被淘汰。
            if document is not None and document["url"] == url:
                document["metadata"] = dict(metadata)
                self._drop_content(docid)
                while self.cached_bytes + size > self.max_bytes:
                    self._drop_content(next(iter(self.contents)))

                entry["last_access"] = time.monotonic()
                self.contents[docid] = entry
                self.cached_bytes += size
                self._touch_document(docid)

            return entry
        finally:
            self.inflight.pop(docid, None)

    @staticmethod
    def _consume_exception(task):
        if not task.cancelled():
            task.exception()

    async def _get_document_entry(self, docid):
        self._purge()
        document = self.documents.get(docid)
        if document is None:
            raise ValueError("Unknown or expired docid; run search again")
        self._touch_document(docid)

        entry = self.contents.get(docid)
        if entry is None:
            task = self.inflight.get(docid)
            if task is None:
                task = asyncio.create_task(
                    self._fetch(docid, document["url"])
                )
                self.inflight[docid] = task
                task.add_done_callback(self._consume_exception)
            entry = await asyncio.shield(task)
        else:
            log.info("Reader cache hit: %s", docid)

        if self.contents.get(docid) is entry:
            entry["last_access"] = time.monotonic()
            self.contents.move_to_end(docid)
        return entry

    async def search_in_document(self, docid, query, k=None):
        requested_k = self.document_search.validate_request(query, k)
        entry = await self._get_document_entry(docid)
        return await asyncio.to_thread(
            self.document_search.search,
            docid,
            entry["text"],
            query,
            requested_k,
        )

    async def read_document(
        self, docid: str, chunk_index: int = 0
    ) -> dict:
        if type(chunk_index) is not int or chunk_index < 0:
            raise ValueError("chunk_index must be a non-negative integer")
        entry = await self._get_document_entry(docid)
        total = entry["char_count"]
        count = max(1, (total + CHUNK_SIZE - 1) // CHUNK_SIZE)
        if chunk_index >= count:
            raise ValueError(
                f"chunk_index {chunk_index} out of range; "
                f"valid range is 0..{count - 1}"
            )

        start = chunk_index * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total)
        has_more = chunk_index + 1 < count

        return {
            "docid": docid,
            "chunk_index": chunk_index,
            "char_start": start,
            "char_end": end,
            "metadata": {
                "char_count": total,
                "chunk_size_chars": CHUNK_SIZE,
                "chunk_count": count,
            },
            "has_more": has_more,
            "next_chunk_index": chunk_index + 1 if has_more else None,
            "text": entry["text"][start:end],
        }

    async def close(self):
        if self.cleaner is not None:
            self.cleaner.cancel()
            with suppress(asyncio.CancelledError):
                await self.cleaner
            self.cleaner = None

        tasks = list(self.inflight.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        try:
            await self.provider.close()
        finally:
            self.inflight.clear()
            self.documents.clear()
            self.contents.clear()
            self.cached_bytes = 0
