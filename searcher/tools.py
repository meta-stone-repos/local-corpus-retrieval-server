import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from searchers.base import BaseSearcher
from transformers import AutoTokenizer


def register_tools(
    mcp: FastMCP,
    searcher: BaseSearcher,
    snippet_max_tokens: int | None = None,
    k: int = 5,
    include_get_document: bool = True,
    bm25_workers: int = 1,
    dense_max_batch_size: int = 16,
    dense_max_wait_ms: float = 5.0,
    dense_max_queue_size: int = 64,
):
    tokenizer = None
    if snippet_max_tokens and snippet_max_tokens > 0:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # BM25 uses multiple CPU workers.
    # Dense keeps exactly one backend worker; concurrency is achieved
    # by batching many queries into one GPU execution.
    if searcher.search_type == "BM25":
        executor_workers = max(1, bm25_workers)
    else:
        executor_workers = 1

    print(
        f"[MCP] search_type={searcher.search_type}, "
        f"executor_workers={executor_workers}"
    )

    executor = ThreadPoolExecutor(
        max_workers=executor_workers,
        thread_name_prefix="mcp-search-worker",
    )

    document_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="mcp-document-worker",
    )

    READ_CHUNK_SIZE_CHARS = 8192

    def _build_search_metadata(
        cand: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add serving-side document reading metadata."""
        metadata = dict(cand.get("metadata") or {})

        char_count = metadata.get("char_count")

        # Fallback for searchers/document sources without precomputed metadata.
        if char_count is None:
            text = cand.get("text") or ""
            char_count = len(text)

        char_count = int(char_count)

        chunk_count = max(
            1,
            (
                char_count
                + READ_CHUNK_SIZE_CHARS
                - 1
            )
            // READ_CHUNK_SIZE_CHARS,
        )

        metadata["char_count"] = char_count
        metadata["chunk_size_chars"] = READ_CHUNK_SIZE_CHARS
        metadata["chunk_count"] = chunk_count

        return metadata

    def _format_candidates(
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply snippet truncation and produce MCP-facing results."""
        if snippet_max_tokens and snippet_max_tokens > 0 and tokenizer:
            # Avoid feeding pathological full documents into the tokenizer
            # when we only need a short preview.
            raw_char_limit = max(
                snippet_max_tokens * 16,
                8192,
            )

            for cand in candidates:
                text = cand["text"]

                guarded_text = text[:raw_char_limit]

                tokens = tokenizer.encode(
                    guarded_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=snippet_max_tokens,
                )

                cand["snippet"] = tokenizer.decode(
                    tokens,
                    skip_special_tokens=True,
                )
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

        results = []

        for cand in candidates:
            if cand.get("score") is None:
                results.append(
                    {
                        "docid": cand["docid"],
                        "snippet": cand["snippet"],
                        "metadata": _build_search_metadata(cand),
                    }
                )
            else:
                results.append(
                    {
                        "docid": cand["docid"],
                        "score": cand["score"],
                        "snippet": cand["snippet"],
                        "metadata": _build_search_metadata(cand),
                    }
                )

        return results

    def _format_batch_candidates(
        batch_candidates: List[List[Dict[str, Any]]],
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch-format Dense retrieval results.

        Unlike _format_candidates(), this tokenizes all Dense result
        documents in one call and truncates during tokenization.
        """
        flat_candidates = [
            cand
            for candidates in batch_candidates
            for cand in candidates
        ]

        if not flat_candidates:
            return [[] for _ in batch_candidates]

        if snippet_max_tokens and snippet_max_tokens > 0 and tokenizer:
            # texts = [
            #     cand["text"]
            #     for cand in flat_candidates
            # ]
            # Raw-text guard before tokenization.
# We only need a 512-token preview, so avoid feeding pathological
# full documents (sometimes 100k+ tokens) into the tokenizer.
            raw_char_limit = max(
                snippet_max_tokens * 16,
                8192,
            )

            texts = [
                cand["text"][:raw_char_limit]
                for cand in flat_candidates
            ]

            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                truncation=True,
                max_length=snippet_max_tokens,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            snippets = tokenizer.batch_decode(
                encoded["input_ids"],
                skip_special_tokens=True,
            )

            for cand, snippet in zip(
                flat_candidates,
                snippets,
            ):
                cand["snippet"] = snippet

        else:
            for cand in flat_candidates:
                cand["snippet"] = cand["text"]

        batch_results = []
        offset = 0

        for candidates in batch_candidates:
            results = []

            for cand in candidates:
                if cand.get("score") is None:
                    results.append(
                        {
                            "docid": cand["docid"],
                            "snippet": cand["snippet"],
                            "metadata": _build_search_metadata(cand),
                        }
                    )
                else:
                    results.append(
                        {
                            "docid": cand["docid"],
                            "score": cand["score"],
                            "snippet": cand["snippet"],
                            "metadata": _build_search_metadata(cand),
                        }
                    )

            batch_results.append(results)
            offset += len(candidates)

        return batch_results


    def _search_sync(
        query: str,
    ) -> List[Dict[str, Any]]:
        candidates = searcher.search(query, k)
        return _format_candidates(candidates)

    # def _search_batch_sync(
    #     queries: List[str],
    # ) -> List[List[Dict[str, Any]]]:
    #     """
    #     Run one true Dense batch and format every query's result.

    #     This function executes inside the single Dense backend worker.
    #     """
    #     batch_candidates = searcher.search_batch(queries, k)

    #     return [
    #         _format_candidates(candidates)
    #         for candidates in batch_candidates
    #     ]
    def _search_batch_sync(
        queries: List[str],
    ) -> List[List[Dict[str, Any]]]:
        """Run one true Dense batch and format its results."""
        batch_candidates = searcher.search_batch(
            queries,
            k,
        )

        return _format_batch_candidates(
            batch_candidates
        )

    # Enable dynamic batching only for a searcher that actually exposes
    # search_batch(). BM25 stays on its worker-pool path.
    dense_batch_enabled = (
        searcher.search_type == "FAISS"
        and callable(getattr(searcher, "search_batch", None))
    )

    class DenseBatchScheduler:
        """
        Aggregate independent async search requests into GPU batches.

        A batch is dispatched when either:
          1) max_batch_size is reached, or
          2) the oldest request has waited max_wait_ms.
        """

        def __init__(
            self,
            max_batch_size: int,
            max_wait_ms: float,
            max_queue_size: int,
        ):
            self.max_batch_size = max(1, max_batch_size)
            self.max_wait_s = max(0.0, max_wait_ms / 1000.0)

            # self.queue = asyncio.Queue()
            self.queue = asyncio.Queue(maxsize=max(1, max_queue_size))
            self.worker_task = None

        def _ensure_worker(self):
            if self.worker_task is None or self.worker_task.done():
                self.worker_task = asyncio.create_task(
                    self._batch_loop()
                )

        async def submit(
            self,
            query: str,
        ) -> List[Dict[str, Any]]:
            loop = asyncio.get_running_loop()
            future = loop.create_future()

            await self.queue.put(
                (
                    query,
                    future,
                    loop.time(),
                )
            )

            self._ensure_worker()

            return await future

        async def _batch_loop(self):
            while True:
                try:
                    first = await self.queue.get()
                except asyncio.CancelledError:
                    raise

                # batch = [first]

                # # The timeout is measured from arrival of the oldest
                # # request, not from when the batch worker happens to
                # # dequeue it.
                # oldest_enqueue_time = first[2]
                # loop = asyncio.get_running_loop()

                # deadline = (
                #     oldest_enqueue_time
                #     + self.max_wait_s
                # )

                # while len(batch) < self.max_batch_size:
                #     remaining = deadline - loop.time()

                #     if remaining <= 0:
                #         break

                #     try:
                #         item = await asyncio.wait_for(
                #             self.queue.get(),
                #             timeout=remaining,
                #         )
                #         batch.append(item)
                #     except asyncio.TimeoutError:
                #         break
                batch = [first]

                oldest_enqueue_time = first[2]
                loop = asyncio.get_running_loop()

                deadline = (
                    oldest_enqueue_time
                    + self.max_wait_s
                )

                while len(batch) < self.max_batch_size:
                    # First, greedily drain requests that are
                    # ALREADY waiting in the queue.
                    #
                    # This is important: even if the oldest
                    # request has exceeded max_wait_ms while
                    # the previous GPU batch was running,
                    # queued requests should still be grouped
                    # together instead of being dispatched
                    # one by one.
                    try:
                        while len(batch) < self.max_batch_size:
                            item = self.queue.get_nowait()
                            batch.append(item)

                    except asyncio.QueueEmpty:
                        pass

                    if len(batch) >= self.max_batch_size:
                        break

                    # No more requests are currently available.
                    # Only now do we consider waiting for new ones.
                    remaining = deadline - loop.time()

                    if remaining <= 0:
                        break

                    try:
                        item = await asyncio.wait_for(
                            self.queue.get(),
                            timeout=remaining,
                        )
                        batch.append(item)

                    except asyncio.TimeoutError:
                        break

                # Requests can be cancelled while waiting.
                active_items = [
                    item
                    for item in batch
                    if not item[1].cancelled()
                ]

                if not active_items:
                    continue

                queries = [
                    item[0]
                    for item in active_items
                ]
                futures = [
                    item[1]
                    for item in active_items
                ]

                try:
                    # Exactly one Dense backend execution at a time.
                    # Parallelism happens inside search_batch() on GPU.
                    batch_results = await loop.run_in_executor(
                        executor,
                        _search_batch_sync,
                        queries,
                    )

                    if len(batch_results) != len(futures):
                        raise RuntimeError(
                            "Dense batch result count mismatch: "
                            f"{len(batch_results)} results for "
                            f"{len(futures)} requests"
                        )

                    for future, result in zip(
                        futures,
                        batch_results,
                    ):
                        if not future.done():
                            future.set_result(result)

                except Exception as exc:
                    for future in futures:
                        if not future.done():
                            future.set_exception(exc)

    dense_scheduler = None

    if dense_batch_enabled:
        # dense_scheduler = DenseBatchScheduler(
        #     max_batch_size=dense_max_batch_size,
        #     max_wait_ms=dense_max_wait_ms,
        # )
        dense_scheduler = DenseBatchScheduler(
            max_batch_size=dense_max_batch_size,
            max_wait_ms=dense_max_wait_ms,
            max_queue_size=dense_max_queue_size,
        )

        # print(
        #     "[MCP] Dense dynamic batching enabled: "
        #     f"max_batch_size={dense_max_batch_size}, "
        #     f"max_wait_ms={dense_max_wait_ms}"
        # )
        print(
            "[MCP] Dense dynamic batching enabled: "
            f"max_batch_size={dense_max_batch_size}, "
            f"max_wait_ms={dense_max_wait_ms}, "
            f"max_queue_size={dense_max_queue_size}"
        )

    @mcp.tool(
        name="search",
        description=searcher.search_description(k),
    )
    async def search(
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Search the index and return top-k hits.
        """
        # Dense path:
        # enqueue the individual Agent request and let the scheduler
        # dynamically combine it with other requests.
        if dense_scheduler is not None:
            return await dense_scheduler.submit(query)

        # BM25 / fallback path:
        # execute individual searches in the configured worker pool.
        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            executor,
            _search_sync,
            query,
        )

    if include_get_document:

        def _read_document_sync(
            docid: str,
            chunk_index: int,
        ) -> Optional[Dict[str, Any]]:
            return searcher.read_document(
                docid,
                chunk_index,
                READ_CHUNK_SIZE_CHARS,
            )

        @mcp.tool(
            name="read_document",
            description=searcher.read_document_description(),
        )
        async def read_document(
            docid: str,
            chunk_index: int = 0,
        ) -> Optional[Dict[str, Any]]:
            """
            Read one bounded chunk of a document by ID.
            """
            loop = asyncio.get_running_loop()

            return await loop.run_in_executor(
                document_executor,
                _read_document_sync,
                docid,
                chunk_index,
            )
