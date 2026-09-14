import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Dict, List, Optional

from fastmcp import FastMCP
from searchers.base import BaseSearcher
from document_search import DocumentSearch
from pydantic import Field


def register_tools(
    mcp: FastMCP,
    searcher: BaseSearcher,
    snippet_max_chars: int = 2048,
    k: int = 5,
    include_get_document: bool = True,
    bm25_workers: int = 1,
    dense_max_batch_size: int = 16,
    dense_max_wait_ms: float = 5.0,
    dense_max_queue_size: int = 64,
    document_search_max_chars: int = 1000,
    document_search_overlap_chars: int = 200,
    document_search_default_k: int = 3,
    document_search_max_k: int = 10,
):
    if type(k) is not int or k <= 0:
        raise ValueError("Default k must be a positive integer")
    if (
        type(snippet_max_chars) is not int
        or (snippet_max_chars != -1 and snippet_max_chars <= 0)
    ):
        raise ValueError("snippet_max_chars must be positive or -1")

    document_search = DocumentSearch(
        max_chars=document_search_max_chars,
        overlap_chars=document_search_overlap_chars,
        default_k=document_search_default_k,
        max_k=document_search_max_k,
    )

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

    def _format_candidates(candidates):
        results = []
        for cand in candidates:
            body = cand["text"]
            snippet = (
                body if snippet_max_chars == -1
                else body[:snippet_max_chars]
            )
            results.append({
                "docid": cand["docid"],
                "score": (
                    float(cand["score"])
                    if cand.get("score") is not None else None
                ),
                "snippet": snippet,
            })
        return results

    def _search_sync(query, requested_k):
        candidates = searcher.search(query, requested_k)
        return _format_candidates(candidates[:requested_k])

    def _search_batch_sync(queries, requested_ks):
        # One FAISS batch uses the largest requested k.
        # Each request is sliced to its own k before formatting.
        batch_candidates = searcher.search_batch(
            queries, max(requested_ks)
        )
        if len(batch_candidates) != len(requested_ks):
            raise RuntimeError("Dense batch result count mismatch")
        return [
            _format_candidates(candidates[:requested_k])
            for candidates, requested_k in zip(
                batch_candidates, requested_ks
            )
        ]

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
            requested_k: int,
        ) -> List[Dict[str, Any]]:
            loop = asyncio.get_running_loop()
            future = loop.create_future()

            await self.queue.put(
                (
                    query,
                    future,
                    loop.time(),
                    requested_k,
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

                requested_ks = [
                    item[3] for item in active_items
                ]

                try:
                    # Exactly one Dense backend execution at a time.
                    # Parallelism happens inside search_batch() on GPU.
                    batch_results = await loop.run_in_executor(
                        executor,
                        _search_batch_sync,
                        queries,
                        requested_ks,
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
        k: Annotated[
            int,
            Field(
                gt=0,
                strict=True,
                description="Maximum number of results to retrieve.",
            ),
        ] = k,
    ) -> List[Dict[str, Any]]:
        """
        Search the index and return top-k hits.
        """
        # Dense path:
        # enqueue the individual Agent request and let the scheduler
        # dynamically combine it with other requests.
        if dense_scheduler is not None:
            return await dense_scheduler.submit(query, k)

        # BM25 / fallback path:
        # execute individual searches in the configured worker pool.
        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            executor,
            _search_sync,
            query,
            k,
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

        def _search_in_document_sync(docid, query, requested_k):
            requested_k = document_search.validate_request(query, requested_k)
            document = searcher.get_document(docid)
            if document is None or not isinstance(document.get("text"), str):
                raise ValueError("Unknown document or missing document text")
            return document_search.search(
                docid, document["text"], query, requested_k
            )

        async def search_in_document(
            docid: str,
            query: str,
            k: int = document_search_default_k,
        ) -> dict:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                document_executor,
                _search_in_document_sync,
                docid,
                query,
                k,
            )

        search_in_document.__annotations__["k"] = Annotated[
            int, Field(gt=0, le=document_search_max_k, strict=True)
        ]
        mcp.tool(
            name="search_in_document",
            description=(
                "Search relevant passages inside one known document. "
                "Use keywords, phrase combinations, or a question. "
                "Matching is lexical BM25, not semantic search. "
                f"Return up to k passages; default {document_search_default_k}, "
                f"maximum {document_search_max_k}. "
                "Use chunk_index/end_chunk_index with read_document "
                "when more context is needed."
            ),
        )(search_in_document)
