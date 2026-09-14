from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Annotated
from pydantic import Field

from fastmcp import FastMCP

from mcp_sse_compat import install_legacy_sse_cleanup_patch
from online_provider import create_provider
from online_service import OnlineService
from document_search import DocumentSearch


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(
            "must be a finite positive number"
        )
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


async def serve(args, config):
    # 在服务进程启动时创建，客户端断开不会释放。
    provider = create_provider(config)
    service = OnlineService(
        provider=provider,
        content_ttl=args.content_cache_idle_ttl_seconds,
        max_bytes=args.content_cache_max_mib * 1024 * 1024,
        mapping_ttl=args.mapping_idle_ttl_seconds,
        mapping_limit=args.mapping_max_entries,
        cleanup_interval=args.cache_cleanup_interval_seconds,
        snippet_max_chars=args.snippet_max_chars,
        k=args.k,
        document_search=DocumentSearch(
            max_chars=args.document_search_max_chars,
            overlap_chars=args.document_search_overlap_chars,
            default_k=args.document_search_default_k,
            max_k=args.document_search_max_k,
        ),
    )

    try:
        await service.start()
        mcp = FastMCP(name="online-search-server")

        @mcp.tool(
            name="search",
            description=(
                f"Search online. Choose k per call (default {args.k}). "
                "Results may be fewer due to provider limits. IQS accepts k=1..50. "
                "Return up to k results "
                "with docid, score and snippet. "
                "score is null. Use docid with read_document "
                "to read the full content."
            ),
        )
        async def search(
            query: str,
            k: Annotated[
                int,
                Field(
                    gt=0,
                    strict=True,
                    description="Maximum results requested; subject to provider limits.",
                ),
            ] = args.k,
        ) -> list[dict]:
            return await service.search(query, k=k)

        @mcp.tool(
            name="read_document",
            description=(
                "Read a web document by docid and zero-based chunk_index. "
                "Chunks contain up to 8192 Unicode characters. "
                "Use metadata.chunk_count, has_more and next_chunk_index "
                "to continue reading."
            ),
        )
        async def read_document(
            docid: str, chunk_index: int = 0
        ) -> dict:
            return await service.read_document(docid, chunk_index)


        async def search_in_document(
            docid: str,
            query: str,
            k: int = args.document_search_default_k,
        ) -> dict:
            return await service.search_in_document(docid, query, k)

        search_in_document.__annotations__["k"] = Annotated[
            int,
            Field(gt=0, le=args.document_search_max_k, strict=True),
        ]
        mcp.tool(
            name="search_in_document",
            description=(
                "Search relevant passages inside one known web document. "
                "Use keywords, phrase combinations, or a question. "
                "Matching is lexical BM25, not semantic search. "
                f"Return up to k passages; default {args.document_search_default_k}, "
                f"maximum {args.document_search_max_k}. "
                "Fetch the document lazily when needed and reuse the read cache. "
                "Use chunk_index/end_chunk_index with read_document "
                "when more context is needed."
            ),
        )(search_in_document)

        if args.transport == "sse":
            install_legacy_sse_cleanup_patch()

        await mcp.run_http_async(
            transport=args.transport,
            host=args.host,
            port=args.port,
            path="/mcp",
        )
    finally:
        await service.close()


def main():
    parser = argparse.ArgumentParser(
        description="Online retrieval MCP server"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--k",
        type=positive_int,
        default=5,
        help="Default search k when the Agent omits it (default: 5).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--transport",
        choices=["sse", "streamable-http"],
        default="sse",
    )
    parser.add_argument(
        "--content-cache-idle-ttl-seconds",
        type=positive_float,
        default=1800,
    )
    parser.add_argument(
        "--content-cache-max-mib",
        type=positive_int,
        default=256,
    )
    parser.add_argument(
        "--mapping-idle-ttl-seconds",
        type=positive_float,
        default=86400,
    )
    parser.add_argument(
        "--mapping-max-entries",
        type=positive_int,
        default=100000,
    )
    parser.add_argument(
        "--cache-cleanup-interval-seconds",
        type=positive_float,
        default=60,
    )
    parser.add_argument(
        "--snippet-max-chars",
        type=int,
        default=2048,
        help="Maximum Unicode characters per snippet; -1 disables truncation.",
    )
    parser.add_argument("--document-search-max-chars", type=int, default=1000)
    parser.add_argument("--document-search-overlap-chars", type=int, default=200)
    parser.add_argument("--document-search-default-k", type=int, default=3)
    parser.add_argument("--document-search-max-k", type=int, default=10)

    args = parser.parse_args()

    if args.snippet_max_chars != -1 and args.snippet_max_chars <= 0:
        parser.error("--snippet-max-chars must be positive or -1")

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config = json.loads(
        Path(args.config).read_text(encoding="utf-8")
    )
    asyncio.run(serve(args, config))


if __name__ == "__main__":
    main()
