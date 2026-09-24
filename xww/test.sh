#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="/home/xww/workspace/local-corpus-retrieval-server"
export BM25_MCP_URL="${BM25_MCP_URL:-http://127.0.0.1:8090/mcp}"
export DENSE_MCP_URL="${DENSE_MCP_URL:-http://127.0.0.1:8091/mcp}"

cd "$PROJECT_DIR"
source .venv/bin/activate

python - <<'PY'
import asyncio
import os
from fastmcp import Client
from fastmcp.client.transports import SSETransport

QUERY = "What is machine learning?"

def format_tool(tool):
    description = " ".join((getattr(tool, "description", "") or "").split())
    if description:
        return f"  - {tool.name}: {description}"
    return f"  - {tool.name}"

async def test_service(name, url):
    try:
        async with Client(SSETransport(url=url)) as client:
            tools = await client.list_tools()
            print(f"\n[{name}] {url}")
            print("Tools:")
            for tool in tools:
                print(format_tool(tool))
            result = await client.call_tool("search", {"query": QUERY})
        print("Result:")
        for item in result:
            print(getattr(item, "text", item))
        return True
    except Exception as exc:
        print(f"\n[{name}] FAILED: {exc}")
        return False

async def main():
    print(f"Query: {QUERY}")
    results = await asyncio.gather(
        test_service("BM25", os.environ["BM25_MCP_URL"]),
        test_service("DENSE", os.environ["DENSE_MCP_URL"]),
    )
    if not all(results):
        raise SystemExit(1)

asyncio.run(main())
PY
