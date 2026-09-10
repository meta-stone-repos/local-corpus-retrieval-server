from __future__ import annotations

import math
import os

import httpx

from .base import check_keys, valid_url


class ZhipuProvider:
    def __init__(self, config: dict):
        check_keys(
            config,
            {"api_key", "api_key_env", "base_url", "timeout_seconds", "search", "read"},
            "zhipu",
        )
        key_name = config.get("api_key_env", "ZHIPU_API_KEY")
        key = config.get("api_key") or os.environ.get(key_name)
        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                f"Set zhipu.api_key in config or environment variable {key_name}"
            )
        key = key.strip()

        search_config = config.get("search", {})
        check_keys(
            search_config,
            {
                "search_engine", "search_intent", "count",
                "search_recency_filter", "content_size",
            },
            "zhipu.search",
        )
        self.search_options = {
            "search_engine": "search_std",
            "search_intent": False,
            "count": 10,
            "search_recency_filter": "noLimit",
            "content_size": "medium",
            **search_config,
        }
        if self.search_options["search_recency_filter"] not in {
            "oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit",
        }:
            raise ValueError("Invalid search_recency_filter")
        count = self.search_options["count"]
        if type(count) is not int or count <= 0:
            raise ValueError("search.count must be a positive integer")

        read_config = config.get("read", {})
        check_keys(
            read_config,
            {
                "no_cache", "return_format", "retain_images",
                "with_images_summary", "with_links_summary",
            },
            "zhipu.read",
        )
        self.read_options = {
            "no_cache": True,
            "return_format": "text",
            "retain_images": False,
            "with_images_summary": False,
            "with_links_summary": False,
            **read_config,
        }
        if self.read_options["return_format"] != "text":
            raise ValueError("read.return_format must be text")

        timeout = float(config.get("timeout_seconds", 60))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_seconds must be finite and positive")

        base_url = config.get(
            "base_url", "https://open.bigmodel.cn/api/paas/v4/"
        ).rstrip("/") + "/"
        valid_url(base_url)

        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )

    async def _post(self, endpoint: str, payload: dict) -> dict:
        try:
            response = await self.client.post(endpoint, json=payload)
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Zhipu {endpoint}: network error ({type(exc).__name__})"
            ) from None

        if response.is_error:
            raise RuntimeError(
                f"Zhipu {endpoint}: HTTP {response.status_code}"
            )
        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(
                f"Zhipu {endpoint}: invalid JSON response"
            ) from None
        if not isinstance(data, dict):
            raise RuntimeError(f"Zhipu {endpoint}: expected JSON object")
        if data.get("error"):
            error = data["error"]
            code = (
                error.get("code", "unknown")
                if isinstance(error, dict)
                else "unknown"
            )
            raise RuntimeError(f"Zhipu {endpoint}: API error {code}")
        return data

    async def search(
        self, query: str, k: int | None = None
    ) -> list[dict]:
        count = self.search_options["count"] if k is None else k
        if type(count) is not int or count <= 0:
            raise ValueError("Zhipu k must be a positive integer")
        data = await self._post(
            "web_search",
            {
                **self.search_options,
                "search_query": query,
                "count": count,
            },
        )
        rows = data.get("search_result")
        if not isinstance(rows, list):
            raise RuntimeError("Zhipu search: missing search_result list")

        results = []
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError("Zhipu search: malformed result")
            snippet = row.get("content") or ""
            if not isinstance(snippet, str):
                raise RuntimeError("Zhipu search: content must be text")
            results.append({
                "url": valid_url(row.get("link")),
                "snippet": snippet,
                "title": row.get("title"),
                "source": row.get("media"),
                "language": row.get("language") or None,
                "publish_date": row.get("publish_date"),
            })
        return results

    async def read(self, url: str) -> dict:
        data = await self._post(
            "reader",
            {**self.read_options, "url": valid_url(url)},
        )
        result = data.get("reader_result")
        if not isinstance(result, dict):
            raise RuntimeError("Zhipu reader: missing reader_result")

        text = result.get("content")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Zhipu reader: empty or missing content")

        metadata = result.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        return {
            "text": text,
            "title": result.get("title") or None,
            "url": result.get("url") or url,
            "source": result.get("source") or None,
            "language": (
                metadata.get("lang")
                or metadata.get("language")
                or None
            ),
        }

    async def close(self):
        await self.client.aclose()
