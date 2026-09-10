from __future__ import annotations

import copy
import math
import os
from urllib.parse import urlsplit

import httpx

from .base import check_keys, valid_url


class IQSProvider:
    def __init__(self, config: dict):
        check_keys(
            config,
            {
                "api_key", "api_key_env", "base_url",
                "timeout_seconds", "search", "read",
            },
            "iqs",
        )

        key_name = config.get("api_key_env", "IQS_API_KEY")
        if not isinstance(key_name, str) or not key_name:
            raise ValueError("iqs.api_key_env must be a nonempty string")
        key = config.get("api_key") or os.environ.get(key_name)
        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                f"Set iqs.api_key in config or environment variable {key_name}"
            )

        search_config = copy.deepcopy(config.get("search", {}))
        check_keys(
            search_config,
            {"engineType", "timeRange", "category", "advancedParams"},
            "iqs.search",
        )
        self.search_options = {
            "engineType": "LiteAdvanced",
            "timeRange": "NoLimit",
            **search_config,
        }
        if self.search_options["engineType"] not in {
            "Generic", "GenericAdvanced", "LiteAdvanced", "Deep",
        }:
            raise ValueError("Invalid iqs.search.engineType")
        if self.search_options["timeRange"] not in {
            "OneDay", "OneWeek", "OneMonth", "OneYear", "NoLimit",
        }:
            raise ValueError("Invalid iqs.search.timeRange")
        if "category" in search_config:
            if not isinstance(search_config["category"], str):
                raise ValueError("iqs.search.category must be a string")

        advanced = self.search_options.get("advancedParams", {})
        check_keys(
            advanced,
            {"numResults", "startPublishedDate", "endPublishedDate"},
            "iqs.search.advancedParams",
        )
        advanced = {"numResults": 10, **advanced}
        count = advanced["numResults"]
        if isinstance(count, str) and count.isdecimal():
            count = int(count)
        if type(count) is not int or not 1 <= count <= 50:
            raise ValueError("iqs.search.advancedParams.numResults must be 1..50")
        advanced["numResults"] = str(count)

        from datetime import date

        for name in ("startPublishedDate", "endPublishedDate"):
            if name in advanced:
                value = advanced[name]
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be YYYY-MM-DD")
                try:
                    parsed = date.fromisoformat(value)
                except ValueError:
                    raise ValueError(f"{name} must be YYYY-MM-DD") from None
                if parsed.isoformat() != value:
                    raise ValueError(f"{name} must be YYYY-MM-DD")
        if (
            "startPublishedDate" in advanced
            and "endPublishedDate" in advanced
            and advanced["startPublishedDate"] > advanced["endPublishedDate"]
        ):
            raise ValueError("startPublishedDate must not exceed endPublishedDate")

        self.search_options["advancedParams"] = advanced
        # Search only needs snippets. Full text is fetched lazily by read().
        self.search_options["contents"] = {
            "mainText": False,
            "markdownText": False,
            "richMainBody": False,
            "summary": False,
            "rerankScore": True,
        }

        read_config = copy.deepcopy(config.get("read", {}))
        check_keys(
            read_config,
            {
                "pageTimeout", "maxAge", "readability",
                "excludeAllImages", "excludeAllLinks", "excludedTags",
            },
            "iqs.read",
        )
        self.read_options = {
            "pageTimeout": 10000,
            "maxAge": 0,
            "readability": {"readabilityMode": "normal"},
            **read_config,
            "formats": ["text"],
        }
        page_timeout = self.read_options["pageTimeout"]
        if type(page_timeout) is not int or not 0 <= page_timeout <= 100000:
            raise ValueError("iqs.read.pageTimeout must be 0..100000 milliseconds")
        max_age = self.read_options["maxAge"]
        if type(max_age) is not int or max_age < 0:
            raise ValueError("iqs.read.maxAge must be a nonnegative integer")

        readability = self.read_options["readability"]
        check_keys(readability, {"readabilityMode"}, "iqs.read.readability")
        if readability.get("readabilityMode", "none") not in {
            "none", "normal", "article",
        }:
            raise ValueError("Invalid iqs.read.readability.readabilityMode")
        for name in ("excludeAllImages", "excludeAllLinks"):
            if name in read_config and type(read_config[name]) is not bool:
                raise ValueError(f"iqs.read.{name} must be boolean")
        if "excludedTags" in read_config:
            tags = read_config["excludedTags"]
            if not isinstance(tags, list) or not all(
                isinstance(tag, str) for tag in tags
            ):
                raise ValueError("iqs.read.excludedTags must be a list of strings")

        timeout = float(config.get("timeout_seconds", 60))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("iqs.timeout_seconds must be finite and positive")
        if timeout <= (page_timeout + 6000) / 1000:
            raise ValueError(
                "iqs.timeout_seconds must exceed (read.pageTimeout + 6000) / 1000"
            )

        base_url = config.get("base_url", "https://cloud-iqs.aliyuncs.com/")
        valid_url(base_url)
        if urlsplit(base_url).scheme != "https":
            raise ValueError("iqs.base_url must use HTTPS")

        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {key.strip()}"},
            timeout=timeout,
        )

    async def _post(self, endpoint: str, payload: dict) -> dict:
        try:
            response = await self.client.post(endpoint, json=payload)
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"IQS {endpoint}: network error ({type(exc).__name__})"
            ) from None

        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"IQS {endpoint}: HTTP {response.status_code}")
        try:
            result = response.json()
        except ValueError:
            raise RuntimeError(f"IQS {endpoint}: invalid JSON response") from None
        if not isinstance(result, dict):
            raise RuntimeError(f"IQS {endpoint}: expected JSON object")
        if result.get("errorCode"):
            raise RuntimeError(
                f"IQS {endpoint}: API error {result['errorCode']}; "
                f"requestId={result.get('requestId', 'unknown')}"
            )
        return result

    async def search(
        self, query: str, k: int | None = None
    ) -> list[dict]:
        count = (
            int(self.search_options["advancedParams"]["numResults"])
            if k is None else k
        )
        if type(count) is not int or not 1 <= count <= 50:
            raise ValueError("IQS k must be an integer from 1 to 50")
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("IQS query must contain 1..500 characters")

        result = await self._post(
            "search/unified",
            {
                **self.search_options,
                "query": query,
                "advancedParams": {
                    **self.search_options["advancedParams"],
                    "numResults": str(count),
                },
            },
        )
        rows = result.get("pageItems")
        if not isinstance(rows, list):
            raise RuntimeError("IQS search: missing pageItems list")

        results = []
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError("IQS search: malformed page item")
            url = valid_url(row.get("link"))
            snippet = row.get("snippet") or ""
            if not isinstance(snippet, str):
                raise RuntimeError("IQS search: snippet must be text")
            results.append({
                "url": url,
                "snippet": snippet,
                "title": row.get("title"),
                "source": row.get("hostname") or urlsplit(url).hostname,
                "language": None,
                "publish_date": row.get("publishedTime"),
            })
        return results

    async def read(self, url: str) -> dict:
        url = valid_url(url)
        result = await self._post(
            "readpage/basic",
            {**self.read_options, "url": url},
        )
        data = result.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("IQS reader: missing data object")

        status = data.get("statusCode")
        if type(status) is not int or not 200 <= status < 300:
            raise RuntimeError(
                f"IQS reader: target statusCode={status}; "
                f"requestId={result.get('requestId', 'unknown')}"
            )
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("IQS reader: empty or missing text")

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            "text": text,
            "url": url,
            "title": metadata.get("title"),
            "source": metadata.get("siteName") or urlsplit(url).hostname,
            "language": metadata.get("language") or None,
        }

    async def close(self) -> None:
        await self.client.aclose()
