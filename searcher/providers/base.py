from __future__ import annotations

from typing import Protocol
from urllib.parse import urlsplit


def check_keys(config: dict, allowed: set[str], label: str):
    if not isinstance(config, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"{label}: unknown options {sorted(unknown)}")


def valid_url(url: str) -> str:
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Expected an absolute HTTP(S) URL")
    return url


class Provider(Protocol):
    async def search(self, query: str, k: int) -> list[dict]: ...
    async def read(self, url: str) -> dict: ...
    async def close(self) -> None: ...
