#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import html
import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


LOGGER = logging.getLogger("wikimedia-loader")

HREF_RE = re.compile(
    r"""href=["']([^"']+\.tar\.gz)["']""",
    re.IGNORECASE,
)

SIZE_RE = re.compile(
    r"</a>\s+\S+\s+\S+\s+(\d+)\s*$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and filter files without downloading or writing manifest",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Replace an existing frozen input manifest",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("YAML root must be a mapping")

    if config.get("config_version") != 1:
        raise ValueError("Only config_version: 1 is supported")

    return config


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")

    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must contain only strings")

    return value


def matches(logical_name: str, pattern: str) -> bool:
    # 不带 / 的规则匹配 basename；带 / 的规则匹配相对路径。
    target = (
        logical_name
        if "/" in pattern
        else Path(logical_name).name
    )

    return fnmatch.fnmatchcase(target, pattern)


def selected(logical_name: str, source: dict[str, Any]) -> bool:
    archive_pattern = str(
        source.get("archive_pattern") or "*.tar.gz"
    )

    include_files = string_list(
        source.get("include_files"),
        "source.include_files",
    )

    exclude_files = string_list(
        source.get("exclude_files"),
        "source.exclude_files",
    )

    if not matches(logical_name, archive_pattern):
        return False

    if include_files and not any(
        matches(logical_name, pattern)
        for pattern in include_files
    ):
        return False

    if any(
        matches(logical_name, pattern)
        for pattern in exclude_files
    ):
        return False

    return True


def discover_local(
    project_root: Path,
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    path_value = source.get("path")

    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(
            "source.path is required for source.type=local"
        )

    root = resolve_path(project_root, path_value)

    if not root.is_dir():
        raise NotADirectoryError(root)

    recursive = bool(source.get("recursive", True))

    iterator = root.rglob("*") if recursive else root.glob("*")
    archives: list[dict[str, Any]] = []

    for path in iterator:
        if not path.is_file():
            continue

        logical_name = path.relative_to(root).as_posix()

        if not selected(logical_name, source):
            continue

        stat = path.stat()

        archives.append(
            {
                "name": Path(logical_name).name,
                "logical_name": logical_name,
                "source": str(path),
                "local_path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

    archives.sort(key=lambda item: item["logical_name"])

    return archives


def read_web_listing(
    url: str,
    user_agent: str,
) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent},
    )

    with urllib.request.urlopen(
        request,
        timeout=60,
    ) as response:
        return response.read().decode(
            "utf-8",
            errors="replace",
        )


def discover_web(
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    url = source.get("url")

    if not isinstance(url, str) or not url.strip():
        raise ValueError(
            "source.url is required for source.type=web"
        )

    user_agent = str(
        source.get("user_agent")
        or "BrowseComp-Plus/1.0"
    )

    listing = read_web_listing(url, user_agent)
    discovered: dict[str, dict[str, Any]] = {}

    for line in listing.splitlines():
        href_match = HREF_RE.search(line)

        if href_match is None:
            continue

        href = html.unescape(href_match.group(1))
        absolute_url = urllib.parse.urljoin(url, href)

        parsed = urllib.parse.urlparse(absolute_url)

        if parsed.scheme not in {"http", "https"}:
            continue

        name = Path(
            urllib.parse.unquote(parsed.path)
        ).name

        if not selected(name, source):
            continue

        size_match = SIZE_RE.search(line)
        expected_size = (
            int(size_match.group(1))
            if size_match
            else None
        )

        discovered[name] = {
            "name": name,
            "logical_name": name,
            "source": absolute_url,
            "local_path": None,
            "size": expected_size,
            "mtime_ns": None,
        }

    return [
        discovered[name]
        for name in sorted(discovered)
    ]


def download_archive(
    archive: dict[str, Any],
    destination_dir: Path,
    user_agent: str,
) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination = destination_dir / archive["name"]
    partial = destination.with_suffix(
        destination.suffix + ".part"
    )

    expected_size = archive.get("size")

    if destination.is_file():
        actual_size = destination.stat().st_size

        if expected_size is None or actual_size == expected_size:
            LOGGER.info(
                "Using cached %s bytes=%d",
                destination.name,
                actual_size,
            )

            archive["local_path"] = str(destination)
            archive["size"] = actual_size
            return

        raise ValueError(
            f"Cached file size mismatch: {destination}; "
            f"expected={expected_size}, actual={actual_size}"
        )

    if (
        partial.is_file()
        and expected_size is not None
        and partial.stat().st_size == expected_size
    ):
        os.replace(partial, destination)
        archive["local_path"] = str(destination)
        return

    for attempt in range(1, 4):
        offset = (
            partial.stat().st_size
            if partial.is_file()
            else 0
        )

        headers = {"User-Agent": user_agent}

        if offset:
            headers["Range"] = f"bytes={offset}-"

        request = urllib.request.Request(
            archive["source"],
            headers=headers,
        )

        try:
            LOGGER.info(
                "Downloading %s attempt=%d offset=%d",
                archive["name"],
                attempt,
                offset,
            )

            with urllib.request.urlopen(
                request,
                timeout=120,
            ) as response:
                status = getattr(response, "status", 200)

                if offset and status == 206:
                    mode = "ab"
                else:
                    mode = "wb"

                with partial.open(mode) as output:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)

                        if not chunk:
                            break

                        output.write(chunk)

            actual_size = partial.stat().st_size

            if (
                expected_size is not None
                and actual_size != expected_size
            ):
                raise ValueError(
                    f"Downloaded size mismatch for "
                    f"{archive['name']}: "
                    f"expected={expected_size}, "
                    f"actual={actual_size}"
                )

            os.replace(partial, destination)

            archive["local_path"] = str(destination)
            archive["size"] = actual_size

            LOGGER.info(
                "Downloaded %s bytes=%d",
                archive["name"],
                actual_size,
            )

            return

        except Exception:
            if attempt == 3:
                raise

            LOGGER.exception(
                "Download failed; retrying %s",
                archive["name"],
            )

            time.sleep(attempt * 2)


def selection_hash(source: dict[str, Any]) -> str:
    relevant = {
        "type": source.get("type"),
        "url": source.get("url"),
        "path": source.get("path"),
        "recursive": source.get("recursive", True),
        "archive_pattern": source.get(
            "archive_pattern",
            "*.tar.gz",
        ),
        "include_files": source.get(
            "include_files",
            [],
        ),
        "exclude_files": source.get(
            "exclude_files",
            [],
        ),
    }

    payload = json.dumps(
        relevant,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def write_manifest(
    path: Path,
    manifest: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(temporary, path)


def print_selection(
    archives: list[dict[str, Any]],
) -> None:
    known_bytes = sum(
        int(item["size"])
        for item in archives
        if item.get("size") is not None
    )

    print(
        f"selected_files={len(archives)} "
        f"known_bytes={known_bytes}"
    )

    for item in archives:
        print(
            f"{item.get('size')} "
            f"{item['logical_name']}"
        )


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(Path(args.config).resolve())

    corpus = config.get("corpus") or {}
    source = config.get("source") or {}
    runtime = config.get("runtime") or {}

    corpus_name = str(
        corpus.get("name") or ""
    ).strip()

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*",
        corpus_name,
    ):
        raise ValueError("Invalid corpus.name")

    source_type = source.get("type")

    if source_type == "local":
        archives = discover_local(
            project_root,
            source,
        )
    elif source_type == "web":
        archives = discover_web(source)
    else:
        raise ValueError(
            "source.type must be web or local"
        )

    if not archives:
        raise ValueError(
            "No archives matched include/exclude rules"
        )

    print_selection(archives)

    dry_run = bool(
        args.dry_run
        or runtime.get("dry_run", False)
    )

    if dry_run:
        return

    manifest_path = (
        project_root
        / "corpora"
        / corpus_name
        / "input_manifest.json"
    )

    current_hash = selection_hash(source)

    if manifest_path.exists() and not args.refresh:
        existing = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if existing.get("selection_hash") != current_hash:
            raise ValueError(
                "Selection config changed. "
                "Use --refresh to replace the frozen manifest."
            )

        if not runtime.get("resume", True):
            raise FileExistsError(
                f"Manifest already exists: {manifest_path}"
            )

        LOGGER.info(
            "Reusing frozen manifest %s",
            manifest_path,
        )

        archives = existing["archives"]

    if source_type == "web":
        download_value = source.get("download_dir")

        if not isinstance(download_value, str):
            raise ValueError(
                "source.download_dir is required "
                "for source.type=web"
            )

        download_dir = resolve_path(
            project_root,
            download_value,
        )

        user_agent = str(
            source.get("user_agent")
            or "BrowseComp-Plus/1.0"
        )

        for archive in archives:
            download_archive(
                archive,
                download_dir,
                user_agent,
            )

    manifest = {
        "manifest_version": 1,
        "corpus": corpus_name,
        "source_type": source_type,
        "selection_hash": current_hash,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "selected_count": len(archives),
        "total_bytes": sum(
            int(item["size"])
            for item in archives
            if item.get("size") is not None
        ),
        "archives": archives,
    }

    write_manifest(
        manifest_path,
        manifest,
    )

    LOGGER.info(
        "Input manifest written: %s",
        manifest_path,
    )


if __name__ == "__main__":
    main()
