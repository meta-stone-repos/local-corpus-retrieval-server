"""Backport cleanup for legacy MCP SSE sessions."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from importlib.metadata import version

from packaging.version import Version

logger = logging.getLogger(__name__)


def install_legacy_sse_cleanup_patch() -> bool:
    """Install the cleanup backport for MCP versions older than 1.29."""

    if Version(version("mcp")) >= Version("1.29.0"):
        return False

    from mcp.server.sse import SseServerTransport

    if getattr(SseServerTransport, "_agentflow_cleanup_patch", False):
        return False

    original_connect_sse = SseServerTransport.connect_sse

    @asynccontextmanager
    async def connect_sse_with_cleanup(self, scope, receive, send):
        owned_session_ids = ()

        try:
            async with original_connect_sse(
                self,
                scope,
                receive,
                send,
            ) as streams:
                read_state = getattr(streams[0], "_state", None)
                writers = getattr(self, "_read_stream_writers", {})

                if read_state is not None:
                    owned_session_ids = tuple(
                        session_id
                        for session_id, writer in tuple(writers.items())
                        if getattr(writer, "_state", None) is read_state
                    )

                yield streams
        finally:
            writers = getattr(self, "_read_stream_writers", {})
            stale_ids = set(owned_session_ids)

            stale_ids.update(
                session_id
                for session_id, writer in tuple(writers.items())
                if getattr(writer, "_closed", False)
            )

            owners = getattr(self, "_session_owners", None)

            for session_id in stale_ids:
                writers.pop(session_id, None)
                if owners is not None:
                    owners.pop(session_id, None)

            if stale_ids:
                logger.debug(
                    "Removed %d disconnected legacy SSE session(s)",
                    len(stale_ids),
                )

    SseServerTransport.connect_sse = connect_sse_with_cleanup
    SseServerTransport._agentflow_cleanup_patch = True
    return True
