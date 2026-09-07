"""
Abstract base class for search implementations.
"""

import argparse
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseSearcher(ABC):
    """Abstract base class for all search implementations."""

    @classmethod
    @abstractmethod
    def parse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add searcher-specific arguments to the argument parser."""
        pass

    @abstractmethod
    def __init__(self, args):
        """Initialize the searcher with parsed arguments."""
        pass

    @abstractmethod
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform search and return results.

        Args:
            query: Search query string
            k: Number of results to return

        Returns:
            List of search results with format: {"docid": str, "score": float, "snippet": str}
        """
        pass

    @abstractmethod
    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve full document by ID.

        Args:
            docid: Document ID to retrieve

        Returns:
            Document dictionary with format: {"docid": str, "snippet": str} or None if not found
        """
        pass

    def read_document(
        self,
        docid: str,
        chunk_index: int = 0,
        chunk_size_chars: int = 8192,
    ) -> Optional[Dict[str, Any]]:
        """
        Read one bounded character chunk from a document.

        The API is stateless: callers explicitly specify the document ID
        and chunk index on every call.
        """
        if chunk_size_chars <= 0:
            raise ValueError("chunk_size_chars must be greater than 0")

        document = self.get_document(docid)
        if document is None:
            return None

        text = document.get("text")
        if text is None:
            return None

        char_count = len(text)
        chunk_count = max(
            1,
            (char_count + chunk_size_chars - 1) // chunk_size_chars,
        )

        if chunk_index < 0 or chunk_index >= chunk_count:
            raise ValueError(
                f"chunk_index {chunk_index} out of range; "
                f"valid range is 0..{chunk_count - 1}"
            )

        char_start = chunk_index * chunk_size_chars
        char_end = min(char_start + chunk_size_chars, char_count)

        has_more = chunk_index + 1 < chunk_count

        return {
            "docid": str(docid),
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "chunk_size_chars": chunk_size_chars,
            "char_start": char_start,
            "char_end": char_end,
            "char_count": char_count,
            "has_more": has_more,
            "next_chunk_index": chunk_index + 1 if has_more else None,
            "text": text[char_start:char_end],
        }

    def read_document_description(self) -> str:
        """Description of the bounded document-reading tool."""
        return (
            "Read one bounded chunk of a document by docid and chunk_index. "
            "Documents are split into fixed-size Unicode character chunks. "
            "Use chunk_count and has_more to determine whether more chunks remain."
        )

    @property
    @abstractmethod
    def search_type(self) -> str:
        """Return the type of search (e.g., 'BM25', 'FAISS')."""
        pass

    def search_description(self, k: int = 10) -> str:
        """
        Description of the search tool to be passed to the LLM.
        """
        return f"Perform a search on a knowledge source. Returns top-{k} hits with docid, score, and snippet. The snippet contains the document's contents (may be truncated based on token limits)."

    def get_document_description(self) -> str:
        """
        Description of the get_document tool to be passed to the LLM.
        """
        return "Retrieve a full document by its docid."
