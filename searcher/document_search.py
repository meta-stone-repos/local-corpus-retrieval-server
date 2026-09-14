"""Transient, character-window BM25 search inside one document."""

import math
import re
from collections import Counter

_HAN = (
    "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    "\U00020000-\U0002fa1f"
)
_HAN_SPLIT = re.compile("([" + _HAN + "]+)")
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text):
    """Lexical matching only: words plus Han unigrams/bigrams."""
    terms = []
    for index, part in enumerate(_HAN_SPLIT.split(text.casefold())):
        if index % 2:
            terms.extend(part)
            terms.extend(
                part[i:i + 2] for i in range(len(part) - 1)
            )
        else:
            terms.extend(_WORDS.findall(part))
    return terms


class DocumentSearch:
    def __init__(
        self,
        max_chars=1000,
        overlap_chars=200,
        default_k=3,
        max_k=10,
    ):
        for name, value in (
            ("max_chars", max_chars),
            ("default_k", default_k),
            ("max_k", max_k),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(overlap_chars) is not int
            or not 0 <= overlap_chars < max_chars
        ):
            raise ValueError(
                "Require 0 <= overlap_chars < max_chars"
            )
        if default_k > max_k:
            raise ValueError("default_k must not exceed max_k")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.default_k = default_k
        self.max_k = max_k

    def validate_request(self, query, k=None):
        requested = self.default_k if k is None else k
        if type(requested) is not int or not 1 <= requested <= self.max_k:
            raise ValueError(f"k must be an integer from 1 to {self.max_k}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        if not tokenize(query):
            raise ValueError("query must contain searchable words or characters")
        return requested

    def search(self, docid, text, query, k=None):
        requested = self.validate_request(query, k)
        if not isinstance(text, str):
            raise ValueError("Document text must be a string")

        response = {
            "docid": str(docid),
            "query": query,
            "results": [],
        }
        if not text:
            return response

        query_terms = set(tokenize(query))
        windows = []
        document_frequency = Counter()
        total_length = 0
        step = self.max_chars - self.overlap_chars

        for start in range(0, len(text), step):
            end = min(start + self.max_chars, len(text))
            terms = tokenize(text[start:end])
            length = len(terms)
            frequencies = Counter(
                term for term in terms if term in query_terms
            )
            document_frequency.update(frequencies.keys())
            windows.append((start, end, length, frequencies))
            total_length += length
            if end == len(text):
                break

        count = len(windows)
        average_length = total_length / count if total_length else 1.0
        idf = {
            term: math.log(
                1.0 + (count - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

        ranked = []
        k1, b = 1.5, 0.75
        for start, end, length, frequencies in windows:
            if not frequencies:
                continue
            normalization = k1 * (
                1.0 - b + b * length / average_length
            )
            score = sum(
                idf[term] * frequency * (k1 + 1.0)
                / (frequency + normalization)
                for term, frequency in frequencies.items()
            )
            ranked.append((score, start, end))

        ranked.sort(key=lambda row: (-row[0], row[1]))
        selected = []

        for _, start, end in ranked:
            # Suppress candidates overlapping >= 50% of the shorter
            # region. The ordinary 200/1000 boundary overlap remains.
            duplicate = False
            for previous_start, previous_end in selected:
                overlap = max(
                    0, min(end, previous_end) - max(start, previous_start)
                )
                shorter = min(end - start, previous_end - previous_start)
                if overlap / shorter >= 0.5:
                    duplicate = True
                    break
            if duplicate:
                continue

            selected.append((start, end))
            response["results"].append({
                "rank": len(selected),
                "char_start": start,
                "char_end": end,
                "chunk_index": start // 8192,
                "end_chunk_index": (end - 1) // 8192,
                "text": text[start:end],
            })
            if len(selected) >= requested:
                break

        return response
