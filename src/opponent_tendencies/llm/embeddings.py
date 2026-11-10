"""Embedding providers for the media index.

The index stores one pinned model ID in its metadata and refuses to query with
an embedder that reports a different ID. :class:`ArcEmbedder` calls the VT ARC
OpenAI-compatible ``/embeddings`` endpoint; :class:`HashingEmbedder` is a
deterministic offline provider used only for tests and chunking evaluation.

No credentials are logged. The VT ARC client here is intentionally small; Stage
6 owns retries, deadlines, and full request logging for the chat client.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections.abc import Sequence
from typing import Any, Final, Protocol

import httpx

__all__ = [
    "ArcEmbedder",
    "Embedder",
    "EmbeddingError",
    "HashingEmbedder",
]

_TOKEN = re.compile(r"[a-z0-9]+")
_BIGRAM_WEIGHT: Final = 0.5


class EmbeddingError(RuntimeError):
    """The embedding provider could not return usable vectors."""


class Embedder(Protocol):
    """One pinned embedding model, used for both indexing and queries."""

    @property
    def model_id(self) -> str:
        """Stable identifier recorded in the index metadata."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed stored chunks."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one search query with the same model."""


class ArcEmbedder:
    """VT ARC OpenAI-compatible embeddings client."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        batch_size: int = 64,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("embedding base_url must be an http(s) URL")
        if not model.strip():
            raise ValueError("embedding model must not be blank")
        if batch_size <= 0:
            raise ValueError("embedding batch_size must be positive")
        if timeout <= 0:
            raise ValueError("embedding timeout must be positive")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._batch_size = batch_size
        self._timeout = timeout
        self._client = client

    @property
    def model_id(self) -> str:
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(list(texts[start : start + self._batch_size])))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self._model, "input": texts}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            response = self._post(payload, headers)
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"embedding request failed: {exc}") from exc
        if response.status_code != httpx.codes.OK:
            raise EmbeddingError(
                f"embedding provider returned HTTP {response.status_code}: "
                f"{_error_excerpt(response)}"
            )
        try:
            document = response.json()
        except ValueError as exc:
            raise EmbeddingError("embedding provider returned non-JSON content") from exc
        return _parse_embeddings(document, expected=len(texts), expected_model=self._model)

    def _post(self, payload: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        if self._client is not None:
            return self._client.post(
                f"{self._base_url}/embeddings",
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
        with httpx.Client(timeout=self._timeout) as client:
            return client.post(
                f"{self._base_url}/embeddings",
                json=payload,
                headers=headers,
            )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


def _parse_embeddings(
    document: object,
    *,
    expected: int,
    expected_model: str,
) -> list[list[float]]:
    if not isinstance(document, dict):
        raise EmbeddingError("embedding response must be a JSON object")
    reported_model = document.get("model")
    if reported_model is not None and reported_model != expected_model:
        raise EmbeddingError(
            f"embedding response reports model {reported_model!r}; "
            f"index is pinned to {expected_model!r}"
        )
    data = document.get("data")
    if not isinstance(data, list) or len(data) != expected:
        raise EmbeddingError(
            f"embedding response has {len(data) if isinstance(data, list) else 'no'} "
            f"vectors; expected {expected}"
        )

    ordered: list[tuple[int, list[float]]] = []
    dimensions: int | None = None
    for position, item in enumerate(data):
        if not isinstance(item, dict):
            raise EmbeddingError("embedding response item must be a JSON object")
        index = item.get("index", position)
        vector = item.get("embedding")
        if isinstance(index, bool) or not isinstance(index, int):
            raise EmbeddingError("embedding response item has a non-integer index")
        if not isinstance(vector, list) or not vector:
            raise EmbeddingError("embedding response item has no vector")
        floats: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EmbeddingError("embedding vector contains a non-numeric value")
            floats.append(float(value))
        if dimensions is None:
            dimensions = len(floats)
        elif len(floats) != dimensions:
            raise EmbeddingError("embedding response mixes vector dimensions")
        ordered.append((index, floats))

    ordered.sort(key=lambda pair: pair[0])
    indexes = [index for index, _ in ordered]
    if indexes != list(range(expected)):
        raise EmbeddingError("embedding response indexes are missing or duplicated")
    return [vector for _, vector in ordered]


def _error_excerpt(response: httpx.Response) -> str:
    text = response.text.strip().replace("\n", " ")
    return text[:200] or "<empty body>"


class HashingEmbedder:
    """Deterministic feature hashing, for offline tests and evaluation only.

    Words and word bigrams are hashed into a fixed number of signed buckets and
    the vector is L2-normalized. It has no semantic training, but it is stable
    across processes, which is what index-reuse and retrieval-filter tests need.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")
        self.dimensions = dimensions

    @property
    def model_id(self) -> str:
        return f"local-hash-v1-{self.dimensions}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _TOKEN.findall(text.lower())
        for token in tokens:
            self._add(vector, token, 1.0)
        for left, right in itertools.pairwise(tokens):
            self._add(vector, f"{left}_{right}", _BIGRAM_WEIGHT)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    def _add(self, vector: list[float], feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % self.dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign * weight
