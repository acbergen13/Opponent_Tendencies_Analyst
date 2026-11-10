"""Embedding determinism, normalization, and the VT ARC wire contract."""

from __future__ import annotations

import json
import math
from typing import Any

import httpx
import pytest

from opponent_tendencies.llm.embeddings import (
    ArcEmbedder,
    EmbeddingError,
    HashingEmbedder,
)


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    first = HashingEmbedder()
    second = HashingEmbedder()

    left = first.embed_documents(["third down play-action pass"])[0]
    right = second.embed_documents(["third down play-action pass"])[0]

    assert left == right
    assert len(left) == 256
    assert _norm(left) == pytest.approx(1.0)
    assert first.model_id == second.model_id == "local-hash-v1-256"


def test_hashing_embedder_ranks_related_text_higher() -> None:
    embedder = HashingEmbedder()
    query = embedder.embed_query("third down passing")
    related = embedder.embed_query("the third down passing plan")
    unrelated = embedder.embed_query("red zone running game")

    assert _dot(query, related) > _dot(query, unrelated)


def test_hashing_embedder_dimensions_change_the_model_id() -> None:
    small = HashingEmbedder(64)
    large = HashingEmbedder(128)

    assert len(small.embed_query("run")) == 64
    assert small.model_id != large.model_id


@pytest.mark.parametrize("dimensions", [0, -1])
def test_hashing_embedder_rejects_bad_dimensions(dimensions: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        HashingEmbedder(dimensions)


def test_arc_embedder_sends_the_pinned_model_and_batches_requests() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/embeddings")
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        calls.append(payload)
        data = [
            {"index": index, "embedding": [float(len(text)), 1.0]}
            for index, text in enumerate(payload["input"])
        ]
        return httpx.Response(200, json={"model": payload["model"], "data": list(reversed(data))})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        embedder = ArcEmbedder(
            "https://llm-api.arc.vt.edu/api/v1/",
            "test-key",
            "Qwen3-Embedding-4B",
            batch_size=2,
            client=client,
        )
        vectors = embedder.embed_documents(["alpha", "beta", "gamma"])

    assert embedder.model_id == "Qwen3-Embedding-4B"
    assert [call["model"] for call in calls] == ["Qwen3-Embedding-4B"] * 2
    assert [call["input"] for call in calls] == [["alpha", "beta"], ["gamma"]]
    assert vectors == [[5.0, 1.0], [4.0, 1.0], [5.0, 1.0]]


def test_arc_embedder_query_uses_the_same_model() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload["model"])
        return httpx.Response(
            200,
            json={"model": payload["model"], "data": [{"index": 0, "embedding": [0.5, 0.5]}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        embedder = ArcEmbedder("https://example.test/v1", "key", "pinned-model", client=client)
        assert embedder.embed_query("third down") == [0.5, 0.5]

    assert seen == ["pinned-model"]


def test_arc_embedder_rejects_a_different_reported_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "other-model", "data": [{"index": 0, "embedding": [1.0]}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        embedder = ArcEmbedder("https://example.test/v1", "key", "pinned-model", client=client)
        with pytest.raises(EmbeddingError, match="pinned-model"):
            embedder.embed_query("third down")


def test_arc_embedder_reports_http_errors_without_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="model unavailable")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        embedder = ArcEmbedder(
            "https://example.test/v1", "secret-key", "pinned-model", client=client
        )
        with pytest.raises(EmbeddingError) as excinfo:
            embedder.embed_query("third down")

    assert "HTTP 500" in str(excinfo.value)
    assert "secret-key" not in str(excinfo.value)


def test_arc_embedder_rejects_malformed_responses() -> None:
    responses = [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"model": "pinned-model", "data": []}),
        httpx.Response(
            200,
            json={"model": "pinned-model", "data": [{"index": 0, "embedding": []}]},
        ),
        httpx.Response(
            200,
            json={"model": "pinned-model", "data": [{"index": 1, "embedding": [1.0]}]},
        ),
    ]

    for response in responses:
        with httpx.Client(transport=httpx.MockTransport(lambda request, r=response: r)) as client:
            embedder = ArcEmbedder("https://example.test/v1", "key", "pinned-model", client=client)
            with pytest.raises(EmbeddingError):
                embedder.embed_query("third down")


@pytest.mark.parametrize(
    ("base_url", "model", "batch_size", "timeout", "message"),
    [
        ("example.test", "m", 1, 1.0, "base_url"),
        ("https://example.test/v1", " ", 1, 1.0, "model"),
        ("https://example.test/v1", "m", 0, 1.0, "batch_size"),
        ("https://example.test/v1", "m", 1, 0.0, "timeout"),
    ],
)
def test_arc_embedder_validates_construction(
    base_url: str,
    model: str,
    batch_size: int,
    timeout: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ArcEmbedder(base_url, "key", model, batch_size=batch_size, timeout=timeout)
