"""Chroma index build, reuse, model pinning, and filtered retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from opponent_tendencies.data.index import (
    IndexConfigurationError,
    IndexModelMismatchError,
    MediaIndex,
    build_index_units,
)
from opponent_tendencies.data.media_store import MediaData
from opponent_tendencies.llm.embeddings import HashingEmbedder
from opponent_tendencies.models import MediaType, Team
from tests.media_fixtures import SEASON, with_media_changes, write_media_store

pytestmark = pytest.mark.integration

COLLECTION = "media_test"


class SpyEmbedder(HashingEmbedder):
    """Hashing embedder that counts how much work the index actually does."""

    def __init__(self, dimensions: int = 256) -> None:
        super().__init__(dimensions)
        self.document_calls = 0
        self.embedded_texts = 0
        self.query_calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        self.embedded_texts += len(texts)
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return super().embed_query(text)


def index_for(tmp_path: Path, embedder: SpyEmbedder | None = None) -> MediaIndex:
    return MediaIndex(
        tmp_path / "chroma",
        embedder or SpyEmbedder(),
        collection_name=COLLECTION,
    )


def test_units_cover_chunks_and_exact_quotes(tmp_path: Path) -> None:
    _, state = write_media_store(tmp_path)

    units = build_index_units(state, chunk_size=300, chunk_overlap=50)

    assert units
    assert len({unit.unit_id for unit in units}) == len(units)
    assert units == build_index_units(state, chunk_size=300, chunk_overlap=50)

    media_by_id = {item.media_id: item for item in state.media}
    quote_units = [unit for unit in units if unit.quote_id is not None]
    assert len(quote_units) == len(state.quotes)
    for unit in quote_units:
        quote = state.quote_by_id(unit.quote_id or "")
        assert quote is not None
        assert unit.text == quote.text
        assert unit.metadata["speaker"] == quote.speaker
        assert unit.metadata["week"] == quote.week

    for unit in units:
        if unit.quote_id is None:
            media = media_by_id[unit.media_id]
            start = int(unit.metadata["char_start"])
            end = int(unit.metadata["char_end"])
            assert unit.text == media.body[start:end]
            assert len(unit.text) <= 300


def test_build_records_model_manifest_and_metadata(tmp_path: Path) -> None:
    _, state = write_media_store(tmp_path)
    embedder = SpyEmbedder()
    index = index_for(tmp_path, embedder)

    report = index.build(state, chunk_size=300, chunk_overlap=50)

    assert report.reused is False
    assert report.model_id == embedder.model_id
    assert report.manifest_hash == state.index_manifest()
    assert report.unit_count == index.count
    assert report.embedded_units == report.unit_count
    assert index.metadata["model_id"] == embedder.model_id
    assert index.metadata["chunk_size"] == 300
    assert index.metadata["chunk_overlap"] == 50
    assert index.metadata["manifest_hash"] == state.index_manifest()
    assert index.metadata["unit_count"] == report.unit_count


def test_second_build_reuses_the_index_without_embedding(tmp_path: Path) -> None:
    _, state = write_media_store(tmp_path)
    embedder = SpyEmbedder()
    index = index_for(tmp_path, embedder)

    first = index.build(state, chunk_size=300, chunk_overlap=50)
    calls_after_first = embedder.document_calls
    embedded_after_first = embedder.embedded_texts

    second = index.build(state, chunk_size=300, chunk_overlap=50)

    assert first.reused is False
    assert second.reused is True
    assert second.embedded_units == 0
    assert embedder.document_calls == calls_after_first
    assert embedder.embedded_texts == embedded_after_first
    assert index.count == first.unit_count


def test_content_change_forces_a_rebuild(tmp_path: Path) -> None:
    store, state = write_media_store(tmp_path)
    embedder = SpyEmbedder()
    index = index_for(tmp_path, embedder)
    index.build(state, chunk_size=300, chunk_overlap=50)
    embedded_before = embedder.embedded_texts

    target = next(item for item in state.media if not state.quotes_for(opponent=item.opponent))
    changed = with_media_changes(
        target, body=target.body + "\n\nA new paragraph changes the index."
    )
    store.refresh((changed,), ())
    report = index.build(store.load(), chunk_size=300, chunk_overlap=50)

    assert report.reused is False
    assert report.manifest_hash != state.index_manifest()
    assert embedder.embedded_texts > embedded_before


def test_chunk_setting_change_forces_a_rebuild(tmp_path: Path) -> None:
    _, state = write_media_store(tmp_path)
    index = index_for(tmp_path)
    index.build(state, chunk_size=1200, chunk_overlap=200)

    report = index.build(state, chunk_size=300, chunk_overlap=50)

    assert report.reused is False
    assert index.metadata["chunk_size"] == 300
    assert index.metadata["chunk_overlap"] == 50


def test_force_rebuild_embeds_even_when_manifest_matches(tmp_path: Path) -> None:
    _, state = write_media_store(tmp_path)
    embedder = SpyEmbedder()
    index = index_for(tmp_path, embedder)
    first = index.build(state, chunk_size=300, chunk_overlap=50)

    report = index.build(state, chunk_size=300, chunk_overlap=50, force=True)

    assert report.reused is False
    assert report.embedded_units == first.unit_count


def test_index_persists_and_reuses_across_clients(tmp_path: Path) -> None:
    _, state = write_media_store(tmp_path)
    first_index = index_for(tmp_path)
    first = first_index.build(state, chunk_size=300, chunk_overlap=50)

    second_index = index_for(tmp_path)
    second = second_index.build(state, chunk_size=300, chunk_overlap=50)

    assert second.reused is True
    assert second_index.count == first.unit_count
    hits = second_index.retrieve(
        "third down",
        opponent=Team.BUF,
        season=SEASON,
        weeks=[8],
        source_types=[MediaType.ARTICLE],
    )
    assert hits


def test_retrieval_filters_opponent_week_and_source_type(tmp_path: Path) -> None:
    _, state = write_media_store(tmp_path)
    index = index_for(tmp_path)
    index.build(state, chunk_size=300, chunk_overlap=50)

    hits = index.retrieve(
        "third down passing plan",
        opponent="BUF",
        season=SEASON,
        weeks=[8],
        source_types=[MediaType.ARTICLE],
        k=5,
    )

    assert hits
    for hit in hits:
        assert hit.metadata["opponent"] == "BUF"
        assert hit.metadata["season"] == SEASON
        assert hit.metadata["week"] == 8
        assert hit.metadata["media_type"] == "article"
        assert hit.kind == "chunk"
        assert hit.week == 8


def test_retrieval_returns_exact_coach_quotes(tmp_path: Path) -> None:
    _, state = write_media_store(tmp_path)
    index = index_for(tmp_path)
    index.build(state, chunk_size=300, chunk_overlap=50)

    hits = index.retrieve(
        "pressure check down",
        opponent=Team.BUF,
        season=SEASON,
        weeks=[9],
        source_types=[MediaType.COACH_QUOTE],
        k=5,
    )

    assert hits
    for hit in hits:
        assert hit.kind == "quote"
        assert hit.quote_id is not None
        assert hit.media_id
        assert hit.week == 9
        assert hit.metadata["media_type"] == "coach_quote"
        assert hit.metadata["speaker"]
        quote = state.quote_by_id(hit.quote_id)
        assert quote is not None
        assert hit.text == quote.text


def test_retrieval_never_returns_another_opponent(tmp_path: Path) -> None:
    _, state = write_media_store(tmp_path)
    index = index_for(tmp_path)
    index.build(state, chunk_size=300, chunk_overlap=50)

    hits = index.retrieve(
        "red zone motion",
        opponent=Team.KC,
        season=SEASON,
        weeks=[6, 8],
        k=10,
    )

    assert hits
    assert {hit.metadata["opponent"] for hit in hits} == {"KC"}


def test_retrieval_week_range_and_undated_background(tmp_path: Path) -> None:
    _, state = write_media_store(tmp_path)
    index = index_for(tmp_path)
    index.build(state, chunk_size=300, chunk_overlap=50)

    ranged = index.retrieve(
        "red zone",
        opponent=Team.BUF,
        season=SEASON,
        min_week=8,
        max_week=9,
        k=10,
    )
    explicit = index.retrieve(
        "red zone",
        opponent=Team.BUF,
        season=SEASON,
        weeks=[2],
        k=10,
    )

    assert ranged
    assert all(hit.week is not None and 8 <= hit.week <= 9 for hit in ranged)
    assert explicit
    assert all(hit.week == 2 for hit in explicit)


def test_retrieval_rejects_a_different_embedding_model(tmp_path: Path) -> None:
    _, state = write_media_store(tmp_path)
    index_for(tmp_path, SpyEmbedder(256)).build(state, chunk_size=300, chunk_overlap=50)
    other = index_for(tmp_path, SpyEmbedder(512))

    with pytest.raises(IndexModelMismatchError, match="built with model"):
        other.retrieve("third down", opponent=Team.BUF, season=SEASON, weeks=[8])


def test_empty_index_returns_no_results(tmp_path: Path) -> None:
    embedder = SpyEmbedder()
    index = index_for(tmp_path, embedder)
    index.build(MediaData(season=SEASON))

    assert index.count == 0
    assert index.retrieve("anything", opponent=Team.BUF, season=SEASON, weeks=[1]) == ()
    assert embedder.query_calls == 0


def test_retrieval_argument_validation(tmp_path: Path) -> None:
    _, state = write_media_store(tmp_path)
    index = index_for(tmp_path)
    index.build(state, chunk_size=300, chunk_overlap=50)

    with pytest.raises(ValueError, match="blank"):
        index.retrieve("   ", opponent=Team.BUF, season=SEASON)
    with pytest.raises(ValueError, match="positive"):
        index.retrieve("run", opponent=Team.BUF, season=SEASON, k=0)
    with pytest.raises(ValueError, match="unknown team"):
        index.retrieve("run", opponent="XYZ", season=SEASON)
    with pytest.raises(ValueError, match="unknown unit kind"):
        index.retrieve(
            "run",
            opponent=Team.BUF,
            season=SEASON,
            kinds=["paragraph"],  # type: ignore[list-item]
        )


def test_empty_filter_sets_return_nothing(tmp_path: Path) -> None:
    _, state = write_media_store(tmp_path)
    index = index_for(tmp_path)
    index.build(state, chunk_size=300, chunk_overlap=50)

    assert index.retrieve("run", opponent=Team.BUF, season=SEASON, weeks=[]) == ()
    assert index.retrieve("run", opponent=Team.BUF, season=SEASON, source_types=[]) == ()


def test_bad_chunk_settings_are_rejected(tmp_path: Path) -> None:
    _, state = write_media_store(tmp_path)
    index = index_for(tmp_path)

    with pytest.raises(IndexConfigurationError, match="chunk_size"):
        index.build(state, chunk_size=0)
    with pytest.raises(IndexConfigurationError, match="chunk_overlap"):
        index.build(state, chunk_size=100, chunk_overlap=100)


def test_collection_name_is_validated(tmp_path: Path) -> None:
    with pytest.raises(IndexConfigurationError, match="3-512"):
        MediaIndex(tmp_path / "chroma", HashingEmbedder(), collection_name="ab")
    with pytest.raises(IndexConfigurationError, match="may only contain"):
        MediaIndex(tmp_path / "chroma", HashingEmbedder(), collection_name="bad name")


def test_store_media_quotes_and_index_units_stay_in_sync(tmp_path: Path) -> None:
    store, state = write_media_store(tmp_path)
    index = index_for(tmp_path)
    report = index.build(state, chunk_size=300, chunk_overlap=50)

    units = build_index_units(state, chunk_size=300, chunk_overlap=50)
    assert report.unit_count == len(units)
    assert {unit.media_id for unit in units} == {item.media_id for item in state.media}
    assert all(unit.text for unit in units)

    # Re-reading the store must not change the derived units.
    assert build_index_units(store.load(), chunk_size=300, chunk_overlap=50) == units
