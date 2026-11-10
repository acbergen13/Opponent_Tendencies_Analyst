"""Shared helpers for the curated media corpus used by Stage 3 tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from opponent_tendencies.data.media_ingest import (
    JsonCorpusConnector,
    MediaCorpus,
    NormalizedMedia,
    normalize_media,
    parse_media_corpus,
)
from opponent_tendencies.data.media_store import MediaData, MediaStore
from opponent_tendencies.models import MediaItem, Quote

SEASON = 2026
REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPO_ROOT / "corpus" / f"media_{SEASON}.json"


def corpus_document() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    return document


def mutated_document(index: int = 0, **changes: object) -> dict[str, Any]:
    """Return a deep copy of the corpus with one item changed."""

    document = copy.deepcopy(corpus_document())
    document["items"][index].update(changes)
    return document


def mutated_quote_document(
    item_index: int = 5,
    quote_index: int = 0,
    **changes: object,
) -> dict[str, Any]:
    document = copy.deepcopy(corpus_document())
    document["items"][item_index]["quotes"][quote_index].update(changes)
    return document


def raw_corpus() -> MediaCorpus:
    return JsonCorpusConnector(CORPUS_PATH).load(SEASON)


def normalized_corpus() -> NormalizedMedia:
    return normalize_media(raw_corpus())


def corpus_from(document: dict[str, Any]) -> MediaCorpus:
    return parse_media_corpus(document, expected_season=SEASON, source="test")


def write_media_store(tmp_path: Path) -> tuple[MediaStore, MediaData]:
    normalized = normalized_corpus()
    store = MediaStore(SEASON, tmp_path / "media" / f"{SEASON}.json")
    store.refresh(normalized.media, normalized.quotes)
    return store, store.load()


def with_media_changes(media: MediaItem, **changes: object) -> MediaItem:
    """Return a revalidated media item, recomputing its derived IDs and hash."""

    data = media.model_dump(mode="json")
    data.pop("media_id")
    data.pop("content_hash")
    data.update(changes)
    return MediaItem.model_validate(data)


def with_quote_changes(quote: Quote, **changes: object) -> Quote:
    """Return a revalidated quote, recomputing its derived quote ID."""

    data = quote.model_dump(mode="json")
    data.pop("quote_id")
    data.update(changes)
    return Quote.model_validate(data)
