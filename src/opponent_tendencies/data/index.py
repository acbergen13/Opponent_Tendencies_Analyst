"""Persistent ChromaDB index over stored media and verbatim quotes.

The index is derived data. :class:`MediaData` is the source of truth; this module
turns each media item into overlapping chunks (or exact quote units for
post-game transcripts), embeds them with one pinned model, and records the
model ID, chunk parameters, and a content manifest in the Chroma collection
metadata.

Retrieval filters on opponent, season, week range, source type, and unit kind
*before* similarity search. A query from a different embedding model is refused
rather than silently mixing vector spaces.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings

from opponent_tendencies.data.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MediaChunk,
    chunk_media,
)
from opponent_tendencies.data.media_store import MediaData
from opponent_tendencies.llm.embeddings import Embedder
from opponent_tendencies.models import MediaItem, MediaType, Quote, Team

__all__ = [
    "INDEX_SCHEMA_VERSION",
    "IndexBuildReport",
    "IndexConfigurationError",
    "IndexModelMismatchError",
    "IndexUnit",
    "MediaIndex",
    "MediaIndexError",
    "RetrievedSource",
    "build_index_units",
]

INDEX_SCHEMA_VERSION: Final = 1

_METADATA_VALUE = str | int | float | bool
_Metadata = dict[str, _METADATA_VALUE]
_UnitKind = Literal["chunk", "quote"]
_KINDS: Final = frozenset({"chunk", "quote"})


class MediaIndexError(RuntimeError):
    """Base class for vector-index failures."""


class IndexConfigurationError(MediaIndexError):
    """The index cannot be built with the requested settings."""


class IndexModelMismatchError(MediaIndexError):
    """The query embedder is not the model that built the stored vectors."""


@dataclass(frozen=True)
class IndexUnit:
    """One embeddable unit: an article chunk or an exact quote."""

    unit_id: str
    media_id: str
    quote_id: str | None
    text: str
    metadata: Mapping[str, _METADATA_VALUE]


@dataclass(frozen=True)
class IndexBuildReport:
    """What one index build did."""

    collection: str
    model_id: str
    manifest_hash: str
    chunk_size: int
    chunk_overlap: int
    unit_count: int
    embedded_units: int
    reused: bool

    def summary(self) -> str:
        action = "reused" if self.reused else f"rebuilt ({self.embedded_units} embedded)"
        return (
            f"collection {self.collection}: {action}; {self.unit_count} units; "
            f"model {self.model_id}; chunk {self.chunk_size}/{self.chunk_overlap}"
        )


@dataclass(frozen=True)
class RetrievedSource:
    """One filtered, ranked retrieval result."""

    unit_id: str
    media_id: str
    quote_id: str | None
    text: str
    score: float
    metadata: Mapping[str, _METADATA_VALUE]

    @property
    def kind(self) -> str:
        return str(self.metadata.get("unit_kind", ""))

    @property
    def week(self) -> int | None:
        value = self.metadata.get("week")
        if isinstance(value, int) and value > 0:
            return value
        return None


def build_index_units(
    media: MediaData,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[IndexUnit, ...]:
    """Turn stored media and quotes into deterministic embeddable units.

    Coach-quote transcripts with extracted quotes index the exact quote spans
    instead of generic chunks, so a coach-quote retrieval returns the words the
    coach actually said. Other media index overlapping chunks; any quotes they
    contain are also indexed as exact units.
    """

    quotes_by_media: dict[str, list[Quote]] = {}
    for quote in media.quotes:
        quotes_by_media.setdefault(quote.media_id, []).append(quote)

    units: list[IndexUnit] = []
    seen: set[str] = set()
    for item in media.media:
        quotes = sorted(
            quotes_by_media.get(item.media_id, []),
            key=lambda quote: (quote.char_start, quote.quote_id),
        )
        quote_units_only = item.source_type is MediaType.COACH_QUOTE and bool(quotes)
        if not quote_units_only:
            for chunk in chunk_media(item, size=chunk_size, overlap=chunk_overlap):
                unit = _chunk_unit(item, chunk)
                _add_unit(units, seen, unit)
        for quote in quotes:
            _add_unit(units, seen, _quote_unit(item, quote))
    return tuple(units)


def _add_unit(units: list[IndexUnit], seen: set[str], unit: IndexUnit) -> None:
    if not unit.text.strip():
        raise IndexConfigurationError(f"unit {unit.unit_id} has no text")
    if unit.unit_id in seen:
        raise IndexConfigurationError(f"duplicate index unit {unit.unit_id}")
    seen.add(unit.unit_id)
    units.append(unit)


def _chunk_unit(media: MediaItem, chunk: MediaChunk) -> IndexUnit:
    metadata = _base_metadata(media)
    metadata.update(
        {
            "unit_kind": "chunk",
            "chunk_index": chunk.index,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "quote_id": "",
            "speaker": "",
            "speaker_role": "",
            "spoken_on": "",
        }
    )
    return IndexUnit(
        unit_id=chunk.chunk_id,
        media_id=media.media_id,
        quote_id=None,
        text=chunk.text,
        metadata=metadata,
    )


def _quote_unit(media: MediaItem, quote: Quote) -> IndexUnit:
    metadata = _base_metadata(media)
    metadata.update(
        {
            "unit_kind": "quote",
            "chunk_index": -1,
            "char_start": quote.char_start,
            "char_end": quote.char_end,
            "quote_id": quote.quote_id,
            "week": quote.week,
            "game_id": quote.game_id or "",
            "speaker": quote.speaker,
            "speaker_role": quote.speaker_role.value,
            "spoken_on": quote.spoken_on.isoformat(),
        }
    )
    return IndexUnit(
        unit_id=quote.quote_id,
        media_id=media.media_id,
        quote_id=quote.quote_id,
        text=quote.text,
        metadata=metadata,
    )


def _base_metadata(media: MediaItem) -> _Metadata:
    return {
        "media_id": media.media_id,
        "source_id": media.source_id,
        "media_type": media.source_type.value,
        "opponent": media.opponent.value,
        "season": media.season,
        "week": media.week if media.week is not None else 0,
        "game_id": media.game_id or "",
        "title": media.title,
        "author": media.author or "",
        "publisher": media.publisher,
        "published_on": media.published_on.isoformat(),
        "url": str(media.url),
        "content_hash": media.content_hash,
    }


class MediaIndex:
    """A persistent Chroma collection over one season's media store."""

    def __init__(
        self,
        path: Path | str | None,
        embedder: Embedder,
        *,
        collection_name: str = "media",
        client: ClientAPI | None = None,
    ) -> None:
        _validate_collection_name(collection_name)
        self._embedder = embedder
        self._collection_name = collection_name
        if client is not None:
            self._client = client
        elif path is None:
            self._client = chromadb.EphemeralClient(_settings())
        else:
            directory = Path(path)
            directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(directory), settings=_settings())
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def model_id(self) -> str:
        return self._embedder.model_id

    @property
    def count(self) -> int:
        return int(self._collection.count())

    @property
    def metadata(self) -> Mapping[str, Any]:
        return dict(self._collection.metadata or {})

    def build(
        self,
        media: MediaData,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        force: bool = False,
    ) -> IndexBuildReport:
        """Build or reuse the collection for ``media``.

        The collection is reused only when the embedding model, chunk settings,
        schema version, unit count, and content manifest all match. Otherwise it
        is deleted and rebuilt in full, so vectors from two models or two chunk
        policies can never be mixed.
        """

        if chunk_size <= 0:
            raise IndexConfigurationError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise IndexConfigurationError("chunk_overlap must be in [0, chunk_size)")
        units = build_index_units(media, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        manifest = media.index_manifest()

        if not force and self._matches(manifest, chunk_size, chunk_overlap, len(units)):
            return IndexBuildReport(
                collection=self._collection_name,
                model_id=self._embedder.model_id,
                manifest_hash=manifest,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                unit_count=len(units),
                embedded_units=0,
                reused=True,
            )

        if self._collection_name in {item.name for item in self._client.list_collections()}:
            self._client.delete_collection(self._collection_name)
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={
                "hnsw:space": "cosine",
                "schema_version": INDEX_SCHEMA_VERSION,
                "model_id": self._embedder.model_id,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "manifest_hash": manifest,
                "unit_count": len(units),
            },
        )

        embedded = 0
        for batch in _batches(units, 128):
            vectors = self._embedder.embed_documents([unit.text for unit in batch])
            if len(vectors) != len(batch):
                raise MediaIndexError(
                    f"embedder returned {len(vectors)} vectors for {len(batch)} units"
                )
            self._collection.add(
                ids=[unit.unit_id for unit in batch],
                embeddings=cast(list[Sequence[float]], vectors),
                documents=[unit.text for unit in batch],
                metadatas=[dict(unit.metadata) for unit in batch],
            )
            embedded += len(batch)

        return IndexBuildReport(
            collection=self._collection_name,
            model_id=self._embedder.model_id,
            manifest_hash=manifest,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            unit_count=len(units),
            embedded_units=embedded,
            reused=False,
        )

    def _matches(
        self,
        manifest: str,
        chunk_size: int,
        chunk_overlap: int,
        unit_count: int,
    ) -> bool:
        metadata = self._collection.metadata or {}
        if metadata.get("model_id") != self._embedder.model_id:
            return False
        if metadata.get("manifest_hash") != manifest:
            return False
        if metadata.get("schema_version") != INDEX_SCHEMA_VERSION:
            return False
        if _metadata_int(metadata, "chunk_size") != chunk_size:
            return False
        if _metadata_int(metadata, "chunk_overlap") != chunk_overlap:
            return False
        if _metadata_int(metadata, "unit_count") != unit_count:
            return False
        return self.count == unit_count

    def retrieve(
        self,
        query: str,
        *,
        opponent: Team | str,
        season: int,
        weeks: Iterable[int] | None = None,
        min_week: int | None = None,
        max_week: int | None = None,
        source_types: Iterable[MediaType | str] | None = None,
        kinds: Iterable[_UnitKind] | None = None,
        k: int = 8,
    ) -> tuple[RetrievedSource, ...]:
        """Return the top ``k`` filtered sources for ``query``.

        ``opponent`` and ``season`` are required: the index must never retrieve
        another opponent's reporting or another season's articles. ``weeks``
        filters an exact selected set; ``min_week``/``max_week`` filter a range.
        Filtering happens in Chroma before the similarity search.
        """

        if not query.strip():
            raise ValueError("query must not be blank")
        if k <= 0:
            raise ValueError("k must be positive")
        if self.count == 0:
            return ()
        self._require_matching_model()

        clauses: list[dict[str, Any]] = [
            {"opponent": _team_value(opponent)},
            {"season": int(season)},
        ]
        if weeks is not None:
            selected = sorted({int(week) for week in weeks})
            if not selected:
                return ()
            clauses.append({"week": {"$in": selected}})
        else:
            if min_week is not None:
                clauses.append({"week": {"$gte": int(min_week)}})
            if max_week is not None:
                clauses.append({"week": {"$lte": int(max_week)}})
        if source_types is not None:
            selected_types = sorted({_media_type_value(item) for item in source_types})
            if not selected_types:
                return ()
            clauses.append({"media_type": {"$in": selected_types}})
        if kinds is not None:
            selected_kinds = sorted(set(kinds))
            if not selected_kinds:
                return ()
            unknown = [kind for kind in selected_kinds if kind not in _KINDS]
            if unknown:
                raise ValueError(f"unknown unit kind(s): {', '.join(unknown)}")
            clauses.append({"unit_kind": {"$in": selected_kinds}})

        where: dict[str, Any] = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        vector = self._embedder.embed_query(query)
        result = cast(
            Any,
            self._collection.query(
                query_embeddings=cast(list[Sequence[float]], [vector]),
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            ),
        )
        return _parse_query_result(result)

    def _require_matching_model(self) -> None:
        stored = self._collection.metadata or {}
        model = stored.get("model_id")
        if model is None:
            raise IndexModelMismatchError(
                f"collection {self._collection_name!r} has no model_id; rebuild the index"
            )
        if model != self._embedder.model_id:
            raise IndexModelMismatchError(
                f"collection {self._collection_name!r} was built with model {model!r}; "
                f"query embedder is {self._embedder.model_id!r}"
            )


def _parse_query_result(result: Mapping[str, Any]) -> tuple[RetrievedSource, ...]:
    ids = _first(result.get("ids"))
    documents = _first(result.get("documents"))
    metadatas = _first(result.get("metadatas"))
    distances = _first(result.get("distances"))
    sources: list[RetrievedSource] = []
    for position, unit_id in enumerate(ids):
        document = documents[position] if position < len(documents) else ""
        metadata = metadatas[position] if position < len(metadatas) else {}
        distance = distances[position] if position < len(distances) else math.inf
        if not isinstance(metadata, Mapping):
            raise MediaIndexError("retrieval returned non-object metadata")
        quote_id = metadata.get("quote_id")
        sources.append(
            RetrievedSource(
                unit_id=str(unit_id),
                media_id=str(metadata.get("media_id", "")),
                quote_id=str(quote_id) if quote_id else None,
                text=str(document),
                score=1.0 - float(distance),
                metadata=cast(dict[str, _METADATA_VALUE], dict(metadata)),
            )
        )
    return tuple(sources)


def _first(value: object) -> list[Any]:
    if not isinstance(value, list) or not value:
        return []
    first = value[0]
    return first if isinstance(first, list) else []


def _batches[T](items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _settings() -> ChromaSettings:
    return ChromaSettings(anonymized_telemetry=False)


def _validate_collection_name(name: str) -> None:
    if not 3 <= len(name) <= 512:
        raise IndexConfigurationError("collection name must be 3-512 characters")
    if not all(character.isalnum() or character in "._-" for character in name):
        raise IndexConfigurationError(
            "collection name may only contain letters, digits, '.', '_', and '-'"
        )
    if not name[0].isalnum() or not name[-1].isalnum():
        raise IndexConfigurationError("collection name must start and end with a letter or digit")


def _metadata_int(metadata: Mapping[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _team_value(team: Team | str) -> str:
    if isinstance(team, Team):
        return team.value
    try:
        return Team(team.upper()).value
    except ValueError as exc:
        raise ValueError(f"unknown team abbreviation {team!r}") from exc


def _media_type_value(media_type: MediaType | str) -> str:
    if isinstance(media_type, MediaType):
        return media_type.value
    try:
        return MediaType(media_type).value
    except ValueError as exc:
        raise ValueError(f"unknown media type {media_type!r}") from exc
