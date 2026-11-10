"""Atomic JSON store for one season's media items and verbatim quotes.

The store is the queryable metadata layer for Stage 3. It holds the full article
or transcript body, the quote spans extracted from it, and the allowlisted
source metadata. Chroma holds vectors only; it is always rebuildable from this
store.

Quotes are replaced as a set per media item: a refresh supplies the complete
quote list for every media item it includes, so correcting a speaker or a span
cannot leave a stale quote behind.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ValidationError

from opponent_tendencies.config import resolve_data_dir
from opponent_tendencies.data.jsonio import atomic_write_text, canonical_dump
from opponent_tendencies.models import (
    TEAM_DISPLAY_NAMES,
    MediaItem,
    MediaType,
    Quote,
    Team,
)

__all__ = [
    "SCHEMA_VERSION",
    "MediaData",
    "MediaRefreshResult",
    "MediaStore",
    "MediaStoreCorruptionError",
    "MediaStoreError",
    "MediaStoreIntegrityError",
    "default_media_store_path",
]

SCHEMA_VERSION: Final = 1

_STORE_DIR: Final = "media"


class MediaStoreError(RuntimeError):
    """Base class for media-store failures."""


class MediaStoreCorruptionError(MediaStoreError):
    """The file on disk is not a media store this version can read."""


class MediaStoreIntegrityError(MediaStoreError):
    """Records are internally inconsistent and must not be written."""


@dataclass(frozen=True)
class MediaRefreshResult:
    """Inserted, updated, skipped, and removed counts for one media refresh."""

    inserted_media: int = 0
    updated_media: int = 0
    skipped_media: int = 0
    inserted_quotes: int = 0
    updated_quotes: int = 0
    skipped_quotes: int = 0
    removed_quotes: int = 0

    @property
    def inserted(self) -> int:
        return self.inserted_media + self.inserted_quotes

    @property
    def updated(self) -> int:
        return self.updated_media + self.updated_quotes

    @property
    def skipped(self) -> int:
        return self.skipped_media + self.skipped_quotes

    @property
    def removed(self) -> int:
        return self.removed_quotes

    @property
    def changed(self) -> bool:
        return bool(self.inserted or self.updated or self.removed)

    def __str__(self) -> str:
        return (
            f"{self.inserted} inserted, {self.updated} updated, "
            f"{self.skipped} skipped, {self.removed} removed"
        )


@dataclass(frozen=True)
class MediaData:
    """A read-only view of one season's stored media and quotes."""

    season: int
    media: tuple[MediaItem, ...] = ()
    quotes: tuple[Quote, ...] = ()

    def media_by_id(self, media_id: str) -> MediaItem | None:
        return next((item for item in self.media if item.media_id == media_id), None)

    def quote_by_id(self, quote_id: str) -> Quote | None:
        return next((quote for quote in self.quotes if quote.quote_id == quote_id), None)

    @property
    def opponents(self) -> tuple[Team, ...]:
        """Teams with at least one stored media item, ordered by display name."""

        present = {item.opponent for item in self.media}
        return tuple(sorted(present, key=TEAM_DISPLAY_NAMES.__getitem__))

    @property
    def weeks(self) -> tuple[int, ...]:
        """Sorted weeks that have at least one stored media item."""

        return tuple(sorted({item.week for item in self.media if item.week is not None}))

    @property
    def source_types(self) -> tuple[MediaType, ...]:
        return tuple(sorted({item.source_type for item in self.media}, key=lambda item: item.value))

    def media_for(
        self,
        *,
        opponent: Team | None = None,
        weeks: Iterable[int] | None = None,
        source_types: Iterable[MediaType] | None = None,
    ) -> tuple[MediaItem, ...]:
        """Filter stored media by opponent, weeks, and source type."""

        wanted_weeks = None if weeks is None else {int(week) for week in weeks}
        wanted_types = None if source_types is None else set(source_types)
        return tuple(
            item
            for item in self.media
            if (opponent is None or item.opponent is opponent)
            and (wanted_weeks is None or (item.week is not None and item.week in wanted_weeks))
            and (wanted_types is None or item.source_type in wanted_types)
        )

    def quotes_for(
        self,
        *,
        opponent: Team | None = None,
        weeks: Iterable[int] | None = None,
    ) -> tuple[Quote, ...]:
        """Filter stored quotes by opponent team and quote week."""

        wanted_weeks = None if weeks is None else {int(week) for week in weeks}
        return tuple(
            quote
            for quote in self.quotes
            if (opponent is None or quote.team is opponent)
            and (wanted_weeks is None or quote.week in wanted_weeks)
        )

    @property
    def revision(self) -> str:
        """Content hash of the full canonical payload."""

        payload = canonical_dump(_payload(self.season, self.media, self.quotes))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def index_manifest(self) -> str:
        """Hash of everything the vector index derives from this store.

        ``retrieved_at`` is excluded: re-fetching identical text must not force
        an index rebuild. Metadata that shapes filters or chunks is included, so
        moving an article to another week does force a rebuild.
        """

        media_payload = [
            {
                key: value
                for key, value in item.model_dump(mode="json").items()
                if key != "retrieved_at"
            }
            for item in self.media
        ]
        quote_payload = [quote.model_dump(mode="json") for quote in self.quotes]
        payload = canonical_dump({"media": media_payload, "quotes": quote_payload})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_media_store_path(season: int, data_dir: Path | None = None) -> Path:
    """Return ``<data_dir>/media/<season>.json``."""

    base = data_dir if data_dir is not None else resolve_data_dir()
    return base / _STORE_DIR / f"{int(season)}.json"


class MediaStore:
    """Read and atomically upsert one season's media items and quotes."""

    def __init__(self, season: int, path: Path | None = None) -> None:
        if int(season) < 1920:
            raise ValueError(f"implausible season: {season}")
        self.season = int(season)
        self.path = path if path is not None else default_media_store_path(self.season)

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> MediaData:
        """Read the store, returning an empty view when the file does not exist."""

        if not self.path.exists():
            return MediaData(season=self.season)
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise MediaStoreCorruptionError(f"cannot read store {self.path}: {exc}") from exc
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MediaStoreCorruptionError(
                f"store {self.path} is not valid JSON ({exc.msg} at line {exc.lineno})"
            ) from exc
        if not isinstance(raw, dict):
            raise MediaStoreCorruptionError(f"store {self.path} must contain a JSON object")

        version = raw.get("schema_version")
        if version != SCHEMA_VERSION:
            raise MediaStoreCorruptionError(
                f"store {self.path} has schema_version {version!r}; "
                f"expected {SCHEMA_VERSION}. Delete the file and re-ingest."
            )
        if raw.get("season") != self.season:
            raise MediaStoreCorruptionError(
                f"store {self.path} holds season {raw.get('season')!r}; expected {self.season}"
            )

        media = _validate_records(MediaItem, raw.get("media"), "media", self.path)
        quotes = _validate_records(Quote, raw.get("quotes"), "quotes", self.path)
        try:
            _check_links(media, quotes)
        except MediaStoreIntegrityError as exc:
            raise MediaStoreCorruptionError(f"store {self.path}: {exc}") from exc

        return MediaData(season=self.season, media=media, quotes=quotes)

    def refresh(
        self,
        media: Iterable[MediaItem],
        quotes: Iterable[Quote],
    ) -> MediaRefreshResult:
        """Upsert media and replace the quote set of every included media item."""

        incoming_media = _index_media(media, expected_season=self.season)
        incoming_quotes = _index_quotes(quotes, expected_season=self.season)
        state = self.load()

        existing_media = {item.media_id: item for item in state.media}
        existing_quotes = {quote.quote_id: quote for quote in state.quotes}

        removed_quotes = 0
        for media_id in incoming_media:
            keep = {
                quote_id
                for quote_id, quote in incoming_quotes.items()
                if quote.media_id == media_id
            }
            for quote_id in [
                quote_id
                for quote_id, quote in existing_quotes.items()
                if quote.media_id == media_id and quote_id not in keep
            ]:
                del existing_quotes[quote_id]
                removed_quotes += 1

        media_counts = _merge(existing_media, incoming_media)
        quote_counts = _merge(existing_quotes, incoming_quotes)
        _check_links(tuple(existing_media.values()), tuple(existing_quotes.values()))

        result = MediaRefreshResult(
            inserted_media=media_counts.inserted,
            updated_media=media_counts.updated,
            skipped_media=media_counts.skipped,
            inserted_quotes=quote_counts.inserted,
            updated_quotes=quote_counts.updated,
            skipped_quotes=quote_counts.skipped,
            removed_quotes=removed_quotes,
        )
        if result.changed:
            self._write(tuple(existing_media.values()), tuple(existing_quotes.values()))
        return result

    def _write(self, media: Sequence[MediaItem], quotes: Sequence[Quote]) -> None:
        payload = _payload(self.season, media, quotes)
        atomic_write_text(self.path, canonical_dump(payload))


@dataclass(frozen=True)
class _UpsertCounts:
    inserted: int = 0
    updated: int = 0
    skipped: int = 0


def _merge[Model: BaseModel](
    existing: dict[str, Model],
    incoming: Mapping[str, Model],
) -> _UpsertCounts:
    inserted = updated = skipped = 0
    for key, record in incoming.items():
        previous = existing.get(key)
        if previous is None:
            inserted += 1
        elif previous == record:
            skipped += 1
        else:
            updated += 1
        existing[key] = record
    return _UpsertCounts(inserted=inserted, updated=updated, skipped=skipped)


def _index_media(media: Iterable[MediaItem], *, expected_season: int) -> dict[str, MediaItem]:
    indexed: dict[str, MediaItem] = {}
    for item in media:
        if item.season != expected_season:
            raise MediaStoreIntegrityError(
                f"media {item.media_id} is season {item.season}; store holds {expected_season}"
            )
        if item.media_id in indexed:
            raise MediaStoreIntegrityError(f"duplicate media {item.media_id!r} in one refresh")
        indexed[item.media_id] = item
    return indexed


def _index_quotes(quotes: Iterable[Quote], *, expected_season: int) -> dict[str, Quote]:
    indexed: dict[str, Quote] = {}
    for quote in quotes:
        if quote.season != expected_season:
            raise MediaStoreIntegrityError(
                f"quote {quote.quote_id} is season {quote.season}; store holds {expected_season}"
            )
        if quote.quote_id in indexed:
            raise MediaStoreIntegrityError(f"duplicate quote {quote.quote_id!r} in one refresh")
        indexed[quote.quote_id] = quote
    return indexed


def _check_links(media: Sequence[MediaItem], quotes: Sequence[Quote]) -> None:
    by_id: dict[str, MediaItem] = {}
    for item in media:
        if item.media_id in by_id:
            raise MediaStoreIntegrityError(f"duplicate media {item.media_id!r}")
        by_id[item.media_id] = item

    seen: set[str] = set()
    for quote in quotes:
        if quote.quote_id in seen:
            raise MediaStoreIntegrityError(f"duplicate quote {quote.quote_id!r}")
        seen.add(quote.quote_id)
        linked = by_id.get(quote.media_id)
        if linked is None:
            raise MediaStoreIntegrityError(
                f"quote {quote.quote_id} references unknown media {quote.media_id!r}; "
                "ingest the article with its quotes"
            )
        if quote.season != linked.season:
            raise MediaStoreIntegrityError(
                f"quote {quote.quote_id} is season {quote.season} but media "
                f"{linked.media_id} is season {linked.season}"
            )
        if quote.team != linked.opponent:
            raise MediaStoreIntegrityError(
                f"quote {quote.quote_id} is about {quote.team.value} but media "
                f"{linked.media_id} covers {linked.opponent.value}"
            )
        if quote.publisher != linked.publisher:
            raise MediaStoreIntegrityError(
                f"quote {quote.quote_id} publisher does not match media {linked.media_id}"
            )
        if quote.url != linked.url:
            raise MediaStoreIntegrityError(
                f"quote {quote.quote_id} URL does not match media {linked.media_id}"
            )
        if not quote.is_verbatim_in(linked):
            raise MediaStoreIntegrityError(
                f"quote {quote.quote_id} span does not reproduce its text verbatim in "
                f"media {linked.media_id}"
            )


def _payload(
    season: int,
    media: Iterable[MediaItem],
    quotes: Iterable[Quote],
) -> dict[str, object]:
    ordered_media = sorted(media, key=lambda item: (item.published_on, item.media_id))
    ordered_quotes = sorted(
        quotes,
        key=lambda quote: (quote.media_id, quote.char_start, quote.quote_id),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "season": season,
        "media": [item.model_dump(mode="json") for item in ordered_media],
        "quotes": [quote.model_dump(mode="json") for quote in ordered_quotes],
    }


def _validate_records[Model: BaseModel](
    model: type[Model],
    raw: object,
    kind: str,
    path: Path,
) -> tuple[Model, ...]:
    if not isinstance(raw, list):
        raise MediaStoreCorruptionError(f"store {path}: {kind!r} must be a list")
    records: list[Model] = []
    for index, item in enumerate(raw):
        try:
            records.append(model.model_validate(item))
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'record'}: {error['msg']}"
                for error in exc.errors()
            )
            raise MediaStoreCorruptionError(
                f"store {path}: {kind}[{index}] is invalid: {detail}"
            ) from exc
    return tuple(records)
