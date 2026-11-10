"""Media corpus ingestion: allowlist, connector interface, and normalization.

The application never scrapes arbitrary URLs. A media item is accepted only
when its ``source_id`` is on the source allowlist, its URL belongs to that
source's domains, its type is permitted for the source, and its author is a
trusted reporter for that source when the allowlist names one.

The first connector reads a curated JSON corpus. Live connectors can implement
the same :class:`MediaConnector` protocol later, but only for sources whose
terms permit it; nothing in this module fetches the network.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, Protocol

from pydantic import HttpUrl, ValidationError

from opponent_tendencies.models import (
    MediaItem,
    MediaType,
    Quote,
    SpeakerRole,
    Team,
)

__all__ = [
    "SOURCE_ALLOWLIST",
    "AllowedSource",
    "JsonCorpusConnector",
    "MediaConnector",
    "MediaCorpus",
    "MediaCorpusError",
    "MediaIngestionError",
    "MediaRefreshReport",
    "NormalizedMedia",
    "QuoteExtractionError",
    "SourcePolicyError",
    "host_is_allowlisted",
    "ingest_media",
    "normalize_media",
    "parse_media_corpus",
]

CORPUS_SCHEMA_VERSION: Final = 1


class MediaIngestionError(RuntimeError):
    """Base class for rejected media input."""


class MediaCorpusError(MediaIngestionError):
    """The corpus document is malformed or does not match the requested season."""


class SourcePolicyError(MediaIngestionError):
    """A source, URL, media type, or author is not on the allowlist."""


class QuoteExtractionError(MediaIngestionError):
    """A quote cannot be reproduced verbatim from its stored article body."""


@dataclass(frozen=True)
class AllowedSource:
    """One trusted publisher and the reporters whose work we accept."""

    source_id: str
    publisher: str
    domains: frozenset[str]
    media_types: frozenset[MediaType]
    trusted_authors: frozenset[str] = frozenset()


#: Curated allowlist. Extend deliberately: each entry is a claim that the
#: publisher and named reporters are acceptable evidence sources.
SOURCE_ALLOWLIST: Final[dict[str, AllowedSource]] = {
    "ap-nfl": AllowedSource(
        source_id="ap-nfl",
        publisher="Associated Press",
        domains=frozenset({"apnews.com"}),
        media_types=frozenset({MediaType.ARTICLE, MediaType.COACH_QUOTE}),
        trusted_authors=frozenset({"Renee Salas"}),
    ),
    "buffalo-news": AllowedSource(
        source_id="buffalo-news",
        publisher="The Buffalo News",
        domains=frozenset({"buffalonews.com"}),
        media_types=frozenset({MediaType.ARTICLE, MediaType.COACH_QUOTE}),
        trusted_authors=frozenset({"Jordan Ellis"}),
    ),
    "espn-nfl": AllowedSource(
        source_id="espn-nfl",
        publisher="ESPN",
        domains=frozenset({"espn.com"}),
        media_types=frozenset({MediaType.ARTICLE, MediaType.FILM_BREAKDOWN, MediaType.COACH_QUOTE}),
        trusted_authors=frozenset({"Chris Okafor", "Maya Chen"}),
    ),
    "kansas-city-star": AllowedSource(
        source_id="kansas-city-star",
        publisher="The Kansas City Star",
        domains=frozenset({"kansascity.com"}),
        media_types=frozenset({MediaType.ARTICLE, MediaType.COACH_QUOTE}),
        trusted_authors=frozenset({"Dana Whitfield"}),
    ),
    "nbc-sports": AllowedSource(
        source_id="nbc-sports",
        publisher="NBC Sports",
        domains=frozenset({"nbcsports.com"}),
        media_types=frozenset({MediaType.ARTICLE, MediaType.FILM_BREAKDOWN, MediaType.COACH_QUOTE}),
        trusted_authors=frozenset({"Tomas Reyes"}),
    ),
    "nfl-com": AllowedSource(
        source_id="nfl-com",
        publisher="NFL.com",
        domains=frozenset({"nfl.com"}),
        media_types=frozenset({MediaType.ARTICLE, MediaType.FILM_BREAKDOWN, MediaType.COACH_QUOTE}),
        trusted_authors=frozenset({"Mara Quinn"}),
    ),
    "si-nfl": AllowedSource(
        source_id="si-nfl",
        publisher="Sports Illustrated",
        domains=frozenset({"si.com"}),
        media_types=frozenset({MediaType.ARTICLE, MediaType.FILM_BREAKDOWN}),
        trusted_authors=frozenset({"Priya Nair"}),
    ),
    "the-athletic": AllowedSource(
        source_id="the-athletic",
        publisher="The Athletic",
        domains=frozenset({"theathletic.com"}),
        media_types=frozenset({MediaType.ARTICLE, MediaType.FILM_BREAKDOWN}),
        trusted_authors=frozenset({"Elliot Park"}),
    ),
}


@dataclass(frozen=True)
class RawQuote:
    """A quote as it appears in the corpus, before its span is resolved."""

    speaker: str
    speaker_role: SpeakerRole
    text: str
    occurrence: int = 0
    team: Team | None = None
    spoken_on: date | None = None
    week: int | None = None
    game_id: str | None = None
    char_start: int | None = None
    char_end: int | None = None


@dataclass(frozen=True)
class RawMediaItem:
    """One corpus item, parsed but not yet policy-checked or normalized."""

    source_id: str
    source_type: MediaType
    title: str
    url: HttpUrl
    published_on: date
    season: int
    opponent: Team
    body: str
    author: str | None = None
    publisher: str | None = None
    week: int | None = None
    game_id: str | None = None
    quotes: tuple[RawQuote, ...] = ()


@dataclass(frozen=True)
class MediaCorpus:
    """One season's curated media corpus."""

    season: int
    retrieved_at: datetime
    items: tuple[RawMediaItem, ...] = ()


class MediaConnector(Protocol):
    """A source of media items for one season."""

    def load(self, season: int) -> MediaCorpus:
        """Return the corpus for ``season``."""


@dataclass(frozen=True)
class NormalizedMedia:
    """Contracts built from a corpus, sorted for canonical storage."""

    media: tuple[MediaItem, ...] = ()
    quotes: tuple[Quote, ...] = ()


@dataclass(frozen=True)
class MediaRefreshReport:
    """What one media refresh fetched, normalized, and wrote."""

    season: int
    normalized: NormalizedMedia
    refresh: Any = field(repr=False)

    @property
    def media_count(self) -> int:
        return len(self.normalized.media)

    @property
    def quote_count(self) -> int:
        return len(self.normalized.quotes)

    def summary(self) -> str:
        return (
            f"season {self.season}: {self.media_count} media items, "
            f"{self.quote_count} quotes; {self.refresh}"
        )


class JsonCorpusConnector:
    """Load a curated JSON corpus from disk."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load(self, season: int) -> MediaCorpus:
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise MediaCorpusError(f"cannot read media corpus {self.path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise MediaCorpusError(
                f"media corpus {self.path} is not valid JSON ({exc.msg} at line {exc.lineno})"
            ) from exc
        return parse_media_corpus(document, expected_season=season, source=str(self.path))


def parse_media_corpus(
    document: object,
    *,
    expected_season: int,
    source: str = "<corpus>",
) -> MediaCorpus:
    """Validate a corpus document and parse it into raw records."""

    where = f"media corpus {source}"
    root = _mapping(document, where)
    _reject_unknown(
        root,
        {"schema_version", "season", "description", "retrieved_at", "items"},
        where,
    )

    version = _required_int(root, "schema_version", where)
    if version != CORPUS_SCHEMA_VERSION:
        raise MediaCorpusError(
            f"{where} has schema_version {version}; expected {CORPUS_SCHEMA_VERSION}"
        )

    season = _required_int(root, "season", where)
    if season != expected_season:
        raise MediaCorpusError(f"{where} holds season {season}; expected {expected_season}")

    retrieved_at = _required_datetime(root, "retrieved_at", where)
    raw_items = root.get("items")
    if not isinstance(raw_items, list):
        raise MediaCorpusError(f"{where}: 'items' must be a list")

    items = tuple(
        _parse_item(item, season=season, where=f"{where} items[{index}]")
        for index, item in enumerate(raw_items)
    )
    return MediaCorpus(season=season, retrieved_at=retrieved_at, items=items)


def ingest_media(
    connector: MediaConnector,
    store: Any,
    *,
    allowlist: Mapping[str, AllowedSource] | None = None,
) -> MediaRefreshReport:
    """Load, normalize, and atomically upsert one season of media."""

    corpus = connector.load(store.season)
    normalized = normalize_media(corpus, allowlist=allowlist)
    result = store.refresh(normalized.media, normalized.quotes)
    return MediaRefreshReport(season=store.season, normalized=normalized, refresh=result)


def normalize_media(
    corpus: MediaCorpus,
    *,
    allowlist: Mapping[str, AllowedSource] | None = None,
) -> NormalizedMedia:
    """Apply the source policy and build the frozen media/quote contracts."""

    sources = SOURCE_ALLOWLIST if allowlist is None else allowlist
    media_items: dict[str, MediaItem] = {}
    quotes: dict[str, Quote] = {}
    for index, raw in enumerate(corpus.items):
        label = f"corpus item[{index}] ({raw.source_id})"
        if raw.season != corpus.season:
            raise MediaCorpusError(
                f"{label}: item season {raw.season} does not match corpus season {corpus.season}"
            )
        allowed = _allowed_source(raw.source_id, sources, label)
        _check_publisher(raw, allowed, label)
        _check_media_type(raw, allowed, label)
        _check_author(raw, allowed, label)
        _check_url(raw, allowed, label)

        media = _build_media(raw, corpus, allowed, label)
        previous = media_items.get(media.media_id)
        if previous is not None:
            raise MediaIngestionError(
                f"{label}: duplicate media identity {media.media_id}; "
                "source_id + url must be unique"
            )
        media_items[media.media_id] = media

        for quote_index, raw_quote in enumerate(raw.quotes):
            quote = _build_quote(
                raw_quote,
                media,
                label=f"{label} quote[{quote_index}]",
            )
            if quote.quote_id in quotes:
                raise MediaIngestionError(
                    f"{label} quote[{quote_index}]: duplicate quote identity {quote.quote_id}"
                )
            quotes[quote.quote_id] = quote

    ordered_media = tuple(
        sorted(media_items.values(), key=lambda item: (item.published_on, item.media_id))
    )
    ordered_quotes = tuple(
        sorted(
            quotes.values(),
            key=lambda quote: (quote.media_id, quote.char_start, quote.quote_id),
        )
    )
    return NormalizedMedia(media=ordered_media, quotes=ordered_quotes)


def _allowed_source(
    source_id: str,
    allowlist: Mapping[str, AllowedSource],
    label: str,
) -> AllowedSource:
    try:
        return allowlist[source_id]
    except KeyError as exc:
        known = ", ".join(sorted(allowlist))
        raise SourcePolicyError(
            f"{label}: source_id {source_id!r} is not on the allowlist ({known})"
        ) from exc


def _check_publisher(raw: RawMediaItem, allowed: AllowedSource, label: str) -> None:
    if raw.publisher is not None and raw.publisher != allowed.publisher:
        raise SourcePolicyError(
            f"{label}: publisher {raw.publisher!r} does not match allowlist "
            f"publisher {allowed.publisher!r} for {allowed.source_id!r}"
        )


def _check_media_type(raw: RawMediaItem, allowed: AllowedSource, label: str) -> None:
    if raw.source_type not in allowed.media_types:
        permitted = ", ".join(sorted(item.value for item in allowed.media_types))
        raise SourcePolicyError(
            f"{label}: {raw.source_type.value!r} is not permitted for {allowed.source_id!r} "
            f"(allowed: {permitted})"
        )


def _check_author(raw: RawMediaItem, allowed: AllowedSource, label: str) -> None:
    if not allowed.trusted_authors:
        return
    if raw.author is None or raw.author not in allowed.trusted_authors:
        trusted = ", ".join(sorted(allowed.trusted_authors))
        raise SourcePolicyError(
            f"{label}: author {raw.author!r} is not a trusted reporter for "
            f"{allowed.source_id!r} (trusted: {trusted})"
        )


def _check_url(raw: RawMediaItem, allowed: AllowedSource, label: str) -> None:
    if raw.url.scheme != "https":
        raise SourcePolicyError(f"{label}: media URLs must use https, got {raw.url.scheme!r}")
    host = (raw.url.host or "").lower().rstrip(".")
    if not host:
        raise SourcePolicyError(f"{label}: media URL has no host")
    if not any(host == domain or host.endswith(f".{domain}") for domain in allowed.domains):
        permitted = ", ".join(sorted(allowed.domains))
        raise SourcePolicyError(
            f"{label}: host {host!r} is not an allowlisted domain for "
            f"{allowed.source_id!r} (allowed: {permitted})"
        )


def _build_media(
    raw: RawMediaItem,
    corpus: MediaCorpus,
    allowed: AllowedSource,
    label: str,
) -> MediaItem:
    try:
        return MediaItem(
            source_id=raw.source_id,
            source_type=raw.source_type,
            title=raw.title,
            author=raw.author,
            publisher=allowed.publisher,
            url=raw.url,
            published_on=raw.published_on,
            season=raw.season,
            week=raw.week,
            game_id=raw.game_id,
            opponent=raw.opponent,
            body=raw.body,
            retrieved_at=corpus.retrieved_at,
        )
    except ValidationError as exc:
        raise MediaCorpusError(f"{label}: {_format_validation_error(exc)}") from exc


def _build_quote(raw: RawQuote, media: MediaItem, *, label: str) -> Quote:
    team = raw.team if raw.team is not None else media.opponent
    if team is not media.opponent:
        raise SourcePolicyError(
            f"{label}: quote team {team.value} does not match the media opponent "
            f"{media.opponent.value}"
        )
    week = raw.week if raw.week is not None else media.week
    if week is None:
        raise MediaCorpusError(f"{label}: a quote must record the week it belongs to")
    game_id = raw.game_id if raw.game_id is not None else media.game_id

    char_start, char_end = _quote_span(raw, media, label)
    if media.body[char_start:char_end] != raw.text:
        raise QuoteExtractionError(f"{label}: stored span does not reproduce the quote verbatim")
    try:
        return Quote(
            media_id=media.media_id,
            speaker=raw.speaker,
            speaker_role=raw.speaker_role,
            team=team,
            season=media.season,
            week=week,
            game_id=game_id,
            spoken_on=raw.spoken_on if raw.spoken_on is not None else media.published_on,
            publisher=media.publisher,
            url=media.url,
            text=raw.text,
            char_start=char_start,
            char_end=char_end,
            occurrence=raw.occurrence,
        )
    except ValidationError as exc:
        raise MediaCorpusError(f"{label}: {_format_validation_error(exc)}") from exc


def _quote_span(raw: RawQuote, media: MediaItem, label: str) -> tuple[int, int]:
    if (raw.char_start is None) != (raw.char_end is None):
        raise QuoteExtractionError(f"{label}: char_start and char_end must be provided together")
    if raw.char_start is not None and raw.char_end is not None:
        if raw.char_start < 0 or raw.char_end > len(media.body) or raw.char_end <= raw.char_start:
            raise QuoteExtractionError(
                f"{label}: span [{raw.char_start}, {raw.char_end}) is outside the body "
                f"of {len(media.body)} characters"
            )
        return raw.char_start, raw.char_end

    cursor = 0
    found = -1
    for _ in range(raw.occurrence + 1):
        found = media.body.find(raw.text, cursor)
        if found < 0:
            raise QuoteExtractionError(
                f"{label}: quote text does not appear occurrence {raw.occurrence} in "
                f"media {media.media_id}"
            )
        cursor = found + 1
    return found, found + len(raw.text)


# ---------------------------------------------------------------------------
# Corpus parsing helpers
# ---------------------------------------------------------------------------


def _mapping(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MediaCorpusError(f"{where} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise MediaCorpusError(f"{where} must use string keys")
    return value


def _reject_unknown(mapping: Mapping[str, object], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        joined = ", ".join(repr(name) for name in unknown)
        raise MediaCorpusError(f"{where} has unexpected field(s): {joined}")


def _required_str(mapping: Mapping[str, object], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MediaCorpusError(f"{where}: {key!r} must be a non-empty string")
    return value


def _optional_str(mapping: Mapping[str, object], key: str, where: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise MediaCorpusError(f"{where}: {key!r} must be a non-empty string when provided")
    return value


def _required_int(mapping: Mapping[str, object], key: str, where: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MediaCorpusError(f"{where}: {key!r} must be an integer")
    return value


def _optional_int(mapping: Mapping[str, object], key: str, where: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise MediaCorpusError(f"{where}: {key!r} must be an integer when provided")
    return value


def _required_date(mapping: Mapping[str, object], key: str, where: str) -> date:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise MediaCorpusError(f"{where}: {key!r} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise MediaCorpusError(f"{where}: {key!r} is not an ISO date: {value!r}") from exc


def _optional_date(mapping: Mapping[str, object], key: str, where: str) -> date | None:
    if mapping.get(key) is None:
        return None
    return _required_date(mapping, key, where)


def _required_datetime(mapping: Mapping[str, object], key: str, where: str) -> datetime:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise MediaCorpusError(f"{where}: {key!r} must be an ISO datetime string")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise MediaCorpusError(f"{where}: {key!r} is not an ISO datetime: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MediaCorpusError(f"{where}: {key!r} must be timezone-aware")
    return parsed


def _parse_item(item: object, *, season: int, where: str) -> RawMediaItem:
    mapping = _mapping(item, where)
    _reject_unknown(
        mapping,
        {
            "source_id",
            "source_type",
            "title",
            "author",
            "publisher",
            "url",
            "published_on",
            "season",
            "week",
            "game_id",
            "opponent",
            "body",
            "quotes",
        },
        where,
    )

    source_type_text = _required_str(mapping, "source_type", where)
    try:
        source_type = MediaType(source_type_text)
    except ValueError as exc:
        permitted = ", ".join(sorted(item.value for item in MediaType))
        raise MediaCorpusError(
            f"{where}: unknown source_type {source_type_text!r} (expected one of {permitted})"
        ) from exc

    url_text = _required_str(mapping, "url", where)
    try:
        url = HttpUrl(url_text)
    except ValidationError as exc:
        raise MediaCorpusError(f"{where}: 'url' is not a valid URL: {url_text!r}") from exc

    opponent_text = _required_str(mapping, "opponent", where)
    try:
        opponent = Team(opponent_text.upper())
    except ValueError as exc:
        raise MediaCorpusError(
            f"{where}: unknown opponent team abbreviation {opponent_text!r}"
        ) from exc

    raw_quotes = mapping.get("quotes", [])
    if not isinstance(raw_quotes, list):
        raise MediaCorpusError(f"{where}: 'quotes' must be a list")

    return RawMediaItem(
        source_id=_required_str(mapping, "source_id", where),
        source_type=source_type,
        title=_required_str(mapping, "title", where),
        author=_optional_str(mapping, "author", where),
        publisher=_optional_str(mapping, "publisher", where),
        url=url,
        published_on=_required_date(mapping, "published_on", where),
        season=_optional_int(mapping, "season", where) or season,
        week=_optional_int(mapping, "week", where),
        game_id=_optional_str(mapping, "game_id", where),
        opponent=opponent,
        body=_required_str(mapping, "body", where),
        quotes=tuple(
            _parse_quote(raw_quote, where=f"{where} quotes[{index}]")
            for index, raw_quote in enumerate(raw_quotes)
        ),
    )


def _parse_quote(item: object, *, where: str) -> RawQuote:
    mapping = _mapping(item, where)
    _reject_unknown(
        mapping,
        {
            "speaker",
            "speaker_role",
            "team",
            "spoken_on",
            "week",
            "game_id",
            "text",
            "occurrence",
            "char_start",
            "char_end",
        },
        where,
    )

    role_text = _required_str(mapping, "speaker_role", where)
    try:
        role = SpeakerRole(role_text)
    except ValueError as exc:
        permitted = ", ".join(sorted(item.value for item in SpeakerRole))
        raise MediaCorpusError(
            f"{where}: unknown speaker_role {role_text!r} (expected one of {permitted})"
        ) from exc

    team_text = _optional_str(mapping, "team", where)
    team: Team | None = None
    if team_text is not None:
        try:
            team = Team(team_text.upper())
        except ValueError as exc:
            raise MediaCorpusError(f"{where}: unknown team abbreviation {team_text!r}") from exc

    occurrence = _optional_int(mapping, "occurrence", where)
    return RawQuote(
        speaker=_required_str(mapping, "speaker", where),
        speaker_role=role,
        text=_required_str(mapping, "text", where),
        occurrence=0 if occurrence is None else occurrence,
        team=team,
        spoken_on=_optional_date(mapping, "spoken_on", where),
        week=_optional_int(mapping, "week", where),
        game_id=_optional_str(mapping, "game_id", where),
        char_start=_optional_int(mapping, "char_start", where),
        char_end=_optional_int(mapping, "char_end", where),
    )


def _format_validation_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'record'}: {error['msg']}"
        for error in exc.errors()
    )


def host_is_allowlisted(url: HttpUrl, domains: Sequence[str]) -> bool:
    """Return whether ``url``'s host belongs to one of ``domains``."""

    host = (url.host or "").lower().rstrip(".")
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)
