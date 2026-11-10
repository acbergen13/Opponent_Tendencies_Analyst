"""Deterministic identifier and content-hash helpers.

Every identifier used by the application is derived from stable source
identity. Nothing here uses randomness or the current clock, so a refresh, an
index rebuild, or a repeated report run cannot fork records that represent the
same underlying source.

nflverse supplies ``game_id`` and ``play_id``. A play is only unique inside its
game, so :func:`play_key` (and :func:`make_play_evidence_id`) always carry both.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Final

_HASH_LENGTH: Final = 16
_SLUG_LIMIT: Final = 48
_NON_SLUG = re.compile(r"[^a-z0-9]+")
_WHITESPACE = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    """Collapse whitespace and lowercase, for stable comparisons in IDs."""

    return _WHITESPACE.sub(" ", value).strip().lower()


def _hash_parts(*parts: object, length: int = _HASH_LENGTH) -> str:
    payload = "\x1f".join(_part_to_text(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _part_to_text(part: object) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, Iterable):
        return ",".join(_part_to_text(item) for item in part)
    return str(part)


def _slug(value: str) -> str:
    slug = _NON_SLUG.sub("_", _normalize_text(value)).strip("_")
    return slug[:_SLUG_LIMIT] or "item"


def play_key(game_id: str, play_id: int) -> str:
    """Return the composite key that identifies a play inside a game."""

    return f"{game_id.strip()}:{int(play_id)}"


def make_media_id(source_id: str, url: str) -> str:
    """Stable media identity from the allowlist source and canonical URL."""

    return f"med_{_hash_parts(source_id.strip().lower(), url.strip())}"


def make_chunk_id(
    media_id: str,
    char_start: int,
    char_end: int,
    occurrence: int = 0,
) -> str:
    """Stable chunk identity from its article and character span.

    Chunk IDs are stored on media evidence and retrieval results, so they must
    not depend on chunking parameters, insertion order, or the current run.
    """

    return "chk_" + _hash_parts(
        media_id.strip(),
        int(char_start),
        int(char_end),
        int(occurrence),
    )


def make_quote_id(
    media_id: str,
    speaker: str,
    text: str,
    occurrence: int = 0,
) -> str:
    """Stable quote identity from its article, speaker, verbatim text, and ordinal."""

    fingerprint = _hash_parts(
        media_id.strip(),
        _normalize_text(speaker),
        _normalize_text(text),
        int(occurrence),
    )
    return f"qte_{fingerprint}"


def make_stat_table_id(
    metric: str,
    dimension: str,
    dimension_value: str | None,
    weeks: Iterable[int],
    game_ids: Iterable[str],
) -> str:
    """Stable table identity from what was measured and which games were used."""

    fingerprint = _hash_parts(
        dimension_value or "all",
        sorted(int(week) for week in weeks),
        sorted(game_id.strip() for game_id in game_ids),
    )
    return f"stt_{metric}_{dimension}_{fingerprint}"


def make_report_id(
    opponent: str,
    weeks: Iterable[int],
    data_revision: str,
    prompt_version: str,
) -> str:
    """Stable report identity for caching and duplicate-run detection."""

    return "rpt_" + _hash_parts(
        opponent.strip().upper(),
        sorted(int(week) for week in weeks),
        data_revision.strip(),
        prompt_version.strip(),
    )


def make_finding_id(
    kind: str,
    statement: str,
    evidence_ids: Iterable[str],
    weeks: Iterable[int],
) -> str:
    """Stable finding identity from its claim, citations, and week window."""

    return "fnd_" + _hash_parts(
        kind,
        _normalize_text(statement),
        sorted(evidence_id.strip() for evidence_id in evidence_ids),
        sorted(int(week) for week in weeks),
    )


def make_agreement_id(
    topic: str,
    statement: str,
    agents: Iterable[str],
    weeks: Iterable[int],
) -> str:
    """Stable identity for an agreement noted by synthesis."""

    return "agr_" + _hash_parts(
        _normalize_text(topic),
        _normalize_text(statement),
        sorted(str(agent) for agent in agents),
        sorted(int(week) for week in weeks),
    )


def make_conflict_id(
    topic: str,
    description: str,
    position_statements: Iterable[str],
    weeks: Iterable[int],
) -> str:
    """Stable identity for a conflict noted by synthesis."""

    return "cfl_" + _hash_parts(
        _normalize_text(topic),
        _normalize_text(description),
        sorted(_normalize_text(statement) for statement in position_statements),
        sorted(int(week) for week in weeks),
    )


def make_gap_id(topic: str, description: str, weeks: Iterable[int]) -> str:
    """Stable identity for a missing-information gap."""

    return "gap_" + _hash_parts(
        _normalize_text(topic),
        _normalize_text(description),
        sorted(int(week) for week in weeks),
    )


def make_weak_evidence_id(
    topic: str,
    description: str,
    evidence_ids: Iterable[str],
    weeks: Iterable[int],
) -> str:
    """Stable identity for a weak-evidence flag."""

    return "wke_" + _hash_parts(
        _normalize_text(topic),
        _normalize_text(description),
        sorted(evidence_id.strip() for evidence_id in evidence_ids),
        sorted(int(week) for week in weeks),
    )


def make_play_evidence_id(game_id: str, play_id: int) -> str:
    return f"ev:play:{play_key(game_id, play_id)}"


def make_media_evidence_id(media_id: str, chunk_id: str | None = None) -> str:
    return f"ev:media:{media_id.strip()}:{_slug(chunk_id or 'full')}"


def make_quote_evidence_id(quote_id: str) -> str:
    return f"ev:quote:{quote_id.strip()}"


def make_stat_evidence_id(stat_table_id: str, label: str | None = None) -> str:
    return f"ev:stat:{stat_table_id.strip()}:{_slug(label or 'table')}"


def make_content_hash(title: str, body: str) -> str:
    """SHA-256 over the retrievable text used to decide when to rebuild an index."""

    return hashlib.sha256(f"{title}\n\n{body}".encode()).hexdigest()
