"""Source, opponent, and week validation for stored data and agent evidence (Stage 5).

Every check returns a :class:`~opponent_tendencies.models.ValidationResult` of
structured issues instead of raising, so the orchestrator can choose between a
hard failure and a warned, partial report (requirements 21, 22). The checks run
before any model sees data and again when agents cite evidence:

* :func:`validate_selection` — the opponent and weeks are playable in the
  stored season.
* :func:`check_plays` — duplicate play keys and missing values are reported,
  never imputed (requirement 24).
* :func:`check_media_alignment` — stored sources line up with the opponent and
  the selected weeks of the analyzed season. Earlier-season reporting is
  labeled ``media_background`` so it can be shown as background but never as
  evidence for the selected weeks; reporting from after the selected weeks and
  other seasons are rejected outright (requirement 26).
* :func:`check_retrieval` — retrieval results resolve to stored media, carry
  text that is an exact span of the stored body, and respect the same
  opponent/week alignment before a model reads them (requirements 21, 22).
* :func:`check_evidence` — every game, play, media, quote, and stat-table ID
  cited by an agent resolves to stored source data, quotes still reproduce
  their span verbatim, and media/quote snippets are grounded in stored text
  (requirements 19, 21).
* :func:`check_stat_tables` — deterministic tables stay inside the selected
  weeks and games, and small samples are flagged for human review
  (requirements 24, 27).
* :func:`check_records` — raw records are schema-validated into structured
  issues instead of a bare traceback (requirement 21).

Weak evidence — small samples, absent sources, empty retrieval — is flagged
with ``requires_human_review`` so the UI can surface it (requirement 27).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ValidationError

from opponent_tendencies.data.index import RetrievedSource
from opponent_tendencies.data.media_store import MediaData
from opponent_tendencies.data.play_store import SeasonData
from opponent_tendencies.ids import play_key
from opponent_tendencies.models import (
    EvidenceKind,
    EvidenceRef,
    IssueSeverity,
    Play,
    PlayType,
    StatTable,
    Team,
    ValidationIssue,
    ValidationResult,
)
from opponent_tendencies.stats.engine import DataQuality

__all__ = [
    "CheckCode",
    "check_evidence",
    "check_media_alignment",
    "check_plays",
    "check_records",
    "check_retrieval",
    "check_stat_tables",
    "validate_selection",
]


class CheckCode(StrEnum):
    """Stable issue codes emitted by the validation layer."""

    SELECTION_WEEKS_EMPTY = "selection_weeks_empty"
    SELECTION_WEEK_DUPLICATES = "selection_week_duplicates"
    SELECTION_WEEK_INVALID = "selection_week_invalid"
    SELECTION_WEEK_NOT_PLAYED = "selection_week_not_played"
    SELECTION_WEEK_WITHOUT_OPPONENT = "selection_week_without_opponent"
    SELECTION_OPPONENT_UNKNOWN = "selection_opponent_unknown"
    SELECTION_NO_GAMES = "selection_no_games"

    DUPLICATE_PLAY = "duplicate_play"
    MISSING_VALUES = "missing_values"
    UNKNOWN_PLAY_TYPE = "unknown_play_type"

    MEDIA_PRIOR_SEASON = "media_prior_season"
    MEDIA_WEEK_LATER = "media_week_later"
    MEDIA_BACKGROUND = "media_background"
    MEDIA_WEEK_UNKNOWN = "media_week_unknown"
    MEDIA_GAME_NOT_FOUND = "media_game_not_found"
    MEDIA_GAME_WEEK_MISMATCH = "media_game_week_mismatch"
    NO_MEDIA_FOR_SELECTION = "no_media_for_selection"
    WEEK_WITHOUT_MEDIA = "week_without_media"

    RETRIEVAL_EMPTY = "retrieval_empty"
    RETRIEVAL_DUPLICATE_UNIT = "retrieval_duplicate_unit"
    RETRIEVAL_SCORE_INVALID = "retrieval_score_invalid"
    RETRIEVAL_KIND_INVALID = "retrieval_kind_invalid"
    RETRIEVAL_MEDIA_NOT_FOUND = "retrieval_media_not_found"
    RETRIEVAL_OPPONENT_MISMATCH = "retrieval_opponent_mismatch"
    RETRIEVAL_SEASON_MISMATCH = "retrieval_season_mismatch"
    RETRIEVAL_TEXT_UNGROUNDED = "retrieval_text_ungrounded"
    RETRIEVAL_QUOTE_MISSING = "retrieval_quote_missing"
    RETRIEVAL_QUOTE_NOT_FOUND = "retrieval_quote_not_found"
    RETRIEVAL_QUOTE_MEDIA_MISMATCH = "retrieval_quote_media_mismatch"

    GAME_NOT_FOUND = "game_not_found"
    PLAY_NOT_FOUND = "play_not_found"
    MEDIA_NOT_FOUND = "media_not_found"
    QUOTE_NOT_FOUND = "quote_not_found"
    QUOTE_MEDIA_NOT_FOUND = "quote_media_not_found"
    QUOTE_NOT_VERBATIM = "quote_not_verbatim"
    STAT_TABLE_NOT_FOUND = "stat_table_not_found"
    SNIPPET_UNGROUNDED = "snippet_ungrounded"

    STAT_TABLE_DUPLICATE = "stat_table_duplicate"
    STAT_TABLE_WEEK_MISALIGNED = "stat_table_week_misaligned"
    STAT_TABLE_GAME_MISALIGNED = "stat_table_game_misaligned"
    STAT_TABLE_SUPPRESSED = "stat_table_suppressed"
    STAT_ROWS_SUPPRESSED = "stat_rows_suppressed"

    INVALID_RECORD = "invalid_record"


_UNIT_KINDS: Final = frozenset({"chunk", "quote"})
_MAX_WEEK: Final = 18


def _issue(
    code: CheckCode,
    message: str,
    *,
    severity: IssueSeverity,
    subject_kind: str | None = None,
    subject_id: str | None = None,
    requires_human_review: bool = False,
    details: Mapping[str, object] | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code.value,
        message=message,
        severity=severity,
        subject_kind=subject_kind,
        subject_id=subject_id,
        requires_human_review=requires_human_review,
        details={str(key): str(value) for key, value in (details or {}).items()},
    )


def _result(
    issues: Iterable[ValidationIssue],
    *,
    checked: Mapping[str, int],
) -> ValidationResult:
    collected = tuple(issues)
    errors = [issue for issue in collected if issue.severity is IssueSeverity.ERROR]
    warnings = [issue for issue in collected if issue.severity is not IssueSeverity.ERROR]
    return ValidationResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        checked=dict(checked),
    )


def _normalize_weeks(weeks: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(week) for week in weeks}))


def _format_weeks(weeks: Sequence[int]) -> str:
    if len(weeks) == 1:
        return f"week {weeks[0]}"
    if list(weeks) == list(range(weeks[0], weeks[-1] + 1)):
        return f"weeks {weeks[0]}-{weeks[-1]}"
    return "weeks " + ", ".join(str(week) for week in weeks)


def _weeks_empty_issue() -> ValidationIssue:
    return _issue(
        CheckCode.SELECTION_WEEKS_EMPTY,
        "select at least one regular-season week before running checks",
        severity=IssueSeverity.ERROR,
        subject_kind="selection",
    )


def _week_issues(
    week: int | None,
    weeks: Sequence[int],
    *,
    subject_kind: str,
    subject_id: str,
) -> list[ValidationIssue]:
    """Classify one source's week against the selected weeks (requirement 26).

    A week inside the selection is aligned. A week earlier in the same season
    is reported as background: it may be labeled background in the report but
    cannot support tendencies for the selected weeks. A week after the
    selection cannot describe those weeks at all and is rejected. Sources
    without a recorded week cannot be aligned.
    """

    if week is None:
        return [
            _issue(
                CheckCode.MEDIA_WEEK_UNKNOWN,
                f"{subject_kind} {subject_id!r} does not record a week, so it cannot be "
                "aligned with the selected weeks",
                severity=IssueSeverity.WARNING,
                subject_kind=subject_kind,
                subject_id=subject_id,
            )
        ]
    if week in weeks:
        return []
    scope = _format_weeks(weeks)
    if week > weeks[-1]:
        return [
            _issue(
                CheckCode.MEDIA_WEEK_LATER,
                f"{subject_kind} {subject_id!r} covers week {week}, after the selected "
                f"{scope}; it cannot be used as evidence",
                severity=IssueSeverity.ERROR,
                subject_kind=subject_kind,
                subject_id=subject_id,
                details={"week": week},
            )
        ]
    return [
        _issue(
            CheckCode.MEDIA_BACKGROUND,
            f"{subject_kind} {subject_id!r} covers week {week}, earlier than the selected "
            f"{scope}; it is labeled background and cannot support tendencies for the "
            "selected weeks",
            severity=IssueSeverity.WARNING,
            subject_kind=subject_kind,
            subject_id=subject_id,
            details={"week": week},
        )
    ]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def validate_selection(
    season_data: SeasonData,
    *,
    opponent: Team,
    weeks: Iterable[int],
) -> ValidationResult:
    """Check that the opponent and weeks describe playable stored games.

    Every selected week must be a regular-season week (1-18) with at least one
    completed game in the store, the opponent must appear in played games, and
    at least one selected week must contain a game involving the opponent.
    """

    issues: list[ValidationIssue] = []
    raw = [int(week) for week in weeks]
    if not raw:
        issues.append(_weeks_empty_issue())
        return _result(issues, checked={"weeks": 0, "games": 0})

    duplicates = sorted({week for week in raw if raw.count(week) > 1})
    if duplicates:
        issues.append(
            _issue(
                CheckCode.SELECTION_WEEK_DUPLICATES,
                f"weeks must not repeat: {duplicates}",
                severity=IssueSeverity.ERROR,
                subject_kind="selection",
                details={"weeks": ",".join(str(week) for week in duplicates)},
            )
        )

    normalized = _normalize_weeks(raw)
    for week in normalized:
        if not 1 <= week <= _MAX_WEEK:
            issues.append(
                _issue(
                    CheckCode.SELECTION_WEEK_INVALID,
                    f"week {week} is outside the regular-season range 1-{_MAX_WEEK}",
                    severity=IssueSeverity.ERROR,
                    subject_kind="week",
                    subject_id=str(week),
                )
            )

    known_opponent = opponent in season_data.teams
    if not known_opponent:
        issues.append(
            _issue(
                CheckCode.SELECTION_OPPONENT_UNKNOWN,
                f"{opponent.value} has no played regular-season games in the stored season",
                severity=IssueSeverity.ERROR,
                subject_kind="opponent",
                subject_id=opponent.value,
            )
        )

    played_weeks = set(season_data.played_weeks)
    for week in normalized:
        if 1 <= week <= _MAX_WEEK and week not in played_weeks:
            issues.append(
                _issue(
                    CheckCode.SELECTION_WEEK_NOT_PLAYED,
                    f"week {week} has no completed regular-season game in the stored season",
                    severity=IssueSeverity.ERROR,
                    subject_kind="week",
                    subject_id=str(week),
                )
            )

    selected = frozenset(normalized)
    opponent_games = tuple(
        game
        for game in season_data.played_games
        if game.involves(opponent) and game.week in selected
    )
    if known_opponent:
        if not opponent_games:
            issues.append(
                _issue(
                    CheckCode.SELECTION_NO_GAMES,
                    f"{opponent.value} does not play in any selected week",
                    severity=IssueSeverity.ERROR,
                    subject_kind="opponent",
                    subject_id=opponent.value,
                )
            )
        else:
            played_by_opponent = {
                game.week for game in season_data.played_games if game.involves(opponent)
            }
            for week in normalized:
                if week in played_weeks and week not in played_by_opponent:
                    issues.append(
                        _issue(
                            CheckCode.SELECTION_WEEK_WITHOUT_OPPONENT,
                            f"{opponent.value} does not play in week {week}; "
                            "that week adds no games",
                            severity=IssueSeverity.INFO,
                            subject_kind="week",
                            subject_id=str(week),
                        )
                    )

    return _result(issues, checked={"weeks": len(normalized), "games": len(opponent_games)})


# ---------------------------------------------------------------------------
# Stored plays
# ---------------------------------------------------------------------------


def check_plays(
    plays: Iterable[Play],
    *,
    opponent: Team | None = None,
    game_ids: Iterable[str] | None = None,
) -> ValidationResult:
    """Report duplicate play keys and missing values before analysis (req 24).

    Duplicates are errors: the season store guarantees unique
    ``(game_id, play_id)`` pairs, so a repeat means the caller is trusting
    forked data. Missing values are informational: they are explained, never
    imputed, and mirror the populations the statistics engine analyzes — the
    opponent's scrimmage plays from the selected games.
    """

    issues: list[ValidationIssue] = []
    input_plays = tuple(plays)

    seen: set[str] = set()
    for play in input_plays:
        if play.key in seen:
            issues.append(
                _issue(
                    CheckCode.DUPLICATE_PLAY,
                    f"play {play.key} appears more than once; the season store guarantees "
                    "unique (game_id, play_id) pairs",
                    severity=IssueSeverity.ERROR,
                    subject_kind="play",
                    subject_id=play.key,
                )
            )
        seen.add(play.key)

    allowed_games = None if game_ids is None else frozenset(game_ids)
    population = [
        play
        for play in input_plays
        if (allowed_games is None or play.game_id in allowed_games)
        and (opponent is None or play.posteam is opponent)
    ]
    scrimmage = [play for play in population if play.is_scrimmage_play]

    missing: dict[str, int] = {
        "down": sum(1 for play in scrimmage if play.down is None),
        "yards_to_go": sum(1 for play in scrimmage if play.yards_to_go is None),
        "yardline_100": sum(1 for play in scrimmage if play.yardline_100 is None),
        "yards_gained": sum(1 for play in scrimmage if play.yards_gained is None),
        "third_down_outcomes": sum(
            1
            for play in scrimmage
            if play.down == 3
            and play.third_down_converted is None
            and play.third_down_failed is None
        ),
    }
    for field, count in missing.items():
        if count:
            issues.append(
                _issue(
                    CheckCode.MISSING_VALUES,
                    f"{count} scrimmage play(s) have no recorded {field.replace('_', ' ')}; "
                    "affected metrics exclude them instead of imputing a value",
                    severity=IssueSeverity.INFO,
                    subject_kind="play",
                    details={"field": field, "plays": count},
                )
            )

    unknown_types = sum(1 for play in population if play.play_type is PlayType.UNKNOWN)
    if unknown_types:
        issues.append(
            _issue(
                CheckCode.UNKNOWN_PLAY_TYPE,
                f"{unknown_types} offensive play(s) have no recorded play type",
                severity=IssueSeverity.INFO,
                subject_kind="play",
                details={"plays": unknown_types},
            )
        )

    return _result(issues, checked={"plays": len(population)})


# ---------------------------------------------------------------------------
# Media alignment
# ---------------------------------------------------------------------------


def check_media_alignment(
    media_data: MediaData,
    *,
    opponent: Team,
    weeks: Iterable[int],
    season: int,
    season_data: SeasonData | None = None,
) -> ValidationResult:
    """Check stored sources for the opponent line up with the selection (req 26).

    Sources from another season are rejected. Earlier-season reporting is
    labeled background, reporting from after the selected weeks is rejected,
    and a source tied to a game must resolve to that game in the season store
    with a matching week. Absent sources are reported so the report can say
    what is missing instead of staying silent (requirement 22).
    """

    issues: list[ValidationIssue] = []
    normalized = _normalize_weeks(weeks)
    if not normalized:
        issues.append(_weeks_empty_issue())
        return _result(issues, checked={"media": 0})

    items = media_data.media_for(opponent=opponent)
    if not items:
        issues.append(
            _issue(
                CheckCode.NO_MEDIA_FOR_SELECTION,
                f"no stored media covers {opponent.value} for the selected weeks; "
                "qualitative sections will have no sources",
                severity=IssueSeverity.WARNING,
                subject_kind="opponent",
                subject_id=opponent.value,
                requires_human_review=True,
                details={"weeks": ",".join(str(week) for week in normalized)},
            )
        )
        return _result(issues, checked={"media": 0})

    games_index = (
        None if season_data is None else {game.game_id: game for game in season_data.games}
    )
    aligned_by_week: dict[int, int] = {week: 0 for week in normalized}

    for item in items:
        if item.season != season:
            issues.append(
                _issue(
                    CheckCode.MEDIA_PRIOR_SEASON,
                    f"media {item.media_id} is from season {item.season}; only season "
                    f"{season} sources may be used",
                    severity=IssueSeverity.ERROR,
                    subject_kind="media",
                    subject_id=item.media_id,
                    details={"season": item.season},
                )
            )
            continue
        if item.week is not None and item.week in aligned_by_week:
            aligned_by_week[item.week] += 1
        issues.extend(
            _week_issues(item.week, normalized, subject_kind="media", subject_id=item.media_id)
        )
        if item.game_id is not None and games_index is not None:
            game = games_index.get(item.game_id)
            if game is None:
                issues.append(
                    _issue(
                        CheckCode.MEDIA_GAME_NOT_FOUND,
                        f"media {item.media_id} references game {item.game_id}, which is "
                        "not stored in the season data",
                        severity=IssueSeverity.ERROR,
                        subject_kind="media",
                        subject_id=item.media_id,
                        details={"game_id": item.game_id},
                    )
                )
            elif item.week is not None and game.week != item.week:
                issues.append(
                    _issue(
                        CheckCode.MEDIA_GAME_WEEK_MISMATCH,
                        f"media {item.media_id} records week {item.week} but its game "
                        f"{game.game_id} is week {game.week}",
                        severity=IssueSeverity.ERROR,
                        subject_kind="media",
                        subject_id=item.media_id,
                        details={"game_id": game.game_id, "week": item.week},
                    )
                )

    for week in normalized:
        if not aligned_by_week[week]:
            issues.append(
                _issue(
                    CheckCode.WEEK_WITHOUT_MEDIA,
                    f"no stored media covers {opponent.value} in week {week}",
                    severity=IssueSeverity.WARNING,
                    subject_kind="week",
                    subject_id=str(week),
                    details={"week": week},
                )
            )

    return _result(issues, checked={"media": len(items)})


# ---------------------------------------------------------------------------
# Retrieval results
# ---------------------------------------------------------------------------


def check_retrieval(
    results: Iterable[RetrievedSource],
    media_data: MediaData,
    *,
    opponent: Team,
    weeks: Iterable[int],
    season: int,
) -> ValidationResult:
    """Validate retrieval results against stored data before a model sees them.

    Every result must resolve to a stored media item for the requested
    opponent and season, respect the selected weeks under the same background
    rule as :func:`check_media_alignment`, and carry text that is an exact span
    of the stored body, so retrieved text is always stored text. Quote results
    must resolve to a stored quote and reproduce it exactly. Empty retrieval is
    reported as a warning rather than silence (requirements 21, 22).
    """

    issues: list[ValidationIssue] = []
    normalized = _normalize_weeks(weeks)
    if not normalized:
        issues.append(_weeks_empty_issue())
        return _result(issues, checked={"retrieval": 0})

    retrieved = tuple(results)
    if not retrieved:
        issues.append(
            _issue(
                CheckCode.RETRIEVAL_EMPTY,
                f"retrieval returned no sources for {opponent.value}; the section must "
                "report what is missing instead of inventing sources",
                severity=IssueSeverity.WARNING,
                subject_kind="retrieval",
                subject_id=opponent.value,
                requires_human_review=True,
            )
        )
        return _result(issues, checked={"retrieval": 0})

    seen: set[str] = set()
    for result in retrieved:
        if result.unit_id in seen:
            issues.append(
                _issue(
                    CheckCode.RETRIEVAL_DUPLICATE_UNIT,
                    f"retrieval returned unit {result.unit_id} more than once",
                    severity=IssueSeverity.ERROR,
                    subject_kind="retrieval",
                    subject_id=result.unit_id,
                )
            )
        seen.add(result.unit_id)

        if not math.isfinite(result.score):
            issues.append(
                _issue(
                    CheckCode.RETRIEVAL_SCORE_INVALID,
                    f"retrieval unit {result.unit_id} has a non-finite similarity score",
                    severity=IssueSeverity.ERROR,
                    subject_kind="retrieval",
                    subject_id=result.unit_id,
                )
            )
        if result.kind not in _UNIT_KINDS:
            issues.append(
                _issue(
                    CheckCode.RETRIEVAL_KIND_INVALID,
                    f"retrieval unit {result.unit_id} has unknown unit kind "
                    f"{result.kind!r}; expected 'chunk' or 'quote'",
                    severity=IssueSeverity.ERROR,
                    subject_kind="retrieval",
                    subject_id=result.unit_id,
                    details={"unit_kind": result.kind},
                )
            )

        stored = media_data.media_by_id(result.media_id)
        if stored is None:
            issues.append(
                _issue(
                    CheckCode.RETRIEVAL_MEDIA_NOT_FOUND,
                    f"retrieval unit {result.unit_id} references media {result.media_id}, "
                    "which is not stored",
                    severity=IssueSeverity.ERROR,
                    subject_kind="retrieval",
                    subject_id=result.unit_id,
                    details={"media_id": result.media_id},
                )
            )
            continue
        if stored.opponent is not opponent:
            issues.append(
                _issue(
                    CheckCode.RETRIEVAL_OPPONENT_MISMATCH,
                    f"retrieval unit {result.unit_id} belongs to {stored.opponent.value} "
                    f"reporting, not {opponent.value}",
                    severity=IssueSeverity.ERROR,
                    subject_kind="retrieval",
                    subject_id=result.unit_id,
                    details={"opponent": stored.opponent.value},
                )
            )
        if stored.season != season:
            issues.append(
                _issue(
                    CheckCode.RETRIEVAL_SEASON_MISMATCH,
                    f"retrieval unit {result.unit_id} belongs to season {stored.season}, "
                    f"not {season}",
                    severity=IssueSeverity.ERROR,
                    subject_kind="retrieval",
                    subject_id=result.unit_id,
                    details={"season": stored.season},
                )
            )
            continue

        issues.extend(
            _week_issues(
                stored.week,
                normalized,
                subject_kind="retrieval",
                subject_id=result.unit_id,
            )
        )

        if not result.text.strip() or result.text not in stored.body:
            issues.append(
                _issue(
                    CheckCode.RETRIEVAL_TEXT_UNGROUNDED,
                    f"retrieval unit {result.unit_id} text is not an exact span of stored "
                    f"media {result.media_id}",
                    severity=IssueSeverity.ERROR,
                    subject_kind="retrieval",
                    subject_id=result.unit_id,
                    details={"media_id": result.media_id},
                )
            )

        if result.quote_id is not None:
            quote = media_data.quote_by_id(result.quote_id)
            if quote is None:
                issues.append(
                    _issue(
                        CheckCode.RETRIEVAL_QUOTE_NOT_FOUND,
                        f"retrieval unit {result.unit_id} references quote "
                        f"{result.quote_id}, which is not stored",
                        severity=IssueSeverity.ERROR,
                        subject_kind="retrieval",
                        subject_id=result.unit_id,
                        details={"quote_id": result.quote_id},
                    )
                )
            elif quote.media_id != result.media_id:
                issues.append(
                    _issue(
                        CheckCode.RETRIEVAL_QUOTE_MEDIA_MISMATCH,
                        f"retrieval unit {result.unit_id} pairs quote {quote.quote_id} "
                        f"with media {result.media_id}, but the quote belongs to "
                        f"{quote.media_id}",
                        severity=IssueSeverity.ERROR,
                        subject_kind="retrieval",
                        subject_id=result.unit_id,
                    )
                )
            elif result.kind == "quote" and result.text != quote.text:
                issues.append(
                    _issue(
                        CheckCode.RETRIEVAL_TEXT_UNGROUNDED,
                        f"retrieval quote unit {result.unit_id} does not reproduce quote "
                        f"{quote.quote_id} verbatim",
                        severity=IssueSeverity.ERROR,
                        subject_kind="retrieval",
                        subject_id=result.unit_id,
                        details={"quote_id": quote.quote_id},
                    )
                )
        elif result.kind == "quote":
            issues.append(
                _issue(
                    CheckCode.RETRIEVAL_QUOTE_MISSING,
                    f"retrieval quote unit {result.unit_id} does not carry a quote_id",
                    severity=IssueSeverity.ERROR,
                    subject_kind="retrieval",
                    subject_id=result.unit_id,
                )
            )

    return _result(issues, checked={"retrieval": len(retrieved)})


# ---------------------------------------------------------------------------
# Agent evidence
# ---------------------------------------------------------------------------


def check_evidence(
    evidence: Iterable[EvidenceRef],
    *,
    season_data: SeasonData,
    media_data: MediaData,
    stat_tables: Iterable[StatTable] = (),
) -> ValidationResult:
    """Check every cited ID resolves to stored source data (reqs 19, 21).

    Play evidence must resolve to a stored game and play. Media evidence must
    resolve to a stored media item, and any snippet it carries must be an
    exact span of the stored body. Quote evidence must resolve to a stored
    quote whose span still reproduces its text verbatim in the stored article,
    and its snippet must be grounded in the quote text. Stat evidence must
    resolve to one of the provided deterministic tables.
    """

    issues: list[ValidationIssue] = []
    refs = tuple(evidence)

    games = {game.game_id: game for game in season_data.games}
    plays = {play.key: play for play in season_data.plays}
    media = {item.media_id: item for item in media_data.media}
    quotes = {quote.quote_id: quote for quote in media_data.quotes}
    tables = {table.stat_table_id for table in stat_tables}

    for ref in refs:
        if ref.kind is EvidenceKind.PLAY:
            if ref.game_id is None or ref.play_id is None:
                issues.append(
                    _issue(
                        CheckCode.PLAY_NOT_FOUND,
                        f"play evidence {ref.evidence_id} is missing its game or play locator",
                        severity=IssueSeverity.ERROR,
                        subject_kind="evidence",
                        subject_id=ref.evidence_id,
                    )
                )
                continue
            if ref.game_id not in games:
                issues.append(
                    _issue(
                        CheckCode.GAME_NOT_FOUND,
                        f"evidence {ref.evidence_id} cites game {ref.game_id}, which is "
                        "not stored in the season data",
                        severity=IssueSeverity.ERROR,
                        subject_kind="evidence",
                        subject_id=ref.evidence_id,
                        details={"game_id": ref.game_id},
                    )
                )
                continue
            key = play_key(ref.game_id, ref.play_id)
            if key not in plays:
                issues.append(
                    _issue(
                        CheckCode.PLAY_NOT_FOUND,
                        f"evidence {ref.evidence_id} cites play {key}, which is not "
                        "stored in the season data",
                        severity=IssueSeverity.ERROR,
                        subject_kind="evidence",
                        subject_id=ref.evidence_id,
                        details={"game_id": ref.game_id, "play_id": ref.play_id},
                    )
                )
        elif ref.kind is EvidenceKind.MEDIA:
            stored = media.get(ref.media_id) if ref.media_id is not None else None
            if stored is None:
                issues.append(
                    _issue(
                        CheckCode.MEDIA_NOT_FOUND,
                        f"evidence {ref.evidence_id} cites media {ref.media_id}, which "
                        "is not stored",
                        severity=IssueSeverity.ERROR,
                        subject_kind="evidence",
                        subject_id=ref.evidence_id,
                        details={"media_id": ref.media_id},
                    )
                )
            elif ref.snippet is not None and ref.snippet not in stored.body:
                issues.append(
                    _issue(
                        CheckCode.SNIPPET_UNGROUNDED,
                        f"evidence {ref.evidence_id} snippet is not an exact span of "
                        f"stored media {stored.media_id}",
                        severity=IssueSeverity.ERROR,
                        subject_kind="evidence",
                        subject_id=ref.evidence_id,
                        details={"media_id": stored.media_id},
                    )
                )
        elif ref.kind is EvidenceKind.QUOTE:
            quote = quotes.get(ref.quote_id) if ref.quote_id is not None else None
            if quote is None:
                issues.append(
                    _issue(
                        CheckCode.QUOTE_NOT_FOUND,
                        f"evidence {ref.evidence_id} cites quote {ref.quote_id}, which "
                        "is not stored",
                        severity=IssueSeverity.ERROR,
                        subject_kind="evidence",
                        subject_id=ref.evidence_id,
                        details={"quote_id": ref.quote_id},
                    )
                )
                continue
            owner = media.get(quote.media_id)
            if owner is None:
                issues.append(
                    _issue(
                        CheckCode.QUOTE_MEDIA_NOT_FOUND,
                        f"quote {quote.quote_id} references media {quote.media_id}, which "
                        "is not stored",
                        severity=IssueSeverity.ERROR,
                        subject_kind="evidence",
                        subject_id=ref.evidence_id,
                        details={"media_id": quote.media_id},
                    )
                )
            elif not quote.is_verbatim_in(owner):
                issues.append(
                    _issue(
                        CheckCode.QUOTE_NOT_VERBATIM,
                        f"quote {quote.quote_id} span no longer reproduces its text "
                        f"verbatim in media {quote.media_id}",
                        severity=IssueSeverity.ERROR,
                        subject_kind="evidence",
                        subject_id=ref.evidence_id,
                        details={"quote_id": quote.quote_id},
                    )
                )
            elif ref.snippet is not None and ref.snippet not in quote.text:
                issues.append(
                    _issue(
                        CheckCode.SNIPPET_UNGROUNDED,
                        f"evidence {ref.evidence_id} snippet is not an exact span of "
                        f"quote {quote.quote_id}",
                        severity=IssueSeverity.ERROR,
                        subject_kind="evidence",
                        subject_id=ref.evidence_id,
                        details={"quote_id": quote.quote_id},
                    )
                )
        else:
            if ref.stat_table_id not in tables:
                issues.append(
                    _issue(
                        CheckCode.STAT_TABLE_NOT_FOUND,
                        f"evidence {ref.evidence_id} cites stat table {ref.stat_table_id}, "
                        "which is not part of the provided tables",
                        severity=IssueSeverity.ERROR,
                        subject_kind="evidence",
                        subject_id=ref.evidence_id,
                        details={"stat_table_id": ref.stat_table_id},
                    )
                )

    return _result(issues, checked={"evidence": len(refs)})


# ---------------------------------------------------------------------------
# Deterministic stat tables
# ---------------------------------------------------------------------------


def check_stat_tables(
    tables: Iterable[StatTable],
    *,
    weeks: Iterable[int],
    game_ids: Iterable[str],
    quality: DataQuality | None = None,
) -> ValidationResult:
    """Check deterministic tables stay in scope and flag small samples (reqs 24, 27).

    Tables must draw only on the selected weeks and games. Suppressed tables
    and suppressed rows keep their values but are flagged for human review,
    and the engine's data-quality counts are surfaced as explanatory notes so
    missing values are reported rather than imputed.
    """

    issues: list[ValidationIssue] = []
    normalized = _normalize_weeks(weeks)
    if not normalized:
        issues.append(_weeks_empty_issue())
        return _result(issues, checked={"stat_tables": 0})

    allowed_games = frozenset(game_ids)
    table_list = tuple(tables)

    seen: set[str] = set()
    for table in table_list:
        if table.stat_table_id in seen:
            issues.append(
                _issue(
                    CheckCode.STAT_TABLE_DUPLICATE,
                    f"stat table {table.stat_table_id} appears more than once",
                    severity=IssueSeverity.ERROR,
                    subject_kind="stat_table",
                    subject_id=table.stat_table_id,
                )
            )
        seen.add(table.stat_table_id)

        extra_weeks = sorted(set(table.weeks) - set(normalized))
        if extra_weeks:
            issues.append(
                _issue(
                    CheckCode.STAT_TABLE_WEEK_MISALIGNED,
                    f"stat table {table.stat_table_id} covers weeks {extra_weeks}, which "
                    "are not part of the selection",
                    severity=IssueSeverity.ERROR,
                    subject_kind="stat_table",
                    subject_id=table.stat_table_id,
                    details={"weeks": ",".join(str(week) for week in extra_weeks)},
                )
            )
        extra_games = sorted(set(table.game_ids) - allowed_games)
        if extra_games:
            issues.append(
                _issue(
                    CheckCode.STAT_TABLE_GAME_MISALIGNED,
                    f"stat table {table.stat_table_id} covers games {extra_games}, which "
                    "are not part of the selection",
                    severity=IssueSeverity.ERROR,
                    subject_kind="stat_table",
                    subject_id=table.stat_table_id,
                    details={"game_ids": ",".join(extra_games)},
                )
            )
        if table.suppressed:
            issues.append(
                _issue(
                    CheckCode.STAT_TABLE_SUPPRESSED,
                    f"stat table {table.stat_table_id} has {table.n_plays} play(s), below "
                    f"its minimum sample of {table.min_sample}; review before relying on it",
                    severity=IssueSeverity.WARNING,
                    subject_kind="stat_table",
                    subject_id=table.stat_table_id,
                    requires_human_review=True,
                    details={"n_plays": table.n_plays, "min_sample": table.min_sample},
                )
            )
        else:
            suppressed_rows = sum(1 for row in table.rows if row.suppressed)
            if suppressed_rows:
                issues.append(
                    _issue(
                        CheckCode.STAT_ROWS_SUPPRESSED,
                        f"stat table {table.stat_table_id} has {suppressed_rows} row(s) below "
                        f"its minimum sample of {table.min_sample}",
                        severity=IssueSeverity.WARNING,
                        subject_kind="stat_table",
                        subject_id=table.stat_table_id,
                        requires_human_review=True,
                        details={"rows": suppressed_rows, "min_sample": table.min_sample},
                    )
                )

    if quality is not None:
        gaps: dict[str, int] = {
            "down": quality.missing_down,
            "yards_to_go": quality.missing_yards_to_go,
            "yardline_100": quality.missing_yardline_100,
            "yards_gained": quality.missing_yards_gained,
            "third_down_outcomes": quality.missing_third_down_outcomes,
        }
        for field, count in gaps.items():
            if count:
                issues.append(
                    _issue(
                        CheckCode.MISSING_VALUES,
                        f"{count} play(s) have no recorded {field.replace('_', ' ')}; "
                        "affected metrics exclude them instead of imputing a value",
                        severity=IssueSeverity.INFO,
                        subject_kind="stat_table",
                        details={"field": field, "plays": count},
                    )
                )
        if quality.unknown_play_types:
            issues.append(
                _issue(
                    CheckCode.UNKNOWN_PLAY_TYPE,
                    f"{quality.unknown_play_types} offensive play(s) have no recorded play type",
                    severity=IssueSeverity.INFO,
                    subject_kind="stat_table",
                    details={"plays": quality.unknown_play_types},
                )
            )

    return _result(issues, checked={"stat_tables": len(table_list)})


# ---------------------------------------------------------------------------
# Raw records
# ---------------------------------------------------------------------------


def check_records(
    model: type[BaseModel],
    records: Sequence[object],
    *,
    kind: str,
) -> ValidationResult:
    """Schema-validate raw records into structured issues (requirement 21).

    Data responses are validated against a contract before they are used as
    evidence; each invalid record becomes one error describing its field
    failures instead of an unhandled validation traceback.
    """

    issues: list[ValidationIssue] = []
    valid = 0
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            issues.append(
                _issue(
                    CheckCode.INVALID_RECORD,
                    f"{kind}[{index}] must be a JSON object",
                    severity=IssueSeverity.ERROR,
                    subject_kind=kind,
                    subject_id=str(index),
                )
            )
            continue
        try:
            model.model_validate(dict(record))
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'record'}: {error['msg']}"
                for error in exc.errors()
            )
            issues.append(
                _issue(
                    CheckCode.INVALID_RECORD,
                    f"{kind}[{index}] fails {model.__name__} validation: {detail}",
                    severity=IssueSeverity.ERROR,
                    subject_kind=kind,
                    subject_id=str(index),
                    details={"index": index, "errors": detail},
                )
            )
        else:
            valid += 1

    return _result(issues, checked={kind: valid})
