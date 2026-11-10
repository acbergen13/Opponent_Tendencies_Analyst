"""Pydantic contracts shared by ingestion, analysis, agents, and the UI.

Two rules drive this module:

1. **Identity is deterministic.** ``game_id`` and ``play_id`` come from
   nflverse; every derived identifier comes from
   :mod:`opponent_tendencies.ids`, so refreshes and repeated runs cannot fork
   records that describe the same source.
2. **Facts are never free-floating.** Findings reference evidence IDs that must
   be present on the same payload, and a report must carry every citation made
   by its findings or by any specialist agent.

Models are frozen. Stages build new values instead of mutating shared state.
Lists keep insertion order where ranking matters (tendencies and observations);
identifier lists are validated as unique and sorted so serialized output is
stable across runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    field_validator,
    model_validator,
)

from opponent_tendencies.ids import (
    make_agreement_id,
    make_conflict_id,
    make_content_hash,
    make_finding_id,
    make_gap_id,
    make_media_evidence_id,
    make_media_id,
    make_play_evidence_id,
    make_quote_evidence_id,
    make_quote_id,
    make_report_id,
    make_stat_evidence_id,
    make_stat_table_id,
    make_weak_evidence_id,
    play_key,
)

# ---------------------------------------------------------------------------
# Shared constrained types
# ---------------------------------------------------------------------------

Season = Annotated[int, Field(ge=1920, le=2100)]
Week = Annotated[int, Field(ge=1, le=18)]
GameId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$"),
]
TrimmedStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
UntouchedStr = Annotated[str, Field(min_length=1)]
Snippet = Annotated[str, Field(max_length=1000)]


def _ensure_sorted_unique(values: Sequence[object], field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")
    if list(values) != sorted(values):  # type: ignore[type-var]
        raise ValueError(f"{field_name} must be sorted")


def _ensure_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------


class SeasonType(StrEnum):
    PRESEASON = "PRE"
    REGULAR = "REG"
    POSTSEASON = "POST"


class Team(StrEnum):
    ARI = "ARI"
    ATL = "ATL"
    BAL = "BAL"
    BUF = "BUF"
    CAR = "CAR"
    CHI = "CHI"
    CIN = "CIN"
    CLE = "CLE"
    DAL = "DAL"
    DEN = "DEN"
    DET = "DET"
    GB = "GB"
    HOU = "HOU"
    IND = "IND"
    JAX = "JAX"
    KC = "KC"
    LA = "LA"
    LAC = "LAC"
    LV = "LV"
    MIA = "MIA"
    MIN = "MIN"
    NE = "NE"
    NO = "NO"
    NYG = "NYG"
    NYJ = "NYJ"
    PHI = "PHI"
    PIT = "PIT"
    SEA = "SEA"
    SF = "SF"
    TB = "TB"
    TEN = "TEN"
    WSH = "WSH"


TEAM_DISPLAY_NAMES: dict[Team, str] = {
    Team.ARI: "Arizona Cardinals",
    Team.ATL: "Atlanta Falcons",
    Team.BAL: "Baltimore Ravens",
    Team.BUF: "Buffalo Bills",
    Team.CAR: "Carolina Panthers",
    Team.CHI: "Chicago Bears",
    Team.CIN: "Cincinnati Bengals",
    Team.CLE: "Cleveland Browns",
    Team.DAL: "Dallas Cowboys",
    Team.DEN: "Denver Broncos",
    Team.DET: "Detroit Lions",
    Team.GB: "Green Bay Packers",
    Team.HOU: "Houston Texans",
    Team.IND: "Indianapolis Colts",
    Team.JAX: "Jacksonville Jaguars",
    Team.KC: "Kansas City Chiefs",
    Team.LA: "Los Angeles Rams",
    Team.LAC: "Los Angeles Chargers",
    Team.LV: "Las Vegas Raiders",
    Team.MIA: "Miami Dolphins",
    Team.MIN: "Minnesota Vikings",
    Team.NE: "New England Patriots",
    Team.NO: "New Orleans Saints",
    Team.NYG: "New York Giants",
    Team.NYJ: "New York Jets",
    Team.PHI: "Philadelphia Eagles",
    Team.PIT: "Pittsburgh Steelers",
    Team.SEA: "Seattle Seahawks",
    Team.SF: "San Francisco 49ers",
    Team.TB: "Tampa Bay Buccaneers",
    Team.TEN: "Tennessee Titans",
    Team.WSH: "Washington Commanders",
}


def team_display_name(team: Team) -> str:
    """Return the full display name used by the team dropdown."""

    return TEAM_DISPLAY_NAMES[team]


def teams_alphabetical() -> tuple[Team, ...]:
    """Return all 32 teams ordered by display name (requirement 4)."""

    return tuple(sorted(TEAM_DISPLAY_NAMES, key=TEAM_DISPLAY_NAMES.__getitem__))


class PlayType(StrEnum):
    RUN = "run"
    PASS = "pass"
    PUNT = "punt"
    FIELD_GOAL = "field_goal"
    KICKOFF = "kickoff"
    EXTRA_POINT = "extra_point"
    QB_KNEEL = "qb_kneel"
    QB_SPIKE = "qb_spike"
    NO_PLAY = "no_play"
    UNKNOWN = "unknown"


class MediaType(StrEnum):
    ARTICLE = "article"
    FILM_BREAKDOWN = "film_breakdown"
    COACH_QUOTE = "coach_quote"


class SpeakerRole(StrEnum):
    HEAD_COACH = "head_coach"
    OFFENSIVE_COORDINATOR = "offensive_coordinator"
    DEFENSIVE_COORDINATOR = "defensive_coordinator"
    ASSISTANT_COACH = "assistant_coach"
    OTHER = "other"


class AgentName(StrEnum):
    STATISTICS = "STA"
    BEAT_REPORTER = "BRA"
    COACH_QUOTE = "CQA"
    SYNTHESIS = "SYN"


SPECIALIST_AGENTS: tuple[AgentName, ...] = (
    AgentName.STATISTICS,
    AgentName.BEAT_REPORTER,
    AgentName.COACH_QUOTE,
)


class SectionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class RunState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"


class RunFailureReason(StrEnum):
    NONE = "none"
    CONFIGURATION_ERROR = "configuration_error"
    VALIDATION_FAILED = "validation_failed"
    NO_GAMES = "no_games"
    STATISTICS_AGENT_FAILED = "statistics_agent_failed"
    SYNTHESIS_FAILED = "synthesis_failed"
    VERIFICATION_FAILED = "verification_failed"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    INTERNAL_ERROR = "internal_error"


class EvidenceKind(StrEnum):
    PLAY = "play"
    MEDIA = "media"
    QUOTE = "quote"
    STAT = "stat"


class StatMetric(StrEnum):
    RUN_RATE = "run_rate"
    PASS_RATE = "pass_rate"
    YARDS_PER_PLAY = "yards_per_play"
    THIRD_DOWN_CONVERSION_RATE = "third_down_conversion_rate"
    PLAY_COUNT = "play_count"


class StatUnit(StrEnum):
    RATE = "rate"
    YARDS = "yards"
    COUNT = "count"


class SplitDimension(StrEnum):
    OVERALL = "overall"
    DOWN = "down"
    DISTANCE_BUCKET = "distance_bucket"
    FIELD_POSITION = "field_position"
    RED_ZONE = "red_zone"
    WEEK = "week"
    CUSTOM = "custom"


class TendencyCategory(StrEnum):
    RUN_PASS = "run_pass"
    SITUATIONAL = "situational"
    FORMATION = "formation"
    PLAY_CALLING = "play_calling"
    RECENT_CHANGE = "recent_change"
    OTHER = "other"


class ObservationCategory(StrEnum):
    SCHEME = "scheme"
    PLAY_CALLING = "play_calling"
    RECENT_CHANGE = "recent_change"
    COACH_EXPLANATION = "coach_explanation"
    COACH_ADJUSTMENT = "coach_adjustment"
    PERFORMANCE = "performance"
    OTHER = "other"


class TendencyDirection(StrEnum):
    INCREASE = "increase"
    DECREASE = "decrease"
    STABLE = "stable"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class GapReason(StrEnum):
    NO_RETRIEVED_SOURCES = "no_retrieved_sources"
    NO_STORED_DATA = "no_stored_data"
    SMALL_SAMPLE = "small_sample"
    AGENT_UNAVAILABLE = "agent_unavailable"
    WEEK_NOT_SELECTED = "week_not_selected"
    OTHER = "other"


class WeakEvidenceReason(StrEnum):
    SMALL_SAMPLE = "small_sample"
    SINGLE_SOURCE = "single_source"
    UNVERIFIED_CLAIM = "unverified_claim"
    AGENT_UNAVAILABLE = "agent_unavailable"
    MISSING_EVIDENCE = "missing_evidence"
    LOW_CONFIDENCE = "low_confidence"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class ContractModel(BaseModel):
    """Frozen, strict base class for every stored or exchanged contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Source records
# ---------------------------------------------------------------------------


class Game(ContractModel):
    """One nflverse game."""

    game_id: GameId
    season: Season
    week: Week
    season_type: SeasonType = SeasonType.REGULAR
    game_date: date
    home_team: Team
    away_team: Team
    home_score: Annotated[int, Field(ge=0)] | None = None
    away_score: Annotated[int, Field(ge=0)] | None = None
    venue: TrimmedStr | None = None
    neutral_site: bool = False

    @model_validator(mode="after")
    def _validate_game(self) -> Self:
        if self.home_team == self.away_team:
            raise ValueError("home_team and away_team must differ")
        if (self.home_score is None) != (self.away_score is None):
            raise ValueError("home_score and away_score must both be set or both be absent")
        return self

    @property
    def teams(self) -> tuple[Team, Team]:
        return (self.away_team, self.home_team)

    def involves(self, team: Team) -> bool:
        return team in self.teams

    def opponent_of(self, team: Team) -> Team:
        if not self.involves(team):
            raise ValueError(f"{team.value} does not play in game {self.game_id}")
        return self.home_team if self.away_team == team else self.away_team

    @property
    def is_regular_season(self) -> bool:
        return self.season_type is SeasonType.REGULAR


def _game_sort_key(game: Game) -> tuple[int, date, str]:
    return (game.week, game.game_date, game.game_id)


class Play(ContractModel):
    """A normalized nflverse play.

    Only fields used by deterministic tendency calculations are kept. Missing
    values stay ``None`` and are reported by validation; they are never imputed.
    """

    game_id: GameId
    play_id: Annotated[int, Field(ge=0)]
    season: Season
    week: Week
    season_type: SeasonType = SeasonType.REGULAR
    posteam: Team | None = None
    defteam: Team | None = None
    down: Annotated[int, Field(ge=1, le=4)] | None = None
    yards_to_go: Annotated[int, Field(ge=0, le=100)] | None = None
    yardline_100: Annotated[int, Field(ge=0, le=100)] | None = None
    play_type: PlayType = PlayType.UNKNOWN
    yards_gained: Annotated[int, Field(ge=-99, le=200)] | None = None
    first_down: bool | None = None
    third_down_converted: bool | None = None
    third_down_failed: bool | None = None
    touchdown: bool | None = None
    turnover: bool | None = None

    @model_validator(mode="after")
    def _validate_play(self) -> Self:
        if self.posteam is not None and self.defteam is not None:
            if self.posteam == self.defteam:
                raise ValueError("posteam and defteam must differ")
        elif (self.posteam is None) != (self.defteam is None):
            raise ValueError("posteam and defteam must both be set or both be absent")
        if self.third_down_converted and self.third_down_failed:
            raise ValueError("a third down cannot be both converted and failed")
        if self.third_down_converted is not None and self.down != 3:
            raise ValueError("third_down_converted is only valid on third down")
        if self.third_down_failed is not None and self.down != 3:
            raise ValueError("third_down_failed is only valid on third down")
        return self

    @property
    def key(self) -> str:
        return play_key(self.game_id, self.play_id)

    @property
    def is_run(self) -> bool:
        return self.play_type is PlayType.RUN

    @property
    def is_pass(self) -> bool:
        return self.play_type is PlayType.PASS

    @property
    def is_scrimmage_play(self) -> bool:
        return self.is_run or self.is_pass

    @property
    def is_third_down_attempt(self) -> bool:
        return self.down == 3 and (
            self.third_down_converted is not None or self.third_down_failed is not None
        )


class MediaItem(ContractModel):
    """An article, film breakdown, or coach-quote source.

    ``body`` is stored verbatim. ``content_hash`` covers the retrievable text
    and drives index-reuse decisions, so editing an article forces a rebuild
    while re-ingesting identical text does not.
    """

    media_id: TrimmedStr = ""
    source_id: TrimmedStr
    source_type: MediaType
    title: TrimmedStr
    author: TrimmedStr | None = None
    publisher: TrimmedStr
    url: HttpUrl
    published_on: date
    season: Season
    week: Week | None = None
    game_id: GameId | None = None
    opponent: Team
    body: UntouchedStr
    retrieved_at: datetime
    content_hash: str = Field(default="", pattern=r"^$|^[0-9a-f]{64}$")

    @field_validator("retrieved_at")
    @classmethod
    def _utc_retrieved_at(cls, value: datetime) -> datetime:
        return _ensure_aware(value, "retrieved_at")

    @model_validator(mode="after")
    def _validate_media(self) -> Self:
        if not self.body.strip():
            raise ValueError("body must contain text")
        if self.source_type is MediaType.COACH_QUOTE and self.week is None:
            raise ValueError("coach-quote sources must record the week they belong to")
        if self.game_id is not None and self.week is None:
            raise ValueError("media tied to a game must also record its week")

        expected_id = make_media_id(self.source_id, str(self.url))
        if self.media_id and self.media_id != expected_id:
            raise ValueError(
                f"media_id {self.media_id!r} does not match source_id + url ({expected_id!r})"
            )
        if not self.media_id:
            object.__setattr__(self, "media_id", expected_id)

        expected_hash = make_content_hash(self.title, self.body)
        if self.content_hash and self.content_hash != expected_hash:
            raise ValueError("content_hash does not match title + body")
        if not self.content_hash:
            object.__setattr__(self, "content_hash", expected_hash)
        return self


class Quote(ContractModel):
    """A verbatim coach quote with its speaker, date, and source metadata."""

    quote_id: TrimmedStr = ""
    media_id: TrimmedStr
    speaker: TrimmedStr
    speaker_role: SpeakerRole
    team: Team
    season: Season
    week: Week
    game_id: GameId | None = None
    spoken_on: date
    publisher: TrimmedStr
    url: HttpUrl
    text: UntouchedStr
    char_start: Annotated[int, Field(ge=0)]
    char_end: Annotated[int, Field(ge=1)]
    occurrence: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def _validate_quote(self) -> Self:
        if not self.text.strip():
            raise ValueError("quote text must not be blank")
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")

        expected_id = make_quote_id(
            self.media_id,
            self.speaker,
            self.text,
            self.occurrence,
        )
        if self.quote_id and self.quote_id != expected_id:
            raise ValueError("quote_id does not match its article, speaker, text, and occurrence")
        if not self.quote_id:
            object.__setattr__(self, "quote_id", expected_id)
        return self

    def is_verbatim_in(self, media: MediaItem) -> bool:
        """Return whether the stored span still reproduces the quote exactly."""

        return (
            self.media_id == media.media_id
            and media.body[self.char_start : self.char_end] == self.text
        )


class EvidenceRef(ContractModel):
    """A pointer from a claim to one stored play, media chunk, quote, or stat table."""

    evidence_id: TrimmedStr = ""
    kind: EvidenceKind
    label: TrimmedStr
    game_id: GameId | None = None
    play_id: Annotated[int, Field(ge=0)] | None = None
    media_id: TrimmedStr | None = None
    quote_id: TrimmedStr | None = None
    stat_table_id: TrimmedStr | None = None
    chunk_id: TrimmedStr | None = None
    snippet: Snippet | None = None
    source_url: HttpUrl | None = None

    @model_validator(mode="after")
    def _validate_evidence(self) -> Self:
        if self.snippet is not None and not self.snippet.strip():
            raise ValueError("snippet must not be blank when provided")

        if self.kind is EvidenceKind.PLAY:
            if self.game_id is None or self.play_id is None:
                raise ValueError("play evidence requires game_id and play_id")
            if any((self.media_id, self.quote_id, self.stat_table_id, self.chunk_id)):
                raise ValueError("play evidence cannot carry media, quote, stat, or chunk IDs")
            expected_id = make_play_evidence_id(self.game_id, self.play_id)
        elif self.kind is EvidenceKind.MEDIA:
            if self.media_id is None:
                raise ValueError("media evidence requires media_id")
            if any((self.game_id, self.play_id, self.quote_id, self.stat_table_id)):
                raise ValueError("media evidence cannot carry play, quote, or stat IDs")
            expected_id = make_media_evidence_id(self.media_id, self.chunk_id)
        elif self.kind is EvidenceKind.QUOTE:
            if self.quote_id is None:
                raise ValueError("quote evidence requires quote_id")
            if any((self.game_id, self.play_id, self.stat_table_id, self.chunk_id)):
                raise ValueError("quote evidence cannot carry play, stat, or chunk IDs")
            expected_id = make_quote_evidence_id(self.quote_id)
        else:
            if self.stat_table_id is None:
                raise ValueError("stat evidence requires stat_table_id")
            if any((self.game_id, self.play_id, self.media_id, self.quote_id, self.chunk_id)):
                raise ValueError("stat evidence cannot carry play, media, quote, or chunk IDs")
            expected_id = make_stat_evidence_id(self.stat_table_id, self.label)

        if self.evidence_id and self.evidence_id != expected_id:
            raise ValueError(
                f"evidence_id {self.evidence_id!r} does not match its locator ({expected_id!r})"
            )
        if not self.evidence_id:
            object.__setattr__(self, "evidence_id", expected_id)
        return self

    @classmethod
    def for_play(
        cls,
        game_id: str,
        play_id: int,
        *,
        label: str,
        snippet: str | None = None,
    ) -> EvidenceRef:
        return cls(
            kind=EvidenceKind.PLAY,
            game_id=game_id,
            play_id=play_id,
            label=label,
            snippet=snippet,
        )

    @classmethod
    def for_media(
        cls,
        media_id: str,
        *,
        label: str,
        chunk_id: str | None = None,
        snippet: str | None = None,
        source_url: HttpUrl | None = None,
    ) -> EvidenceRef:
        return cls(
            kind=EvidenceKind.MEDIA,
            media_id=media_id,
            chunk_id=chunk_id,
            label=label,
            snippet=snippet,
            source_url=source_url,
        )

    @classmethod
    def for_quote(
        cls,
        quote_id: str,
        *,
        label: str,
        media_id: str | None = None,
        snippet: str | None = None,
        source_url: HttpUrl | None = None,
    ) -> EvidenceRef:
        return cls(
            kind=EvidenceKind.QUOTE,
            quote_id=quote_id,
            media_id=media_id,
            label=label,
            snippet=snippet,
            source_url=source_url,
        )

    @classmethod
    def for_stat(
        cls,
        stat_table_id: str,
        *,
        label: str,
        snippet: str | None = None,
    ) -> EvidenceRef:
        return cls(
            kind=EvidenceKind.STAT,
            stat_table_id=stat_table_id,
            label=label,
            snippet=snippet,
        )


# ---------------------------------------------------------------------------
# Deterministic statistics
# ---------------------------------------------------------------------------


class StatRow(ContractModel):
    """One row of a deterministic statistic table."""

    label: TrimmedStr
    value: float | None = None
    n: Annotated[int, Field(ge=0)]
    numerator: int | None = None
    denominator: Annotated[int, Field(gt=0)] | None = None
    unit: StatUnit
    suppressed: bool = False
    notes: list[TrimmedStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_row(self) -> Self:
        if self.denominator is not None and self.denominator != self.n:
            raise ValueError("denominator must equal n (the number of plays included)")
        if self.n == 0:
            if self.value is not None:
                raise ValueError("a row with no plays must not report a value")
            if not self.suppressed:
                raise ValueError("a row with no plays must be marked suppressed")
        if self.unit is StatUnit.RATE and self.value is not None and not 0.0 <= self.value <= 1.0:
            raise ValueError("rate values must be between 0 and 1")
        if (
            self.unit is StatUnit.RATE
            and self.numerator is not None
            and self.denominator is not None
            and not 0 <= self.numerator <= self.denominator
        ):
            raise ValueError("rate numerator must be between 0 and the denominator")
        return self


class StatTable(ContractModel):
    """A deterministic table for one metric and one split."""

    stat_table_id: TrimmedStr = ""
    metric: StatMetric
    title: TrimmedStr
    dimension: SplitDimension
    dimension_value: TrimmedStr | None = None
    rows: list[StatRow] = Field(default_factory=list)
    n_plays: Annotated[int, Field(ge=0)]
    game_ids: list[GameId] = Field(default_factory=list)
    weeks: list[Week] = Field(default_factory=list)
    min_sample: Annotated[int, Field(ge=1)]
    suppressed: bool = False
    notes: list[TrimmedStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_table(self) -> Self:
        _ensure_sorted_unique(self.game_ids, "game_ids")
        _ensure_sorted_unique(self.weeks, "weeks")

        if self.n_plays > 0 and (not self.game_ids or not self.weeks):
            raise ValueError("a non-empty table must record its games and weeks")

        expected_suppressed = self.n_plays < self.min_sample
        if self.suppressed != expected_suppressed:
            raise ValueError(
                f"suppressed must be {expected_suppressed} for n={self.n_plays}, "
                f"min_sample={self.min_sample}"
            )

        expected_unit = _expected_unit(self.metric)
        if len({row.label for row in self.rows}) != len(self.rows):
            raise ValueError("stat rows must not repeat a label")
        for row in self.rows:
            if row.unit is not expected_unit:
                raise ValueError(f"{self.metric.value} rows must use unit {expected_unit.value!r}")
            expected_row_suppressed = row.n < self.min_sample
            if row.suppressed != expected_row_suppressed:
                raise ValueError(
                    f"row {row.label!r} suppressed must be {expected_row_suppressed} "
                    f"for n={row.n}, min_sample={self.min_sample}"
                )
            if row.n > self.n_plays:
                raise ValueError(f"row {row.label!r} has more plays than the table")

        expected_id = make_stat_table_id(
            self.metric.value,
            self.dimension.value,
            self.dimension_value,
            self.weeks,
            self.game_ids,
        )
        if self.stat_table_id and self.stat_table_id != expected_id:
            raise ValueError("stat_table_id does not match metric, split, weeks, and games")
        if not self.stat_table_id:
            object.__setattr__(self, "stat_table_id", expected_id)
        return self

    def evidence(self, row_label: str | None = None) -> EvidenceRef:
        """Return an evidence pointer to this table or one of its rows."""

        label = f"{self.metric.value} / {self.dimension.value}"
        if self.dimension_value:
            label += f" / {self.dimension_value}"
        if row_label:
            label += f" / {row_label}"
        return EvidenceRef.for_stat(
            self.stat_table_id,
            label=label,
            snippet=f"n={self.n_plays}, min_sample={self.min_sample}",
        )


def _expected_unit(metric: StatMetric) -> StatUnit:
    if metric in {
        StatMetric.RUN_RATE,
        StatMetric.PASS_RATE,
        StatMetric.THIRD_DOWN_CONVERSION_RATE,
    }:
        return StatUnit.RATE
    if metric is StatMetric.YARDS_PER_PLAY:
        return StatUnit.YARDS
    return StatUnit.COUNT


# ---------------------------------------------------------------------------
# Findings and agent output
# ---------------------------------------------------------------------------


class AgentFinding(ContractModel):
    """Base class for a claim that must cite at least one evidence item."""

    finding_id: TrimmedStr = ""
    statement: TrimmedStr
    evidence_ids: list[TrimmedStr] = Field(min_length=1)
    weeks: list[Week] = Field(min_length=1)
    confidence: Confidence = Confidence.MEDIUM
    weak_evidence: bool = False
    direction: TendencyDirection = TendencyDirection.UNKNOWN
    metric: TrimmedStr | None = None
    value: float | None = None
    comparison: TrimmedStr | None = None
    treat_as_context: bool = False

    @property
    def finding_kind(self) -> str:
        return "finding"

    @model_validator(mode="after")
    def _validate_finding(self) -> Self:
        _ensure_sorted_unique(self.weeks, "weeks")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids must not contain duplicates")
        if self.weak_evidence and self.confidence is Confidence.HIGH:
            raise ValueError("weak evidence cannot carry high confidence")
        if self.value is not None and self.metric is None:
            raise ValueError("a numeric value requires the metric it belongs to")
        if self.finding_kind == "tendency" and self.treat_as_context:
            raise ValueError("a coach statement is context, not a tendency")

        expected_id = make_finding_id(
            self.finding_kind,
            self.statement,
            self.evidence_ids,
            self.weeks,
        )
        if self.finding_id and self.finding_id != expected_id:
            raise ValueError("finding_id does not match claim, evidence, and weeks")
        if not self.finding_id:
            object.__setattr__(self, "finding_id", expected_id)
        return self


class Tendency(AgentFinding):
    """A described tendency with an explicit direction (requirement 20)."""

    category: TendencyCategory
    direction: TendencyDirection

    @property
    def finding_kind(self) -> str:
        return "tendency"


class Observation(AgentFinding):
    """Qualitative reporting, scheme context, or a coach statement."""

    category: ObservationCategory

    @property
    def finding_kind(self) -> str:
        return "observation"


class AgentOutput(ContractModel):
    """Shared output contract for STA, BRA, and CQA (requirement 25).

    Identity, games, and weeks are stamped by deterministic Python so a model
    cannot drop them or drift between agents.
    """

    agent: AgentName
    opponent: Team
    games: list[Game] = Field(min_length=1)
    weeks: list[Week] = Field(min_length=1)
    tendencies: list[Tendency]
    observations: list[Observation]
    evidence: list[EvidenceRef]
    limitations: list[TrimmedStr]
    stat_tables: list[StatTable] = Field(default_factory=list)
    status: SectionStatus = SectionStatus.COMPLETED
    prompt_version: TrimmedStr
    model: TrimmedStr | None = None

    @model_validator(mode="after")
    def _validate_output(self) -> Self:
        if self.agent not in SPECIALIST_AGENTS:
            raise ValueError("AgentOutput is only produced by STA, BRA, or CQA")
        _ensure_sorted_unique(self.weeks, "weeks")
        _ensure_sorted_unique(self.limitations, "limitations")

        game_ids = [game.game_id for game in self.games]
        if len(set(game_ids)) != len(game_ids):
            raise ValueError("games must not repeat a game_id")
        if self.games != sorted(self.games, key=_game_sort_key):
            raise ValueError("games must be sorted by week, date, and game_id")
        if any(not game.involves(self.opponent) for game in self.games):
            raise ValueError("every game must involve the selected opponent")
        if any(not game.is_regular_season for game in self.games):
            raise ValueError("only regular-season games may be analyzed")
        if any(game.week not in self.weeks for game in self.games):
            raise ValueError("every game week must be one of the selected weeks")
        if len({game.season for game in self.games}) > 1:
            raise ValueError("all games must belong to one season")

        if self.status is not SectionStatus.COMPLETED:
            if self.tendencies or self.observations:
                raise ValueError("an unavailable or failed section must not report findings")
            if not self.limitations:
                raise ValueError("an unavailable or failed section must explain itself")
        elif not self.tendencies and not self.observations and not self.limitations:
            raise ValueError("an empty result must carry an explicit limitation")

        _validate_evidence_lists(self.evidence, "evidence")
        available_ids = {item.evidence_id for item in self.evidence}

        referenced: set[str] = set()
        for finding in [*self.tendencies, *self.observations]:
            if any(week not in self.weeks for week in finding.weeks):
                raise ValueError("finding weeks must be a subset of the selected weeks")
            referenced.update(finding.evidence_ids)
        missing = referenced - available_ids
        if missing:
            raise ValueError(f"findings cite evidence not present on the output: {sorted(missing)}")

        if self.stat_tables and self.agent is not AgentName.STATISTICS:
            raise ValueError("only the statistics agent may attach stat tables")
        _validate_stat_tables(self.stat_tables, weeks=self.weeks, game_ids=game_ids)
        table_ids = {table.stat_table_id for table in self.stat_tables}
        for item in self.evidence:
            if item.kind is EvidenceKind.STAT and item.stat_table_id not in table_ids:
                raise ValueError(
                    f"stat evidence {item.evidence_id!r} does not resolve to a table on this output"
                )
            if self.agent is AgentName.STATISTICS and item.kind in {
                EvidenceKind.MEDIA,
                EvidenceKind.QUOTE,
            }:
                raise ValueError("the statistics agent cannot cite retrieved media or quotes")
            if self.agent is AgentName.BEAT_REPORTER and item.kind is not EvidenceKind.MEDIA:
                raise ValueError("the beat-reporter agent may only cite media evidence")
            if self.agent is AgentName.COACH_QUOTE and item.kind not in {
                EvidenceKind.MEDIA,
                EvidenceKind.QUOTE,
            }:
                raise ValueError("the coach-quote agent may only cite media or quote evidence")
        return self

    @property
    def is_available(self) -> bool:
        return self.status is SectionStatus.COMPLETED


def _validate_evidence_lists(evidence: Sequence[EvidenceRef], field_name: str) -> None:
    ids = [item.evidence_id for item in evidence]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{field_name} must not repeat an evidence_id")
    if list(ids) != sorted(ids):
        raise ValueError(f"{field_name} must be sorted by evidence_id")


def _validate_stat_tables(
    tables: Sequence[StatTable],
    *,
    weeks: Sequence[int],
    game_ids: Iterable[str],
) -> None:
    ids = [table.stat_table_id for table in tables]
    if len(set(ids)) != len(ids):
        raise ValueError("stat_tables must not repeat a stat_table_id")
    if list(ids) != sorted(ids):
        raise ValueError("stat_tables must be sorted by stat_table_id")

    allowed_game_ids = set(game_ids)
    for table in tables:
        if not set(table.weeks).issubset(weeks):
            raise ValueError("stat table weeks must be a subset of the selected weeks")
        if not set(table.game_ids).issubset(allowed_game_ids):
            raise ValueError("stat table games must be a subset of the selected games")


# ---------------------------------------------------------------------------
# Validation and run status
# ---------------------------------------------------------------------------


class ValidationIssue(ContractModel):
    """One structured error, warning, or informational note."""

    code: TrimmedStr
    message: TrimmedStr
    severity: IssueSeverity
    subject_kind: TrimmedStr | None = None
    subject_id: TrimmedStr | None = None
    requires_human_review: bool = False
    details: dict[TrimmedStr, TrimmedStr] = Field(default_factory=dict)


class ValidationResult(ContractModel):
    """Structured outcome of source, schema, and alignment checks."""

    ok: bool
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    checked: dict[TrimmedStr, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        if any(issue.severity is not IssueSeverity.ERROR for issue in self.errors):
            raise ValueError("errors must all have error severity")
        if any(issue.severity is IssueSeverity.ERROR for issue in self.warnings):
            raise ValueError("warnings must not contain error-severity issues")
        if self.ok == bool(self.errors):
            raise ValueError("ok must be true exactly when there are no errors")
        return self

    @property
    def requires_human_review(self) -> bool:
        return any(issue.requires_human_review for issue in [*self.errors, *self.warnings])

    @classmethod
    def merge(cls, *results: ValidationResult) -> ValidationResult:
        errors = [issue for result in results for issue in result.errors]
        warnings = [issue for result in results for issue in result.warnings]
        checked: dict[str, int] = {}
        for result in results:
            for name, count in result.checked.items():
                checked[name] = checked.get(name, 0) + count
        return cls(ok=not errors, errors=errors, warnings=warnings, checked=checked)


class RunStatus(ContractModel):
    """Overall and per-agent progress for one report run."""

    state: RunState = RunState.PENDING
    failure_reason: RunFailureReason = RunFailureReason.NONE
    message: TrimmedStr | None = None
    agent_status: dict[AgentName, SectionStatus] = Field(default_factory=dict)
    current_step: TrimmedStr | None = None
    progress: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    deadline_seconds: Annotated[float, Field(gt=0.0)] | None = None

    @field_validator("started_at", "finished_at")
    @classmethod
    def _utc_times(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _ensure_aware(value, "run timestamp")

    @model_validator(mode="after")
    def _validate_status(self) -> Self:
        if (
            self.started_at is not None
            and self.finished_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("finished_at must not precede started_at")

        if self.state is RunState.PENDING:
            if self.started_at is not None or self.finished_at is not None:
                raise ValueError("a pending run cannot have started or finished")
            if self.failure_reason is not RunFailureReason.NONE:
                raise ValueError("a pending run cannot have a failure reason")
            if self.progress != 0.0:
                raise ValueError("a pending run must report zero progress")

        elif self.state is RunState.RUNNING:
            if self.started_at is None:
                raise ValueError("a running run must record started_at")
            if self.finished_at is not None:
                raise ValueError("a running run cannot be finished")
            if self.failure_reason is not RunFailureReason.NONE:
                raise ValueError("a running run cannot have a failure reason")

        elif self.state is RunState.COMPLETED:
            if self.finished_at is None:
                raise ValueError("a completed run must record finished_at")
            if self.failure_reason is not RunFailureReason.NONE:
                raise ValueError("a completed run cannot have a failure reason")
            if self.progress != 1.0:
                raise ValueError("a completed run must report full progress")
            if not self.agent_status:
                raise ValueError("a completed run must record agent statuses")
            if any(status is not SectionStatus.COMPLETED for status in self.agent_status.values()):
                raise ValueError("a completed run requires every agent to be completed")

        elif self.state is RunState.DEGRADED:
            if self.finished_at is None:
                raise ValueError("a degraded run must record finished_at")
            if self.failure_reason is not RunFailureReason.NONE:
                raise ValueError("a degraded run is partial, not failed")
            if self.progress != 1.0:
                raise ValueError("a degraded run must report full progress")
            statuses = list(self.agent_status.values())
            if not statuses:
                raise ValueError("a degraded run must record agent statuses")
            if not any(status is SectionStatus.COMPLETED for status in statuses):
                raise ValueError("a degraded run needs at least one completed section")
            if all(status is SectionStatus.COMPLETED for status in statuses):
                raise ValueError("a degraded run needs at least one unavailable section")

        else:  # FAILED
            if self.failure_reason is RunFailureReason.NONE:
                raise ValueError("a failed run must record why it failed")
            if self.finished_at is None:
                raise ValueError("a failed run must record finished_at")
            if not self.message:
                raise ValueError("a failed run must explain the failure to the user")
        return self

    @property
    def elapsed_seconds(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def is_terminal(self) -> bool:
        return self.state in {RunState.COMPLETED, RunState.DEGRADED, RunState.FAILED}


# ---------------------------------------------------------------------------
# Synthesis structures
# ---------------------------------------------------------------------------


class Agreement(ContractModel):
    agreement_id: TrimmedStr = ""
    topic: TrimmedStr
    statement: TrimmedStr
    agents: list[AgentName] = Field(min_length=2)
    evidence_ids: list[TrimmedStr] = Field(min_length=1)
    weeks: list[Week] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_agreement(self) -> Self:
        _ensure_sorted_unique(self.weeks, "weeks")
        if len(set(self.agents)) != len(self.agents):
            raise ValueError("agents must not repeat")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids must not repeat")

        expected_id = make_agreement_id(
            self.topic,
            self.statement,
            [agent.value for agent in self.agents],
            self.weeks,
        )
        if self.agreement_id and self.agreement_id != expected_id:
            raise ValueError("agreement_id does not match its content")
        if not self.agreement_id:
            object.__setattr__(self, "agreement_id", expected_id)
        return self


class ConflictPosition(ContractModel):
    agent: AgentName
    statement: TrimmedStr
    evidence_ids: list[TrimmedStr] = Field(min_length=1)


class Conflict(ContractModel):
    conflict_id: TrimmedStr = ""
    topic: TrimmedStr
    description: TrimmedStr
    positions: list[ConflictPosition] = Field(min_length=2)
    evidence_ids: list[TrimmedStr] = Field(min_length=1)
    weeks: list[Week] = Field(min_length=1)
    resolution: TrimmedStr | None = None

    @model_validator(mode="after")
    def _validate_conflict(self) -> Self:
        _ensure_sorted_unique(self.weeks, "weeks")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids must not repeat")
        position_evidence = {
            evidence_id for position in self.positions for evidence_id in position.evidence_ids
        }
        missing = position_evidence - set(self.evidence_ids)
        if missing:
            raise ValueError(
                f"conflict evidence_ids must include position evidence: {sorted(missing)}"
            )

        expected_id = make_conflict_id(
            self.topic,
            self.description,
            [position.statement for position in self.positions],
            self.weeks,
        )
        if self.conflict_id and self.conflict_id != expected_id:
            raise ValueError("conflict_id does not match its content")
        if not self.conflict_id:
            object.__setattr__(self, "conflict_id", expected_id)
        return self


class Gap(ContractModel):
    gap_id: TrimmedStr = ""
    topic: TrimmedStr
    description: TrimmedStr
    reason: GapReason
    agents: list[AgentName] = Field(default_factory=list)
    weeks: list[Week] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_gap(self) -> Self:
        _ensure_sorted_unique(self.weeks, "weeks")
        if len(set(self.agents)) != len(self.agents):
            raise ValueError("agents must not repeat")

        expected_id = make_gap_id(self.topic, self.description, self.weeks)
        if self.gap_id and self.gap_id != expected_id:
            raise ValueError("gap_id does not match its content")
        if not self.gap_id:
            object.__setattr__(self, "gap_id", expected_id)
        return self


class WeakEvidence(ContractModel):
    weak_evidence_id: TrimmedStr = ""
    topic: TrimmedStr
    description: TrimmedStr
    reason: WeakEvidenceReason
    evidence_ids: list[TrimmedStr] = Field(default_factory=list)
    weeks: list[Week] = Field(min_length=1)
    agent: AgentName | None = None

    @model_validator(mode="after")
    def _validate_weak_evidence(self) -> Self:
        _ensure_sorted_unique(self.weeks, "weeks")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids must not repeat")

        expected_id = make_weak_evidence_id(
            self.topic,
            self.description,
            self.evidence_ids,
            self.weeks,
        )
        if self.weak_evidence_id and self.weak_evidence_id != expected_id:
            raise ValueError("weak_evidence_id does not match its content")
        if not self.weak_evidence_id:
            object.__setattr__(self, "weak_evidence_id", expected_id)
        return self


class DataUsage(ContractModel):
    """Counts of stored data that fed the report (requirement 5)."""

    games: Annotated[int, Field(ge=0)]
    plays: Annotated[int, Field(ge=0)]
    media_items: Annotated[int, Field(ge=0)]
    quotes: Annotated[int, Field(ge=0)]
    source_ids: list[TrimmedStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_usage(self) -> Self:
        _ensure_sorted_unique(self.source_ids, "source_ids")
        return self


class Report(ContractModel):
    """The final evidence-backed report."""

    report_id: TrimmedStr = ""
    opponent: Team
    season: Season
    weeks: list[Week] = Field(min_length=1)
    games: list[Game] = Field(min_length=1)
    generated_at: datetime
    data_revision: TrimmedStr
    prompt_version: TrimmedStr
    model: TrimmedStr | None = None
    status: RunStatus
    overview: TrimmedStr
    data_used: DataUsage
    stat_tables: list[StatTable] = Field(default_factory=list)
    tendencies: list[Tendency] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    agreements: list[Agreement] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    weak_evidence: list[WeakEvidence] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    agent_outputs: list[AgentOutput] = Field(min_length=3)

    @field_validator("generated_at")
    @classmethod
    def _utc_generated_at(cls, value: datetime) -> datetime:
        return _ensure_aware(value, "generated_at")

    @model_validator(mode="after")
    def _validate_report(self) -> Self:
        if self.status.state not in {RunState.COMPLETED, RunState.DEGRADED}:
            raise ValueError("a report is only produced for completed or degraded runs")
        if any(issue.severity is IssueSeverity.ERROR for issue in self.warnings):
            raise ValueError("report warnings must not contain errors")

        _ensure_sorted_unique(self.weeks, "weeks")
        game_ids = [game.game_id for game in self.games]
        if len(set(game_ids)) != len(game_ids):
            raise ValueError("games must not repeat a game_id")
        if self.games != sorted(self.games, key=_game_sort_key):
            raise ValueError("games must be sorted by week, date, and game_id")
        if any(not game.involves(self.opponent) for game in self.games):
            raise ValueError("every game must involve the selected opponent")
        if any(not game.is_regular_season for game in self.games):
            raise ValueError("only regular-season games may be analyzed")
        if any(game.week not in self.weeks for game in self.games):
            raise ValueError("every game week must be one of the selected weeks")
        if any(game.season != self.season for game in self.games):
            raise ValueError("every game must belong to the report season")
        if self.data_used.games != len(self.games):
            raise ValueError("data_used.games must match the game list")

        output_agents = [output.agent for output in self.agent_outputs]
        if len(set(output_agents)) != len(output_agents):
            raise ValueError("agent_outputs must not repeat an agent")
        if set(output_agents) != set(SPECIALIST_AGENTS):
            raise ValueError("a report needs exactly one STA, BRA, and CQA output")
        for output in self.agent_outputs:
            if output.opponent is not self.opponent:
                raise ValueError("every agent output must target the report opponent")
            if output.weeks != self.weeks:
                raise ValueError("every agent output must cover the selected weeks")
            if output.games != self.games:
                raise ValueError("every agent output must use the report games")
            expected_status = self.status.agent_status.get(output.agent)
            if expected_status is not output.status:
                raise ValueError(
                    f"run status for {output.agent.value} does not match its output status"
                )

        if self.status.agent_status.get(AgentName.SYNTHESIS) is not SectionStatus.COMPLETED:
            raise ValueError("synthesis must complete before a report is produced")
        degraded_outputs = [output for output in self.agent_outputs if not output.is_available]
        if self.status.state is RunState.COMPLETED and degraded_outputs:
            raise ValueError("a completed run cannot contain an unavailable section")
        if self.status.state is RunState.DEGRADED and not degraded_outputs:
            raise ValueError("a degraded report must contain an unavailable section")
        statistics_output = next(
            output for output in self.agent_outputs if output.agent is AgentName.STATISTICS
        )
        if not statistics_output.is_available:
            raise ValueError("no report may be produced when the statistics agent fails")

        _validate_evidence_lists(self.evidence, "evidence")
        available_evidence = {item.evidence_id for item in self.evidence}
        referenced: set[str] = set()

        for finding in [*self.tendencies, *self.observations]:
            if any(week not in self.weeks for week in finding.weeks):
                raise ValueError("finding weeks must be a subset of the selected weeks")
            referenced.update(finding.evidence_ids)
        for agreement in self.agreements:
            if not set(agreement.weeks).issubset(self.weeks):
                raise ValueError("agreement weeks must be a subset of the selected weeks")
            referenced.update(agreement.evidence_ids)
        for conflict in self.conflicts:
            if not set(conflict.weeks).issubset(self.weeks):
                raise ValueError("conflict weeks must be a subset of the selected weeks")
            referenced.update(conflict.evidence_ids)
        for gap in self.gaps:
            if not set(gap.weeks).issubset(self.weeks):
                raise ValueError("gap weeks must be a subset of the selected weeks")
        for weak in self.weak_evidence:
            if not set(weak.weeks).issubset(self.weeks):
                raise ValueError("weak-evidence weeks must be a subset of the selected weeks")
            referenced.update(weak.evidence_ids)
        for output in self.agent_outputs:
            referenced.update(item.evidence_id for item in output.evidence)

        missing = referenced - available_evidence
        if missing:
            raise ValueError(f"report claims cite evidence not present: {sorted(missing)}")

        _validate_stat_tables(self.stat_tables, weeks=self.weeks, game_ids=game_ids)
        agent_tables = {
            table.stat_table_id: table
            for output in self.agent_outputs
            for table in output.stat_tables
        }
        report_tables = {table.stat_table_id: table for table in self.stat_tables}
        if report_tables.keys() != agent_tables.keys():
            raise ValueError("report stat tables must exactly match the agents' stat tables")

        expected_id = make_report_id(
            self.opponent.value,
            self.weeks,
            self.data_revision,
            self.prompt_version,
        )
        if self.report_id and self.report_id != expected_id:
            raise ValueError("report_id does not match opponent, weeks, data, and prompt version")
        if not self.report_id:
            object.__setattr__(self, "report_id", expected_id)
        return self

    @property
    def is_partial(self) -> bool:
        return self.status.state is RunState.DEGRADED

    @property
    def unavailable_sections(self) -> tuple[AgentName, ...]:
        return tuple(output.agent for output in self.agent_outputs if not output.is_available)

    @property
    def human_review_required(self) -> bool:
        return bool(self.weak_evidence) or any(
            issue.requires_human_review for issue in self.warnings
        )


# Re-export the identifiers module's helpers that callers commonly pair with models.
__all__: tuple[str, ...] = (
    "SPECIALIST_AGENTS",
    "TEAM_DISPLAY_NAMES",
    "AgentFinding",
    "AgentName",
    "AgentOutput",
    "Agreement",
    "Confidence",
    "Conflict",
    "ConflictPosition",
    "ContractModel",
    "DataUsage",
    "EvidenceKind",
    "EvidenceRef",
    "Game",
    "GameId",
    "Gap",
    "GapReason",
    "IssueSeverity",
    "MediaItem",
    "MediaType",
    "Observation",
    "ObservationCategory",
    "Play",
    "PlayType",
    "Quote",
    "Report",
    "RunFailureReason",
    "RunState",
    "RunStatus",
    "Season",
    "SeasonType",
    "SectionStatus",
    "SpeakerRole",
    "SplitDimension",
    "StatMetric",
    "StatRow",
    "StatTable",
    "StatUnit",
    "Team",
    "Tendency",
    "TendencyCategory",
    "TendencyDirection",
    "ValidationIssue",
    "ValidationResult",
    "WeakEvidence",
    "WeakEvidenceReason",
    "Week",
    "team_display_name",
    "teams_alphabetical",
)
