"""Deterministic tendency calculations for one opponent's offense (Stage 4).

The engine is pure Python over the frozen :class:`~opponent_tendencies.models.Play`
contracts: no pandas, no randomness, no clock. Identical inputs produce
byte-identical :meth:`StatsBundle.to_payload` output, which the golden-file test
pins. Deterministic Python owns every calculation, sort, filter, and duplicate
check (requirement 24); nothing is imputed:

* a play whose split key (down, distance, field position) is missing lands in an
  explicit ``"Unknown"`` bucket with a table note explaining the gap;
* a play missing ``yards_gained`` or a third-down outcome is excluded from the
  affected metric, with the exclusion counted in :class:`DataQuality` and
  explained in the table notes;
* duplicate ``(game_id, play_id)`` input is rejected outright.

Populations are the selected opponent's offensive plays from the selected
games. Scrimmage plays (run or pass) drive run rate, pass rate, and yards per
play; third-down attempts are scrimmage plays on third down with a recorded
outcome, which matches the standard conversion statistic and keeps flagged
penalties, punts, and kneels from counting as attempts. Every row reports ``n``
and rows below ``min_sample`` keep their value but are flagged ``suppressed``
(requirement 11). :func:`compare_week_windows` compares two disjoint week
windows per metric and classifies the direction of change with fixed
thresholds, reporting ``unknown`` when either window is below ``min_sample``
(requirement 15).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Final

from opponent_tendencies.models import (
    Game,
    Play,
    PlayType,
    SplitDimension,
    StatMetric,
    StatRow,
    StatTable,
    StatUnit,
    Team,
    TendencyDirection,
)

__all__ = [
    "DEFAULT_MIN_SAMPLE",
    "DISTANCE_LABELS",
    "DOWN_LABELS",
    "FIELD_POSITION_LABELS",
    "STABLE_RATE_DELTA",
    "STABLE_YARDS_DELTA",
    "DataQuality",
    "DuplicatePlayKeyError",
    "InvalidSelectionError",
    "StatsBundle",
    "StatsError",
    "WindowComparison",
    "analyze_opponent",
    "compare_week_windows",
    "distance_bucket",
    "down_label",
    "field_position_bucket",
]

#: Minimum plays per row before the value is flagged as an unreliable split.
DEFAULT_MIN_SAMPLE: Final = 10

#: Absolute change below which a rate comparison is classified as stable.
STABLE_RATE_DELTA: Final = 0.05

#: Absolute change below which a yards-per-play comparison is stable.
STABLE_YARDS_DELTA: Final = 0.5

_RATE_DECIMALS: Final = 4
_YARDS_DECIMALS: Final = 2

DOWN_LABELS: Final[tuple[str, ...]] = ("1st down", "2nd down", "3rd down", "4th down")
UNKNOWN_DOWN_LABEL: Final = "Unknown down"
DISTANCE_LABELS: Final[tuple[str, ...]] = ("Short (0-3)", "Medium (4-6)", "Long (7+)")
UNKNOWN_DISTANCE_LABEL: Final = "Unknown distance"
FIELD_POSITION_LABELS: Final[tuple[str, ...]] = (
    "Red zone (0-20)",
    "Opponent territory (21-50)",
    "Own territory (51-100)",
)
UNKNOWN_FIELD_POSITION_LABEL: Final = "Unknown field position"

_UNKNOWN_LABELS: Final = frozenset(
    {UNKNOWN_DOWN_LABEL, UNKNOWN_DISTANCE_LABEL, UNKNOWN_FIELD_POSITION_LABEL}
)

_SCRIMMAGE_DIMENSIONS: Final[tuple[SplitDimension, ...]] = (
    SplitDimension.OVERALL,
    SplitDimension.DOWN,
    SplitDimension.DISTANCE_BUCKET,
    SplitDimension.FIELD_POSITION,
    SplitDimension.WEEK,
)
_THIRD_DOWN_DIMENSIONS: Final[tuple[SplitDimension, ...]] = (
    SplitDimension.OVERALL,
    SplitDimension.DISTANCE_BUCKET,
    SplitDimension.FIELD_POSITION,
    SplitDimension.WEEK,
)


class StatsError(ValueError):
    """Base class for rejected statistics-engine input."""


class DuplicatePlayKeyError(StatsError):
    """The same ``(game_id, play_id)`` appeared twice in the analyzed input."""


class InvalidSelectionError(StatsError):
    """The opponent, game, or week-window selection cannot be analyzed."""


@dataclass(frozen=True)
class DataQuality:
    """Counts of what was included, excluded, or left unrecorded (requirement 24)."""

    offensive_plays: int
    scrimmage_plays: int
    third_down_attempts: int
    unknown_play_types: int
    missing_down: int
    missing_yards_to_go: int
    missing_yardline_100: int
    missing_yards_gained: int
    missing_third_down_outcomes: int

    def to_payload(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class WindowComparison:
    """One metric compared across two disjoint week windows (requirement 15).

    ``delta`` is the late value minus the early value, computed from the
    published (rounded) values so it always matches what a reader sees.
    ``reliable`` is false when either window is below ``min_sample``; such
    comparisons still report values but never a direction.
    """

    metric: StatMetric
    early_weeks: tuple[int, ...]
    late_weeks: tuple[int, ...]
    early_n: int
    early_value: float | None
    late_n: int
    late_value: float | None
    delta: float | None
    direction: TendencyDirection
    reliable: bool
    min_sample: int

    def to_payload(self) -> dict[str, object]:
        return {
            "metric": self.metric.value,
            "early_weeks": list(self.early_weeks),
            "late_weeks": list(self.late_weeks),
            "early_n": self.early_n,
            "early_value": self.early_value,
            "late_n": self.late_n,
            "late_value": self.late_value,
            "delta": self.delta,
            "direction": self.direction.value,
            "reliable": self.reliable,
            "min_sample": self.min_sample,
        }


@dataclass(frozen=True)
class StatsBundle:
    """Deterministic tables and quality counts for one opponent and selection.

    Tables are sorted by ``stat_table_id`` and draw only on the selected games
    and weeks, so the statistics agent can attach them to its
    :class:`~opponent_tendencies.models.AgentOutput` unchanged.
    """

    opponent: Team
    weeks: tuple[int, ...]
    game_ids: tuple[str, ...]
    min_sample: int
    tables: tuple[StatTable, ...]
    quality: DataQuality

    def to_payload(self) -> dict[str, object]:
        """Return the canonical, JSON-safe document for this bundle."""

        return {
            "opponent": self.opponent.value,
            "weeks": list(self.weeks),
            "game_ids": list(self.game_ids),
            "min_sample": self.min_sample,
            "tables": [table.model_dump(mode="json") for table in self.tables],
            "quality": self.quality.to_payload(),
        }


@dataclass(frozen=True)
class _MetricSpec:
    metric: StatMetric
    title: str
    overall_row_label: str
    unit: StatUnit
    decimals: int
    dimensions: tuple[SplitDimension, ...]
    stable_delta: float


_SPECS: Final[tuple[_MetricSpec, ...]] = (
    _MetricSpec(
        metric=StatMetric.RUN_RATE,
        title="run rate",
        overall_row_label="All scrimmage plays",
        unit=StatUnit.RATE,
        decimals=_RATE_DECIMALS,
        dimensions=_SCRIMMAGE_DIMENSIONS,
        stable_delta=STABLE_RATE_DELTA,
    ),
    _MetricSpec(
        metric=StatMetric.PASS_RATE,
        title="pass rate",
        overall_row_label="All scrimmage plays",
        unit=StatUnit.RATE,
        decimals=_RATE_DECIMALS,
        dimensions=_SCRIMMAGE_DIMENSIONS,
        stable_delta=STABLE_RATE_DELTA,
    ),
    _MetricSpec(
        metric=StatMetric.YARDS_PER_PLAY,
        title="yards per play",
        overall_row_label="All scrimmage plays",
        unit=StatUnit.YARDS,
        decimals=_YARDS_DECIMALS,
        dimensions=_SCRIMMAGE_DIMENSIONS,
        stable_delta=STABLE_YARDS_DELTA,
    ),
    _MetricSpec(
        metric=StatMetric.THIRD_DOWN_CONVERSION_RATE,
        title="third-down conversion rate",
        overall_row_label="All third-down attempts",
        unit=StatUnit.RATE,
        decimals=_RATE_DECIMALS,
        dimensions=_THIRD_DOWN_DIMENSIONS,
        stable_delta=STABLE_RATE_DELTA,
    ),
)

_DIMENSION_TITLES: Final[dict[SplitDimension, str]] = {
    SplitDimension.DOWN: "down",
    SplitDimension.DISTANCE_BUCKET: "distance",
    SplitDimension.FIELD_POSITION: "field position",
    SplitDimension.WEEK: "week",
}


def down_label(down: int | None) -> str:
    """Return the row label for a down, or the explicit unknown label."""

    if down is None:
        return UNKNOWN_DOWN_LABEL
    return DOWN_LABELS[down - 1]


def distance_bucket(yards_to_go: int | None) -> str:
    """Bucket a yards-to-go value into a fixed, ordered distance label."""

    if yards_to_go is None:
        return UNKNOWN_DISTANCE_LABEL
    if yards_to_go <= 3:
        return DISTANCE_LABELS[0]
    if yards_to_go <= 6:
        return DISTANCE_LABELS[1]
    return DISTANCE_LABELS[2]


def field_position_bucket(yardline_100: int | None) -> str:
    """Bucket yards-from-goal into red zone, opponent, or own territory."""

    if yardline_100 is None:
        return UNKNOWN_FIELD_POSITION_LABEL
    if yardline_100 <= 20:
        return FIELD_POSITION_LABELS[0]
    if yardline_100 <= 50:
        return FIELD_POSITION_LABELS[1]
    return FIELD_POSITION_LABELS[2]


def _validate_min_sample(min_sample: int) -> None:
    if min_sample < 1:
        raise StatsError(f"min_sample must be at least 1, got {min_sample}")


def _validate_games(games: Sequence[Game], opponent: Team) -> tuple[Game, ...]:
    selected = tuple(games)
    if not selected:
        raise InvalidSelectionError("analyze at least one game")
    seen: set[str] = set()
    for game in selected:
        if game.game_id in seen:
            raise InvalidSelectionError(f"duplicate game {game.game_id!r} in the selection")
        seen.add(game.game_id)
        if not game.involves(opponent):
            raise InvalidSelectionError(f"{opponent.value} does not play in game {game.game_id}")
    return selected


def _reject_duplicate_plays(plays: Sequence[Play]) -> None:
    seen: set[str] = set()
    for play in plays:
        key = play.key
        if key in seen:
            raise DuplicatePlayKeyError(
                f"duplicate play {key} in the analyzed input; "
                "the season store guarantees unique (game_id, play_id)"
            )
        seen.add(key)


def _offensive_plays(
    plays: Sequence[Play],
    game_ids: frozenset[str],
    opponent: Team,
) -> tuple[Play, ...]:
    return tuple(play for play in plays if play.game_id in game_ids and play.posteam is opponent)


def _populations(offense: Sequence[Play]) -> dict[StatMetric, tuple[Play, ...]]:
    scrimmage = tuple(play for play in offense if play.is_scrimmage_play)
    with_yards = tuple(play for play in scrimmage if play.yards_gained is not None)
    attempts = tuple(
        play
        for play in scrimmage
        if play.down == 3
        and (play.third_down_converted is not None or play.third_down_failed is not None)
    )
    return {
        StatMetric.RUN_RATE: scrimmage,
        StatMetric.PASS_RATE: scrimmage,
        StatMetric.YARDS_PER_PLAY: with_yards,
        StatMetric.THIRD_DOWN_CONVERSION_RATE: attempts,
    }


def _data_quality(
    offense: Sequence[Play],
    populations: dict[StatMetric, tuple[Play, ...]],
) -> DataQuality:
    scrimmage = populations[StatMetric.RUN_RATE]
    return DataQuality(
        offensive_plays=len(offense),
        scrimmage_plays=len(scrimmage),
        third_down_attempts=len(populations[StatMetric.THIRD_DOWN_CONVERSION_RATE]),
        unknown_play_types=sum(1 for play in offense if play.play_type is PlayType.UNKNOWN),
        missing_down=sum(1 for play in scrimmage if play.down is None),
        missing_yards_to_go=sum(1 for play in scrimmage if play.yards_to_go is None),
        missing_yardline_100=sum(1 for play in scrimmage if play.yardline_100 is None),
        missing_yards_gained=sum(1 for play in scrimmage if play.yards_gained is None),
        missing_third_down_outcomes=sum(
            1
            for play in scrimmage
            if play.down == 3
            and play.third_down_converted is None
            and play.third_down_failed is None
        ),
    )


def _numerator(spec: _MetricSpec, plays: Sequence[Play]) -> int:
    if spec.metric is StatMetric.RUN_RATE:
        return sum(1 for play in plays if play.is_run)
    if spec.metric is StatMetric.PASS_RATE:
        return sum(1 for play in plays if play.is_pass)
    if spec.metric is StatMetric.THIRD_DOWN_CONVERSION_RATE:
        return sum(1 for play in plays if play.third_down_converted is True)
    return sum(play.yards_gained for play in plays if play.yards_gained is not None)


def _metric_value(spec: _MetricSpec, plays: Sequence[Play]) -> float | None:
    if not plays:
        return None
    return round(_numerator(spec, plays) / len(plays), spec.decimals)


def _stat_row(spec: _MetricSpec, label: str, plays: Sequence[Play], min_sample: int) -> StatRow:
    n = len(plays)
    numerator: int | None = None
    value: float | None = None
    if plays:
        numerator = _numerator(spec, plays)
        value = round(numerator / n, spec.decimals)
    return StatRow(
        label=label,
        value=value,
        n=n,
        numerator=numerator,
        denominator=n if n else None,
        unit=spec.unit,
        suppressed=n < min_sample,
    )


def _rows(
    spec: _MetricSpec,
    dimension: SplitDimension,
    population: Sequence[Play],
    min_sample: int,
) -> list[StatRow]:
    if dimension is SplitDimension.OVERALL:
        return [_stat_row(spec, spec.overall_row_label, population, min_sample)]

    groups: dict[str, list[Play]] = {}
    if dimension is SplitDimension.DOWN:
        ordered: tuple[str, ...] = (*DOWN_LABELS, UNKNOWN_DOWN_LABEL)
        for play in population:
            groups.setdefault(down_label(play.down), []).append(play)
    elif dimension is SplitDimension.DISTANCE_BUCKET:
        ordered = (*DISTANCE_LABELS, UNKNOWN_DISTANCE_LABEL)
        for play in population:
            groups.setdefault(distance_bucket(play.yards_to_go), []).append(play)
    elif dimension is SplitDimension.FIELD_POSITION:
        ordered = (*FIELD_POSITION_LABELS, UNKNOWN_FIELD_POSITION_LABEL)
        for play in population:
            groups.setdefault(field_position_bucket(play.yardline_100), []).append(play)
    else:
        for week in sorted({play.week for play in population}):
            groups[f"Week {week}"] = [play for play in population if play.week == week]
        ordered = tuple(groups)

    rows: list[StatRow] = []
    for label in ordered:
        bucket = groups.get(label, [])
        if not bucket and label in _UNKNOWN_LABELS:
            continue
        rows.append(_stat_row(spec, label, bucket, min_sample))
    return rows


def _unknown_note(missing: int, human: str, unknown_label: str) -> str | None:
    if not missing:
        return None
    return f"{missing} play(s) have no recorded {human}; they are shown as '{unknown_label}'."


def _table_notes(
    spec: _MetricSpec,
    dimension: SplitDimension,
    population: Sequence[Play],
    quality: DataQuality,
) -> list[str]:
    notes: list[str] = []
    if dimension is SplitDimension.DOWN:
        note = _unknown_note(
            sum(1 for play in population if play.down is None), "down", UNKNOWN_DOWN_LABEL
        )
    elif dimension is SplitDimension.DISTANCE_BUCKET:
        note = _unknown_note(
            sum(1 for play in population if play.yards_to_go is None),
            "yards to go",
            UNKNOWN_DISTANCE_LABEL,
        )
    elif dimension is SplitDimension.FIELD_POSITION:
        note = _unknown_note(
            sum(1 for play in population if play.yardline_100 is None),
            "field position",
            UNKNOWN_FIELD_POSITION_LABEL,
        )
    else:
        note = None
    if note:
        notes.append(note)
    if spec.metric is StatMetric.YARDS_PER_PLAY and quality.missing_yards_gained:
        notes.append(
            f"{quality.missing_yards_gained} scrimmage play(s) excluded: yards_gained is missing."
        )
    if spec.metric is StatMetric.THIRD_DOWN_CONVERSION_RATE and quality.missing_third_down_outcomes:
        notes.append(
            f"{quality.missing_third_down_outcomes} third down(s) excluded: "
            "the conversion outcome is missing."
        )
    return notes


def _format_weeks(weeks: Sequence[int]) -> str:
    if not weeks:
        return "no plays"
    if len(weeks) == 1:
        return f"week {weeks[0]}"
    if list(weeks) == list(range(weeks[0], weeks[-1] + 1)):
        return f"weeks {weeks[0]}-{weeks[-1]}"
    return "weeks " + ", ".join(str(week) for week in weeks)


def _table_title(
    spec: _MetricSpec,
    dimension: SplitDimension,
    opponent: Team,
    weeks: Sequence[int],
) -> str:
    scope = _format_weeks(weeks)
    if dimension is SplitDimension.OVERALL:
        return f"{opponent.value} {spec.title} ({scope})"
    return f"{opponent.value} {spec.title} by {_DIMENSION_TITLES[dimension]} ({scope})"


def _build_table(
    spec: _MetricSpec,
    dimension: SplitDimension,
    population: tuple[Play, ...],
    *,
    opponent: Team,
    min_sample: int,
    quality: DataQuality,
) -> StatTable:
    weeks = sorted({play.week for play in population})
    game_ids = sorted({play.game_id for play in population})
    n_plays = len(population)
    return StatTable(
        metric=spec.metric,
        title=_table_title(spec, dimension, opponent, weeks),
        dimension=dimension,
        rows=_rows(spec, dimension, population, min_sample),
        n_plays=n_plays,
        game_ids=game_ids,
        weeks=weeks,
        min_sample=min_sample,
        suppressed=n_plays < min_sample,
        notes=_table_notes(spec, dimension, population, quality),
    )


def analyze_opponent(
    plays: Iterable[Play],
    games: Sequence[Game],
    *,
    opponent: Team,
    min_sample: int = DEFAULT_MIN_SAMPLE,
) -> StatsBundle:
    """Compute every tendency table for one opponent's offense.

    ``games`` is the selection (each must involve ``opponent``); ``plays`` may
    be the whole season, since only plays from the selected games with
    ``posteam`` equal to the opponent are analyzed. Duplicate play keys in
    ``plays`` are rejected even outside the selection, because the caller is
    trusting that data.
    """

    _validate_min_sample(min_sample)
    selected_games = _validate_games(games, opponent)
    input_plays = tuple(plays)
    _reject_duplicate_plays(input_plays)

    game_ids = frozenset(game.game_id for game in selected_games)
    offense = _offensive_plays(input_plays, game_ids, opponent)
    populations = _populations(offense)
    quality = _data_quality(offense, populations)

    tables = [
        _build_table(
            spec,
            dimension,
            populations[spec.metric],
            opponent=opponent,
            min_sample=min_sample,
            quality=quality,
        )
        for spec in _SPECS
        for dimension in spec.dimensions
    ]
    tables.sort(key=lambda table: table.stat_table_id)

    return StatsBundle(
        opponent=opponent,
        weeks=tuple(sorted({game.week for game in selected_games})),
        game_ids=tuple(sorted(game_ids)),
        min_sample=min_sample,
        tables=tuple(tables),
        quality=quality,
    )


def _normalize_weeks(weeks: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(week) for week in weeks}))


def compare_week_windows(
    plays: Iterable[Play],
    games: Sequence[Game],
    *,
    opponent: Team,
    early_weeks: Iterable[int],
    late_weeks: Iterable[int],
    min_sample: int = DEFAULT_MIN_SAMPLE,
) -> tuple[WindowComparison, ...]:
    """Compare each metric across two disjoint week windows (requirement 15).

    Both windows must be non-empty, disjoint, and drawn from the weeks of the
    selected games. Each comparison reports the sample size and value of both
    windows, the delta between the published values, and a direction
    classified with fixed thresholds (``STABLE_RATE_DELTA`` for rates,
    ``STABLE_YARDS_DELTA`` for yards). A window below ``min_sample`` keeps its
    values but is marked unreliable with an ``unknown`` direction.
    """

    _validate_min_sample(min_sample)
    selected_games = _validate_games(games, opponent)
    input_plays = tuple(plays)
    _reject_duplicate_plays(input_plays)

    early = _normalize_weeks(early_weeks)
    late = _normalize_weeks(late_weeks)
    if not early or not late:
        raise InvalidSelectionError("both week windows must contain at least one week")
    overlap = sorted(set(early) & set(late))
    if overlap:
        raise InvalidSelectionError(f"week windows must not overlap; shared weeks: {overlap}")
    covered = {game.week for game in selected_games}
    outside = sorted((set(early) | set(late)) - covered)
    if outside:
        raise InvalidSelectionError(f"weeks {outside} are not part of the selected games")

    game_ids = frozenset(game.game_id for game in selected_games)
    offense = _offensive_plays(input_plays, game_ids, opponent)
    populations = _populations(offense)

    comparisons: list[WindowComparison] = []
    for spec in _SPECS:
        population = populations[spec.metric]
        early_plays = [play for play in population if play.week in set(early)]
        late_plays = [play for play in population if play.week in set(late)]
        early_value = _metric_value(spec, early_plays)
        late_value = _metric_value(spec, late_plays)
        reliable = len(early_plays) >= min_sample and len(late_plays) >= min_sample
        delta = (
            round(late_value - early_value, spec.decimals)
            if early_value is not None and late_value is not None
            else None
        )
        direction = TendencyDirection.UNKNOWN
        if reliable and delta is not None:
            if abs(delta) <= spec.stable_delta:
                direction = TendencyDirection.STABLE
            elif delta > 0:
                direction = TendencyDirection.INCREASE
            else:
                direction = TendencyDirection.DECREASE
        comparisons.append(
            WindowComparison(
                metric=spec.metric,
                early_weeks=early,
                late_weeks=late,
                early_n=len(early_plays),
                early_value=early_value,
                late_n=len(late_plays),
                late_value=late_value,
                delta=delta,
                direction=direction,
                reliable=reliable,
                min_sample=min_sample,
            )
        )
    return tuple(comparisons)
