"""Deterministic statistics engine: metrics, splits, suppression, gaps, windows."""

from __future__ import annotations

import os
import random
from datetime import date, timedelta
from pathlib import Path

import pytest

from opponent_tendencies.data.jsonio import canonical_dump
from opponent_tendencies.data.nflverse import normalize_games, normalize_plays
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
from opponent_tendencies.stats.engine import (
    DISTANCE_LABELS,
    DOWN_LABELS,
    FIELD_POSITION_LABELS,
    DataQuality,
    DuplicatePlayKeyError,
    InvalidSelectionError,
    StatsBundle,
    StatsError,
    WindowComparison,
    analyze_opponent,
    compare_week_windows,
    distance_bucket,
    down_label,
    field_position_bucket,
)
from tests.nflverse_fixtures import pbp_rows, schedule_rows

OPPONENT = Team.BUF
DEFENSE = Team.KC
SEASON = 2026

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "stats_golden_2026_SEA.json"


def _game(
    week: int,
    *,
    game_id: str,
    home: Team = DEFENSE,
    away: Team = OPPONENT,
) -> Game:
    return Game(
        game_id=game_id,
        season=SEASON,
        week=week,
        game_date=date(2026, 9, 6) + timedelta(days=7 * (week - 1)),
        home_team=home,
        away_team=away,
        home_score=21,
        away_score=17,
    )


def _play(
    game_id: str,
    play_id: int,
    *,
    week: int,
    posteam: Team = OPPONENT,
    down: int | None = 1,
    yards_to_go: int | None = 10,
    yardline_100: int | None = 75,
    play_type: PlayType = PlayType.RUN,
    yards_gained: int | None = 4,
    third_down_converted: bool | None = None,
    third_down_failed: bool | None = None,
) -> Play:
    return Play(
        game_id=game_id,
        play_id=play_id,
        season=SEASON,
        week=week,
        posteam=posteam,
        defteam=DEFENSE if posteam is OPPONENT else OPPONENT,
        down=down,
        yards_to_go=yards_to_go,
        yardline_100=yardline_100,
        play_type=play_type,
        yards_gained=yards_gained,
        third_down_converted=third_down_converted,
        third_down_failed=third_down_failed,
    )


def _scrimmage_dataset() -> tuple[list[Game], list[Play]]:
    """Two games and 17 offensive plays: 16 scrimmage plus one punt.

    Week 1 has five runs and three passes (26 yards); week 2 has two runs and
    six passes (50 yards). Five third-down attempts, three converted.
    """

    game_1 = "2026_01_BUF_KC"
    game_2 = "2026_02_BUF_KC"
    plays = [
        _play(
            game_1,
            1,
            week=1,
            play_type=PlayType.RUN,
            down=1,
            yards_to_go=10,
            yardline_100=75,
            yards_gained=4,
        ),
        _play(
            game_1,
            2,
            week=1,
            play_type=PlayType.RUN,
            down=2,
            yards_to_go=6,
            yardline_100=71,
            yards_gained=3,
        ),
        _play(
            game_1,
            3,
            week=1,
            play_type=PlayType.RUN,
            down=3,
            yards_to_go=3,
            yardline_100=62,
            yards_gained=5,
            third_down_converted=True,
            third_down_failed=False,
        ),
        _play(
            game_1,
            4,
            week=1,
            play_type=PlayType.RUN,
            down=1,
            yards_to_go=10,
            yardline_100=67,
            yards_gained=2,
        ),
        _play(
            game_1,
            5,
            week=1,
            play_type=PlayType.PASS,
            down=2,
            yards_to_go=4,
            yardline_100=73,
            yards_gained=9,
        ),
        _play(
            game_1,
            6,
            week=1,
            play_type=PlayType.PASS,
            down=3,
            yards_to_go=7,
            yardline_100=70,
            yards_gained=2,
            third_down_converted=False,
            third_down_failed=True,
        ),
        _play(
            game_1,
            7,
            week=1,
            play_type=PlayType.RUN,
            down=1,
            yards_to_go=10,
            yardline_100=72,
            yards_gained=1,
        ),
        _play(
            game_1,
            8,
            week=1,
            play_type=PlayType.PASS,
            down=4,
            yards_to_go=8,
            yardline_100=74,
            yards_gained=0,
        ),
        _play(
            game_1,
            9,
            week=1,
            play_type=PlayType.PUNT,
            down=4,
            yards_to_go=8,
            yardline_100=74,
            yards_gained=0,
        ),
        _play(
            game_2,
            10,
            week=2,
            play_type=PlayType.RUN,
            down=1,
            yards_to_go=10,
            yardline_100=18,
            yards_gained=3,
        ),
        _play(
            game_2,
            11,
            week=2,
            play_type=PlayType.RUN,
            down=2,
            yards_to_go=2,
            yardline_100=45,
            yards_gained=4,
        ),
        _play(
            game_2,
            12,
            week=2,
            play_type=PlayType.PASS,
            down=3,
            yards_to_go=5,
            yardline_100=40,
            yards_gained=12,
            third_down_converted=True,
            third_down_failed=False,
        ),
        _play(
            game_2,
            13,
            week=2,
            play_type=PlayType.PASS,
            down=1,
            yards_to_go=10,
            yardline_100=55,
            yards_gained=8,
        ),
        _play(
            game_2,
            14,
            week=2,
            play_type=PlayType.PASS,
            down=3,
            yards_to_go=2,
            yardline_100=30,
            yards_gained=6,
            third_down_converted=True,
            third_down_failed=False,
        ),
        _play(
            game_2,
            15,
            week=2,
            play_type=PlayType.PASS,
            down=2,
            yards_to_go=8,
            yardline_100=60,
            yards_gained=15,
        ),
        _play(
            game_2,
            16,
            week=2,
            play_type=PlayType.PASS,
            down=3,
            yards_to_go=4,
            yardline_100=55,
            yards_gained=2,
            third_down_converted=False,
            third_down_failed=True,
        ),
        _play(
            game_2,
            17,
            week=2,
            play_type=PlayType.PASS,
            down=1,
            yards_to_go=10,
            yardline_100=65,
            yards_gained=0,
        ),
    ]
    games = [_game(1, game_id=game_1), _game(2, game_id=game_2)]
    return games, plays


def _gaps_dataset() -> tuple[list[Game], list[Play]]:
    """One game with every kind of missing value the engine must explain."""

    game_id = "2026_03_BUF_KC"
    plays = [
        _play(game_id, 1, week=3, down=None, yards_to_go=None, yardline_100=None, yards_gained=5),
        _play(game_id, 2, week=3, down=3, yards_to_go=3, yardline_100=60, yards_gained=2),
        _play(
            game_id,
            3,
            week=3,
            down=1,
            yards_to_go=10,
            yardline_100=70,
            play_type=PlayType.PASS,
            yards_gained=None,
        ),
        _play(game_id, 4, week=3, down=2, yards_to_go=5, yardline_100=50, yards_gained=3),
    ]
    return [_game(3, game_id=game_id)], plays


def _rate_dataset(
    early_runs: int,
    early_passes: int,
    late_runs: int,
    late_passes: int,
    *,
    early_yards: int = 5,
    late_yards: int = 5,
) -> tuple[list[Game], list[Play]]:
    games = [_game(1, game_id="2026_01_BUF_KC"), _game(2, game_id="2026_02_BUF_KC")]
    plays: list[Play] = []
    play_id = 1
    for week, runs, passes, yards in (
        (1, early_runs, early_passes, early_yards),
        (2, late_runs, late_passes, late_yards),
    ):
        for play_type, count in ((PlayType.RUN, runs), (PlayType.PASS, passes)):
            for _ in range(count):
                plays.append(
                    _play(
                        games[week - 1].game_id,
                        play_id,
                        week=week,
                        play_type=play_type,
                        yards_gained=yards,
                    )
                )
                play_id += 1
    return games, plays


def _table(bundle: StatsBundle, metric: StatMetric, dimension: SplitDimension) -> StatTable:
    matching = [
        table for table in bundle.tables if table.metric is metric and table.dimension is dimension
    ]
    assert len(matching) == 1
    return matching[0]


def _overall(bundle: StatsBundle, metric: StatMetric) -> StatTable:
    return _table(bundle, metric, SplitDimension.OVERALL)


def _row(table: StatTable, label: str) -> StatRow:
    matching = [row for row in table.rows if row.label == label]
    assert len(matching) == 1
    return matching[0]


def _comparisons(
    plays: list[Play],
    games: list[Game],
    *,
    min_sample: int = 5,
) -> tuple[WindowComparison, ...]:
    return compare_week_windows(
        plays,
        games,
        opponent=OPPONENT,
        early_weeks=[1],
        late_weeks=[2],
        min_sample=min_sample,
    )


def test_bucket_boundaries() -> None:
    assert down_label(1) == "1st down"
    assert down_label(4) == "4th down"
    assert down_label(None) == "Unknown down"

    for yards_to_go, expected in (
        (0, "Short (0-3)"),
        (3, "Short (0-3)"),
        (4, "Medium (4-6)"),
        (6, "Medium (4-6)"),
        (7, "Long (7+)"),
        (30, "Long (7+)"),
        (None, "Unknown distance"),
    ):
        assert distance_bucket(yards_to_go) == expected

    for yardline, expected in (
        (0, "Red zone (0-20)"),
        (20, "Red zone (0-20)"),
        (21, "Opponent territory (21-50)"),
        (50, "Opponent territory (21-50)"),
        (51, "Own territory (51-100)"),
        (100, "Own territory (51-100)"),
        (None, "Unknown field position"),
    ):
        assert field_position_bucket(yardline) == expected


def test_overall_metrics_and_row_counts() -> None:
    games, plays = _scrimmage_dataset()
    bundle = analyze_opponent(plays, games, opponent=OPPONENT, min_sample=5)

    assert bundle.weeks == (1, 2)
    assert bundle.game_ids == ("2026_01_BUF_KC", "2026_02_BUF_KC")
    assert bundle.quality == DataQuality(
        offensive_plays=17,
        scrimmage_plays=16,
        third_down_attempts=5,
        unknown_play_types=0,
        missing_down=0,
        missing_yards_to_go=0,
        missing_yardline_100=0,
        missing_yards_gained=0,
        missing_third_down_outcomes=0,
    )

    run = _overall(bundle, StatMetric.RUN_RATE).rows[0]
    assert (run.value, run.n, run.numerator, run.denominator) == (0.4375, 16, 7, 16)
    assert run.unit is StatUnit.RATE
    assert run.suppressed is False

    passes = _overall(bundle, StatMetric.PASS_RATE).rows[0]
    assert (passes.value, passes.numerator) == (0.5625, 9)

    yards = _overall(bundle, StatMetric.YARDS_PER_PLAY).rows[0]
    assert (yards.value, yards.numerator, yards.unit) == (4.75, 76, StatUnit.YARDS)

    third_down = _overall(bundle, StatMetric.THIRD_DOWN_CONVERSION_RATE).rows[0]
    assert (third_down.value, third_down.n, third_down.numerator) == (0.6, 5, 3)


def test_split_rows_are_complete_ordered_and_flagged() -> None:
    games, plays = _scrimmage_dataset()
    bundle = analyze_opponent(plays, games, opponent=OPPONENT, min_sample=5)

    down = _table(bundle, StatMetric.RUN_RATE, SplitDimension.DOWN)
    assert [row.label for row in down.rows] == list(DOWN_LABELS)
    assert [row.n for row in down.rows] == [6, 4, 5, 1]
    assert _row(down, "4th down").suppressed is True
    assert _row(down, "1st down").suppressed is False

    distance = _table(bundle, StatMetric.RUN_RATE, SplitDimension.DISTANCE_BUCKET)
    assert [row.label for row in distance.rows] == list(DISTANCE_LABELS)
    assert [row.n for row in distance.rows] == [3, 4, 9]

    field = _table(bundle, StatMetric.RUN_RATE, SplitDimension.FIELD_POSITION)
    assert [row.label for row in field.rows] == list(FIELD_POSITION_LABELS)
    assert [row.n for row in field.rows] == [1, 3, 12]
    assert _row(field, "Red zone (0-20)").suppressed is True

    week = _table(bundle, StatMetric.RUN_RATE, SplitDimension.WEEK)
    assert [row.label for row in week.rows] == ["Week 1", "Week 2"]
    assert [row.n for row in week.rows] == [8, 8]
    assert _row(week, "Week 1").value == 0.625
    assert _row(week, "Week 2").value == 0.25

    third_down_down = [
        table
        for table in bundle.tables
        if table.metric is StatMetric.THIRD_DOWN_CONVERSION_RATE
        and table.dimension is SplitDimension.DOWN
    ]
    assert third_down_down == []


def test_missing_values_are_explained_never_imputed() -> None:
    games, plays = _gaps_dataset()
    bundle = analyze_opponent(plays, games, opponent=OPPONENT, min_sample=5)

    assert bundle.quality == DataQuality(
        offensive_plays=4,
        scrimmage_plays=4,
        third_down_attempts=0,
        unknown_play_types=0,
        missing_down=1,
        missing_yards_to_go=1,
        missing_yardline_100=1,
        missing_yards_gained=1,
        missing_third_down_outcomes=1,
    )

    down = _table(bundle, StatMetric.RUN_RATE, SplitDimension.DOWN)
    assert [row.label for row in down.rows] == [*DOWN_LABELS, "Unknown down"]
    assert [row.n for row in down.rows] == [1, 1, 1, 0, 1]
    assert down.notes == ["1 play(s) have no recorded down; they are shown as 'Unknown down'."]

    distance = _table(bundle, StatMetric.RUN_RATE, SplitDimension.DISTANCE_BUCKET)
    assert [row.label for row in distance.rows] == [*DISTANCE_LABELS, "Unknown distance"]
    assert distance.notes == [
        "1 play(s) have no recorded yards to go; they are shown as 'Unknown distance'."
    ]

    field = _table(bundle, StatMetric.RUN_RATE, SplitDimension.FIELD_POSITION)
    assert [row.label for row in field.rows] == [*FIELD_POSITION_LABELS, "Unknown field position"]
    assert [row.n for row in field.rows] == [0, 1, 2, 1]

    yards = _overall(bundle, StatMetric.YARDS_PER_PLAY)
    assert (yards.rows[0].value, yards.rows[0].n, yards.rows[0].numerator) == (3.33, 3, 10)
    assert yards.notes == ["1 scrimmage play(s) excluded: yards_gained is missing."]

    third_down = _overall(bundle, StatMetric.THIRD_DOWN_CONVERSION_RATE)
    assert third_down.n_plays == 0
    assert third_down.suppressed is True
    assert third_down.rows[0].value is None
    assert third_down.notes == ["1 third down(s) excluded: the conversion outcome is missing."]

    run = _overall(bundle, StatMetric.RUN_RATE).rows[0]
    assert (run.value, run.n) == (0.75, 4)

    rendered = canonical_dump(bundle.to_payload())
    assert '"missing_down":1' in rendered


def test_short_samples_are_flagged_not_hidden() -> None:
    games, plays = _scrimmage_dataset()
    bundle = analyze_opponent(plays, games, opponent=OPPONENT, min_sample=100)

    overall = _overall(bundle, StatMetric.RUN_RATE)
    assert overall.suppressed is True
    row = overall.rows[0]
    assert (row.value, row.n, row.suppressed) == (0.4375, 16, True)


def test_no_offensive_plays_yields_suppressed_tables() -> None:
    games = [_game(1, game_id="2026_01_BUF_KC"), _game(2, game_id="2026_02_BUF_KC")]
    bundle = analyze_opponent([], games, opponent=OPPONENT)

    assert bundle.weeks == (1, 2)
    assert bundle.game_ids == ("2026_01_BUF_KC", "2026_02_BUF_KC")
    assert bundle.quality.offensive_plays == 0
    assert len(bundle.tables) == 19
    assert all(table.suppressed for table in bundle.tables)
    assert all(table.n_plays == 0 for table in bundle.tables)
    assert all(table.game_ids == [] and table.weeks == [] for table in bundle.tables)

    row = _overall(bundle, StatMetric.RUN_RATE).rows[0]
    assert (row.n, row.value, row.suppressed) == (0, None, True)
    assert _table(bundle, StatMetric.RUN_RATE, SplitDimension.WEEK).rows == []


def test_defensive_and_foreign_plays_are_excluded() -> None:
    games = [_game(1, game_id="2026_01_BUF_KC")]
    plays = [
        _play("2026_01_BUF_KC", 1, week=1),
        _play("2026_01_BUF_KC", 2, week=1, posteam=DEFENSE, yards_gained=-3),
        _play("2026_09_BUF_MIA", 3, week=9),
    ]
    bundle = analyze_opponent(plays, games, opponent=OPPONENT, min_sample=1)

    assert bundle.quality.offensive_plays == 1
    overall = _overall(bundle, StatMetric.YARDS_PER_PLAY)
    assert (overall.rows[0].value, overall.rows[0].n) == (4.0, 1)
    assert overall.game_ids == ["2026_01_BUF_KC"]


def test_punts_do_not_count_as_third_down_attempts() -> None:
    games = [_game(1, game_id="2026_01_BUF_KC")]
    plays = [
        _play(
            "2026_01_BUF_KC",
            1,
            week=1,
            play_type=PlayType.PUNT,
            down=3,
            yards_to_go=5,
            yards_gained=0,
        ),
        _play(
            "2026_01_BUF_KC",
            2,
            week=1,
            play_type=PlayType.RUN,
            down=3,
            yards_to_go=5,
            yards_gained=6,
            third_down_converted=True,
            third_down_failed=False,
        ),
    ]
    bundle = analyze_opponent(plays, games, opponent=OPPONENT, min_sample=1)

    assert bundle.quality.third_down_attempts == 1
    row = _overall(bundle, StatMetric.THIRD_DOWN_CONVERSION_RATE).rows[0]
    assert (row.value, row.n) == (1.0, 1)


def test_duplicate_plays_and_bad_selections_are_rejected() -> None:
    game = _game(1, game_id="2026_01_BUF_KC")
    duplicate = [_play("2026_01_BUF_KC", 1, week=1), _play("2026_01_BUF_KC", 1, week=1)]
    with pytest.raises(DuplicatePlayKeyError, match=r"2026_01_BUF_KC:1"):
        analyze_opponent(duplicate, [game], opponent=OPPONENT)

    with pytest.raises(InvalidSelectionError, match="at least one game"):
        analyze_opponent([], [], opponent=OPPONENT)

    other = _game(1, game_id="2026_01_KC_MIA", away=Team.MIA)
    with pytest.raises(InvalidSelectionError, match="does not play"):
        analyze_opponent([], [other], opponent=OPPONENT)

    with pytest.raises(InvalidSelectionError, match="duplicate game"):
        analyze_opponent([], [game, _game(1, game_id="2026_01_BUF_KC")], opponent=OPPONENT)

    with pytest.raises(StatsError, match="min_sample"):
        analyze_opponent([], [game], opponent=OPPONENT, min_sample=0)


def test_output_is_order_invariant_and_repeatable() -> None:
    games, plays = _scrimmage_dataset()
    baseline = analyze_opponent(plays, games, opponent=OPPONENT).to_payload()

    shuffled = list(plays)
    random.Random(2026).shuffle(shuffled)
    assert analyze_opponent(shuffled, games, opponent=OPPONENT).to_payload() == baseline
    assert analyze_opponent(plays, games, opponent=OPPONENT).to_payload() == baseline


def test_week_window_comparison_values_and_directions() -> None:
    games, plays = _scrimmage_dataset()
    run, passes, yards, third_down = _comparisons(plays, games)

    assert run.metric is StatMetric.RUN_RATE
    assert (run.early_weeks, run.late_weeks) == ((1,), (2,))
    assert (run.early_n, run.early_value) == (8, 0.625)
    assert (run.late_n, run.late_value) == (8, 0.25)
    assert run.delta == -0.375
    assert run.direction is TendencyDirection.DECREASE
    assert run.reliable is True

    assert (passes.delta, passes.direction) == (0.375, TendencyDirection.INCREASE)
    assert (yards.delta, yards.direction) == (3.0, TendencyDirection.INCREASE)

    assert (third_down.early_n, third_down.early_value) == (2, 0.5)
    assert (third_down.late_n, third_down.late_value) == (3, 0.6667)
    assert third_down.delta == 0.1667
    assert third_down.direction is TendencyDirection.UNKNOWN
    assert third_down.reliable is False


def test_week_window_stable_thresholds_are_inclusive() -> None:
    games, plays = _rate_dataset(10, 10, 11, 9)
    run, _, _, _ = _comparisons(plays, games)
    assert (run.delta, run.direction) == (0.05, TendencyDirection.STABLE)

    games, plays = _rate_dataset(10, 10, 12, 8)
    run, _, _, _ = _comparisons(plays, games)
    assert (run.delta, run.direction) == (0.1, TendencyDirection.INCREASE)

    games, plays = _rate_dataset(11, 9, 10, 10)
    run, _, _, _ = _comparisons(plays, games)
    assert (run.delta, run.direction) == (-0.05, TendencyDirection.STABLE)

    games, plays = _rate_dataset(10, 10, 10, 10, early_yards=5, late_yards=5)
    _, _, yards, _ = _comparisons(plays, games)
    assert (yards.delta, yards.direction) == (0.0, TendencyDirection.STABLE)


def _yards_dataset(early_yards: list[int], late_yards: list[int]) -> tuple[list[Game], list[Play]]:
    games = [_game(1, game_id="2026_01_BUF_KC"), _game(2, game_id="2026_02_BUF_KC")]
    plays: list[Play] = []
    for week, yards_list in ((1, early_yards), (2, late_yards)):
        for index, yards in enumerate(yards_list):
            plays.append(
                _play(
                    games[week - 1].game_id,
                    (week - 1) * 100 + index + 1,
                    week=week,
                    play_type=PlayType.RUN if index % 2 else PlayType.PASS,
                    yards_gained=yards,
                )
            )
    return games, plays


def test_week_window_yards_threshold() -> None:
    games, plays = _yards_dataset([5] * 10, [5, 6] * 5)
    _, _, yards, _ = _comparisons(plays, games)
    assert (yards.early_value, yards.late_value) == (5.0, 5.5)
    assert (yards.delta, yards.direction) == (0.5, TendencyDirection.STABLE)

    games, plays = _yards_dataset([5] * 10, [6] * 10)
    _, _, yards, _ = _comparisons(plays, games)
    assert (yards.delta, yards.direction) == (1.0, TendencyDirection.INCREASE)


def test_week_window_validation() -> None:
    games, plays = _scrimmage_dataset()
    with pytest.raises(InvalidSelectionError, match="overlap"):
        compare_week_windows(plays, games, opponent=OPPONENT, early_weeks=[1], late_weeks=[1])
    with pytest.raises(InvalidSelectionError, match="at least one week"):
        compare_week_windows(plays, games, opponent=OPPONENT, early_weeks=[], late_weeks=[2])
    with pytest.raises(InvalidSelectionError, match="not part of the selected games"):
        compare_week_windows(plays, games, opponent=OPPONENT, early_weeks=[1], late_weeks=[9])


def test_empty_comparison_reports_unknown_direction() -> None:
    games = [_game(1, game_id="2026_01_BUF_KC"), _game(2, game_id="2026_02_BUF_KC")]
    comparisons = compare_week_windows(
        [], games, opponent=OPPONENT, early_weeks=[1], late_weeks=[2]
    )

    assert len(comparisons) == 4
    for comparison in comparisons:
        assert comparison.early_value is None
        assert comparison.late_value is None
        assert comparison.delta is None
        assert comparison.direction is TendencyDirection.UNKNOWN
        assert comparison.reliable is False


def _sea_selection() -> tuple[list[Game], tuple[Play, ...]]:
    games = [
        game
        for game in normalize_games(schedule_rows()).games
        if game.involves(Team.SEA) and game.home_score is not None
    ]
    plays = normalize_plays(pbp_rows()).plays
    return games, plays


def _golden_document() -> dict[str, object]:
    games, plays = _sea_selection()
    bundle = analyze_opponent(plays, games, opponent=Team.SEA)
    comparisons = compare_week_windows(
        plays, games, opponent=Team.SEA, early_weeks=[1], late_weeks=[2]
    )
    return {
        "bundle": bundle.to_payload(),
        "comparisons": [comparison.to_payload() for comparison in comparisons],
    }


def test_fixture_bundle_follows_contract_rules() -> None:
    games, plays = _sea_selection()
    bundle = analyze_opponent(plays, games, opponent=Team.SEA)

    assert bundle.weeks == (1, 2)
    assert len(bundle.tables) == 19
    ids = [table.stat_table_id for table in bundle.tables]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    assert {table.metric for table in bundle.tables} == {
        StatMetric.RUN_RATE,
        StatMetric.PASS_RATE,
        StatMetric.YARDS_PER_PLAY,
        StatMetric.THIRD_DOWN_CONVERSION_RATE,
    }
    for table in bundle.tables:
        assert set(table.weeks) <= set(bundle.weeks)
        assert set(table.game_ids) <= set(bundle.game_ids)
        for row in table.rows:
            assert row.n <= table.n_plays
    assert bundle.quality.offensive_plays > 0
    assert bundle.quality.scrimmage_plays > 0
    assert bundle.quality.third_down_attempts > 0


def test_every_fixture_team_analyzes() -> None:
    played = [
        game for game in normalize_games(schedule_rows()).games if game.home_score is not None
    ]
    plays = normalize_plays(pbp_rows()).plays
    teams = sorted({team for game in played for team in game.teams})
    assert teams

    for team in teams:
        games = [game for game in played if game.involves(team)]
        bundle = analyze_opponent(plays, games, opponent=team)
        assert bundle.tables
        if len(bundle.weeks) >= 2:
            first, last = bundle.weeks[0], bundle.weeks[-1]
            comparisons = compare_week_windows(
                plays, games, opponent=team, early_weeks=[first], late_weeks=[last]
            )
            assert len(comparisons) == 4


def test_golden_bundle_is_byte_identical() -> None:
    """Pin the engine output for the cached 2026 SEA fixture.

    Regenerate deliberately with ``UPDATE_GOLDEN=1 pytest
    tests/unit/test_stats_engine.py`` after an intentional change to bucket
    labels, ordering, rounding, or notes.
    """

    rendered = canonical_dump(_golden_document())
    assert canonical_dump(_golden_document()) == rendered

    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN_PATH.write_text(rendered, encoding="utf-8")
        pytest.skip("golden file regenerated")

    assert GOLDEN_PATH.is_file(), "golden file missing; run with UPDATE_GOLDEN=1"
    assert rendered == GOLDEN_PATH.read_text(encoding="utf-8")
