"""Normalization of cached nflverse frames into the frozen contracts."""

from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl
import pytest

from opponent_tendencies.data.nflverse import (
    DuplicatePlayError,
    IngestionError,
    MissingColumnsError,
    NflverseClient,
    RowNormalizationError,
    normalize_games,
    normalize_plays,
)
from opponent_tendencies.models import Play, PlayType, SeasonType, Team
from tests.nflverse_fixtures import SEASON, pbp_rows, schedule_rows

FIXTURE_GAMES = {
    "2026_01_NE_SEA",
    "2026_01_SF_LA",
    "2026_02_SEA_ARI",
    "2026_02_MIA_SF",
}


def frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def one_row(**changes: object) -> list[dict[str, Any]]:
    row = pbp_rows()[0]
    row.update(changes)
    return [row]


def test_normalizes_pbp_fixture_into_sorted_plays() -> None:
    batch = normalize_plays(frame(pbp_rows()), season=SEASON)

    assert len(batch.plays) == len(pbp_rows())
    assert all(isinstance(play, Play) for play in batch.plays)
    keys = [(play.game_id, play.play_id) for play in batch.plays]
    assert keys == sorted(keys)
    assert {play.game_id for play in batch.plays} == FIXTURE_GAMES
    assert {play.week for play in batch.plays} == {1, 2}
    assert {play.season for play in batch.plays} == {SEASON}
    assert batch.skipped_deleted == 0
    assert batch.skipped_non_regular == 0
    assert batch.unknown_play_types == 0


def test_third_down_flags_only_survive_on_third_down() -> None:
    plays = normalize_plays(frame(pbp_rows()), season=SEASON).plays

    third_downs = [play for play in plays if play.down == 3]
    assert third_downs
    assert any(play.third_down_converted for play in third_downs)
    assert any(play.third_down_failed for play in third_downs)

    for play in plays:
        if play.down != 3:
            assert play.third_down_converted is None
            assert play.third_down_failed is None


def test_missing_values_are_preserved_not_imputed() -> None:
    plays = normalize_plays(frame(pbp_rows()), season=SEASON).plays
    opening_play = next(
        play for play in plays if play.game_id == "2026_01_NE_SEA" and play.play_id == 1
    )

    assert opening_play.posteam is None
    assert opening_play.down is None
    assert opening_play.yardline_100 is None
    assert opening_play.play_type is PlayType.UNKNOWN

    no_gain_recorded = next(
        play for play in plays if play.down is not None and play.yards_gained is None
    )
    assert no_gain_recorded.yards_gained is None


def test_deleted_play_is_dropped_and_counted() -> None:
    rows = pbp_rows()
    rows[0]["play_deleted"] = 1

    batch = normalize_plays(frame(rows), season=SEASON)

    assert len(batch.plays) == len(rows) - 1
    assert batch.skipped_deleted == 1
    assert batch.skipped_non_regular == 0


def test_non_regular_play_is_dropped_and_counted() -> None:
    batch = normalize_plays(frame(one_row(season_type="POST")), season=SEASON)

    assert batch.plays == ()
    assert batch.skipped_non_regular == 1
    assert batch.skipped_deleted == 0


def test_unknown_play_type_is_kept_as_unknown_and_counted() -> None:
    batch = normalize_plays(frame(one_row(play_type="scramble_drill")), season=SEASON)

    assert len(batch.plays) == 1
    assert batch.plays[0].play_type is PlayType.UNKNOWN
    assert batch.unknown_play_types == 1


def test_missing_required_column_is_reported_by_name() -> None:
    rows = pbp_rows()
    for row in rows:
        del row["yardline_100"]

    with pytest.raises(MissingColumnsError, match="yardline_100"):
        normalize_plays(frame(rows), season=SEASON)


def test_empty_frame_is_an_empty_batch() -> None:
    batch = normalize_plays(pd.DataFrame(columns=list(pbp_rows()[0])), season=SEASON)

    assert batch.plays == ()
    assert batch.skipped_deleted == 0
    assert batch.skipped_non_regular == 0


def test_invalid_play_id_reports_the_source_key() -> None:
    with pytest.raises(RowNormalizationError, match="play_id"):
        normalize_plays(frame(one_row(play_id=1.5)), season=SEASON)


def test_duplicate_play_keys_are_rejected() -> None:
    rows = pbp_rows()
    rows.append(dict(rows[0]))

    with pytest.raises(DuplicatePlayError, match="2026_01_NE_SEA:1"):
        normalize_plays(frame(rows), season=SEASON)


def test_wrong_season_is_rejected() -> None:
    with pytest.raises(IngestionError, match="expected season 2025"):
        normalize_plays(frame(pbp_rows()), season=2025)


def test_team_abbreviations_are_canonicalized() -> None:
    batch = normalize_plays(frame(one_row(posteam="WAS", defteam="PHI")), season=SEASON)

    play = batch.plays[0]
    assert play.posteam is Team.WSH
    assert play.defteam is Team.PHI


def test_unknown_team_abbreviation_is_rejected() -> None:
    with pytest.raises(RowNormalizationError, match="XYZ"):
        normalize_plays(frame(one_row(posteam="XYZ")), season=SEASON)


def test_polars_frame_matches_pandas_frame() -> None:
    rows = pbp_rows()

    from_pandas = normalize_plays(frame(rows), season=SEASON).plays
    from_polars = normalize_plays(pl.DataFrame(rows), season=SEASON).plays

    assert from_polars == from_pandas


def test_schedule_fixture_normalizes_games_and_future_week() -> None:
    batch = normalize_games(frame(schedule_rows()), season=SEASON)

    assert len(batch.games) == 5
    assert batch.skipped_non_regular == 0
    assert {game.week for game in batch.games} == {1, 2, 18}
    assert all(game.season_type is SeasonType.REGULAR for game in batch.games)

    future = next(game for game in batch.games if game.week == 18)
    assert future.home_score is None
    assert future.away_score is None

    neutral = next(game for game in batch.games if game.game_id == "2026_01_SF_LA")
    assert neutral.neutral_site is True
    assert neutral.venue == "Melbourne Cricket Ground"


def test_non_regular_games_are_dropped_and_counted() -> None:
    rows = schedule_rows()
    exhibition = dict(rows[0], game_id="2026_PRE_NE_SEA", game_type="PRE")
    playoff = dict(rows[0], game_id="2026_POST_NE_SEA", game_type="POST", week=19)
    rows.extend([exhibition, playoff])

    batch = normalize_games(frame(rows), season=SEASON)

    assert len(batch.games) == 5
    assert batch.skipped_non_regular == 2
    assert all(game.game_id not in {"2026_PRE_NE_SEA", "2026_POST_NE_SEA"} for game in batch.games)


def test_one_sided_score_is_rejected() -> None:
    rows = [dict(schedule_rows()[0], home_score=13, away_score=None)]

    with pytest.raises(RowNormalizationError, match="both"):
        normalize_games(frame(rows), season=SEASON)


def test_bad_game_date_is_rejected() -> None:
    rows = [dict(schedule_rows()[0], gameday="not-a-date")]

    with pytest.raises(RowNormalizationError, match="gameday"):
        normalize_games(frame(rows), season=SEASON)


def test_empty_schedule_frame_is_an_empty_batch() -> None:
    batch = normalize_games(pd.DataFrame(columns=list(schedule_rows()[0])), season=SEASON)

    assert batch.games == ()
    assert batch.skipped_non_regular == 0


def test_client_uses_injected_loaders() -> None:
    client = NflverseClient(
        SEASON,
        schedule_loader=lambda season: frame(schedule_rows()),
        play_loader=lambda season: frame(pbp_rows()),
    )

    assert len(client.fetch_games().games) == 5
    assert len(client.fetch_plays().plays) == len(pbp_rows())
    with pytest.raises(ValueError, match="implausible season"):
        NflverseClient(1800)
