"""Atomic JSON store behavior, exercised through cached fixtures."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from opponent_tendencies.data.nflverse import (
    IngestionError,
    NflverseClient,
    normalize_games,
    normalize_plays,
)
from opponent_tendencies.data.play_store import (
    SCHEMA_VERSION,
    SeasonStore,
    StoreCorruptionError,
    StoreIntegrityError,
)
from opponent_tendencies.models import Game, Play, SeasonType, Team
from tests.nflverse_fixtures import SEASON, pbp_rows, schedule_rows

pytestmark = pytest.mark.integration


def fixture_records() -> tuple[tuple[Game, ...], tuple[Play, ...]]:
    games = normalize_games(pd.DataFrame(schedule_rows()), season=SEASON).games
    plays = normalize_plays(pd.DataFrame(pbp_rows()), season=SEASON).plays
    return games, plays


def new_store(tmp_path: Path) -> SeasonStore:
    return SeasonStore(SEASON, tmp_path / f"{SEASON}.json")


def make_game(
    game_id: str = "2026_01_LV_LA",
    *,
    week: int = 1,
    season: int = SEASON,
    season_type: SeasonType = SeasonType.REGULAR,
    home_team: Team = Team.LA,
    away_team: Team = Team.LV,
    home_score: int | None = 20,
    away_score: int | None = 17,
) -> Game:
    return Game(
        game_id=game_id,
        season=season,
        week=week,
        season_type=season_type,
        game_date=date(2026, 9, 13),
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
    )


def make_play(
    game_id: str = "2026_01_LV_LA",
    *,
    play_id: int = 1,
    season: int = SEASON,
    week: int = 1,
    season_type: SeasonType = SeasonType.REGULAR,
    posteam: Team | None = Team.LV,
    defteam: Team | None = Team.LA,
) -> Play:
    return Play(
        game_id=game_id,
        play_id=play_id,
        season=season,
        week=week,
        season_type=season_type,
        posteam=posteam,
        defteam=defteam,
    )


def write_store(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_first_refresh_inserts_then_second_refresh_is_idempotent(tmp_path: Path) -> None:
    games, plays = fixture_records()
    store = new_store(tmp_path)

    first = store.refresh(games, plays)

    assert first.inserted_games == 5
    assert first.inserted_plays == len(plays)
    assert first.updated == 0
    assert first.skipped == 0
    assert first.changed is True

    written = store.path.read_bytes()
    second = store.refresh(games, plays)

    assert second.inserted == 0
    assert second.updated == 0
    assert second.skipped_games == 5
    assert second.skipped_plays == len(plays)
    assert second.changed is False
    assert store.path.read_bytes() == written

    state = store.load()
    assert len(state.games) == 5
    assert len(state.plays) == len(plays)


def test_changed_play_updates_in_place(tmp_path: Path) -> None:
    games, plays = fixture_records()
    store = new_store(tmp_path)
    store.refresh(games, plays)
    before = store.load()
    target = next(play for play in plays if play.yards_gained is not None)
    changed = target.model_copy(update={"yards_gained": 99})

    result = store.refresh((), (changed,))

    assert result.inserted_plays == 0
    assert result.updated_plays == 1
    assert result.skipped_plays == 0
    after = store.load()
    assert after.revision != before.revision
    stored = next(play for play in after.plays if play.key == target.key)
    assert stored.yards_gained == 99


def test_teams_are_alphabetical_by_display_name(tmp_path: Path) -> None:
    store = new_store(tmp_path)

    store.refresh((make_game(),), ())

    # "Las Vegas Raiders" precedes "Los Angeles Rams" even though LA < LV.
    assert store.load().teams == (Team.LV, Team.LA)


def test_played_weeks_exclude_future_and_non_regular_games(tmp_path: Path) -> None:
    games, plays = fixture_records()
    store = new_store(tmp_path)
    exhibition = make_game(
        "2026_PRE_NE_SEA",
        week=3,
        season_type=SeasonType.PRESEASON,
        home_team=Team.SEA,
        away_team=Team.NE,
    )

    store.refresh((*games, exhibition), plays)

    state = store.load()
    assert state.played_weeks == (1, 2)
    assert Team.NE in state.teams
    assert len(state.games_for_weeks([2])) == 2
    assert {game.game_id for game in state.games_for_weeks([1, 2])} == {
        "2026_01_NE_SEA",
        "2026_01_SF_LA",
        "2026_02_MIA_SF",
        "2026_02_SEA_ARI",
    }


def test_empty_refresh_creates_no_file(tmp_path: Path) -> None:
    store = new_store(tmp_path)

    result = store.refresh((), ())

    assert result.inserted == result.updated == result.skipped == 0
    assert result.changed is False
    assert store.exists() is False
    state = store.load()
    assert state.games == ()
    assert state.plays == ()
    assert state.teams == ()
    assert state.played_weeks == ()


def test_play_without_its_game_is_rejected(tmp_path: Path) -> None:
    store = new_store(tmp_path)

    with pytest.raises(StoreIntegrityError, match="unknown game"):
        store.refresh((), (make_play(),))

    assert store.exists() is False


def test_play_week_must_match_its_game(tmp_path: Path) -> None:
    store = new_store(tmp_path)

    with pytest.raises(StoreIntegrityError, match="week 1"):
        store.refresh((make_game(),), (make_play(week=2),))


def test_play_team_must_play_in_its_game(tmp_path: Path) -> None:
    store = new_store(tmp_path)

    with pytest.raises(StoreIntegrityError, match="NE"):
        store.refresh((make_game(),), (make_play(posteam=Team.NE, defteam=Team.LA),))


def test_wrong_season_is_rejected(tmp_path: Path) -> None:
    store = new_store(tmp_path)

    with pytest.raises(StoreIntegrityError, match="season 2025"):
        store.refresh((make_game(season=2025),), ())


def test_corrupt_json_reports_the_path(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{not json", encoding="utf-8")

    with pytest.raises(StoreCorruptionError, match=str(store.path)):
        store.load()


def test_unknown_schema_version_reports_recovery_step(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    write_store(store.path, {"schema_version": 99, "season": SEASON, "games": [], "plays": []})

    with pytest.raises(StoreCorruptionError, match="Delete the file and refresh"):
        store.load()


def test_store_season_must_match_the_file(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    write_store(
        store.path, {"schema_version": SCHEMA_VERSION, "season": 2025, "games": [], "plays": []}
    )

    with pytest.raises(StoreCorruptionError, match="season 2025"):
        store.load()


def test_invalid_stored_record_reports_its_index(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    write_store(
        store.path,
        {
            "schema_version": SCHEMA_VERSION,
            "season": SEASON,
            "games": [{"game_id": "x"}],
            "plays": [],
        },
    )

    with pytest.raises(StoreCorruptionError, match=r"games\[0\]"):
        store.load()


def test_duplicate_stored_plays_are_corruption(tmp_path: Path) -> None:
    play = make_play()
    store = new_store(tmp_path)
    write_store(
        store.path,
        {
            "schema_version": SCHEMA_VERSION,
            "season": SEASON,
            "games": [make_game().model_dump(mode="json")],
            "plays": [play.model_dump(mode="json"), play.model_dump(mode="json")],
        },
    )

    with pytest.raises(StoreCorruptionError, match="duplicate play"):
        store.load()


def test_store_file_is_canonical(tmp_path: Path) -> None:
    games, plays = fixture_records()
    store = new_store(tmp_path)
    store.refresh(games, plays)

    document = json.loads(store.path.read_text(encoding="utf-8"))

    assert list(document) == sorted(document)
    assert document["schema_version"] == SCHEMA_VERSION
    assert document["season"] == SEASON
    keys = [(play["game_id"], play["play_id"]) for play in document["plays"]]
    assert keys == sorted(keys)
    assert keys == [
        (play.game_id, play.play_id) for play in sorted(plays, key=lambda p: (p.game_id, p.play_id))
    ]


def test_client_refreshes_store_end_to_end(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    client = NflverseClient(
        SEASON,
        schedule_loader=lambda season: pd.DataFrame(schedule_rows()),
        play_loader=lambda season: pd.DataFrame(pbp_rows()),
    )

    report = client.refresh(store)

    assert report.refresh.inserted_games == 5
    assert report.refresh.inserted_plays == len(pbp_rows())
    assert report.games.skipped_non_regular == 0
    assert report.plays.skipped_deleted == 0
    assert "5 games" in report.summary()
    assert len(store.load().plays) == len(pbp_rows())


def test_client_rejects_a_store_for_another_season(tmp_path: Path) -> None:
    client = NflverseClient(
        SEASON,
        schedule_loader=lambda season: pd.DataFrame(schedule_rows()),
        play_loader=lambda season: pd.DataFrame(pbp_rows()),
    )
    other = SeasonStore(2025, tmp_path / "2025.json")

    with pytest.raises(IngestionError, match="does not match client season"):
        client.refresh(other)


def test_default_store_path_lives_under_data_dir() -> None:
    from opponent_tendencies.data.play_store import default_store_path

    assert default_store_path(2026, Path("/tmp/opponent-tendencies")) == Path(
        "/tmp/opponent-tendencies/seasons/2026.json"
    )
