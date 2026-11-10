"""Atomic JSON store for one nflverse season.

The store holds two normalized record sets — regular-season games and plays —
and is written as a single document so a refresh can never leave games without
their plays or vice versa. Upserts are keyed by ``game_id`` and
``(game_id, play_id)``; a repeat refresh of identical data reports every record
as skipped and does not rewrite the file, so bytes stay stable across runs.

Only deterministic values are persisted: no timestamps, no fetch counters, and
keys/records sorted before serialization. ``SeasonData.revision`` hashes that
canonical payload for later report caching.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ValidationError

from opponent_tendencies.config import resolve_data_dir
from opponent_tendencies.models import (
    TEAM_DISPLAY_NAMES,
    Game,
    Play,
    Team,
)

__all__ = [
    "SCHEMA_VERSION",
    "RefreshResult",
    "SeasonData",
    "SeasonStore",
    "StoreCorruptionError",
    "StoreError",
    "StoreIntegrityError",
    "default_store_path",
]

SCHEMA_VERSION: Final = 1

_STORE_DIR: Final = "seasons"


class StoreError(RuntimeError):
    """Base class for store failures."""


class StoreCorruptionError(StoreError):
    """The file on disk is not a store this version can read."""


class StoreIntegrityError(StoreError):
    """Records are internally inconsistent and must not be written."""


@dataclass(frozen=True)
class RefreshResult:
    """Inserted, updated, and skipped counts for one refresh."""

    inserted_games: int = 0
    updated_games: int = 0
    skipped_games: int = 0
    inserted_plays: int = 0
    updated_plays: int = 0
    skipped_plays: int = 0

    @property
    def inserted(self) -> int:
        return self.inserted_games + self.inserted_plays

    @property
    def updated(self) -> int:
        return self.updated_games + self.updated_plays

    @property
    def skipped(self) -> int:
        return self.skipped_games + self.skipped_plays

    @property
    def changed(self) -> bool:
        return bool(self.inserted or self.updated)

    def __str__(self) -> str:
        return f"{self.inserted} inserted, {self.updated} updated, {self.skipped} skipped"


@dataclass(frozen=True)
class SeasonData:
    """A read-only view of one season's stored games and plays."""

    season: int
    games: tuple[Game, ...] = ()
    plays: tuple[Play, ...] = ()

    @property
    def played_games(self) -> tuple[Game, ...]:
        """Regular-season games with final scores; future and non-regular games excluded."""

        return tuple(
            game
            for game in self.games
            if game.is_regular_season
            and game.home_score is not None
            and game.away_score is not None
        )

    @property
    def teams(self) -> tuple[Team, ...]:
        """Alphabetical teams that appear in played regular-season games (req 4)."""

        present = {team for game in self.played_games for team in game.teams}
        return tuple(sorted(present, key=TEAM_DISPLAY_NAMES.__getitem__))

    @property
    def played_weeks(self) -> tuple[int, ...]:
        """Sorted regular-season weeks with at least one completed game (req 4)."""

        return tuple(sorted({game.week for game in self.played_games}))

    def games_for_weeks(self, weeks: Iterable[int]) -> tuple[Game, ...]:
        """Return played games in ``weeks``, ordered by week then date."""

        wanted = {int(week) for week in weeks}
        selected = (game for game in self.played_games if game.week in wanted)
        return tuple(sorted(selected, key=lambda game: (game.week, game.game_date, game.game_id)))

    @property
    def revision(self) -> str:
        """Content hash of the canonical payload, used to key report caches."""

        payload = _dump(_payload(self.season, self.games, self.plays))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_store_path(season: int, data_dir: Path | None = None) -> Path:
    """Return ``<data_dir>/seasons/<season>.json``."""

    base = data_dir if data_dir is not None else resolve_data_dir()
    return base / _STORE_DIR / f"{int(season)}.json"


class SeasonStore:
    """Read and atomically upsert one season's games and plays."""

    def __init__(self, season: int, path: Path | None = None) -> None:
        if int(season) < 1920:
            raise ValueError(f"implausible season: {season}")
        self.season = int(season)
        self.path = path if path is not None else default_store_path(self.season)

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> SeasonData:
        """Read the store, returning an empty view when the file does not exist."""

        if not self.path.exists():
            return SeasonData(season=self.season)
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise StoreCorruptionError(f"cannot read store {self.path}: {exc}") from exc
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise StoreCorruptionError(
                f"store {self.path} is not valid JSON ({exc.msg} at line {exc.lineno})"
            ) from exc
        if not isinstance(raw, dict):
            raise StoreCorruptionError(f"store {self.path} must contain a JSON object")

        version = raw.get("schema_version")
        if version != SCHEMA_VERSION:
            raise StoreCorruptionError(
                f"store {self.path} has schema_version {version!r}; "
                f"expected {SCHEMA_VERSION}. Delete the file and refresh."
            )
        if raw.get("season") != self.season:
            raise StoreCorruptionError(
                f"store {self.path} holds season {raw.get('season')!r}; expected {self.season}"
            )

        games = _validate_records(Game, raw.get("games"), "games", self.path)
        plays = _validate_records(Play, raw.get("plays"), "plays", self.path)
        try:
            _check_play_links(games, plays)
        except StoreIntegrityError as exc:
            raise StoreCorruptionError(f"store {self.path}: {exc}") from exc

        return SeasonData(season=self.season, games=games, plays=plays)

    def refresh(self, games: Iterable[Game], plays: Iterable[Play]) -> RefreshResult:
        """Upsert games and plays in one atomic write.

        A record whose values are unchanged counts as skipped; a record with the
        same key but different values replaces the stored one and counts as
        updated. Nothing is written when no record changed.
        """

        incoming_games = _index_games(games, expected_season=self.season)
        incoming_plays = _index_plays(plays, expected_season=self.season)
        state = self.load()

        existing_games = {game.game_id: game for game in state.games}
        existing_plays = {play.key: play for play in state.plays}
        game_counts = _merge(existing_games, incoming_games)
        play_counts = _merge(existing_plays, incoming_plays)

        _check_play_links(tuple(existing_games.values()), tuple(existing_plays.values()))

        result = RefreshResult(
            inserted_games=game_counts.inserted,
            updated_games=game_counts.updated,
            skipped_games=game_counts.skipped,
            inserted_plays=play_counts.inserted,
            updated_plays=play_counts.updated,
            skipped_plays=play_counts.skipped,
        )
        if result.changed:
            self._write(tuple(existing_games.values()), tuple(existing_plays.values()))
        return result

    def _write(self, games: Sequence[Game], plays: Sequence[Play]) -> None:
        payload = _payload(self.season, games, plays)
        _atomic_write(self.path, _dump(payload))


@dataclass(frozen=True)
class _UpsertCounts:
    inserted: int = 0
    updated: int = 0
    skipped: int = 0


def _merge[Model: BaseModel](
    existing: dict[str, Model], incoming: Mapping[str, Model]
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


def _index_games(games: Iterable[Game], *, expected_season: int) -> dict[str, Game]:
    indexed: dict[str, Game] = {}
    for game in games:
        if game.season != expected_season:
            raise StoreIntegrityError(
                f"game {game.game_id} is season {game.season}; store holds {expected_season}"
            )
        if game.game_id in indexed:
            raise StoreIntegrityError(f"duplicate game {game.game_id!r} in one refresh")
        indexed[game.game_id] = game
    return indexed


def _index_plays(plays: Iterable[Play], *, expected_season: int) -> dict[str, Play]:
    indexed: dict[str, Play] = {}
    for play in plays:
        if play.season != expected_season:
            raise StoreIntegrityError(
                f"play {play.key} is season {play.season}; store holds {expected_season}"
            )
        if play.key in indexed:
            raise StoreIntegrityError(f"duplicate play {play.key} in one refresh")
        indexed[play.key] = play
    return indexed


def _check_play_links(games: Iterable[Game], plays: Iterable[Play]) -> None:
    by_id: dict[str, Game] = {}
    for game in games:
        if game.game_id in by_id:
            raise StoreIntegrityError(f"duplicate game {game.game_id!r}")
        by_id[game.game_id] = game

    seen: set[str] = set()
    for play in plays:
        if play.key in seen:
            raise StoreIntegrityError(f"duplicate play {play.key}")
        seen.add(play.key)
        linked_game = by_id.get(play.game_id)
        if linked_game is None:
            raise StoreIntegrityError(
                f"play {play.key} references unknown game {play.game_id!r}; "
                "refresh schedules and plays together"
            )
        if play.week != linked_game.week:
            raise StoreIntegrityError(
                f"play {play.key} is week {play.week} but game {linked_game.game_id} "
                f"is week {linked_game.week}"
            )
        if play.season_type is not linked_game.season_type:
            raise StoreIntegrityError(
                f"play {play.key} is {play.season_type.value} but game {linked_game.game_id} "
                f"is {linked_game.season_type.value}"
            )
        for team in (play.posteam, play.defteam):
            if team is not None and not linked_game.involves(team):
                raise StoreIntegrityError(
                    f"play {play.key} lists {team.value}, who does not play in "
                    f"{linked_game.game_id}"
                )


def _payload(
    season: int,
    games: Iterable[Game],
    plays: Iterable[Play],
) -> dict[str, object]:
    ordered_games = sorted(games, key=lambda game: (game.week, game.game_date, game.game_id))
    ordered_plays = sorted(plays, key=lambda play: (play.game_id, play.play_id))
    return {
        "schema_version": SCHEMA_VERSION,
        "season": season,
        "games": [game.model_dump(mode="json") for game in ordered_games],
        "plays": [play.model_dump(mode="json") for play in ordered_plays],
    }


def _dump(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, 0o644)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory sync so the rename survives a crash."""

    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - platform dependent
        pass
    finally:
        os.close(descriptor)


def _validate_records[Model: BaseModel](
    model: type[Model],
    raw: object,
    kind: str,
    path: Path,
) -> tuple[Model, ...]:
    if not isinstance(raw, list):
        raise StoreCorruptionError(f"store {path}: {kind!r} must be a list")
    records: list[Model] = []
    for index, item in enumerate(raw):
        try:
            records.append(model.model_validate(item))
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'record'}: {error['msg']}"
                for error in exc.errors()
            )
            raise StoreCorruptionError(
                f"store {path}: {kind}[{index}] is invalid: {detail}"
            ) from exc
    return tuple(records)
