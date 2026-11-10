"""nflverse play-by-play and schedule ingestion.

nflverse is the only structured source of play-by-play. This module keeps the
boundary thin: it reads the columns the deterministic statistics engine needs,
converts them into the frozen contracts in
:mod:`opponent_tendencies.models`, and discards the rest of the ~370-column
payload. Missing values stay :data:`None`; nothing is imputed.

``nflreadpy`` returns Polars frames, but the normalizers accept any record
container (Polars, pandas, or a list of mappings) so tests and cached fixtures
exercise exactly the same code path as a live fetch.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Final

from pydantic import ValidationError

from opponent_tendencies.models import Game, Play, PlayType, SeasonType, Team

if TYPE_CHECKING:
    from opponent_tendencies.data.play_store import RefreshResult, SeasonStore

__all__ = [
    "GAME_COLUMNS",
    "PLAY_COLUMNS",
    "DuplicatePlayError",
    "IngestionError",
    "IngestionReport",
    "MissingColumnsError",
    "NflverseClient",
    "NormalizedGames",
    "NormalizedPlays",
    "RowNormalizationError",
    "current_season",
    "normalize_games",
    "normalize_plays",
]

#: Columns the statistics engine reads from ``load_pbp``. Anything else is
#: payload we deliberately do not persist.
PLAY_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "play_id",
    "season",
    "week",
    "season_type",
    "posteam",
    "defteam",
    "down",
    "ydstogo",
    "yardline_100",
    "play_type",
    "yards_gained",
    "first_down",
    "third_down_converted",
    "third_down_failed",
    "touchdown",
    "interception",
    "fumble_lost",
)

#: Columns the game selector and team/week derivation need from ``load_schedules``.
GAME_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
)

#: Historical/franchise abbreviations that nflverse or cached exports may use.
TEAM_ALIASES: Final[dict[str, str]] = {
    "WAS": "WSH",
    "LAR": "LA",
    "STL": "LA",
    "SD": "LAC",
    "OAK": "LV",
    "JAC": "JAX",
}

_SEASON_TYPES: Final[dict[str, SeasonType]] = {
    "PRE": SeasonType.PRESEASON,
    "REG": SeasonType.REGULAR,
    "POST": SeasonType.POSTSEASON,
}


class IngestionError(RuntimeError):
    """Base class for rejected nflverse input."""


class MissingColumnsError(IngestionError):
    """The source frame does not look like the nflverse schema we pinned."""


class RowNormalizationError(IngestionError):
    """A single row contains a value the contracts cannot accept."""


class DuplicatePlayError(IngestionError):
    """The same ``(game_id, play_id)`` appeared twice in one fetch."""


@dataclass(frozen=True)
class NormalizedPlays:
    """The kept plays plus counts explaining what was dropped or flattened."""

    plays: tuple[Play, ...] = ()
    skipped_deleted: int = 0
    skipped_non_regular: int = 0
    unknown_play_types: int = 0


@dataclass(frozen=True)
class NormalizedGames:
    """Regular-season games plus a count of non-regular rows that were dropped."""

    games: tuple[Game, ...] = ()
    skipped_non_regular: int = 0


_Loader = Callable[[int], object]


class _Skip(StrEnum):
    """Why a source row was deliberately dropped during normalization."""

    NON_REGULAR = "non_regular"
    DELETED = "deleted"


def _frame_rows(frame: object) -> list[Mapping[str, object]]:
    """Return rows as mappings from a Polars, pandas, or record container."""

    if isinstance(frame, Mapping):
        raise IngestionError(
            "expected a row-oriented frame or a list of mappings, got a single mapping"
        )
    if isinstance(frame, (list, tuple)):
        if not all(isinstance(row, Mapping) for row in frame):
            raise IngestionError("every row must be a mapping of column name to value")
        return list(frame)

    to_dicts = getattr(frame, "to_dicts", None)
    if callable(to_dicts):
        rows = to_dicts()
        if not isinstance(rows, list):  # pragma: no cover - Polars always returns a list
            raise IngestionError("frame.to_dicts() must return a list of rows")
        return rows

    to_dict = getattr(frame, "to_dict", None)
    if callable(to_dict):
        try:
            rows = to_dict(orient="records")
        except TypeError as exc:  # pragma: no cover - defensive against odd frames
            raise IngestionError("frame.to_dict(orient='records') failed") from exc
        if not isinstance(rows, list):  # pragma: no cover
            raise IngestionError("frame.to_dict(orient='records') must return a list")
        return rows

    raise IngestionError(
        f"unsupported frame type {type(frame).__name__}; expected Polars, pandas, or records"
    )


def _row_label(row: Mapping[str, object]) -> str:
    return f"game_id={row.get('game_id')!r}, play_id={row.get('play_id')!r}"


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if type(value).__name__ in {"NAType", "NaTType"}:  # pandas.NA / pandas.NaT
        return True
    if isinstance(value, Real):
        number = float(value)
        return number != number
    try:
        return bool(value != value)
    except (TypeError, ValueError):  # pragma: no cover - exotic scalar types
        return False


def _required_text(value: object, column: str, label: str) -> str:
    if _is_missing(value):
        raise RowNormalizationError(f"{label}: required column {column!r} is empty")
    if not isinstance(value, str):
        raise RowNormalizationError(f"{label}: {column!r} must be text, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise RowNormalizationError(f"{label}: required column {column!r} is empty")
    return text


def _optional_text(value: object, column: str, label: str) -> str | None:
    if _is_missing(value):
        return None
    if not isinstance(value, str):
        raise RowNormalizationError(f"{label}: {column!r} must be text, got {type(value).__name__}")
    text = value.strip()
    return text or None


def _as_int(value: object, column: str, label: str) -> int | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        raise RowNormalizationError(f"{label}: {column!r} must be a whole number, got a bool")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if number.is_integer():
            return int(number)
        raise RowNormalizationError(f"{label}: {column!r} must be a whole number, got {number!r}")
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError as exc:
            raise RowNormalizationError(
                f"{label}: {column!r} must be a whole number, got {text!r}"
            ) from exc
        if number.is_integer():
            return int(number)
    raise RowNormalizationError(f"{label}: {column!r} must be a whole number, got {value!r}")


def _as_flag(value: object, column: str, label: str) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, Real) and float(value) in (0.0, 1.0):
        return bool(value)
    raise RowNormalizationError(f"{label}: {column!r} must be 0/1, got {value!r}")


def _as_team(value: object, column: str, label: str) -> Team | None:
    text = _optional_text(value, column, label)
    if text is None:
        return None
    abbreviation = TEAM_ALIASES.get(text.upper(), text.upper())
    try:
        return Team(abbreviation)
    except ValueError as exc:
        raise RowNormalizationError(
            f"{label}: {column!r} has unknown team abbreviation {text!r}"
        ) from exc


def _required_team(value: object, column: str, label: str) -> Team:
    team = _as_team(value, column, label)
    if team is None:
        raise RowNormalizationError(f"{label}: required column {column!r} is empty")
    return team


def _as_season_type(value: object, column: str, label: str) -> SeasonType:
    text = _required_text(value, column, label).upper()
    try:
        return _SEASON_TYPES[text]
    except KeyError as exc:
        raise RowNormalizationError(
            f"{label}: {column!r} has unknown season type {text!r}"
        ) from exc


def _as_play_type(value: object) -> tuple[PlayType, bool]:
    """Map a source play type; unknown values become ``UNKNOWN`` and are counted."""

    if _is_missing(value):
        return PlayType.UNKNOWN, False
    if not isinstance(value, str):
        return PlayType.UNKNOWN, True
    text = value.strip().lower()
    if not text:
        return PlayType.UNKNOWN, False
    try:
        return PlayType(text), False
    except ValueError:
        return PlayType.UNKNOWN, True


def _as_date(value: object, column: str, label: str) -> date:
    if _is_missing(value):
        raise RowNormalizationError(f"{label}: required column {column!r} is empty")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return date.fromisoformat(text[:10])
        except ValueError as exc:
            raise RowNormalizationError(
                f"{label}: {column!r} is not an ISO date: {text!r}"
            ) from exc
    raise RowNormalizationError(f"{label}: {column!r} must be a date, got {type(value).__name__}")


def _check_columns(
    rows: list[Mapping[str, object]],
    required: Iterable[str],
    kind: str,
) -> None:
    if not rows:
        return
    present = set(rows[0])
    for row in rows[1:]:
        present.update(row)
    missing = sorted(set(required) - present)
    if missing:
        joined = ", ".join(repr(name) for name in missing)
        raise MissingColumnsError(
            f"{kind} frame is missing required column(s): {joined}. "
            "The nflverse schema may have changed; re-run the Stage 2 spike before ingesting."
        )


def _validate_season(season: int, expected: int | None, label: str) -> None:
    if expected is not None and season != expected:
        raise IngestionError(f"{label}: expected season {expected}, got {season}")


def _format_validation_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'record'}: {error['msg']}"
        for error in exc.errors()
    )


def _play_from_row(
    row: Mapping[str, object],
    *,
    expected_season: int | None,
    label: str,
) -> Play | _Skip:
    """Build one play, or say why the row is intentionally dropped."""

    game_id = _required_text(row.get("game_id"), "game_id", label)
    play_id = _as_int(row.get("play_id"), "play_id", label)
    if play_id is None:
        raise RowNormalizationError(f"{label}: required column 'play_id' is empty")
    season = _as_int(row.get("season"), "season", label)
    if season is None:
        raise RowNormalizationError(f"{label}: required column 'season' is empty")
    _validate_season(season, expected_season, label)
    week = _as_int(row.get("week"), "week", label)
    if week is None:
        raise RowNormalizationError(f"{label}: required column 'week' is empty")
    season_type = _as_season_type(row.get("season_type"), "season_type", label)

    if season_type is not SeasonType.REGULAR:
        return _Skip.NON_REGULAR
    if _as_flag(row.get("play_deleted"), "play_deleted", label):
        return _Skip.DELETED

    down = _as_int(row.get("down"), "down", label)
    play_type, _unknown = _as_play_type(row.get("play_type"))
    third_down = down == 3
    interception = _as_flag(row.get("interception"), "interception", label)
    fumble_lost = _as_flag(row.get("fumble_lost"), "fumble_lost", label)
    if interception is None and fumble_lost is None:
        turnover: bool | None = None
    else:
        turnover = bool(interception) or bool(fumble_lost)

    try:
        return Play(
            game_id=game_id,
            play_id=play_id,
            season=season,
            week=week,
            posteam=_as_team(row.get("posteam"), "posteam", label),
            defteam=_as_team(row.get("defteam"), "defteam", label),
            down=down,
            yards_to_go=_as_int(row.get("ydstogo"), "ydstogo", label),
            yardline_100=_as_int(row.get("yardline_100"), "yardline_100", label),
            play_type=play_type,
            yards_gained=_as_int(row.get("yards_gained"), "yards_gained", label),
            first_down=_as_flag(row.get("first_down"), "first_down", label),
            # nflverse fills 0/1 on every row; the flags only mean anything on
            # third down, and the contract rejects them elsewhere.
            third_down_converted=(
                _as_flag(row.get("third_down_converted"), "third_down_converted", label)
                if third_down
                else None
            ),
            third_down_failed=(
                _as_flag(row.get("third_down_failed"), "third_down_failed", label)
                if third_down
                else None
            ),
            touchdown=_as_flag(row.get("touchdown"), "touchdown", label),
            turnover=turnover,
        )
    except ValidationError as exc:
        raise RowNormalizationError(f"{label}: {_format_validation_error(exc)}") from exc


def normalize_plays(frame: object, *, season: int | None = None) -> NormalizedPlays:
    """Normalize a ``load_pbp`` frame into sorted, unique regular-season plays."""

    rows = _frame_rows(frame)
    _check_columns(rows, PLAY_COLUMNS, "play-by-play")

    kept: dict[str, Play] = {}
    skipped_deleted = 0
    skipped_non_regular = 0
    unknown_play_types = 0
    for row in rows:
        label = _row_label(row)
        result = _play_from_row(row, expected_season=season, label=label)
        if isinstance(result, _Skip):
            if result is _Skip.NON_REGULAR:
                skipped_non_regular += 1
            else:
                skipped_deleted += 1
            continue
        if result.play_type is PlayType.UNKNOWN and not _is_missing(row.get("play_type")):
            unknown_play_types += 1
        if result.key in kept:
            raise DuplicatePlayError(
                f"duplicate play {result.key} in one fetch; nflverse keys must be unique"
            )
        kept[result.key] = result

    return NormalizedPlays(
        plays=tuple(sorted(kept.values(), key=lambda play: (play.game_id, play.play_id))),
        skipped_deleted=skipped_deleted,
        skipped_non_regular=skipped_non_regular,
        unknown_play_types=unknown_play_types,
    )


def _game_from_row(
    row: Mapping[str, object],
    *,
    expected_season: int | None,
    label: str,
) -> Game | _Skip:
    game_id = _required_text(row.get("game_id"), "game_id", label)
    season = _as_int(row.get("season"), "season", label)
    if season is None:
        raise RowNormalizationError(f"{label}: required column 'season' is empty")
    _validate_season(season, expected_season, label)
    game_type = _required_text(row.get("game_type"), "game_type", label).upper()
    if game_type not in _SEASON_TYPES:
        raise RowNormalizationError(f"{label}: 'game_type' has unknown value {game_type!r}")
    if _SEASON_TYPES[game_type] is not SeasonType.REGULAR:
        return _Skip.NON_REGULAR

    week = _as_int(row.get("week"), "week", label)
    if week is None:
        raise RowNormalizationError(f"{label}: required column 'week' is empty")
    location = _optional_text(row.get("location"), "location", label)
    try:
        return Game(
            game_id=game_id,
            season=season,
            week=week,
            season_type=SeasonType.REGULAR,
            game_date=_as_date(row.get("gameday"), "gameday", label),
            home_team=_required_team(row.get("home_team"), "home_team", label),
            away_team=_required_team(row.get("away_team"), "away_team", label),
            home_score=_as_int(row.get("home_score"), "home_score", label),
            away_score=_as_int(row.get("away_score"), "away_score", label),
            venue=_optional_text(row.get("stadium"), "stadium", label),
            neutral_site=bool(location and location.lower() == "neutral"),
        )
    except ValidationError as exc:
        raise RowNormalizationError(f"{label}: {_format_validation_error(exc)}") from exc


def normalize_games(frame: object, *, season: int | None = None) -> NormalizedGames:
    """Normalize a ``load_schedules`` frame, keeping regular-season games only."""

    rows = _frame_rows(frame)
    _check_columns(rows, GAME_COLUMNS, "schedule")

    kept: dict[str, Game] = {}
    skipped_non_regular = 0
    for row in rows:
        label = f"game_id={row.get('game_id')!r}"
        result = _game_from_row(row, expected_season=season, label=label)
        if isinstance(result, _Skip):
            skipped_non_regular += 1
            continue
        if result.game_id in kept:
            raise IngestionError(f"duplicate game {result.game_id!r} in one fetch")
        kept[result.game_id] = result

    ordered = tuple(
        sorted(kept.values(), key=lambda game: (game.week, game.game_date, game.game_id))
    )
    return NormalizedGames(games=ordered, skipped_non_regular=skipped_non_regular)


@dataclass(frozen=True)
class IngestionReport:
    """What one refresh fetched, normalized, and wrote."""

    season: int
    games: NormalizedGames
    plays: NormalizedPlays
    refresh: RefreshResult

    def summary(self) -> str:
        return (
            f"season {self.season}: "
            f"{len(self.games.games)} games, {len(self.plays.plays)} plays; "
            f"{self.refresh}"
        )


def _nflverse() -> Any:
    """Import ``nflreadpy`` lazily so tests and the UI do not pay for Polars."""

    return importlib.import_module("nflreadpy")


def _default_schedule_loader(season: int) -> object:
    return _nflverse().load_schedules(season)


def _default_play_loader(season: int) -> object:
    return _nflverse().load_pbp(season)


def current_season() -> int:
    """Return nflverse's current season, used when the CLI is not given one."""

    return int(_nflverse().get_current_season())


class NflverseClient:
    """Fetch and normalize one season of nflverse data.

    Loaders are injectable so tests can pin cached fixtures and never touch the
    network. The defaults call ``nflreadpy`` directly.
    """

    def __init__(
        self,
        season: int,
        *,
        schedule_loader: _Loader | None = None,
        play_loader: _Loader | None = None,
    ) -> None:
        if season < 1920:
            raise ValueError(f"implausible season: {season}")
        self.season = season
        self._load_schedules = schedule_loader or _default_schedule_loader
        self._load_pbp = play_loader or _default_play_loader

    def fetch_games(self) -> NormalizedGames:
        return normalize_games(self._load_schedules(self.season), season=self.season)

    def fetch_plays(self) -> NormalizedPlays:
        return normalize_plays(self._load_pbp(self.season), season=self.season)

    def refresh(self, store: SeasonStore) -> IngestionReport:
        """Normalize both feeds and upsert them atomically into ``store``."""

        if store.season != self.season:
            raise IngestionError(
                f"store season {store.season} does not match client season {self.season}"
            )
        games = self.fetch_games()
        plays = self.fetch_plays()
        result = store.refresh(games.games, plays.plays)
        return IngestionReport(season=self.season, games=games, plays=plays, refresh=result)
