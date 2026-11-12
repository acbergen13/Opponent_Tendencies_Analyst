"""Stage 5 validation layer: table-driven rejection and warning paths."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from typing import Any

import pytest
from pydantic import HttpUrl

from opponent_tendencies.data.index import RetrievedSource
from opponent_tendencies.data.media_store import MediaData
from opponent_tendencies.data.play_store import SeasonData
from opponent_tendencies.models import (
    EvidenceRef,
    Game,
    IssueSeverity,
    MediaItem,
    MediaType,
    Play,
    PlayType,
    Quote,
    SeasonType,
    SpeakerRole,
    SplitDimension,
    StatMetric,
    StatRow,
    StatTable,
    StatUnit,
    Team,
)
from opponent_tendencies.stats.engine import DataQuality
from opponent_tendencies.validation.checks import (
    CheckCode,
    check_evidence,
    check_media_alignment,
    check_plays,
    check_records,
    check_retrieval,
    check_stat_tables,
    validate_selection,
)

SEASON = 2026
OPPONENT = Team.BUF
DEFENSE = Team.KC
GAME_ID = "2026_09_BUF_KC"
NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
GAME_DATE = date(2026, 11, 8)
QUOTE_TEXT = "We have to run the ball better on first down."
PRESSER_TAIL = " We will keep working at it."
ARTICLE_BODY = "Buffalo leans on outside-zone runs on early downs."


def _game(
    week: int = 9,
    *,
    game_id: str | None = None,
    home: Team = DEFENSE,
    away: Team = OPPONENT,
    scored: bool = True,
    season_type: SeasonType = SeasonType.REGULAR,
) -> Game:
    return Game(
        game_id=game_id or f"2026_{week:02d}_{away.value}_{home.value}",
        season=SEASON,
        week=week,
        season_type=season_type,
        game_date=GAME_DATE,
        home_team=home,
        away_team=away,
        home_score=24 if scored else None,
        away_score=20 if scored else None,
    )


def _season_data(games: tuple[Game, ...], plays: tuple[Play, ...] = ()) -> SeasonData:
    return SeasonData(season=SEASON, games=games, plays=plays)


def _play(
    game_id: str,
    play_id: int,
    *,
    week: int = 9,
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


def _media(
    *,
    tag: str,
    week: int | None = 9,
    season: int = SEASON,
    opponent: Team = OPPONENT,
    source_type: MediaType = MediaType.ARTICLE,
    game_id: str | None = None,
    body: str = ARTICLE_BODY,
) -> MediaItem:
    return MediaItem(
        source_id=f"test-{tag}",
        source_type=source_type,
        title=f"Source {tag}",
        author="A. Reporter",
        publisher="Test Publisher",
        url=HttpUrl(f"https://example.com/{tag}"),
        published_on=GAME_DATE,
        season=season,
        week=week,
        game_id=game_id,
        opponent=opponent,
        body=body,
        retrieved_at=NOW,
    )


def _presser_media() -> MediaItem:
    return _media(
        tag="presser",
        source_type=MediaType.COACH_QUOTE,
        body=QUOTE_TEXT + PRESSER_TAIL,
        game_id=GAME_ID,
    )


def _quote(
    media: MediaItem,
    *,
    text: str = QUOTE_TEXT,
    char_start: int = 0,
    occurrence: int = 0,
    media_id: str | None = None,
) -> Quote:
    return Quote(
        media_id=media_id if media_id is not None else media.media_id,
        speaker="Sean McDermott",
        speaker_role=SpeakerRole.HEAD_COACH,
        team=OPPONENT,
        season=SEASON,
        week=9,
        game_id=GAME_ID,
        spoken_on=GAME_DATE,
        publisher=media.publisher,
        url=media.url,
        text=text,
        char_start=char_start,
        char_end=char_start + len(text),
        occurrence=occurrence,
    )


def _retrieved(
    unit_id: str,
    media_id: str,
    text: str,
    *,
    kind: str = "chunk",
    quote_id: str | None = None,
    score: float = 0.9,
) -> RetrievedSource:
    metadata: dict[str, Any] = {"unit_kind": kind, "week": 9}
    return RetrievedSource(
        unit_id=unit_id,
        media_id=media_id,
        quote_id=quote_id,
        text=text,
        score=score,
        metadata=metadata,
    )


def _stat_row(label: str, n: int, *, numerator: int) -> StatRow:
    return StatRow(
        label=label,
        value=round(numerator / n, 4) if n else None,
        n=n,
        numerator=numerator if n else None,
        denominator=n if n else None,
        unit=StatUnit.RATE,
        suppressed=n < 5,
    )


def _table(
    *,
    weeks: tuple[int, ...] = (9,),
    game_ids: tuple[str, ...] = (GAME_ID,),
    n_plays: int = 10,
    rows: list[StatRow] | None = None,
) -> StatTable:
    if rows is None:
        rows = [_stat_row("All scrimmage plays", n_plays, numerator=5)]
    return StatTable(
        metric=StatMetric.RUN_RATE,
        title="Run rate",
        dimension=SplitDimension.OVERALL,
        rows=rows,
        n_plays=n_plays,
        game_ids=list(game_ids),
        weeks=list(weeks),
        min_sample=5,
        suppressed=n_plays < 5,
    )


def _game_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "game_id": GAME_ID,
        "season": SEASON,
        "week": 9,
        "game_date": "2026-11-08",
        "home_team": "KC",
        "away_team": "BUF",
        "home_score": 24,
        "away_score": 20,
    }
    base.update(overrides)
    return base


def _codes(issues: Any) -> set[str]:
    return {issue.code for issue in issues}


# ---------------------------------------------------------------------------
# validate_selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("games", "weeks", "opponent", "expected_errors", "expected_warnings", "expected_games"),
    [
        (
            (_game(9, game_id=GAME_ID),),
            [9],
            OPPONENT,
            set(),
            set(),
            1,
        ),
        (
            (_game(9, game_id=GAME_ID),),
            [],
            OPPONENT,
            {CheckCode.SELECTION_WEEKS_EMPTY},
            set(),
            0,
        ),
        (
            (_game(9, game_id=GAME_ID),),
            [9, 9],
            OPPONENT,
            {CheckCode.SELECTION_WEEK_DUPLICATES},
            set(),
            1,
        ),
        (
            (_game(9, game_id=GAME_ID),),
            [19],
            OPPONENT,
            {CheckCode.SELECTION_WEEK_INVALID, CheckCode.SELECTION_NO_GAMES},
            set(),
            0,
        ),
        (
            (_game(9, game_id=GAME_ID),),
            [8],
            OPPONENT,
            {CheckCode.SELECTION_WEEK_NOT_PLAYED, CheckCode.SELECTION_NO_GAMES},
            set(),
            0,
        ),
        (
            (_game(9, game_id=GAME_ID, scored=False),),
            [9],
            OPPONENT,
            {CheckCode.SELECTION_OPPONENT_UNKNOWN, CheckCode.SELECTION_WEEK_NOT_PLAYED},
            set(),
            0,
        ),
        (
            (_game(9, game_id=GAME_ID, season_type=SeasonType.POSTSEASON),),
            [9],
            OPPONENT,
            {CheckCode.SELECTION_OPPONENT_UNKNOWN, CheckCode.SELECTION_WEEK_NOT_PLAYED},
            set(),
            0,
        ),
        (
            (_game(9, game_id=GAME_ID),),
            [9],
            Team.SF,
            {CheckCode.SELECTION_OPPONENT_UNKNOWN},
            set(),
            0,
        ),
        (
            (_game(9, game_id=GAME_ID), _game(10, home=Team.KC, away=Team.DEN)),
            [10],
            OPPONENT,
            {CheckCode.SELECTION_NO_GAMES},
            set(),
            0,
        ),
        (
            (_game(9, game_id=GAME_ID), _game(10, home=Team.KC, away=Team.DEN)),
            [9, 10],
            OPPONENT,
            set(),
            {CheckCode.SELECTION_WEEK_WITHOUT_OPPONENT},
            1,
        ),
    ],
)
def test_validate_selection_table(
    games: tuple[Game, ...],
    weeks: list[int],
    opponent: Team,
    expected_errors: set[str],
    expected_warnings: set[str],
    expected_games: int,
) -> None:
    result = validate_selection(_season_data(games), opponent=opponent, weeks=weeks)

    assert result.ok == (not expected_errors)
    assert _codes(result.errors) == expected_errors
    assert _codes(result.warnings) == expected_warnings
    assert result.checked == {"weeks": len(set(weeks)), "games": expected_games}


# ---------------------------------------------------------------------------
# check_plays
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plays", "expected_errors", "expected_info", "expected_population"),
    [
        ([_play(GAME_ID, 1)], set(), [], 1),
        (
            [_play(GAME_ID, 1), _play(GAME_ID, 1, yards_gained=5)],
            {CheckCode.DUPLICATE_PLAY},
            [],
            2,
        ),
        (
            [
                _play(
                    GAME_ID,
                    1,
                    down=None,
                    yards_to_go=None,
                    yardline_100=None,
                    yards_gained=None,
                )
            ],
            set(),
            [
                (CheckCode.MISSING_VALUES, "down"),
                (CheckCode.MISSING_VALUES, "yards_to_go"),
                (CheckCode.MISSING_VALUES, "yardline_100"),
                (CheckCode.MISSING_VALUES, "yards_gained"),
            ],
            1,
        ),
        (
            [_play(GAME_ID, 1, down=3)],
            set(),
            [(CheckCode.MISSING_VALUES, "third_down_outcomes")],
            1,
        ),
        (
            [_play(GAME_ID, 1, play_type=PlayType.UNKNOWN)],
            set(),
            [(CheckCode.UNKNOWN_PLAY_TYPE, None)],
            1,
        ),
        ([_play(GAME_ID, 1, posteam=DEFENSE, down=None)], set(), [], 0),
        (
            [_play(GAME_ID, 1, play_type=PlayType.PUNT, down=None, yards_gained=None)],
            set(),
            [],
            1,
        ),
    ],
)
def test_check_plays_table(
    plays: list[Play],
    expected_errors: set[str],
    expected_info: list[tuple[CheckCode, str | None]],
    expected_population: int,
) -> None:
    result = check_plays(plays, opponent=OPPONENT, game_ids=[GAME_ID])

    assert result.ok == (not expected_errors)
    assert _codes(result.errors) == expected_errors
    assert result.checked == {"plays": expected_population}
    observed = sorted((issue.code, issue.details.get("field")) for issue in result.warnings)
    assert observed == sorted(expected_info)


def test_check_plays_missing_values_are_explained_not_imputed() -> None:
    result = check_plays([_play(GAME_ID, 1, down=None, yards_gained=None)], opponent=OPPONENT)

    assert result.ok
    assert all(issue.severity is IssueSeverity.INFO for issue in result.warnings)
    assert all("instead of imputing" in issue.message for issue in result.warnings)


# ---------------------------------------------------------------------------
# check_media_alignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("items", "weeks", "season", "season_data", "expected_errors", "expected_warnings"),
    [
        ([_media(tag="wk9")], [9], SEASON, None, set(), set()),
        (
            [_media(tag="wk5", week=5)],
            [9],
            SEASON,
            None,
            set(),
            {CheckCode.MEDIA_BACKGROUND, CheckCode.WEEK_WITHOUT_MEDIA},
        ),
        (
            [_media(tag="wk7", week=7)],
            [5, 9],
            SEASON,
            None,
            set(),
            {CheckCode.MEDIA_BACKGROUND, CheckCode.WEEK_WITHOUT_MEDIA},
        ),
        (
            [_media(tag="wk12", week=12)],
            [9],
            SEASON,
            None,
            {CheckCode.MEDIA_WEEK_LATER},
            {CheckCode.WEEK_WITHOUT_MEDIA},
        ),
        (
            [_media(tag="nowk", week=None)],
            [9],
            SEASON,
            None,
            set(),
            {CheckCode.MEDIA_WEEK_UNKNOWN, CheckCode.WEEK_WITHOUT_MEDIA},
        ),
        (
            [_media(tag="old", week=5, season=2025)],
            [9],
            SEASON,
            None,
            {CheckCode.MEDIA_PRIOR_SEASON},
            {CheckCode.WEEK_WITHOUT_MEDIA},
        ),
        ([], [9], SEASON, None, set(), {CheckCode.NO_MEDIA_FOR_SELECTION}),
        (
            [_media(tag="wk5", week=5)],
            [5, 9],
            SEASON,
            None,
            set(),
            {CheckCode.WEEK_WITHOUT_MEDIA},
        ),
        (
            [_media(tag="wk9g", game_id="2026_09_BUF_XXX")],
            [9],
            SEASON,
            _season_data((_game(9, game_id=GAME_ID),)),
            {CheckCode.MEDIA_GAME_NOT_FOUND},
            set(),
        ),
        (
            [_media(tag="wk9g", game_id="2026_08_BUF_KC")],
            [9],
            SEASON,
            _season_data((_game(8, game_id="2026_08_BUF_KC"),)),
            {CheckCode.MEDIA_GAME_WEEK_MISMATCH},
            set(),
        ),
        (
            [_media(tag="kc", week=5, opponent=DEFENSE), _media(tag="wk9")],
            [9],
            SEASON,
            None,
            set(),
            set(),
        ),
    ],
)
def test_check_media_alignment_table(
    items: list[MediaItem],
    weeks: list[int],
    season: int,
    season_data: SeasonData | None,
    expected_errors: set[str],
    expected_warnings: set[str],
) -> None:
    media_data = MediaData(season=SEASON, media=tuple(items), quotes=())
    expected_checked = sum(1 for item in items if item.opponent is OPPONENT)

    result = check_media_alignment(
        media_data,
        opponent=OPPONENT,
        weeks=weeks,
        season=season,
        season_data=season_data,
    )

    assert result.ok == (not expected_errors)
    assert _codes(result.errors) == expected_errors
    assert _codes(result.warnings) == expected_warnings
    assert result.checked == {"media": expected_checked}


def test_background_sources_are_labeled_not_rejected() -> None:
    media_data = MediaData(season=SEASON, media=(_media(tag="wk5", week=5),), quotes=())

    result = check_media_alignment(media_data, opponent=OPPONENT, weeks=[9], season=SEASON)

    assert result.ok
    background = next(
        issue for issue in result.warnings if issue.code == CheckCode.MEDIA_BACKGROUND
    )
    assert background.details["week"] == "5"
    assert "background" in background.message
    assert not background.requires_human_review


def test_missing_sources_require_human_review() -> None:
    result = check_media_alignment(
        MediaData(season=SEASON, media=(), quotes=()),
        opponent=OPPONENT,
        weeks=[9],
        season=SEASON,
    )

    assert result.ok
    assert result.requires_human_review
    assert _codes(result.warnings) == {CheckCode.NO_MEDIA_FOR_SELECTION}


# ---------------------------------------------------------------------------
# check_retrieval
# ---------------------------------------------------------------------------


def _media_id_of(media_data: MediaData, source_id: str) -> str:
    return next(item.media_id for item in media_data.media if item.source_id == source_id)


def _retrieval_store() -> tuple[MediaData, MediaItem, MediaItem, Quote, MediaItem]:
    week9 = _media(tag="wk9")
    week5 = _media(tag="wk5", week=5)
    week12 = _media(tag="wk12", week=12)
    kc = _media(tag="kc", opponent=DEFENSE)
    presser = _presser_media()
    quote = _quote(presser)
    media_data = MediaData(
        season=SEASON,
        media=(week9, week5, week12, kc, presser),
        quotes=(quote,),
    )
    return media_data, week9, presser, quote, kc


@pytest.mark.parametrize(
    ("label", "keys", "weeks", "season", "expected_errors", "expected_warnings"),
    [
        ("empty", [], [9], SEASON, set(), {CheckCode.RETRIEVAL_EMPTY}),
        ("grounded-chunk", ["wk9-chunk"], [9], SEASON, set(), set()),
        (
            "duplicate-units",
            ["wk9-chunk", "wk9-chunk"],
            [9],
            SEASON,
            {CheckCode.RETRIEVAL_DUPLICATE_UNIT},
            set(),
        ),
        (
            "media-not-found",
            ["missing"],
            [9],
            SEASON,
            {CheckCode.RETRIEVAL_MEDIA_NOT_FOUND},
            set(),
        ),
        (
            "ungrounded-text",
            ["wk9-fabricated"],
            [9],
            SEASON,
            {CheckCode.RETRIEVAL_TEXT_UNGROUNDED},
            set(),
        ),
        (
            "blank-text",
            ["wk9-blank"],
            [9],
            SEASON,
            {CheckCode.RETRIEVAL_TEXT_UNGROUNDED},
            set(),
        ),
        (
            "opponent-mismatch",
            ["kc-chunk"],
            [9],
            SEASON,
            {CheckCode.RETRIEVAL_OPPONENT_MISMATCH},
            set(),
        ),
        (
            "season-mismatch",
            ["wk9-chunk"],
            [9],
            2025,
            {CheckCode.RETRIEVAL_SEASON_MISMATCH},
            set(),
        ),
        ("background-week", ["wk5-chunk"], [9], SEASON, set(), {CheckCode.MEDIA_BACKGROUND}),
        ("later-week", ["wk12-chunk"], [9], SEASON, {CheckCode.MEDIA_WEEK_LATER}, set()),
        ("quote-ok", ["presser-quote"], [9], SEASON, set(), set()),
        (
            "quote-not-found",
            ["ghost-quote"],
            [9],
            SEASON,
            {CheckCode.RETRIEVAL_QUOTE_NOT_FOUND},
            set(),
        ),
        (
            "quote-missing-id",
            ["quote-no-id"],
            [9],
            SEASON,
            {CheckCode.RETRIEVAL_QUOTE_MISSING},
            set(),
        ),
        (
            "quote-text-wrong",
            ["quote-wrong-text"],
            [9],
            SEASON,
            {CheckCode.RETRIEVAL_TEXT_UNGROUNDED},
            set(),
        ),
        (
            "quote-media-mismatch",
            ["quote-cross-media"],
            [9],
            SEASON,
            {CheckCode.RETRIEVAL_QUOTE_MEDIA_MISMATCH},
            set(),
        ),
        (
            "kind-invalid",
            ["wk9-video"],
            [9],
            SEASON,
            {CheckCode.RETRIEVAL_KIND_INVALID},
            set(),
        ),
        (
            "score-invalid",
            ["wk9-nan"],
            [9],
            SEASON,
            {CheckCode.RETRIEVAL_SCORE_INVALID},
            set(),
        ),
    ],
)
def test_check_retrieval_table(
    label: str,
    keys: list[str],
    weeks: list[int],
    season: int,
    expected_errors: set[str],
    expected_warnings: set[str],
) -> None:
    media_data, week9, presser, quote, kc = _retrieval_store()
    built: dict[str, RetrievedSource] = {
        "wk9-chunk": _retrieved("chk-1", week9.media_id, ARTICLE_BODY),
        "wk9-fabricated": _retrieved("chk-2", week9.media_id, "A sentence no article contains."),
        "wk9-blank": _retrieved("chk-3", week9.media_id, "   "),
        "kc-chunk": _retrieved("chk-4", kc.media_id, ARTICLE_BODY),
        "wk5-chunk": _retrieved("chk-5", _media_id_of(media_data, "test-wk5"), ARTICLE_BODY),
        "wk12-chunk": _retrieved("chk-6", _media_id_of(media_data, "test-wk12"), ARTICLE_BODY),
        "wk9-video": _retrieved("unt-7", week9.media_id, ARTICLE_BODY, kind="video"),
        "wk9-nan": _retrieved("unt-8", week9.media_id, ARTICLE_BODY, score=math.nan),
        "missing": _retrieved("unt-9", "med_missing", ARTICLE_BODY),
        "presser-quote": _retrieved(
            "qte-1", presser.media_id, quote.text, kind="quote", quote_id=quote.quote_id
        ),
        "ghost-quote": _retrieved(
            "qte-2", presser.media_id, quote.text, kind="quote", quote_id="qte_missing"
        ),
        "quote-no-id": _retrieved("qte-3", presser.media_id, quote.text, kind="quote"),
        "quote-wrong-text": _retrieved(
            "qte-4", presser.media_id, PRESSER_TAIL.strip(), kind="quote", quote_id=quote.quote_id
        ),
        "quote-cross-media": _retrieved(
            "qte-5", week9.media_id, ARTICLE_BODY, kind="quote", quote_id=quote.quote_id
        ),
    }
    retrieved = tuple(built[key] for key in keys)

    result = check_retrieval(
        retrieved,
        media_data,
        opponent=OPPONENT,
        weeks=weeks,
        season=season,
    )

    assert result.ok == (not expected_errors), label
    assert _codes(result.errors) == expected_errors, label
    assert _codes(result.warnings) == expected_warnings, label
    assert result.checked == {"retrieval": len(retrieved)}, label


def test_empty_retrieval_requires_human_review() -> None:
    media_data, _, _, _, _ = _retrieval_store()

    result = check_retrieval((), media_data, opponent=OPPONENT, weeks=[9], season=SEASON)

    assert result.ok
    assert result.requires_human_review
    assert _codes(result.warnings) == {CheckCode.RETRIEVAL_EMPTY}


# ---------------------------------------------------------------------------
# check_evidence
# ---------------------------------------------------------------------------


def _evidence_stores() -> tuple[SeasonData, MediaData, StatTable]:
    game = _game(9, game_id=GAME_ID)
    play = _play(GAME_ID, 42)
    season_data = _season_data((game,), (play,))
    article = _media(tag="article")
    presser = _presser_media()
    good_quote = _quote(presser)
    ghost_quote = _quote(presser, media_id="med_ghost")
    drifted_quote = _quote(presser, char_start=5, occurrence=1)
    media_data = MediaData(
        season=SEASON,
        media=(article, presser),
        quotes=(good_quote, ghost_quote, drifted_quote),
    )
    return season_data, media_data, _table()


@pytest.mark.parametrize(
    ("label", "ref", "with_tables", "expected_errors"),
    [
        ("play-resolves", "play", True, set()),
        ("game-missing", "bad-game", True, {CheckCode.GAME_NOT_FOUND}),
        ("play-missing", "bad-play", True, {CheckCode.PLAY_NOT_FOUND}),
        ("media-resolves", "media", True, set()),
        ("media-missing", "bad-media", True, {CheckCode.MEDIA_NOT_FOUND}),
        ("media-snippet-ungrounded", "media-bad-snippet", True, {CheckCode.SNIPPET_UNGROUNDED}),
        ("quote-resolves", "quote", True, set()),
        ("quote-missing", "bad-quote", True, {CheckCode.QUOTE_NOT_FOUND}),
        ("quote-media-missing", "ghost-quote", True, {CheckCode.QUOTE_MEDIA_NOT_FOUND}),
        ("quote-not-verbatim", "drifted-quote", True, {CheckCode.QUOTE_NOT_VERBATIM}),
        ("quote-snippet-ungrounded", "quote-bad-snippet", True, {CheckCode.SNIPPET_UNGROUNDED}),
        ("stat-resolves", "stat", True, set()),
        ("stat-missing", "stat", False, {CheckCode.STAT_TABLE_NOT_FOUND}),
    ],
)
def test_check_evidence_table(
    label: str,
    ref: str,
    with_tables: bool,
    expected_errors: set[str],
) -> None:
    season_data, media_data, table = _evidence_stores()
    article = media_data.media[0]
    good_quote = media_data.quotes[0]
    ghost_quote = media_data.quotes[1]
    drifted_quote = media_data.quotes[2]
    refs: dict[str, EvidenceRef] = {
        "play": EvidenceRef.for_play(GAME_ID, 42, label="run"),
        "bad-game": EvidenceRef.for_play("2026_09_BUF_XXX", 42, label="run"),
        "bad-play": EvidenceRef.for_play(GAME_ID, 999, label="run"),
        "media": EvidenceRef.for_media(
            article.media_id, label="breakdown", snippet=ARTICLE_BODY[:20]
        ),
        "bad-media": EvidenceRef.for_media("med_missing", label="breakdown"),
        "media-bad-snippet": EvidenceRef.for_media(
            article.media_id, label="breakdown", snippet="a paraphrase, not a span"
        ),
        "quote": EvidenceRef.for_quote(good_quote.quote_id, label="quote", snippet=QUOTE_TEXT),
        "bad-quote": EvidenceRef.for_quote("qte_missing", label="quote"),
        "ghost-quote": EvidenceRef.for_quote(ghost_quote.quote_id, label="quote"),
        "drifted-quote": EvidenceRef.for_quote(drifted_quote.quote_id, label="quote"),
        "quote-bad-snippet": EvidenceRef.for_quote(
            good_quote.quote_id, label="quote", snippet="paraphrased wording"
        ),
        "stat": EvidenceRef.for_stat(table.stat_table_id, label="run rate"),
    }

    result = check_evidence(
        [refs[ref]],
        season_data=season_data,
        media_data=media_data,
        stat_tables=[table] if with_tables else [],
    )

    assert result.ok == (not expected_errors), label
    assert _codes(result.errors) == expected_errors, label
    assert result.checked == {"evidence": 1}, label


def test_check_evidence_mixed_batch_reports_each_failure() -> None:
    season_data, media_data, table = _evidence_stores()
    refs = [
        EvidenceRef.for_play(GAME_ID, 42, label="run"),
        EvidenceRef.for_play("2026_09_BUF_XXX", 42, label="run"),
        EvidenceRef.for_media("med_missing", label="breakdown"),
    ]

    result = check_evidence(
        refs,
        season_data=season_data,
        media_data=media_data,
        stat_tables=[table],
    )

    assert not result.ok
    assert _codes(result.errors) == {CheckCode.GAME_NOT_FOUND, CheckCode.MEDIA_NOT_FOUND}
    assert result.checked == {"evidence": 3}


# ---------------------------------------------------------------------------
# check_stat_tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "tables", "quality", "expected_errors", "expected_warnings"),
    [
        ("aligned", [_table()], None, set(), set()),
        ("duplicate", [_table(), _table()], None, {CheckCode.STAT_TABLE_DUPLICATE}, set()),
        (
            "week-misaligned",
            [_table(weeks=(8,))],
            None,
            {CheckCode.STAT_TABLE_WEEK_MISALIGNED},
            set(),
        ),
        (
            "game-misaligned",
            [_table(game_ids=("2026_08_BUF_KC",))],
            None,
            {CheckCode.STAT_TABLE_GAME_MISALIGNED},
            set(),
        ),
        (
            "suppressed-table",
            [_table(n_plays=3, rows=[_stat_row("All scrimmage plays", 3, numerator=1)])],
            None,
            set(),
            {CheckCode.STAT_TABLE_SUPPRESSED},
        ),
        (
            "suppressed-rows",
            [
                _table(
                    rows=[
                        _stat_row("Short (0-3)", 3, numerator=2),
                        _stat_row("Medium (4-6)", 7, numerator=4),
                    ]
                )
            ],
            None,
            set(),
            {CheckCode.STAT_ROWS_SUPPRESSED},
        ),
        (
            "quality-gaps",
            [_table()],
            DataQuality(
                offensive_plays=12,
                scrimmage_plays=12,
                third_down_attempts=3,
                unknown_play_types=2,
                missing_down=1,
                missing_yards_to_go=0,
                missing_yardline_100=0,
                missing_yards_gained=4,
                missing_third_down_outcomes=0,
            ),
            set(),
            {CheckCode.MISSING_VALUES, CheckCode.UNKNOWN_PLAY_TYPE},
        ),
    ],
)
def test_check_stat_tables_table(
    label: str,
    tables: list[StatTable],
    quality: DataQuality | None,
    expected_errors: set[str],
    expected_warnings: set[str],
) -> None:
    result = check_stat_tables(tables, weeks=[9], game_ids=[GAME_ID], quality=quality)

    assert result.ok == (not expected_errors), label
    assert _codes(result.errors) == expected_errors, label
    assert _codes(result.warnings) == expected_warnings, label
    assert result.checked == {"stat_tables": len(tables)}, label


def test_suppressed_tables_require_human_review() -> None:
    table = _table(n_plays=3, rows=[_stat_row("All scrimmage plays", 3, numerator=1)])

    result = check_stat_tables([table], weeks=[9], game_ids=[GAME_ID])

    assert result.ok
    assert result.requires_human_review
    assert _codes(result.warnings) == {CheckCode.STAT_TABLE_SUPPRESSED}


def test_stat_tables_empty_weeks_is_structured_error() -> None:
    result = check_stat_tables([_table()], weeks=[], game_ids=[GAME_ID])

    assert not result.ok
    assert _codes(result.errors) == {CheckCode.SELECTION_WEEKS_EMPTY}
    assert result.checked == {"stat_tables": 0}


# ---------------------------------------------------------------------------
# check_records
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "records", "expected_errors", "expected_valid"),
    [
        ("valid", [_game_dict(), _game_dict(game_id="2026_10_BUF_KC", week=10)], set(), 2),
        ("invalid-week", [_game_dict(week=19)], {CheckCode.INVALID_RECORD}, 0),
        ("unknown-field", [_game_dict(unexpected=1)], {CheckCode.INVALID_RECORD}, 0),
        ("not-an-object", ["oops"], {CheckCode.INVALID_RECORD}, 0),
    ],
)
def test_check_records_table(
    label: str,
    records: list[Any],
    expected_errors: set[str],
    expected_valid: int,
) -> None:
    result = check_records(Game, records, kind="games")

    assert result.ok == (not expected_errors), label
    assert _codes(result.errors) == expected_errors, label
    assert result.checked == {"games": expected_valid}, label


def test_check_records_reports_field_paths() -> None:
    result = check_records(Game, [_game_dict(week=19, home_team="NOPE")], kind="games")

    assert not result.ok
    (issue,) = result.errors
    assert issue.code == CheckCode.INVALID_RECORD
    assert "week" in issue.details["errors"]
    assert "home_team" in issue.details["errors"]
