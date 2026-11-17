"""Live media refresh: feeds, week assignment, verbatim quote checks, freshness."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from opponent_tendencies.data.live_media import (
    LEAGUE_FEEDS,
    refresh_live_media,
)
from opponent_tendencies.data.media_store import MediaStore
from opponent_tendencies.data.play_store import SeasonData
from opponent_tendencies.llm.schemas import ExtractedQuote, QuoteExtractionResponse
from opponent_tendencies.models import Game, MediaType, SpeakerRole, Team

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
QUOTE = "We want to run the ball early and make them defend the whole field."
BODY = (
    "The Bills used condensed formations again and ran the ball on early downs. "
    "Buffalo coach Sean McDermott said the plan was simple. "
    f'"{QUOTE}" The Bills kept that approach through the fourth quarter. '
    "Buffalo's offense stayed on schedule and the Bills avoided long third downs. "
) * 2
ARTICLE_URL = "https://www.espn.com/nfl/story/_/id/1/bills-run-game"


def _rss(items: str) -> str:
    return f"<?xml version='1.0'?><rss version='2.0'><channel>{items}</channel></rss>"


def _item(title: str, link: str, day: str = "Mon, 16 Nov 2026 10:00:00 GMT") -> str:
    return (
        f"<item><title>{title}</title><link>{link}</link><pubDate>{day}</pubDate>"
        "<description>Bills news</description></item>"
    )


def _game(week: int, day: date) -> Game:
    return Game(
        game_id=f"2026_{week:02d}_BUF_NYJ",
        season=2026,
        week=week,
        game_date=day,
        home_team=Team.NYJ,
        away_team=Team.BUF,
    )


def _season() -> SeasonData:
    return SeasonData(
        season=2026, games=(_game(9, date(2026, 11, 9)), _game(10, date(2026, 11, 15)))
    )


def _transport(requests: list[str]) -> httpx.MockTransport:
    espn = LEAGUE_FEEDS[0].url

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        requests.append(url)
        if url == espn:
            return httpx.Response(
                200,
                text=_rss(
                    _item("Bills run game", ARTICLE_URL)
                    + _item("Bills odds and picks", ARTICLE_URL + "-odds")
                    + _item("Bills off-domain", "https://example.com/story")
                    + _item("Bills stale", ARTICLE_URL + "-old", "Mon, 01 Jun 2026 10:00:00 GMT")
                ),
            )
        if url == ARTICLE_URL:
            paragraphs = "".join(f"<p>{part.strip()}</p>" for part in BODY.split(". ") if part)
            return httpx.Response(200, text=f"<html><body>{paragraphs}</body></html>")
        return httpx.Response(404)

    return httpx.MockTransport(handler)


class _Client:
    def __init__(self, quotes: list[ExtractedQuote]) -> None:
        self.quotes = quotes
        self.calls = 0

    def complete_structured(self, messages: Any, output: Any, /, **_: Any) -> Any:
        self.calls += 1
        assert output == "quote_extraction_response"
        return SimpleNamespace(parsed=QuoteExtractionResponse(quotes=self.quotes))


def _run(
    tmp_path: Path, client: _Client | None, **kwargs: Any
) -> tuple[Any, MediaStore, list[str]]:
    requests: list[str] = []
    store = MediaStore(2026, tmp_path / "media" / "2026.json")
    http = httpx.Client(transport=_transport(requests), follow_redirects=True)
    report = refresh_live_media(
        Team.BUF,
        season_data=_season(),
        store=store,
        client=client,
        http=http,
        now=lambda: NOW,
        **kwargs,
    )
    return report, store, requests


def test_fetches_team_articles_with_week_and_verbatim_quotes(tmp_path: Path) -> None:
    good = ExtractedQuote(speaker="Sean McDermott", speaker_role=SpeakerRole.HEAD_COACH, text=QUOTE)
    invented = ExtractedQuote(
        speaker="Sean McDermott",
        speaker_role=SpeakerRole.HEAD_COACH,
        text="This sentence was never printed anywhere in the article.",
    )
    player = ExtractedQuote(speaker="A Player", speaker_role=SpeakerRole.OTHER, text=QUOTE)
    client = _Client([good, invented, player])

    report, store, _ = _run(tmp_path, client)

    data = store.load()
    assert report.added == 1 and report.quotes == 1 and report.changed
    (item,) = data.media
    assert item.opponent is Team.BUF and item.week == 10
    assert item.source_type is MediaType.ARTICLE and str(item.url) == ARTICLE_URL
    (quote,) = data.quotes
    assert quote.text == QUOTE and item.body[quote.char_start : quote.char_end] == QUOTE


def test_skips_offdomain_noise_and_stale_entries(tmp_path: Path) -> None:
    _, _, requests = _run(tmp_path, None)

    assert ARTICLE_URL in requests
    assert not any(
        "odds" in url or "example.com" in url or url.endswith("-old") for url in requests
    )


def test_second_run_inside_ttl_makes_no_requests(tmp_path: Path) -> None:
    _run(tmp_path, None)
    report, _, requests = _run(tmp_path, None)

    assert report.skipped_fresh and requests == []


def test_stored_articles_are_not_refetched_or_reextracted(tmp_path: Path) -> None:
    client = _Client([])
    _run(tmp_path, client)
    later = NOW + timedelta(days=1)
    store = MediaStore(2026, tmp_path / "media" / "2026.json")
    requests: list[str] = []
    report = refresh_live_media(
        Team.BUF,
        season_data=_season(),
        store=store,
        client=client,
        http=httpx.Client(transport=_transport(requests)),
        now=lambda: later,
    )

    assert report.added == 0 and ARTICLE_URL not in requests and client.calls == 1


def test_feed_failure_is_not_fatal_and_does_not_stamp(tmp_path: Path) -> None:
    store = MediaStore(2026, tmp_path / "media" / "2026.json")
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    report = refresh_live_media(
        Team.BUF, season_data=_season(), store=store, client=None, http=http, now=lambda: NOW
    )

    assert report.added == 0 and report.errors
    assert not (tmp_path / "media" / "live_2026.json").exists()
