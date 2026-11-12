"""Shared fixture builders for agent, synthesis, orchestrator, and UI tests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from opponent_tendencies.data.index import MediaIndex
from opponent_tendencies.data.media_store import MediaData
from opponent_tendencies.data.play_store import SeasonData
from opponent_tendencies.llm.client import ArcLLMClient
from opponent_tendencies.llm.embeddings import HashingEmbedder
from opponent_tendencies.models import Game, Play, PlayType, Team
from opponent_tendencies.stats.engine import StatsBundle, analyze_opponent
from tests.media_fixtures import write_media_store

SEASON = 2026
OPPONENT = Team.BUF
WEEKS = (5, 6, 7, 8, 9)
API_KEY = "secret-test-key"
BASE_URL = "http://vt-arc.test/v1"

_EVIDENCE_LINE = re.compile(r"^evidence: (?P<unit_id>\S+) kind=(?P<kind>\w+)", re.MULTILINE)


@dataclass(frozen=True)
class FakeRetrieved:
    """A minimal retrieval result for grounding-failure tests."""

    unit_id: str
    text: str
    metadata: dict[str, str | int | float | bool]
    score: float = 0.9

    @property
    def media_id(self) -> str:
        return str(self.metadata.get("media_id", ""))

    @property
    def kind(self) -> str:
        return str(self.metadata.get("unit_kind", ""))

    @property
    def week(self) -> int | None:
        value = self.metadata.get("week")
        return value if isinstance(value, int) and value > 0 else None


def season_data(*, team: Team = OPPONENT) -> SeasonData:
    """Five weeks of games with enough offensive plays to compute tables."""

    games: list[Game] = []
    plays: list[Play] = []
    for offset, week in enumerate(WEEKS):
        game_id = f"2026_{week:02d}_BUF_OPP"
        opponent = Team.KC if team is Team.BUF else Team.BUF
        games.append(
            Game(
                game_id=game_id,
                season=SEASON,
                week=week,
                game_date=date(2026, 10, 4) + timedelta(weeks=offset),
                home_team=team,
                away_team=opponent,
                home_score=24,
                away_score=20,
            )
        )
        for index in range(12):
            down = (index % 3) + 1
            run = index % 2 == 0
            third = down == 3
            plays.append(
                Play(
                    game_id=game_id,
                    play_id=index + 1,
                    season=SEASON,
                    week=week,
                    posteam=team,
                    defteam=opponent,
                    down=down,
                    yards_to_go=(3, 6, 9)[index % 3],
                    yardline_100=(15, 40, 75)[index % 3],
                    play_type=PlayType.RUN if run else PlayType.PASS,
                    yards_gained=(2, 7, -1, 12)[index % 4],
                    first_down=index % 4 == 1,
                    third_down_converted=(third and index % 2 == 0) or None,
                    third_down_failed=(third and index % 2 == 1) or None,
                )
            )
    return SeasonData(season=SEASON, games=tuple(games), plays=tuple(plays))


def opponent_games(*, team: Team = OPPONENT) -> tuple[Game, ...]:
    return tuple(game for game in season_data(team=team).played_games if game.involves(team))


def stats_bundle(*, data: SeasonData | None = None, min_sample: int = 10) -> StatsBundle:
    state = data if data is not None else season_data()
    games = tuple(game for game in state.played_games if game.involves(OPPONENT))
    return analyze_opponent(state.plays, games, opponent=OPPONENT, min_sample=min_sample)


def media_data(tmp_path: Path) -> MediaData:
    _, media = write_media_store(tmp_path)
    return media


def media_index(media: MediaData, *, dimensions: int = 256) -> MediaIndex:
    index = MediaIndex(None, HashingEmbedder(dimensions=dimensions))
    index.build(media)
    return index


def completion(
    content: str,
    *,
    model: str = "test-model",
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
) -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def json_completion(document: object, **kwargs: Any) -> dict[str, Any]:
    return completion(json.dumps(document), **kwargs)


def make_client(handler: Any, **kwargs: Any) -> tuple[ArcLLMClient, list[dict[str, Any]]]:
    """Build an ``ArcLLMClient`` over ``httpx.MockTransport`` and record bodies."""

    requests: list[dict[str, Any]] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        response = handler(request)
        assert isinstance(response, httpx.Response)
        return response

    client = ArcLLMClient(
        BASE_URL,
        API_KEY,
        "test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(recording_handler)),
        retry_backoff_seconds=0.0,
        **kwargs,
    )
    return client, requests


def user_text(body: dict[str, Any]) -> str:
    messages = body["messages"]
    return "\n".join(message["content"] for message in messages if message["role"] == "user")


def evidence_ids(body: dict[str, Any], *, kind: str | None = None) -> list[str]:
    text = user_text(body)
    return [
        match.group("unit_id")
        for match in _EVIDENCE_LINE.finditer(text)
        if kind is None or match.group("kind") == kind
    ]


def agent_payload(
    *,
    evidence_ids: list[str],
    weeks: list[int],
    statement: str = "Buffalo leans on early-down runs.",
    direction: str = "stable",
    category: str = "run_pass",
    observation_category: str = "scheme",
    confidence: str = "high",
    metric: str | None = "run_rate",
    value: float | None = 0.5,
    limitations: list[str] | None = None,
    treat_as_context: bool = False,
) -> dict[str, object]:
    return {
        "tendencies": [
            {
                "statement": statement,
                "category": category,
                "direction": direction,
                "evidence_ids": evidence_ids,
                "confidence": confidence,
                "weeks": weeks,
                "metric": metric,
                "value": value,
                "comparison": "weeks 5-9",
            }
        ],
        "observations": [
            {
                "statement": "Reporters describe condensed formations on third down.",
                "category": observation_category,
                "evidence_ids": evidence_ids,
                "confidence": confidence,
                "weeks": weeks,
                "treat_as_context": treat_as_context,
            }
        ],
        "limitations": limitations or [],
    }


def fixed_now() -> datetime:
    return datetime(2026, 11, 12, 12, 0, tzinfo=UTC)
