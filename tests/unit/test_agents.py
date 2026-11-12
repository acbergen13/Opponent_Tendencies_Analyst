"""Stage 7 specialist agents: grounding, stamping, and explicit limitations.

Every LLM call runs on ``httpx.MockTransport``; retrieval uses the tracked media
corpus with ``HashingEmbedder`` and an ephemeral Chroma collection. The tests
check the split of responsibilities: the model chooses wording, Python owns
evidence IDs, weeks, status, and every rejection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from opponent_tendencies.agents import (
    beat_reporter_query,
    coach_quote_query,
    run_beat_reporter_agent,
    run_coach_quote_agent,
    run_statistics_agent,
)
from opponent_tendencies.agents.common import format_weeks
from opponent_tendencies.data.index import MediaIndex
from opponent_tendencies.models import AgentName, SectionStatus, Team
from tests.agent_fixtures import (
    WEEKS,
    agent_payload,
    evidence_ids,
    json_completion,
    make_client,
    media_data,
    media_index,
    opponent_games,
    season_data,
    stats_bundle,
)

SELECTED = list(WEEKS)


def _respond(payload_builder: Any) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json=json_completion(payload_builder(body)))

    return handler


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def test_format_weeks() -> None:
    assert format_weeks([5]) == "5"
    assert format_weeks([5, 6, 7]) == "5-7"
    assert format_weeks([5, 7, 9]) == "5, 7, 9"
    with pytest.raises(ValueError, match="must not be empty"):
        format_weeks([])


def test_queries_are_deterministic_and_name_the_opponent() -> None:
    assert beat_reporter_query(Team.BUF, [5, 6]) == beat_reporter_query(Team.BUF, [5, 6])
    assert "BUF" in beat_reporter_query(Team.BUF, [5, 6])
    assert "weeks 5-6" in beat_reporter_query(Team.BUF, [5, 6])
    assert "BUF" in coach_quote_query(Team.BUF, [9])


# ---------------------------------------------------------------------------
# Statistics agent
# ---------------------------------------------------------------------------


def test_statistics_agent_renders_tables_and_stamps_identity() -> None:
    bundle = stats_bundle()
    games = opponent_games()

    def payload(body: dict[str, Any]) -> dict[str, object]:
        ids = evidence_ids(body, kind="stat")
        return agent_payload(evidence_ids=[ids[0]], weeks=SELECTED)

    client, requests = make_client(_respond(payload))
    result = run_statistics_agent(
        client=client,
        games=games,
        weeks=SELECTED,
        opponent=Team.BUF,
        bundle=bundle,
    )

    assert len(requests) == 1
    prompt = requests[0]["messages"][1]["content"]
    assert "<deterministic_statistics>" in prompt
    assert "<available_evidence>" in prompt
    assert "<retrieved_data" not in prompt

    output = result.output
    assert output.agent is AgentName.STATISTICS
    assert output.opponent is Team.BUF
    assert output.games == list(games)
    assert output.weeks == SELECTED
    assert output.stat_tables == list(bundle.tables)
    assert output.is_available
    assert output.tendencies and output.observations
    assert output.evidence == sorted(output.evidence, key=lambda item: item.evidence_id)
    available = {item.evidence_id for item in output.evidence}
    assert set(output.tendencies[0].evidence_ids) <= available
    assert result.llm_calls == 1
    assert result.usage.total_tokens == 18


def test_statistics_agent_drops_invented_evidence_and_explains() -> None:
    bundle = stats_bundle()
    client, _ = make_client(
        _respond(
            lambda body: agent_payload(
                evidence_ids=["ev:stat:invented"],
                weeks=SELECTED,
            )
        )
    )
    result = run_statistics_agent(
        client=client,
        games=opponent_games(),
        weeks=SELECTED,
        opponent=Team.BUF,
        bundle=bundle,
    )
    output = result.output
    assert output.tendencies == []
    assert output.observations == []
    assert any("Dropped 2 finding(s)" in limitation for limitation in output.limitations)


def test_statistics_agent_rejects_weeks_outside_the_selection() -> None:
    bundle = stats_bundle()
    client, _ = make_client(
        _respond(
            lambda body: agent_payload(
                evidence_ids=[evidence_ids(body, kind="stat")[0]],
                weeks=[1],
            )
        )
    )
    result = run_statistics_agent(
        client=client,
        games=opponent_games(),
        weeks=SELECTED,
        opponent=Team.BUF,
        bundle=bundle,
    )
    assert result.output.tendencies == []
    assert any("Dropped 2 finding(s)" in item for item in result.output.limitations)


def test_statistics_agent_marks_suppressed_tables_as_weak_evidence() -> None:
    bundle = stats_bundle(min_sample=10_000)
    assert all(table.suppressed for table in bundle.tables)
    client, _ = make_client(
        _respond(
            lambda body: agent_payload(
                evidence_ids=[evidence_ids(body, kind="stat")[0]],
                weeks=SELECTED,
                confidence="high",
            )
        )
    )
    result = run_statistics_agent(
        client=client,
        games=opponent_games(),
        weeks=SELECTED,
        opponent=Team.BUF,
        bundle=bundle,
    )
    for finding in [*result.output.tendencies, *result.output.observations]:
        assert finding.weak_evidence
        assert finding.confidence.value == "medium"


def test_statistics_agent_rejects_a_bundle_for_another_selection() -> None:
    bundle = stats_bundle()
    client, _ = make_client(_respond(lambda body: agent_payload(evidence_ids=[], weeks=[])))
    with pytest.raises(ValueError, match="outside the selection"):
        run_statistics_agent(
            client=client,
            games=opponent_games(),
            weeks=SELECTED[:-1],
            opponent=Team.BUF,
            bundle=bundle,
        )


# ---------------------------------------------------------------------------
# Beat-reporter agent
# ---------------------------------------------------------------------------


def test_beat_reporter_agent_retrieves_validated_media(tmp_path: Path) -> None:
    media = media_data(tmp_path)
    index = media_index(media)

    def payload(body: dict[str, Any]) -> dict[str, object]:
        ids = evidence_ids(body, kind="media")
        return agent_payload(evidence_ids=[ids[0]], weeks=SELECTED, category="formation")

    client, requests = make_client(_respond(payload))
    result = run_beat_reporter_agent(
        client=client,
        index=index,
        media_data=media,
        games=opponent_games(),
        weeks=SELECTED,
        opponent=Team.BUF,
    )

    prompt = requests[0]["messages"][1]["content"]
    assert "<retrieved_data" in prompt
    assert "<source " in prompt
    assert "publisher=" in prompt
    assert "<deterministic_statistics>" not in prompt

    output = result.output
    assert output.agent is AgentName.BEAT_REPORTER
    assert output.is_available
    assert output.evidence
    assert all(item.kind.value == "media" for item in output.evidence)
    assert any("background context only" in item for item in output.limitations)
    assert any(issue.code == "media_background" for issue in result.warnings)


def test_beat_reporter_agent_skips_the_model_when_nothing_matched(tmp_path: Path) -> None:
    media = media_data(tmp_path)
    index = media_index(media)

    def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the model must not be called without sources")

    client, requests = make_client(forbidden)
    data = season_data(team=Team.ARI)
    games = tuple(game for game in data.played_games if game.involves(Team.ARI))
    result = run_beat_reporter_agent(
        client=client,
        index=index,
        media_data=media,
        games=games,
        weeks=SELECTED,
        opponent=Team.ARI,
    )
    assert requests == []
    assert result.output.is_available
    assert result.output.status is SectionStatus.COMPLETED
    assert result.output.tendencies == []
    assert any(
        "No stored articles or film breakdowns" in item for item in result.output.limitations
    )


def test_beat_reporter_agent_refuses_ungrounded_retrieval(tmp_path: Path) -> None:
    media = media_data(tmp_path)
    fake = cast(MediaIndex, _BadIndex())
    client, requests = make_client(
        lambda request: (_ for _ in ()).throw(AssertionError("must not call the model"))
    )
    result = run_beat_reporter_agent(
        client=client,
        index=fake,
        media_data=media,
        games=opponent_games(),
        weeks=SELECTED,
        opponent=Team.BUF,
    )
    assert requests == []
    assert result.output.status is SectionStatus.UNAVAILABLE
    assert any("failed grounding checks" in item for item in result.output.limitations)


class _BadIndex:
    def retrieve(self, query: str, **kwargs: Any) -> tuple[Any, ...]:
        from tests.agent_fixtures import FakeRetrieved

        return (
            FakeRetrieved(
                unit_id="chk_bad",
                text="This text is not stored in the media store.",
                metadata={
                    "media_id": "med_missing",
                    "unit_kind": "chunk",
                    "publisher": "Unknown",
                },
            ),
        )


# ---------------------------------------------------------------------------
# Coach-quote agent
# ---------------------------------------------------------------------------


def test_coach_quote_agent_preserves_quote_evidence_and_context(tmp_path: Path) -> None:
    media = media_data(tmp_path)
    index = media_index(media)

    def payload(body: dict[str, Any]) -> dict[str, object]:
        ids = evidence_ids(body, kind="quote")
        return agent_payload(
            evidence_ids=[ids[0]],
            weeks=[9],
            category="other",
            observation_category="coach_explanation",
            confidence="medium",
            metric=None,
            value=None,
            treat_as_context=False,
        )

    client, requests = make_client(_respond(payload))
    games = tuple(game for game in opponent_games() if game.week == 9)
    result = run_coach_quote_agent(
        client=client,
        index=index,
        media_data=media,
        games=games,
        weeks=[9],
        opponent=Team.BUF,
    )

    prompt = requests[0]["messages"][1]["content"]
    assert "speaker=" in prompt
    assert "spoken_on=" in prompt
    assert "<retrieved_data" in prompt

    output = result.output
    assert output.agent is AgentName.COACH_QUOTE
    assert output.evidence
    assert all(item.kind.value == "quote" for item in output.evidence)
    assert output.observations
    assert all(observation.treat_as_context for observation in output.observations)


def test_coach_quote_agent_skips_the_model_without_quotes(tmp_path: Path) -> None:
    media = media_data(tmp_path)
    index = media_index(media)
    client, requests = make_client(
        lambda request: (_ for _ in ()).throw(AssertionError("must not call the model"))
    )
    data = season_data(team=Team.ARI)
    games = tuple(game for game in data.played_games if game.involves(Team.ARI))
    result = run_coach_quote_agent(
        client=client,
        index=index,
        media_data=media,
        games=games,
        weeks=SELECTED,
        opponent=Team.ARI,
    )
    assert requests == []
    assert result.output.is_available
    assert any("No stored post-game coach quotes" in item for item in result.output.limitations)
