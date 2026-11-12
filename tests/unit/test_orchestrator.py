"""Stage 9 orchestration: happy path, failure matrix, and bounded stages.

Every LLM call runs through ``httpx.MockTransport``; retrieval uses the tracked
media corpus with ``HashingEmbedder`` in a temporary Chroma collection, so the
tests never touch the network or VT ARC.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from opponent_tendencies.data.play_store import SeasonData
from opponent_tendencies.llm.prompts import PROMPT_VERSION
from opponent_tendencies.models import (
    AgentName,
    GapReason,
    RunFailureReason,
    RunState,
    SectionStatus,
    Team,
)
from opponent_tendencies.orchestrator import ReportRun, run_report
from opponent_tendencies.validation.checks import CheckCode
from tests.agent_fixtures import (
    WEEKS,
    agent_payload,
    evidence_ids,
    fixed_now,
    json_completion,
    make_client,
    media_data,
    media_index,
    season_data,
)

SELECTED = list(WEEKS)
AGENT_NAMES = {
    AgentName.STATISTICS,
    AgentName.BEAT_REPORTER,
    AgentName.COACH_QUOTE,
}

_EVIDENCE = re.compile(r"^evidence: (?P<unit_id>\S+) kind=(?P<kind>\w+)", re.MULTILINE)
_CLAIM = re.compile(
    r'<claim claim_id="(?P<claim_id>[^"]+)">(?P<body>.*?)</claim>',
    re.DOTALL,
)
_SNIPPET = re.compile(r'<snippet evidence_id="(?P<evidence_id>[^"]+)">')


def _prompt(request: httpx.Request) -> str:
    body = json.loads(request.content)
    return "\n".join(
        message["content"] for message in body["messages"] if message["role"] == "user"
    )


def _classify(text: str) -> str:
    if "<claims_to_verify" in text:
        return "verifier"
    if "<agent_findings" in text:
        return "synthesis"
    if "<deterministic_statistics>" in text:
        return "sta"
    if "speaker=" in text:
        return "cqa"
    if "<retrieved_data" in text:
        return "bra"
    raise AssertionError("unrecognized prompt")


def _catalog(text: str) -> dict[str, str]:
    return {match.group("unit_id"): match.group("kind") for match in _EVIDENCE.finditer(text)}


def _agent_document(kind: str, body: dict[str, Any]) -> dict[str, object]:
    if kind == "sta":
        ids = evidence_ids(body, kind="stat")
        return agent_payload(evidence_ids=[ids[0]], weeks=SELECTED)
    if kind == "bra":
        ids = evidence_ids(body, kind="media")
        return agent_payload(
            evidence_ids=[ids[0]],
            weeks=SELECTED,
            category="formation",
        )
    ids = evidence_ids(body, kind="quote")
    return agent_payload(
        evidence_ids=[ids[0]],
        weeks=[9],
        category="other",
        observation_category="coach_explanation",
        confidence="medium",
        metric=None,
        value=None,
        treat_as_context=True,
    )


def _synthesis_document(text: str) -> dict[str, object]:
    catalog = _catalog(text)
    assert catalog, "synthesis must see an evidence catalog"
    stats = [item for item, kind in catalog.items() if kind == "stat"]
    media = [item for item, kind in catalog.items() if kind == "media"]
    tendency_evidence = stats[0] if stats else sorted(catalog)[0]
    observation_evidence = media[0] if media else tendency_evidence
    return {
        "overview": "Buffalo leans on early-down runs; reporting adds context.",
        "tendencies": [
            {
                "statement": "Buffalo runs on early downs.",
                "category": "run_pass",
                "direction": "stable",
                "evidence_ids": [tendency_evidence],
                "confidence": "high",
                "weeks": SELECTED,
                "metric": "run_rate",
                "value": 0.5,
                "comparison": "weeks 5-9",
            }
        ],
        "observations": [
            {
                "statement": "Reporters describe condensed formations on third down.",
                "category": "scheme",
                "evidence_ids": [observation_evidence],
                "confidence": "medium",
                "weeks": SELECTED,
                "treat_as_context": False,
            }
        ],
        "agreements": (
            [
                {
                    "topic": "Early-down identity",
                    "statement": "The tables and reporting agree on early-down runs.",
                    "agents": ["STA", "BRA"],
                    "evidence_ids": [tendency_evidence, observation_evidence],
                    "weeks": SELECTED,
                }
            ]
            if media
            else []
        ),
        "conflicts": [],
        "gaps": [
            {
                "topic": "Personnel usage",
                "description": "No player-level snap counts are stored for the selection.",
                "reason": "other",
                "agents": [],
                "weeks": SELECTED,
            }
        ],
        "weak_evidence": [],
    }


def _verifier_document(text: str, *, verdict: str = "supported") -> dict[str, object]:
    claims: list[dict[str, object]] = []
    for match in _CLAIM.finditer(text):
        snippets = [
            snippet.group("evidence_id") for snippet in _SNIPPET.finditer(match.group("body"))
        ]
        if verdict == "supported" and snippets:
            claims.append(
                {
                    "claim_id": match.group("claim_id"),
                    "verdict": "supported",
                    "reason": "The attached snippet supports the claim.",
                    "evidence_ids": [snippets[0]],
                    "correction": None,
                }
            )
        else:
            claims.append(
                {
                    "claim_id": match.group("claim_id"),
                    "verdict": "unsupported",
                    "reason": "The attached snippets do not support the claim.",
                    "evidence_ids": [],
                    "correction": None,
                }
            )
    return {"claims": claims}


def _routing_handler(
    *,
    fail: frozenset[str] = frozenset(),
    delay: dict[str, float] | None = None,
    verdict: str = "supported",
) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        text = _prompt(request)
        kind = _classify(text)
        if kind in fail:
            return httpx.Response(500, json={"error": {"message": "injected failure"}})
        if delay and kind in delay:
            time.sleep(delay[kind])
        if kind == "verifier":
            document = _verifier_document(text, verdict=verdict)
        elif kind == "synthesis":
            document = _synthesis_document(text)
        else:
            document = _agent_document(kind, json.loads(request.content))
        return httpx.Response(200, json=json_completion(document))

    return handler


def _run(
    *,
    tmp_path: Path | None,
    handler: Any,
    data: SeasonData | None = None,
    weeks: list[int] | None = None,
    **kwargs: Any,
) -> tuple[ReportRun, list[dict[str, Any]]]:
    media = media_data(tmp_path) if tmp_path is not None else None
    index = media_index(media) if media is not None else None
    client, requests = make_client(handler, max_retries=0)
    run = run_report(
        opponent=Team.BUF,
        weeks=weeks if weeks is not None else SELECTED,
        season_data=data if data is not None else season_data(),
        media_data=media,
        index=index,
        client=client,
        clock=fixed_now,
        **kwargs,
    )
    return run, requests


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_report_completes_end_to_end(tmp_path: Path) -> None:
    run, requests = _run(tmp_path=tmp_path, handler=_routing_handler())

    assert len(requests) == 5
    report = run.report
    assert report is not None
    assert run.status.state is RunState.COMPLETED
    assert report.status.state is RunState.COMPLETED
    assert report.opponent is Team.BUF
    assert report.weeks == SELECTED
    assert report.season == 2026
    assert {output.agent for output in report.agent_outputs} == AGENT_NAMES
    assert all(output.status is SectionStatus.COMPLETED for output in report.agent_outputs)
    assert report.status.agent_status[AgentName.SYNTHESIS] is SectionStatus.COMPLETED
    assert report.report_id.startswith("rpt_")
    assert report.data_revision
    assert report.prompt_version == PROMPT_VERSION
    assert report.model == "test-model"

    assert report.data_used.games == len(report.games) == len(WEEKS)
    assert report.data_used.plays == 60
    assert report.data_used.quotes >= 1
    assert report.data_used.source_ids
    assert report.data_used.source_ids == sorted(set(report.data_used.source_ids))

    available = {item.evidence_id for item in report.evidence}
    referenced = {
        evidence_id
        for finding in [*report.tendencies, *report.observations]
        for evidence_id in finding.evidence_ids
    }
    referenced.update(
        evidence_id for agreement in report.agreements for evidence_id in agreement.evidence_ids
    )
    referenced.update(
        evidence_id for weak in report.weak_evidence for evidence_id in weak.evidence_ids
    )
    assert referenced <= available

    assert run.llm_calls == 5
    assert run.usage.total_tokens == 90
    assert run.validation is not None and run.validation.ok
    assert run.synthesis is not None and run.verification is not None


def test_run_report_emits_progress_events(tmp_path: Path) -> None:
    events: list[Any] = []
    run, _ = _run(tmp_path=tmp_path, handler=_routing_handler(), on_progress=events.append)

    assert run.report is not None
    starts = {(event.agent, event.status) for event in events}
    for agent in AGENT_NAMES:
        assert (agent, SectionStatus.RUNNING) in starts
        assert (agent, SectionStatus.COMPLETED) in starts
    assert (AgentName.SYNTHESIS, SectionStatus.COMPLETED) in starts
    assert any(
        event.step == "verification" and event.status is SectionStatus.COMPLETED for event in events
    )
    assert any(
        event.step == "report" and event.status is SectionStatus.COMPLETED for event in events
    )


def test_run_report_is_deterministic_for_the_same_inputs(tmp_path: Path) -> None:
    first, _ = _run(tmp_path=tmp_path, handler=_routing_handler())
    second, _ = _run(tmp_path=tmp_path, handler=_routing_handler())

    assert first.report is not None and second.report is not None
    assert first.report.report_id == second.report.report_id
    assert first.report.data_revision == second.report.data_revision


# ---------------------------------------------------------------------------
# Degraded failures: BRA/CQA
# ---------------------------------------------------------------------------


def test_missing_media_degrades_bra_and_cqa() -> None:
    run, requests = _run(tmp_path=None, handler=_routing_handler())

    report = run.report
    assert report is not None
    assert run.status.state is RunState.DEGRADED
    assert report.is_partial
    assert set(report.unavailable_sections) == {
        AgentName.BEAT_REPORTER,
        AgentName.COACH_QUOTE,
    }
    assert any(issue.code == CheckCode.AGENT_UNAVAILABLE.value for issue in report.warnings)
    assert any(gap.reason is GapReason.AGENT_UNAVAILABLE for gap in report.gaps)
    # STA + synthesis + verification; the unavailable sections never call a model.
    assert len(requests) == 3
    assert run.llm_calls == 3


def test_bra_failure_degrades_only_that_section(tmp_path: Path) -> None:
    run, _ = _run(
        tmp_path=tmp_path,
        handler=_routing_handler(fail=frozenset({"bra"})),
    )

    report = run.report
    assert report is not None
    assert run.status.state is RunState.DEGRADED
    assert report.unavailable_sections == (AgentName.BEAT_REPORTER,)
    assert any(
        issue.code == CheckCode.AGENT_UNAVAILABLE.value
        and issue.subject_id == AgentName.BEAT_REPORTER.value
        for issue in report.warnings
    )
    assert report.tendencies


def test_stalled_cqa_worker_is_abandoned_within_the_timeout(tmp_path: Path) -> None:
    started = time.monotonic()
    run, _ = _run(
        tmp_path=tmp_path,
        handler=_routing_handler(delay={"cqa": 0.3}),
        agent_timeout_seconds=0.05,
    )
    elapsed = time.monotonic() - started

    report = run.report
    assert elapsed < 1.0
    assert report is not None
    assert run.status.state is RunState.DEGRADED
    assert report.status.agent_status[AgentName.COACH_QUOTE] is SectionStatus.TIMED_OUT
    assert AgentName.COACH_QUOTE in report.unavailable_sections
    assert any(issue.code == CheckCode.AGENT_TIMEOUT.value for issue in report.warnings)


# ---------------------------------------------------------------------------
# Hard failures: validation, STA, deadlines, budgets
# ---------------------------------------------------------------------------


def test_invalid_selection_fails_without_calling_a_model() -> None:
    run, requests = _run(tmp_path=None, handler=_routing_handler(), weeks=[1])

    assert requests == []
    assert run.report is None
    assert run.status.state is RunState.FAILED
    assert run.status.failure_reason is RunFailureReason.VALIDATION_FAILED
    assert run.status.message
    assert run.llm_calls == 0


def test_duplicate_plays_fail_validation_before_the_agents() -> None:
    data = season_data()
    forked = SeasonData(
        season=data.season,
        games=data.games,
        plays=(*data.plays, data.plays[0]),
    )
    run, requests = _run(tmp_path=None, handler=_routing_handler(), data=forked)

    assert requests == []
    assert run.report is None
    assert run.status.failure_reason is RunFailureReason.VALIDATION_FAILED
    assert "appears more than once" in (run.status.message or "").lower()


def test_statistics_agent_failure_is_a_hard_failure(tmp_path: Path) -> None:
    run, _ = _run(
        tmp_path=tmp_path,
        handler=_routing_handler(fail=frozenset({"sta"})),
    )

    assert run.report is None
    assert run.status.state is RunState.FAILED
    assert run.status.failure_reason is RunFailureReason.STATISTICS_AGENT_FAILED
    assert run.status.message


def test_statistics_agent_timeout_is_a_hard_failure(tmp_path: Path) -> None:
    run, _ = _run(
        tmp_path=tmp_path,
        handler=_routing_handler(delay={"sta": 0.3}),
        agent_timeout_seconds=0.05,
    )

    assert run.report is None
    assert run.status.state is RunState.FAILED
    assert run.status.failure_reason is RunFailureReason.STATISTICS_AGENT_FAILED
    assert "time" in (run.status.message or "").lower()


def test_overall_deadline_is_a_hard_failure(tmp_path: Path) -> None:
    run, _ = _run(
        tmp_path=tmp_path,
        handler=_routing_handler(delay={"sta": 0.3}),
        deadline_seconds=0.001,
    )

    assert run.report is None
    assert run.status.state is RunState.FAILED
    assert run.status.failure_reason is RunFailureReason.DEADLINE_EXCEEDED


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_llm_calls": 2}, "step cap"),
        ({"token_budget": 10}, "token budget"),
    ],
)
def test_resource_budgets_stop_the_run(
    tmp_path: Path,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    run, _ = _run(tmp_path=tmp_path, handler=_routing_handler(), **kwargs)

    assert run.report is None
    assert run.status.failure_reason is RunFailureReason.DEADLINE_EXCEEDED
    assert message in (run.status.message or "").lower()


# ---------------------------------------------------------------------------
# Verification removes unsupported claims
# ---------------------------------------------------------------------------


def test_unsupported_claims_are_removed_before_the_report(tmp_path: Path) -> None:
    run, _ = _run(
        tmp_path=tmp_path,
        handler=_routing_handler(verdict="unsupported"),
    )

    report = run.report
    assert report is not None
    assert run.status.state is RunState.COMPLETED
    assert report.tendencies == []
    assert report.observations == []
    assert report.weak_evidence
    assert any(
        issue.code == CheckCode.VERIFICATION_REMOVED_CLAIM.value for issue in report.warnings
    )
    assert run.verified is not None
    assert run.verified.removed == 2
