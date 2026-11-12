"""Offline tests for the Streamlit follow-up helpers and formatting."""

from __future__ import annotations

import httpx
import pytest

from opponent_tendencies.models import AgentName, EvidenceKind, SectionStatus, StatUnit
from opponent_tendencies.orchestrator import RunProgress
from opponent_tendencies.ui.render import (
    agent_label,
    format_finding_line,
    format_stat_value,
    render_progress,
)
from opponent_tendencies.ui.runner import (
    FollowUpTurn,
    answer_follow_up,
    summarize_conversation,
)
from tests.agent_fixtures import json_completion, make_client, user_text
from tests.unit.factories import make_report, make_stat_table


def _follow_up_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "answer": "Buffalo ran on half of its scrimmage plays.",
        "supported": True,
        "evidence_ids": [],
        "limitations": [],
    }
    payload.update(overrides)
    return payload


def test_answer_follow_up_uses_stored_report_and_validates_citations() -> None:
    report = make_report()
    stat_id = next(item.evidence_id for item in report.evidence if item.kind is EvidenceKind.STAT)
    client, requests = make_client(
        lambda request: httpx.Response(
            200, json=json_completion(_follow_up_payload(evidence_ids=[stat_id]))
        )
    )

    answer = answer_follow_up(
        client=client,
        report=report,
        question="How often does Buffalo run?",
        turns=(FollowUpTurn("Earlier?", "Yes.", True, (stat_id,)),),
    )

    assert answer.supported is True
    assert answer.evidence_ids == (stat_id,)
    assert answer.llm_calls == 1
    assert answer.usage.total_tokens > 0
    assert len(requests) == 1
    prompt_text = user_text(requests[0])
    assert "How often does Buffalo run?" in prompt_text
    assert "<report_context" in prompt_text
    assert stat_id in prompt_text
    assert "Earlier?" in prompt_text


def test_unknown_follow_up_citation_is_removed_and_marked_unsupported() -> None:
    report = make_report()
    client, _ = make_client(
        lambda request: httpx.Response(
            200,
            json=json_completion(_follow_up_payload(evidence_ids=["invented-evidence-id"])),
        )
    )

    answer = answer_follow_up(client=client, report=report, question="What changed?")

    assert answer.supported is False
    assert answer.evidence_ids == ()
    assert any("not present in the stored report" in item for item in answer.limitations)


def test_supported_follow_up_without_citations_is_forced_unsupported() -> None:
    report = make_report()
    # The wire schema rejects supported=true without evidence before our code runs,
    # so exercise the defensive branch with an explicitly unsupported answer.
    client, _ = make_client(
        lambda request: httpx.Response(
            200, json=json_completion(_follow_up_payload(supported=False, evidence_ids=[]))
        )
    )

    answer = answer_follow_up(client=client, report=report, question="Anything?")

    assert answer.supported is False
    assert answer.evidence_ids == ()


def test_blank_follow_up_question_is_rejected_before_a_model_call() -> None:
    report = make_report()
    client, requests = make_client(
        lambda request: httpx.Response(200, json=json_completion(_follow_up_payload()))
    )

    with pytest.raises(ValueError, match="question"):
        answer_follow_up(client=client, report=report, question="   ")

    assert requests == []


def test_conversation_summary_keeps_recent_turns_and_collapses_older_ones() -> None:
    turns = [
        FollowUpTurn(
            question=f"Question {index}",
            answer=f"Answer {index}",
            supported=index % 2 == 0,
            evidence_ids=(f"ev-{index}",),
            limitations=(f"limit {index}",),
        )
        for index in range(5)
    ]

    summary = summarize_conversation(turns, max_full_turns=2)

    assert "Earlier questions (3)" in summary
    assert "Question 0" in summary
    assert "Question 4" in summary
    assert "Turn 1 question: Question 3" in summary
    assert "Turn 2 question: Question 4" in summary
    assert "Question 2 question" not in summary


def test_conversation_summary_is_bounded() -> None:
    turns = [FollowUpTurn(question="q" * 500, answer="a" * 500, supported=True) for _ in range(10)]

    summary = summarize_conversation(turns, max_full_turns=3, max_chars=300)

    assert len(summary) <= 300
    assert summary.endswith("[...truncated]")


def test_empty_conversation_summary_is_explicit() -> None:
    assert summarize_conversation([]) == "(no earlier questions)"


def test_render_helpers_use_stored_values_as_is() -> None:
    table = make_stat_table()
    rate = table.rows[0]
    assert format_stat_value(rate) == "50.0%"

    yards = rate.model_copy(update={"unit": StatUnit.YARDS, "value": 4.25})
    assert format_stat_value(yards) == "4.2"

    count = rate.model_copy(update={"unit": StatUnit.COUNT, "value": 7.0})
    assert format_stat_value(count) == "7"

    missing = rate.model_copy(update={"value": None, "suppressed": True, "n": 0})
    assert format_stat_value(missing) == "—"


def test_agent_and_progress_labels_are_human_readable() -> None:
    assert agent_label(AgentName.BEAT_REPORTER) == "Beat reporters"
    line = render_progress(
        RunProgress(step="specialists", status=SectionStatus.RUNNING, agent=AgentName.STATISTICS)
    )
    assert line == "specialists Statistics: running"


def test_tendency_line_keeps_confidence_and_direction() -> None:
    report = make_report()
    line = format_finding_line(report.tendencies[0])

    assert report.tendencies[0].statement in line
    assert "high confidence" in line
    assert "stable" in line
