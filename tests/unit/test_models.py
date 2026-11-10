"""Round-trip and validation tests for the frozen contract layer."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from opponent_tendencies.models import (
    AgentName,
    AgentOutput,
    Confidence,
    EvidenceKind,
    EvidenceRef,
    IssueSeverity,
    Observation,
    ObservationCategory,
    RunFailureReason,
    RunState,
    RunStatus,
    SectionStatus,
    SplitDimension,
    StatMetric,
    StatRow,
    StatTable,
    StatUnit,
    Team,
    Tendency,
    TendencyCategory,
    TendencyDirection,
    ValidationIssue,
    ValidationResult,
    team_display_name,
    teams_alphabetical,
)

from .factories import (
    GAME_ID,
    NOW,
    PLAY_ID,
    WEEK,
    make_bra_output,
    make_cqa_output,
    make_evidence,
    make_game,
    make_media,
    make_play,
    make_quote,
    make_quote_media,
    make_report,
    make_sta_output,
    make_stat_table,
)

ROUND_TRIP_CASES: dict[str, Callable[[], BaseModel]] = {
    "game": make_game,
    "play": make_play,
    "media": make_media,
    "quote": lambda: make_quote(make_quote_media()),
    "play-evidence": lambda: make_evidence()[0],
    "media-evidence": lambda: make_evidence()[1],
    "quote-evidence": lambda: make_evidence()[2],
    "stat-evidence": lambda: make_evidence()[3],
    "stat-table": make_stat_table,
    "tendency": lambda: make_sta_output()[1],
    "observation": lambda: make_bra_output()[1],
    "sta-output": lambda: make_sta_output()[0],
    "bra-output": lambda: make_bra_output()[0],
    "cqa-output": lambda: make_cqa_output()[0],
    "report": make_report,
    "degraded-report": lambda: make_report(degraded=True),
    "validation-result": lambda: ValidationResult(
        ok=True,
        warnings=[
            ValidationIssue(
                code="SMALL_SAMPLE",
                message="Only one game was selected.",
                severity=IssueSeverity.WARNING,
                requires_human_review=True,
            )
        ],
        checked={"games": 1, "plays": 10},
    ),
    "run-status": lambda: make_report().status,
}


@pytest.mark.parametrize("factory", ROUND_TRIP_CASES.values(), ids=ROUND_TRIP_CASES.keys())
def test_contracts_round_trip_through_json(factory: Callable[[], BaseModel]) -> None:
    model = factory()

    restored = type(model).model_validate(model.model_dump(mode="json"))
    assert restored == model

    restored_from_text = type(model).model_validate_json(model.model_dump_json())
    assert restored_from_text == model


def test_report_round_trip_preserves_generated_ids() -> None:
    report = make_report()

    restored = type(report).model_validate_json(report.model_dump_json())

    assert restored.report_id == report.report_id
    assert restored.status.state is RunState.COMPLETED
    assert restored.human_review_required is True


def test_agent_output_schema_requires_requirement_25_fields() -> None:
    schema = AgentOutput.model_json_schema()
    required = set(schema["required"])

    assert {
        "opponent",
        "games",
        "weeks",
        "tendencies",
        "observations",
        "evidence",
        "limitations",
    } <= required


def test_agent_output_rejects_opponent_mismatch() -> None:
    data: dict[str, Any] = make_sta_output()[0].model_dump()
    data["opponent"] = Team.NYJ

    with pytest.raises(ValidationError, match="every game must involve"):
        AgentOutput.model_validate(data)


def test_agent_output_rejects_postseason_games() -> None:
    data: dict[str, Any] = make_sta_output()[0].model_dump()
    data["games"][0]["season_type"] = "POST"

    with pytest.raises(ValidationError, match="regular-season"):
        AgentOutput.model_validate(data)


def test_report_rejects_postseason_games() -> None:
    report = make_report()
    data: dict[str, Any] = report.model_dump()
    data["games"][0]["season_type"] = "POST"

    with pytest.raises(ValidationError, match="regular-season"):
        type(report).model_validate(data)


def test_agent_output_rejects_unresolved_evidence_reference() -> None:
    output, tendency, _play_ref, stat_ref, table = make_sta_output()
    data: dict[str, Any] = output.model_dump()
    data["evidence"] = [stat_ref.model_dump()]

    with pytest.raises(ValidationError, match="cite evidence not present"):
        AgentOutput.model_validate(data)

    # The untouched output remains valid and every citation resolves.
    assert tendency.evidence_ids
    assert table.stat_table_id == stat_ref.stat_table_id


def test_agent_output_requires_limitations_for_unavailable_section() -> None:
    data: dict[str, Any] = make_sta_output()[0].model_dump()
    data.update(
        status=SectionStatus.UNAVAILABLE,
        tendencies=[],
        observations=[],
        evidence=[],
        limitations=[],
        stat_tables=[],
    )

    with pytest.raises(ValidationError, match="must explain itself"):
        AgentOutput.model_validate(data)

    data["limitations"] = ["The statistics tool failed."]
    restored = AgentOutput.model_validate(data)
    assert restored.status is SectionStatus.UNAVAILABLE
    assert restored.is_available is False


def test_agent_output_rejects_findings_on_unavailable_section() -> None:
    data: dict[str, Any] = make_sta_output()[0].model_dump()
    data.update(
        status=SectionStatus.TIMED_OUT,
        limitations=["The statistics agent timed out."],
    )

    with pytest.raises(ValidationError, match="must not report findings"):
        AgentOutput.model_validate(data)


def test_empty_completed_output_must_explain_itself() -> None:
    data: dict[str, Any] = make_sta_output()[0].model_dump()
    data.update(
        tendencies=[],
        observations=[],
        evidence=[],
        limitations=[],
        stat_tables=[],
    )

    with pytest.raises(ValidationError, match="must carry an explicit limitation"):
        AgentOutput.model_validate(data)


def test_agent_output_rejects_wrong_evidence_kind_for_agent() -> None:
    output, _, _, _, _ = make_sta_output()
    media_ref = make_evidence()[1]
    data: dict[str, Any] = output.model_dump()
    data["evidence"] = sorted([*output.evidence, media_ref], key=lambda item: item.evidence_id)

    with pytest.raises(ValidationError, match="cannot cite retrieved media"):
        AgentOutput.model_validate(data)


def test_finding_requires_evidence_and_sorted_weeks() -> None:
    with pytest.raises(ValidationError):
        Tendency(
            statement="Buffalo runs on early downs.",
            category=TendencyCategory.RUN_PASS,
            direction=TendencyDirection.INCREASE,
            evidence_ids=[],
            weeks=[WEEK],
            confidence=Confidence.MEDIUM,
        )

    with pytest.raises(ValidationError, match="weeks must be sorted"):
        Observation(
            statement="Reporters expect more play action.",
            category=ObservationCategory.SCHEME,
            evidence_ids=[make_evidence()[1].evidence_id],
            weeks=[10, 9],
            confidence=Confidence.MEDIUM,
        )


def test_coach_statement_cannot_be_a_tendency() -> None:
    with pytest.raises(ValidationError, match="context, not a tendency"):
        Tendency(
            statement="The coach said Buffalo will run more.",
            category=TendencyCategory.RUN_PASS,
            direction=TendencyDirection.INCREASE,
            evidence_ids=[make_evidence()[2].evidence_id],
            weeks=[WEEK],
            confidence=Confidence.LOW,
            treat_as_context=True,
        )


def test_stat_table_flags_small_samples() -> None:
    with pytest.raises(ValidationError, match="suppressed must be True"):
        StatTable(
            metric=StatMetric.RUN_RATE,
            title="Run rate",
            dimension=SplitDimension.OVERALL,
            rows=[
                StatRow(
                    label="All plays",
                    value=0.5,
                    n=4,
                    numerator=2,
                    denominator=4,
                    unit=StatUnit.RATE,
                    suppressed=False,
                )
            ],
            n_plays=4,
            game_ids=[GAME_ID],
            weeks=[WEEK],
            min_sample=5,
        )


def test_stat_table_requires_sorted_weeks() -> None:
    with pytest.raises(ValidationError, match="weeks must be sorted"):
        StatTable(
            metric=StatMetric.RUN_RATE,
            title="Run rate",
            dimension=SplitDimension.OVERALL,
            rows=[],
            n_plays=0,
            game_ids=[],
            weeks=[10, 9],
            min_sample=5,
            suppressed=True,
        )


def test_stat_row_cannot_report_a_value_without_plays() -> None:
    with pytest.raises(ValidationError, match="must not report a value"):
        StatRow(
            label="Empty",
            value=0.5,
            n=0,
            unit=StatUnit.RATE,
            suppressed=True,
        )


def test_evidence_requires_matching_locator() -> None:
    with pytest.raises(ValidationError, match="requires game_id and play_id"):
        EvidenceRef(kind=EvidenceKind.PLAY, label="Missing play locator")

    with pytest.raises(ValidationError, match="requires media_id"):
        EvidenceRef(kind=EvidenceKind.MEDIA, label="Missing media locator")

    with pytest.raises(ValidationError, match="does not match its locator"):
        EvidenceRef(
            kind=EvidenceKind.PLAY,
            game_id=GAME_ID,
            play_id=PLAY_ID,
            label="Forged id",
            evidence_id="ev:play:forged:1",
        )


def test_quote_must_still_be_verbatim() -> None:
    media = make_quote_media()
    quote = make_quote(media)

    assert quote.is_verbatim_in(media)

    edited = media.model_copy(update={"body": "The quote was rewritten."})
    assert quote.is_verbatim_in(edited) is False


def test_media_content_hash_is_verified() -> None:
    media = make_media()
    edited = media.model_copy(update={"body": "Different body", "content_hash": media.content_hash})

    with pytest.raises(ValidationError, match="content_hash does not match"):
        type(media).model_validate(edited.model_dump())


def test_report_rejects_hard_failure_state() -> None:
    report = make_report()
    data: dict[str, Any] = report.model_dump()
    data["status"] = RunStatus(
        state=RunState.FAILED,
        failure_reason=RunFailureReason.DEADLINE_EXCEEDED,
        message="The run exceeded its deadline.",
        progress=0.5,
        started_at=datetime(2026, 11, 10, 11, 55, tzinfo=UTC),
        finished_at=NOW,
    ).model_dump()

    with pytest.raises(ValidationError, match="only produced for completed or degraded"):
        type(report).model_validate(data)


def test_report_requires_all_three_specialist_outputs() -> None:
    report = make_report()
    data: dict[str, Any] = report.model_dump()
    data["agent_outputs"] = data["agent_outputs"][:2]

    with pytest.raises(ValidationError):
        type(report).model_validate(data)


def test_report_requires_every_cited_evidence_item() -> None:
    report = make_report()
    data: dict[str, Any] = report.model_dump()
    data["evidence"] = data["evidence"][1:]

    with pytest.raises(ValidationError, match="cite evidence not present"):
        type(report).model_validate(data)


def test_degraded_report_marks_unavailable_section() -> None:
    report = make_report(degraded=True)

    assert report.is_partial is True
    assert report.unavailable_sections == (AgentName.COACH_QUOTE,)
    assert report.status.state is RunState.DEGRADED
    assert report.status.agent_status[AgentName.COACH_QUOTE] is SectionStatus.UNAVAILABLE


def test_completed_run_status_cannot_leave_a_section_unavailable() -> None:
    report = make_report(degraded=True)
    data: dict[str, Any] = report.model_dump()
    data["status"] = make_report().status.model_dump()

    with pytest.raises(ValidationError, match="does not match its output status"):
        type(report).model_validate(data)


def test_validation_result_keeps_ok_in_sync_with_errors() -> None:
    error = ValidationIssue(
        code="SOURCE_MISSING",
        message="Play does not exist.",
        severity=IssueSeverity.ERROR,
    )

    with pytest.raises(ValidationError, match="ok must be true"):
        ValidationResult(ok=True, errors=[error])

    with pytest.raises(ValidationError, match="ok must be true"):
        ValidationResult(ok=False)

    with pytest.raises(ValidationError, match="errors must all have error severity"):
        ValidationResult(
            ok=False,
            errors=[
                ValidationIssue(
                    code="BAD",
                    message="Wrong severity.",
                    severity=IssueSeverity.WARNING,
                )
            ],
        )


def test_validation_result_merge_sums_counts_and_flags_review() -> None:
    first = ValidationResult(
        ok=True,
        warnings=[
            ValidationIssue(
                code="SMALL_SAMPLE",
                message="Five plays.",
                severity=IssueSeverity.WARNING,
                requires_human_review=True,
            )
        ],
        checked={"plays": 5},
    )
    second = ValidationResult(ok=True, checked={"plays": 3, "media": 2})

    merged = ValidationResult.merge(first, second)

    assert merged.ok is True
    assert merged.checked == {"plays": 8, "media": 2}
    assert merged.requires_human_review is True


def test_failed_run_status_explains_itself() -> None:
    with pytest.raises(ValidationError, match="must record why it failed"):
        RunStatus(
            state=RunState.FAILED,
            finished_at=NOW,
        )

    with pytest.raises(ValidationError, match="must explain the failure"):
        RunStatus(
            state=RunState.FAILED,
            failure_reason=RunFailureReason.DEADLINE_EXCEEDED,
            finished_at=NOW,
        )


def test_running_run_status_cannot_be_finished() -> None:
    with pytest.raises(ValidationError, match="cannot be finished"):
        RunStatus(
            state=RunState.RUNNING,
            started_at=NOW,
            finished_at=NOW,
        )


def test_teams_are_alphabetical_by_display_name() -> None:
    ordered = teams_alphabetical()

    assert len(ordered) == 32
    assert team_display_name(ordered[0]) == "Arizona Cardinals"
    assert team_display_name(ordered[-1]) == "Washington Commanders"
    assert [team_display_name(team) for team in ordered] == sorted(
        team_display_name(team) for team in ordered
    )


def test_game_helpers_resolve_the_opponent() -> None:
    game = make_game()

    assert game.involves(Team.BUF)
    assert game.opponent_of(Team.BUF) is Team.KC
    assert game.opponent_of(Team.KC) is Team.BUF
    assert game.is_regular_season

    with pytest.raises(ValueError):
        game.opponent_of(Team.NYJ)
