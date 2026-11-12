"""Streamlit AppTest coverage for selection, reports, failures, and follow-ups."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from streamlit.testing.v1 import AppTest

from opponent_tendencies.models import (
    RunFailureReason,
    RunState,
    RunStatus,
    SectionStatus,
    Team,
)
from opponent_tendencies.orchestrator import ReportRun, RunProgress
from opponent_tendencies.ui.runner import FollowUpAnswer
from tests.unit.factories import NOW, make_report


def _app() -> AppTest:
    return AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)


def _failed_run() -> ReportRun:
    return ReportRun(
        status=RunStatus(
            state=RunState.FAILED,
            failure_reason=RunFailureReason.VALIDATION_FAILED,
            message="The selected opponent has no stored games.",
            started_at=NOW,
            finished_at=NOW,
        )
    )


def test_selection_uses_alphabetical_teams_and_played_weeks_only() -> None:
    app = _app()
    app.run()

    assert not app.exception
    assert app.selectbox[0].value == Team.ARI.value
    assert [int(value) for value in app.multiselect[0].options] == list(range(1, 10))
    assert app.multiselect[0].value == [6, 7, 8, 9]


def test_selection_summary_lists_games_and_scores() -> None:
    app = _app()
    app.run()
    app.selectbox[0].set_value(Team.BUF.value)
    app.multiselect[0].set_value([9])
    app.run()

    assert not app.exception
    assert next(item.value for item in app.subheader) == "Selection"
    assert app.dataframe
    row = app.dataframe[0].value.iloc[0]
    assert row["Week"] == 9
    assert row["Opponent"] == "BUF"
    assert "BUF" in str(row["Score"])


def test_stored_report_renders_tables_findings_citations_and_weak_evidence() -> None:
    report = make_report()
    app = _app()
    app.session_state["report"] = report
    app.run()

    assert not app.exception
    subheaders = [item.value for item in app.subheader]
    assert "Overview" in subheaders
    assert "Deterministic statistics" in subheaders
    assert "Tendencies" in subheaders
    assert any(report.overview == item.value for item in app.markdown)
    assert any("Weak evidence" in item.value for item in app.warning)
    assert any(
        "Evidence:" in item.value and report.evidence[0].evidence_id in item.value
        for item in app.caption
    )
    assert any("All evidence" in item.label for item in app.expander)


def test_degraded_report_shows_partial_banner_and_unavailable_section() -> None:
    report = make_report(degraded=True)
    app = _app()
    app.session_state["report"] = report
    app.run()

    assert any("Partial report" in item.value for item in app.warning)
    assert "Unavailable sections" in [item.value for item in app.subheader]
    assert any(item.value == "Coach quotes unavailable." for item in app.caption)


def test_hard_failure_shows_message_and_no_report() -> None:
    app = _app()
    app.session_state["report_runner"] = lambda **_: _failed_run()
    app.run()
    _click_build(app)

    assert app.session_state["report"] is None
    assert any("no stored games" in item.value for item in app.error)
    assert any("validation_failed" in item.value for item in app.caption)


def test_build_report_stores_session_state_and_emits_progress() -> None:
    report = make_report()
    events: list[RunProgress] = []

    def runner(
        *,
        opponent: Team,
        weeks: list[int],
        on_progress: Callable[[RunProgress], None],
        **_: object,
    ) -> ReportRun:
        on_progress(RunProgress("statistics", SectionStatus.RUNNING, None))
        on_progress(RunProgress("report", SectionStatus.COMPLETED, None))
        events.append(RunProgress("runner", SectionStatus.COMPLETED, None))
        return ReportRun(status=report.status, report=report)

    app = _app()
    app.session_state["report_runner"] = runner
    app.run()
    _click_build(app)

    assert not app.exception
    assert app.session_state["report"] == report
    assert app.session_state["opponent"] is Team.ARI
    assert app.session_state["weeks"] == [6, 7, 8, 9]
    assert app.session_state["evidence"] == report.evidence
    assert events

    # A plain rerun keeps the stored report visible.
    app.run()
    assert app.session_state["report"] == report
    assert "Overview" in [item.value for item in app.subheader]


def test_build_is_disabled_without_weeks() -> None:
    app = _app()
    app.run()
    app.multiselect[0].set_value([])
    app.run()

    assert app.button[0].label == "Build report"
    assert app.button[0].disabled


def test_build_is_disabled_while_a_run_is_in_progress() -> None:
    app = _app()
    app.session_state["run_in_progress"] = True
    app.run()

    assert app.button[0].label == "Build report"
    assert app.button[0].disabled


def test_follow_up_appends_a_validated_turn() -> None:
    report = make_report()
    stat_id = report.evidence[0].evidence_id
    calls: list[dict[str, object]] = []

    def follow_up_runner(*, report: object, question: str, turns: object) -> FollowUpAnswer:
        calls.append({"report": report, "question": question, "turns": turns})
        return FollowUpAnswer(
            answer="The stored table reports a 50% run rate.",
            supported=True,
            evidence_ids=(stat_id,),
            limitations=("Single-game sample.",),
        )

    app = _app()
    app.session_state["report"] = report
    app.session_state["follow_up_runner"] = follow_up_runner
    app.run()

    app.text_input[0].set_value("What is the run rate?")
    _click(app, "Ask")

    assert not app.exception
    turns = app.session_state["follow_up_turns"]
    assert len(turns) == 1
    assert turns[0].question == "What is the run rate?"
    assert turns[0].answer == "The stored table reports a 50% run rate."
    assert turns[0].evidence_ids == (stat_id,)
    assert calls and calls[0]["question"] == "What is the run rate?"
    assert any("Supported by stored evidence" in item.value for item in app.caption)


def test_follow_up_without_a_report_is_not_offered() -> None:
    app = _app()
    app.run()

    assert not app.text_input
    assert any("Build a report" in item.value for item in app.caption)


def _click_build(app: AppTest) -> None:
    _click(app, "Build report")


def _click(app: AppTest, label: str) -> None:
    button = next(item for item in app.button if item.label == label)
    button.click()
    app.run()
