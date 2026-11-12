"""Streamlit rendering for selection, reports, and failures.

Pure formatting helpers live at the top and are unit-tested without a browser;
the ``render_*`` functions only place already-validated contract data on the
page. They never recompute a statistic, sort evidence differently from the
report, or hide a warning.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import streamlit as st

from opponent_tendencies.data.play_store import SeasonData
from opponent_tendencies.models import (
    AgentName,
    EvidenceRef,
    Game,
    Report,
    RunFailureReason,
    StatRow,
    StatTable,
    StatUnit,
    Team,
    Tendency,
    ValidationIssue,
)
from opponent_tendencies.orchestrator import ReportRun, RunProgress

__all__ = [
    "agent_label",
    "format_finding_line",
    "format_stat_value",
    "render_failure",
    "render_follow_up",
    "render_progress",
    "render_report",
    "render_selection_summary",
]

_AGENT_LABELS: dict[AgentName, str] = {
    AgentName.STATISTICS: "Statistics",
    AgentName.BEAT_REPORTER: "Beat reporters",
    AgentName.COACH_QUOTE: "Coach quotes",
    AgentName.SYNTHESIS: "Synthesis",
}


def agent_label(agent: AgentName) -> str:
    return _AGENT_LABELS.get(agent, agent.value.replace("_", " ").title())


def format_stat_value(row: StatRow) -> str:
    """Format a frozen row exactly as stored, without recomputing anything."""

    if row.value is None:
        return "—"
    if row.unit is StatUnit.RATE:
        return f"{row.value:.1%}"
    if row.unit is StatUnit.YARDS:
        return f"{row.value:.1f}"
    return f"{row.value:g}"


def format_finding_line(finding: Tendency) -> str:
    direction = finding.direction.value.replace("_", " ")
    return f"{finding.statement} ({finding.confidence.value.lower()} confidence, {direction})"


def render_progress(progress: RunProgress) -> str:
    """Return a human-readable progress line for one orchestrator event."""

    who = f" {agent_label(progress.agent)}" if progress.agent is not None else ""
    return f"{progress.step}{who}: {progress.status.value}"


def render_selection_summary(
    season_data: SeasonData,
    opponent: Team,
    weeks: Sequence[int],
) -> None:
    """Show the opponent, weeks, games, dates, and scores for the selection."""

    games = [game for game in season_data.games_for_weeks(weeks) if game.involves(opponent)]
    st.subheader("Selection")
    columns = st.columns(4)
    columns[0].metric("Opponent", opponent.value)
    columns[1].metric("Weeks", ", ".join(str(week) for week in sorted(weeks)) or "—")
    columns[2].metric("Games", str(len(games)))
    columns[3].metric("Season", str(season_data.season))

    if not games:
        st.info(
            "No completed regular-season games for this opponent in the selected weeks. "
            "Pick another week or refresh the data.",
            icon="ℹ️",
        )
        return

    rows = [_game_row(game, opponent) for game in games]
    st.dataframe(rows, hide_index=True, width="stretch")


def render_report(report: Report) -> None:
    """Render every report piece, with warnings and weak evidence kept visible."""

    if report.is_partial:
        unavailable = ", ".join(agent_label(agent) for agent in report.unavailable_sections)
        st.warning(
            f"Partial report: {unavailable} unavailable. The statistics and completed "
            "sections are still shown; review the warnings before using it.",
            icon="⚠️",
        )

    st.subheader("Overview")
    st.write(report.overview)
    st.caption(
        f"Report {report.report_id} · {report.season} · weeks "
        f"{', '.join(str(week) for week in report.weeks)} · data {report.data_revision[:12]} "
        f"· prompt {report.prompt_version}"
    )

    if report.weak_evidence:
        topics = "; ".join(item.topic for item in report.weak_evidence)
        st.warning(f"Weak evidence — human review recommended: {topics}", icon="🔎")

    _render_data_used(report)
    _render_tables(report)
    _render_findings(report)
    _render_synthesis(report)
    _render_warnings(report)
    _render_unavailable(report)
    _render_evidence(report)


def render_failure(run: ReportRun) -> None:
    """Show a hard failure without pretending a report exists."""

    status = run.status
    st.error(
        status.message or "The report could not be produced.",
        icon="⛔",
    )
    reason = status.failure_reason
    if reason is not RunFailureReason.NONE:
        st.caption(f"Failure reason: `{reason.value}`")
    if run.warnings:
        st.markdown("**Warnings recorded before the failure**")
        for issue in run.warnings:
            st.markdown(f"- `{issue.code}` — {issue.message}")


def render_follow_up(turns: Iterable[object]) -> None:
    """Render stored question/answer turns as a simple chat transcript."""

    for turn in turns:
        question = getattr(turn, "question", "")
        answer = getattr(turn, "answer", "")
        supported = bool(getattr(turn, "supported", False))
        evidence_ids = tuple(getattr(turn, "evidence_ids", ()))
        limitations = tuple(getattr(turn, "limitations", ()))

        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
            if supported:
                st.caption("Supported by stored evidence: " + ", ".join(evidence_ids))
            else:
                st.caption("Not supported by the stored report evidence.")
            for limitation in limitations:
                st.caption(f"Limitation: {limitation}")


def _game_row(game: Game, opponent: Team) -> dict[str, object]:
    return {
        "Week": game.week,
        "Date": game.game_date.isoformat(),
        "Matchup": f"{game.away_team.value} @ {game.home_team.value}",
        "Opponent": opponent.value,
        "Score": _score(game),
    }


def _score(game: Game) -> str:
    if game.home_score is None or game.away_score is None:
        return "—"
    return f"{game.away_team.value} {game.away_score} – {game.home_team.value} {game.home_score}"


def _render_data_used(report: Report) -> None:
    usage = report.data_used
    st.subheader("Data used")
    columns = st.columns(5)
    columns[0].metric("Games", usage.games)
    columns[1].metric("Plays", usage.plays)
    columns[2].metric("Media items", usage.media_items)
    columns[3].metric("Quotes", usage.quotes)
    columns[4].metric("Sources", len(usage.source_ids))
    if usage.source_ids:
        st.caption("Sources: " + ", ".join(usage.source_ids))


def _render_tables(report: Report) -> None:
    if not report.stat_tables:
        return
    st.subheader("Deterministic statistics")
    st.caption("Computed by Python and passed through unchanged; the model only interprets them.")
    for table in report.stat_tables:
        title = table.title
        if table.dimension_value:
            title = f"{title} — {table.dimension_value}"
        with st.expander(f"{title} (n={table.n_plays})", expanded=False):
            st.dataframe(_table_rows(table), hide_index=True, width="stretch")
            if table.suppressed:
                st.warning("Suppressed: below the minimum sample size.", icon="⚠️")
            for note in table.notes:
                st.caption(note)


def _table_rows(table: StatTable) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in table.rows:
        rows.append(
            {
                "Label": row.label,
                "Value": format_stat_value(row),
                "n": row.n,
                "Notes": "; ".join(row.notes) or ("suppressed" if row.suppressed else ""),
            }
        )
    return rows


def _render_findings(report: Report) -> None:
    evidence = {item.evidence_id: item for item in report.evidence}
    if report.tendencies:
        st.subheader("Tendencies")
        for tendency in report.tendencies:
            st.markdown(f"- {format_finding_line(tendency)}")
            _render_citations(tendency.evidence_ids, evidence)
    if report.observations:
        st.subheader("Reporting and coach context")
        for observation in report.observations:
            context = " (coach context, not proof)" if observation.treat_as_context else ""
            st.markdown(
                f"- {observation.statement} "
                f"({observation.confidence.value.lower()} confidence, "
                f"{observation.category.value.replace('_', ' ')}{context})"
            )
            _render_citations(observation.evidence_ids, evidence)


def _render_citations(ids: Sequence[str], evidence: dict[str, EvidenceRef]) -> None:
    if not ids:
        st.caption("No evidence attached.")
        return
    labels: list[str] = []
    for evidence_id in ids:
        item = evidence.get(evidence_id)
        if item is None:
            labels.append(f"{evidence_id} (missing from stored evidence)")
            continue
        label = f"{item.label} [{item.kind.value}] `{item.evidence_id}`"
        if item.source_url is not None:
            label = f"{label} — {item.source_url}"
        labels.append(label)
    st.caption("Evidence: " + " | ".join(labels))


def _render_synthesis(report: Report) -> None:
    if report.agreements:
        st.subheader("Where sources agree")
        for agreement in report.agreements:
            st.markdown(
                f"- **{agreement.topic}** — {agreement.statement} "
                f"({', '.join(agent_label(agent) for agent in agreement.agents)})"
            )
    if report.conflicts:
        st.subheader("Conflicts")
        for conflict in report.conflicts:
            st.markdown(f"- **{conflict.topic}** — {conflict.description}")
            for position in conflict.positions:
                st.caption(f"{agent_label(position.agent)}: {position.statement}")
            if conflict.resolution:
                st.caption(f"Resolution: {conflict.resolution}")
    if report.gaps:
        st.subheader("Gaps")
        for gap in report.gaps:
            st.markdown(
                f"- **{gap.topic}** — {gap.description} (`{gap.reason.value.replace('_', ' ')}`)"
            )


def _render_warnings(report: Report) -> None:
    if not report.warnings:
        return
    with st.expander(f"Warnings ({len(report.warnings)})", expanded=True):
        for issue in report.warnings:
            st.markdown(_issue_line(issue))


def _issue_line(issue: ValidationIssue) -> str:
    review = " · human review" if issue.requires_human_review else ""
    return f"- `{issue.code}` ({issue.severity.value}) — {issue.message}{review}"


def _render_unavailable(report: Report) -> None:
    unavailable = [output for output in report.agent_outputs if not output.is_available]
    if not unavailable:
        return
    st.subheader("Unavailable sections")
    for output in unavailable:
        st.markdown(f"- **{agent_label(output.agent)}** (`{output.status.value}`)")
        for limitation in output.limitations:
            st.caption(limitation)


def _render_evidence(report: Report) -> None:
    if not report.evidence:
        return
    with st.expander(f"All evidence ({len(report.evidence)})", expanded=False):
        for item in report.evidence:
            line = f"- `{item.evidence_id}` **{item.label}** ({item.kind.value})"
            if item.source_url is not None:
                line = f"{line} — {item.source_url}"
            st.markdown(line)
