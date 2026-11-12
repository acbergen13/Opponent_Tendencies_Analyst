"""Streamlit app: select data, build a report, and ask follow-up questions.

The app is deliberately thin. It loads stored data, calls the Stage 9
orchestrator, and renders contracts; it never computes a statistic, retrieves a
source, or validates a citation itself. Tests inject ``report_runner`` and
``follow_up_runner`` into ``st.session_state`` so the UI flow stays offline.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from opponent_tendencies.config import ConfigurationError, get_settings
from opponent_tendencies.data.play_store import SeasonStore
from opponent_tendencies.llm.client import ArcLLMClient
from opponent_tendencies.models import TEAM_DISPLAY_NAMES, Team
from opponent_tendencies.orchestrator import RunProgress
from opponent_tendencies.ui.render import (
    render_failure,
    render_follow_up,
    render_progress,
    render_report,
    render_selection_summary,
)
from opponent_tendencies.ui.runner import (
    DEFAULT_SEASON,
    FollowUpTurn,
    answer_follow_up,
    build_report,
    load_season,
)

st.set_page_config(page_title="Opponent Tendencies Analyst", layout="wide")


def main() -> None:
    st.title("Opponent Tendencies Analyst")
    st.caption(
        "Deterministic NFL statistics with cited reporting and coach context. "
        "The model interprets stored data; Python owns every number and source ID."
    )

    season = int(st.session_state.get("season", DEFAULT_SEASON))
    data_dir = _path_setting("data_dir")
    chroma_dir = _path_setting("chroma_dir")

    try:
        season_data = load_season(season, data_dir)
    except Exception as exc:
        st.error(f"Could not read the {season} season store: {exc}")
        st.stop()
        return

    _render_data_sidebar(season, data_dir)

    if not season_data.teams:
        st.warning(
            f"No completed {season} regular-season games are stored yet. "
            "Use the refresh button in the sidebar after the season starts.",
            icon="📭",
        )
        return

    teams = season_data.teams
    labels = {team.value: f"{TEAM_DISPLAY_NAMES[team]} ({team.value})" for team in teams}
    choice = st.selectbox(
        "Opponent",
        options=[team.value for team in teams],
        format_func=lambda value: labels[value],
        key="opponent_select",
        help="Teams are listed alphabetically by display name.",
    )
    opponent = next(team for team in teams if team.value == choice)

    played_weeks = season_data.played_weeks
    default_weeks = list(played_weeks[-4:]) if played_weeks else []
    weeks = st.multiselect(
        "Regular-season weeks",
        options=list(played_weeks),
        default=default_weeks,
        key="weeks_select",
        help="Only weeks with at least one completed regular-season game are selectable.",
    )

    st.session_state["opponent"] = opponent
    st.session_state["weeks"] = list(weeks)

    render_selection_summary(season_data, opponent, weeks)

    _render_run_controls(season, opponent, weeks, data_dir, chroma_dir)
    _render_stored_report()

    st.divider()
    _render_follow_up()


def _render_data_sidebar(season: int, data_dir: Path | None) -> None:
    with st.sidebar:
        st.header("Data")
        st.caption(f"Season {season}")
        if st.button("Refresh from nflverse", help="Download the latest play-by-play data."):
            with st.spinner("Refreshing nflverse data…"):
                try:
                    from opponent_tendencies.data.nflverse import NflverseClient

                    store = SeasonStore(
                        season,
                        None if data_dir is None else data_dir / "seasons" / f"{season}.json",
                    )
                    result = NflverseClient(season).refresh(store)
                except Exception as exc:
                    st.error(f"Refresh failed: {exc}")
                else:
                    st.success(f"Refresh complete: {result.summary()}.")
                    st.caption(f"Store: {store.path}")
                    st.rerun()
        st.caption(
            "The Chroma index is rebuilt by `scripts/build_index.py`; the app only queries it."
        )


def _render_run_controls(
    season: int,
    opponent: Team,
    weeks: list[int],
    data_dir: Path | None,
    chroma_dir: Path | None,
) -> None:
    st.subheader("Build report")
    in_progress = bool(st.session_state.get("run_in_progress", False))
    disabled = in_progress or not weeks
    if st.button("Build report", type="primary", disabled=disabled):
        st.session_state["run_in_progress"] = True
        st.session_state["last_run"] = None
        st.session_state["last_error"] = None
        st.session_state["report"] = None
        st.session_state["follow_up_turns"] = []
        progress_box = st.empty()
        events: list[str] = []

        def on_progress(event: RunProgress) -> None:
            line = render_progress(event)
            events.append(line)
            progress_box.info("\n\n".join(events[-4:]))

        runner = st.session_state.get("report_runner", build_report)
        try:
            with st.spinner("Running the specialist agents…"):
                run = runner(
                    opponent=opponent,
                    weeks=weeks,
                    season=season,
                    data_dir=data_dir,
                    chroma_dir=chroma_dir,
                    on_progress=on_progress,
                )
        except ConfigurationError as exc:
            st.session_state["last_error"] = str(exc)
            st.session_state["report"] = None
        except Exception as exc:
            st.session_state["last_error"] = f"Report build failed: {exc}"
            st.session_state["report"] = None
        else:
            st.session_state["last_run"] = run
            st.session_state["report"] = run.report
            st.session_state["evidence"] = (
                list(run.report.evidence) if run.report is not None else []
            )
        finally:
            st.session_state["run_in_progress"] = False
            progress_box.empty()

    error = st.session_state.get("last_error")
    if error:
        st.error(str(error))


def _render_stored_report() -> None:
    report = st.session_state.get("report")
    if report is not None:
        st.divider()
        render_report(report)
        run = st.session_state.get("last_run")
        if run is not None:
            cached = " · cache hit" if run.cached else ""
            st.caption(
                f"Run telemetry: {run.llm_calls} LLM calls · "
                f"{run.usage.total_tokens:,} tokens · {run.latency_seconds:.1f}s{cached}"
            )
        return
    run = st.session_state.get("last_run")
    if run is not None:
        st.divider()
        render_failure(run)


def _render_follow_up() -> None:
    report = st.session_state.get("report")
    if report is None:
        st.caption("Build a report to ask follow-up questions grounded in its stored evidence.")
        return

    st.subheader("Follow-up questions")
    st.caption("Answers reuse this report and its stored evidence; no agents are re-run.")
    turns: list[FollowUpTurn] = st.session_state.get("follow_up_turns") or []
    st.session_state["follow_up_turns"] = turns
    render_follow_up(turns)

    with st.form("follow_up_form", clear_on_submit=True):
        question = st.text_input(
            "Ask about the report",
            placeholder="e.g. Which weeks support the third-down finding?",
        )
        submitted = st.form_submit_button("Ask")

    if not submitted:
        return
    if not question.strip():
        st.warning("Enter a question first.")
        return

    follow_up_runner = st.session_state.get("follow_up_runner")
    try:
        with st.spinner("Answering from the stored report…"):
            if follow_up_runner is not None:
                answer = follow_up_runner(report=report, question=question, turns=turns)
            else:
                settings = get_settings()
                answer = answer_follow_up(
                    client=ArcLLMClient.from_settings(settings),
                    report=report,
                    question=question,
                    turns=turns,
                )
    except ConfigurationError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Follow-up failed: {exc}")
        return

    turns.append(
        FollowUpTurn(
            question=question.strip(),
            answer=answer.answer,
            supported=answer.supported,
            evidence_ids=answer.evidence_ids,
            limitations=answer.limitations,
        )
    )
    st.session_state["follow_up_turns"] = turns
    st.rerun()


def _path_setting(name: str) -> Path | None:
    value = st.session_state.get(name)
    if value is None or str(value).strip() == "":
        return None
    return Path(value)


if __name__ == "__main__":
    main()
