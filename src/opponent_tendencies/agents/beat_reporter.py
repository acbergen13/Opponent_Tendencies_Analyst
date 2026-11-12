"""Beat-Reporter agent (BRA): cited qualitative reporting and film breakdowns.

Python retrieves and validates the sources *before* the model sees them. If a
retrieval unit does not resolve to stored media or is not an exact stored span,
the agent refuses to call the model and returns an explicit unavailable section
instead of letting ungrounded text into a report.
"""

from __future__ import annotations

from collections.abc import Sequence

from opponent_tendencies.agents.common import (
    AgentResult,
    LLMCompleter,
    as_agent_response,
    build_media_evidence,
    build_output,
    convert_agent_response,
    format_weeks,
    prompt_context,
    run_agent_prompt,
    unavailable_output,
)
from opponent_tendencies.data.index import MediaIndex, RetrievedSource
from opponent_tendencies.data.media_store import MediaData
from opponent_tendencies.llm.prompts import PROMPT_VERSION
from opponent_tendencies.models import AgentName, Game, MediaType, SectionStatus, Team
from opponent_tendencies.validation.checks import check_retrieval

__all__ = ["beat_reporter_query", "run_beat_reporter_agent"]

DEFAULT_RETRIEVAL_K = 8


def beat_reporter_query(opponent: Team, weeks: Sequence[int]) -> str:
    """Deterministic retrieval query for articles and film breakdowns."""

    return (
        f"{opponent.value} offense tendencies formations play calling recent changes "
        f"film breakdown weeks {format_weeks(weeks)}"
    )


def run_beat_reporter_agent(
    *,
    client: LLMCompleter,
    index: MediaIndex,
    media_data: MediaData,
    games: Sequence[Game],
    weeks: Sequence[int],
    opponent: Team,
    k: int = DEFAULT_RETRIEVAL_K,
    prompt_version: str = PROMPT_VERSION,
) -> AgentResult:
    """Retrieve, validate, and interpret beat-reporter reporting."""

    ordered_games = _ordered_games(games)
    selected_weeks = tuple(sorted({int(week) for week in weeks}))
    season = ordered_games[0].season

    sources = index.retrieve(
        beat_reporter_query(opponent, selected_weeks),
        opponent=opponent,
        season=season,
        max_week=max(selected_weeks),
        source_types=(MediaType.ARTICLE, MediaType.FILM_BREAKDOWN),
        kinds=("chunk",),
        k=k,
    )
    validation = check_retrieval(
        sources,
        media_data,
        opponent=opponent,
        weeks=selected_weeks,
        season=season,
    )
    if not validation.ok:
        reason = "; ".join(issue.message for issue in validation.errors)
        return AgentResult(
            output=unavailable_output(
                agent=AgentName.BEAT_REPORTER,
                opponent=opponent,
                games=ordered_games,
                weeks=selected_weeks,
                prompt_version=prompt_version,
                limitation=f"Retrieved beat-reporter sources failed grounding checks: {reason}",
            ),
            warnings=tuple(validation.warnings),
        )

    if not sources:
        return AgentResult(
            output=unavailable_output(
                agent=AgentName.BEAT_REPORTER,
                opponent=opponent,
                games=ordered_games,
                weeks=selected_weeks,
                prompt_version=prompt_version,
                limitation=(
                    f"No stored articles or film breakdowns matched {opponent.value} for "
                    f"weeks {format_weeks(selected_weeks)}; the section has no reporting "
                    "to summarize and is limited to the deterministic statistics."
                ),
                status=SectionStatus.COMPLETED,
            ),
            warnings=tuple(validation.warnings),
        )

    evidence = build_media_evidence(sources)
    result = run_agent_prompt(
        client,
        prompt_name="beat_reporter",
        context=prompt_context(
            opponent=opponent,
            season=season,
            weeks=selected_weeks,
            sources=sources,
            evidence=evidence,
        ),
    )
    response = as_agent_response(result)
    conversion = convert_agent_response(
        response,
        agent=AgentName.BEAT_REPORTER,
        catalog=evidence,
        weeks=selected_weeks,
    )
    limitations = list(response.limitations)
    limitations.extend(_background_limitations(sources, selected_weeks))
    output = build_output(
        agent=AgentName.BEAT_REPORTER,
        opponent=opponent,
        games=ordered_games,
        weeks=selected_weeks,
        evidence=evidence,
        conversion=conversion,
        limitations=limitations,
        prompt_version=prompt_version,
        model=result.model,
    )
    return AgentResult(
        output=output,
        usage=result.usage,
        llm_calls=result.llm_calls,
        latency_seconds=result.latency_seconds,
        warnings=tuple(validation.warnings),
    )


def _ordered_games(games: Sequence[Game]) -> tuple[Game, ...]:
    if not games:
        raise ValueError("the beat-reporter agent needs at least one selected game")
    ordered = tuple(sorted(games, key=lambda game: (game.week, game.game_date, game.game_id)))
    if len({game.season for game in ordered}) > 1:
        raise ValueError("all selected games must belong to one season")
    return ordered


def _background_limitations(
    sources: Sequence[RetrievedSource],
    weeks: Sequence[int],
) -> list[str]:
    last_week = max(weeks)
    background = [
        source.week
        for source in sources
        if source.week is not None and source.week not in weeks and source.week <= last_week
    ]
    if not background:
        return []
    unique = ", ".join(str(week) for week in sorted(set(background)))
    return [
        f"Retrieved source(s) from week(s) {unique} are background context only and "
        "cannot support tendencies for the selected weeks."
    ]
