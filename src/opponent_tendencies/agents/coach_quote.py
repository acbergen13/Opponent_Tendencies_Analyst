"""Coach-Quote agent (CQA): verbatim post-game quotes as context.

The index returns exact quote spans for coach-quote media, so every quote the
model sees and every quote ID it may cite resolve to stored text. Coach
statements are context, not proof: Python marks explanations and adjustments
``treat_as_context`` regardless of what the model returned.
"""

from __future__ import annotations

from collections.abc import Sequence

from opponent_tendencies.agents.common import (
    AgentResult,
    LLMCompleter,
    as_agent_response,
    build_output,
    build_quote_evidence,
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

__all__ = ["coach_quote_query", "run_coach_quote_agent"]

DEFAULT_RETRIEVAL_K = 8


def coach_quote_query(opponent: Team, weeks: Sequence[int]) -> str:
    """Deterministic retrieval query for post-game coach quotes."""

    return (
        f"{opponent.value} head coach offensive coordinator post-game press conference "
        f"quotes adjustments offense weeks {format_weeks(weeks)}"
    )


def run_coach_quote_agent(
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
    """Retrieve, validate, and interpret verbatim coach quotes."""

    ordered_games = _ordered_games(games)
    selected_weeks = tuple(sorted({int(week) for week in weeks}))
    season = ordered_games[0].season

    sources = index.retrieve(
        coach_quote_query(opponent, selected_weeks),
        opponent=opponent,
        season=season,
        max_week=max(selected_weeks),
        source_types=(MediaType.COACH_QUOTE,),
        kinds=("quote",),
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
                agent=AgentName.COACH_QUOTE,
                opponent=opponent,
                games=ordered_games,
                weeks=selected_weeks,
                prompt_version=prompt_version,
                limitation=f"Retrieved coach quotes failed grounding checks: {reason}",
            ),
            warnings=tuple(validation.warnings),
        )

    if not sources:
        return AgentResult(
            output=unavailable_output(
                agent=AgentName.COACH_QUOTE,
                opponent=opponent,
                games=ordered_games,
                weeks=selected_weeks,
                prompt_version=prompt_version,
                limitation=(
                    f"No stored post-game coach quotes matched {opponent.value} for "
                    f"weeks {format_weeks(selected_weeks)}; the section has no quotes to "
                    "summarize."
                ),
                status=SectionStatus.COMPLETED,
            ),
            warnings=tuple(validation.warnings),
        )

    evidence = build_quote_evidence(sources)
    result = run_agent_prompt(
        client,
        prompt_name="coach_quote",
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
        agent=AgentName.COACH_QUOTE,
        catalog=evidence,
        weeks=selected_weeks,
    )
    limitations = list(response.limitations)
    limitations.extend(_background_limitations(sources, selected_weeks))
    output = build_output(
        agent=AgentName.COACH_QUOTE,
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
        raise ValueError("the coach-quote agent needs at least one selected game")
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
        f"Retrieved quote(s) from week(s) {unique} are background context only and "
        "cannot support tendencies for the selected weeks."
    ]
