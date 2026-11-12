"""Statistics agent (STA): interpret deterministic tables, never compute.

The tables come from :func:`opponent_tendencies.stats.engine.analyze_opponent`
and are passed through unchanged. The model chooses what to emphasize and how
to phrase it; Python owns every number, the evidence catalog, and the final
contract.
"""

from __future__ import annotations

from collections.abc import Sequence

from opponent_tendencies.agents.common import (
    AgentResult,
    LLMCompleter,
    as_agent_response,
    build_output,
    convert_agent_response,
    format_weeks,
    prompt_context,
    run_agent_prompt,
    stat_evidence,
)
from opponent_tendencies.llm.prompts import PROMPT_VERSION
from opponent_tendencies.models import AgentName, Game, Team
from opponent_tendencies.stats.engine import StatsBundle

__all__ = ["run_statistics_agent"]


def run_statistics_agent(
    *,
    client: LLMCompleter,
    games: Sequence[Game],
    weeks: Sequence[int],
    opponent: Team,
    bundle: StatsBundle,
    prompt_version: str = PROMPT_VERSION,
) -> AgentResult:
    """Run STA over one opponent/season/weeks selection.

    ``bundle`` is the deterministic output for the same selection; a mismatch is
    a programming error and raises instead of letting the model describe the
    wrong games.
    """

    ordered_games = _ordered_games(games)
    selected_weeks = tuple(sorted({int(week) for week in weeks}))
    _validate_bundle(bundle, opponent=opponent, games=ordered_games, weeks=selected_weeks)

    evidence = stat_evidence(bundle.tables)
    weak_evidence_ids = {
        table.evidence().evidence_id for table in bundle.tables if table.suppressed
    }
    result = run_agent_prompt(
        client,
        prompt_name="statistics",
        context=prompt_context(
            opponent=opponent,
            season=ordered_games[0].season,
            weeks=selected_weeks,
            games=ordered_games,
            stat_tables=bundle.tables,
            evidence=evidence,
        ),
    )
    response = as_agent_response(result)
    conversion = convert_agent_response(
        response,
        agent=AgentName.STATISTICS,
        catalog=evidence,
        weeks=selected_weeks,
        weak_evidence_ids=weak_evidence_ids,
    )
    output = build_output(
        agent=AgentName.STATISTICS,
        opponent=opponent,
        games=ordered_games,
        weeks=selected_weeks,
        evidence=evidence,
        conversion=conversion,
        limitations=response.limitations,
        prompt_version=prompt_version,
        model=result.model,
        stat_tables=bundle.tables,
    )
    return AgentResult(
        output=output,
        usage=result.usage,
        llm_calls=result.llm_calls,
        latency_seconds=result.latency_seconds,
    )


def _ordered_games(games: Sequence[Game]) -> tuple[Game, ...]:
    if not games:
        raise ValueError("the statistics agent needs at least one selected game")
    ordered = tuple(sorted(games, key=lambda game: (game.week, game.game_date, game.game_id)))
    if len({game.season for game in ordered}) > 1:
        raise ValueError("all selected games must belong to one season")
    return ordered


def _validate_bundle(
    bundle: StatsBundle,
    *,
    opponent: Team,
    games: Sequence[Game],
    weeks: Sequence[int],
) -> None:
    if bundle.opponent is not opponent:
        raise ValueError(f"stats bundle is for {bundle.opponent.value}, not {opponent.value}")
    expected_games = sorted(game.game_id for game in games)
    if list(bundle.game_ids) != expected_games:
        raise ValueError("stats bundle games do not match the selected games")
    played_weeks = tuple(sorted({game.week for game in games}))
    if list(bundle.weeks) != list(played_weeks):
        raise ValueError("stats bundle weeks do not match the selected games")
    if not set(bundle.weeks).issubset(weeks):
        raise ValueError(
            f"stats bundle covers weeks outside the selection: {format_weeks(bundle.weeks)}"
        )
