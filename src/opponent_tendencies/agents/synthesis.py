"""Synthesis agent (SYN): merge validated findings into report-level pieces.

The model sees only already-validated specialist outputs and the deterministic
tables. Python then enforces the same rules it enforced for the specialists:
citations must exist, weeks must be selected, agreement agents must be
specialists, conflict evidence must cover every position, and an unavailable
agent always yields an explicit gap. Nothing the model returns can introduce a
new fact or a new source.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from opponent_tendencies.agents.common import (
    AgentConversion,
    LLMCompleter,
    build_stat_evidence,
    convert_agent_response,
    format_weeks,
    run_agent_prompt,
)
from opponent_tendencies.llm.client import TokenUsage
from opponent_tendencies.llm.prompts import (
    PROMPT_VERSION,
    render_agent_findings,
    render_evidence_catalog,
    render_stat_tables,
)
from opponent_tendencies.llm.schemas import (
    AgentResponse,
    DraftAgreement,
    DraftConflict,
    DraftGap,
    DraftWeakEvidence,
    SynthesisResponse,
)
from opponent_tendencies.models import (
    SPECIALIST_AGENTS,
    AgentFinding,
    AgentName,
    AgentOutput,
    Agreement,
    Conflict,
    ConflictPosition,
    EvidenceRef,
    Game,
    Gap,
    GapReason,
    IssueSeverity,
    Observation,
    StatTable,
    Team,
    Tendency,
    ValidationIssue,
    WeakEvidence,
    WeakEvidenceReason,
)
from opponent_tendencies.validation.checks import CheckCode

__all__ = ["SYNTHESIS_MAX_TOKENS", "SynthesisResult", "run_synthesis_agent"]

# Synthesis restates every validated finding plus agreements, conflicts, gaps,
# and weak-evidence flags, so it needs more room than a specialist. Above the
# endpoint's buffered cap the client switches to a streamed completion.
SYNTHESIS_MAX_TOKENS = 16384


@dataclass(frozen=True)
class SynthesisResult:
    """Report-level content produced by SYN, before the report is stamped."""

    overview: str
    tendencies: tuple[Tendency, ...]
    observations: tuple[Observation, ...]
    agreements: tuple[Agreement, ...]
    conflicts: tuple[Conflict, ...]
    gaps: tuple[Gap, ...]
    weak_evidence: tuple[WeakEvidence, ...]
    evidence: tuple[EvidenceRef, ...]
    dropped: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    llm_calls: int = 0
    latency_seconds: float = 0.0
    model: str | None = None
    warnings: tuple[ValidationIssue, ...] = ()


def run_synthesis_agent(
    *,
    client: LLMCompleter,
    opponent: Team,
    season: int,
    weeks: Sequence[int],
    games: Sequence[Game],
    agent_outputs: Sequence[AgentOutput],
    stat_tables: Sequence[StatTable],
    prompt_version: str = PROMPT_VERSION,
) -> SynthesisResult:
    """Merge three specialist outputs into report-level, validated pieces."""

    selected_weeks = tuple(sorted({int(week) for week in weeks}))
    _validate_inputs(
        opponent=opponent,
        weeks=selected_weeks,
        games=games,
        agent_outputs=agent_outputs,
    )
    evidence = _evidence_catalog(agent_outputs, stat_tables)
    result = run_agent_prompt(
        client,
        prompt_name="synthesis",
        context={
            "opponent": opponent.value,
            "season": str(season),
            "weeks": format_weeks(selected_weeks),
            "agent_findings": render_agent_findings(agent_outputs),
            "stat_tables": render_stat_tables(stat_tables),
            "available_evidence": render_evidence_catalog(evidence),
        },
        max_tokens=SYNTHESIS_MAX_TOKENS,
    )
    if not isinstance(result.parsed, SynthesisResponse):
        raise TypeError(f"expected SynthesisResponse, got {type(result.parsed).__name__}")
    response = result.parsed

    conversion = convert_agent_response(
        # AGENT response and SYN response share the finding draft shapes.
        _as_agent_response(response),
        agent=AgentName.SYNTHESIS,
        catalog=evidence,
        weeks=selected_weeks,
    )
    allowed = {item.evidence_id for item in evidence}
    dropped = conversion.dropped

    agreements = _agreements(response.agreements, allowed, selected_weeks)
    conflicts, conflict_dropped = _conflicts(response.conflicts, allowed, selected_weeks)
    gaps = _gaps(response.gaps, selected_weeks)
    weak = _weak_evidence(response.weak_evidence, allowed, selected_weeks)
    dropped += conflict_dropped
    dropped += len(response.agreements) - len(agreements)
    dropped += len(response.gaps) - len(gaps)
    dropped += len(response.weak_evidence) - len(weak)

    unavailable = _unavailable_gaps(agent_outputs, selected_weeks)
    gaps = _dedupe([*gaps, *unavailable], key=lambda gap: gap.gap_id)
    weak = _dedupe(
        [*weak, *_finding_weak_evidence(conversion, agent_outputs)],
        key=lambda item: item.weak_evidence_id,
    )

    warnings: list[ValidationIssue] = []
    if dropped:
        warnings.append(
            _warning(
                CheckCode.SYNTHESIS_UNGROUNDED_CLAIM,
                f"Synthesis dropped {dropped} item(s) that cited evidence, weeks, or "
                "agents outside the validated inputs.",
            )
        )

    return SynthesisResult(
        overview=response.overview,
        tendencies=tuple(conversion.tendencies),
        observations=tuple(conversion.observations),
        agreements=tuple(sorted(agreements, key=lambda item: item.agreement_id)),
        conflicts=tuple(sorted(conflicts, key=lambda item: item.conflict_id)),
        gaps=tuple(sorted(gaps, key=lambda item: item.gap_id)),
        weak_evidence=tuple(sorted(weak, key=lambda item: item.weak_evidence_id)),
        evidence=evidence,
        dropped=dropped,
        usage=result.usage,
        llm_calls=result.llm_calls,
        latency_seconds=result.latency_seconds,
        model=result.model,
        warnings=tuple(warnings),
    )


def _as_agent_response(response: SynthesisResponse) -> AgentResponse:
    """Narrow SYN drafts to the shared finding-only wire model."""

    return AgentResponse(
        tendencies=response.tendencies,
        observations=response.observations,
        limitations=[],
    )


def _validate_inputs(
    *,
    opponent: Team,
    weeks: Sequence[int],
    games: Sequence[Game],
    agent_outputs: Sequence[AgentOutput],
) -> None:
    if not weeks:
        raise ValueError("synthesis needs at least one selected week")
    if not games:
        raise ValueError("synthesis needs at least one selected game")
    if len(agent_outputs) != len(SPECIALIST_AGENTS):
        raise ValueError("synthesis needs exactly one STA, BRA, and CQA output")
    if {output.agent for output in agent_outputs} != set(SPECIALIST_AGENTS):
        raise ValueError("synthesis inputs must be STA, BRA, and CQA exactly once")
    for output in agent_outputs:
        if output.opponent is not opponent:
            raise ValueError("every agent output must target the synthesis opponent")
        if list(output.weeks) != list(weeks):
            raise ValueError("every agent output must cover the selected weeks")
        if list(output.games) != list(games):
            raise ValueError("every agent output must use the selected games")


def _evidence_catalog(
    agent_outputs: Sequence[AgentOutput],
    stat_tables: Sequence[StatTable],
) -> tuple[EvidenceRef, ...]:
    by_id = {item.evidence_id: item for item in build_stat_evidence(stat_tables)}
    for output in agent_outputs:
        for item in output.evidence:
            by_id[item.evidence_id] = item
    return tuple(by_id[evidence_id] for evidence_id in sorted(by_id))


def _valid_evidence(ids: Iterable[str], allowed: set[str]) -> list[str] | None:
    unique = sorted({str(evidence_id) for evidence_id in ids})
    if not unique or not set(unique).issubset(allowed):
        return None
    return unique


def _valid_weeks(weeks: Iterable[int], selected: Sequence[int]) -> list[int] | None:
    unique = sorted({int(week) for week in weeks})
    if not unique or not set(unique).issubset(selected):
        return None
    return unique


def _agreements(
    drafts: Sequence[DraftAgreement],
    allowed: set[str],
    weeks: Sequence[int],
) -> list[Agreement]:
    agreements: list[Agreement] = []
    for draft in drafts:
        agents = sorted({agent for agent in draft.agents if agent in SPECIALIST_AGENTS})
        evidence_ids = _valid_evidence(draft.evidence_ids, allowed)
        selected = _valid_weeks(draft.weeks, weeks)
        if len(agents) < 2 or evidence_ids is None or selected is None:
            continue
        agreements.append(
            Agreement(
                topic=draft.topic,
                statement=draft.statement,
                agents=agents,
                evidence_ids=evidence_ids,
                weeks=selected,
            )
        )
    return agreements


def _conflicts(
    drafts: Sequence[DraftConflict],
    allowed: set[str],
    weeks: Sequence[int],
) -> tuple[list[Conflict], int]:
    conflicts: list[Conflict] = []
    dropped = 0
    for draft in drafts:
        positions: list[ConflictPosition] = []
        seen_agents: set[AgentName] = set()
        for position in draft.positions:
            if position.agent not in SPECIALIST_AGENTS or position.agent in seen_agents:
                dropped += 1
                continue
            evidence_ids = _valid_evidence(position.evidence_ids, allowed)
            if evidence_ids is None:
                dropped += 1
                continue
            seen_agents.add(position.agent)
            positions.append(
                ConflictPosition(
                    agent=position.agent,
                    statement=position.statement,
                    evidence_ids=evidence_ids,
                )
            )
        selected = _valid_weeks(draft.weeks, weeks)
        model_evidence = _valid_evidence(draft.evidence_ids, allowed) or []
        if len(positions) < 2 or selected is None:
            dropped += 1
            continue
        evidence_ids = sorted(
            set(model_evidence).union(
                evidence_id for position in positions for evidence_id in position.evidence_ids
            )
        )
        conflicts.append(
            Conflict(
                topic=draft.topic,
                description=draft.description,
                positions=positions,
                evidence_ids=evidence_ids,
                weeks=selected,
                resolution=draft.resolution,
            )
        )
    return conflicts, dropped


def _gaps(drafts: Sequence[DraftGap], weeks: Sequence[int]) -> list[Gap]:
    gaps: list[Gap] = []
    for draft in drafts:
        selected = _valid_weeks(draft.weeks, weeks)
        if selected is None:
            continue
        agents = sorted({agent for agent in draft.agents if agent in SPECIALIST_AGENTS})
        gaps.append(
            Gap(
                topic=draft.topic,
                description=draft.description,
                reason=draft.reason,
                agents=agents,
                weeks=selected,
            )
        )
    return gaps


def _weak_evidence(
    drafts: Sequence[DraftWeakEvidence],
    allowed: set[str],
    weeks: Sequence[int],
) -> list[WeakEvidence]:
    items: list[WeakEvidence] = []
    for draft in drafts:
        selected = _valid_weeks(draft.weeks, weeks)
        if selected is None:
            continue
        evidence_ids = sorted({item for item in draft.evidence_ids if item in allowed})
        if draft.evidence_ids and not evidence_ids:
            continue
        agent = draft.agent if draft.agent in SPECIALIST_AGENTS else None
        items.append(
            WeakEvidence(
                topic=draft.topic,
                description=draft.description,
                reason=draft.reason,
                evidence_ids=evidence_ids,
                weeks=selected,
                agent=agent,
            )
        )
    return items


def _unavailable_gaps(
    agent_outputs: Sequence[AgentOutput],
    weeks: Sequence[int],
) -> list[Gap]:
    gaps: list[Gap] = []
    for output in agent_outputs:
        if output.is_available:
            continue
        gaps.append(
            Gap(
                topic=f"{output.agent.value} section unavailable",
                description=output.limitations[0],
                reason=GapReason.AGENT_UNAVAILABLE,
                agents=[output.agent],
                weeks=list(weeks),
            )
        )
    return gaps


def _finding_weak_evidence(
    conversion: AgentConversion,
    agent_outputs: Sequence[AgentOutput],
) -> list[WeakEvidence]:
    """Turn uncontested weak-confidence findings into explicit flags.

    The synthesis model is told to flag weak evidence, but Python guarantees it:
    a final finding that any agent marked weak always reaches human review.
    """

    source_agents: dict[str, AgentName] = {}
    for output in agent_outputs:
        for finding in [*output.tendencies, *output.observations]:
            source_agents[finding.finding_id] = output.agent

    items: list[WeakEvidence] = []
    for finding, kind in _iter_findings(conversion):
        if not finding.weak_evidence:
            continue
        items.append(
            WeakEvidence(
                topic=f"Weak {kind}",
                description=finding.statement,
                reason=WeakEvidenceReason.LOW_CONFIDENCE,
                evidence_ids=list(finding.evidence_ids),
                weeks=list(finding.weeks),
                agent=source_agents.get(finding.finding_id),
            )
        )
    return items


def _iter_findings(conversion: AgentConversion) -> list[tuple[AgentFinding, str]]:
    found: list[tuple[AgentFinding, str]] = []
    for tendency in conversion.tendencies:
        found.append((tendency, "tendency"))
    for observation in conversion.observations:
        found.append((observation, "observation"))
    return found


def _warning(code: CheckCode, message: str) -> ValidationIssue:
    return ValidationIssue(
        code=code.value,
        message=message,
        severity=IssueSeverity.WARNING,
        subject_kind="synthesis",
        requires_human_review=True,
    )


def _dedupe[T](items: Iterable[T], *, key: Callable[[T], str]) -> list[T]:
    by_key: dict[str, T] = {}
    for item in items:
        by_key[key(item)] = item
    return list(by_key.values())
