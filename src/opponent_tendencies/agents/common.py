"""Shared machinery for the three specialist agents.

Stage 7 agents differ only in what deterministic Python hands them and which
prompt they render: the Statistics agent gets computed tables, the Beat-Reporter
agent gets filtered articles and film breakdowns, and the Coach-Quote agent gets
verbatim post-game quotes. Everything after the model replies is shared:

* Python builds the evidence catalog (stable IDs, sorted, deduplicated).
* Python rejects any citation or week the model invented and records why.
* Python stamps opponent, games, weeks, prompt version, and status.
* Python never lets a low-confidence or suppressed-table claim pass as strong.

The model only interprets the supplied data; it never owns identity, ordering,
or the arithmetic behind a number.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import HttpUrl, ValidationError

from opponent_tendencies.llm.client import LLMMessage, StructuredResult, TokenUsage
from opponent_tendencies.llm.prompts import (
    get_prompt,
    render_evidence_catalog,
    render_games_summary,
    render_retrieved_sources,
    render_stat_tables,
)
from opponent_tendencies.llm.prompts.blocks import RetrievedText
from opponent_tendencies.llm.schemas import (
    AgentResponse,
    DraftObservation,
    DraftTendency,
    WireModel,
)
from opponent_tendencies.models import (
    AgentName,
    AgentOutput,
    Confidence,
    EvidenceRef,
    Game,
    Observation,
    ObservationCategory,
    SectionStatus,
    StatTable,
    Team,
    Tendency,
    ValidationIssue,
)

__all__ = [
    "AgentConversion",
    "AgentResult",
    "LLMCompleter",
    "as_agent_response",
    "build_media_evidence",
    "build_output",
    "build_quote_evidence",
    "build_stat_evidence",
    "convert_agent_response",
    "format_weeks",
    "prompt_context",
    "run_agent_prompt",
    "stat_evidence",
    "unavailable_output",
]


class LLMCompleter(Protocol):
    """The one client method an agent needs (implemented by ``ArcLLMClient``)."""

    def complete_structured(
        self,
        messages: Sequence[LLMMessage],
        output: type[WireModel] | str,
        *,
        temperature: float = ...,
        max_tokens: int | None = ...,
    ) -> StructuredResult[WireModel]: ...


@dataclass(frozen=True)
class AgentResult:
    """One specialist agent's output plus the LLM telemetry behind it."""

    output: AgentOutput
    usage: TokenUsage = field(default_factory=TokenUsage)
    llm_calls: int = 0
    latency_seconds: float = 0.0
    warnings: tuple[ValidationIssue, ...] = ()

    @property
    def is_available(self) -> bool:
        return self.output.is_available


def format_weeks(weeks: Sequence[int]) -> str:
    """Render a sorted week selection for a prompt (``5-9`` or ``5, 7``)."""

    ordered = sorted({int(week) for week in weeks})
    if not ordered:
        raise ValueError("weeks must not be empty")
    if len(ordered) == 1:
        return str(ordered[0])
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]}-{ordered[-1]}"
    return ", ".join(str(week) for week in ordered)


# ---------------------------------------------------------------------------
# Deterministic evidence catalogs
# ---------------------------------------------------------------------------


def stat_evidence(tables: Sequence[StatTable]) -> tuple[EvidenceRef, ...]:
    """One table-level evidence pointer per deterministic statistic table."""

    return _sorted_unique(table.evidence() for table in tables)


def build_stat_evidence(tables: Sequence[StatTable]) -> tuple[EvidenceRef, ...]:
    """Alias used by the Statistics agent and synthesis."""

    return stat_evidence(tables)


def build_media_evidence(sources: Sequence[RetrievedText]) -> tuple[EvidenceRef, ...]:
    """Build media evidence from validated retrieval units.

    Retrieval text is validated as an exact stored span before this runs, so a
    snippet built from it stays grounded. Snippets are capped at the contract's
    1,000-character limit by taking a prefix, which is still an exact span.
    """

    refs: list[EvidenceRef] = []
    for source in sources:
        unit_id = source.unit_id
        media_id = str(source.metadata.get("media_id", ""))
        text = source.text
        metadata = source.metadata
        kind = str(metadata.get("unit_kind", ""))
        chunk_id = unit_id if kind == "chunk" else None
        title = str(metadata.get("title") or metadata.get("source_id") or media_id)
        source_url = metadata.get("url")
        refs.append(
            EvidenceRef.for_media(
                media_id,
                label=f"{title} ({kind or 'media'})",
                chunk_id=chunk_id,
                snippet=text[:1000] if text.strip() else None,
                source_url=_url(source_url),
            )
        )
    return _sorted_unique(refs)


def build_quote_evidence(sources: Sequence[RetrievedText]) -> tuple[EvidenceRef, ...]:
    """Build quote evidence from validated retrieval units.

    Quote units cite the stored quote by ID; any generic chunk from a
    coach-quote transcript falls back to media evidence.
    """

    refs: list[EvidenceRef] = []
    for source in sources:
        metadata = source.metadata
        kind = str(metadata.get("unit_kind", ""))
        text = source.text
        quote_id = metadata.get("quote_id")
        media_id = str(metadata.get("media_id", ""))
        source_url = metadata.get("url")
        if kind == "quote" and isinstance(quote_id, str) and quote_id:
            speaker = str(metadata.get("speaker") or "Coach")
            refs.append(
                EvidenceRef.for_quote(
                    quote_id,
                    label=f"{speaker} quote",
                    media_id=media_id,
                    snippet=text[:1000] if text.strip() else None,
                    source_url=_url(source_url),
                )
            )
        else:
            title = str(metadata.get("title") or metadata.get("source_id") or media_id)
            refs.append(
                EvidenceRef.for_media(
                    media_id,
                    label=f"{title} (media)",
                    chunk_id=source.unit_id,
                    snippet=text[:1000] if text.strip() else None,
                    source_url=_url(source_url),
                )
            )
    return _sorted_unique(refs)


def _url(value: object) -> HttpUrl | None:
    if isinstance(value, str) and value.strip():
        return HttpUrl(value)
    return None


def _sorted_unique(refs: Iterable[EvidenceRef]) -> tuple[EvidenceRef, ...]:
    by_id = {ref.evidence_id: ref for ref in refs}
    return tuple(by_id[evidence_id] for evidence_id in sorted(by_id))


# ---------------------------------------------------------------------------
# Prompt execution
# ---------------------------------------------------------------------------


def run_agent_prompt(
    client: LLMCompleter,
    *,
    prompt_name: str,
    context: dict[str, str],
    max_tokens: int | None = None,
) -> StructuredResult[WireModel]:
    """Render one prompt with deterministic blocks and request its schema."""

    prompt = get_prompt(prompt_name)
    missing = sorted(set(prompt.required_context) - set(context))
    extra = sorted(set(context) - set(prompt.required_context))
    if missing or extra:
        raise ValueError(
            f"prompt {prompt_name!r} context mismatch: missing {missing}, unexpected {extra}"
        )
    rendered = prompt.render(**context)
    if max_tokens is None:
        return client.complete_structured(rendered.messages, prompt.structured_output)
    return client.complete_structured(
        rendered.messages, prompt.structured_output, max_tokens=max_tokens
    )


def prompt_context(
    *,
    opponent: Team,
    season: int,
    weeks: Sequence[int],
    stat_tables: Sequence[StatTable] | None = None,
    games: Sequence[Game] | None = None,
    sources: Sequence[RetrievedText] | None = None,
    evidence: Sequence[EvidenceRef],
) -> dict[str, str]:
    """Build the context keys shared by the three specialist prompts."""

    context = {
        "opponent": opponent.value,
        "season": str(season),
        "weeks": format_weeks(weeks),
        "available_evidence": render_evidence_catalog(evidence),
    }
    if games is not None:
        context["games_summary"] = render_games_summary(games)
    if stat_tables is not None:
        context["stat_tables"] = render_stat_tables(stat_tables)
    if sources is not None:
        context["retrieved_sources"] = render_retrieved_sources(sources)
    return context


# ---------------------------------------------------------------------------
# Draft -> stored contracts
# ---------------------------------------------------------------------------


@dataclass
class AgentConversion:
    tendencies: list[Tendency] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    dropped: int = 0


def convert_agent_response(
    response: AgentResponse,
    *,
    agent: AgentName,
    catalog: Sequence[EvidenceRef],
    weeks: Sequence[int],
    weak_evidence_ids: Iterable[str] = (),
) -> AgentConversion:
    """Turn wire drafts into frozen contracts, dropping ungrounded claims.

    A draft whose evidence IDs or weeks are not in the deterministic catalog is
    discarded and counted, never repaired by guesswork. Low confidence and any
    citation of a suppressed statistic table become explicit weak evidence.
    """

    selected = {int(week) for week in weeks}
    allowed_evidence = {ref.evidence_id for ref in catalog}
    weak = set(weak_evidence_ids)
    conversion = AgentConversion()

    for draft_tendency in response.tendencies:
        tendency = _to_tendency(draft_tendency, allowed_evidence, selected, weak)
        if tendency is None:
            conversion.dropped += 1
        else:
            conversion.tendencies.append(tendency)

    for draft_observation in response.observations:
        observation = _to_observation(draft_observation, allowed_evidence, selected, weak, agent)
        if observation is None:
            conversion.dropped += 1
        else:
            conversion.observations.append(observation)

    return conversion


def _evidence_ids(ids: Sequence[str], allowed: set[str]) -> list[str] | None:
    unique = sorted({str(evidence_id) for evidence_id in ids})
    if not unique or not set(unique).issubset(allowed):
        return None
    return unique


def _weeks(weeks: Sequence[int], selected: set[int]) -> list[int] | None:
    unique = sorted({int(week) for week in weeks})
    if not unique or not set(unique).issubset(selected):
        return None
    return unique


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    return value if math.isfinite(value) else None


def _weak(confidence: Confidence, evidence_ids: Sequence[str], weak_ids: set[str]) -> bool:
    return confidence is Confidence.LOW or any(
        evidence_id in weak_ids for evidence_id in evidence_ids
    )


def _safe_confidence(confidence: Confidence, is_weak: bool) -> Confidence:
    if is_weak and confidence is Confidence.HIGH:
        return Confidence.MEDIUM
    return confidence


def _to_tendency(
    draft: DraftTendency,
    allowed_evidence: set[str],
    selected: set[int],
    weak_ids: set[str],
) -> Tendency | None:
    evidence_ids = _evidence_ids(draft.evidence_ids, allowed_evidence)
    weeks = _weeks(draft.weeks, selected)
    if evidence_ids is None or weeks is None:
        return None
    is_weak = _weak(draft.confidence, evidence_ids, weak_ids)
    try:
        return Tendency(
            statement=draft.statement,
            category=draft.category,
            direction=draft.direction,
            evidence_ids=evidence_ids,
            weeks=weeks,
            confidence=_safe_confidence(draft.confidence, is_weak),
            weak_evidence=is_weak,
            metric=draft.metric,
            value=_finite(draft.value),
            comparison=draft.comparison,
        )
    except (ValidationError, ValueError):
        return None


def _to_observation(
    draft: DraftObservation,
    allowed_evidence: set[str],
    selected: set[int],
    weak_ids: set[str],
    agent: AgentName,
) -> Observation | None:
    evidence_ids = _evidence_ids(draft.evidence_ids, allowed_evidence)
    weeks = _weeks(draft.weeks, selected)
    if evidence_ids is None or weeks is None:
        return None
    is_weak = _weak(draft.confidence, evidence_ids, weak_ids)
    treat_as_context = draft.treat_as_context
    if agent is AgentName.COACH_QUOTE and draft.category in {
        ObservationCategory.COACH_EXPLANATION,
        ObservationCategory.COACH_ADJUSTMENT,
    }:
        # A coach explaining or adjusting is context by contract, whatever the
        # model returned (requirement: quotes are context, not proof).
        treat_as_context = True
    try:
        return Observation(
            statement=draft.statement,
            category=draft.category,
            evidence_ids=evidence_ids,
            weeks=weeks,
            confidence=_safe_confidence(draft.confidence, is_weak),
            weak_evidence=is_weak,
            treat_as_context=treat_as_context,
        )
    except (ValidationError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Agent output construction
# ---------------------------------------------------------------------------


def build_output(
    *,
    agent: AgentName,
    opponent: Team,
    games: Sequence[Game],
    weeks: Sequence[int],
    evidence: Sequence[EvidenceRef],
    conversion: AgentConversion,
    limitations: Iterable[str],
    prompt_version: str,
    model: str | None,
    stat_tables: Sequence[StatTable] = (),
    status: SectionStatus = SectionStatus.COMPLETED,
) -> AgentOutput:
    """Stamp deterministic identity onto a parsed model response."""

    collected = [str(limitation) for limitation in limitations if str(limitation).strip()]
    if conversion.dropped:
        collected.append(
            f"Dropped {conversion.dropped} finding(s) that cited evidence or weeks "
            "outside the deterministic catalog."
        )
    if (
        status is SectionStatus.COMPLETED
        and not conversion.tendencies
        and not conversion.observations
        and not collected
    ):
        collected.append(
            "The model reported no supported findings for the selected weeks and games."
        )

    return AgentOutput(
        agent=agent,
        opponent=opponent,
        games=list(games),
        weeks=sorted({int(week) for week in weeks}),
        tendencies=conversion.tendencies,
        observations=conversion.observations,
        evidence=list(evidence),
        limitations=sorted(set(collected)),
        stat_tables=sorted(stat_tables, key=lambda table: table.stat_table_id),
        status=status,
        prompt_version=prompt_version,
        model=model,
    )


def unavailable_output(
    *,
    agent: AgentName,
    opponent: Team,
    games: Sequence[Game],
    weeks: Sequence[int],
    prompt_version: str,
    limitation: str,
    status: SectionStatus = SectionStatus.UNAVAILABLE,
) -> AgentOutput:
    """Build an explicit unavailable/failed section with no findings."""

    return AgentOutput(
        agent=agent,
        opponent=opponent,
        games=list(games),
        weeks=sorted({int(week) for week in weeks}),
        tendencies=[],
        observations=[],
        evidence=[],
        limitations=[limitation],
        stat_tables=[],
        status=status,
        prompt_version=prompt_version,
        model=None,
    )


def as_agent_response(result: StructuredResult[WireModel]) -> AgentResponse:
    """Narrow a parsed wire model to the agent response it must be."""

    if not isinstance(result.parsed, AgentResponse):
        raise TypeError(f"expected AgentResponse, got {type(result.parsed).__name__}")
    return result.parsed
