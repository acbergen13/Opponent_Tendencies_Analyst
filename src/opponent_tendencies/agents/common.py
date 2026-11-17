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
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import HttpUrl, ValidationError

from opponent_tendencies.data.media_store import MediaData
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
    EvidenceKind,
    EvidenceRef,
    Game,
    MediaType,
    Observation,
    ObservationCategory,
    SectionStatus,
    StatTable,
    Team,
    Tendency,
    ValidationIssue,
)

if TYPE_CHECKING:
    from opponent_tendencies.data.index import MediaIndex, RetrievedSource

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
    "evidence_outside_weeks",
    "format_weeks",
    "prompt_context",
    "retrieve_selected_first",
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


def retrieve_selected_first(
    index: MediaIndex,
    query: str,
    *,
    opponent: Team,
    season: int,
    weeks: Sequence[int],
    source_types: Sequence[MediaType],
    kinds: Sequence[Literal["chunk", "quote"]],
    k: int,
) -> tuple[RetrievedSource, ...]:
    """Retrieve units from the selected weeks first, then backfill with background.

    Earlier-week sources may still be shown as background context, but they
    only fill slots the selected weeks could not, so they never crowd out
    in-window reporting.
    """

    selected = tuple(sorted({int(week) for week in weeks}))
    primary = index.retrieve(
        query,
        opponent=opponent,
        season=season,
        weeks=selected,
        source_types=source_types,
        kinds=kinds,
        k=k,
    )
    if len(primary) >= k:
        return primary
    background = index.retrieve(
        query,
        opponent=opponent,
        season=season,
        max_week=max(selected),
        source_types=source_types,
        kinds=kinds,
        k=k,
    )
    seen = {source.unit_id for source in primary}
    extra = [source for source in background if source.unit_id not in seen]
    return (*primary, *extra[: k - len(primary)])


def evidence_outside_weeks(
    evidence: Iterable[EvidenceRef],
    media_data: MediaData,
    weeks: Sequence[int],
) -> frozenset[str]:
    """Return the media/quote evidence IDs that cannot support the selected weeks.

    A media item or quote from a week outside the selection is background, and
    one with no recorded week cannot be aligned at all; neither may be the
    only support for a finding about the selected weeks (requirement 26).
    Statistic-table and play evidence is computed from the selected games, so
    it is always in the window.
    """

    selected = {int(week) for week in weeks}
    outside: set[str] = set()
    for ref in evidence:
        week: int | None
        if ref.kind is EvidenceKind.QUOTE:
            quote = media_data.quote_by_id(ref.quote_id or "")
            week = quote.week if quote is not None else None
        elif ref.kind is EvidenceKind.MEDIA:
            item = media_data.media_by_id(ref.media_id or "")
            week = item.week if item is not None else None
        else:
            continue
        if week is None or week not in selected:
            outside.add(ref.evidence_id)
    return frozenset(outside)


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
    extra_messages: Sequence[LLMMessage] = (),
) -> StructuredResult[WireModel]:
    """Render one prompt with deterministic blocks and request its schema.

    ``extra_messages`` are appended after the rendered prompt; they carry
    deterministic, Python-authored instructions such as a retry constraint.
    """

    prompt = get_prompt(prompt_name)
    missing = sorted(set(prompt.required_context) - set(context))
    extra = sorted(set(context) - set(prompt.required_context))
    if missing or extra:
        raise ValueError(
            f"prompt {prompt_name!r} context mismatch: missing {missing}, unexpected {extra}"
        )
    rendered = prompt.render(**context)
    messages = (*rendered.messages, *extra_messages)
    if max_tokens is None:
        return client.complete_structured(messages, prompt.structured_output)
    return client.complete_structured(messages, prompt.structured_output, max_tokens=max_tokens)


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
    out_of_window: int = 0


def convert_agent_response(
    response: AgentResponse,
    *,
    agent: AgentName,
    catalog: Sequence[EvidenceRef],
    weeks: Sequence[int],
    weak_evidence_ids: Iterable[str] = (),
    out_of_window_ids: Iterable[str] = (),
) -> AgentConversion:
    """Turn wire drafts into frozen contracts, dropping ungrounded claims.

    A draft whose evidence IDs or weeks are not in the deterministic catalog is
    discarded and counted, never repaired by guesswork. Low confidence and any
    citation of a suppressed statistic table become explicit weak evidence.

    ``out_of_window_ids`` are sources from outside the selected weeks (see
    :func:`evidence_outside_weeks`). A tendency supported *only* by them is
    dropped; an observation supported only by them is kept as weak, low
    confidence background; any finding citing one alongside in-window evidence
    is marked weak.
    """

    selected = {int(week) for week in weeks}
    allowed_evidence = {ref.evidence_id for ref in catalog}
    weak = set(weak_evidence_ids)
    outside = set(out_of_window_ids)
    conversion = AgentConversion()

    for draft_tendency in response.tendencies:
        cited = set(draft_tendency.evidence_ids)
        if outside and cited and cited <= outside and cited <= allowed_evidence:
            conversion.out_of_window += 1
            continue
        tendency = _to_tendency(draft_tendency, allowed_evidence, selected, weak | outside)
        if tendency is None:
            conversion.dropped += 1
        else:
            conversion.tendencies.append(tendency)

    for draft_observation in response.observations:
        cited = set(draft_observation.evidence_ids)
        background_only = bool(outside and cited and cited <= outside)
        observation = _to_observation(
            draft_observation,
            allowed_evidence,
            selected,
            weak | outside,
            agent,
            background_only=background_only,
        )
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
    *,
    background_only: bool = False,
) -> Observation | None:
    evidence_ids = _evidence_ids(draft.evidence_ids, allowed_evidence)
    weeks = _weeks(draft.weeks, selected)
    if evidence_ids is None or weeks is None:
        return None
    is_weak = _weak(draft.confidence, evidence_ids, weak_ids)
    confidence = _safe_confidence(draft.confidence, is_weak)
    if background_only:
        # Out-of-window sources are background only: never stronger than low.
        confidence = Confidence.LOW
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
            confidence=confidence,
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
    if conversion.out_of_window:
        collected.append(
            f"Dropped {conversion.out_of_window} tendency(ies) supported only by sources "
            "from outside the selected weeks; background sources cannot support tendencies."
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
