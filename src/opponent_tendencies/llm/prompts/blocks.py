"""Deterministic rendering of the data blocks used in agent prompts.

Python owns every character the model sees around retrieved text: the
delimiters, the provenance labels, and the neutralization of delimiter
look-alikes inside untrusted text (requirements 23 and 29 in the design
write-up). Retrieved text is always wrapped in a labeled, delimited block so
it reads as data, never as instructions, and the LLM never formats data
itself.

Neutralization is deliberately minimal: only the literal block tags are
rewritten to look-alike Unicode angle brackets, so article and quote wording
stays copyable from the prompt view.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from opponent_tendencies.models import (
    AgentFinding,
    AgentOutput,
    EvidenceRef,
    Game,
    Report,
    StatTable,
)

__all__ = [
    "ClaimSnippet",
    "VerifierClaim",
    "mentioned_block_tags",
    "neutralize_tags",
    "render_agent_findings",
    "render_evidence_catalog",
    "render_games_summary",
    "render_report_context",
    "render_retrieved_sources",
    "render_stat_tables",
    "render_verifier_claims",
]

BLOCK_TAGS: tuple[str, ...] = (
    "retrieved_data",
    "deterministic_statistics",
    "available_evidence",
    "agent_findings",
    "claims_to_verify",
    "report_context",
    "source",
)

_PROVENANCE_KEYS: tuple[str, ...] = (
    "media_id",
    "source_id",
    "media_type",
    "unit_kind",
    "publisher",
    "author",
    "title",
    "url",
    "week",
    "published_on",
    "game_id",
    "speaker",
    "speaker_role",
    "spoken_on",
)


class RetrievedText(Protocol):
    """Structural type satisfied by retrieval results from the media index."""

    @property
    def unit_id(self) -> str: ...

    @property
    def text(self) -> str: ...

    @property
    def metadata(self) -> Mapping[str, str | int | float | bool]: ...


@dataclass(frozen=True)
class ClaimSnippet:
    """One source snippet attached to a claim for verification."""

    evidence_id: str
    text: str


@dataclass(frozen=True)
class VerifierClaim:
    """One claim to verify, with the snippets it must be checked against."""

    claim_id: str
    statement: str
    snippets: tuple[ClaimSnippet, ...] = ()


def neutralize_tags(text: str) -> str:
    """Rewrite literal block tags inside untrusted text to look-alikes.

    ``<retrieved_data>`` becomes ``⟨retrieved_data⟩`` so text inside a data
    block cannot close the block early or open a fake one.
    """

    for tag in BLOCK_TAGS:
        text = text.replace(f"<{tag}", f"⟨{tag}")
        text = text.replace(f"</{tag}>", f"⟨/{tag}⟩")
    return text


def mentioned_block_tags(text: str) -> tuple[str, ...]:
    """Return which data-block tags a prompt text references."""

    return tuple(tag for tag in BLOCK_TAGS if f"<{tag}" in text or f"</{tag}>" in text)


def render_games_summary(games: Sequence[Game]) -> str:
    """One deterministic line per selected game (dates and scores included)."""

    if not games:
        return "(no games selected)"
    lines: list[str] = []
    for game in games:
        score = ""
        if game.home_score is not None and game.away_score is not None:
            score = (
                f", {game.away_team.value} {game.away_score} at "
                f"{game.home_team.value} {game.home_score}"
            )
        lines.append(f"week {game.week}: {game.game_id} on {game.game_date.isoformat()}{score}")
    return "\n".join(lines)


def render_retrieved_sources(sources: Sequence[RetrievedText]) -> str:
    """Wrap validated retrieval units in a delimited, provenance-labeled block."""

    lines: list[str] = ['<retrieved_data source="untrusted">']
    if not sources:
        lines.append("(no sources retrieved; report this as a limitation)")
    for source in sources:
        attributes = [f'unit_id="{_escape_attribute(source.unit_id)}"']
        for key in _PROVENANCE_KEYS:
            value = source.metadata.get(key)
            if value is None or value == "":
                continue
            if key == "week" and value == 0:
                continue
            attributes.append(f'{key}="{_escape_attribute(str(value))}"')
        lines.append(f"<source {' '.join(attributes)}>")
        lines.append(neutralize_tags(source.text))
        lines.append("</source>")
    lines.append("</retrieved_data>")
    return "\n".join(lines)


def render_stat_tables(tables: Sequence[StatTable]) -> str:
    """Render deterministic statistic tables as a delimited ground-truth block."""

    lines: list[str] = ["<deterministic_statistics>"]
    if not tables:
        lines.append("(no statistics computed)")
    for table in tables:
        attributes = [
            f'stat_table_id="{_escape_attribute(table.stat_table_id)}"',
            f'metric="{table.metric.value}"',
            f'dimension="{table.dimension.value}"',
        ]
        if table.dimension_value is not None:
            attributes.append(f'dimension_value="{_escape_attribute(table.dimension_value)}"')
        attributes.append(f"n_plays={table.n_plays}")
        attributes.append(f"min_sample={table.min_sample}")
        attributes.append(f"suppressed={str(table.suppressed).lower()}")
        lines.append(f"<table {' '.join(attributes)}>")
        lines.append(f"title: {table.title}")
        lines.append(f"weeks: {_join_values(table.weeks)}")
        lines.append(f"games: {_join_values(table.game_ids)}")
        for note in table.notes:
            lines.append(f"note: {note}")
        for row in table.rows:
            parts = [
                f"label={row.label!r}",
                f"value={_format_value(row.value)}",
                f"n={row.n}",
                f"unit={row.unit.value}",
                f"suppressed={str(row.suppressed).lower()}",
            ]
            if row.numerator is not None:
                parts.append(f"numerator={row.numerator}")
            if row.denominator is not None:
                parts.append(f"denominator={row.denominator}")
            lines.append(f"row: {' '.join(parts)}")
            for note in row.notes:
                lines.append(f"row_note: {note}")
        lines.append("</table>")
    lines.append("</deterministic_statistics>")
    return "\n".join(lines)


def render_evidence_catalog(evidence: Sequence[EvidenceRef]) -> str:
    """List the evidence IDs a response may cite, with kind and label."""

    lines: list[str] = ["<available_evidence>"]
    if not evidence:
        lines.append("(no evidence available; do not cite anything and report a limitation)")
    for item in evidence:
        lines.append(
            f"evidence: {item.evidence_id} kind={item.kind.value} "
            f"label={item.label!r} locator={_evidence_locator(item)}"
        )
    lines.append("</available_evidence>")
    return "\n".join(lines)


def render_agent_findings(outputs: Sequence[AgentOutput]) -> str:
    """Render validated agent outputs as an untrusted, delimited block for SYN."""

    lines: list[str] = ['<agent_findings source="untrusted">']
    if not outputs:
        lines.append("(no agent findings)")
    for output in outputs:
        lines.append(f'<agent_output agent="{output.agent.value}" status="{output.status.value}">')
        for tendency in output.tendencies:
            lines.append(_format_finding("tendency", tendency))
        for observation in output.observations:
            lines.append(_format_finding("observation", observation))
        for limitation in output.limitations:
            lines.append(f"limitation: {neutralize_tags(limitation)}")
        if output.stat_tables:
            lines.append(
                "stat_tables: " + ",".join(table.stat_table_id for table in output.stat_tables)
            )
        lines.append("</agent_output>")
    lines.append("</agent_findings>")
    return "\n".join(lines)


def render_verifier_claims(claims: Sequence[VerifierClaim]) -> str:
    """Render claims and their source snippets for the verification call."""

    lines: list[str] = ['<claims_to_verify source="untrusted">']
    if not claims:
        lines.append("(no claims to verify)")
    for claim in claims:
        lines.append(f'<claim claim_id="{_escape_attribute(claim.claim_id)}">')
        lines.append(f"statement: {neutralize_tags(claim.statement)}")
        if claim.snippets:
            for snippet in claim.snippets:
                lines.append(f'<snippet evidence_id="{_escape_attribute(snippet.evidence_id)}">')
                lines.append(neutralize_tags(snippet.text))
                lines.append("</snippet>")
        else:
            lines.append("(no snippets attached)")
        lines.append("</claim>")
    lines.append("</claims_to_verify>")
    return "\n".join(lines)


def render_report_context(report: Report) -> str:
    """Render the stored report for follow-up questions."""

    lines: list[str] = ['<report_context source="stored-report">']
    lines.append(f"opponent: {report.opponent.value}")
    lines.append(f"season: {report.season}")
    lines.append(f"weeks: {_join_values(report.weeks)}")
    lines.append(f"overview: {neutralize_tags(report.overview)}")
    for tendency in report.tendencies:
        lines.append(_format_finding("tendency", tendency))
    for observation in report.observations:
        lines.append(_format_finding("observation", observation))
    for agreement in report.agreements:
        lines.append(
            f"agreement: {neutralize_tags(agreement.statement)} "
            f"agents={','.join(agent.value for agent in agreement.agents)} "
            f"evidence={','.join(agreement.evidence_ids)}"
        )
    for conflict in report.conflicts:
        lines.append(f"conflict: {neutralize_tags(conflict.description)}")
    for gap in report.gaps:
        lines.append(f"gap: {neutralize_tags(gap.description)}")
    for weak in report.weak_evidence:
        lines.append(f"weak_evidence: {neutralize_tags(weak.description)}")
    if report.evidence:
        lines.append("evidence:")
        for item in report.evidence:
            lines.append(
                f"  evidence: {item.evidence_id} kind={item.kind.value} label={item.label!r}"
            )
    else:
        lines.append("evidence: (none)")
    lines.append("</report_context>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _format_finding(kind: str, finding: AgentFinding) -> str:
    parts = [f"{kind}: {neutralize_tags(finding.statement)!r}"]
    category = getattr(finding, "category", None)
    if category is not None:
        parts.append(f"category={category.value}")
    if kind == "tendency":
        parts.append(f"direction={finding.direction.value}")
    parts.append(f"confidence={finding.confidence.value}")
    if finding.metric is not None:
        parts.append(f"metric={finding.metric}")
    if finding.value is not None:
        parts.append(f"value={_format_value(finding.value)}")
    if finding.comparison is not None:
        parts.append(f"comparison={finding.comparison!r}")
    if kind == "observation" and finding.treat_as_context:
        parts.append("treat_as_context=true")
    if finding.weak_evidence:
        parts.append("weak_evidence=true")
    parts.append(f"weeks={_join_values(finding.weeks)}")
    parts.append(f"evidence={','.join(finding.evidence_ids)}")
    return " | ".join(parts)


def _evidence_locator(item: EvidenceRef) -> str:
    if item.kind.value == "play":
        return f"play {item.game_id}:{item.play_id}"
    if item.kind.value == "media":
        locator = item.media_id or "?"
        return f"media {locator}" + (f" chunk {item.chunk_id}" if item.chunk_id else "")
    if item.kind.value == "quote":
        return f"quote {item.quote_id}"
    return f"stat table {item.stat_table_id}"


def _escape_attribute(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _format_value(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _join_values(values: Sequence[object]) -> str:
    return ",".join(str(value) for value in values)
