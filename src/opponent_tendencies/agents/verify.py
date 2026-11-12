"""Separate claim-verification call (not a fifth agent).

The verifier sees each claim beside deterministic snippets resolved from stored
sources: play descriptions, exact media spans, verbatim quotes, and rendered
statistic tables. Python applies the verdicts — it never asks the verifier to
rewrite the report:

* ``supported`` keeps a claim only when the verdict cites snippets attached to
  that claim.
* ``partially_supported`` replaces the statement with the verifier's correction.
* ``unsupported``/``contradicted``/missing verdicts remove the claim and leave an
  explicit weak-evidence flag plus a warning for human review.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from opponent_tendencies.agents.common import LLMCompleter, format_weeks
from opponent_tendencies.data.media_store import MediaData
from opponent_tendencies.data.play_store import SeasonData
from opponent_tendencies.llm.client import TokenUsage
from opponent_tendencies.llm.prompts import (
    ClaimSnippet,
    VerifierClaim,
    get_prompt,
    render_verifier_claims,
)
from opponent_tendencies.llm.schemas import ClaimVerdict, VerifiedClaim, VerifierResponse
from opponent_tendencies.models import (
    AgentFinding,
    EvidenceKind,
    EvidenceRef,
    Game,
    IssueSeverity,
    Observation,
    Play,
    StatTable,
    Team,
    Tendency,
    ValidationIssue,
    WeakEvidence,
    WeakEvidenceReason,
)
from opponent_tendencies.validation.checks import CheckCode

__all__ = [
    "VerificationResult",
    "VerifiedContent",
    "apply_verification",
    "build_verifier_claims",
    "run_verifier",
]


@dataclass(frozen=True)
class VerificationResult:
    """Verdicts from one verification call plus call telemetry."""

    claims: tuple[VerifiedClaim, ...] = ()
    usage: TokenUsage = field(default_factory=TokenUsage)
    llm_calls: int = 0
    latency_seconds: float = 0.0
    model: str | None = None

    def by_claim_id(self) -> dict[str, VerifiedClaim]:
        return {claim.claim_id: claim for claim in self.claims}


@dataclass(frozen=True)
class VerifiedContent:
    """Findings after deterministic verdict application."""

    tendencies: tuple[Tendency, ...]
    observations: tuple[Observation, ...]
    weak_evidence: tuple[WeakEvidence, ...]
    warnings: tuple[ValidationIssue, ...]
    removed: int = 0
    corrected: int = 0


def build_verifier_claims(
    findings: Sequence[AgentFinding],
    evidence: Sequence[EvidenceRef],
    *,
    season_data: SeasonData,
    media_data: MediaData,
    stat_tables: Sequence[StatTable],
) -> tuple[VerifierClaim, ...]:
    """Attach resolved, deterministic snippets to every claim."""

    resolver = _SnippetResolver(
        evidence=evidence,
        season_data=season_data,
        media_data=media_data,
        stat_tables=stat_tables,
    )
    claims: list[VerifierClaim] = []
    for finding in findings:
        snippets: list[ClaimSnippet] = []
        seen: set[str] = set()
        for evidence_id in finding.evidence_ids:
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            text = resolver.text_for(evidence_id)
            if text is None:
                continue
            snippets.append(ClaimSnippet(evidence_id=evidence_id, text=text))
        claims.append(
            VerifierClaim(
                claim_id=finding.finding_id,
                statement=finding.statement,
                snippets=tuple(snippets),
            )
        )
    return tuple(claims)


def run_verifier(
    *,
    client: LLMCompleter,
    opponent: Team,
    weeks: Sequence[int],
    claims: Sequence[VerifierClaim],
) -> VerificationResult:
    """Ask the verifier to judge every claim against its attached snippets."""

    if not claims:
        return VerificationResult()
    prompt = get_prompt("verifier")
    rendered = prompt.render(
        opponent=opponent.value,
        weeks=format_weeks(weeks),
        claims=render_verifier_claims(claims),
    )
    result = client.complete_structured(rendered.messages, prompt.structured_output)
    if not isinstance(result.parsed, VerifierResponse):
        raise TypeError(f"expected VerifierResponse, got {type(result.parsed).__name__}")
    return VerificationResult(
        claims=tuple(result.parsed.claims),
        usage=result.usage,
        llm_calls=result.llm_calls,
        latency_seconds=result.latency_seconds,
        model=result.model,
    )


def apply_verification(
    *,
    tendencies: Sequence[Tendency],
    observations: Sequence[Observation],
    claims: Sequence[VerifierClaim],
    verdicts: Mapping[str, VerifiedClaim],
    weeks: Sequence[int],
) -> VerifiedContent:
    """Apply verdicts deterministically and record every removal/correction."""

    attached = {
        claim.claim_id: {snippet.evidence_id for snippet in claim.snippets} for claim in claims
    }
    selected = list(weeks)
    kept_tendencies: list[Tendency] = []
    kept_observations: list[Observation] = []
    weak: list[WeakEvidence] = []
    warnings: list[ValidationIssue] = []
    removed = 0
    corrected = 0

    for finding in [*tendencies, *observations]:
        verdict = verdicts.get(finding.finding_id)
        kind = "tendency" if isinstance(finding, Tendency) else "observation"
        claim_evidence = attached.get(finding.finding_id, set())

        if verdict is None:
            removed += 1
            warnings.append(
                _warning(
                    CheckCode.VERIFICATION_MISSING_VERDICT,
                    f"The verifier returned no verdict for {kind} {finding.finding_id}; "
                    "the claim was removed rather than published unverified.",
                    finding.finding_id,
                )
            )
            weak.append(_removed_weak(finding, kind, selected, "no verifier verdict"))
            continue

        invalid_citations = bool(verdict.evidence_ids) and not set(verdict.evidence_ids).issubset(
            claim_evidence
        )
        if verdict.verdict in {ClaimVerdict.UNSUPPORTED, ClaimVerdict.CONTRADICTED} or (
            invalid_citations
        ):
            removed += 1
            reason = (
                f"a verdict cited evidence not attached to the claim ({verdict.verdict.value})"
                if invalid_citations
                else verdict.verdict.value
            )
            warnings.append(
                _warning(
                    CheckCode.VERIFICATION_REMOVED_CLAIM,
                    f"Removed {kind} {finding.finding_id}: {reason}. Verifier reason: "
                    f"{verdict.reason}",
                    finding.finding_id,
                )
            )
            weak.append(_removed_weak(finding, kind, selected, verdict.reason))
            continue

        if verdict.verdict is ClaimVerdict.PARTIALLY_SUPPORTED:
            replacement = verdict.correction
            if replacement is None or not replacement.strip():
                removed += 1
                warnings.append(
                    _warning(
                        CheckCode.VERIFICATION_REMOVED_CLAIM,
                        f"Removed {kind} {finding.finding_id}: the verifier marked it "
                        "partially supported without a correction.",
                        finding.finding_id,
                    )
                )
                weak.append(_removed_weak(finding, kind, selected, verdict.reason))
                continue
            corrected += 1
            warnings.append(
                _warning(
                    CheckCode.VERIFICATION_CORRECTED_CLAIM,
                    f"Corrected {kind} {finding.finding_id} using the verifier's "
                    f"correction. Verifier reason: {verdict.reason}",
                    finding.finding_id,
                )
            )
            adjusted = _replace_statement(finding, replacement)
            if isinstance(adjusted, Tendency):
                kept_tendencies.append(adjusted)
            elif isinstance(adjusted, Observation):
                kept_observations.append(adjusted)
            continue

        if isinstance(finding, Tendency):
            kept_tendencies.append(finding)
        elif isinstance(finding, Observation):
            kept_observations.append(finding)

    return VerifiedContent(
        tendencies=tuple(kept_tendencies),
        observations=tuple(kept_observations),
        weak_evidence=tuple(weak),
        warnings=tuple(warnings),
        removed=removed,
        corrected=corrected,
    )


class _SnippetResolver:
    """Resolve evidence IDs to deterministic, stored snippets."""

    def __init__(
        self,
        *,
        evidence: Sequence[EvidenceRef],
        season_data: SeasonData,
        media_data: MediaData,
        stat_tables: Sequence[StatTable],
    ) -> None:
        self._refs = {item.evidence_id: item for item in evidence}
        self._games: dict[str, Game] = {game.game_id: game for game in season_data.games}
        self._plays: dict[str, Play] = {play.key: play for play in season_data.plays}
        self._media = {item.media_id: item for item in media_data.media}
        self._quotes = {quote.quote_id: quote for quote in media_data.quotes}
        self._tables = {table.stat_table_id: table for table in stat_tables}

    def text_for(self, evidence_id: str) -> str | None:
        ref = self._refs.get(evidence_id)
        if ref is None:
            return None
        if ref.kind is EvidenceKind.PLAY:
            return self._play_text(ref)
        if ref.kind is EvidenceKind.MEDIA:
            return self._media_text(ref)
        if ref.kind is EvidenceKind.QUOTE:
            return self._quote_text(ref)
        return self._stat_text(ref)

    def _play_text(self, ref: EvidenceRef) -> str | None:
        if ref.game_id is None or ref.play_id is None:
            return None
        play = self._plays.get(f"{ref.game_id}:{ref.play_id}")
        if play is None:
            return None
        game = self._games.get(play.game_id)
        parts = [f"{play.game_id} play {play.play_id}"]
        if game is not None:
            parts.append(f"week {game.week}")
        if play.posteam is not None:
            parts.append(str(play.posteam.value))
        if play.down is not None:
            parts.append(f"down {play.down}")
        if play.yards_to_go is not None:
            parts.append(f"{play.yards_to_go} to go")
        parts.append(play.play_type.value)
        if play.yards_gained is not None:
            parts.append(f"{play.yards_gained} yards")
        if ref.snippet:
            parts.append(ref.snippet)
        return "; ".join(parts)

    def _media_text(self, ref: EvidenceRef) -> str | None:
        stored = self._media.get(ref.media_id or "")
        if stored is None:
            return None
        if ref.snippet and ref.snippet in stored.body:
            return ref.snippet
        return stored.body[:1000]

    def _quote_text(self, ref: EvidenceRef) -> str | None:
        quote = self._quotes.get(ref.quote_id or "")
        return None if quote is None else quote.text

    def _stat_text(self, ref: EvidenceRef) -> str | None:
        table = self._tables.get(ref.stat_table_id or "")
        if table is None:
            return None
        rows = table.rows
        label = ref.label
        targeted = [row for row in rows if label.endswith(f"/ {row.label}")]
        if targeted:
            rows = targeted
        rendered = "; ".join(
            f"{row.label}={row.value if row.value is not None else 'n/a'} (n={row.n})"
            for row in rows
        )
        scope = table.dimension.value
        if table.dimension_value:
            scope += f"/{table.dimension_value}"
        return (
            f"{table.title}; metric={table.metric.value}; scope={scope}; "
            f"suppressed={str(table.suppressed).lower()}; {rendered}"
        )


def _replace_statement[T: AgentFinding](finding: T, statement: str) -> T:
    data = finding.model_dump(mode="python")
    data.pop("finding_id")
    data["statement"] = statement
    return type(finding).model_validate(data)


def _removed_weak(
    finding: AgentFinding,
    kind: str,
    weeks: Sequence[int],
    reason: str,
) -> WeakEvidence:
    return WeakEvidence(
        topic=f"Removed unverified {kind}",
        description=f"{finding.statement} (verifier: {reason})",
        reason=WeakEvidenceReason.UNVERIFIED_CLAIM,
        evidence_ids=list(finding.evidence_ids),
        weeks=list(weeks),
        agent=None,
    )


def _warning(code: CheckCode, message: str, subject_id: str) -> ValidationIssue:
    return ValidationIssue(
        code=code.value,
        message=message,
        severity=IssueSeverity.WARNING,
        subject_kind="claim",
        subject_id=subject_id,
        requires_human_review=True,
    )
