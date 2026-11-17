"""JSON schemas for LLM structured outputs.

The models here are the *wire* contracts for model responses, not the stored
contracts in :mod:`opponent_tendencies.models`. Server-owned identity
(``opponent``, ``games``, ``weeks``, ``prompt_version``, stable IDs) is stamped
by deterministic Python after a response is parsed. A model therefore cannot
drop, invent, or drift those fields between agents (requirement 25 and issue 16
in the design write-up).

Every wire model:

* forbids additional properties,
* gives every field a required key (optional values are explicitly nullable),
* cites evidence with IDs supplied by Python, and
* uses the same enums as the stored contracts.

:func:`write_json_schemas` materializes the schemas for offline prompt work and
for OpenAI-compatible ``response_format`` payloads.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from opponent_tendencies.models import (
    AgentName,
    Confidence,
    GapReason,
    ObservationCategory,
    SpeakerRole,
    TendencyCategory,
    TendencyDirection,
    TrimmedStr,
    WeakEvidenceReason,
    Week,
)

# Bounded free text for model-authored fields. Constrained decoding honors
# ``maxLength``/``maxItems``, so a degenerate generation cannot grow one string
# or one list until the completion cap truncates the whole reply (a live
# 2026-11-17 synthesis hit 32,768 completion tokens with no reasoning). The
# stored contracts in ``models.py`` stay unbounded; only the wire is capped.
LABEL_MAX_CHARS = 120
STATEMENT_MAX_CHARS = 600
OVERVIEW_MAX_CHARS = 1500

WireLabel = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=LABEL_MAX_CHARS)
]
WireText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=STATEMENT_MAX_CHARS)
]
WireOverview = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=OVERVIEW_MAX_CHARS)
]


class WireModel(BaseModel):
    """Strict base class for model-facing JSON payloads."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DraftTendency(WireModel):
    statement: WireText
    category: TendencyCategory
    direction: TendencyDirection
    evidence_ids: list[TrimmedStr] = Field(min_length=1)
    confidence: Confidence
    weeks: list[Week] = Field(min_length=1)
    metric: WireLabel | None
    value: float | None
    comparison: WireLabel | None


class DraftObservation(WireModel):
    statement: WireText
    category: ObservationCategory
    evidence_ids: list[TrimmedStr] = Field(min_length=1)
    confidence: Confidence
    weeks: list[Week] = Field(min_length=1)
    treat_as_context: bool


class AgentResponse(WireModel):
    """STA, BRA, or CQA response body."""

    tendencies: list[DraftTendency]
    observations: list[DraftObservation]
    limitations: list[WireText]


class DraftAgreement(WireModel):
    topic: WireLabel
    statement: WireText
    agents: list[AgentName] = Field(min_length=2)
    evidence_ids: list[TrimmedStr] = Field(min_length=1)
    weeks: list[Week] = Field(min_length=1)


class DraftConflictPosition(WireModel):
    agent: AgentName
    statement: WireText
    evidence_ids: list[TrimmedStr] = Field(min_length=1)


class DraftConflict(WireModel):
    topic: WireLabel
    description: WireText
    positions: list[DraftConflictPosition] = Field(min_length=2)
    evidence_ids: list[TrimmedStr] = Field(min_length=1)
    weeks: list[Week] = Field(min_length=1)
    resolution: WireText | None


class DraftGap(WireModel):
    topic: WireLabel
    description: WireText
    reason: GapReason
    agents: list[AgentName]
    weeks: list[Week] = Field(min_length=1)


class DraftWeakEvidence(WireModel):
    topic: WireLabel
    description: WireText
    reason: WeakEvidenceReason
    evidence_ids: list[TrimmedStr]
    weeks: list[Week] = Field(min_length=1)
    agent: AgentName | None


class SynthesisResponse(WireModel):
    """SYN response body, before Python assigns IDs and validates citations.

    Every list and string is bounded in the schema so constrained decoding
    cannot spin in an endless array or string until the completion cap
    truncates the reply.
    """

    overview: WireOverview
    tendencies: list[DraftTendency] = Field(max_length=12)
    observations: list[DraftObservation] = Field(max_length=12)
    agreements: list[DraftAgreement] = Field(max_length=10)
    conflicts: list[DraftConflict] = Field(max_length=8)
    gaps: list[DraftGap] = Field(max_length=10)
    weak_evidence: list[DraftWeakEvidence] = Field(max_length=10)


class ClaimVerdict(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class VerifiedClaim(WireModel):
    claim_id: TrimmedStr
    verdict: ClaimVerdict
    reason: WireText
    evidence_ids: list[TrimmedStr]
    correction: WireText | None

    @model_validator(mode="after")
    def _validate_verdict(self) -> Self:
        if self.verdict is ClaimVerdict.SUPPORTED:
            if not self.evidence_ids:
                raise ValueError("a supported claim must cite evidence")
            if self.correction is not None:
                raise ValueError("a supported claim must not carry a correction")
        if self.verdict is ClaimVerdict.PARTIALLY_SUPPORTED and self.correction is None:
            raise ValueError("a partially supported claim must provide a correction")
        return self


class VerifierResponse(WireModel):
    """Separate verification call; this is not a fifth agent (requirement 10)."""

    claims: list[VerifiedClaim]


class FollowUpResponse(WireModel):
    """Follow-up answer over stored report evidence (requirement 17)."""

    answer: TrimmedStr
    supported: bool
    evidence_ids: list[TrimmedStr]
    limitations: list[TrimmedStr]

    @model_validator(mode="after")
    def _validate_support(self) -> Self:
        if self.supported and not self.evidence_ids:
            raise ValueError("a supported answer must cite stored evidence")
        return self


class ExtractedQuote(WireModel):
    """One candidate coach quote; Python verifies it is verbatim in the article."""

    speaker: TrimmedStr
    speaker_role: SpeakerRole
    text: TrimmedStr


class QuoteExtractionResponse(WireModel):
    """Candidate quotes proposed from one live article (never trusted unverified)."""

    quotes: list[ExtractedQuote]


STRUCTURED_OUTPUT_MODELS: Mapping[str, type[WireModel]] = {
    "agent_response": AgentResponse,
    "synthesis_response": SynthesisResponse,
    "verifier_response": VerifierResponse,
    "follow_up_response": FollowUpResponse,
    "quote_extraction_response": QuoteExtractionResponse,
}

_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def _resolve(model: type[BaseModel] | str) -> type[BaseModel]:
    if isinstance(model, str):
        try:
            return STRUCTURED_OUTPUT_MODELS[model]
        except KeyError as exc:
            known = ", ".join(sorted(STRUCTURED_OUTPUT_MODELS))
            raise KeyError(f"unknown structured output {model!r}; known: {known}") from exc
    return model


def json_schema_for(model: type[BaseModel] | str) -> dict[str, Any]:
    """Return a fresh JSON schema dict for a structured-output model."""

    schema = _resolve(model).model_json_schema()
    schema["$schema"] = _SCHEMA_DIALECT
    return schema


def structured_output_schemas() -> dict[str, dict[str, Any]]:
    """Return every schema keyed by the name used in ``response_format``."""

    return {name: json_schema_for(name) for name in STRUCTURED_OUTPUT_MODELS}


def write_json_schemas(directory: Path | str) -> tuple[Path, ...]:
    """Write deterministic schema files to ``directory`` and return their paths."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, schema in structured_output_schemas().items():
        path = target / f"{name}.schema.json"
        path.write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return tuple(paths)


__all__ = (
    "STRUCTURED_OUTPUT_MODELS",
    "AgentResponse",
    "ClaimVerdict",
    "DraftAgreement",
    "DraftConflict",
    "DraftConflictPosition",
    "DraftGap",
    "DraftObservation",
    "DraftTendency",
    "DraftWeakEvidence",
    "ExtractedQuote",
    "FollowUpResponse",
    "QuoteExtractionResponse",
    "SynthesisResponse",
    "VerifiedClaim",
    "VerifierResponse",
    "WireModel",
    "json_schema_for",
    "structured_output_schemas",
    "write_json_schemas",
)
