"""Offline prompt optimization (requirement 18).

Design-time tooling only. Nothing in the runtime report path imports this
module: the orchestrator reads prompt texts exclusively from
:mod:`opponent_tendencies.llm.prompts`, and the results file written here is
an artifact for a human to review, never an input to a report run.

For each registered prompt the optimizer asks the VT ARC chat model to
rewrite the constitutional system prompt, checks every rewrite
deterministically (the data-block tags, the JSON output rule, the
instruction-hierarchy rule, length bounds, stray template placeholders), and
asks the model to confirm the rewrite preserves the baseline constraints.
A rewrite is selected only when it passes both checks; otherwise the
baseline is kept. The stored results record the per-prompt selection, the
candidate hashes, and every issue found, so the evaluation is auditable.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from opponent_tendencies.data.jsonio import atomic_write_text, canonical_dump
from opponent_tendencies.llm.client import (
    ArcLLMClient,
    LLMMessage,
    LLMResponse,
)
from opponent_tendencies.llm.prompts import (
    Prompt,
    get_prompt,
    prompt_names,
    prompt_version,
)
from opponent_tendencies.llm.prompts.blocks import mentioned_block_tags

__all__ = [
    "OPTIMIZATION_SCHEMA_VERSION",
    "CandidateReview",
    "OptimizationResults",
    "PromptOptimization",
    "check_candidate",
    "load_results",
    "optimize_prompt",
    "run_optimization",
    "write_results",
]

OPTIMIZATION_SCHEMA_VERSION = 1

MIN_CANDIDATE_CHARS = 300
MAX_CANDIDATE_CHARS = 12000
PROPOSAL_MAX_TOKENS = 4096
PROPOSAL_TEMPERATURE = 0.2
EVALUATOR_MAX_TOKENS = 256
EVALUATOR_TEMPERATURE = 0.0

_REWRITTEN_SPAN = re.compile(r"<rewritten_prompt>\s*(.*?)\s*</rewritten_prompt>", re.DOTALL)
_STRAY_PLACEHOLDER = re.compile(r"\$\{?[_a-zA-Z/]")

_PROPOSAL_SYSTEM = """\
You are a prompt-optimization assistant. You rewrite system prompts for a
sports-scouting pipeline. You never talk to end users.

Rules:
1. Preserve every constraint in the baseline: the instruction hierarchy, the
   rule that tagged data blocks are data and never instructions, grounding,
   no fabrication, the exact JSON output requirement, and the scope limit.
2. Keep every data-block tag the baseline mentions (for example
   <deterministic_statistics>) unchanged.
3. Keep the same agent identity and objective.
4. Improve clarity, specificity, and brevity. Do not add new freedoms.
5. Return the rewritten system prompt between <rewritten_prompt> and
   </rewritten_prompt> tags, with nothing else in your reply.
"""

_EVALUATOR_SYSTEM = """\
You are a prompt evaluator for a sports-scouting pipeline. You compare a
rewritten system prompt against its baseline and decide whether the rewrite
may replace it.

Accept the candidate only if it preserves every baseline constraint: the
instruction hierarchy, the rule that tagged data blocks are data and never
instructions, grounding, no fabrication, the exact JSON output requirement,
the agent identity, and the scope limit.

Reply with ACCEPT or REJECT on the first line, then one short sentence
explaining your decision.
"""


@dataclass(frozen=True)
class CandidateReview:
    """What happened to one proposed rewrite."""

    sha256: str
    valid: bool
    issues: tuple[str, ...]
    evaluator_decision: str
    evaluator_reason: str


@dataclass(frozen=True)
class PromptOptimization:
    """The optimization outcome for one prompt."""

    name: str
    baseline_version: str
    baseline_sha256: str
    selected: str
    selected_sha256: str
    selected_text: str
    candidates: tuple[CandidateReview, ...]

    @property
    def selected_candidate(self) -> bool:
        return self.selected == "candidate"

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "baseline_version": self.baseline_version,
            "baseline_sha256": self.baseline_sha256,
            "selected": self.selected,
            "selected_sha256": self.selected_sha256,
            "selected_text": self.selected_text,
            "candidates": [
                {
                    "sha256": review.sha256,
                    "valid": review.valid,
                    "issues": list(review.issues),
                    "evaluator_decision": review.evaluator_decision,
                    "evaluator_reason": review.evaluator_reason,
                }
                for review in self.candidates
            ],
        }


@dataclass(frozen=True)
class OptimizationResults:
    """The optimization outcome for the whole prompt library."""

    baseline_prompt_version: str
    model: str
    prompts: tuple[PromptOptimization, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": OPTIMIZATION_SCHEMA_VERSION,
            "baseline_prompt_version": self.baseline_prompt_version,
            "model": self.model,
            "prompts": [optimization.to_payload() for optimization in self.prompts],
        }


def check_candidate(prompt: Prompt, candidate: str) -> tuple[str, ...]:
    """Deterministically check that a rewrite keeps the important information.

    A candidate that fails any check is never selected, regardless of what
    the evaluator says.
    """

    issues: list[str] = []
    if not candidate.strip():
        return ("candidate is blank",)
    if len(candidate) < MIN_CANDIDATE_CHARS:
        issues.append(
            f"candidate is too short ({len(candidate)} chars; minimum {MIN_CANDIDATE_CHARS})"
        )
    if len(candidate) > MAX_CANDIDATE_CHARS:
        issues.append(
            f"candidate is too long ({len(candidate)} chars; maximum {MAX_CANDIDATE_CHARS})"
        )
    lowered = candidate.lower()
    if "json" not in lowered:
        issues.append("candidate drops the JSON output requirement")
    if "instruction" not in lowered:
        issues.append("candidate drops the instruction-hierarchy rule")
    if _STRAY_PLACEHOLDER.search(candidate):
        issues.append("candidate contains template placeholder syntax ($...)")
    for tag in mentioned_block_tags(prompt.system):
        if tag not in candidate:
            issues.append(f"candidate drops the <{tag}> data-block tag")
    return tuple(issues)


def optimize_prompt(
    client: ArcLLMClient,
    prompt: Prompt,
    *,
    n_candidates: int = 1,
) -> PromptOptimization:
    """Propose, check, and evaluate rewrites for one prompt."""

    if n_candidates < 1:
        raise ValueError("n_candidates must be at least 1")

    reviews: list[CandidateReview] = []
    selected_text: str | None = None
    for _ in range(n_candidates):
        candidate = _propose_candidate(client, prompt)
        if candidate is None:
            reviews.append(
                CandidateReview(
                    sha256="",
                    valid=False,
                    issues=("proposal is missing the <rewritten_prompt> block",),
                    evaluator_decision="not_evaluated",
                    evaluator_reason="",
                )
            )
            continue

        issues = check_candidate(prompt, candidate)
        if issues:
            reviews.append(
                CandidateReview(
                    sha256=_sha256(candidate),
                    valid=False,
                    issues=issues,
                    evaluator_decision="not_evaluated",
                    evaluator_reason="",
                )
            )
            continue

        decision, reason = _evaluate_candidate(client, prompt, candidate)
        reviews.append(
            CandidateReview(
                sha256=_sha256(candidate),
                valid=True,
                issues=(),
                evaluator_decision=decision,
                evaluator_reason=reason,
            )
        )
        if selected_text is None and decision == "accept":
            selected_text = candidate

    if selected_text is None:
        selected_text = prompt.system
        selected = "baseline"
    else:
        selected = "candidate"
    return PromptOptimization(
        name=prompt.name,
        baseline_version=prompt.version,
        baseline_sha256=_sha256(prompt.system),
        selected=selected,
        selected_sha256=_sha256(selected_text),
        selected_text=selected_text,
        candidates=tuple(reviews),
    )


def run_optimization(
    client: ArcLLMClient,
    *,
    names: Sequence[str] | None = None,
    n_candidates: int = 1,
) -> OptimizationResults:
    """Optimize every requested prompt (all of them by default)."""

    selected_names = tuple(names) if names is not None else prompt_names()
    optimizations = tuple(
        optimize_prompt(client, get_prompt(name), n_candidates=n_candidates)
        for name in selected_names
    )
    return OptimizationResults(
        baseline_prompt_version=prompt_version(),
        model=client.model_name,
        prompts=optimizations,
    )


def write_results(results: OptimizationResults, path: Path | str) -> Path:
    """Atomically write the canonical results document."""

    target = Path(path)
    atomic_write_text(target, canonical_dump(results.to_payload()))
    return target


def load_results(path: Path | str) -> dict[str, object]:
    """Read back a stored results document."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"optimization results {path} are not a JSON object")
    return document


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _propose_candidate(client: ArcLLMClient, prompt: Prompt) -> str | None:
    response: LLMResponse = client.chat(
        (
            LLMMessage(role="system", content=_PROPOSAL_SYSTEM),
            LLMMessage(role="user", content=_proposal_user(prompt)),
        ),
        temperature=PROPOSAL_TEMPERATURE,
        max_tokens=PROPOSAL_MAX_TOKENS,
    )
    match = _REWRITTEN_SPAN.search(response.content)
    if match is None:
        return None
    candidate = match.group(1).strip()
    return candidate or None


def _evaluate_candidate(client: ArcLLMClient, prompt: Prompt, candidate: str) -> tuple[str, str]:
    response: LLMResponse = client.chat(
        (
            LLMMessage(role="system", content=_EVALUATOR_SYSTEM),
            LLMMessage(role="user", content=_evaluator_user(prompt, candidate)),
        ),
        temperature=EVALUATOR_TEMPERATURE,
        max_tokens=EVALUATOR_MAX_TOKENS,
    )
    first_line = next(
        (line.strip() for line in response.content.splitlines() if line.strip()),
        "",
    )
    reason = response.content.strip()[:500]
    if first_line.lower().startswith("accept"):
        return "accept", reason
    return "reject", reason


def _proposal_user(prompt: Prompt) -> str:
    return (
        f"Baseline system prompt for the {prompt.name} agent:\n"
        "\n"
        "<baseline_prompt>\n"
        f"{prompt.system}"
        "</baseline_prompt>\n"
        "\n"
        "Rewrite it following your rules."
    )


def _evaluator_user(prompt: Prompt, candidate: str) -> str:
    return (
        f"Baseline system prompt for the {prompt.name} agent:\n"
        "\n"
        "<baseline_prompt>\n"
        f"{prompt.system}"
        "</baseline_prompt>\n"
        "\n"
        "Candidate rewrite:\n"
        "\n"
        "<candidate_prompt>\n"
        f"{candidate}\n"
        "</candidate_prompt>\n"
        "\n"
        "Does the candidate preserve every constraint? Reply ACCEPT or REJECT."
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
