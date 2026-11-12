"""Specialist agents (STA, BRA, CQA), synthesis, and claim verification.

Stages 7–8 live here. Each specialist turns deterministic inputs into an
:class:`~opponent_tendencies.models.AgentOutput`; synthesis merges validated
outputs into report-level pieces; verification is a separate call, not a fifth
agent.
"""

from opponent_tendencies.agents.beat_reporter import (
    beat_reporter_query,
    run_beat_reporter_agent,
)
from opponent_tendencies.agents.coach_quote import coach_quote_query, run_coach_quote_agent
from opponent_tendencies.agents.common import AgentResult
from opponent_tendencies.agents.statistics import run_statistics_agent
from opponent_tendencies.agents.synthesis import SynthesisResult, run_synthesis_agent
from opponent_tendencies.agents.verify import (
    VerificationResult,
    VerifiedContent,
    apply_verification,
    build_verifier_claims,
    run_verifier,
)

__all__ = [
    "AgentResult",
    "SynthesisResult",
    "VerificationResult",
    "VerifiedContent",
    "apply_verification",
    "beat_reporter_query",
    "build_verifier_claims",
    "coach_quote_query",
    "run_beat_reporter_agent",
    "run_coach_quote_agent",
    "run_statistics_agent",
    "run_synthesis_agent",
    "run_verifier",
]
