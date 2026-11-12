"""Versioned constitutional prompts and the deterministic data-block renderers.

Public API:

* :func:`get_prompt` / :func:`all_prompts` / :func:`prompt_names` -- the six
  versioned prompts (STA, BRA, CQA, SYN, verifier, follow-up).
* :func:`prompt_version` -- the version stamped onto reports.
* The ``render_*`` helpers from :mod:`opponent_tendencies.llm.prompts.blocks`
  build the delimited, provenance-labeled data blocks the templates reference.
"""

from opponent_tendencies.llm.prompts.blocks import (
    BLOCK_TAGS,
    ClaimSnippet,
    VerifierClaim,
    mentioned_block_tags,
    neutralize_tags,
    render_agent_findings,
    render_evidence_catalog,
    render_games_summary,
    render_report_context,
    render_retrieved_sources,
    render_stat_tables,
    render_verifier_claims,
)
from opponent_tendencies.llm.prompts.library import (
    PROMPT_NAMES,
    PROMPT_VERSION,
    Prompt,
    RenderedPrompt,
    all_prompts,
    get_prompt,
    prompt_names,
    prompt_version,
    template_placeholders,
)

__all__ = [
    "BLOCK_TAGS",
    "PROMPT_NAMES",
    "PROMPT_VERSION",
    "ClaimSnippet",
    "Prompt",
    "RenderedPrompt",
    "VerifierClaim",
    "all_prompts",
    "get_prompt",
    "mentioned_block_tags",
    "neutralize_tags",
    "prompt_names",
    "prompt_version",
    "render_agent_findings",
    "render_evidence_catalog",
    "render_games_summary",
    "render_report_context",
    "render_retrieved_sources",
    "render_stat_tables",
    "render_verifier_claims",
    "template_placeholders",
]
