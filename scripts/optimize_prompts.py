"""Offline prompt optimization (requirement 18). Never used at runtime.

Usage::

    python scripts/optimize_prompts.py [--candidates N] [--prompts a,b] [--out PATH]

For every registered agent prompt this asks the VT ARC chat model to rewrite
the constitutional system prompt, checks each rewrite deterministically
(data-block tags, JSON rule, instruction hierarchy, length), and asks the
model to confirm the rewrite preserves the baseline constraints. Results --
including the selected prompt text per agent -- are stored under
``DATA_DIR/prompt_optimization/results.json``.

This is a design-time artifact. The runtime pipeline reads prompts only from
``opponent_tendencies.llm.prompts``; to adopt an accepted rewrite, copy it
into ``src/opponent_tendencies/llm/prompts/library.py`` and bump
``PROMPT_VERSION`` deliberately.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from opponent_tendencies.config import (
    ConfigurationError,
    get_settings,
    resolve_data_dir,
)
from opponent_tendencies.llm.client import ArcLLMClient
from opponent_tendencies.llm.optimize import run_optimization, write_results
from opponent_tendencies.llm.prompts import prompt_names


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=int,
        default=1,
        help="candidate rewrites per prompt (default: 1)",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="comma-separated prompt names (default: all prompts)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="results path (default: DATA_DIR/prompt_optimization/results.json)",
    )
    args = parser.parse_args(argv)

    if args.candidates < 1:
        parser.error("--candidates must be at least 1")

    names: list[str] | None = None
    if args.prompts is not None:
        names = [name.strip() for name in args.prompts.split(",") if name.strip()]
        unknown = [name for name in names if name not in prompt_names()]
        if unknown:
            parser.error(
                f"unknown prompt(s) {', '.join(unknown)}; known: {', '.join(prompt_names())}"
            )
        if not names:
            parser.error("--prompts must name at least one prompt")

    try:
        settings = get_settings()
    except ConfigurationError as exc:
        parser.error(str(exc))

    client = ArcLLMClient.from_settings(settings)
    results = run_optimization(client, names=names, n_candidates=args.candidates)

    default_out = resolve_data_dir() / "prompt_optimization" / "results.json"
    out_path = args.out if args.out is not None else default_out
    path = write_results(results, out_path)

    print(f"Model: {results.model}")
    print(f"Baseline prompt version: {results.baseline_prompt_version}")
    for optimization in results.prompts:
        invalid = sum(1 for review in optimization.candidates if not review.valid)
        print(
            f"{optimization.name}: selected {optimization.selected} "
            f"({len(optimization.candidates)} candidate(s), {invalid} invalid)"
        )
    print(f"Results: {path}")
    print(
        "Promote accepted rewrites into src/opponent_tendencies/llm/prompts/library.py "
        "and bump PROMPT_VERSION; the runtime never reads this file."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
