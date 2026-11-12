"""Command-line runner for the full report pipeline.

Usage::

    python scripts/report.py --opponent BUF --weeks 5 6 7 8 9

This is the non-Streamlit entry point recommended by ``PLAN.md``. It loads the
same stores, pinned embedder, and LLM client as the app, calls
:func:`opponent_tendencies.ui.runner.build_report`, prints a deterministic
summary, and can write the frozen ``Report`` as JSON. The pipeline logic stays in
the orchestrator; this module only parses arguments, calls it, and formats the
result. It is never imported by the app or the agents.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from opponent_tendencies.config import (
    ConfigurationError,
    get_settings,
    resolve_data_dir,
)
from opponent_tendencies.data.jsonio import atomic_write_text
from opponent_tendencies.data.play_store import SeasonStore, default_store_path
from opponent_tendencies.models import RunFailureReason, Team
from opponent_tendencies.orchestrator import (
    DEFAULT_AGENT_TIMEOUT_SECONDS,
    DEFAULT_DEADLINE_SECONDS,
    DEFAULT_MAX_LLM_CALLS,
    DEFAULT_TOKEN_BUDGET,
    ReportRun,
    RunProgress,
)
from opponent_tendencies.stats.engine import (
    DEFAULT_MIN_SAMPLE,
    StatsBundle,
    analyze_opponent,
)
from opponent_tendencies.ui.runner import DEFAULT_SEASON, build_report
from opponent_tendencies.validation.checks import validate_selection

__all__ = [
    "build_parser",
    "main",
    "render_stats_summary",
    "render_summary",
    "run_payload",
]


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser (kept separate so tests can inspect it)."""

    parser = argparse.ArgumentParser(
        description="Build one evidence-backed opponent tendencies report.",
    )
    parser.add_argument("--opponent", required=True, help="team abbreviation, e.g. BUF")
    parser.add_argument("--weeks", nargs="+", type=int, required=True, help="selected weeks")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON, help="season (default: 2026)")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="print the deterministic statistics only; no credentials or index needed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the full report and telemetry as JSON to this path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the JSON payload instead of the human-readable summary",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="always run the pipeline; never reuse a stored report",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print per-stage progress to stderr",
    )
    parser.add_argument(
        "--min-sample",
        type=int,
        default=DEFAULT_MIN_SAMPLE,
        help=f"suppress stat rows below this play count (default: {DEFAULT_MIN_SAMPLE})",
    )
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=DEFAULT_DEADLINE_SECONDS,
        help=f"overall run deadline (default: {DEFAULT_DEADLINE_SECONDS:g})",
    )
    parser.add_argument(
        "--agent-timeout-seconds",
        type=float,
        default=DEFAULT_AGENT_TIMEOUT_SECONDS,
        help=f"per-stage timeout (default: {DEFAULT_AGENT_TIMEOUT_SECONDS:g})",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=DEFAULT_MAX_LLM_CALLS,
        help=f"step cap across the pipeline (default: {DEFAULT_MAX_LLM_CALLS})",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
        help=f"token budget across the pipeline (default: {DEFAULT_TOKEN_BUDGET})",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the CLI and return a process exit code."""

    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        opponent = Team(args.opponent.upper())
    except ValueError:
        parser.error(f"unknown team abbreviation: {args.opponent}")

    weeks = tuple(sorted({int(week) for week in args.weeks}))
    if not weeks:
        parser.error("--weeks must contain at least one week")
    if any(week < 1 for week in weeks):
        parser.error("--weeks must be positive regular-season weeks")
    if args.min_sample < 1:
        parser.error("--min-sample must be at least 1")
    if args.deadline_seconds <= 0:
        parser.error("--deadline-seconds must be positive")
    if args.agent_timeout_seconds <= 0:
        parser.error("--agent-timeout-seconds must be positive")
    if args.max_llm_calls < 1:
        parser.error("--max-llm-calls must be at least 1")
    if args.token_budget < 1:
        parser.error("--token-budget must be at least 1")

    if args.stats_only:
        return _stats_only(
            opponent=opponent,
            weeks=weeks,
            season=int(args.season),
            min_sample=int(args.min_sample),
            output=args.output,
            as_json=bool(args.json),
            out=out,
            err=err,
        )

    try:
        settings = get_settings()
    except ConfigurationError as exc:
        parser.error(str(exc))

    progress = None if args.quiet else _progress_printer(err)
    run = build_report(
        opponent=opponent,
        weeks=weeks,
        season=int(args.season),
        data_dir=settings.data_dir,
        chroma_dir=settings.chroma_dir,
        on_progress=progress,
        min_sample=int(args.min_sample),
        deadline_seconds=float(args.deadline_seconds),
        agent_timeout_seconds=float(args.agent_timeout_seconds),
        max_llm_calls=int(args.max_llm_calls),
        token_budget=int(args.token_budget),
        use_cache=not args.no_cache,
    )

    payload = run_payload(run)
    if args.output is not None:
        atomic_write_text(args.output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.output}", file=out)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True), file=out)
    else:
        print(render_summary(run), file=out)

    return 0 if run.report is not None else 1


def render_summary(run: ReportRun) -> str:
    """Render a deterministic human-readable summary of one run."""

    status = run.status
    lines = [f"Status: {status.state.value}"]
    if status.failure_reason is not RunFailureReason.NONE:
        lines.append(f"Failure: {status.failure_reason.value}")
    if status.message:
        lines.append(f"Message: {status.message}")

    report = run.report
    if report is not None:
        lines.extend(
            [
                f"Report: {report.report_id}",
                f"Opponent: {report.opponent.value} ({report.season})",
                f"Weeks: {', '.join(str(week) for week in report.weeks)}",
                (
                    f"Games: {report.data_used.games} | "
                    f"Offensive plays: {report.data_used.plays} | "
                    f"Media items: {report.data_used.media_items} | "
                    f"Quotes: {report.data_used.quotes}"
                ),
                (
                    f"Tendencies: {len(report.tendencies)} | "
                    f"Observations: {len(report.observations)} | "
                    f"Agreements: {len(report.agreements)} | "
                    f"Conflicts: {len(report.conflicts)} | "
                    f"Gaps: {len(report.gaps)} | "
                    f"Weak evidence: {len(report.weak_evidence)}"
                ),
                f"Prompt: {report.prompt_version} | Model: {report.model or 'unknown'}",
                f"Overview: {_clip(report.overview, 400)}",
            ]
        )

    lines.extend(
        [
            f"Report cache: {'hit' if run.cached else 'miss'}",
            (
                f"LLM calls: {run.llm_calls} | "
                f"Tokens: {run.usage.total_tokens} "
                f"(prompt {run.usage.prompt_tokens}, completion {run.usage.completion_tokens})"
            ),
            f"Latency: {run.latency_seconds:.2f}s",
            f"Warnings: {len(run.warnings)}",
        ]
    )
    return "\n".join(lines)


def run_payload(run: ReportRun) -> dict[str, object]:
    """Return the stable JSON document for one run (report plus telemetry)."""

    return {
        "cached": run.cached,
        "latency_seconds": round(run.latency_seconds, 3),
        "llm_calls": run.llm_calls,
        "opponent": run.report.opponent.value if run.report is not None else None,
        "report": None if run.report is None else run.report.model_dump(mode="json"),
        "season": run.report.season if run.report is not None else None,
        "status": run.status.model_dump(mode="json"),
        "usage": {
            "completion_tokens": run.usage.completion_tokens,
            "prompt_tokens": run.usage.prompt_tokens,
            "total_tokens": run.usage.total_tokens,
        },
        "warnings": [issue.model_dump(mode="json") for issue in run.warnings],
        "weeks": None if run.report is None else list(run.report.weeks),
    }


def render_stats_summary(bundle: StatsBundle, season: int) -> str:
    """Render the deterministic statistics bundle without any model output."""

    quality = [
        f"{name}={value}"
        for name, value in bundle.quality.to_payload().items()
        if value and name != "offensive_plays"
    ]
    lines = [
        f"Opponent: {bundle.opponent.value} ({season})",
        f"Weeks: {', '.join(str(week) for week in bundle.weeks)}",
        (
            f"Games: {len(bundle.game_ids)} | "
            f"Offensive plays: {bundle.quality.offensive_plays} | "
            f"Min sample: {bundle.min_sample}"
        ),
    ]
    for table in bundle.tables:
        suppressed = sum(1 for row in table.rows if row.suppressed)
        lines.append(
            f"{table.title}: {len(table.rows)} rows ({suppressed} suppressed, n={table.n_plays})"
        )
    if quality:
        lines.append(f"Data quality: {', '.join(quality)}")
    return "\n".join(lines)


def _stats_only(
    *,
    opponent: Team,
    weeks: tuple[int, ...],
    season: int,
    min_sample: int,
    output: Path | None,
    as_json: bool,
    out: TextIO,
    err: TextIO,
) -> int:
    data_dir = resolve_data_dir()
    state = SeasonStore(season, default_store_path(season, data_dir)).load()
    selection = validate_selection(state, opponent=opponent, weeks=weeks)
    if not selection.ok:
        print("The selection is not playable:", file=err)
        for issue in selection.errors:
            print(f"- {issue.message}", file=err)
        return 1

    games = tuple(game for game in state.games_for_weeks(weeks) if game.involves(opponent))
    if not games:
        print(
            f"{opponent.value} has no stored regular-season games in the selected weeks.",
            file=err,
        )
        return 1

    try:
        bundle = analyze_opponent(state.plays, games, opponent=opponent, min_sample=min_sample)
    except Exception as exc:
        print(f"The deterministic statistics could not be computed: {exc}", file=err)
        return 1

    payload = bundle.to_payload()
    if output is not None:
        atomic_write_text(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {output}", file=out)
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True), file=out)
    else:
        print(render_stats_summary(bundle, season), file=out)
    return 0


def _progress_printer(stream: TextIO) -> Callable[[RunProgress], None]:
    def report(progress: RunProgress) -> None:
        print(f"[{progress.step}] {progress.status.value}", file=stream, flush=True)

    return report


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."
