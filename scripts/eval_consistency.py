"""Run the same report several times and check narrative consistency.

Usage::

    python scripts/eval_consistency.py --opponent BUF --weeks 5 6 7 8 9

This is a **live** evaluation: it needs VT ARC credentials, the stored season
and media data, and the built Chroma collection. It is never imported by the
app or the agents. The deterministic checks live in
:mod:`opponent_tendencies.evaluation`; this script only drives repeated runs and
prints the JSON result (optionally written with ``--output``).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from opponent_tendencies.config import ConfigurationError, get_settings
from opponent_tendencies.data.index import MediaIndex
from opponent_tendencies.data.jsonio import atomic_write_text
from opponent_tendencies.data.media_store import MediaStore
from opponent_tendencies.data.play_store import SeasonStore
from opponent_tendencies.evaluation import DEFAULT_RUNS, evaluate_consistency
from opponent_tendencies.llm.client import ArcLLMClient
from opponent_tendencies.llm.embeddings import ArcEmbedder
from opponent_tendencies.models import Team
from opponent_tendencies.orchestrator import run_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", required=True, help="team abbreviation, e.g. BUF")
    parser.add_argument("--weeks", nargs="+", type=int, required=True, help="selected weeks")
    parser.add_argument("--season", type=int, default=2026, help="season (default: 2026)")
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"repeated runs to compare (default: {DEFAULT_RUNS})",
    )
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output path")
    args = parser.parse_args(argv)

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    try:
        opponent = Team(args.opponent.upper())
    except ValueError:
        parser.error(f"unknown team abbreviation: {args.opponent}")
    weeks = sorted({int(week) for week in args.weeks})
    if not weeks:
        parser.error("--weeks must contain at least one week")

    try:
        settings = get_settings()
    except ConfigurationError as exc:
        parser.error(str(exc))

    season = int(args.season)
    season_store = SeasonStore(season, settings.data_dir / "seasons" / f"{season}.json")
    media_store = MediaStore(season, settings.data_dir / "media" / f"{season}.json")
    season_data = season_store.load()
    media_data = media_store.load()

    embedder = ArcEmbedder(
        settings.vt_arc_base_url,
        settings.vt_arc_api_key.get_secret_value(),
        settings.embedding_model,
    )
    index = MediaIndex(
        settings.chroma_dir,
        embedder,
        collection_name=f"media_{season}",
    )
    if index.count == 0 and media_data.media:
        print("Index is empty; building it before evaluation…")
        print(index.build(media_data).summary())

    client = ArcLLMClient.from_settings(settings)
    reports = []
    for index_number in range(1, args.runs + 1):
        print(f"Run {index_number}/{args.runs}…")
        run = run_report(
            opponent=opponent,
            weeks=weeks,
            season_data=season_data,
            media_data=media_data,
            index=index,
            client=client,
        )
        if run.report is None:
            print(
                f"Run {index_number} failed: {run.status.failure_reason.value} — "
                f"{run.status.message}"
            )
            return 2
        reports.append(run.report)

    result = evaluate_consistency(reports, media_data=media_data)
    payload = {
        "opponent": opponent.value,
        "season": season,
        "weeks": weeks,
        **result.as_dict(),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text)
    if args.output is not None:
        atomic_write_text(args.output, text)
        print(f"Wrote {args.output}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
