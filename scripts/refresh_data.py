"""Refresh the JSON play store from nflverse.

Usage::

    python scripts/refresh_data.py --season 2026

The store path comes from ``DATA_DIR`` (default ``data/``); see
:func:`opponent_tendencies.data.play_store.default_store_path`. No LLM
credentials are required.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from opponent_tendencies.data.nflverse import NflverseClient, current_season
from opponent_tendencies.data.play_store import SeasonStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="nflverse season to ingest (default: current season)",
    )
    args = parser.parse_args(argv)

    season = int(args.season) if args.season is not None else current_season()
    store = SeasonStore(season)
    report = NflverseClient(season).refresh(store)

    print(f"Wrote {store.path}")
    print(report.summary())
    print(f"Skipped deleted plays: {report.plays.skipped_deleted}")
    print(f"Skipped non-regular plays: {report.plays.skipped_non_regular}")
    print(f"Unknown play types (kept as 'unknown'): {report.plays.unknown_play_types}")

    state = store.load()
    print(f"Played regular-season weeks: {list(state.played_weeks)}")
    print(f"Teams: {', '.join(team.value for team in state.teams)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
