"""Load the cached nflverse fixtures shared by unit and integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
SEASON = 2026
PBP_FIXTURE = "nflverse_pbp_2026_sample.json"
SCHEDULE_FIXTURE = "nflverse_schedules_2026_sample.json"


def load_document(name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8")))


def pbp_rows() -> list[dict[str, Any]]:
    """Return a deep-enough copy of the cached play-by-play rows."""

    document = load_document(PBP_FIXTURE)
    return [dict(row) for row in cast(list[dict[str, Any]], document["rows"])]


def schedule_rows() -> list[dict[str, Any]]:
    document = load_document(SCHEDULE_FIXTURE)
    return [dict(row) for row in cast(list[dict[str, Any]], document["rows"])]
