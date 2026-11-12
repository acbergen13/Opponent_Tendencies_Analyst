"""Offline tests for content-addressed report caching."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opponent_tendencies.cache import (
    CACHE_SCHEMA_VERSION,
    CacheKey,
    ReportCache,
    ReportCacheError,
    cache_key,
)
from opponent_tendencies.models import Team
from tests.unit.factories import make_report


def test_cache_round_trips_a_report(tmp_path: Path) -> None:
    report = make_report()
    cache = ReportCache(tmp_path)

    path = cache.store(report)
    loaded = cache.load(cache_key(report.opponent, report.weeks, "test-v1", report.data_revision))

    assert path.is_file()
    assert loaded == report


def test_cache_key_normalizes_weeks_and_team() -> None:
    first = cache_key(Team.BUF, [9, 5, 9], "v1", "rev")
    second = cache_key("BUF", [5, 9], "v1", "rev")

    assert first == second
    assert first.weeks == (5, 9)


def test_cache_key_changes_with_prompt_or_data_revision() -> None:
    first = cache_key(Team.BUF, [9], "v1", "rev")
    second = cache_key(Team.BUF, [9], "v2", "rev")
    third = cache_key(Team.BUF, [9], "v1", "new-rev")

    assert len({first.digest, second.digest, third.digest}) == 3
    assert ReportCache(Path("/tmp/unused")).path_for(first) != ReportCache(
        Path("/tmp/unused")
    ).path_for(second)


def test_cache_miss_is_none(tmp_path: Path) -> None:
    cache = ReportCache(tmp_path)
    assert cache.load(cache_key(Team.BUF, [9], "v1", "rev")) is None


def test_cache_rejects_corrupt_or_mismatched_entries(tmp_path: Path) -> None:
    cache = ReportCache(tmp_path)
    key = cache_key(Team.BUF, [9], "v1", "rev")
    path = cache.path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ReportCacheError, match="not valid JSON"):
        cache.load(key)

    path.write_text('{"schema_version": 999}', encoding="utf-8")
    with pytest.raises(ReportCacheError, match="schema_version"):
        cache.load(key)

    path.write_text(
        f'{{"schema_version": {CACHE_SCHEMA_VERSION}, "key": {{}}, "report": {{}}}}',
        encoding="utf-8",
    )
    with pytest.raises(ReportCacheError, match="does not match"):
        cache.load(key)


def test_cache_rejects_invalid_stored_report(tmp_path: Path) -> None:
    cache = ReportCache(tmp_path)
    key = cache_key(Team.BUF, [9], "v1", "rev")
    path = cache.path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            f'{{"schema_version": {CACHE_SCHEMA_VERSION}, '
            f'"key": {_key_json(key)}, "report": {{"opponent": "BUF"}}}}'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReportCacheError, match="failed contract validation"):
        cache.load(key)


def test_cache_key_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="week"):
        cache_key(Team.BUF, [], "v1", "rev")
    with pytest.raises(ValueError, match="prompt_version"):
        cache_key(Team.BUF, [1], "  ", "rev")
    with pytest.raises(ValueError, match="data_revision"):
        cache_key(Team.BUF, [1], "v1", "")


def _key_json(key: CacheKey) -> str:
    return json.dumps(
        {
            "opponent": key.opponent,
            "weeks": list(key.weeks),
            "prompt_version": key.prompt_version,
            "data_revision": key.data_revision,
        }
    )
