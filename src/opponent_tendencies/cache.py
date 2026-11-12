"""Content-addressed cache for finished reports.

A cached report is reusable only when the opponent, selected weeks, prompt
version, and combined season/media data revision all match. That tuple is the
cache key, so a data refresh or a prompt edit can never serve a stale report.

The cache stores the frozen :class:`~opponent_tendencies.models.Report` as
canonical JSON through the same atomic writer as the data stores. Corrupt or
mismatched entries fail loudly instead of being trusted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from opponent_tendencies.data.jsonio import atomic_write_text, canonical_dump
from opponent_tendencies.models import Report, Team

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CacheKey",
    "ReportCache",
    "ReportCacheError",
    "cache_key",
]

CACHE_SCHEMA_VERSION: Final = 1


class ReportCacheError(RuntimeError):
    """A cache entry exists but cannot be read as this schema version."""


@dataclass(frozen=True)
class CacheKey:
    """Everything a report depends on, normalized for a stable digest."""

    opponent: str
    weeks: tuple[int, ...]
    prompt_version: str
    data_revision: str

    @property
    def digest(self) -> str:
        payload = canonical_dump(
            {
                "opponent": self.opponent,
                "weeks": list(self.weeks),
                "prompt_version": self.prompt_version,
                "data_revision": self.data_revision,
            }
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_key(
    opponent: Team | str,
    weeks: Sequence[int],
    prompt_version: str,
    data_revision: str,
) -> CacheKey:
    """Build the documented ``(opponent, weeks, prompt_version, data_revision)`` key."""

    team = opponent.value if isinstance(opponent, Team) else str(opponent)
    selected = tuple(sorted({int(week) for week in weeks}))
    if not selected:
        raise ValueError("at least one week is required to key a report cache")
    if not prompt_version.strip():
        raise ValueError("prompt_version must not be blank")
    if not data_revision.strip():
        raise ValueError("data_revision must not be blank")
    return CacheKey(
        opponent=team,
        weeks=selected,
        prompt_version=prompt_version,
        data_revision=data_revision,
    )


class ReportCache:
    """Read and write finished reports in one directory."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def path_for(self, key: CacheKey) -> Path:
        return self.directory / f"{key.digest}.json"

    def load(self, key: CacheKey) -> Report | None:
        """Return a cached report, or ``None`` when the key is not stored."""

        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ReportCacheError(f"cannot read cached report {path}: {exc}") from exc
        try:
            envelope = json.loads(text)
        except ValueError as exc:
            raise ReportCacheError(f"cached report {path} is not valid JSON") from exc
        if not isinstance(envelope, dict):
            raise ReportCacheError(f"cached report {path} must contain a JSON object")
        if envelope.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ReportCacheError(
                f"cached report {path} has schema_version {envelope.get('schema_version')!r}; "
                f"expected {CACHE_SCHEMA_VERSION}"
            )
        if envelope.get("key") != _key_payload(key):
            raise ReportCacheError(f"cached report {path} does not match its requested key")
        try:
            return Report.model_validate(envelope.get("report"))
        except ValidationError as exc:
            raise ReportCacheError(f"cached report {path} failed contract validation") from exc

    def store(self, report: Report) -> Path:
        """Atomically store ``report`` and return its cache path."""

        key = cache_key(
            report.opponent,
            report.weeks,
            report.prompt_version,
            report.data_revision,
        )
        path = self.path_for(key)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "key": _key_payload(key),
            "report": report.model_dump(mode="json"),
        }
        atomic_write_text(path, canonical_dump(payload))
        return path


def _key_payload(key: CacheKey) -> dict[str, object]:
    return {
        "opponent": key.opponent,
        "weeks": list(key.weeks),
        "prompt_version": key.prompt_version,
        "data_revision": key.data_revision,
    }
