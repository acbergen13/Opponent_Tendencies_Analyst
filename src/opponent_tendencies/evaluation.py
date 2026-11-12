"""Deterministic consistency checks for repeated report runs (Stage 11).

The live model is allowed to vary its wording, but the acceptance criteria are
mechanical: the top-five tendencies should overlap by at least 80% in at least
four of five runs, no tendency may flip direction between runs, deterministic
stat tables must be byte-identical, and every cited quote must resolve to the
stored quote text.

All comparisons run on frozen contracts; no model call happens here.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from opponent_tendencies.data.jsonio import canonical_dump
from opponent_tendencies.data.media_store import MediaData
from opponent_tendencies.models import EvidenceKind, Report, Tendency

__all__ = [
    "DEFAULT_MIN_OVERLAP",
    "DEFAULT_RUNS",
    "DEFAULT_TOP_N",
    "ConsistencyResult",
    "evaluate_consistency",
    "tendency_key",
    "top_tendencies",
]

DEFAULT_RUNS: Final = 5
DEFAULT_TOP_N: Final = 5
DEFAULT_MIN_OVERLAP: Final = 0.8


@dataclass(frozen=True)
class ConsistencyResult:
    """One consistency evaluation over repeated reports."""

    runs: int
    top_n: int
    consensus: tuple[str, ...]
    overlaps: tuple[float, ...]
    passing_runs: int
    required_passing_runs: int
    direction_contradictions: tuple[str, ...]
    stats_identical: bool
    quote_traceability: float
    stats_traceability: float

    @property
    def passed(self) -> bool:
        return (
            self.passing_runs >= self.required_passing_runs
            and not self.direction_contradictions
            and self.stats_identical
            and self.quote_traceability == 1.0
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "runs": self.runs,
            "top_n": self.top_n,
            "consensus": list(self.consensus),
            "overlaps": list(self.overlaps),
            "passing_runs": self.passing_runs,
            "required_passing_runs": self.required_passing_runs,
            "direction_contradictions": list(self.direction_contradictions),
            "stats_identical": self.stats_identical,
            "quote_traceability": self.quote_traceability,
            "stats_traceability": self.stats_traceability,
            "passed": self.passed,
        }


def tendency_key(tendency: Tendency) -> str:
    """Stable comparison key: category plus whitespace/case-normalized statement."""

    statement = " ".join(tendency.statement.lower().split())
    return f"{tendency.category.value}|{statement}"


def top_tendencies(report: Report, n: int = DEFAULT_TOP_N) -> tuple[str, ...]:
    """Return the first ``n`` distinct tendency keys in report order."""

    if n < 1:
        raise ValueError("n must be at least 1")
    keys: list[str] = []
    seen: set[str] = set()
    for tendency in report.tendencies:
        key = tendency_key(tendency)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
        if len(keys) == n:
            break
    return tuple(keys)


def evaluate_consistency(
    reports: Sequence[Report],
    *,
    media_data: MediaData | None = None,
    top_n: int = DEFAULT_TOP_N,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> ConsistencyResult:
    """Compare repeated reports against a deterministic consensus top list."""

    if not reports:
        raise ValueError("at least one report is required")
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap must be between 0 and 1")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    tops = [top_tendencies(report, top_n) for report in reports]
    counts: Counter[str] = Counter(key for top in tops for key in top)
    consensus = tuple(
        key for key, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    )
    denominator = max(1, min(top_n, len(consensus)))
    overlaps = tuple(len(set(top) & set(consensus)) / denominator if top else 0.0 for top in tops)
    passing = sum(1 for overlap in overlaps if overlap >= min_overlap)
    required = max(1, math.ceil(min_overlap * len(reports)))

    directions: dict[str, set[str]] = defaultdict(set)
    for report in reports:
        for tendency in report.tendencies:
            directions[tendency_key(tendency)].add(tendency.direction.value)
    contradictions = tuple(sorted(key for key, values in directions.items() if len(values) > 1))

    stats_identical = _stats_identical(reports)
    quote_traceability, stats_traceability = _traceability(reports, media_data=media_data)

    return ConsistencyResult(
        runs=len(reports),
        top_n=top_n,
        consensus=consensus,
        overlaps=overlaps,
        passing_runs=passing,
        required_passing_runs=required,
        direction_contradictions=contradictions,
        stats_identical=stats_identical,
        quote_traceability=quote_traceability,
        stats_traceability=stats_traceability,
    )


def _stats_identical(reports: Sequence[Report]) -> bool:
    if not reports:
        return True
    dumps = [
        canonical_dump({"tables": [table.model_dump(mode="json") for table in report.stat_tables]})
        for report in reports
    ]
    return all(dump == dumps[0] for dump in dumps[1:])


def _traceability(
    reports: Sequence[Report], *, media_data: MediaData | None
) -> tuple[float, float]:
    quotes_total = 0
    quotes_traceable = 0
    stats_total = 0
    stats_traceable = 0
    for report in reports:
        table_ids = {table.stat_table_id for table in report.stat_tables}
        for item in report.evidence:
            if item.kind is EvidenceKind.QUOTE:
                quotes_total += 1
                if media_data is not None and _quote_resolves(
                    report, item.quote_id, item.snippet, media_data
                ):
                    quotes_traceable += 1
            elif item.kind is EvidenceKind.STAT:
                stats_total += 1
                if item.stat_table_id in table_ids:
                    stats_traceable += 1
    quote_ratio = 1.0 if quotes_total == 0 else quotes_traceable / quotes_total
    stat_ratio = 1.0 if stats_total == 0 else stats_traceable / stats_total
    return quote_ratio, stat_ratio


def _quote_resolves(
    report: Report,
    quote_id: str | None,
    snippet: str | None,
    media_data: MediaData,
) -> bool:
    if quote_id is None:
        return False
    quote = media_data.quote_by_id(quote_id)
    if quote is None:
        return False
    if snippet is not None and quote.text != snippet:
        return False
    return quote.team is report.opponent and quote.season == report.season


def iter_tendency_keys(reports: Iterable[Report]) -> Iterable[str]:
    """Convenience iterator used by callers that only need the comparison keys."""

    for report in reports:
        for tendency in report.tendencies:
            yield tendency_key(tendency)
