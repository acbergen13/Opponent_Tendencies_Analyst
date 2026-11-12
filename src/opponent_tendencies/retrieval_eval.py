"""Deterministic chunking and retrieval evaluation on an opponent sample (Stage 11).

The evaluator answers two questions before a report run is trusted:

* Does the configured chunking cover every stored body without hard cuts or
  duplicate chunks?
* Does filtered retrieval return only stored, opponent-matched, week-aligned
  sources for the same deterministic queries the agents use?

Everything here is plain Python and the checks reuse
:func:`opponent_tendencies.validation.checks.check_retrieval`; no model is
called. :mod:`scripts.eval_retrieval` wraps it for offline (hashing embedder) or
live (pinned VT ARC embedder) use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from opponent_tendencies.agents.beat_reporter import beat_reporter_query
from opponent_tendencies.agents.coach_quote import coach_quote_query
from opponent_tendencies.data.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_text,
    evaluate_chunking,
)
from opponent_tendencies.data.index import MediaIndex
from opponent_tendencies.data.media_store import MediaData
from opponent_tendencies.models import MediaType, Team
from opponent_tendencies.validation.checks import check_retrieval

__all__ = [
    "ChunkingEvaluation",
    "RetrievalEvaluation",
    "RetrievalSample",
    "evaluate_corpus_chunking",
    "evaluate_retrieval",
]


@dataclass(frozen=True)
class ChunkingEvaluation:
    """Aggregate chunk quality across every stored media body."""

    documents: int
    chunks: int
    total_chars: int
    covered_chars: int
    hard_cuts: int
    duplicate_texts: int
    max_chunk_chars: int

    @property
    def coverage(self) -> float:
        if self.total_chars == 0:
            return 1.0
        return self.covered_chars / self.total_chars

    @property
    def passed(self) -> bool:
        return self.coverage == 1.0 and self.hard_cuts == 0 and self.duplicate_texts == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "chunks": self.chunks,
            "coverage": round(self.coverage, 6),
            "documents": self.documents,
            "duplicate_texts": self.duplicate_texts,
            "hard_cuts": self.hard_cuts,
            "max_chunk_chars": self.max_chunk_chars,
            "passed": self.passed,
            "total_chars": self.total_chars,
        }


@dataclass(frozen=True)
class RetrievalSample:
    """One deterministic retrieval query and its validation outcome."""

    name: str
    query: str
    source_types: tuple[str, ...]
    kinds: tuple[Literal["chunk", "quote"], ...]
    expected: bool
    retrieved: int
    aligned: int
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        if self.errors:
            return False
        return self.aligned > 0 or not self.expected

    def as_dict(self) -> dict[str, object]:
        return {
            "aligned": self.aligned,
            "errors": list(self.errors),
            "expected": self.expected,
            "kinds": list(self.kinds),
            "name": self.name,
            "passed": self.passed,
            "query": self.query,
            "retrieved": self.retrieved,
            "source_types": list(self.source_types),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class RetrievalEvaluation:
    """The retrieval samples for one opponent/week selection."""

    opponent: Team
    season: int
    weeks: tuple[int, ...]
    samples: tuple[RetrievalSample, ...]

    @property
    def passed(self) -> bool:
        return bool(self.samples) and all(sample.passed for sample in self.samples)

    def as_dict(self) -> dict[str, object]:
        return {
            "opponent": self.opponent.value,
            "passed": self.passed,
            "samples": [sample.as_dict() for sample in self.samples],
            "season": self.season,
            "weeks": list(self.weeks),
        }


def evaluate_corpus_chunking(
    media: MediaData,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> ChunkingEvaluation:
    """Measure coverage and boundary quality for every stored media body."""

    documents = 0
    chunks = 0
    total_chars = 0
    covered_chars = 0
    hard_cuts = 0
    duplicate_texts = 0
    max_chunk_chars = 0

    for item in media.media:
        spans = chunk_text(item.body, size=chunk_size, overlap=chunk_overlap)
        report = evaluate_chunking(item.body, spans)
        documents += 1
        chunks += report.chunk_count
        total_chars += report.total_chars
        covered_chars += report.covered_chars
        hard_cuts += report.hard_cuts
        duplicate_texts += report.duplicate_texts
        max_chunk_chars = max(max_chunk_chars, report.max_chars)

    return ChunkingEvaluation(
        documents=documents,
        chunks=chunks,
        total_chars=total_chars,
        covered_chars=covered_chars,
        hard_cuts=hard_cuts,
        duplicate_texts=duplicate_texts,
        max_chunk_chars=max_chunk_chars,
    )


def evaluate_retrieval(
    index: MediaIndex,
    media: MediaData,
    *,
    opponent: Team,
    season: int,
    weeks: Sequence[int],
    k: int = 8,
) -> RetrievalEvaluation:
    """Run the BRA/CQA retrieval queries and validate every returned unit."""

    selected_weeks = tuple(sorted({int(week) for week in weeks}))
    if not selected_weeks:
        raise ValueError("weeks must contain at least one week")
    if k <= 0:
        raise ValueError("k must be positive")

    samples = (
        _sample(
            index,
            media,
            name="beat_reporter",
            query=beat_reporter_query(opponent, selected_weeks),
            source_types=(MediaType.ARTICLE, MediaType.FILM_BREAKDOWN),
            kinds=("chunk",),
            expected=bool(
                media.media_for(
                    opponent=opponent,
                    weeks=selected_weeks,
                    source_types=(MediaType.ARTICLE, MediaType.FILM_BREAKDOWN),
                )
            ),
            opponent=opponent,
            season=season,
            weeks=selected_weeks,
            k=k,
        ),
        _sample(
            index,
            media,
            name="coach_quote",
            query=coach_quote_query(opponent, selected_weeks),
            source_types=(MediaType.COACH_QUOTE,),
            kinds=("quote",),
            expected=bool(media.quotes_for(opponent=opponent, weeks=selected_weeks)),
            opponent=opponent,
            season=season,
            weeks=selected_weeks,
            k=k,
        ),
    )
    return RetrievalEvaluation(
        opponent=opponent,
        season=season,
        weeks=selected_weeks,
        samples=samples,
    )


def _sample(
    index: MediaIndex,
    media: MediaData,
    *,
    name: str,
    query: str,
    source_types: tuple[MediaType, ...],
    kinds: tuple[Literal["chunk", "quote"], ...],
    expected: bool,
    opponent: Team,
    season: int,
    weeks: tuple[int, ...],
    k: int,
) -> RetrievalSample:
    sources = index.retrieve(
        query,
        opponent=opponent,
        season=season,
        max_week=weeks[-1],
        source_types=source_types,
        kinds=kinds,
        k=k,
    )
    validation = check_retrieval(
        sources,
        media,
        opponent=opponent,
        weeks=weeks,
        season=season,
    )
    aligned = sum(
        1
        for source in sources
        if (stored := media.media_by_id(source.media_id)) is not None and stored.week in weeks
    )
    return RetrievalSample(
        name=name,
        query=query,
        source_types=tuple(source_type.value for source_type in source_types),
        kinds=kinds,
        expected=expected,
        retrieved=len(sources),
        aligned=aligned,
        warnings=tuple(issue.message for issue in validation.warnings),
        errors=tuple(issue.message for issue in validation.errors),
    )
