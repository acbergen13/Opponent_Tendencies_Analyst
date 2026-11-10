"""Deterministic, word-aware chunking for the media index.

Chunking is deliberately boring: no tokenizer, no model, no randomness. It walks
the stored text by character budget, prefers paragraph and sentence boundaries,
and rewinds by a fixed overlap so a fact that straddles a boundary still appears
in one complete chunk. Every chunk carries its exact character span, which is
what lets a citation resolve back to the stored article or quote.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from opponent_tendencies.ids import make_chunk_id
from opponent_tendencies.models import MediaItem

__all__ = [
    "ChunkSpan",
    "ChunkingReport",
    "MediaChunk",
    "chunk_media",
    "chunk_text",
    "evaluate_chunking",
]

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200

_SENTENCE_MARKERS = (". ", "? ", "! ", '." ', '?" ', '!" ')


@dataclass(frozen=True)
class ChunkSpan:
    """One chunk's location inside the stored text."""

    index: int
    char_start: int
    char_end: int

    @property
    def length(self) -> int:
        return self.char_end - self.char_start


@dataclass(frozen=True)
class MediaChunk:
    """A chunk of a stored media item, ready to embed."""

    chunk_id: str
    media_id: str
    index: int
    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True)
class ChunkingReport:
    """Evaluation of a chunking run, used by the Stage 3 chunking check."""

    chunk_count: int
    total_chars: int
    covered_chars: int
    min_chars: int
    max_chars: int
    mean_chars: float
    overlapped_pairs: int
    hard_cuts: int
    duplicate_texts: int

    @property
    def coverage(self) -> float:
        if self.total_chars == 0:
            return 1.0
        return self.covered_chars / self.total_chars


def chunk_text(
    text: str,
    *,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[ChunkSpan, ...]:
    """Split ``text`` into overlapping spans of at most ``size`` characters.

    ``overlap`` must be smaller than ``size``. Boundaries prefer a paragraph
    break, then a line break, then a sentence end, then whitespace; if none is
    available the span is cut at the character budget.
    """

    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0:
        raise ValueError("chunk overlap must not be negative")
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    if not text:
        return ()
    if len(text) <= size:
        return (ChunkSpan(index=0, char_start=0, char_end=len(text)),)

    spans: list[ChunkSpan] = []
    start = 0
    length = len(text)
    while start < length:
        hard_end = min(start + size, length)
        end = hard_end
        if hard_end < length:
            minimum_end = start + max(overlap + 1, size // 2)
            if minimum_end < hard_end:
                boundary = _find_boundary(text, minimum_end, hard_end)
                if boundary is not None:
                    end = boundary
        if end <= start:  # pragma: no cover - defensive against pathological input
            end = hard_end
        spans.append(ChunkSpan(index=len(spans), char_start=start, char_end=end))
        if end >= length:
            break
        next_start = max(start + 1, end - overlap)
        while next_start < end and text[next_start].isspace() and end - next_start > overlap:
            next_start += 1
        if next_start >= end:  # pragma: no cover - overlap < size guarantees progress
            next_start = end
        start = next_start
    return tuple(spans)


def _find_boundary(text: str, minimum_end: int, hard_end: int) -> int | None:
    """Return the best cut position in ``[minimum_end, hard_end]``, if any."""

    paragraph = text.rfind("\n\n", minimum_end, hard_end)
    if paragraph >= minimum_end:
        return paragraph + 2

    line = text.rfind("\n", minimum_end, hard_end)
    if line >= minimum_end:
        return line + 1

    sentence = max(text.rfind(marker, minimum_end, hard_end) for marker in _SENTENCE_MARKERS)
    if sentence >= minimum_end:
        return sentence + 1

    space = max(text.rfind(" ", minimum_end, hard_end), text.rfind("\t", minimum_end, hard_end))
    if space >= minimum_end:
        return space
    return None


def chunk_media(
    media: MediaItem,
    *,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[MediaChunk, ...]:
    """Return the chunks of one stored media item."""

    return tuple(
        MediaChunk(
            chunk_id=make_chunk_id(media.media_id, span.char_start, span.char_end),
            media_id=media.media_id,
            index=span.index,
            char_start=span.char_start,
            char_end=span.char_end,
            text=media.body[span.char_start : span.char_end],
        )
        for span in chunk_text(media.body, size=size, overlap=overlap)
    )


def evaluate_chunking(text: str, spans: tuple[ChunkSpan, ...]) -> ChunkingReport:
    """Measure coverage, size bounds, overlap, and boundary quality."""

    if not spans:
        return ChunkingReport(
            chunk_count=0,
            total_chars=len(text),
            covered_chars=0,
            min_chars=0,
            max_chars=0,
            mean_chars=0.0,
            overlapped_pairs=0,
            hard_cuts=0,
            duplicate_texts=0,
        )

    covered: set[int] = set()
    for span in spans:
        covered.update(range(span.char_start, span.char_end))
    lengths = [span.length for span in spans]
    overlapped_pairs = sum(
        1
        for previous, current in itertools.pairwise(spans)
        if current.char_start < previous.char_end
    )
    hard_cuts = sum(
        1
        for span in spans
        if span.char_end < len(text)
        and not text[span.char_end - 1].isspace()
        and not text[span.char_end].isspace()
    )
    texts = [text[span.char_start : span.char_end] for span in spans]
    duplicate_texts = len(texts) - len(set(texts))
    return ChunkingReport(
        chunk_count=len(spans),
        total_chars=len(text),
        covered_chars=len(covered),
        min_chars=min(lengths),
        max_chars=max(lengths),
        mean_chars=sum(lengths) / len(lengths),
        overlapped_pairs=overlapped_pairs,
        hard_cuts=hard_cuts,
        duplicate_texts=duplicate_texts,
    )
