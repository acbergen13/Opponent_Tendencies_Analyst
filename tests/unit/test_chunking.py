"""Deterministic chunking and the chunking-evaluation report."""

from __future__ import annotations

import itertools

import pytest

from opponent_tendencies.data.chunking import (
    ChunkSpan,
    chunk_media,
    chunk_text,
    evaluate_chunking,
)
from opponent_tendencies.ids import make_chunk_id
from tests.media_fixtures import normalized_corpus

LONG_TEXT = "".join(
    f"Note {index}: Buffalo runs outside zone behind guard {index} on first down. "
    f"Quarterback {index} keeps on a bootleg when edge defender {index} crashes. "
    f"On third down {index} the offense condenses its splits and sends receiver {index} "
    f"in motion before the snap count {index}.\n\n"
    for index in range(8)
)


def test_spans_are_bounded_complete_and_overlapping() -> None:
    spans = chunk_text(LONG_TEXT, size=160, overlap=40)

    assert len(spans) > 2
    assert spans[0].char_start == 0
    assert spans[-1].char_end == len(LONG_TEXT)
    assert all(span.length <= 160 for span in spans)
    assert all(span.char_start < span.char_end for span in spans)
    assert all(current.char_start > previous.char_start for previous, current in _pairs(spans))
    assert all(current.char_start < previous.char_end for previous, current in _pairs(spans))

    report = evaluate_chunking(LONG_TEXT, spans)
    assert report.coverage == 1.0
    assert report.chunk_count == len(spans)
    assert report.max_chars <= 160
    assert report.overlapped_pairs == len(spans) - 1
    assert report.hard_cuts == 0
    assert report.duplicate_texts == 0


def test_short_text_is_one_span() -> None:
    spans = chunk_text("A short note.", size=100, overlap=10)

    assert spans == (ChunkSpan(index=0, char_start=0, char_end=13),)


def test_empty_text_has_no_spans() -> None:
    assert chunk_text("", size=100, overlap=10) == ()
    report = evaluate_chunking("", ())
    assert report.chunk_count == 0
    assert report.coverage == 1.0


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-1, 0), (100, -1), (100, 100), (100, 101)],
)
def test_invalid_parameters_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="chunk"):
        chunk_text(LONG_TEXT, size=size, overlap=overlap)


def test_chunk_media_uses_exact_body_spans_and_stable_ids() -> None:
    media = normalized_corpus().media[0]

    chunks = chunk_media(media, size=180, overlap=30)

    assert chunks == chunk_media(media, size=180, overlap=30)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        assert chunk.media_id == media.media_id
        assert chunk.text == media.body[chunk.char_start : chunk.char_end]
        assert chunk.chunk_id == make_chunk_id(media.media_id, chunk.char_start, chunk.char_end)


def test_chunk_id_changes_with_span_and_media() -> None:
    assert make_chunk_id("med_a", 0, 100) == make_chunk_id("med_a", 0, 100)
    assert make_chunk_id("med_a", 0, 100) != make_chunk_id("med_a", 1, 100)
    assert make_chunk_id("med_a", 0, 100) != make_chunk_id("med_b", 0, 100)


def test_every_corpus_media_chunks_with_full_coverage() -> None:
    for media in normalized_corpus().media:
        spans = chunk_text(media.body, size=300, overlap=50)
        report = evaluate_chunking(media.body, spans)
        assert report.coverage == 1.0
        assert report.max_chars <= 300
        assert report.hard_cuts == 0


def _pairs(spans: tuple[ChunkSpan, ...]) -> list[tuple[ChunkSpan, ChunkSpan]]:
    return list(itertools.pairwise(spans))
