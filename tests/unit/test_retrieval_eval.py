"""Offline chunking/retrieval evaluation on the tracked media corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

from opponent_tendencies.models import Team
from opponent_tendencies.retrieval_eval import (
    evaluate_corpus_chunking,
    evaluate_retrieval,
)
from tests.agent_fixtures import OPPONENT, SEASON, WEEKS, media_data, media_index


def test_chunking_evaluation_covers_every_stored_body(tmp_path: Path) -> None:
    media = media_data(tmp_path)

    report = evaluate_corpus_chunking(media, chunk_size=300, chunk_overlap=50)

    assert report.documents == len(media.media)
    assert report.chunks > report.documents
    assert report.coverage == 1.0
    assert report.hard_cuts == 0
    assert report.duplicate_texts == 0
    assert report.max_chunk_chars <= 300
    assert report.passed
    assert report.as_dict()["passed"] is True


def test_retrieval_evaluation_returns_aligned_sources(tmp_path: Path) -> None:
    media = media_data(tmp_path)
    index = media_index(media)

    result = evaluate_retrieval(
        index,
        media,
        opponent=OPPONENT,
        season=SEASON,
        weeks=WEEKS,
    )

    assert result.passed
    assert result.weeks == WEEKS
    assert [sample.name for sample in result.samples] == ["beat_reporter", "coach_quote"]
    for sample in result.samples:
        assert sample.expected
        assert sample.retrieved > 0
        assert sample.aligned > 0
        assert sample.errors == ()
        assert sample.passed
    payload = result.as_dict()
    assert payload["passed"] is True
    assert isinstance(payload["samples"], list)


def test_retrieval_evaluation_handles_a_selection_without_coach_quotes(tmp_path: Path) -> None:
    media = media_data(tmp_path)
    index = media_index(media)

    result = evaluate_retrieval(
        index,
        media,
        opponent=Team.MIA,
        season=SEASON,
        weeks=(3,),
    )

    by_name = {sample.name: sample for sample in result.samples}
    assert by_name["beat_reporter"].expected
    assert by_name["beat_reporter"].aligned > 0
    assert by_name["coach_quote"].expected is False
    assert by_name["coach_quote"].passed
    assert result.passed


def test_retrieval_evaluation_rejects_empty_weeks(tmp_path: Path) -> None:
    media = media_data(tmp_path)
    index = media_index(media)

    with pytest.raises(ValueError, match="week"):
        evaluate_retrieval(
            index,
            media,
            opponent=OPPONENT,
            season=SEASON,
            weeks=(),
        )
