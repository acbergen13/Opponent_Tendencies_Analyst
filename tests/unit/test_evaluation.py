"""Offline tests for the Stage 11 consistency evaluator."""

from __future__ import annotations

import pytest

from opponent_tendencies.data.media_store import MediaData
from opponent_tendencies.evaluation import (
    evaluate_consistency,
    tendency_key,
    top_tendencies,
)
from opponent_tendencies.models import Report, TendencyDirection
from tests.unit.factories import SEASON, make_quote, make_quote_media, make_report


def _with_tendencies(
    statements: list[str],
    *,
    directions: list[TendencyDirection] | None = None,
) -> Report:
    report = make_report()
    if directions is None:
        directions = [TendencyDirection.STABLE for _ in statements]
    tendencies = [
        report.tendencies[0].model_copy(update={"statement": statement, "direction": direction})
        for statement, direction in zip(statements, directions, strict=True)
    ]
    return report.model_copy(update={"tendencies": tendencies})


def _quote_media() -> MediaData:
    media = make_quote_media()
    return MediaData(season=SEASON, media=(media,), quotes=(make_quote(media),))


def test_top_tendencies_is_ordered_deduplicated_and_bounded() -> None:
    report = _with_tendencies(["A", "A", "B", "C"])

    assert top_tendencies(report, n=2) == (tendency_key(report.tendencies[0]), "run_pass|b")


def test_tendency_key_ignores_case_and_whitespace() -> None:
    report = _with_tendencies(["Buffalo  RUNS early"])
    noisy = report.tendencies[0].model_copy(update={"statement": " buffalo runs   early "})

    assert tendency_key(report.tendencies[0]) == tendency_key(noisy)


def test_four_of_five_runs_with_enough_overlap_passes() -> None:
    reports = [
        _with_tendencies(["A", "B", "C"]),
        _with_tendencies(["A", "B", "C"]),
        _with_tendencies(["A", "B", "other"]),
        _with_tendencies(["A", "B", "C"]),
        _with_tendencies(["A", "B", "C"]),
    ]

    result = evaluate_consistency(reports, media_data=_quote_media(), top_n=3)

    assert result.runs == 5
    assert result.required_passing_runs == 4
    assert result.passing_runs == 4
    assert result.overlaps[2] == pytest.approx(2 / 3)
    assert result.stats_identical is True
    assert result.direction_contradictions == ()
    assert result.passed is True


def test_too_few_passing_runs_fails() -> None:
    reports = [
        _with_tendencies(["A", "B", "C"]),
        _with_tendencies(["A", "other1", "other2"]),
        _with_tendencies(["A", "other1", "other2"]),
    ]

    result = evaluate_consistency(reports, media_data=_quote_media(), top_n=3)

    assert result.passing_runs < result.required_passing_runs
    assert result.passed is False


def test_direction_flip_is_a_contradiction() -> None:
    reports = [
        _with_tendencies(["Same statement"], directions=[TendencyDirection.INCREASE]),
        _with_tendencies(["Same statement"], directions=[TendencyDirection.DECREASE]),
    ]

    result = evaluate_consistency(reports, top_n=1)

    assert len(result.direction_contradictions) == 1
    assert result.passed is False


def test_changed_stat_tables_fail_even_when_narratives_match() -> None:
    first = _with_tendencies(["A"])
    second = _with_tendencies(["A"]).model_copy(update={"stat_tables": []})

    result = evaluate_consistency([first, second], top_n=1)

    assert result.stats_identical is False
    assert result.passed is False


def test_quote_traceability_requires_stored_verbatim_quotes() -> None:
    reports = [_with_tendencies(["A"])]

    good = evaluate_consistency(reports, media_data=_quote_media(), top_n=1)
    assert good.quote_traceability == 1.0

    bad = evaluate_consistency(reports, media_data=MediaData(season=SEASON), top_n=1)
    assert bad.quote_traceability == 0.0
    assert bad.passed is False


def test_empty_or_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one report"):
        evaluate_consistency([])
    with pytest.raises(ValueError, match="min_overlap"):
        evaluate_consistency([_with_tendencies(["A"])], min_overlap=1.5)
    with pytest.raises(ValueError, match="n must be at least 1"):
        top_tendencies(_with_tendencies(["A"]), n=0)
