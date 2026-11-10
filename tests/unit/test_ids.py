"""Tests for deterministic identifiers and content hashes."""

from __future__ import annotations

from opponent_tendencies.ids import (
    make_agreement_id,
    make_conflict_id,
    make_content_hash,
    make_gap_id,
    make_media_evidence_id,
    make_media_id,
    make_play_evidence_id,
    make_quote_evidence_id,
    make_quote_id,
    make_report_id,
    make_stat_evidence_id,
    make_stat_table_id,
    make_weak_evidence_id,
    play_key,
)

GAME_ID = "2026_09_BUF_KC"


def test_play_key_is_composite() -> None:
    assert play_key(GAME_ID, 42) == f"{GAME_ID}:42"
    assert play_key(GAME_ID, 42) != play_key("2026_10_BUF_NYJ", 42)


def test_media_id_is_stable_and_source_case_insensitive() -> None:
    first = make_media_id("Kansas-City-Star", "https://example.com/a")
    second = make_media_id("kansas-city-star", "https://example.com/a")

    assert first == second
    assert first.startswith("med_")
    assert first != make_media_id("kansas-city-star", "https://example.com/b")


def test_quote_id_depends_on_speaker_text_and_occurrence() -> None:
    base = make_quote_id("med_1", "Sean McDermott", "We must run better.")
    assert base == make_quote_id("med_1", "sean   mcdermott", "We must run better.")
    assert base != make_quote_id("med_1", "Sean McDermott", "We must pass better.")
    assert base != make_quote_id("med_1", "Sean McDermott", "We must run better.", 1)


def test_stat_table_id_ignores_input_order() -> None:
    first = make_stat_table_id("run_rate", "down", "1", [9, 8], [GAME_ID, "2026_08_BUF_MIA"])
    second = make_stat_table_id("run_rate", "down", "1", [8, 9], ["2026_08_BUF_MIA", GAME_ID])

    assert first == second
    assert first.startswith("stt_run_rate_down_")


def test_report_id_ignores_week_order_but_tracks_data_and_prompt() -> None:
    base = make_report_id("BUF", [9, 8], "rev-1", "v1")
    assert base == make_report_id("buf", [8, 9], "rev-1", "v1")
    assert base != make_report_id("BUF", [8, 9], "rev-2", "v1")
    assert base != make_report_id("BUF", [8, 9], "rev-1", "v2")


def test_evidence_ids_are_readable_and_derived_from_locators() -> None:
    assert make_play_evidence_id(GAME_ID, 42) == f"ev:play:{GAME_ID}:42"
    assert make_media_evidence_id("med_1") == "ev:media:med_1:full"
    assert make_media_evidence_id("med_1", "Chunk 2") == "ev:media:med_1:chunk_2"
    assert make_quote_evidence_id("qte_1") == "ev:quote:qte_1"
    assert make_stat_evidence_id("stt_1", "3rd & 2-4") == "ev:stat:stt_1:3rd_2_4"


def test_synthesis_ids_are_stable() -> None:
    assert make_agreement_id("T", "S", ["STA", "BRA"], [9]) == make_agreement_id(
        "t", "s", ["BRA", "STA"], [9]
    )
    assert make_conflict_id("T", "D", ["one", "two"], [9]) == make_conflict_id(
        "t", "d", ["two", "one"], [9]
    )
    assert make_gap_id("T", "D", [9]) == make_gap_id("t", "d", [9])
    assert make_weak_evidence_id("T", "D", ["ev:1"], [9]) == make_weak_evidence_id(
        "t", "d", ["ev:1"], [9]
    )


def test_content_hash_tracks_retrievable_text() -> None:
    first = make_content_hash("Title", "Body")
    assert first == make_content_hash("Title", "Body")
    assert first != make_content_hash("Title", "Body changed")
    assert len(first) == 64
