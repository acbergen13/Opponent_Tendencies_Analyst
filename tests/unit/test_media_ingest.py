"""Source allowlist, corpus parsing, verbatim quotes, and media normalization."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import HttpUrl

from opponent_tendencies.data.media_ingest import (
    MediaCorpusError,
    MediaIngestionError,
    QuoteExtractionError,
    SourcePolicyError,
    host_is_allowlisted,
    ingest_media,
    normalize_media,
    parse_media_corpus,
)
from opponent_tendencies.data.media_store import MediaStore
from opponent_tendencies.models import MediaType, Team
from tests.media_fixtures import (
    CORPUS_PATH,
    SEASON,
    corpus_document,
    corpus_from,
    mutated_document,
    mutated_quote_document,
    normalized_corpus,
    raw_corpus,
)


def test_corpus_normalizes_into_sorted_contracts() -> None:
    normalized = normalized_corpus()

    assert len(normalized.media) == 12
    assert len(normalized.quotes) == 5
    assert {item.opponent for item in normalized.media} == {
        Team.BAL,
        Team.BUF,
        Team.DET,
        Team.KC,
        Team.MIA,
        Team.SF,
    }
    assert normalized.media == tuple(
        sorted(normalized.media, key=lambda item: (item.published_on, item.media_id))
    )
    assert len({item.media_id for item in normalized.media}) == len(normalized.media)
    assert len({quote.quote_id for quote in normalized.quotes}) == len(normalized.quotes)


def test_normalization_is_deterministic() -> None:
    assert normalized_corpus() == normalized_corpus()


def test_quotes_are_verbatim_spans_of_their_media() -> None:
    normalized = normalized_corpus()
    by_id = {item.media_id: item for item in normalized.media}

    for quote in normalized.quotes:
        media = by_id[quote.media_id]
        assert quote.is_verbatim_in(media)
        assert media.body[quote.char_start : quote.char_end] == quote.text


def test_quote_metadata_defaults_come_from_the_media() -> None:
    normalized = normalized_corpus()
    by_media = {item.media_id: item for item in normalized.media}
    quote = next(
        item for item in normalized.quotes if item.text.startswith("We have to run the ball")
    )
    media = by_media[quote.media_id]

    assert quote.publisher == media.publisher
    assert quote.url == media.url
    assert quote.team == media.opponent
    assert quote.season == media.season
    assert quote.week == 9
    assert quote.spoken_on.isoformat() == "2026-11-09"


def test_quote_occurrence_selects_the_right_span() -> None:
    document = mutated_document(
        5,
        body=(
            "We have to run the ball better on first down. "
            "We have to run the ball better on first down."
        ),
        quotes=[
            {
                "speaker": "Sean McDermott",
                "speaker_role": "head_coach",
                "team": "BUF",
                "spoken_on": "2026-11-09",
                "text": "We have to run the ball better on first down.",
                "occurrence": 1,
            }
        ],
    )

    normalized = normalize_media(corpus_from(document))
    text = "We have to run the ball better on first down."
    body = document["items"][5]["body"]
    expected = body.find(text, body.find(text) + 1)
    quote = next(item for item in normalized.quotes if item.text == text)

    assert quote.char_start == expected
    assert quote.is_verbatim_in(
        next(item for item in normalized.media if item.media_id == quote.media_id)
    )


def test_unknown_source_is_rejected() -> None:
    document = mutated_document(0, source_id="unknown-blog")

    with pytest.raises(SourcePolicyError, match="not on the allowlist"):
        normalize_media(corpus_from(document))


def test_publisher_spoofing_is_rejected() -> None:
    document = mutated_document(0, publisher="Fake Sports Daily")

    with pytest.raises(SourcePolicyError, match="does not match allowlist publisher"):
        normalize_media(corpus_from(document))


def test_media_type_disallowed_for_source_is_rejected() -> None:
    document = mutated_document(5, source_type="film_breakdown")

    with pytest.raises(SourcePolicyError, match="not permitted"):
        normalize_media(corpus_from(document))


def test_untrusted_author_is_rejected() -> None:
    document = mutated_document(0, author="Unknown Reporter")

    with pytest.raises(SourcePolicyError, match="not a trusted reporter"):
        normalize_media(corpus_from(document))


def test_non_allowlisted_host_is_rejected() -> None:
    document = mutated_document(0, url="https://www.example.com/nfl/story")

    with pytest.raises(SourcePolicyError, match="not an allowlisted domain"):
        normalize_media(corpus_from(document))


def test_http_url_is_rejected() -> None:
    document = mutated_document(0, url="http://www.espn.com/nfl/story")

    with pytest.raises(SourcePolicyError, match="https"):
        normalize_media(corpus_from(document))


def test_subdomains_of_allowlisted_domains_are_accepted() -> None:
    assert host_is_allowlisted(HttpUrl("https://abc.espn.com/x"), ["espn.com"])
    assert not host_is_allowlisted(HttpUrl("https://espn.com.evil.test/x"), ["espn.com"])


def test_quote_text_must_appear_in_the_body() -> None:
    document = mutated_quote_document(5, 0, text="This sentence was never spoken.")

    with pytest.raises(QuoteExtractionError, match="does not appear"):
        normalize_media(corpus_from(document))


def test_explicit_quote_span_must_match_the_text() -> None:
    document = mutated_quote_document(5, 0, char_start=0, char_end=12)

    with pytest.raises(QuoteExtractionError, match="does not reproduce"):
        normalize_media(corpus_from(document))


def test_only_one_end_of_an_explicit_span_is_rejected() -> None:
    document = mutated_quote_document(5, 0, char_start=0)

    with pytest.raises(QuoteExtractionError, match="provided together"):
        normalize_media(corpus_from(document))


def test_coach_quote_source_requires_a_week() -> None:
    document = mutated_document(5)
    document["items"][5].pop("week")

    with pytest.raises(MediaCorpusError, match="week"):
        normalize_media(corpus_from(document))


def test_quote_team_must_match_the_media_opponent() -> None:
    document = mutated_quote_document(5, 0, team="KC")

    with pytest.raises(SourcePolicyError, match="does not match the media opponent"):
        normalize_media(corpus_from(document))


def test_duplicate_media_identity_is_rejected() -> None:
    document = corpus_document()
    document["items"].append(copy.deepcopy(document["items"][0]))

    with pytest.raises(MediaIngestionError, match="duplicate media identity"):
        normalize_media(corpus_from(document))


def test_duplicate_quote_identity_is_rejected() -> None:
    document = corpus_document()
    document["items"][5]["quotes"].append(copy.deepcopy(document["items"][5]["quotes"][0]))

    with pytest.raises(MediaIngestionError, match="duplicate quote identity"):
        normalize_media(corpus_from(document))


def test_unknown_corpus_field_is_rejected() -> None:
    document = mutated_document(0, surprise="unexpected")

    with pytest.raises(MediaCorpusError, match="unexpected field"):
        corpus_from(document)


def test_wrong_corpus_season_is_rejected() -> None:
    with pytest.raises(MediaCorpusError, match="expected 2025"):
        parse_media_corpus(corpus_document(), expected_season=2025)


def test_unknown_media_type_is_rejected() -> None:
    document = mutated_document(0, source_type="podcast")

    with pytest.raises(MediaCorpusError, match="unknown source_type"):
        corpus_from(document)


def test_naive_retrieved_at_is_rejected() -> None:
    document = corpus_document()
    document["retrieved_at"] = "2026-11-10T12:00:00"

    with pytest.raises(MediaCorpusError, match="timezone-aware"):
        corpus_from(document)


def test_empty_corpus_normalizes_to_empty_contracts() -> None:
    document = corpus_document()
    document["items"] = []

    normalized = normalize_media(corpus_from(document))

    assert normalized.media == ()
    assert normalized.quotes == ()


def test_connector_reports_missing_and_malformed_files(tmp_path: Path) -> None:
    from opponent_tendencies.data.media_ingest import JsonCorpusConnector

    missing = JsonCorpusConnector(tmp_path / "missing.json")
    with pytest.raises(MediaCorpusError, match="cannot read"):
        missing.load(SEASON)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    with pytest.raises(MediaCorpusError, match="not valid JSON"):
        JsonCorpusConnector(malformed).load(SEASON)


def test_connector_rejects_a_corpus_for_another_season(tmp_path: Path) -> None:
    from opponent_tendencies.data.media_ingest import JsonCorpusConnector

    document = corpus_document()
    document["season"] = 2025
    path = tmp_path / "media.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(MediaCorpusError, match="expected 2026"):
        JsonCorpusConnector(path).load(SEASON)


def test_ingest_media_writes_to_the_store(tmp_path: Path) -> None:
    from opponent_tendencies.data.media_ingest import JsonCorpusConnector

    store = MediaStore(SEASON, tmp_path / "media" / f"{SEASON}.json")

    report = ingest_media(JsonCorpusConnector(CORPUS_PATH), store)

    assert report.season == SEASON
    assert report.media_count == 12
    assert report.quote_count == 5
    assert report.refresh.inserted_media == 12
    assert report.refresh.inserted_quotes == 5
    assert "12 media items" in report.summary()
    assert store.load().opponents == (
        Team.BAL,
        Team.BUF,
        Team.DET,
        Team.KC,
        Team.MIA,
        Team.SF,
    )


def test_source_types_reported_by_store_are_sorted(tmp_path: Path) -> None:
    from tests.media_fixtures import write_media_store

    _, state = write_media_store(tmp_path)

    assert state.source_types == (
        MediaType.ARTICLE,
        MediaType.COACH_QUOTE,
        MediaType.FILM_BREAKDOWN,
    )
    assert state.media_for(opponent=Team.BUF, weeks=[9], source_types=[MediaType.COACH_QUOTE])
    assert len(state.media_for(opponent=Team.BUF, weeks=[9])) == 1
    assert state.quotes_for(opponent=Team.BUF, weeks=[9])
    assert not state.quotes_for(opponent=Team.BUF, weeks=[8])


def test_raw_corpus_matches_the_fixture_file() -> None:
    assert raw_corpus().season == SEASON
    assert len(raw_corpus().items) == 12
