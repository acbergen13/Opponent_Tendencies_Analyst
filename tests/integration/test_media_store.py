"""Atomic media-store behavior, quote replacement, and manifest stability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opponent_tendencies.data.media_store import (
    SCHEMA_VERSION,
    MediaStore,
    MediaStoreCorruptionError,
    MediaStoreIntegrityError,
)
from opponent_tendencies.models import MediaType, Team
from tests.media_fixtures import (
    SEASON,
    normalized_corpus,
    with_media_changes,
    with_quote_changes,
    write_media_store,
)

pytestmark = pytest.mark.integration


def test_first_refresh_inserts_then_second_refresh_is_idempotent(tmp_path: Path) -> None:
    normalized = normalized_corpus()
    store = MediaStore(SEASON, tmp_path / "media" / f"{SEASON}.json")

    first = store.refresh(normalized.media, normalized.quotes)

    assert first.inserted_media == 12
    assert first.inserted_quotes == 5
    assert first.updated == 0
    assert first.removed == 0
    assert first.changed is True
    written = store.path.read_bytes()

    second = store.refresh(normalized.media, normalized.quotes)

    assert second.inserted == 0
    assert second.updated == 0
    assert second.removed == 0
    assert second.skipped_media == 12
    assert second.skipped_quotes == 5
    assert second.changed is False
    assert store.path.read_bytes() == written


def test_changed_body_updates_in_place(tmp_path: Path) -> None:
    normalized = normalized_corpus()
    store = MediaStore(SEASON, tmp_path / "media" / f"{SEASON}.json")
    store.refresh(normalized.media, normalized.quotes)
    before = store.load()
    target = normalized.media[0]
    changed = with_media_changes(target, body=target.body + "\n\nAn added note.")

    result = store.refresh((changed,), ())

    assert result.updated_media == 1
    assert result.removed_quotes == 0
    after = store.load()
    assert after.revision != before.revision
    stored = after.media_by_id(target.media_id)
    assert stored is not None and stored.body.endswith("An added note.")


def test_corrected_quote_replaces_the_stale_one(tmp_path: Path) -> None:
    normalized = normalized_corpus()
    store = MediaStore(SEASON, tmp_path / "media" / f"{SEASON}.json")
    store.refresh(normalized.media, normalized.quotes)
    before = store.load()
    target = next(quote for quote in normalized.quotes if quote.text.startswith("We have to run"))
    corrected = with_quote_changes(target, speaker="Sean McDermott (corrected)")
    media = before.media_by_id(target.media_id)
    assert media is not None
    incoming = tuple(
        corrected if quote.quote_id == target.quote_id else quote
        for quote in normalized.quotes
        if quote.media_id == target.media_id
    )

    result = store.refresh((media,), incoming)

    assert result.removed_quotes == 1
    assert result.inserted_quotes == 1
    after = store.load()
    assert after.quote_by_id(target.quote_id) is None
    assert after.quote_by_id(corrected.quote_id) is not None
    assert after.quotes_for(opponent=Team.BUF, weeks=[9])


def test_removing_all_quotes_from_a_media_item_prunes_them(tmp_path: Path) -> None:
    normalized = normalized_corpus()
    store = MediaStore(SEASON, tmp_path / "media" / f"{SEASON}.json")
    store.refresh(normalized.media, normalized.quotes)
    quote = next(quote for quote in normalized.quotes if quote.week == 9)
    media = store.load().media_by_id(quote.media_id)
    assert media is not None

    result = store.refresh((media,), ())

    assert result.removed_quotes >= 1
    assert store.load().quotes_for(opponent=media.opponent, weeks=[9]) == ()


def test_index_manifest_ignores_retrieval_time_but_tracks_content(tmp_path: Path) -> None:
    normalized = normalized_corpus()
    store = MediaStore(SEASON, tmp_path / "media" / f"{SEASON}.json")
    store.refresh(normalized.media, normalized.quotes)
    state = store.load()
    manifest = state.index_manifest()
    revision = state.revision

    refetched = tuple(
        with_media_changes(item, retrieved_at=item.retrieved_at.replace(hour=13).isoformat())
        for item in normalized.media
    )
    store.refresh(refetched, normalized.quotes)
    refreshed = store.load()

    assert refreshed.revision != revision
    assert refreshed.index_manifest() == manifest

    target = normalized.media[0]
    changed = with_media_changes(target, body=target.body + "\n\nNew reporting.")
    store.refresh((changed,), ())
    assert store.load().index_manifest() != manifest


def test_empty_refresh_creates_no_file(tmp_path: Path) -> None:
    store = MediaStore(SEASON, tmp_path / "media" / f"{SEASON}.json")

    result = store.refresh((), ())

    assert result.inserted == result.updated == result.skipped == 0
    assert result.changed is False
    assert store.exists() is False
    assert store.load().media == ()


def test_quote_without_media_is_rejected(tmp_path: Path) -> None:
    normalized = normalized_corpus()
    store = MediaStore(SEASON, tmp_path / "media" / f"{SEASON}.json")

    with pytest.raises(MediaStoreIntegrityError, match="unknown media"):
        store.refresh((), (normalized.quotes[0],))

    assert store.exists() is False


def test_quote_span_must_still_be_verbatim(tmp_path: Path) -> None:
    normalized = normalized_corpus()
    store = MediaStore(SEASON, tmp_path / "media" / f"{SEASON}.json")
    quote = normalized.quotes[0]
    media = next(item for item in normalized.media if item.media_id == quote.media_id)

    with pytest.raises(MediaStoreIntegrityError, match="verbatim"):
        store.refresh((media,), (quote.model_copy(update={"char_end": quote.char_end - 1}),))


def test_wrong_season_is_rejected(tmp_path: Path) -> None:
    normalized = normalized_corpus()
    store = MediaStore(SEASON, tmp_path / "media" / f"{SEASON}.json")

    with pytest.raises(MediaStoreIntegrityError, match="season 2025"):
        store.refresh((normalized.media[0].model_copy(update={"season": 2025}),), ())


def test_duplicate_records_in_one_refresh_are_rejected(tmp_path: Path) -> None:
    normalized = normalized_corpus()
    store = MediaStore(SEASON, tmp_path / "media" / f"{SEASON}.json")

    with pytest.raises(MediaStoreIntegrityError, match="duplicate media"):
        store.refresh((normalized.media[0], normalized.media[0]), ())
    with pytest.raises(MediaStoreIntegrityError, match="duplicate quote"):
        store.refresh((normalized.media[0],), (normalized.quotes[0], normalized.quotes[0]))


def test_stored_quote_team_must_match_its_media(tmp_path: Path) -> None:
    _, state = write_media_store(tmp_path)
    quote = state.quotes[0]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "season": SEASON,
        "media": [item.model_dump(mode="json") for item in state.media],
        "quotes": [
            quote.model_copy(update={"team": Team.KC}).model_dump(mode="json")
            if item.quote_id == quote.quote_id
            else item.model_dump(mode="json")
            for item in state.quotes
        ],
    }
    path = tmp_path / "media" / f"{SEASON}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MediaStoreCorruptionError, match="covers"):
        MediaStore(SEASON, path).load()


def test_corrupt_json_reports_the_path(tmp_path: Path) -> None:
    path = tmp_path / "media" / f"{SEASON}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(MediaStoreCorruptionError, match=str(path)):
        MediaStore(SEASON, path).load()


def test_unknown_schema_version_reports_recovery_step(tmp_path: Path) -> None:
    path = tmp_path / "media" / f"{SEASON}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"schema_version": 99, "season": SEASON, "media": [], "quotes": []}),
        encoding="utf-8",
    )

    with pytest.raises(MediaStoreCorruptionError, match="Delete the file and re-ingest"):
        MediaStore(SEASON, path).load()


def test_invalid_stored_record_reports_its_index(tmp_path: Path) -> None:
    path = tmp_path / "media" / f"{SEASON}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "season": SEASON,
                "media": [{"media_id": "broken"}],
                "quotes": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(MediaStoreCorruptionError, match=r"media\[0\]"):
        MediaStore(SEASON, path).load()


def test_store_file_is_canonical(tmp_path: Path) -> None:
    store, _ = write_media_store(tmp_path)

    document = json.loads(store.path.read_text(encoding="utf-8"))

    assert list(document) == sorted(document)
    assert document["schema_version"] == SCHEMA_VERSION
    media_ids = [item["media_id"] for item in document["media"]]
    assert media_ids == [item.media_id for item in store.load().media]
    quote_keys = [(quote["media_id"], quote["char_start"]) for quote in document["quotes"]]
    assert quote_keys == sorted(quote_keys)


def test_media_views_filter_by_opponent_week_and_type(tmp_path: Path) -> None:
    _, state = write_media_store(tmp_path)

    assert state.opponents == (
        Team.BAL,
        Team.BUF,
        Team.DET,
        Team.KC,
        Team.MIA,
        Team.SF,
    )
    assert state.weeks == (2, 3, 4, 5, 6, 8, 9)
    assert {item.week for item in state.media_for(opponent=Team.BUF, weeks=[5, 8])} == {5, 8}
    assert all(
        item.source_type is MediaType.COACH_QUOTE
        for item in state.media_for(source_types=[MediaType.COACH_QUOTE])
    )


def test_default_store_path_lives_under_data_dir() -> None:
    from opponent_tendencies.data.media_store import default_media_store_path

    assert default_media_store_path(2026, Path("/tmp/opponent-tendencies")) == Path(
        "/tmp/opponent-tendencies/media/2026.json"
    )
