"""Ingest the curated media corpus and build/reuse the Chroma index.

Usage::

    python scripts/build_index.py --season 2026

The media store is written under ``DATA_DIR/media/<season>.json`` and the index
under ``CHROMA_DIR``. The default embedder is the pinned VT ARC model from
``EMBEDDING_MODEL``. ``--embedder hashing`` is an explicit offline mode for
chunking evaluation and tests; it records a different model ID, so a later
VT ARC query refuses to reuse the collection.

The index is rebuilt only when the embedding model, chunk settings, or the
stored content manifest changes. Repeat runs reuse it and embed nothing.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from opponent_tendencies.config import (
    ConfigurationError,
    get_settings,
    resolve_chroma_dir,
    resolve_data_dir,
)
from opponent_tendencies.data.index import MediaIndex
from opponent_tendencies.data.media_ingest import JsonCorpusConnector, ingest_media
from opponent_tendencies.data.media_store import MediaStore, default_media_store_path
from opponent_tendencies.llm.embeddings import ArcEmbedder, Embedder, HashingEmbedder

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026, help="season to ingest (default: 2026)")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="curated media corpus JSON (default: corpus/media_<season>.json)",
    )
    parser.add_argument(
        "--embedder",
        choices=("arc", "hashing"),
        default="arc",
        help="arc uses EMBEDDING_MODEL; hashing is offline-only",
    )
    parser.add_argument("--collection", default=None, help="Chroma collection name")
    parser.add_argument("--chunk-size", type=int, default=1200, help="chunk size in characters")
    parser.add_argument(
        "--chunk-overlap", type=int, default=200, help="chunk overlap in characters"
    )
    parser.add_argument(
        "--force", action="store_true", help="rebuild even when the manifest matches"
    )
    args = parser.parse_args(argv)

    season = int(args.season)
    corpus_path = (
        Path(args.corpus)
        if args.corpus is not None
        else _REPO_ROOT / "corpus" / f"media_{season}.json"
    )
    if not corpus_path.is_file():
        parser.error(f"media corpus not found: {corpus_path}")

    if args.embedder == "arc":
        try:
            settings = get_settings()
        except ConfigurationError as exc:
            parser.error(str(exc))
        embedder: Embedder = ArcEmbedder(
            settings.vt_arc_base_url,
            settings.vt_arc_api_key.get_secret_value(),
            settings.embedding_model,
        )
        data_dir = settings.data_dir
        chroma_dir = settings.chroma_dir
    else:
        embedder = HashingEmbedder()
        data_dir = resolve_data_dir()
        chroma_dir = resolve_chroma_dir()

    collection_name = args.collection or f"media_{season}"
    store = MediaStore(season, default_media_store_path(season, data_dir))
    report = ingest_media(JsonCorpusConnector(corpus_path), store)
    state = store.load()

    print(f"Corpus: {corpus_path}")
    print(f"Store:  {store.path}")
    print(report.summary())
    print(f"Opponents: {', '.join(team.value for team in state.opponents)}")
    print(f"Weeks: {list(state.weeks)}")

    index = MediaIndex(chroma_dir, embedder, collection_name=collection_name)
    build = index.build(
        state,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force=args.force,
    )
    print(f"Index:  {chroma_dir} ({collection_name})")
    print(build.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
