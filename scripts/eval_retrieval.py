"""Evaluate chunking and filtered retrieval on a small opponent sample.

Usage::

    python scripts/eval_retrieval.py --opponent BUF --weeks 5 6 7 8 9
    python scripts/eval_retrieval.py --opponent BUF --weeks 5 6 7 8 9 --embedder arc

The default ``--embedder hashing`` is fully offline: it builds an ephemeral
index with the deterministic test embedder and needs no credentials. ``arc``
uses ``EMBEDDING_MODEL`` and the persistent Chroma collection so the evaluation
runs against the same vectors the app uses. The deterministic checks live in
:mod:`opponent_tendencies.retrieval_eval`; this script only loads data, builds or
reuses an index, and prints the JSON result.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from opponent_tendencies.config import (
    ConfigurationError,
    get_settings,
    resolve_data_dir,
)
from opponent_tendencies.data.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from opponent_tendencies.data.index import MediaIndex
from opponent_tendencies.data.jsonio import atomic_write_text
from opponent_tendencies.data.media_store import MediaStore, default_media_store_path
from opponent_tendencies.llm.embeddings import ArcEmbedder, Embedder, HashingEmbedder
from opponent_tendencies.models import Team
from opponent_tendencies.retrieval_eval import (
    evaluate_corpus_chunking,
    evaluate_retrieval,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", required=True, help="team abbreviation, e.g. BUF")
    parser.add_argument("--weeks", nargs="+", type=int, required=True, help="selected weeks")
    parser.add_argument("--season", type=int, default=2026, help="season (default: 2026)")
    parser.add_argument(
        "--embedder",
        choices=("hashing", "arc"),
        default="hashing",
        help="hashing is offline; arc uses EMBEDDING_MODEL and the stored index",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Chroma collection name (default: media_<season>)",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="chunk size")
    parser.add_argument(
        "--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="chunk overlap"
    )
    parser.add_argument("--k", type=int, default=8, help="retrieval depth (default: 8)")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output path")
    args = parser.parse_args(argv)

    try:
        opponent = Team(args.opponent.upper())
    except ValueError:
        parser.error(f"unknown team abbreviation: {args.opponent}")
    weeks = tuple(sorted({int(week) for week in args.weeks}))
    if not weeks:
        parser.error("--weeks must contain at least one week")
    if any(week < 1 for week in weeks):
        parser.error("--weeks must be positive regular-season weeks")
    if args.k < 1:
        parser.error("--k must be at least 1")
    if args.chunk_size < 1 or not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("--chunk-overlap must be in [0, --chunk-size)")

    season = int(args.season)
    collection = args.collection or f"media_{season}"
    if args.embedder == "arc":
        try:
            settings = get_settings()
        except ConfigurationError as exc:
            parser.error(str(exc))
        data_dir = settings.data_dir
        embedder: Embedder = ArcEmbedder(
            settings.vt_arc_base_url,
            settings.vt_arc_api_key.get_secret_value(),
            settings.embedding_model,
        )
        index = MediaIndex(settings.chroma_dir, embedder, collection_name=collection)
    else:
        data_dir = resolve_data_dir()
        embedder = HashingEmbedder()
        index = MediaIndex(None, embedder, collection_name=collection)

    media = MediaStore(season, default_media_store_path(season, data_dir)).load()
    if not media.media:
        print(f"No stored media for season {season} under {data_dir / 'media'}.")
        return 2

    if index.count == 0:
        build = index.build(
            media,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"Index: {build.summary()}")

    chunking = evaluate_corpus_chunking(
        media,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    retrieval = evaluate_retrieval(
        index,
        media,
        opponent=opponent,
        season=season,
        weeks=weeks,
        k=args.k,
    )
    passed = chunking.passed and retrieval.passed
    payload = {
        "chunking": chunking.as_dict(),
        "passed": passed,
        "retrieval": retrieval.as_dict(),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text)
    if args.output is not None:
        atomic_write_text(args.output, text)
        print(f"Wrote {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
