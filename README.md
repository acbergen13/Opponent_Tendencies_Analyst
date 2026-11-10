# Opponent Tendencies Analyst

An NFL opponent-scouting assistant: deterministic Python computes the statistics,
and an LLM (VT ARC) interprets and cites them in a scouting report. The full
design lives in [`PLAN.md`](PLAN.md); the graded write-up lives in
[`docs/`](docs/).

> **Status: foundation complete.** Stages 0–3 of `PLAN.md` §4 are done —
> packaging/config, frozen contracts, nflverse ingestion into the JSON play
> store, and curated media ingestion plus a persistent Chroma index. The
> Streamlit app is still a placeholder, and the stats engine, validation,
> agents, and report workflow are **not implemented yet**. See
> [What works today](#what-works-today).

---

## Quick start

### Prerequisites

- Python **3.12+** (developed and verified on 3.13)
- `git`
- Network access for the first data refresh (nflverse downloads from GitHub
  releases)

Everything installs from `pyproject.toml`; no system packages are required.

### 1. Clone and create a virtual environment

```bash
git clone <repo-url> opponent_tendencies
cd opponent_tendencies

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install the package (editable, with dev tools)

```bash
pip install -e ".[dev]"
```

### 3. (Optional, for now) Configure credentials

Copy the template and fill in real values:

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
| --- | --- | --- |
| `VT_ARC_BASE_URL` | yes* | VT ARC OpenAI-compatible endpoint |
| `VT_ARC_API_KEY` | yes* | API key (never commit this file) |
| `VT_ARC_MODEL` | yes* | Model ID the agents will call |
| `EMBEDDING_MODEL` | yes* | Pinned embedding model for the media index |
| `DATA_DIR` | no | Where stores live (default `data/`) |
| `CHROMA_DIR` | no | Where the Chroma index lives (default `data/chroma`) |

\* Required only when code calls `Settings.from_env()` / `get_settings()`. The
placeholder app and `scripts/refresh_data.py` do **not** read credentials.
`scripts/build_index.py --embedder arc` (the default) does, because it sends
documents to the VT ARC embedding endpoint. Anything that calls `get_settings()`
fails fast with a `ConfigurationError` naming every missing variable — there are
no silent defaults for secrets. For an offline index build, pass
`--embedder hashing`; it records a different model ID, so a later VT ARC query
refuses to reuse that collection.

### 4. Run the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (normally <http://localhost:8501>). Right now
you will see a single "under construction" banner — that is expected; it is the
Stage 0 placeholder that Stage 10 replaces with the selection → report →
follow-up workflow.

To use a different port or run without opening a browser:

```bash
streamlit run app.py --server.port 8502 --server.headless true
```

### 5. Load season data (optional but useful)

```bash
python scripts/refresh_data.py --season 2026
```

This fetches nflverse play-by-play and schedules, keeps regular-season rows
only, normalizes them into the frozen contracts, and upserts them into
`data/seasons/2026.json`. It prints the store path, a one-line
`season <year>: N games, M plays; X inserted, Y updated, Z skipped` summary, the
counts of rows deliberately dropped (deleted / non-regular / unknown play type),
the played regular-season weeks, and the alphabetical team list.

Notes:

- Omit `--season` to ingest nflverse's current season.
- `DATA_DIR` is honored without any LLM credentials.
- The refresh is **idempotent**: re-running on unchanged data reports everything
  as skipped and does not rewrite the file.
- The JSON store is generated output — never hand-edit it. To rebuild, delete
  `data/seasons/<season>.json` and refresh again.
- Ingesting a season before it has games is fine: the refresh writes no file and
  the store reads as empty (`[]` weeks, no teams).

### 6. Build the media store and vector index (optional but useful)

```bash
python scripts/build_index.py --season 2026
```

This ingests the curated corpus at `corpus/media_2026.json` into
`data/media/2026.json`, then builds or reuses a persistent Chroma collection
(`media_2026` by default) under `data/chroma`. It prints the store path, the
insert/update/skip counts, the opponents and weeks present, and whether the
index was rebuilt or reused.

Notes:

- The default `--embedder arc` reads `EMBEDDING_MODEL` and sends text to VT ARC
  (`Qwen3-Embedding-4B` is verified). Use `--embedder hashing` for a fully
  offline build; tests use the same hashing embedder and never hit the network.
- The index is rebuilt only when the embedding model, chunk settings, or the
  stored content/quote manifest changes. A repeat run embeds nothing and leaves
  the store byte-identical.
- Tune chunking with `--chunk-size` / `--chunk-overlap` (defaults 1200/200
  characters). A chunk-setting change forces a full rebuild.
- The media store and Chroma index are generated output — never hand-edit them.
  Delete `data/media/<season>.json` and the collection (or pass `--force`) to
  rebuild.
- `--collection` selects the Chroma collection name; retrieval filters by
  opponent, season, week range, and source type before similarity search.

---

## What works today

| Area | State |
| --- | --- |
| Package + tooling (`pyproject.toml`, `ruff`, `mypy`, `pytest`) | ✅ Stage 0 |
| Environment config with fail-fast validation (`config.py`) | ✅ Stage 0 |
| Frozen Pydantic contracts (`models.py`) and LLM JSON schemas (`llm/schemas.py`) | ✅ Stage 1 |
| Deterministic ID helpers (`ids.py`) | ✅ Stage 1 |
| nflverse ingestion → atomic JSON store (`data/nflverse.py`, `data/play_store.py`) | ✅ Stage 2 |
| Data refresh CLI (`scripts/refresh_data.py`) | ✅ Stage 2 |
| Media allowlist, curated corpus, verbatim quotes (`data/media_ingest.py`, `data/media_store.py`) | ✅ Stage 3 |
| Deterministic chunking + evaluation (`data/chunking.py`) | ✅ Stage 3 |
| Persistent Chroma index, reuse detection, filtered retrieval (`data/index.py`) | ✅ Stage 3 |
| Pinned VT ARC / offline embeddings (`llm/embeddings.py`) | ✅ Stage 3 |
| Media/index CLI (`scripts/build_index.py`) | ✅ Stage 3 |
| Streamlit app | 🚧 Placeholder only |
| Deterministic stats engine | ⛔ Not started |
| Validation layer | ⛔ Not started |
| LLM client, prompts, agents, orchestrator | ⛔ Not started |
| UI report flow and follow-up chat | ⛔ Not started |

Do not assume a module exists because `PLAN.md` lists it — check the tree.

---

## Development

Run the local quality gates:

```bash
pytest -q                                  # full suite (fixture-driven, no network)
ruff check . && ruff format --check .      # lint + formatting
mypy src tests scripts                     # strict typing
```

Apply formatting:

```bash
ruff format .
```

Tests never hit the network: nflverse fixtures are trimmed real 2026 pulls in
`tests/fixtures/` and are injected through `NflverseClient`'s loader arguments.
Media tests use the tracked `corpus/media_2026.json` fixture with the offline
`HashingEmbedder`; `tests/unit/test_embeddings.py` mocks the VT ARC HTTP
contract rather than calling it.

### Project layout

```text
app.py                              # Streamlit entrypoint (placeholder)
corpus/media_2026.json              # curated, tracked Stage 3 input
scripts/refresh_data.py             # nflverse -> data/seasons/<season>.json
scripts/build_index.py              # corpus -> data/media/<season>.json + Chroma
src/opponent_tendencies/
  config.py                         # env parsing, Settings, ConfigurationError
  models.py                         # frozen contracts: Game, Play, MediaItem, Quote, ...
  ids.py                            # deterministic ID/hash helpers
  llm/schemas.py                    # JSON schemas for structured LLM output
  llm/embeddings.py                 # ArcEmbedder + offline HashingEmbedder
  data/nflverse.py                  # fetch + normalize play-by-play and schedules
  data/play_store.py                # atomic JSON store, upsert by (game_id, play_id)
  data/jsonio.py                    # canonical JSON + atomic writes
  data/media_ingest.py              # allowlist, corpus parser, quote spans
  data/media_store.py               # atomic media/quote store + index manifest
  data/chunking.py                  # deterministic chunk spans + evaluation
  data/index.py                     # Chroma build/reuse + filtered retrieval
  agents/ ui/ stats/ validation/    # package stubs for later stages
tests/
  unit/                             # contracts, config, ingestion, chunking, embeddings
  integration/                      # stores, idempotency, index reuse/retrieval
  fixtures/                         # trimmed real nflverse samples
  media_fixtures.py                 # shared curated-corpus helpers
docs/                               # graded Quarto deliverable (separate from the app)
data/                               # gitignored, generated by the scripts
```

### Ground rules for contributors

- **Deterministic Python owns the facts.** All math, filtering, sorting,
  formatting, duplicate checks, and source-ID checks stay in Python. The LLM only
  interprets, extracts, and synthesizes (`PLAN.md` §1).
- **Contracts are frozen.** Changing anything in `models.py` is a deliberate
  design change: update `llm/schemas.py` and keep `tests/unit/test_models.py`
  green.
- **Pin the source schema.** `PLAY_COLUMNS` / `GAME_COLUMNS` in
  `data/nflverse.py` are the nflverse columns we depend on; a missing column
  raises `MissingColumnsError` instead of being silently skipped. Re-spike
  nflverse before changing that list.
- **Keep quotes verbatim and IDs derived.** `data/media_ingest.py` only accepts
  allowlisted publishers/domains/types/authors and requires every quote to be an
  exact span of the stored body. Never rewrite a direct quote. Construct modified
  media/quotes by revalidation so `media_id`, `quote_id`, and `content_hash`
  stay consistent (`tests/media_fixtures.py` has helpers).
- **The index is derived.** `data/media_store.py` is the source of truth and
  `data/index.py` is a disposable Chroma collection. Reuse requires matching
  model ID, schema version, chunk settings, unit count, and manifest; anything
  else triggers a full rebuild. Do not mix embedders — `ArcEmbedder` and
  `HashingEmbedder` record different model IDs on purpose.
- **Never commit** `.env`, API keys, or anything under `data/`. Generated data is
  reproduced with `scripts/refresh_data.py` and `scripts/build_index.py`.
- **Keep the deliverable clean.** `docs/` is graded and separate from the app;
  implementation code belongs outside it. If you edit the `.qmd`, re-render the
  PDF:

  ```bash
  quarto render docs/final_project_reqs_and_arch_zachary_and_adam.qmd
  ```

  (This repo's environment has Quarto at `/opt/quarto/bin/quarto` if it is not
  on your `PATH`.)

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ConfigurationError: Missing required environment variable(s): ...` | Copy `.env.example` to `.env` and set every listed value. Paths are resolved from the directory you run the command in. |
| `StoreCorruptionError: ... has schema_version ...; delete the file and refresh` | The store predates a schema change. Delete `data/seasons/<season>.json` and re-run the refresh. |
| `MissingColumnsError: ... frame is missing required column(s)` | The nflverse schema changed. Re-spike the client before ingesting (see `PLAN.md` §7). |
| `SourcePolicyError: ...` during `build_index.py` | The corpus item is not allowlisted (publisher, HTTPS domain, source type, or author). Fix the corpus or extend `SOURCE_ALLOWLIST` deliberately. |
| `QuoteExtractionError: ... does not appear / does not reproduce` | The quote is not an exact span of the item's `body`. Quote text is never rewritten; fix the corpus body or use `char_start`/`char_end`/`occurrence`. |
| `MediaStoreCorruptionError: ... delete the file and re-ingest` | The generated media store is old or damaged. Delete `data/media/<season>.json` and re-run `scripts/build_index.py`. |
| `IndexModelMismatchError: ... built with model ...` | The query embedder differs from the one that built the Chroma collection. Rebuild with the same `EMBEDDING_MODEL` or pass `--force` after switching embedders. |
| `Port 8501 is already in use` | Another Streamlit is running. Use `--server.port <free port>`. |
| Refresh is slow | It downloads a full season of play-by-play; the second run is fast because nothing changed. |

## Roadmap

`PLAN.md` §4 tracks the remaining stages (stats engine → validation → LLM/agents
→ orchestrator → UI → evaluation). Milestone and requirement traceability tables
are in `PLAN.md` §6 and §8. Stage 3 is complete; the next slice is the
deterministic statistics engine.
