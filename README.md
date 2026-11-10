# Opponent Tendencies Analyst

An NFL opponent-scouting assistant: deterministic Python computes the statistics,
and an LLM (VT ARC) interprets and cites them in a scouting report. The full
design lives in [`PLAN.md`](PLAN.md); the graded write-up lives in
[`docs/`](docs/).

> **Status: early.** Stages 0–2 of `PLAN.md` §4 are complete — packaging/config,
> frozen contracts, and nflverse ingestion into the JSON play store. The
> Streamlit app is still a placeholder, and the stats engine, media/RAG layer,
> agents, validation, and report workflow are **not implemented yet**. See
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
current placeholder app and the ingestion script do **not** read credentials, so
you can run both before the LLM stages land. Anything that does call
`get_settings()` fails fast with a `ConfigurationError` naming every missing
variable — there are no silent defaults for secrets.

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
| Streamlit app | 🚧 Placeholder only |
| Deterministic stats engine | ⛔ Not started |
| Media ingestion, ChromaDB index, retrieval | ⛔ Not started |
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

### Project layout

```text
app.py                              # Streamlit entrypoint (placeholder)
scripts/refresh_data.py             # nflverse -> data/seasons/<season>.json
src/opponent_tendencies/
  config.py                         # env parsing, Settings, ConfigurationError
  models.py                         # frozen contracts: Game, Play, Report, ...
  ids.py                            # deterministic ID/hash helpers
  llm/schemas.py                    # JSON schemas for structured LLM output
  data/nflverse.py                  # fetch + normalize play-by-play and schedules
  data/play_store.py                # atomic JSON store, upsert by (game_id, play_id)
  agents/ ui/ stats/ validation/    # package stubs for later stages
tests/
  unit/                             # contracts, config, ingestion, IDs
  integration/                      # play-store round-trip and idempotency
  fixtures/                         # trimmed real nflverse samples
docs/                               # graded Quarto deliverable (separate from the app)
data/                               # gitignored, generated by the refresh script
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
- **Never commit** `.env`, API keys, or anything under `data/`. Generated data is
  reproduced with `scripts/refresh_data.py`.
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
| `Port 8501 is already in use` | Another Streamlit is running. Use `--server.port <free port>`. |
| Refresh is slow | It downloads a full season of play-by-play; the second run is fast because nothing changed. |

## Roadmap

`PLAN.md` §4 tracks the remaining stages (stats engine → validation → media/RAG →
LLM/agents → orchestrator → UI → evaluation). Milestone and requirement
traceability tables are in `PLAN.md` §6 and §8.
