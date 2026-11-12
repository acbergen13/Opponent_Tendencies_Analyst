# AGENTS.md

## What this repo is

Two things live here:

1. **The graded Quarto deliverable** in `docs/` (design write-up for the NFL
   "Opponent Tendencies Analyst"). This is the assignment artifact.
2. **The implementation** described by `PLAN.md`: a Python package under
   `src/opponent_tendencies/` with a Streamlit app. Development proceeds stage
   by stage in `PLAN.md` §4. Stages 0–6 are done: packaging/config, the frozen
   Pydantic contracts in `models.py` with the LLM JSON schemas in
   `llm/schemas.py`, nflverse play-by-play ingestion into the atomic JSON store
   at `data/seasons/<season>.json`, curated media ingestion plus a
   persistent Chroma index at `data/media/<season>.json` / `data/chroma`, the
   deterministic statistics engine in `stats/engine.py`, the structured
   validation layer in `validation/checks.py` (pure checks returning
   `ValidationResult`s — `CheckCode` issues for selection, plays, media
   alignment, retrieval grounding, evidence resolution, stat tables, and raw
   records), and the LLM layer in `llm/client.py` + `llm/prompts/` (the
   OpenAI-compatible VT ARC chat client with bounded retries and one-shot
   structured-output repair, six versioned constitutional prompts, and
   deterministic data-block renderers) plus offline prompt optimization in
   `llm/optimize.py` / `scripts/optimize_prompts.py`. Agents (Stage 7+),
   orchestrator, and UI are not implemented yet — do not assume modules
   that the plan lists but the tree lacks.

`docs/` is graded and separate from the app. Implementation code must live
outside `docs/`.

## Docs deliverable (`docs/`)

- `docs/final_project_reqs_and_arch_zachary_and_adam.qmd` — the only editable source
- `docs/final_project_reqs_and_arch_zachary_and_adam.pdf` — rendered deliverable (committed)
- `docs/opponent-tendencies-sequence-diagram.png` — rendered sequence diagram (committed)

Render from the repo root (Quarto 1.8.24 is installed at `/opt/quarto/bin/quarto`):

```bash
quarto render docs/final_project_reqs_and_arch_zachary_and_adam.qmd
```

The PDF is written next to the `.qmd` (`docs/`). It is a graded deliverable, so re-render and commit it whenever the `.qmd` changes. Never hand-edit the PDF or PNG; regenerate them.

Quick config check without rendering: `quarto inspect docs/final_project_reqs_and_arch_zachary_and_adam.qmd`.

## Implementation toolchain

Python 3.13 with a local venv (`.venv/`, gitignored). Dependencies are pinned in
`pyproject.toml`; `chromadb`, `nflreadpy`, and the rest install cleanly on 3.13
(verified), so no 3.12 fallback is in use.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ruff check . && ruff format --check .
mypy src tests scripts
python scripts/refresh_data.py --season 2026   # nflverse -> data/seasons/2026.json
python scripts/build_index.py --season 2026    # corpus/media_2026.json -> store + Chroma
python scripts/optimize_prompts.py             # offline prompt optimization (design-time only)
streamlit run app.py
```

Configuration is environment-driven (see `.env.example`); `opponent_tendencies.config`
fails fast with `ConfigurationError` when `VT_ARC_BASE_URL`, `VT_ARC_API_KEY`,
`VT_ARC_MODEL`, or `EMBEDDING_MODEL` is missing. Never commit `.env`, keys, or
generated `data/` / Chroma directories.

## Editing gotchas

- Requirements are numbered 1–30 and issues 1–18. Later sections ("LLM / RAG Issues", "Solution Mitigation") reference those numbers explicitly. If you add, remove, or reorder an item, update every cross-reference.
- The Mermaid `sequenceDiagram` in the `.qmd` and `docs/opponent-tendencies-sequence-diagram.png` depict the same flow. If you change one, update the other.
- The diagram is referenced by a relative filename, so the PNG must stay in `docs/` beside the `.qmd`.
- Assignment text in the `.qmd` states the write-up must be produced without AI assistance. Treat content generation accordingly and limit agent help to repo mechanics unless the user says otherwise.
- Deterministic Python owns all math, filtering, sorting, formatting, duplicate checks, and source-ID checks; the LLM only interprets, extracts, and synthesizes (`PLAN.md` §1). Keep that split when adding code.
- Stage 1 contracts in `opponent_tendencies.models` are frozen. Change them deliberately and keep `tests/unit/test_models.py` green; `llm/schemas.py` builds the LLM wire schemas from the same enums/types, with server-owned identity stamped by Python.
- Stage 2 ingestion in `data/nflverse.py` pins the nflverse columns it reads (`PLAY_COLUMNS`, `GAME_COLUMNS`); a missing column raises `MissingColumnsError` instead of being skipped. `nflreadpy` returns Polars, 2026 uses `WAS` (aliased to `WSH`), and nflverse fills third-down flags on every row, so the normalizer keeps them only on down 3. Re-spike the client before changing the column list.
- `data/seasons/<season>.json` is generated. `SeasonStore` upserts by `game_id` / `(game_id, play_id)` in one atomic write and does not rewrite unchanged data, so a double refresh is byte-identical. Never hand-edit it; delete and re-run `scripts/refresh_data.py`. Fixtures in `tests/fixtures/nflverse_*_sample.json` are trimmed real 2026 pulls.
- Stage 3 media lives in `data/media/<season>.json` (generated, atomic, canonical) and the curated input lives in `corpus/media_<season>.json` (tracked). `MediaStore` upserts media by `media_id` and replaces each incoming item's whole quote set, so a corrected/removed speaker cannot leave stale quotes. Never hand-edit the generated store; delete and re-run `scripts/build_index.py`. Refetching identical text keeps the index manifest stable and reuses the index; only title/body/quote/filter changes force a rebuild.
- `data/media_ingest.py` enforces the source allowlist (publisher, HTTPS allowlisted domain, media type, trusted authors). Every quote must be an exact character span of its stored body. Corpus documents reject unknown fields, and media/quote IDs derive from `source_id` + URL and media/speaker/text/occurrence respectively; construct changed records by revalidation (`tests/media_fixtures.py` helpers), not `model_copy` alone.
- `data/chunking.py` owns all chunk boundaries and IDs (`make_chunk_id`); defaults are 1200/200 characters and cuts prefer paragraphs, lines, sentences, then whitespace. `data/index.py` stores one cosine Chroma collection per season and records `model_id`, `schema_version`, `chunk_size`, `chunk_overlap`, `manifest_hash`, and `unit_count`. Any mismatch deletes and fully rebuilds the collection; retrieval requires opponent and season and filters weeks/source types/unit kinds in Chroma before similarity search.
- Embeddings are pinned. `ArcEmbedder` uses `EMBEDDING_MODEL` (VT ARC `Qwen3-Embedding-4B`, 2560 dimensions, verified); `HashingEmbedder` (`local-hash-v1-<dimensions>`) is offline/test-only. A query embedder whose model ID differs from the stored index model is refused rather than mixing vector spaces. `tests/` and the default `pytest` run never call VT ARC; use `--embedder hashing` for an offline index build.
- Stage 4 statistics in `stats/engine.py` are pure Python and byte-deterministic: bucket labels, row order, rounding, note wording, and the golden payload in `tests/fixtures/stats_golden_2026_SEA.json` are frozen. Changing any of them requires regenerating the golden with `UPDATE_GOLDEN=1 pytest tests/unit/test_stats_engine.py`. The engine analyzes one opponent's offense only, treats third-down attempts as scrimmage plays with a recorded outcome, and never imputes: missing split keys become explicit "Unknown" rows, missing yards/outcomes are excluded with notes, and duplicate play keys raise.
- Stage 6 LLM calls live in `llm/client.py`: retries are bounded (default 2, capped exponential backoff over timeouts, transport errors, and HTTP 408/429/5xx), every log line records only model, status, attempts, latency, and token counts (never prompts, completions, headers, or keys), and `complete_structured` allows exactly one repair attempt before raising `StructuredOutputError`. The six prompts in `llm/prompts/library.py` share one constitutional operating-rules block and one `PROMPT_VERSION` (`2026.11.0`); editing any prompt text requires bumping `PROMPT_VERSION` deliberately since it is stamped on agent outputs/reports and feeds `report_id`. Only `llm/prompts/blocks.py` renders data blocks, and `neutralize_tags` keeps untrusted text from closing or opening one. `llm/optimize.py` / `scripts/optimize_prompts.py` are design-time only — never import them from the runtime path — and tests must stay on `httpx.MockTransport`, never VT ARC.
