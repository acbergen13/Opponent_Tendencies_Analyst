# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` is the detailed source of truth for invariants and gotchas; `PLAN.md` holds the design and `README.md` the status table. Read `AGENTS.md` before changing a subsystem. This file is the short version.

## What this is

NFL opponent-scouting assistant. Deterministic Python computes all statistics; an LLM (VT ARC, OpenAI-compatible) only interprets, extracts, and synthesizes, citing evidence IDs. `docs/` holds the graded Quarto write-up (no implementation code there); the implementation is in `src/opponent_tendencies/`, `app.py`, `scripts/`, `tests/`. The assignment says the write-up must be produced without AI assistance, so limit agent help in `docs/` to repo mechanics unless told otherwise.

## Commands (repo root; use `.venv/bin/python`, don't recreate the venv)

```bash
.venv/bin/python -m pytest -q                                  # full suite, offline
.venv/bin/python -m pytest -q tests/unit/test_stats_engine.py  # one module
.venv/bin/python -m pytest -q tests/unit/test_validation_checks.py -k retrieval
.venv/bin/python -m pytest -q -m integration                   # tmp_path + Chroma only
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy src tests scripts                     # strict
.venv/bin/python scripts/refresh_data.py --season 2026         # nflverse -> data/seasons/2026.json
.venv/bin/python scripts/build_index.py --season 2026          # corpus -> data/media + Chroma
.venv/bin/python scripts/report.py --opponent BUF --weeks 5 6 7 8 9 [--stats-only]
streamlit run app.py
quarto render docs/final_project_reqs_and_arch_zachary_and_adam.qmd   # re-render and commit the PDF
```

No CI or pre-commit: run pytest, ruff, then mypy yourself. Register new pytest markers in `pyproject.toml` (`--strict-markers`). Scripts assume the repo root is the CWD. Regenerate the golden stats fixture only deliberately: `UPDATE_GOLDEN=1 .venv/bin/python -m pytest tests/unit/test_stats_engine.py`.

## Architecture

Pipeline: `data/nflverse.py` → `data/play_store.py` (`data/seasons/<season>.json`) and `corpus/media_2026.json` → `data/media_ingest.py`/`media_store.py` + `data/index.py` (Chroma) → `orchestrator.run_report`:

1. validate selection and compute stats once (`stats/engine.py`, pure, byte-deterministic)
2. run the statistics, beat-reporter, and coach-quote agents (`agents/`) concurrently
3. `agents/synthesis.py`, then `agents/verify.py`
4. re-check citations and stamp the frozen `Report` (`models.py`)

STA, synthesis, and verification failures are hard failures. BRA and CQA failures degrade to an explicit unavailable section.

- `llm/`: `client.py` (bounded retries, streaming above 8,000 max tokens, one structured-output repair), `prompts/` (six prompts sharing one `PROMPT_VERSION` that feeds `report_id` and the cache key), `schemas.py` (wire schemas, converted to contracts in `agents/common.py`), and `optimize.py` (design-time only, never import at runtime).
- `data/live_media.py`: at report time, fetches current team articles from allowlisted RSS feeds and stores them (see AGENTS.md); `LIVE_MEDIA=false` disables.
- `cache.py`: content-addressed by opponent, weeks, prompt version, and data revision. Only successful reports are stored.
- `ui/runner.py` is the shared entry used by `app.py` and `report_cli.py`/`scripts/report.py`. `ui/render.py` renders contracts.
- `validation/checks.py`: pure checks returning `ValidationResult`s, not exceptions.
- `evaluation.py` and `retrieval_eval.py` back `scripts/eval_*.py`; the live consistency eval is never imported by runtime code.

## Key invariants

- Keep math, filtering, sorting, and ID/duplicate checks in Python; the LLM never does them. Agents may cite only evidence IDs built in `agents/common.py`. Unknown IDs are dropped, not trusted.
- `models.py` contracts are frozen. Change them deliberately and update `llm/schemas.py` and `tests/unit/test_models.py`. Bump `PROMPT_VERSION` on any prompt edit.
- `data/seasons/*.json` and `data/media/*.json` are generated. Never hand-edit them; delete and re-run the refresh or build script.
- Embeddings are pinned. A query embedder whose model ID differs from the stored index is refused. Use `--embedder hashing` for offline builds.
- Tests must stay offline: inject fixtures through loader arguments, use `httpx.MockTransport` for the LLM, and use `HashingEmbedder` with `tmp_path` for media/Chroma tests.
- Credentials: `Settings.from_env()` / `get_settings()` needs `VT_ARC_BASE_URL`, `VT_ARC_API_KEY`, `VT_ARC_MODEL`, and `EMBEDDING_MODEL`. Data-only paths and `--stats-only` don't. Never print or commit `.env` values.
- When a stage or feature changes status, update the README status table to match.
