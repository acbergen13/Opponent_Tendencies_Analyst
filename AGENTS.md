# AGENTS.md

## What this repo is

Two things live here:

1. **The graded Quarto deliverable** in `docs/` (design write-up for the NFL
   "Opponent Tendencies Analyst"). This is the assignment artifact.
2. **The implementation** described by `PLAN.md`: a Python package under
   `src/opponent_tendencies/` with a Streamlit app. Development proceeds stage
   by stage in `PLAN.md` §4. Stages 0–2 are done: packaging/config, the frozen
   Pydantic contracts in `models.py` with the LLM JSON schemas in
   `llm/schemas.py`, and nflverse play-by-play ingestion into the atomic JSON
   store at `data/seasons/<season>.json`. Stats, validation, agents, and UI are
   not implemented yet — do not assume modules that the plan lists but the tree
   lacks.

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
