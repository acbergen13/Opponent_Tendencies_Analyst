# PLAN.md — Opponent Tendencies Analyst

## 0. Context

- Source of truth: `docs/final_project_reqs_and_arch_zachary_and_adam.qmd` (requirements 1–30, issues 1–18, architecture, mitigations).
- This repo is currently docs-only: no manifest, tests, lint config, or CI. `docs/` is the graded deliverable; implementation code should live outside it.
- Environment: Python 3.13 (`/opt/anaconda3`); `streamlit`, `pydantic`, `pandas`, `pytest`, `httpx` installed; `chromadb`, `openai`, `nflreadpy` not installed.
- `PLAN.md` is engineering planning only. The assignment's no-AI note still applies to the written deliverable.

## 1. Design invariants

- Deterministic Python owns all math, filtering, sorting, formatting, duplicate checks, and source-ID checks. The LLM only interprets, extracts, and synthesizes.
- Every factual claim carries evidence resolving to a stored play, article, breakdown, or quote.
- Sources must match the selected opponent and 2026 regular-season weeks. Earlier-2026 reporting may be labeled "background"; prior seasons are rejected.
- Four agents: Statistics (STA), Beat-Reporter (BRA), Coach-Quote (CQA), Synthesis (SYN). STA/BRA/CQA run in parallel; SYN fans in. Claim verification is a separate LLM call, not a fifth agent.
- Failure policy: STA failure or overall deadline → hard failure, no report. BRA/CQA failure/timeout → mark section unavailable, warn, continue.
- All LLM calls use the VT ARC OpenAI-compatible endpoint with narrow constitutional prompts. Retrieved text is delimited and labeled as data.

## 2. Proposed layout

```text
PLAN.md
pyproject.toml
.env.example
app.py                          # Streamlit entrypoint
src/opponent_tendencies/
  config.py                     # env + settings, fail-fast validation
  models.py                     # Pydantic contracts for all stages
  data/
    nflverse.py                 # PBP fetch wrapper
    play_store.py               # JSON store, upsert by game_id+play_id
    media_ingest.py             # source allowlist + connectors
    media_store.py              # metadata/full text/quotes
    index.py                    # ChromaDB build/retrieve/rebuild detection
  stats/engine.py               # deterministic tendency calculations
  validation/checks.py          # source existence, opponent/week alignment
  llm/
    client.py                   # VT ARC client, retries, timeouts, logging
    schemas.py                  # JSON schemas for structured outputs
    prompts/                    # versioned agent + verifier + follow-up prompts
  agents/
    statistics.py
    beat_reporter.py
    coach_quote.py
    synthesis.py
    verify.py
  orchestrator.py               # fan-out/fan-in, deadlines, failure matrix
  ui/app.py                     # selection, progress, report, chat
tests/
  fixtures/                     # cached PBP + media corpus + golden stats
  unit/
  integration/
scripts/
  refresh_data.py
  build_index.py
  optimize_prompts.py           # offline prompt optimization (req 18)
  eval_consistency.py           # 5-run reliability evaluation
data/                           # gitignored: JSON store, Chroma index, caches
```

## 3. Tooling

- Use `pip` + `venv` (no `uv` installed). Pin dependencies in `pyproject.toml`; add `ruff` + `mypy` + `pytest`.
- Commands:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -e ".[dev]"`
  - `streamlit run app.py`
  - `pytest -q`
  - `ruff check . && mypy src`
- `.env.example` variables: `VT_ARC_BASE_URL`, `VT_ARC_API_KEY`, `VT_ARC_MODEL`, `EMBEDDING_MODEL`, `DATA_DIR`, `CHROMA_DIR`.
- Never commit `.env`, API keys, or generated data/index directories.

## 4. Development stages

### Stage 0 — Skeleton and tooling
- Create package skeleton, `pyproject.toml`, `.env.example`, `.gitignore`, and empty test layout.
- Verify dependency install on Python 3.13; fall back to a 3.12 environment if `chromadb` or the nflverse client is incompatible.
- **Exit:** package imports, `pytest` runs, config fails clearly when `VT_ARC_API_KEY` is missing.
- **Reqs:** foundation for all.

### Stage 1 — Contracts and schemas
- Define Pydantic models: `Game`, `Play`, `MediaItem`, `Quote`, `EvidenceRef`, `StatTable`, `AgentFinding`, `AgentOutput`, `Report`, `ValidationResult`, `RunStatus`.
- `AgentOutput` must always include opponent, games, weeks, tendencies, observations, evidence, limitations (req 25).
- Stable IDs: `game_id`, `play_id`, `media_id`, `quote_id`, `evidence_id`.
- Generate JSON schemas for LLM structured outputs.
- **Exit:** round-trip/schema tests; contracts frozen before agent work.
- **Reqs:** 5, 25.

### Stage 2 — Play-by-play ingestion and JSON store
- Spike the maintained nflverse Python client; verify actual 2026 regular-season columns.
- Normalize fields needed for stats: game/week/team, down, distance, field position, play type, yards, conversions; discard unneeded payload.
- Implement atomic JSON store with upsert keyed by `(game_id, play_id)`; refresh returns inserted/skipped counts.
- Derive alphabetical team list and played regular-season weeks only (exclude preseason, future, postseason).
- **Exit:** double-refresh idempotency test; missing-column and empty-week handling; fixture-based.
- **Reqs:** 2, 3, 4.

### Stage 3 — Media ingestion, metadata store, and ChromaDB
- Build a source allowlist and connector interface for trusted sports sites/reporters; start with a cached/curated fixture corpus, then add live connectors only where terms permit.
- Store article/breakdown/quote text plus author, publisher, date, URL, opponent, season, week, game, source type, and speaker/coach.
- Preserve quotes verbatim; never rewrite direct quotes.
- Chunk with configurable size/overlap; embed with one pinned model; persist the Chroma index and record the model ID in index metadata.
- Rebuild only when a content hash/manifest changes; otherwise reuse.
- Retrieval must filter by opponent, season, week range, and source type before similarity search.
- **Exit:** retrieval fixtures return correct opponent/week sources; index-reuse and embedding-consistency tests; chunking evaluation.
- **Reqs:** 6, 7, 8, 12, 13.

### Stage 4 — Deterministic statistics engine
- Implement pure Python/pandas functions: run rate, pass rate, yards per play, third-down conversion; splits by down, distance bucket, and field position (including red zone/opponent territory).
- Report `n` plays per split; suppress or flag splits below a minimum sample threshold.
- Detect duplicate plays and missing values; explain gaps, never impute.
- Add tendency-change comparison across week windows (req 15).
- **Exit:** golden-file tests assert byte-identical tables for identical inputs; edge cases for empty/short samples.
- **Reqs:** 11, 15, 24.

### Stage 5 — Validation layer
- Check every game, play, media item, and quote ID resolves to stored source data.
- Enforce opponent/week alignment; label earlier-2026 sources as background; reject prior seasons.
- Validate schemas, duplicates, missing values, and retrieval results before any model sees them.
- Flag weak evidence (small samples, missing sources) for human review.
- Return structured errors/warnings rather than raising vague exceptions.
- **Exit:** table-driven tests for each rejection and warning path.
- **Reqs:** 19, 21, 22, 24, 26, 27.

### Stage 6 — LLM client, prompts, and offline prompt optimization
- Implement an OpenAI-compatible client for VT ARC using environment config; bounded retries, per-call timeout, latency/token logging, no secrets in logs.
- Write narrow constitutional prompts for STA, BRA, CQA, SYN, verifier, and follow-up.
- Use delimiters, provenance labels, and instruction hierarchy so retrieved text is data, not instructions.
- Parse structured outputs with Pydantic; allow one bounded repair attempt on invalid output.
- Add `scripts/optimize_prompts.py` for offline prompt optimization; store the selected prompt version and evaluation results. Do not run it at runtime.
- **Exit:** mocked contract tests; invalid JSON triggers bounded repair; report records prompt version.
- **Reqs:** 10, 18, 23, 29.

### Stage 7 — Specialist agents
- STA: call deterministic stats tools only; LLM interprets/report results.
- BRA: retrieve filtered articles and written film breakdowns; extract tendencies, formations, play calling, and recent changes with citations.
- CQA: retrieve filtered post-game coach quotes; preserve verbatim wording with coach, date, and source.
- All three emit the same `AgentOutput` schema and explicit limitations/empty results.
- **Exit:** fixture-driven tests; every claim has an evidence ref; no evidence produces an explicit limitation, not fabrication.
- **Reqs:** 10, 11, 12, 13, 25.

### Stage 8 — Synthesis and claim verification
- Merge validated findings; state agreement, conflict, and gaps; compare week windows; treat coach quotes as context, not proof.
- Produce quantitative tables and qualitative notes with citations.
- Run a separate LLM verification call against source snippets; correct/remove unsupported claims.
- Python re-checks calculations, format, and source IDs after synthesis.
- Flag weak evidence for human review.
- **Exit:** contradictory fixture yields a conflict note; unsupported claim is removed; all citations resolve; stats unchanged by LLM.
- **Reqs:** 14, 15, 16, 19, 27, 28.

### Stage 9 — Orchestrator and failure handling
- Implement `run_report(opponent, weeks)`: validate → parallel STA/BRA/CQA → fan-in validation → SYN → verify → report.
- Enforce per-agent timeouts, overall deadline, step cap, token budget, and stall detection.
- Apply the failure matrix exactly: STA/overall → hard failure; BRA/CQA → unavailable section + warning + partial report.
- Bound retries; never loop indefinitely on missing evidence.
- **Exit:** failure-injection tests with mocked agents; no report on hard failure; partial report on degraded failure.
- **Reqs:** 9, 10, 22, 30.

### Stage 10 — Streamlit UI and session state
- Alphabetical team dropdown; multiselect of played regular-season weeks only.
- Refresh button showing inserted/skipped counts.
- Selection summary: opponent, weeks, games, dates, scores, data used.
- Build-report button with per-agent progress; prevent concurrent duplicate runs.
- Render quantitative tables, qualitative notes, citations, warnings, unavailable sections, and weak-evidence banners.
- Store opponent, weeks, report, and evidence in `st.session_state`.
- Follow-up chat reuses SYN and stored evidence without a full re-run; summarize older turns to control context.
- **Exit:** Streamlit AppTest or script-level UI tests; session persistence verified; hard-failure message shown.
- **Reqs:** 1, 4, 5, 9, 16, 17, 22, 30.

### Stage 11 — Evaluation, hardening, and docs
- Build unit, integration (fixtures), mocked-LLM contract, and end-to-end tests.
- Add `scripts/eval_consistency.py`: same inputs 5 times → top-5 tendency overlap ≥80% in 4/5 runs, no direction contradictions; stats tables identical; quotes 100% traceable.
- Add prompt/chunking evaluation on a small opponent sample.
- Measure latency and token cost; cache reports by `(opponent, weeks, prompt_version, data_revision)`.
- Add prompt-injection fixtures and source/instruction separation tests.
- Write README/runbook: setup, refresh, index rebuild, evaluation commands.
- Update `AGENTS.md` once app code exists (current file correctly describes the docs-only state).
- **Exit:** all acceptance checks pass; report completes within the target time; failure modes documented.
- **Reqs:** 20, 23, 28, 30.

## 5. Verification matrix

| Check | Type | Gate |
|---|---|---|
| Stats functions and splits | Unit + golden files | Exact deterministic output |
| Play upsert / refresh | Integration | No duplicates after repeat refresh |
| Media retrieval | Integration + fixtures | Correct opponent/week, no prior seasons |
| Agent outputs | Contract | Valid `AgentOutput`, resolvable evidence |
| Failure matrix | Integration (mocks) | Hard vs degraded behavior matches req 30 |
| Report citations | End-to-end | 100% resolve to stored sources |
| 5-run narrative consistency | Live evaluation | ≥80% top-5 overlap in 4/5, no direction conflicts |
| UI flow | Streamlit AppTest | Selection → report → follow-up persists |

## 6. Milestones

| Milestone | Stages | Outcome |
|---|---|---|
| M1 Data foundation | 0–3 | Real PBP + media corpus + persistent index |
| M2 Deterministic analysis | 4–5 | Trustworthy stats and validated inputs |
| M3 Agent pipeline | 6–9 | Parallel agents, synthesis, verification, failure handling |
| M4 User interface | 10 | End-to-end Streamlit workflow with chat |
| M5 Reliability | 11 | Determinism, traceability, latency, hardening, docs |

Recommended vertical slices: stats-only CLI after Stage 5; full-report CLI after Stage 9; then UI.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| 2026 nflverse schema/packaging changes | Early spike; pin version; fixture snapshots; verify actual columns |
| Media scraping is brittle / ToS-sensitive | Curated fixture corpus first; connector abstraction; allowlist; cache raw text |
| No VT ARC embedding endpoint | Pin a local embedding model fallback; record model ID; rebuild when changed |
| Python 3.13 dependency incompatibility | Isolated venv; pin versions; fall back to 3.12 |
| LLM nondeterminism | Low temperature + structured outputs + verification; enforce determinism only where required (stats/quotes) |
| Runtime exceeds "few minutes" | Parallel agents, caching, bounded retrieval, per-agent deadlines |
| Prompt injection in retrieved text | Delimiters, provenance labels, instruction hierarchy, tool allowlist, schema validation |

## 8. Requirement traceability

| Requirement(s) | Stage(s) |
|---|---|
| 1 | 10 |
| 2, 3 | 2 |
| 4 | 2, 10 |
| 5 | 1, 10 |
| 6 | 3, 7 |
| 7, 8 | 3 |
| 9 | 9, 10 |
| 10 | 6–10 |
| 11 | 4, 7 |
| 12, 13 | 3, 7 |
| 14 | 8 |
| 15 | 4, 8 |
| 16 | 8, 10 |
| 17 | 10 |
| 18 | 6 |
| 19 | 5, 8 |
| 20 | 11 |
| 21 | 5 |
| 22 | 5, 9, 10 |
| 23 | 3, 6 |
| 24 | 4, 5 |
| 25 | 1, 7 |
| 26 | 3, 5 |
| 27 | 5, 8, 10 |
| 28 | 6, 8 |
| 29 | 6 |
| 30 | 9, 10 |

## 9. Decisions to confirm

1. **PLAN.md location:** repo root (recommended) rather than `docs/`, so the graded deliverable folder stays clean.
2. **Media strategy:** start with a curated/cached corpus behind a connector interface (recommended), then add live fetching only for sources whose terms allow it.
3. **Media metadata store:** SQLite for queryable metadata + ChromaDB for vectors (recommended), versus keeping metadata in JSON.
4. **Embeddings:** use VT ARC embeddings if available; otherwise pin a local model and record it in index metadata.
5. **Model alias:** confirm which VT ARC model ID the app should target (the OpenCode config lists aliases, but the app should read `VT_ARC_MODEL` from env).
