# AGENTS.md

## What this repo is

- `docs/` holds the **graded Quarto write-up** (the assignment artifact). The
  **implementation** it describes lives in `src/opponent_tendencies/`, `app.py`,
  `scripts/`, and `tests/`. Implementation code must stay out of `docs/`.
- Stages 0–11 of `PLAN.md` §4 are done: packaging/config, frozen Pydantic
  contracts (`models.py` + wire schemas in `llm/schemas.py`), nflverse ingestion
  into `data/seasons/<season>.json`, curated media + Chroma index under
  `data/media/` / `data/chroma`, deterministic stats (`stats/engine.py`),
  validation (`validation/checks.py`), the LLM layer (`llm/client.py`,
  `llm/prompts/`, offline `llm/optimize.py`), specialist agents
  (`agents/statistics.py`, `beat_reporter.py`, `coach_quote.py`),
  synthesis/verification (`agents/synthesis.py`, `agents/verify.py`), the
  orchestrator (`orchestrator.py`), the Streamlit workflow (`app.py`, `ui/`), the
  full-report/stats-only CLI (`report_cli.py`, `scripts/report.py`),
  content-addressed report caching (`cache.py`), and consistency plus
  chunking/retrieval evaluation (`evaluation.py`, `retrieval_eval.py`, and their
  scripts).
- `app.py` is the real UI entry point; `ui/runner.py` loads stores/index/client
  and calls the orchestrator, while `ui/render.py` renders contracts.
  `scripts/report.py` is the non-Streamlit entry point over the same
  `build_report`; `--stats-only` stops before credentials and the LLM. Tests
  inject fixtures and runners, so AppTest and CLI tests never touch the network
  or VT ARC.
- `README.md` mirrors this status (status banner, "What works today" table,
  roadmap). Update it when a stage completes so docs and tree stay consistent.

## Commands (run from the repo root)

Dependencies are already installed in `.venv` (Python 3.13.5; `pyproject.toml`
targets `>=3.12` and ruff/mypy target `py312`). Do not recreate the venv.

```bash
.venv/bin/python -m pytest -q                                  # full suite, offline
.venv/bin/python -m pytest -q tests/unit/test_stats_engine.py  # one module
.venv/bin/python -m pytest -q tests/unit/test_validation_checks.py -k retrieval
.venv/bin/python -m pytest -q -m integration                   # tmp_path + Chroma only
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy src tests scripts
.venv/bin/python scripts/refresh_data.py --season 2026   # nflverse -> data/seasons/2026.json
.venv/bin/python scripts/build_index.py --season 2026    # corpus -> data/media + Chroma
.venv/bin/python scripts/report.py --opponent BUF --weeks 5 6 7 8 9  # full report
.venv/bin/python scripts/report.py --opponent BUF --weeks 5 6 7 8 9 --stats-only
.venv/bin/python scripts/optimize_prompts.py             # design-time only
.venv/bin/python scripts/eval_consistency.py --opponent BUF --weeks 5 6 7 8 9  # live
.venv/bin/python scripts/eval_retrieval.py --opponent BUF --weeks 5 6 7 8 9    # offline
streamlit run app.py
```

- No CI or pre-commit: run the gates yourself (pytest → ruff → mypy; all green on
  HEAD). mypy is strict (`disallow_untyped_defs`), and pytest runs with
  `--strict-markers`/`--strict-config`, so register any new marker in
  `pyproject.toml` before using it.
- `pytest` never touches the network or VT ARC. nflverse fixtures in
  `tests/fixtures/` are trimmed real 2026 pulls injected through loader
  arguments; media/integration tests use the tracked `corpus/media_2026.json`
  with `HashingEmbedder` and `tmp_path`; LLM tests use `httpx.MockTransport`.
  Keep it that way when adding tests.
- `DATA_DIR` / `CHROMA_DIR` default to relative `data/` paths, so scripts assume
  the repo root is the CWD.
- Offline paths need no credentials (`scripts/report.py --stats-only`,
  `scripts/eval_retrieval.py` with its default hashing embedder). `get_settings()`
  fails fast with `ConfigurationError` unless `VT_ARC_BASE_URL`,
  `VT_ARC_API_KEY`, `VT_ARC_MODEL`, and `EMBEDDING_MODEL` are set; data-only code
  should call `resolve_data_dir()` / `resolve_chroma_dir()` instead. A local
  `.env` exists but is gitignored — never print, commit, or copy its values into
  logs/tests.
- The golden stats payload is frozen in
  `tests/fixtures/stats_golden_2026_SEA.json`; regenerate only deliberately with
  `UPDATE_GOLDEN=1 .venv/bin/python -m pytest tests/unit/test_stats_engine.py`.

## Docs deliverable (`docs/`)

- `docs/final_project_reqs_and_arch_zachary_and_adam.qmd` — the only editable source
- `docs/final_project_reqs_and_arch_zachary_and_adam.pdf` — rendered deliverable (committed)
- `docs/opponent-tendencies-sequence-diagram.png` — rendered sequence diagram (committed)

```bash
quarto render docs/final_project_reqs_and_arch_zachary_and_adam.qmd
quarto inspect docs/final_project_reqs_and_arch_zachary_and_adam.qmd  # config check
```

- Quarto 1.8.24 is on `PATH` and at `/opt/quarto/bin/quarto`; the PDF is written
  next to the `.qmd`. It is graded, so re-render and commit it whenever the
  `.qmd` changes.
- The PNG has no generator in this repo: regenerate it externally from the Mermaid
  `sequenceDiagram` block in the `.qmd` and keep the two in sync. It is
  referenced by relative filename, so it must stay in `docs/`.
- Requirements are numbered 1–30 and issues 1–18; later sections ("LLM / RAG
  Issues", "Solution Mitigation") reference those numbers. Update every
  cross-reference when renumbering.
- Assignment text states the write-up must be produced without AI assistance.
  Treat content generation accordingly and limit agent help to repo mechanics
  unless the user says otherwise.

## Invariants and gotchas

- Deterministic Python owns all math, filtering, sorting, formatting, duplicate
  checks, and source-ID checks; the LLM only interprets, extracts, and
  synthesizes (`PLAN.md` §1). Keep new code on the correct side of that split.
- Stage 1 contracts in `opponent_tendencies.models` are frozen. Change them
  deliberately, update `llm/schemas.py`, and keep `tests/unit/test_models.py`
  green; server-owned identity is stamped by Python.
- Ingestion (`data/nflverse.py`) pins `PLAY_COLUMNS` / `GAME_COLUMNS`; a missing
  column raises `MissingColumnsError` instead of being skipped. `nflreadpy`
  returns Polars, 2026 uses `WAS` (aliased to `WSH`), and nflverse fills
  third-down flags on every row, so the normalizer keeps them only on down 3.
  Re-spike the client before changing the column list.
- `data/seasons/<season>.json` is generated: `SeasonStore` upserts by `game_id` /
  `(game_id, play_id)` in one atomic write and does not rewrite unchanged data, so
  a double refresh is byte-identical. Never hand-edit it; delete and re-run
  `scripts/refresh_data.py`.
- `data/media/<season>.json` is generated atomically from the tracked
  `corpus/media_2026.json`; `MediaStore` replaces each incoming item's whole quote
  set, so removed speakers cannot leave stale quotes. Never hand-edit it; delete
  and re-run `scripts/build_index.py`. Refetching identical text keeps the index
  manifest stable; only title/body/quote/filter changes force a rebuild.
- `data/media_ingest.py` enforces the source allowlist (publisher, HTTPS
  allowlisted domain, media type, trusted authors). Every quote must be an exact
  character span of its stored body, corpus documents reject unknown fields, and
  media/quote IDs derive from `source_id` + URL and media/speaker/text/occurrence
  respectively. Build changed records by revalidation (`tests/media_fixtures.py`
  helpers), not `model_copy` alone.
- `data/chunking.py` owns all chunk boundaries and IDs (`make_chunk_id`); defaults
  are 1200/200 characters, cutting paragraphs → lines → sentences → whitespace.
  `data/index.py` stores one cosine Chroma collection per season recording
  `model_id`, `schema_version`, `chunk_size`, `chunk_overlap`, `manifest_hash`,
  and `unit_count`; any mismatch deletes and fully rebuilds it. Retrieval requires
  opponent and season and filters weeks/source types/unit kinds in Chroma before
  similarity search.
- Embeddings are pinned. `ArcEmbedder` uses `EMBEDDING_MODEL` (VT ARC
  `Qwen3-Embedding-4B`, 2560 dimensions, verified); `HashingEmbedder`
  (`local-hash-v1-<dimensions>`) is offline/test-only. A query embedder whose
  model ID differs from the stored index model is refused rather than mixing
  vector spaces; use `--embedder hashing` for an offline index build.
- `stats/engine.py` is pure Python and byte-deterministic: bucket labels, row
  order, rounding, note wording, and the golden payload are frozen. It analyzes
  one opponent's offense only, treats third-down attempts as scrimmage plays with
  a recorded outcome, and never imputes: missing split keys become explicit
  "Unknown" rows, missing yards/outcomes are excluded with notes, duplicate play
  keys raise.
- `validation/checks.py` exposes pure checks returning `ValidationResult`s with
  `CheckCode` issues instead of raising: `validate_selection`, `check_plays`,
  `check_media_alignment`, `check_retrieval`, `check_evidence`,
  `check_stat_tables`, `check_records`.
- `llm/client.py`: retries are bounded (default 2, capped exponential backoff over
  timeouts, transport errors, empty non-truncated replies, and HTTP 408/429/5xx);
  log lines record only model, status, attempts, latency, and token counts
  (never prompts, completions, headers, or keys). Provider-side reasoning is
  disabled by default for runtime calls (`chat_template_kwargs.enable_thinking=
  false`) because the pinned model otherwise spends the completion cap before
  emitting JSON; `VT_ARC_ENABLE_THINKING=true` opts back in. An empty
  `finish_reason=length` reply is not retried; a truncated structured reply
  raises `LLMResponseError` with the finish reason/token counts instead of
  spending the one repair. VT ARC's proxy rejects non-streaming requests above
  `DEFAULT_BUFFERED_MAX_TOKENS` (8,000) with HTTP 413, so `chat` sets
  `stream=true` above that cap and accumulates SSE deltas/usage (a buffered JSON
  body is still accepted if a gateway ignores streaming). Otherwise
  `complete_structured` allows exactly one repair attempt before raising
  `StructuredOutputError`.
- `llm/prompts/library.py` holds six prompts (`statistics`, `beat_reporter`,
  `coach_quote`, `synthesis`, `verifier`, `follow_up`) sharing one constitutional
  operating-rules block and one `PROMPT_VERSION` (`2026.11.0`). The version is
  stamped on agent outputs/reports and feeds `report_id`, so bump it deliberately
  on any prompt edit. `Prompt.render` fails unless the caller supplies exactly the
  declared context keys, and only `llm/prompts/blocks.py` renders data blocks;
  `neutralize_tags` rewrites literal block tags to `⟨…⟩` look-alikes so untrusted
  text cannot open or close one.
- `llm/optimize.py` / `scripts/optimize_prompts.py` are design-time only — never
  import them from the runtime path.
- `agents/common.py` owns the evidence catalogs and the wire→contract conversion:
  agents may only cite IDs built by `build_stat_evidence`, `build_media_evidence`,
  or `build_quote_evidence`, and unknown IDs/weeks are dropped with warnings, not
  trusted. Suppressed-table citations and low-confidence findings set
  `weak_evidence=True` (and downgrade `HIGH` to `MEDIUM`); CQA coach
  explanations/adjustments are forced to `treat_as_context=True`. Empty BRA/CQA
  retrieval is an explicit completed limitation; a grounding failure is
  `UNAVAILABLE`.
- `agents/synthesis.py` and `agents/verify.py` keep their claims on the same
  deterministic split: synthesis drops ungrounded agreements/conflicts/gaps with
  `SYNTHESIS_UNGROUNDED_CLAIM` and adds unavailable/low-confidence gaps itself;
  verification keeps `supported` claims only with attached snippet IDs, replaces
  the statement on `partially_supported`, and removes unsupported/contradicted/
  missing-verdict claims with `WeakEvidence` + warnings. Never let synthesis
  rewrite `StatTable`s.
- `orchestrator.run_report` is the only report entry point. It validates and
  computes stats once, runs STA/BRA/CQA concurrently, re-checks citations, and
  stamps the `Report`. STA/synthesis/verification/validation/deadline/budget
  failures are hard failures with no report; BRA/CQA failure or timeout degrades
  to an explicit unavailable section with a warning and partial report. Timeouts
  are bounded `Future.result` waits (`shutdown(wait=False, cancel_futures=True)`);
  a worker that produces no result within its budget is the stall detector — it
  is abandoned and follows the same failure matrix, so no stage can wait forever.
  `DEADLINE_EXCEEDED` also covers step-cap/token-budget overruns. Progress goes
  through `on_progress(RunProgress)` from the main thread; `data_revision` combines
  the season and media store revisions.
- `cache.py` is content-addressed: `(opponent, sorted weeks, prompt_version,
  data_revision)` hashes to the filename, entries are canonical JSON envelopes
  (`schema_version` + exact key + `Report`), reads revalidate the key and report,
  and writes use `atomic_write_text`. Only successful reports are stored; never
  serve a stale report after a data or prompt change.
- `ui/runner.build_report` loads stores/index/client, computes the cache key, and
  calls the orchestrator; `ui/answer_follow_up` reuses the stored report and
  `follow_up` prompt, summarizes older turns deterministically, and validates
  every cited ID against `report.evidence` (unknown IDs are removed and
  `supported` is forced false). `app.py` only orchestrates widgets, session
  state, and `ui/render.py`; AppTest injects `report_runner`/`follow_up_runner`.
- `report_cli.py` / `scripts/report.py` are thin over `ui.runner.build_report`:
  they parse/validate arguments, expose the pipeline budgets, and format the
  frozen report plus telemetry. `--stats-only` bypasses `get_settings`, the
  index, and the LLM and calls `analyze_opponent` directly; `--no-cache` must
  pass `use_cache=False` rather than deleting entries.
- `evaluation.py` is pure and compares frozen reports: `tendency_key` normalizes
  category/statement, the consensus top list and `ceil(0.8 * runs)` threshold
  encode the 4/5 acceptance rule, and direction flips, changed stat tables, or
  untraceable quotes fail. `scripts/eval_consistency.py` is the only live caller
  and is never imported by runtime code.
- `retrieval_eval.py` reuses `chunk_text`/`evaluate_chunking` and
  `check_retrieval`; it must not duplicate chunk boundaries or week/opponent
  filtering. `scripts/eval_retrieval.py` defaults to the offline
  `HashingEmbedder`; `--embedder arc` evaluates the pinned persistent index. A
  retrieval sample passes only when every returned unit validates and, when the
  store has matching sources, at least one unit is week-aligned.
