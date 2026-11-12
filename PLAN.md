# PLAN.md — Opponent Tendencies Analyst

## 0. Context

- Source of truth: `docs/final_project_reqs_and_arch_zachary_and_adam.qmd` (requirements 1–30, issues 1–18, architecture, mitigations).
- Stages 0–11 are implemented: packaging/config, frozen contracts, nflverse play-by-play ingestion, curated media ingestion plus a persistent Chroma index, the deterministic statistics engine, the structured validation layer, the LLM layer (VT ARC chat client, versioned constitutional prompts, offline prompt optimization), specialist agents, synthesis/verification, the orchestrator, the Streamlit report/follow-up UI, content-addressed report caching, and the consistency evaluation harness. `docs/` remains the graded deliverable; implementation code lives outside it.
- Environment: Python 3.13 in a local `.venv`; `chromadb`, `nflreadpy`, `streamlit`, `pydantic`, `pytest`, `httpx`, `ruff`, and `mypy` are pinned in `pyproject.toml` and verified on 3.13.
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
    jsonio.py                   # canonical JSON + atomic writes (shared)
    media_ingest.py             # source allowlist + connectors
    media_store.py              # metadata/full text/quotes
    chunking.py                 # deterministic word-aware chunking + evaluation
    index.py                    # ChromaDB build/retrieve/rebuild detection
  stats/engine.py               # deterministic tendency calculations
  validation/checks.py          # source existence, opponent/week alignment
  llm/
    client.py                   # VT ARC client, retries, timeouts, logging
    embeddings.py               # pinned VT ARC / offline hashing embedders
    schemas.py                  # JSON schemas for structured outputs
    prompts/                    # versioned agent + verifier + follow-up prompts
  agents/
    common.py                   # evidence catalogs, conversion, telemetry
    statistics.py
    beat_reporter.py
    coach_quote.py
    synthesis.py
    verify.py
  orchestrator.py               # fan-out/fan-in, deadlines, failure matrix
  report_cli.py                 # full-report + stats-only command line
  cache.py                      # content-addressed report cache
  evaluation.py                 # deterministic 5-run consistency checks
  retrieval_eval.py             # chunking + filtered-retrieval evaluation
  ui/
    __init__.py
    runner.py                   # data/index/client loading, cache, follow-ups
    render.py                   # selection/report/failure/progress rendering
tests/
  fixtures/                     # cached PBP + media corpus + golden stats
  unit/
  integration/
scripts/
  refresh_data.py
  build_index.py
  report.py                     # full report or --stats-only from the CLI
  optimize_prompts.py           # offline prompt optimization (req 18)
  eval_consistency.py           # 5-run reliability evaluation
  eval_retrieval.py             # offline/live chunking + retrieval evaluation
corpus/                         # tracked curated media corpus (fixture input)
data/                           # gitignored: JSON stores, Chroma index, caches
```

## 3. Tooling

- Use `pip` + `venv` (no `uv` installed). Pin dependencies in `pyproject.toml`; add `ruff` + `mypy` + `pytest`.
- Commands:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -e ".[dev]"`
  - `streamlit run app.py`
  - `pytest -q`
  - `ruff check . && mypy src`
- `.env.example` variables: `VT_ARC_BASE_URL`, `VT_ARC_API_KEY`, `VT_ARC_MODEL`, `VT_ARC_ENABLE_THINKING` (optional, default `false`), `EMBEDDING_MODEL`, `DATA_DIR`, `CHROMA_DIR`.
- Never commit `.env`, API keys, or generated data/index directories.

## 4. Development stages

### Stage 0 — Skeleton and tooling ✅
- Create package skeleton, `pyproject.toml`, `.env.example`, `.gitignore`, and empty test layout.
- Verify dependency install on Python 3.13; fall back to a 3.12 environment if `chromadb` or the nflverse client is incompatible.
- **Exit:** package imports, `pytest` runs, config fails clearly when `VT_ARC_API_KEY` is missing.
- **Reqs:** foundation for all.

### Stage 1 — Contracts and schemas ✅
- Define Pydantic models: `Game`, `Play`, `MediaItem`, `Quote`, `EvidenceRef`, `StatTable`, `AgentFinding`, `AgentOutput`, `Report`, `ValidationResult`, `RunStatus`.
- `AgentOutput` must always include opponent, games, weeks, tendencies, observations, evidence, limitations (req 25).
- Stable IDs: `game_id`, `play_id`, `media_id`, `quote_id`, `evidence_id`.
- Generate JSON schemas for LLM structured outputs.
- **Exit:** round-trip/schema tests; contracts frozen before agent work.
- **Reqs:** 5, 25.

### Stage 2 — Play-by-play ingestion and JSON store ✅
- Spike the maintained nflverse Python client; verify actual 2026 regular-season columns.
- Normalize fields needed for stats: game/week/team, down, distance, field position, play type, yards, conversions; discard unneeded payload.
- Implement atomic JSON store with upsert keyed by `(game_id, play_id)`; refresh returns inserted/skipped counts.
- Derive alphabetical team list and played regular-season weeks only (exclude preseason, future, postseason).
- **Exit:** double-refresh idempotency test; missing-column and empty-week handling; fixture-based.
- **Reqs:** 2, 3, 4.

### Stage 3 — Media ingestion, metadata store, and ChromaDB ✅
- Build a source allowlist and connector interface for trusted sports sites/reporters; start with a cached/curated fixture corpus, then add live connectors only where terms permit.
- Store article/breakdown/quote text plus author, publisher, date, URL, opponent, season, week, game, source type, and speaker/coach.
- Preserve quotes verbatim; never rewrite direct quotes.
- Chunk with configurable size/overlap; embed with one pinned model; persist the Chroma index and record the model ID in index metadata.
- Rebuild only when a content hash/manifest changes; otherwise reuse.
- Retrieval must filter by opponent, season, week range, and source type before similarity search.
- **Exit:** retrieval fixtures return correct opponent/week sources; index-reuse and embedding-consistency tests; chunking evaluation.
- **Implementation notes:**
  - Media lives in the atomic, canonical store `data/media/<season>.json`; an incoming item replaces its whole quote set so corrected or removed speakers cannot leave stale quotes.
  - `MediaItem` IDs derive from `source_id` + URL and `content_hash` from title + body. The index manifest also covers quote text and filter metadata but excludes `retrieved_at`, so refetching identical text reuses the index.
  - The allowlist checks publisher, HTTPS domain, media type, and trusted authors; every quote must be an exact span of the stored body.
  - Chunking defaults to 1200/200 characters and prefers paragraph, line, sentence, then whitespace boundaries.
  - The index is one cosine Chroma collection per season (`media_<season>`) recording `model_id`, `schema_version`, `chunk_size`, `chunk_overlap`, `manifest_hash`, and `unit_count`; any mismatch deletes and rebuilds the collection.
  - Coach-quote media index exact quote spans; other media index overlapping chunks plus exact quote units.
  - `ArcEmbedder` uses `EMBEDDING_MODEL` (VT ARC `Qwen3-Embedding-4B` is verified at 2560 dimensions); `HashingEmbedder` is an explicit offline/test provider. A query embedder whose model ID differs from the stored one is refused.
  - Retrieval requires opponent and season, then filters exact weeks or a week range plus source types/unit kinds before similarity search. Build with `python scripts/build_index.py --season 2026`.
- **Reqs:** 6, 7, 8, 12, 13.

### Stage 4 — Deterministic statistics engine ✅
- Implement pure Python/pandas functions: run rate, pass rate, yards per play, third-down conversion; splits by down, distance bucket, and field position (including red zone/opponent territory).
- Report `n` plays per split; suppress or flag splits below a minimum sample threshold.
- Detect duplicate plays and missing values; explain gaps, never impute.
- Add tendency-change comparison across week windows (req 15).
- **Exit:** golden-file tests assert byte-identical tables for identical inputs; edge cases for empty/short samples.
- **Implementation notes:**
  - `stats/engine.py` is pure Python over the frozen `Play` contracts (no pandas): identical inputs produce byte-identical `StatsBundle.to_payload()` output, pinned by `tests/fixtures/stats_golden_2026_SEA.json` (regenerate deliberately with `UPDATE_GOLDEN=1 pytest tests/unit/test_stats_engine.py`).
  - Populations are the selected opponent's offensive plays from the selected games: scrimmage plays drive run rate, pass rate, and yards per play; third-down attempts are scrimmage plays on third down with a recorded outcome, so flagged penalties, punts, and kneels never count as attempts.
  - Splits: down 1–4, distance buckets 0–3 / 4–6 / 7+, field position red zone 0–20 / opponent territory 21–50 / own territory 51–100, and week. Fixed buckets always emit a row (an empty bucket says so explicitly); "Unknown" rows appear only when populated.
  - Every row reports `n`; rows below `min_sample` (default 10) keep their value but are flagged `suppressed`, and a table is suppressed when its total is below `min_sample`.
  - Missing values are never imputed: missing split keys land in explicit "Unknown" rows with a table note, missing `yards_gained` or third-down outcomes exclude the play from that metric with a note plus a `DataQuality` count, and duplicate `(game_id, play_id)` input raises.
  - `compare_week_windows` requires disjoint windows drawn from the selected weeks and classifies each metric's direction with fixed thresholds (0.05 for rates, 0.5 for yards); below `min_sample` the comparison keeps its values but reports an `unknown` direction and is marked unreliable.
  - Tables are sorted by `stat_table_id` and draw only on the selected weeks and games, so STA can attach them to its `AgentOutput` unchanged.
- **Reqs:** 11, 15, 24.

### Stage 5 — Validation layer ✅
- Check every game, play, media item, and quote ID resolves to stored source data.
- Enforce opponent/week alignment; label earlier-2026 sources as background; reject prior seasons.
- Validate schemas, duplicates, missing values, and retrieval results before any model sees them.
- Flag weak evidence (small samples, missing sources) for human review.
- Return structured errors/warnings rather than raising vague exceptions.
- **Exit:** table-driven tests for each rejection and warning path.
- **Implementation notes:**
  - `validation/checks.py` exposes pure checks that each return a structured
    `ValidationResult` instead of raising: `validate_selection`, `check_plays`,
    `check_media_alignment`, `check_retrieval`, `check_evidence`,
    `check_stat_tables`, and `check_records` (raw payload schema validation).
    Issue codes are the `CheckCode` enum; nothing raises a vague exception.
  - `validate_selection` requires every selected week to be a played
    regular-season week containing a game for the opponent; duplicates,
    out-of-range weeks, unplayed weeks, unknown opponents, and selections with
    no opponent games are errors, while a selected week where the opponent is
    idle is only informational.
  - Week alignment treats a source inside the selection as aligned, an
    earlier-2026 week as `media_background` (warning, labeled background),
    a week after the selection as an error, a missing week as a warning, and
    any other season as `media_prior_season` error. Media tied to a
    `game_id` must resolve in the season store with a matching week.
  - `check_plays` reports duplicate `(game_id, play_id)` keys as errors and
    mirrors the stats engine's populations (opponent scrimmage plays from the
    selected games) for missing-value notes, so gaps are explained, never
    imputed. `check_stat_tables` reuses the engine's `DataQuality` counts and
    flags suppressed tables/rows for human review; row flags are suppressed
    when the whole table is already suppressed.
  - `check_retrieval` validates retrieval results against the media store
    before a model sees them: units must resolve to stored media for the
    requested opponent/season, carry exact body spans (quote units reproduce
    the quote verbatim), respect the background/later-week rule via the
    stored week (not the unit's own metadata), and duplicate units,
    non-finite scores, or unknown unit kinds are errors. Empty retrieval and
    missing sources are `requires_human_review` warnings so the report says
    what is missing (req 22).
  - `check_evidence` resolves every `EvidenceRef` kind: play evidence must
    resolve to stored games and plays, media/quote snippets must be exact
    spans of stored text, quotes must still reproduce their span verbatim,
    and stat evidence must resolve to a provided deterministic table.
  - `check_records` turns `ValidationError`s into per-record
    `invalid_record` issues with field paths, for validating raw data
    responses before they become evidence.
- **Reqs:** 19, 21, 22, 24, 26, 27.

### Stage 6 — LLM client, prompts, and offline prompt optimization ✅
- Implement an OpenAI-compatible client for VT ARC using environment config; bounded retries, per-call timeout, latency/token logging, no secrets in logs.
- Write narrow constitutional prompts for STA, BRA, CQA, SYN, verifier, and follow-up.
- Use delimiters, provenance labels, and instruction hierarchy so retrieved text is data, not instructions.
- Parse structured outputs with Pydantic; allow one bounded repair attempt on invalid output.
- Add `scripts/optimize_prompts.py` for offline prompt optimization; store the selected prompt version and evaluation results. Do not run it at runtime.
- **Exit:** mocked contract tests; invalid JSON triggers bounded repair; report records prompt version.
- **Implementation notes:**
  - `llm/client.py` owns every chat call: bounded retries (default 2) with capped exponential backoff over timeouts, transport errors, empty non-truncated replies, and HTTP 408/429/5xx; a per-call timeout; and logging that records only model, status, attempts, latency, and token counts — never prompts, completions, headers, or keys. Provider-side reasoning is disabled by default (`chat_template_kwargs.enable_thinking=false`, opt in with `VT_ARC_ENABLE_THINKING=true`) because the pinned reasoning model can spend the whole completion cap before emitting JSON; an empty `finish_reason=length` reply is reported immediately, and a truncated structured reply raises `LLMResponseError` with the finish reason/token counts instead of spending the repair. The endpoint buffers non-streaming replies and rejects `max_tokens` above 8,000 with HTTP 413, so `chat` automatically sets `stream=true` (with `stream_options.include_usage`) above that cap and accumulates SSE deltas, finish reason, and usage; a gateway that ignores `stream=true` and returns one buffered JSON body is accepted through the same parser. `complete_structured` parses the Pydantic wire models from `llm/schemas.py` and allows exactly one repair attempt that quotes the validation errors back to the model; a second failure raises `StructuredOutputError`. JSON extraction tolerates markdown fences and surrounding prose without spending a repair.
  - `llm/prompts/` holds six versioned constitutional prompts (statistics, beat_reporter, coach_quote, synthesis, verifier, follow_up). Each shares one operating-rules block — instruction hierarchy, grounding, no fabrication, exactly-one-JSON-object output, scope limits — plus agent-specific rules. `PROMPT_VERSION` (`2026.11.0`) is the single value stamped on agent outputs and reports (and feeds `report_id`), so any prompt edit requires bumping it deliberately. `Prompt.render` fails fast unless the caller supplies exactly the declared context keys.
  - `llm/prompts/blocks.py` renders every data block deterministically: retrieval units are wrapped in `<retrieved_data source="untrusted">` with provenance attributes (publisher, author, title, url, week, speaker, …), stat tables render as `<deterministic_statistics>` ground truth, and `neutralize_tags` rewrites literal block tags inside untrusted text to Unicode look-alikes so injected text cannot close or open a data block.
  - `llm/optimize.py` plus `scripts/optimize_prompts.py` implement offline prompt optimization: propose rewrites through the chat model, run deterministic constraint checks (data-block tags, JSON rule, instruction hierarchy, length bounds, stray placeholders), then a model ACCEPT/REJECT evaluation, and store results atomically under `DATA_DIR/prompt_optimization/results.json`. The runtime never imports it; an accepted rewrite is promoted by editing `llm/prompts/library.py` and bumping `PROMPT_VERSION`.
  - All Stage 6 tests run on `httpx.MockTransport`; `pytest` never calls VT ARC.
- **Reqs:** 10, 18, 23, 29.

### Stage 7 — Specialist agents ✅
- STA: call deterministic stats tools only; LLM interprets/report results.
- BRA: retrieve filtered articles and written film breakdowns; extract tendencies, formations, play calling, and recent changes with citations.
- CQA: retrieve filtered post-game coach quotes; preserve verbatim wording with coach, date, and source.
- All three emit the same `AgentOutput` schema and explicit limitations/empty results.
- **Exit:** fixture-driven tests; every claim has an evidence ref; no evidence produces an explicit limitation, not fabrication.
- **Implementation notes:**
  - `agents/common.py` is the shared spine: `AgentResult` bundles the `AgentOutput` with token usage, call count, warnings, and availability; `prompt_context` renders the exact prompt context; `convert_agent_response` turns wire models into frozen findings while dropping unknown evidence IDs, out-of-selection weeks, non-finite numbers, and empty statements with deterministic `ValidationIssue` warnings. `build_output` stamps agent identity, prompt version, games, weeks, tables, evidence, and limitations; `unavailable_output` produces the explicit empty result used by the failure matrix.
  - Evidence catalogs are built only in Python: `build_stat_evidence` (one ref per stat table), `build_media_evidence` (media + body span), and `build_quote_evidence` (media + quote span). The model may cite only IDs from that catalog; the contract never accepts a model-invented ID. Findings whose only citations are suppressed stat tables or that the model marks low-confidence set `weak_evidence=True`, and `HIGH` confidence is downgraded to `MEDIUM` so weak claims cannot be presented as certain.
  - `run_statistics_agent` passes the frozen `StatsBundle` tables through unchanged (a selection mismatch raises), so STA can rephrase but never recompute. `beat_reporter_query` and `coach_quote_query` request media then filter the retrieved units through `check_retrieval`; empty or missing retrieval yields `SectionStatus.COMPLETED` with an explicit limitation rather than a fabrication. Retrieval is bounded at `max_week=max(weeks)`, so earlier-week sources remain background-only with deterministic limitations/warnings.
  - CQA forces coach explanation/adjustment observations to `treat_as_context=True`, keeping quotes as context rather than proof. Every agent path records model, prompt version, token usage, and call counts without logging prompts or completions.
- **Reqs:** 10, 11, 12, 13, 25.

### Stage 8 — Synthesis and claim verification ✅
- Merge validated findings; state agreement, conflict, and gaps; compare week windows; treat coach quotes as context, not proof.
- Produce quantitative tables and qualitative notes with citations.
- Run a separate LLM verification call against source snippets; correct/remove unsupported claims.
- Python re-checks calculations, format, and source IDs after synthesis.
- Flag weak evidence for human review.
- **Exit:** contradictory fixture yields a conflict note; unsupported claim is removed; all citations resolve; stats unchanged by LLM.
- **Implementation notes:**
  - `agents/synthesis.py` accepts only completed specialist outputs and validates that they share opponent, season, weeks, and games before the call. `run_synthesis_agent` gives the model the union evidence catalog plus the deterministic tables; `_agreements`, `_conflicts`, and `_gaps` accept only claims whose cited IDs resolve in that catalog and whose weeks stay inside the selection, dropping the rest with `SYNTHESIS_UNGROUNDED_CLAIM`. Python adds a `Gap(AGENT_UNAVAILABLE)` for each unavailable specialist and `WeakEvidence(LOW_CONFIDENCE)` for each weak finding, so gaps and review flags survive even if the model omits them. Synthesis restates every validated finding plus agreements, conflicts, gaps, and weak-evidence flags, so `run_synthesis_agent` requests `SYNTHESIS_MAX_TOKENS` (16,384); above the endpoint's buffered cap the client streams the reply, and an exhausted budget raises the explicit truncation error instead of returning a partial report.
  - `agents/verify.py` builds one claim per report finding with the exact resolved snippet text and runs a separate verifier call (not a fifth agent). `apply_verification` interprets verdicts deterministically: `supported` keeps the claim only when snippet IDs are attached; `partially_supported` replaces the statement with the verifier's correction; `unsupported`, `contradicted`, a missing verdict, or an invalid citation removes the claim and records `WeakEvidence` plus a warning. Removed/corrected claims are reported through `VERIFICATION_REMOVED_CLAIM`/`VERIFICATION_CORRECTED_CLAIM`.
  - Synthesis never edits the deterministic tables; the orchestrator re-runs `check_stat_tables` and `check_evidence` on the final report and requires the report tables to equal the agent tables, so the LLM cannot change a number or a citation ID.
- **Reqs:** 14, 15, 16, 19, 27, 28.

### Stage 9 — Orchestrator and failure handling ✅
- Implement `run_report(opponent, weeks)`: validate → parallel STA/BRA/CQA → fan-in validation → SYN → verify → report.
- Enforce per-agent timeouts, overall deadline, step cap, token budget, and stall detection.
- Apply the failure matrix exactly: STA/overall → hard failure; BRA/CQA → unavailable section + warning + partial report.
- Bound retries; never loop indefinitely on missing evidence.
- **Exit:** failure-injection tests with mocked agents; no report on hard failure; partial report on degraded failure.
- **Implementation notes:**
  - `orchestrator.run_report` is the single UI/CLI entry point. It validates the selection (`validate_selection`, `check_plays`), computes the deterministic `StatsBundle` once, runs STA/BRA/CQA concurrently on a `ThreadPoolExecutor`, re-checks citations (`check_evidence`, `check_retrieval`), then runs synthesis, the separate verifier, and the final evidence/stat re-checks before stamping a `Report`. `ReportRun` carries the status, optional report, warnings, agent results, validation, token usage, call count, latency, and the synthesis/verification artifacts.
  - Timeouts are bounded `Future.result(timeout=...)` waits (never a model-controlled stop); a timed-out call is abandoned with `shutdown(wait=False, cancel_futures=True)` and the client's own timeout bounds the request. The overall deadline, `max_llm_calls` step cap, and `token_budget` are checked between stages and reuse `DEADLINE_EXCEEDED` for bounded-resource overruns. Stall detection is the same bounded wait: a worker that produces no result within its budget is treated as stalled/timed out, abandoned, and routed through the failure matrix (hard failure for STA/synthesis/verification; unavailable section for BRA/CQA), so no stage can wait forever.
  - Failure matrix: validation, overall deadline/budget, STA, synthesis, verification, or final re-check failure is a hard failure with **no** `Report`; BRA/CQA failure or timeout yields an explicit unavailable section, an `AGENT_UNAVAILABLE`/`AGENT_TIMEOUT` warning, and a `DEGRADED` report carrying the statistics and completed sections. `RunStatus.FAILED` always records `failure_reason`, `finished_at`, and a human-readable message.
  - `on_progress(RunProgress)` emits deterministic `step`/`status`/`agent` events from the main thread only (one per specialist and pipeline stage), so the UI can render progress without touching worker state. `data_revision` is derived from the season and media store revisions, and `DataUsage` counts games, opponent offensive plays, cited media/quote IDs, and sorted cited `source_id`s in Python.
  - `report_cli.py` / `scripts/report.py` wrap `ui.runner.build_report` for terminal use, expose the pipeline budgets, and offer `--no-cache`, `--json`, and `--output`. `--stats-only` stops after `analyze_opponent` and never reads credentials or calls the index/model.
- **Reqs:** 9, 10, 22, 30.

### Stage 10 — Streamlit UI and session state ✅
- Alphabetical team dropdown; multiselect of played regular-season weeks only.
- Refresh button showing inserted/skipped counts.
- Selection summary: opponent, weeks, games, dates, scores, data used.
- Build-report button with per-agent progress; prevent concurrent duplicate runs.
- Render quantitative tables, qualitative notes, citations, warnings, unavailable sections, and weak-evidence banners.
- Store opponent, weeks, report, and evidence in `st.session_state`.
- Follow-up chat reuses SYN and stored evidence without a full re-run; summarize older turns to control context.
- **Exit:** Streamlit AppTest or script-level UI tests; session persistence verified; hard-failure message shown.
- **Implementation notes:**
  - `app.py` is the workflow shell: it loads the season store, lists `SeasonData.teams` (already alphabetical by display name), offers only `SeasonData.played_weeks`, and renders the selection summary before any model call. The sidebar refresh button calls `NflverseClient.refresh` and prints the store's inserted/updated/skipped counts; the Chroma index remains a script-managed artifact.
  - `ui/runner.py` keeps the UI thin and testable: `load_season`/`load_media` need no credentials, `create_index` uses the pinned `ArcEmbedder`/`media_<season>` collection, and `build_report` calls the Stage 9 orchestrator. A finished report is reused from `ReportCache` only when `(opponent, weeks, prompt_version, data_revision)` match; no failed or partial build is cached. Tests inject data, index, client, `report_runner`, or `follow_up_runner` so AppTest never touches the network.
  - `ui/render.py` renders the stored contracts without recomputing anything: selection metrics and game scores, deterministic tables (suppressed rows/notes preserved), findings with evidence labels/IDs/URLs, agreements/conflicts/gaps, warnings, unavailable sections, the complete evidence list, and weak-evidence banners. Hard failures show the recorded `failure_reason`/message and never a fake report.
  - Session state holds `opponent`, `weeks`, `report`, `evidence`, `last_run`, and `follow_up_turns`. `run_in_progress` disables the build button, and the report run's latency/token/call counts are displayed as telemetry.
  - Follow-up questions call the `follow_up` prompt with `render_report_context(report)` and a deterministic `summarize_conversation` (recent turns in full, older questions collapsed, hard character cap). Python validates every returned evidence ID against `report.evidence`; unknown IDs are removed, `supported` is forced to `False`, and a limitation is appended. No specialist agent or retrieval step runs again.
- **Reqs:** 1, 4, 5, 9, 16, 17, 22, 30.

### Stage 11 — Evaluation, hardening, and docs ✅
- Build unit, integration (fixtures), mocked-LLM contract, and end-to-end tests.
- Add `scripts/eval_consistency.py`: same inputs 5 times → top-5 tendency overlap ≥80% in 4/5 runs, no direction contradictions; stats tables identical; quotes 100% traceable.
- Add prompt/chunking evaluation on a small opponent sample.
- Measure latency and token cost; cache reports by `(opponent, weeks, prompt_version, data_revision)`.
- Add prompt-injection fixtures and source/instruction separation tests.
- Write README/runbook: setup, refresh, index rebuild, evaluation commands.
- Update `AGENTS.md` once app code exists (current file correctly describes the docs-only state).
- **Exit:** all acceptance checks pass; report completes within the target time; failure modes documented.
- **Implementation notes:**
  - `evaluation.py` is pure and deterministic. `tendency_key` normalizes category/statement; `top_tendencies` takes the first five distinct keys; each run is compared with the consensus top list built from all runs (count, then key), and `passing_runs >= ceil(0.8 * runs)` encodes "4/5 at ≥80%". Direction flips for the same key, non-identical canonical stat tables, or any untraceable quote fail the result. Quote traceability requires the stored `Quote` to exist, match the evidence snippet verbatim, and match the report opponent/season; stat evidence must resolve to a report table. `as_dict()` is the stable JSON output.
  - `scripts/eval_consistency.py` is the live harness: it loads settings, season/media stores, the pinned index, and the chat client, runs `run_report` N times, prints the deterministic JSON result, and exits non-zero when the acceptance check fails. It is never imported by runtime code.
  - `retrieval_eval.py` + `scripts/eval_retrieval.py` cover the prompt/chunking slice: `evaluate_corpus_chunking` aggregates `evaluate_chunking` over every stored body (coverage, hard cuts, duplicate chunks) and `evaluate_retrieval` runs the exact BRA/CQA queries through the filtered index, then reuses `check_retrieval` to require stored, opponent-matched, week-aligned sources. The script defaults to the offline `HashingEmbedder` and `--embedder arc` evaluates the pinned persistent index; it prints the same stable JSON shape and exits non-zero on failure.
  - `cache.py` implements a content-addressed `ReportCache`: the documented tuple is normalized (sorted unique weeks, team value, prompt version, combined season/media revision) into a SHA-256 filename. Entries are canonical JSON envelopes with `schema_version`, the exact key, and the frozen report; reads validate the schema, key, and `Report` contract, and writes use the shared atomic writer. `ui.runner.build_report` computes the revision before the model call, returns a cache hit as `ReportRun(cached=True)`, and stores only successful reports.
  - Prompt-injection and source/instruction separation are covered by `tests/unit/test_llm_prompts.py`: untrusted blocks keep provenance and literal block tags are neutralized, including inside retrieved sources, agent findings, verifier claims, and report context. The orchestrator/agent suites add mocked-LLM end-to-end failure and determinism coverage, and AppTest covers the UI flow. The README runbook documents setup, refresh, index rebuild, app use, and the consistency/caching behavior. `ReportRun` exposes call count, token usage, and latency for measurement.
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
| Chunking + filtered retrieval | Offline/live evaluation | Full coverage, no hard cuts, aligned sources |
| Full report / stats CLI | Unit + offline run | Report or explicit failure; stats without credentials |
| UI flow | Streamlit AppTest | Selection → report → follow-up persists |

## 6. Milestones

| Milestone | Stages | Outcome |
|---|---|---|
| M1 Data foundation | 0–3 | Real PBP + media corpus + persistent index |
| M2 Deterministic analysis | 4–5 | Trustworthy stats and validated inputs |
| M3 Agent pipeline | 6–9 | Parallel agents, synthesis, verification, failure handling |
| M4 User interface | 10 | End-to-end Streamlit workflow with chat |
| M5 Reliability | 11 | Determinism, traceability, latency, hardening, docs |

Recommended vertical slices: stats-only CLI after Stage 5 and full-report CLI after Stage 9 (both via `scripts/report.py`; `--stats-only` needs no credentials), then the UI. All three are implemented.

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

## 9. Decisions

1. **PLAN.md location:** repo root (confirmed), so the graded deliverable folder stays clean.
2. **Media strategy:** curated/cached corpus behind a connector interface first (confirmed); live fetching only for sources whose terms allow it.
3. **Media metadata store:** per-season atomic JSON (`data/media/<season>.json`) rather than SQLite (confirmed for Stage 3); the media store is the source of truth and Chroma remains derived. Revisit SQLite only if metadata queries outgrow the in-memory store.
4. **Embeddings:** VT ARC embeddings are available and pinned through `EMBEDDING_MODEL` (confirmed, `Qwen3-Embedding-4B` verified); an offline `HashingEmbedder` exists for tests and chunking evaluation, and the resolved model ID is recorded in index metadata.
5. **Model alias:** the app reads `VT_ARC_MODEL` from env (confirmed); `EMBEDDING_MODEL` is separate so the chat and embedding models can change independently.

