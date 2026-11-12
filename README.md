# Opponent Tendencies Analyst

An NFL opponent-scouting assistant: deterministic Python computes the statistics,
and an LLM (VT ARC) interprets and cites them in a scouting report. The full
design lives in [`PLAN.md`](PLAN.md); the graded write-up lives in
[`docs/`](docs/).

> **Status: Stages 0–11 complete.** Packaging/config, frozen contracts, nflverse
> ingestion into the JSON play store, curated media ingestion plus a persistent
> Chroma index, the deterministic statistics engine, the validation layer, the
> LLM client + versioned prompts, the specialist agents, synthesis/verification,
> the orchestrator, the Streamlit report/follow-up UI, the full-report and
> stats-only CLI, content-addressed report caching, and the consistency plus
> chunking/retrieval evaluation harnesses are implemented and tested. See
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

### 3. Configure credentials (needed for VT ARC calls)

Copy the template and fill in real values:

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
| --- | --- | --- |
| `VT_ARC_BASE_URL` | yes* | VT ARC OpenAI-compatible endpoint |
| `VT_ARC_API_KEY` | yes* | API key (never commit this file) |
| `VT_ARC_MODEL` | yes* | Model ID the LLM client calls |
| `VT_ARC_ENABLE_THINKING` | no | Opt into provider-side reasoning (default `false`; the pinned reasoning model otherwise spends the completion cap before emitting JSON) |
| `EMBEDDING_MODEL` | yes* | Pinned embedding model for the media index |
| `DATA_DIR` | no | Where stores live (default `data/`) |
| `CHROMA_DIR` | no | Where the Chroma index lives (default `data/chroma`) |

\* Required only when code calls `Settings.from_env()` / `get_settings()`. The
data-only UI paths (season selection and summary), `scripts/refresh_data.py`,
`scripts/report.py --stats-only`, `scripts/eval_retrieval.py` (hashing default),
the stats engine, and the validation checks do **not** read credentials.
Building a report in the app or via `scripts/report.py`,
`scripts/build_index.py --embedder arc` (the default), and
`scripts/optimize_prompts.py` do, because they send text to VT ARC. Anything
that calls `get_settings()` fails fast with a `ConfigurationError` naming every
missing variable — there are no silent defaults for secrets. For an offline
index build, pass `--embedder hashing`; it records a different model ID, so a
later VT ARC query refuses to reuse that collection.

### 4. Run the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (normally <http://localhost:8501>). The app
walks through the full workflow:

1. Pick an opponent from the alphabetical team list and one or more played
   regular-season weeks. The selection summary shows the games, dates, and
   scores before anything is generated.
2. Optionally refresh the play store from nflverse from the sidebar; the result
   reports inserted/updated/skipped counts.
3. Click **Build report**. Specialist agents run in parallel with per-agent
   progress; the button is disabled while a run is in flight.
4. Read the deterministic tables, cited tendencies/reporting/coach context,
   agreements, conflicts, gaps, warnings, and unavailable sections. Partial
   reports and weak evidence stay visibly labeled.
5. Ask follow-up questions in the chat at the bottom. Answers reuse the stored
   report and evidence only — no agent is re-run.

A finished report is reused automatically when the opponent, weeks, prompt
version, and stored data revision all match. Delete
`data/reports/<digest>.json` (or change the data/prompt) to force a rebuild.

To use a different port or run without opening a browser:

```bash
streamlit run app.py --server.port 8502 --server.headless true
```

#### Command-line alternative

The same pipeline is available without Streamlit:

```bash
python scripts/report.py --opponent BUF --weeks 5 6 7 8 9
python scripts/report.py --opponent BUF --weeks 5 6 7 8 9 --output report.json
python scripts/report.py --opponent BUF --weeks 5 6 7 8 9 --no-cache
```

It prints the status, report ID, data counts, finding counts, model/prompt
versions, token usage, and latency; `--json` prints the full frozen report plus
telemetry instead, and `--output` writes it atomically. Exit code `0` means a
report was produced (including a degraded one), `1` means a hard failure, and `2`
is an argument/configuration error.

For deterministic statistics only — no credentials, media index, or model call:

```bash
python scripts/report.py --opponent BUF --weeks 5 6 7 8 9 --stats-only
python scripts/report.py --opponent BUF --weeks 5 6 7 8 9 --stats-only --json --output stats.json
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

### 7. Evaluate consistency, chunking, and retrieval (occasional)

```bash
python scripts/eval_consistency.py --opponent BUF --weeks 5 6 7 8 9
python scripts/eval_retrieval.py --opponent BUF --weeks 5 6 7 8 9
python scripts/eval_retrieval.py --opponent BUF --weeks 5 6 7 8 9 --embedder arc
```

The consistency harness runs the same report five times against VT ARC, then
applies deterministic acceptance checks: top-five tendency overlap of at least
80% in at least four of five runs, no tendency direction flips, byte-identical
deterministic stat tables, and 100% traceable quotes. It prints a JSON summary
(add `--output path.json` to save it) and exits non-zero when a check fails. It
needs credentials and the built index.

`eval_retrieval.py` is the chunking/retrieval check. It evaluates every stored
media body (coverage, hard cuts, duplicate chunks) and runs the exact BRA/CQA
queries through the filtered index, requiring stored, opponent-matched,
week-aligned sources. The default `--embedder hashing` is fully offline; use
`--embedder arc` to evaluate the pinned persistent index. It prints stable JSON
and exits non-zero when either check fails. Normal tests never run either
script against VT ARC.

---

## Using the components from Python

The deterministic layers are pure Python over the stored contracts, so they can
be exercised without any LLM credentials once a season store exists (step 5):

```python
from opponent_tendencies.data.play_store import SeasonStore
from opponent_tendencies.models import Team
from opponent_tendencies.stats.engine import analyze_opponent
from opponent_tendencies.validation.checks import validate_selection

state = SeasonStore(2026).load()
opponent = Team.SEA
weeks = [1, 2, 3]
games = [g for g in state.played_games if g.involves(opponent) and g.week in weeks]

selection = validate_selection(state, opponent=opponent, weeks=weeks)
if not selection.ok:
    raise SystemExit("; ".join(issue.message for issue in selection.errors))

bundle = analyze_opponent(state.plays, games, opponent=opponent)
print(bundle.to_payload())  # canonical, JSON-safe payload
```

`analyze_opponent` computes run/pass rate, yards per play, and third-down
conversion split by down, distance, field position, and week, with every row's
`n` and a `suppressed` flag below `min_sample` (default 10). It analyzes only the
opponent's offense from the selected games, treats third-down attempts as
scrimmage plays with a recorded outcome, never imputes (missing keys become
explicit "Unknown" rows, missing outcomes are excluded with notes), and rejects
duplicate `(game_id, play_id)` input. `compare_week_windows(...)` adds the
early-vs-late comparison.

`validation/checks.py` returns `ValidationResult`s (errors, warnings,
`requires_human_review`) instead of raising; see its docstrings for
`check_plays`, `check_media_alignment`, `check_retrieval`, `check_evidence`,
`check_stat_tables`, and `check_records`.

LLM calls go through the Stage 6 client and prompts:

```python
from opponent_tendencies.config import get_settings
from opponent_tendencies.llm.client import ArcLLMClient
from opponent_tendencies.llm.prompts import (
    get_prompt,
    render_evidence_catalog,
    render_games_summary,
    render_stat_tables,
)

client = ArcLLMClient.from_settings(get_settings())
rendered = get_prompt("statistics").render(
    opponent="SEA",
    season="2026",
    weeks="1-3",
    games_summary=render_games_summary(games),
    stat_tables=render_stat_tables(bundle.tables),
    available_evidence=render_evidence_catalog([]),
)
result = client.complete_structured(rendered.messages, "agent_response")
print(result.parsed.tendencies)
```

The full pipeline is one call. `run_report` validates the selection, computes
the tables, runs STA/BRA/CQA concurrently, synthesizes, verifies, and returns a
`ReportRun` (never raises for a deterministic validation failure):

```python
from opponent_tendencies.data.media_store import MediaStore
from opponent_tendencies.orchestrator import run_report
from opponent_tendencies.ui.runner import create_index

media = MediaStore(2026).load()
index = create_index(2026)
run = run_report(
    opponent=opponent,
    weeks=weeks,
    season_data=state,
    media_data=media,
    index=index,
    client=client,
)
if run.report is None:
    print(run.status.failure_reason.value, run.status.message)
else:
    print(run.report.overview, run.usage.total_tokens, run.latency_seconds)
```

`Build report` in the app uses `ui.runner.build_report`, which wraps this call,
adds the report cache, and loads the stores/pinned embedder for you.

Six prompts live in `llm/prompts/library.py` (`statistics`, `beat_reporter`,
`coach_quote`, `synthesis`, `verifier`, `follow_up`); all share one
constitutional operating-rules block and one `PROMPT_VERSION`, which is stamped
on agent outputs and reports. `Prompt.render` fails unless the caller supplies
exactly the declared context keys, and only `llm/prompts/blocks.py` renders
delimited data blocks.

### Prompt optimization (design-time)

```bash
python scripts/optimize_prompts.py --prompts statistics --candidates 2
```

This proposes rewrites, runs deterministic constraint checks, then a model
ACCEPT/REJECT evaluation, and writes results to
`data/prompt_optimization/results.json`. It needs VT ARC credentials and is
**never imported at runtime**; promote an accepted rewrite by editing
`llm/prompts/library.py` and bumping `PROMPT_VERSION`.

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
| Deterministic stats engine: run/pass rate, yards per play, third-down conversion, splits, suppression, week-window comparison (`stats/engine.py`) | ✅ Stage 4 |
| Validation layer: selection, plays, media alignment, retrieval grounding, evidence resolution, stat tables, raw records (`validation/checks.py`) | ✅ Stage 5 |
| OpenAI-compatible LLM client with bounded retries, telemetry, streamed long completions above the proxy's buffered cap, one-shot structured-output repair (`llm/client.py`) | ✅ Stage 6 |
| Six versioned constitutional prompts + deterministic data-block renderers (`llm/prompts/`) | ✅ Stage 6 |
| Offline prompt optimization (`llm/optimize.py`, `scripts/optimize_prompts.py`) | ✅ Stage 6 |
| Specialist agents (STA, BRA, CQA) | ✅ Stage 7 |
| Synthesis agent + claim verification | ✅ Stage 8 |
| Orchestrator and failure handling (`orchestrator.run_report`) | ✅ Stage 9 |
| Full-report + stats-only CLI (`report_cli.py`, `scripts/report.py`) | ✅ Stage 9 |
| Streamlit UI: selection, refresh, progress, report/failure rendering, session state, follow-up chat (`app.py`, `ui/`) | ✅ Stage 10 |
| Content-addressed report cache (`cache.py`) and run telemetry | ✅ Stage 11 |
| Consistency evaluation (`evaluation.py`, `scripts/eval_consistency.py`) | ✅ Stage 11 |
| Chunking/retrieval evaluation (`retrieval_eval.py`, `scripts/eval_retrieval.py`) | ✅ Stage 11 |
| Evaluation, hardening, and docs | ✅ Stage 11 |

Every stage in `PLAN.md` §4 is now implemented; the tables in §8 remain the
requirement traceability reference.

---

## Development

Run the local quality gates (there is no CI or pre-commit hook, so run them
yourself):

```bash
pytest -q                                  # full suite (fixture-driven, no network)
ruff check . && ruff format --check .      # lint + formatting
mypy src tests scripts                     # strict typing
```

Focused runs while developing:

```bash
pytest -q tests/unit/test_stats_engine.py
pytest -q tests/unit/test_validation_checks.py -k retrieval
pytest -q tests/unit/test_llm_client.py
pytest -q tests/unit/test_orchestrator.py tests/unit/test_report_cli.py
pytest -q tests/unit/test_ui_app.py tests/unit/test_evaluation.py tests/unit/test_cache.py
pytest -q tests/unit/test_retrieval_eval.py
pytest -q -m integration                   # stores + Chroma under tmp_path
```

Apply formatting:

```bash
ruff format .
```

mypy is strict (`disallow_untyped_defs`), and pytest runs with
`--strict-markers` / `--strict-config`, so register any new marker in
`pyproject.toml` before using it.

Tests never hit the network: nflverse fixtures are trimmed real 2026 pulls in
`tests/fixtures/` and are injected through `NflverseClient`'s loader arguments.
Media tests use the tracked `corpus/media_2026.json` fixture with the offline
`HashingEmbedder`; `tests/unit/test_embeddings.py` mocks the VT ARC HTTP
contract, and every LLM test runs on `httpx.MockTransport` — nothing calls
VT ARC. AppTest injects fixtures/`report_runner`/`follow_up_runner` rather than
clicking refresh or building against the live model. The stats engine is pinned
by a golden payload; regenerate it only deliberately:

```bash
UPDATE_GOLDEN=1 pytest tests/unit/test_stats_engine.py
```

### Project layout

```text
app.py                              # Streamlit workflow: select -> build -> follow-up
corpus/media_2026.json              # curated, tracked Stage 3 input
scripts/refresh_data.py             # nflverse -> data/seasons/<season>.json
scripts/build_index.py              # corpus -> data/media/<season>.json + Chroma
scripts/report.py                   # full report or --stats-only from the CLI
scripts/optimize_prompts.py         # offline prompt optimization (design-time)
scripts/eval_consistency.py         # live 5-run consistency evaluation
scripts/eval_retrieval.py           # chunking + retrieval evaluation (offline/live)
src/opponent_tendencies/
  config.py                         # env parsing, Settings, ConfigurationError
  models.py                         # frozen contracts: Game, Play, MediaItem, Quote, ...
  ids.py                            # deterministic ID/hash helpers
  cache.py                          # content-addressed ReportCache
  orchestrator.py                   # validate -> agents -> synthesis -> verify -> Report
  report_cli.py                     # full-report + stats-only command line
  evaluation.py                     # pure 5-run consistency checks
  retrieval_eval.py                 # chunking + filtered-retrieval evaluation
  data/nflverse.py                  # fetch + normalize play-by-play and schedules
  data/play_store.py                # atomic JSON store, upsert by (game_id, play_id)
  data/jsonio.py                    # canonical JSON + atomic writes
  data/media_ingest.py              # allowlist, corpus parser, quote spans
  data/media_store.py               # atomic media/quote store + index manifest
  data/chunking.py                  # deterministic chunk spans + evaluation
  data/index.py                     # Chroma build/reuse + filtered retrieval
  stats/engine.py                   # deterministic tendency tables + week comparison
  validation/checks.py              # structured source/alignment/grounding checks
  llm/schemas.py                    # JSON schemas for structured LLM output
  llm/client.py                     # VT ARC chat client, retries, structured repair
  llm/embeddings.py                 # ArcEmbedder + offline HashingEmbedder
  llm/prompts/library.py            # six versioned constitutional prompts
  llm/prompts/blocks.py             # deterministic data-block renderers
  llm/optimize.py                   # offline prompt optimization (design-time only)
  agents/common.py                  # evidence catalogs, conversion, telemetry
  agents/statistics.py              # STA
  agents/beat_reporter.py           # BRA
  agents/coach_quote.py             # CQA
  agents/synthesis.py               # SYN + deterministic agreement/conflict/gap checks
  agents/verify.py                  # separate verifier + deterministic verdicts
  ui/runner.py                      # data/index/client loading, cache, follow-ups
  ui/render.py                      # selection/report/failure/progress rendering
tests/
  unit/                             # contracts, data, stats, checks, agents, UI, cache/eval
  integration/                      # stores, idempotency, index reuse/retrieval
  fixtures/                         # trimmed nflverse samples + stats golden payload
  agent_fixtures.py                 # shared agent/orchestrator fixtures
  media_fixtures.py                 # shared curated-corpus helpers
  nflverse_fixtures.py              # shared play-by-play fixture helpers
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
- **Checks return results, not exceptions.** `validation/checks.py` reports
  problems as `ValidationIssue`s with a `CheckCode` and a severity inside a
  `ValidationResult`; new checks should extend `CheckCode` rather than raise
  ad-hoc errors.
- **The LLM layer owns calls, not facts.** `llm/client.py` retries are bounded
  (over timeouts, transport errors, empty non-truncated replies, and HTTP
  408/429/5xx), and logs contain only model, status, attempts, latency, and
  token counts — never prompts, completions, headers, or keys. Provider-side
  reasoning is disabled by default (`VT_ARC_ENABLE_THINKING=true` opts in), and
  replies that exhaust the completion cap are reported instead of retried.
  Requests above the endpoint's 8,000-token buffered cap are sent with
  `stream=true` (and a buffered JSON body is still accepted if a gateway ignores
  it); `complete_structured` permits exactly one repair attempt.
- **Prompts are versioned.** All prompt text lives in `llm/prompts/library.py`
  under one shared operating-rules block; `PROMPT_VERSION` is stamped on agent
  outputs and reports and feeds `report_id`, so bump it deliberately on any
  prompt edit. Render data blocks only through `llm/prompts/blocks.py`, which
  neutralizes block tags inside untrusted text. `llm/optimize.py` is
  design-time only — never import it from the runtime path.
- **Agents cite, they do not compute.** `agents/common.py` builds the only
  evidence IDs a model may cite; unknown IDs/weeks are dropped with warnings,
  suppressed-table and low-confidence findings are marked weak, and CQA coach
  statements stay context. `agents/verify.py` removes unsupported claims and
  replaces partially supported ones; synthesis may not rewrite `StatTable`s.
- **The orchestrator owns the failure decision.** `orchestrator.run_report`
  validates first, runs STA/BRA/CQA concurrently, re-checks evidence/tables, and
  only then stamps a `Report`. STA/validation/synthesis/verification/deadline/
  budget failures produce no report; BRA/CQA failures produce an explicit
  unavailable section and a degraded report.
- **The cache is keyed, not guessed.** `cache.py` reuses a report only for the
  exact `(opponent, weeks, prompt_version, data_revision)` tuple; entries are
  canonical JSON and revalidated against `Report` on read. Failed runs are never
  cached.
- **The UI and CLI are thin.** `app.py`, `ui/`, and `report_cli.py` load stores,
  call the orchestrator, validate follow-up citations, and render contracts.
  They never compute a statistic, retrieve a source, or trust a model-invented
  evidence ID. `--stats-only` calls the same deterministic engine and never
  reads credentials.
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
| `StructuredOutputError: structured output '...' failed validation after one repair attempt` | The model returned non-conforming JSON twice; the client never loops on repairs. Inspect the schema (`llm/schemas.py`) and prompt, then retry. |
| `LLMResponseError: chat completion message has no content (finish_reason=length...)` or `... truncated at the completion limit` | The model spent the completion cap on hidden reasoning or a truncated reply. Keep `VT_ARC_ENABLE_THINKING=false` (the default), raise the completion cap, or shorten the prompt. Transient empty replies are retried; `finish_reason=length` is reported immediately. |
| `LLMResponseError: ... HTTP 413 ... exceeds the proxy's buffered-mode cap of 8000` | A non-streaming request asked for more than 8,000 completion tokens. Leave the client's automatic streaming on (requests above `DEFAULT_BUFFERED_MAX_TOKENS` set `stream=true`), or lower `max_tokens` to fit the buffered cap. |
| `LLMTimeoutError` / `LLMConnectionError` | VT ARC timed out or was unreachable after the bounded retries. Check `VT_ARC_BASE_URL` and network access, then retry. |
| `ValueError: prompt '...' context mismatch: missing [...], unexpected [...]` | `Prompt.render` requires exactly the keys declared in `required_context`; use `template_placeholders()` to see what a template uses. |
| Golden stats test fails after an intentional change | Regenerate deliberately with `UPDATE_GOLDEN=1 pytest tests/unit/test_stats_engine.py` and review the diff before committing. |
| `Port 8501 is already in use` | Another Streamlit is running. Use `--server.port <free port>`. |
| A report looks stale after refreshing data or editing a prompt | The report cache key includes the combined data revision and `PROMPT_VERSION`; a real data/prompt change misses automatically. To force a rebuild anyway, delete the matching `data/reports/*.json` entry. |
| `ReportCacheError: cached report ...` | The cache entry is corrupt or from another schema. Delete that file; the next build rewrites it. |
| `scripts/report.py` exits `2` | Argument or configuration error (for example a missing VT ARC variable). The message on stderr names it; `--stats-only` needs no credentials. |
| `IndexModelMismatchError` from retrieval/evaluation | The stored collection was built with another embedder. Rebuild with `scripts/build_index.py --season 2026` using the same `EMBEDDING_MODEL`. |
| Refresh is slow | It downloads a full season of play-by-play; the second run is fast because nothing changed. |

## Roadmap

All stages in `PLAN.md` §4 are implemented (Stages 0–11), including the
Streamlit workflow, the full-report/stats-only CLI, report caching, and the
consistency plus chunking/retrieval evaluation harnesses. Milestone and
requirement traceability tables are in `PLAN.md` §6 and §8; the detailed
implementation notes live beside each stage in §4. Remaining work is operational
rather than architectural: run the live consistency/retrieval checks with real
credentials, refresh the 2026 data as the season progresses, and review the
report with a human before using it.
