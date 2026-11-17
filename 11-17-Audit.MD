# Code Audit — 2026-11-17

**Repo:** `final_project` @ `38a8a54` (branch `master`).
**Work tree:** one uncommitted change, `src/opponent_tendencies/llm/schemas.py` (`SynthesisResponse` list
bounds, see H1). Nothing else is modified.
**Scope:** runtime Python in `src/opponent_tendencies/` and `app.py`, plus the test and tooling gates.
**Gates at HEAD + the schemas.py change:**

| Gate | Command | Result |
|---|---|---|
| Tests | `.venv/bin/python -m pytest -q` | **2 failed, 497 passed** (M3) |
| Lint | `.venv/bin/ruff check .` | pass |
| Types | `.venv/bin/mypy src` | **8 errors**, all missing `reportlab` stubs (L1) |

**Severity counts:** 4 High, 13 Medium, 17 Low.

> **Status update (later 2026-11-17):** H1–H4 are **fixed in the work tree** (uncommitted) with
> regression tests, and `PROMPT_VERSION` is bumped to `2026.11.2`. A live `BUF` weeks 6–10 run completed
> in 5 LLM calls. Start with the Medium items. Not done from H1's list: the optional anti-repetition
> setting (step 5). The usage of a failed synthesis or verification call is still not added to the run
> budget (M1).

**Severity scale**
- **High:** the report fails outright, or a published report states something its own rules forbid.
- **Medium:** wrong behavior in a reachable path (budgets, caching, concurrency, grounding gaps), or a
  broken gate.
- **Low:** consistency, maintainability, performance or defense-in-depth.

**How to work this list**
- Each finding gives its location, the defect, a concrete failure scenario, a fix direction and an
  acceptance check. Line numbers are at `38a8a54`.
- Project rule (AGENTS.md / PLAN.md): deterministic Python owns validation, statistics, evidence IDs and
  failure decisions; the model only interprets. Fixes should keep enforcement in Python, not in prompt
  text.
- Keep `.venv/bin/python -m pytest -q` green and add a regression test per fix. Tests must stay offline;
  inject clients, data and `httpx.MockTransport`.

---

## High

### H1. Synthesis replies run away to the completion cap, so report builds fail with `LLMResponseError`

**Observed in production (2026-11-17):** `BUF`, weeks including 6–10:
`LLMResponseError: chat completion response was truncated at the completion limit
(finish_reason=length, completion_tokens=32768, reasoning_tokens=0, max_tokens=32768)` → run state
`synthesis_failed`.

**Where:** `llm/schemas.py:120` (`SynthesisResponse`), `agents/synthesis.py:69` (`SYNTHESIS_MAX_TOKENS =
32768`), `llm/client.py:403–470` (`complete_structured`), `llm/client.py:536–645` (streaming path).

**Defect:**
- `reasoning_tokens=0`, so hidden reasoning didn't use up the budget. The model emitted 32k tokens of
  visible JSON.
- The synthesis inputs for this selection are small. I measured 19 stat tables plus the stat evidence
  catalog at ≈4k tokens. A valid reply capped at 12 + 12 findings is a few thousand tokens.
- So the generation degenerated: an array or string that kept growing, or whitespace looping under
  guided JSON decoding.
- Commit `6037d56` doubled the cap from 16k to 32k and added "at most 12" to the prompt. That made the
  runaway take twice as long without fixing it. Prompt text isn't enforced.

**Partial fix already in the work tree (uncommitted):**
- `SynthesisResponse` lists now have `max_length` (12/12/10/8/10/10).
- The emitted JSON schema now contains `maxItems`.
- Not yet verified against the live VT ARC endpoint. vLLM's guided-decoding backend must honor
  `maxItems` for this to bound generation.

**Remaining work:**
1. **Bound strings.** `TrimmedStr` (`models.py:64`) has no `max_length`, so a single `statement` or
   `overview` can still run to the cap. Add a wire-only bounded string type to `llm/schemas.py` and use it
   for every free-text field in `Draft*` and `SynthesisResponse`. Suggested limits: `overview` ≤ 1500
   chars, statements and descriptions ≤ 600, topics ≤ 120. Don't change the stored contracts in
   `models.py`.
2. **Lower the cap.** With bounded output, put `SYNTHESIS_MAX_TOKENS` back to ≤ 16384 so a runaway fails
   fast. Update `tests/unit/test_synthesis_verify.py:105`, which asserts `== 32768`.
3. **Diagnose truncation.** On `finish_reason == "length"`, log the length and the last ~200 chars of
   `content` at WARNING (the content tail only, never the prompt). That tells repetition apart from
   whitespace loops.
4. **Retry once.** On truncation, retry once with a tighter instruction ("return at most 6 tendencies and
   6 observations") instead of failing immediately. Count that call toward `llm_calls`.
5. **Consider a light anti-repetition setting** for structured calls if the endpoint supports it
   (`frequency_penalty` or vLLM `repetition_penalty`). Put it behind a constant so tests can assert the
   payload.

**Acceptance:**
- A unit test with a fake client whose first reply truncates gets a successful result on the retry.
- A schema test asserts `maxItems` and `maxLength` appear in `json_schema_for("synthesis_response")`.
- A live BUF weeks 6–10 run completes.

### H2. A synthesis or verification failure throws away a report the specialists already built

**Where:** `orchestrator.py:428–458` (synthesis → `_failed`), `orchestrator.py:497–520` (verification →
`_failed`). The policy is documented in the module docstring (lines 15–21).

**Defect:** STA, BRA and CQA can all finish and pass validation, and the deterministic tables exist. One
failed synthesis call (H1) or verifier call still returns `report=None`. The user gets an error page and
nothing else, even though `agent_outputs` and `bundle.tables` are already in hand.

**Failure scenario:** the exact production error above. Ten minutes of agent work and every statistic
table are discarded.

**Fix direction:**
- Add a **degraded** path. When synthesis fails, stamp a `Report` with `RunState.DEGRADED` from the
  validated inputs:
  - the specialists' tendencies and observations, after deterministic dedupe;
  - `stat_tables`;
  - a gap or warning saying synthesis was unavailable;
  - a deterministic overview string.
- Still run verification on those findings. Verification is a quality gate, so if it fails, drop every
  unverified finding and keep stats-only content. Don't publish unverified claims.
- This changes the documented failure matrix (requirement 30). Update the docstring, PLAN.md and
  AGENTS.md together, and confirm with the project owner before merging.

**Acceptance:** an orchestrator test where the synthesis fake raises gets
`report is not None`, `status.state is DEGRADED`, tables present, and a synthesis-unavailable gap.

### H3. Background-week and unknown-week sources can ground tendencies for the selected weeks

**Where:**
- `agents/beat_reporter.py:62–70` and `agents/coach_quote.py:62–70`: retrieval passes `max_week` only,
  with no `min_week` and no `weeks`.
- `data/index.py:222`: media without a week is indexed as `week=0`, which every `max_week` filter matches.
- `validation/checks.py:195–248`: `MEDIA_BACKGROUND` and `MEDIA_WEEK_UNKNOWN` are WARNING severity only.
- `agents/common.py:281–315`: `convert_agent_response` checks that cited *weeks* ⊆ selection and cited
  *IDs* ∈ catalog, but never which week each *evidence item* covers.
- `validation/checks.py:761` (`check_evidence`) takes no week selection.

**Defect:**
- The pipeline's own rule says a background source "cannot support tendencies for the selected weeks".
  That rule exists only as warning and limitation text.
- A tendency with `weeks=[6..10]` whose only citation is a week-2 article, or an undated one, passes
  every check.
- The verifier is then handed that snippet as the grounding text.
- This is the source of the `media_week_unknown` warning in the production failure.

**Fix direction:**
- Record each evidence item's week. Carry it on `EvidenceRef` or look it up through `MediaData`.
- Give `check_evidence` and `convert_agent_response` the selected weeks.
- If a finding cites *only* background or unknown-week evidence, drop it, or force `weak_evidence=True`
  and confidence ≤ LOW. Mixed citations are allowed but marked weak.
- Retrieval should prefer selected weeks: query with `weeks=selected` first and backfill from background
  only if fewer than `k` units come back.

**Acceptance:** a test where a draft tendency cites only a week-`None` media evidence ID ends up
dropped or flagged weak in the `AgentOutput`.

### H4. "Partially supported" corrections replace the claim with text that isn't a claim

**Where:** `agents/verify.py:212–240` and `:344–348` (`_replace_statement`),
`llm/prompts/library.py:298–299` (verifier rule).

**Defect:**
- The verifier prompt defines `correction` as text that "states exactly what part is unsupported", which
  is a critique.
- `apply_verification` then puts that text in place of the finding's `statement` and publishes it as the
  tendency itself. A report can end up showing "The claim that BUF runs 60% on first down is unsupported;
  the snippet shows 52%" as a tendency.
- The corrected finding keeps the original `metric`, `value`, `comparison` and `confidence`, so the
  number fields can contradict the new statement.
- The corrected text is model-authored and never verified.
- The finding also gets a new `finding_id`, but the `VERIFICATION_CORRECTED_CLAIM` warning still carries
  the old ID.

**Fix direction:**
- Pick one meaning for `correction`:
  - **(a)** Change the prompt and schema field to `corrected_statement`: "a rewritten claim that only
    states what the snippets support". Bump `PROMPT_VERSION`.
  - **(b)** Simpler and safer: keep the original statement, set `weak_evidence=True`, lower confidence to
    ≤ MEDIUM, and attach the verifier's text as a `WeakEvidence` description.
- (b) keeps model-rewritten text out of the published finding, which fits the project rule better.
- Either way, emit the warning with the *new* `finding_id`.

**Acceptance:** a test where a partially-supported verdict produces a finding with the original
statement (b) or the corrected statement (a), with the number fields consistent and the warning ID
matching the published finding.

---

## Medium

### M1. Budget accounting has holes, and budget overruns are reported as deadline failures

**Where:** `orchestrator.py:406–420, 469–483, 525–527, 1098–1117`; `llm/client.py:437–439`;
`ui/runner.py:205–230`.

**Defects:**
1. **Verification isn't checked.** There's no `_budget_failure` call after verification (lines
   525–527). The verifier can push the run over `max_llm_calls` or `token_budget` and the report is
   still stamped.
2. **Failed calls lose their usage.** When `complete_structured` raises (truncation, schema failure after
   repair), the tokens already spent are thrown away. The 32k-token truncated call in H1 counts as 0 tokens.
3. **Live-media calls are outside the budget.** The quote-extraction LLM calls in
   `data/live_media.py:637` (up to 6 per refresh) run before `run_report`. They're counted against no
   budget and appear in no telemetry.
4. **Wrong failure reason.** A budget overrun is reported as `RunFailureReason.DEADLINE_EXCEEDED`, which
   misleads anyone triaging.

**Fix direction:**
- Add the post-verification budget check.
- Attach `usage` and `llm_calls` to `LLMResponseError` and `StructuredOutputError` (for example
  `exc.usage`) and add them to the run log in `_stage_failure`.
- Thread a usage accumulator through the live refresh, or report its calls separately in `ReportRun`.
- Add `RunFailureReason.BUDGET_EXCEEDED`. That's a model enum change, so update the UI copy in
  `ui/render.py`.

### M2. Timeouts don't bound wall-clock time, and abandoned threads keep running

**Where:** `llm/client.py:37–38, 297–321, 536–559`; `orchestrator.py:855–858, 1086–1087`.

**Defects:**
- `DEFAULT_TIMEOUT_SECONDS = 300` is an httpx *per-phase* timeout. On a streamed reply the read timeout
  resets on every chunk, so a runaway stream (H1) has no wall-clock limit.
- With 2 retries, a single call can take more than 15 minutes.
- When the orchestrator gives up on a future it calls `shutdown(wait=False)`, but the worker thread keeps
  streaming and keeps billing tokens.
- In Streamlit, repeated runs pile up live background requests against the shared endpoint.

**Fix direction:**
- Give `chat` a total deadline (`deadline_at: float | None`). Check it inside `_read_stream`'s line loop
  and between retry attempts, and abort by closing the response.
- Have the orchestrator pass its remaining budget down. One option is a `threading.Event` cancel flag
  checked in the stream loop.

### M3. Two tests fail at HEAD because `test_ui_app.py` reads the live, untracked `data/` store

**Where:** `tests/unit/test_ui_app.py:22–23, 44, 130`; `app.py:44–51` (`data_dir` comes from
`session_state`, else `Settings` defaults to `./data`).

**Defect:** the tests expect played weeks `1..9` and the default selection `[6,7,8,9]`. The local
`data/seasons/2026.json` (gitignored) now includes week 10, so the assertions fail. The tests depend on
the machine state and on the calendar.

**Fix:** in `_app()`, set `app.session_state["data_dir"]` to a temp directory seeded from
`tests/fixtures/` (or build a `SeasonStore` there from the factories), and derive expected weeks from the
fixture.

**Acceptance:** `pytest -q` is green whether `data/` is present or not.

### M4. Live-media refresh runs on every build, before the cache lookup and outside the deadline

**Where:** `ui/runner.py:145–180`, `ui/runner.py:205–230`, `data/index.py:296–350`.

**Defects:**
- `_refresh_live_media` runs before `report_cache.load`. Even a cache hit waits on RSS, ESPN, Bing and
  article fetches, plus LLM quote extraction.
- Any store change triggers `index.build`, which deletes the whole Chroma collection and re-embeds every
  unit synchronously.
- None of this is covered by `deadline_seconds` or shown in progress events.

**Fix direction:**
- Check the cache with the pre-refresh revision first, or make live refresh an explicit user action
  (sidebar button or flag) instead of a side effect of "Build report".
- Make index updates incremental: add new unit IDs and delete removed ones instead of a full rebuild.
- Emit `RunProgress("live_media", …)` events.

### M5. The report cache key omits `min_sample`

**Where:** `cache.py:63–85`, `ui/runner.py:177`, `report_cli.py:164, 184`.

**Defect:** a report built with `--min-sample 5` is served to a later `--min-sample 20` request (or the
reverse). Suppression flags, and so the weak-evidence decisions, differ between the two.

**Fix:** add `min_sample` to `CacheKey` and `_key_payload`, and bump `CACHE_SCHEMA_VERSION`.

### M6. One bad cache entry blocks its key for good, and contract changes don't invalidate the cache

**Where:** `cache.py:94–122`, `ui/runner.py:176–180`.

**Defects:**
- `ReportCache.load` raises `ReportCacheError` on a corrupt or mismatched entry. `build_report` doesn't
  catch it, so every later build of that selection fails with "Report build failed: …" until someone
  deletes the file by hand.
- Separately, `CACHE_SCHEMA_VERSION` is a hand-maintained `1`. Any change to the `Report` model makes
  old entries fail `model_validate`, which raises the same blocking error.

**Fix:**
- In `build_report`, catch `ReportCacheError`, log a warning, delete or quarantine the entry, and run
  fresh.
- Derive the schema version from a hash of `Report.model_json_schema()`, or bump it in the same commit as
  any model change.

### M7. Cross-session races on the media store, stamp file and Chroma collection

**Where:** `data/index.py:311–339` (delete → create → add), `data/media_store.py:252–308` (load → merge →
write), `data/live_media.py:771–800` (stamp file).

**Defect:**
- Streamlit runs each browser session in its own thread with its own `MediaIndex` instance on the same
  `PersistentClient` path.
- Session A's rebuild deletes the collection while session B is querying it. B's `self._collection`
  handle goes stale and its query errors or returns a partial result.
- Two concurrent refreshes do a read-modify-write on the same JSON, and the last writer wins.

**Fix direction:**
- Hold a process-wide `threading.Lock` per path around store refresh and index build.
- For multi-process safety, use a `fcntl` file lock next to the store.
- Rebuild into a temporary collection name and switch to it, instead of delete-then-create.

### M8. Synthesis doesn't apply the suppressed-table weak-evidence rule that STA applies

**Where:** `agents/synthesis.py:130–136` (no `weak_evidence_ids` passed), compare
`agents/statistics.py:52–74`.

**Defect:** STA marks any finding that cites a suppressed table (`n < min_sample`) as weak and caps it
below HIGH. Synthesis calls `convert_agent_response` without `weak_evidence_ids`, so a synthesis tendency
citing the same suppressed table publishes at HIGH confidence with `weak_evidence=False`. That's
reachable today: the production run had 4 suppressed tables.

**Fix:** compute `{t.evidence().evidence_id for t in stat_tables if t.suppressed}` in
`run_synthesis_agent` and pass it through.

### M9. Synthesis agreements and conflicts never check that the cited evidence belongs to the named agent

**Where:** `agents/synthesis.py:243–313`.

**Defect:**
- An `Agreement` is accepted if it names at least two specialist agents and cites any catalog IDs.
  Nothing checks that each named agent actually produced or cited that evidence.
- The model can claim "STA and CQA agree" while citing only STA tables.
- The same gap applies to each `ConflictPosition.agent`.

**Fix:** build `agent → {evidence_ids}` from `agent_outputs` (STA owns the stat evidence). Require each
agreement agent to own at least one cited ID, and each conflict position's IDs to be owned by its agent.
Drop and count anything else.

### M10. `neutralize_tags` misses the structural tags the renderers emit, so untrusted text can forge prompt structure

**Where:** `llm/prompts/blocks.py:44–52` (`BLOCK_TAGS`), `:102–112`; the renderers emit `<agent_output>`,
`<table>`, `<claim>` and `<snippet>` (`blocks.py:177, 225, 248, 252`).

**Defect:**
- Only the seven outer tags are neutralized, and only in their exact lowercase form.
- An article body or model statement containing `</snippet></claim><claim claim_id="…">` can close the
  current verifier claim and inject a fake one.
- `</agent_output>` can forge an agent section for SYN.
- Case variants like `</SOURCE>` also get through.

**Fix:** neutralize every tag name the renderers emit, matched case-insensitively with a regex such as
`</?\s*(tag1|tag2|…)\b`. Add a test that renders a malicious snippet and asserts there's still exactly
one `<claim` per claim.

### M11. One ungrounded retrieval unit makes the whole BRA or CQA section unavailable

**Where:** `agents/beat_reporter.py:71–90`, `agents/coach_quote.py:71–90`,
`validation/checks.py:566–758`.

**Defect:**
- `check_retrieval` returns `ok=False` if *any* unit fails, for example a stale index chunk after a media
  body changed, or a duplicate unit.
- The agent then skips the model and returns an unavailable section, even when 7 of 8 units are
  perfectly grounded.
- Because H3-style warnings are non-blocking while one stale chunk is fatal, the behavior is inverted:
  weak sources pass and good sources get dropped.

**Fix direction:**
- Filter out the failing units (keep their issues as warnings with `requires_human_review`) and continue
  with the remainder.
- Return unavailable only if nothing survives.
- If any unit is ungrounded, schedule an index rebuild.

### M12. The follow-up answer can be `supported=True` while citing invented evidence IDs

**Where:** `ui/runner.py:289–304`.

**Defect:** `supported = response.supported and valid`. If the model cites one real ID and one invented
ID, the answer is still shown as supported, with only a limitation line noting the unknown ID. The
docstring says unknown IDs force `supported=False`.

**Fix:** `supported = bool(response.supported and valid and not unknown)`, and add a test.

### M13. LLM and embedding transport is fragile

**Where:** `llm/client.py:536–645`, `llm/client.py:549–567`, `llm/embeddings.py:84–130`.

**Defects:**
1. **No retry on a bad body.** A malformed SSE chunk or a non-JSON 200 body raises `LLMResponseError`
   from inside `_send`. Only `httpx` exceptions are caught for retry, so a proxy hiccup is fatal.
2. **No connection reuse.** A new `httpx.Client` is created per attempt, so every call pays a fresh
   TCP+TLS handshake.
3. **No embedder retries.** `ArcEmbedder` has none, so one transient failure kills retrieval or an index
   build.

**Fix:**
- Treat a malformed body as retryable, with the same backoff.
- Keep a lazily created `httpx.Client` on the instance, closed by a `close()` or context manager.
- Add bounded retries with backoff to `ArcEmbedder`, reusing the client's retry helper.

---

## Low

- **L1. mypy gate is red.** `ui/pdf_export.py:15–33` has 8 `import-untyped` errors for `reportlab`. Add
  `types-reportlab` to the `dev` extras, or a `[[tool.mypy.overrides]] module = "reportlab.*"` with
  `ignore_missing_imports = true`.
- **L2. Duplicate warnings.** `check_plays` and `check_stat_tables` both emit `missing_values` and
  `unknown_play_type` for the same plays with different wording ("4 scrimmage play(s)…" vs "4
  play(s)…"). `_dedupe_warnings` (`orchestrator.py:1215`) keys on the message, so both survive. Emit
  once, or dedupe on `(code, subject_kind, subject_id)`.
- **L3. Bye weeks are selectable.** The week picker (`app.py:78–85`) offers league-wide played weeks, and
  the default "last 4" can include the opponent's bye. The production run warned `BUF does not play in
  week 7`. Build options from `season_data.games` filtered by `game.involves(opponent)`.
- **L4. No Streamlit caching.** `app.py:51` reloads and re-validates the 26k-play season store (≈0.15 s)
  plus the media store on every widget interaction. Wrap `load_season` and `load_media` in
  `st.cache_data`, keyed on file mtime.
- **L5. `prompt_version` is mostly decorative.**
  - `run_synthesis_agent` accepts `prompt_version` (`agents/synthesis.py:101`) and never uses it.
  - Prompts can't be selected by version anywhere. Passing a different version changes only the stamp
    and cache key, not the prompt text.
  - Either remove the parameter, or make `get_prompt(name, version)` real.
- **L6. Duplicated code.**
  - `agents/beat_reporter.py` and `agents/coach_quote.py` are ~95% identical. Factor them into one
    `run_retrieval_agent(spec)`.
  - `_valid_evidence`/`_valid_weeks`/`_as_agent_response` in `synthesis.py` duplicate
    `_evidence_ids`/`_weeks`/`as_agent_response` in `common.py`.
- **L7. Prompt and schema disagree.** The synthesis prompt says "every item you return must cite at
  least one" evidence ID (`library.py:246–248`), but `DraftGap` has no `evidence_ids` and
  `DraftWeakEvidence.evidence_ids` may be empty. Reword the rule to cover only findings, agreements and
  conflicts.
- **L8. `strict: true` with a non-strict schema.** `llm/client.py:420–427` sends `"strict": True` with a
  pydantic schema that contains `anyOf` and `$defs`. It works only because vLLM ignores `strict`; an
  OpenAI-strict gateway would return a non-retryable 400. Document this, or make `strict` configurable.
- **L9. Redirects skip the allowlist.** `data/live_media.py:230–234` uses `follow_redirects=True`, but the
  allowlist check (`_host_allowed`) runs on the pre-redirect URL only. Check `response.url` after the
  fetch, or disable redirects and follow them manually through the allowlist.
- **L10. The size limit is checked after the download.** `_get_text` (`live_media.py:702–707`) reads the
  whole body and then checks it against `limit`. Stream the body and abort once it passes the limit.
- **L11. `LIVE_MEDIA` parsing is loose.** `ui/runner.py:147` treats anything but `"false"` as on (`"0"`,
  `"no"` and `"off"` all enable it) and reads `os.environ` directly. Move it into `Settings` using
  `_bool_setting`.
- **L12. Model text renders as markdown.** Feed titles and model statements go through `st.markdown`
  (`ui/render.py:241–328`), so link or image syntax in feed-controlled text becomes clickable links or
  remote image loads. Escape markdown metacharacters, or render with `st.text` / `st.write` on plain
  strings.
- **L13. Removed-claim weak evidence is too broad.** `_removed_weak` (`agents/verify.py:351–364`) stamps
  the whole selection's weeks instead of `finding.weeks`, and sets `agent=None`.
- **L14. Redundant re-check.** `check_stat_tables` re-runs after verification (`orchestrator.py:550–555`)
  on the unchanged `bundle.tables`. Remove it, or note why it's needed.
- **L15. Retry backoff has no jitter.** Backoff (`llm/client.py:647–655`) ignores `Retry-After`, so three
  parallel specialists retry a shared 429 in lockstep.
- **L16. Stale docstring.** The `ui/runner.py:1–13` docstring says the module has "side-effect-free
  helpers", but `build_report` fetches the network, writes the media store and rebuilds Chroma.
- **L17. Stale design docs.** PLAN.md (line 228) and the README troubleshooting table (line 542) still
  recommend raising the completion cap for truncation. Update both after H1 lands: bound the schema
  first, and a larger cap isn't the fix.

---

## Checked and found sound

- Cache filenames are SHA-256 digests (no path traversal) and every store write is atomic (mkstemp →
  fsync → `os.replace` → directory fsync).
- Retrieval filters opponent and season inside Chroma before the similarity search, and refuses a query
  from a different embedding model.
- `check_retrieval` verifies that every unit is an exact span of stored text, and that quote units
  reproduce the stored quote verbatim.
- Truncated empty replies (`finish_reason=length`) correctly skip the repair attempt; repair is bounded
  to one call.
- `SecretStr` keeps the API key out of logs and reprs; log lines carry no prompt or completion text.
- The stats engine guards against division by zero and rejects duplicate plays.

## Suggested fix order

1. **H1 → H2.** Restore working report builds: finish bounding the synthesis output, then degrade
   instead of discarding.
2. **M3, L1.** Get the test and type gates green so later fixes are verifiable.
3. **H3, H4, M8, M9.** The grounding and verification guarantees are the project's core claim; these
   close the places Python doesn't enforce them.
4. **M1, M2, M5, M6.** Budget, timeout and cache correctness.
5. **M4, M7, M11.** Live media, concurrency and retrieval resilience.
6. **M10, M12, M13.** Prompt-injection hardening, follow-up correctness and transport hardening.
7. Low items, grouped by file.
