"""End-to-end report orchestration (Stage 9).

:func:`run_report` is the one entry point the UI and CLI share:

1. Validate the opponent/week selection and the stored plays.
2. Compute every deterministic statistic table for the selection.
3. Run STA, BRA, and CQA in parallel, each behind a bounded wait.
4. Re-check every agent's citations against stored sources.
5. Merge the validated outputs with synthesis and apply verifier verdicts.
6. Re-check evidence and tables, then stamp a :class:`Report`.

Deterministic Python owns validation, statistics, evidence IDs, sorting, and the
failure decision. The models only interpret the data they are given.

Failure matrix (requirement 30):

* A validation failure, an overall deadline/token-budget overrun, or a failure
  in STA, synthesis, or verification is a **hard failure**: no report is built.
* A failure or timeout in BRA or CQA produces an explicit unavailable section,
  a warning for human review, and a **degraded** report that still carries the
  statistics and any section that completed.

Timeouts are enforced by waiting on each future for a bounded budget rather
than by trusting a model to stop. A worker thread cannot be killed, so a timed
out call is abandoned (``shutdown(wait=False)``) and the run moves on or fails;
the client's own per-call timeout bounds the abandoned request.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

from pydantic import ValidationError

from opponent_tendencies.agents.beat_reporter import run_beat_reporter_agent
from opponent_tendencies.agents.coach_quote import run_coach_quote_agent
from opponent_tendencies.agents.common import AgentResult, LLMCompleter, unavailable_output
from opponent_tendencies.agents.statistics import run_statistics_agent
from opponent_tendencies.agents.synthesis import SynthesisResult, run_synthesis_agent
from opponent_tendencies.agents.verify import (
    VerificationResult,
    VerifiedContent,
    apply_verification,
    build_verifier_claims,
    run_verifier,
)
from opponent_tendencies.config import ConfigurationError
from opponent_tendencies.data.index import MediaIndex
from opponent_tendencies.data.media_store import MediaData
from opponent_tendencies.data.play_store import SeasonData
from opponent_tendencies.llm.client import TokenUsage
from opponent_tendencies.llm.prompts import PROMPT_VERSION
from opponent_tendencies.models import (
    SPECIALIST_AGENTS,
    AgentName,
    DataUsage,
    EvidenceKind,
    EvidenceRef,
    Game,
    IssueSeverity,
    Report,
    RunFailureReason,
    RunState,
    RunStatus,
    SectionStatus,
    StatTable,
    Team,
    ValidationIssue,
    ValidationResult,
    WeakEvidence,
)
from opponent_tendencies.stats.engine import DEFAULT_MIN_SAMPLE, StatsBundle, analyze_opponent
from opponent_tendencies.validation.checks import (
    CheckCode,
    check_evidence,
    check_plays,
    check_stat_tables,
    validate_selection,
)

__all__ = [
    "DEFAULT_AGENT_TIMEOUT_SECONDS",
    "DEFAULT_DEADLINE_SECONDS",
    "DEFAULT_MAX_LLM_CALLS",
    "DEFAULT_TOKEN_BUDGET",
    "ReportRun",
    "RunProgress",
    "data_revision",
    "run_report",
]

DEFAULT_DEADLINE_SECONDS: Final = 600.0
DEFAULT_AGENT_TIMEOUT_SECONDS: Final = 360.0
# Three specialists + synthesis + verification, each allowed one repair call.
DEFAULT_MAX_LLM_CALLS: Final = 10
DEFAULT_TOKEN_BUDGET: Final = 250_000

ProgressCallback = Callable[["RunProgress"], None]


@dataclass(frozen=True)
class RunProgress:
    """One progress event a UI can render while a report is being built."""

    step: str
    status: SectionStatus
    agent: AgentName | None = None


@dataclass(frozen=True)
class ReportRun:
    """The result of one orchestration run, successful or not."""

    status: RunStatus
    report: Report | None = None
    warnings: tuple[ValidationIssue, ...] = ()
    agent_results: tuple[AgentResult, ...] = ()
    validation: ValidationResult | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    llm_calls: int = 0
    latency_seconds: float = 0.0
    synthesis: SynthesisResult | None = None
    verification: VerificationResult | None = None
    verified: VerifiedContent | None = None
    cached: bool = False


@dataclass(frozen=True)
class _StageFailure:
    reason: RunFailureReason
    message: str


@dataclass(frozen=True)
class _Call[T]:
    value: T | None
    failure: _StageFailure | None


@dataclass
class _RunLog:
    """Mutable bookkeeping for one run."""

    usage: TokenUsage = field(default_factory=TokenUsage)
    llm_calls: int = 0
    warnings: list[ValidationIssue] = field(default_factory=list)
    agent_results: list[AgentResult] = field(default_factory=list)
    agent_status: dict[AgentName, SectionStatus] = field(default_factory=dict)
    validation: ValidationResult | None = None
    synthesis: SynthesisResult | None = None
    verification: VerificationResult | None = None
    verified: VerifiedContent | None = None

    def add_result(self, result: AgentResult) -> None:
        self.agent_results.append(result)
        self.agent_status[result.output.agent] = result.output.status
        self.usage += result.usage
        self.llm_calls += result.llm_calls
        self.warnings.extend(result.warnings)

    def add_validation(self, result: ValidationResult) -> None:
        self.validation = (
            result if self.validation is None else ValidationResult.merge(self.validation, result)
        )
        self.warnings.extend(result.warnings)


def run_report(
    *,
    opponent: Team,
    weeks: Sequence[int],
    season_data: SeasonData,
    media_data: MediaData | None,
    index: MediaIndex | None,
    client: LLMCompleter,
    min_sample: int = DEFAULT_MIN_SAMPLE,
    prompt_version: str = PROMPT_VERSION,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
    agent_timeout_seconds: float = DEFAULT_AGENT_TIMEOUT_SECONDS,
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    clock: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] | None = None,
    on_progress: ProgressCallback | None = None,
) -> ReportRun:
    """Validate, run the agent pipeline, and return a report or a failure.

    ``media_data``/``index`` may be ``None`` when no media is stored; BRA and
    CQA then degrade explicitly instead of inventing sources. ``clock`` and
    ``monotonic`` are injectable so tests can drive timestamps and deadlines.
    """

    if deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be positive")
    if agent_timeout_seconds <= 0:
        raise ValueError("agent_timeout_seconds must be positive")
    if max_llm_calls < 1:
        raise ValueError("max_llm_calls must be at least 1")
    if token_budget < 1:
        raise ValueError("token_budget must be at least 1")

    now = clock if clock is not None else (lambda: datetime.now(UTC))
    tick = monotonic if monotonic is not None else time.monotonic
    started_at = now()
    started = tick()
    deadline_at = started + deadline_seconds
    log = _RunLog()

    try:
        return _execute(
            opponent=opponent,
            weeks=weeks,
            season_data=season_data,
            media_data=media_data,
            index=index,
            client=client,
            min_sample=min_sample,
            prompt_version=prompt_version,
            deadline_seconds=deadline_seconds,
            agent_timeout_seconds=agent_timeout_seconds,
            max_llm_calls=max_llm_calls,
            token_budget=token_budget,
            started_at=started_at,
            started=started,
            deadline_at=deadline_at,
            now=now,
            tick=tick,
            log=log,
            on_progress=on_progress,
        )
    except ConfigurationError as exc:
        return _failed(
            _StageFailure(
                RunFailureReason.CONFIGURATION_ERROR,
                f"The LLM is not configured correctly: {_describe(exc)}",
            ),
            log=log,
            started_at=started_at,
            finished_at=now(),
            deadline_seconds=deadline_seconds,
            latency_seconds=tick() - started,
        )


def _execute(
    *,
    opponent: Team,
    weeks: Sequence[int],
    season_data: SeasonData,
    media_data: MediaData | None,
    index: MediaIndex | None,
    client: LLMCompleter,
    min_sample: int,
    prompt_version: str,
    deadline_seconds: float,
    agent_timeout_seconds: float,
    max_llm_calls: int,
    token_budget: int,
    started_at: datetime,
    started: float,
    deadline_at: float,
    now: Callable[[], datetime],
    tick: Callable[[], float],
    log: _RunLog,
    on_progress: ProgressCallback | None,
) -> ReportRun:
    selected_weeks = tuple(sorted({int(week) for week in weeks}))
    resolved_media = media_data if media_data is not None else MediaData(season=season_data.season)

    # ------------------------------------------------------------------
    # 1. Deterministic validation and statistics
    # ------------------------------------------------------------------

    _notify(on_progress, "validation", SectionStatus.RUNNING)
    selection = validate_selection(season_data, opponent=opponent, weeks=weeks)
    log.add_validation(selection)
    if not selection.ok:
        return _failed(
            _StageFailure(
                RunFailureReason.VALIDATION_FAILED,
                f"The selection is not playable: {_errors(selection)}",
            ),
            log=log,
            started_at=started_at,
            finished_at=now(),
            deadline_seconds=deadline_seconds,
            latency_seconds=tick() - started,
        )

    games = tuple(
        game for game in season_data.games_for_weeks(selected_weeks) if game.involves(opponent)
    )
    if not games:
        return _failed(
            _StageFailure(
                RunFailureReason.NO_GAMES,
                f"{opponent.value} has no stored regular-season games in the selected weeks.",
            ),
            log=log,
            started_at=started_at,
            finished_at=now(),
            deadline_seconds=deadline_seconds,
            latency_seconds=tick() - started,
        )

    plays = check_plays(
        season_data.plays,
        opponent=opponent,
        game_ids=[game.game_id for game in games],
    )
    log.add_validation(plays)
    if not plays.ok:
        return _failed(
            _StageFailure(
                RunFailureReason.VALIDATION_FAILED,
                f"The stored plays failed validation: {_errors(plays)}",
            ),
            log=log,
            started_at=started_at,
            finished_at=now(),
            deadline_seconds=deadline_seconds,
            latency_seconds=tick() - started,
        )

    try:
        bundle = analyze_opponent(
            season_data.plays,
            games,
            opponent=opponent,
            min_sample=min_sample,
        )
    except Exception as exc:  # the engine rejects unplayable or forked input
        return _failed(
            _StageFailure(
                RunFailureReason.VALIDATION_FAILED,
                f"The deterministic statistics could not be computed: {_describe(exc)}",
            ),
            log=log,
            started_at=started_at,
            finished_at=now(),
            deadline_seconds=deadline_seconds,
            latency_seconds=tick() - started,
        )

    tables = check_stat_tables(
        bundle.tables,
        weeks=selected_weeks,
        game_ids=[game.game_id for game in games],
        quality=bundle.quality,
    )
    log.add_validation(tables)
    if not tables.ok:
        return _failed(
            _StageFailure(
                RunFailureReason.VALIDATION_FAILED,
                f"The deterministic statistics failed validation: {_errors(tables)}",
            ),
            log=log,
            started_at=started_at,
            finished_at=now(),
            deadline_seconds=deadline_seconds,
            latency_seconds=tick() - started,
        )

    # ------------------------------------------------------------------
    # 2. Parallel specialist fan-out
    # ------------------------------------------------------------------

    _notify(on_progress, "validation", SectionStatus.COMPLETED)
    for agent in SPECIALIST_AGENTS:
        _notify(on_progress, agent.value.lower(), SectionStatus.RUNNING, agent=agent)

    fan = _run_specialists(
        client=client,
        opponent=opponent,
        games=games,
        weeks=selected_weeks,
        bundle=bundle,
        index=index,
        media_data=media_data,
        season_data=season_data,
        prompt_version=prompt_version,
        agent_timeout_seconds=agent_timeout_seconds,
        deadline_at=deadline_at,
        tick=tick,
        on_progress=on_progress,
    )
    for result in fan.results:
        log.add_result(result)
    if fan.failure is not None:
        return _failed(
            fan.failure,
            log=log,
            started_at=started_at,
            finished_at=now(),
            deadline_seconds=deadline_seconds,
            latency_seconds=tick() - started,
        )

    budget_failure = _budget_failure(
        log,
        max_llm_calls=max_llm_calls,
        token_budget=token_budget,
        stage="the specialist agents",
    )
    if budget_failure is not None:
        return _failed(
            budget_failure,
            log=log,
            started_at=started_at,
            finished_at=now(),
            deadline_seconds=deadline_seconds,
            latency_seconds=tick() - started,
        )

    outputs = tuple(result.output for result in fan.results)

    # ------------------------------------------------------------------
    # 3. Synthesis (a failure here is a hard failure)
    # ------------------------------------------------------------------

    log.agent_status[AgentName.SYNTHESIS] = SectionStatus.RUNNING
    _notify(on_progress, "synthesis", SectionStatus.RUNNING, agent=AgentName.SYNTHESIS)
    synthesis_call = _call_with_deadline(
        lambda: run_synthesis_agent(
            client=client,
            opponent=opponent,
            season=season_data.season,
            weeks=selected_weeks,
            games=games,
            agent_outputs=outputs,
            stat_tables=bundle.tables,
            prompt_version=prompt_version,
        ),
        timeout_seconds=agent_timeout_seconds,
        deadline_at=deadline_at,
        tick=tick,
        on_timeout="Synthesis did not finish within its time limit.",
        on_error="Synthesis failed",
        failure_reason=RunFailureReason.SYNTHESIS_FAILED,
    )
    if synthesis_call.failure is not None:
        log.agent_status[AgentName.SYNTHESIS] = SectionStatus.FAILED
        _notify(on_progress, "synthesis", SectionStatus.FAILED, agent=AgentName.SYNTHESIS)
        return _failed(
            synthesis_call.failure,
            log=log,
            started_at=started_at,
            finished_at=now(),
            deadline_seconds=deadline_seconds,
            latency_seconds=tick() - started,
        )
    synthesis = synthesis_call.value
    if synthesis is None:  # pragma: no cover - guarded by the failure branch
        raise RuntimeError("synthesis returned neither a result nor a failure")
    log.synthesis = synthesis
    log.usage += synthesis.usage
    log.llm_calls += synthesis.llm_calls
    log.warnings.extend(synthesis.warnings)
    log.agent_status[AgentName.SYNTHESIS] = SectionStatus.COMPLETED
    _notify(on_progress, "synthesis", SectionStatus.COMPLETED, agent=AgentName.SYNTHESIS)

    budget_failure = _budget_failure(
        log,
        max_llm_calls=max_llm_calls,
        token_budget=token_budget,
        stage="synthesis",
    )
    if budget_failure is not None:
        return _failed(
            budget_failure,
            log=log,
            started_at=started_at,
            finished_at=now(),
            deadline_seconds=deadline_seconds,
            latency_seconds=tick() - started,
        )

    # ------------------------------------------------------------------
    # 4. Verification (a separate call, not a fifth agent)
    # ------------------------------------------------------------------

    claims = build_verifier_claims(
        [*synthesis.tendencies, *synthesis.observations],
        synthesis.evidence,
        season_data=season_data,
        media_data=resolved_media,
        stat_tables=bundle.tables,
    )
    _notify(on_progress, "verification", SectionStatus.RUNNING)
    verification_call = _call_with_deadline(
        lambda: run_verifier(
            client=client,
            opponent=opponent,
            weeks=selected_weeks,
            claims=claims,
        ),
        timeout_seconds=agent_timeout_seconds,
        deadline_at=deadline_at,
        tick=tick,
        on_timeout="Claim verification did not finish within its time limit.",
        on_error="Claim verification failed",
        failure_reason=RunFailureReason.VERIFICATION_FAILED,
    )
    if verification_call.failure is not None:
        _notify(on_progress, "verification", SectionStatus.FAILED)
        return _failed(
            verification_call.failure,
            log=log,
            started_at=started_at,
            finished_at=now(),
            deadline_seconds=deadline_seconds,
            latency_seconds=tick() - started,
        )
    verification = verification_call.value
    if verification is None:  # pragma: no cover - guarded by the failure branch
        raise RuntimeError("verification returned neither a result nor a failure")
    _notify(on_progress, "verification", SectionStatus.COMPLETED)
    log.verification = verification
    log.usage += verification.usage
    log.llm_calls += verification.llm_calls

    verified = apply_verification(
        tendencies=synthesis.tendencies,
        observations=synthesis.observations,
        claims=claims,
        verdicts=verification.by_claim_id(),
        weeks=selected_weeks,
    )
    log.verified = verified
    log.warnings.extend(verified.warnings)

    # ------------------------------------------------------------------
    # 5. Deterministic re-checks after synthesis and verification
    # ------------------------------------------------------------------

    final_evidence = synthesis.evidence
    evidence_check = check_evidence(
        final_evidence,
        season_data=season_data,
        media_data=resolved_media,
        stat_tables=bundle.tables,
    )
    final_tables = check_stat_tables(
        bundle.tables,
        weeks=selected_weeks,
        game_ids=[game.game_id for game in games],
        quality=bundle.quality,
    )
    log.warnings.extend(evidence_check.warnings)
    log.warnings.extend(final_tables.warnings)
    if not evidence_check.ok or not final_tables.ok:
        problems = "; ".join(
            message
            for result in (evidence_check, final_tables)
            for message in (issue.message for issue in result.errors)
        )
        return _failed(
            _StageFailure(
                RunFailureReason.VALIDATION_FAILED,
                f"Post-synthesis validation failed: {problems}",
            ),
            log=log,
            started_at=started_at,
            finished_at=now(),
            deadline_seconds=deadline_seconds,
            latency_seconds=tick() - started,
        )

    # ------------------------------------------------------------------
    # 6. Stamp the report
    # ------------------------------------------------------------------

    finished_at = now()
    agent_status = {output.agent: output.status for output in outputs}
    agent_status[AgentName.SYNTHESIS] = SectionStatus.COMPLETED
    state = (
        RunState.COMPLETED
        if all(status is SectionStatus.COMPLETED for status in agent_status.values())
        else RunState.DEGRADED
    )
    status = RunStatus(
        state=state,
        agent_status=agent_status,
        current_step="complete",
        progress=1.0,
        started_at=started_at,
        finished_at=finished_at,
        deadline_seconds=deadline_seconds,
    )

    weak_evidence = _merge_weak_evidence(synthesis.weak_evidence, verified.weak_evidence)
    warnings = _dedupe_warnings(log.warnings)
    data_revision_value = data_revision(season_data, resolved_media)
    data_used = _data_usage(
        bundle=bundle,
        games=games,
        evidence=final_evidence,
        media_data=resolved_media,
    )

    try:
        report = Report(
            opponent=opponent,
            season=season_data.season,
            weeks=list(selected_weeks),
            games=list(games),
            generated_at=finished_at,
            data_revision=data_revision_value,
            prompt_version=prompt_version,
            model=_model_name(client, synthesis),
            status=status,
            overview=synthesis.overview,
            data_used=data_used,
            stat_tables=list(bundle.tables),
            tendencies=list(verified.tendencies),
            observations=list(verified.observations),
            agreements=list(synthesis.agreements),
            conflicts=list(synthesis.conflicts),
            gaps=list(synthesis.gaps),
            weak_evidence=list(weak_evidence),
            warnings=list(warnings),
            evidence=list(final_evidence),
            agent_outputs=list(outputs),
        )
    except (ValidationError, ValueError) as exc:
        return _failed(
            _StageFailure(
                RunFailureReason.INTERNAL_ERROR,
                f"Assembling the report failed validation: {_describe(exc)}",
            ),
            log=log,
            started_at=started_at,
            finished_at=finished_at,
            deadline_seconds=deadline_seconds,
            latency_seconds=tick() - started,
        )

    _notify(on_progress, "report", SectionStatus.COMPLETED)
    return ReportRun(
        status=status,
        report=report,
        warnings=warnings,
        agent_results=tuple(log.agent_results),
        validation=log.validation,
        usage=log.usage,
        llm_calls=log.llm_calls,
        latency_seconds=tick() - started,
        synthesis=synthesis,
        verification=verification,
        verified=verified,
    )


# ---------------------------------------------------------------------------
# Specialist fan-out
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FanOut:
    results: tuple[AgentResult, ...]
    failure: _StageFailure | None


def _run_specialists(
    *,
    client: LLMCompleter,
    opponent: Team,
    games: Sequence[Game],
    weeks: Sequence[int],
    bundle: StatsBundle,
    index: MediaIndex | None,
    media_data: MediaData | None,
    season_data: SeasonData,
    prompt_version: str,
    agent_timeout_seconds: float,
    deadline_at: float,
    tick: Callable[[], float],
    on_progress: ProgressCallback | None,
) -> _FanOut:
    """Run STA, BRA, and CQA concurrently behind bounded waits."""

    def statistics() -> AgentResult:
        return run_statistics_agent(
            client=client,
            games=games,
            weeks=weeks,
            opponent=opponent,
            bundle=bundle,
            prompt_version=prompt_version,
        )

    def beat_reporter() -> AgentResult:
        if index is None or media_data is None:
            return _unavailable_media_result(
                AgentName.BEAT_REPORTER,
                opponent=opponent,
                games=games,
                weeks=weeks,
                prompt_version=prompt_version,
            )
        return run_beat_reporter_agent(
            client=client,
            index=index,
            media_data=media_data,
            games=games,
            weeks=weeks,
            opponent=opponent,
            prompt_version=prompt_version,
        )

    def coach_quote() -> AgentResult:
        if index is None or media_data is None:
            return _unavailable_media_result(
                AgentName.COACH_QUOTE,
                opponent=opponent,
                games=games,
                weeks=weeks,
                prompt_version=prompt_version,
            )
        return run_coach_quote_agent(
            client=client,
            index=index,
            media_data=media_data,
            games=games,
            weeks=weeks,
            opponent=opponent,
            prompt_version=prompt_version,
        )

    tasks: Mapping[AgentName, Callable[[], AgentResult]] = {
        AgentName.STATISTICS: statistics,
        AgentName.BEAT_REPORTER: beat_reporter,
        AgentName.COACH_QUOTE: coach_quote,
    }

    pool = ThreadPoolExecutor(max_workers=len(tasks), thread_name_prefix="opponent-tendencies")
    futures = {agent: pool.submit(task) for agent, task in tasks.items()}
    results: list[AgentResult] = []
    failure: _StageFailure | None = None
    try:
        for agent in SPECIALIST_AGENTS:
            remaining = deadline_at - tick()
            if remaining <= 0:
                failure = _StageFailure(
                    RunFailureReason.DEADLINE_EXCEEDED,
                    "The run reached its overall deadline before every agent finished.",
                )
                break

            budget = min(agent_timeout_seconds, remaining)
            outcome = _await(futures[agent], budget=budget, deadline_at=deadline_at, tick=tick)
            if outcome.result is not None:
                problem = _agent_problem(
                    outcome.result,
                    agent=agent,
                    opponent=opponent,
                    games=games,
                    weeks=weeks,
                    season_data=season_data,
                    media_data=media_data,
                    tables=bundle.tables,
                )
                if problem is None:
                    results.append(outcome.result)
                    _notify(
                        on_progress,
                        agent.value.lower(),
                        outcome.result.output.status,
                        agent=agent,
                    )
                    continue
                if agent is AgentName.STATISTICS:
                    _notify(
                        on_progress,
                        agent.value.lower(),
                        SectionStatus.FAILED,
                        agent=agent,
                    )
                    failure = _StageFailure(
                        RunFailureReason.STATISTICS_AGENT_FAILED,
                        f"The statistics agent output failed validation: {problem}",
                    )
                    break
                results.append(
                    _failed_agent_result(
                        agent,
                        problem=problem,
                        opponent=opponent,
                        games=games,
                        weeks=weeks,
                        prompt_version=prompt_version,
                    )
                )
                _notify(on_progress, agent.value.lower(), SectionStatus.UNAVAILABLE, agent=agent)
                continue

            if outcome.timed_out:
                if agent is AgentName.STATISTICS:
                    _notify(
                        on_progress,
                        agent.value.lower(),
                        SectionStatus.FAILED,
                        agent=agent,
                    )
                    failure = _StageFailure(
                        RunFailureReason.DEADLINE_EXCEEDED
                        if outcome.deadline_hit
                        else RunFailureReason.STATISTICS_AGENT_FAILED,
                        "The statistics agent did not finish within its time limit.",
                    )
                    break
                results.append(
                    _timed_out_agent_result(
                        agent,
                        opponent=opponent,
                        games=games,
                        weeks=weeks,
                        prompt_version=prompt_version,
                    )
                )
                _notify(on_progress, agent.value.lower(), SectionStatus.TIMED_OUT, agent=agent)
                continue

            if agent is AgentName.STATISTICS:
                _notify(
                    on_progress,
                    agent.value.lower(),
                    SectionStatus.FAILED,
                    agent=agent,
                )
                failure = _StageFailure(
                    RunFailureReason.STATISTICS_AGENT_FAILED,
                    f"The statistics agent failed: {outcome.error}",
                )
                break
            results.append(
                _failed_agent_result(
                    agent,
                    problem=f"the agent failed: {outcome.error}",
                    opponent=opponent,
                    games=games,
                    weeks=weeks,
                    prompt_version=prompt_version,
                )
            )
            _notify(on_progress, agent.value.lower(), SectionStatus.UNAVAILABLE, agent=agent)
    finally:
        # A timed-out worker cannot be killed; abandon it instead of blocking
        # the run forever. Its client call is bounded by the client timeout.
        pool.shutdown(wait=False, cancel_futures=True)

    return _FanOut(results=tuple(results), failure=failure)


@dataclass(frozen=True)
class _Outcome:
    result: AgentResult | None = None
    error: str | None = None
    timed_out: bool = False
    deadline_hit: bool = False


def _await(
    future: Future[AgentResult],
    *,
    budget: float,
    deadline_at: float,
    tick: Callable[[], float],
) -> _Outcome:
    """Wait for one specialist future without ever waiting unbounded."""

    try:
        return _Outcome(result=future.result(timeout=budget))
    except FutureTimeoutError:
        if not future.done():
            return _Outcome(
                timed_out=True,
                deadline_hit=tick() + 1e-9 >= deadline_at,
            )
        exception = future.exception()
        if exception is None:
            return _Outcome(result=future.result())
        return _Outcome(error=_describe(exception))
    except Exception as exc:
        return _Outcome(error=_describe(exc))


def _agent_problem(
    result: AgentResult,
    *,
    agent: AgentName,
    opponent: Team,
    games: Sequence[Game],
    weeks: Sequence[int],
    season_data: SeasonData,
    media_data: MediaData | None,
    tables: Sequence[StatTable],
) -> str | None:
    """Return why an agent result is untrustworthy, or ``None`` if it is sound."""

    output = result.output
    if output.agent is not agent:
        return f"expected a {agent.value} output, got {output.agent.value}"
    if output.opponent is not opponent:
        return f"output targets {output.opponent.value}, not {opponent.value}"
    if list(output.games) != list(games):
        return "output games do not match the selected games"
    if list(output.weeks) != list(weeks):
        return "output weeks do not match the selected weeks"

    evidence = check_evidence(
        output.evidence,
        season_data=season_data,
        media_data=media_data if media_data is not None else MediaData(season=season_data.season),
        stat_tables=tables,
    )
    if not evidence.ok:
        return f"citations do not resolve to stored sources: {_errors(evidence)}"

    if agent is AgentName.STATISTICS:
        expected = [table.stat_table_id for table in tables]
        actual = [table.stat_table_id for table in output.stat_tables]
        if actual != expected:
            return "the statistics agent returned tables other than the deterministic ones"
    return None


def _unavailable_media_result(
    agent: AgentName,
    *,
    opponent: Team,
    games: Sequence[Game],
    weeks: Sequence[int],
    prompt_version: str,
) -> AgentResult:
    limitation = (
        "No media index or stored reporting is available for this season, so this "
        "section has no retrieved sources."
    )
    return AgentResult(
        output=unavailable_output(
            agent=agent,
            opponent=opponent,
            games=games,
            weeks=weeks,
            prompt_version=prompt_version,
            limitation=limitation,
        ),
        warnings=(
            _warning(
                CheckCode.AGENT_UNAVAILABLE,
                f"{agent.value} is unavailable: {limitation}",
                subject_kind="agent",
                subject_id=agent.value,
            ),
        ),
    )


def _failed_agent_result(
    agent: AgentName,
    *,
    problem: str,
    opponent: Team,
    games: Sequence[Game],
    weeks: Sequence[int],
    prompt_version: str,
) -> AgentResult:
    limitation = f"The {agent.value} section is unavailable because {problem}."
    return AgentResult(
        output=unavailable_output(
            agent=agent,
            opponent=opponent,
            games=games,
            weeks=weeks,
            prompt_version=prompt_version,
            limitation=limitation,
        ),
        warnings=(
            _warning(
                CheckCode.AGENT_UNAVAILABLE,
                limitation,
                subject_kind="agent",
                subject_id=agent.value,
            ),
        ),
    )


def _timed_out_agent_result(
    agent: AgentName,
    *,
    opponent: Team,
    games: Sequence[Game],
    weeks: Sequence[int],
    prompt_version: str,
) -> AgentResult:
    limitation = f"The {agent.value} section timed out and produced no findings."
    return AgentResult(
        output=unavailable_output(
            agent=agent,
            opponent=opponent,
            games=games,
            weeks=weeks,
            prompt_version=prompt_version,
            limitation=limitation,
            status=SectionStatus.TIMED_OUT,
        ),
        warnings=(
            _warning(
                CheckCode.AGENT_TIMEOUT,
                limitation,
                subject_kind="agent",
                subject_id=agent.value,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Bounded stage calls
# ---------------------------------------------------------------------------


def _call_with_deadline[T](
    call: Callable[[], T],
    *,
    timeout_seconds: float,
    deadline_at: float,
    tick: Callable[[], float],
    on_timeout: str,
    on_error: str,
    failure_reason: RunFailureReason,
) -> _Call[T]:
    """Run ``call`` behind a per-stage timeout and the overall deadline."""

    remaining = deadline_at - tick()
    if remaining <= 0:
        return _Call(
            None,
            _StageFailure(
                RunFailureReason.DEADLINE_EXCEEDED,
                "The run reached its overall deadline before this stage could start.",
            ),
        )

    budget = min(timeout_seconds, remaining)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="opponent-tendencies-stage")
    future = pool.submit(call)
    try:
        try:
            return _Call(future.result(timeout=budget), None)
        except FutureTimeoutError:
            if future.done():
                exception = future.exception()
                if exception is None:
                    return _Call(future.result(), None)
                return _Call(None, _stage_failure(exception, on_error, failure_reason))
            if tick() + 1e-9 >= deadline_at:
                return _Call(
                    None,
                    _StageFailure(
                        RunFailureReason.DEADLINE_EXCEEDED,
                        "The run reached its overall deadline before this stage finished.",
                    ),
                )
            return _Call(None, _StageFailure(failure_reason, on_timeout))
        except ConfigurationError as exc:
            return _Call(
                None,
                _StageFailure(
                    RunFailureReason.CONFIGURATION_ERROR,
                    f"The LLM is not configured correctly: {_describe(exc)}",
                ),
            )
        except Exception as exc:
            return _Call(None, _stage_failure(exc, on_error, failure_reason))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _stage_failure(
    exc: BaseException,
    prefix: str,
    reason: RunFailureReason,
) -> _StageFailure:
    return _StageFailure(reason, f"{prefix}: {_describe(exc)}")


def _budget_failure(
    log: _RunLog,
    *,
    max_llm_calls: int,
    token_budget: int,
    stage: str,
) -> _StageFailure | None:
    if log.llm_calls > max_llm_calls:
        return _StageFailure(
            RunFailureReason.DEADLINE_EXCEEDED,
            f"The run exceeded its step cap after {stage}: {log.llm_calls} LLM call(s), "
            f"limit {max_llm_calls}.",
        )
    if log.usage.total_tokens > token_budget:
        return _StageFailure(
            RunFailureReason.DEADLINE_EXCEEDED,
            f"The run exceeded its token budget after {stage}: "
            f"{log.usage.total_tokens} token(s), limit {token_budget}.",
        )
    return None


# ---------------------------------------------------------------------------
# Stamping helpers
# ---------------------------------------------------------------------------


def _failed(
    failure: _StageFailure,
    *,
    log: _RunLog,
    started_at: datetime,
    finished_at: datetime,
    deadline_seconds: float,
    latency_seconds: float,
) -> ReportRun:
    status = RunStatus(
        state=RunState.FAILED,
        failure_reason=failure.reason,
        message=failure.message,
        agent_status=dict(log.agent_status),
        current_step="failed",
        progress=0.0,
        started_at=started_at,
        finished_at=finished_at,
        deadline_seconds=deadline_seconds,
    )
    return ReportRun(
        status=status,
        report=None,
        warnings=_dedupe_warnings(log.warnings),
        agent_results=tuple(log.agent_results),
        validation=log.validation,
        usage=log.usage,
        llm_calls=log.llm_calls,
        latency_seconds=latency_seconds,
        synthesis=log.synthesis,
        verification=log.verification,
        verified=log.verified,
    )


def data_revision(season_data: SeasonData, media_data: MediaData) -> str:
    """Content revision covering both stores, used to key report caches."""

    return f"{season_data.revision[:16]}-{media_data.revision[:16]}"


def _data_usage(
    *,
    bundle: StatsBundle,
    games: Sequence[Game],
    evidence: Sequence[EvidenceRef],
    media_data: MediaData,
) -> DataUsage:
    """Count only the data that actually fed the report (requirement 5)."""

    media_ids: set[str] = set()
    quote_ids: set[str] = set()
    for ref in evidence:
        if ref.kind is EvidenceKind.MEDIA and ref.media_id:
            media_ids.add(ref.media_id)
        elif ref.kind is EvidenceKind.QUOTE:
            if ref.quote_id:
                quote_ids.add(ref.quote_id)
                quote = media_data.quote_by_id(ref.quote_id)
                if quote is not None:
                    media_ids.add(quote.media_id)
            elif ref.media_id:
                media_ids.add(ref.media_id)

    source_ids = sorted(
        {
            item.source_id
            for media_id in media_ids
            if (item := media_data.media_by_id(media_id)) is not None
        }
    )
    return DataUsage(
        games=len(games),
        plays=bundle.quality.offensive_plays,
        media_items=len(media_ids),
        quotes=len(quote_ids),
        source_ids=source_ids,
    )


def _merge_weak_evidence(
    *groups: Sequence[WeakEvidence],
) -> tuple[WeakEvidence, ...]:
    by_id: dict[str, WeakEvidence] = {}
    for group in groups:
        for item in group:
            by_id[item.weak_evidence_id] = item
    return tuple(by_id[item_id] for item_id in sorted(by_id))


def _dedupe_warnings(issues: Sequence[ValidationIssue]) -> tuple[ValidationIssue, ...]:
    seen: set[tuple[str, str | None, str | None, str]] = set()
    unique: list[ValidationIssue] = []
    for issue in issues:
        if issue.severity is IssueSeverity.ERROR:  # errors are handled as failures
            continue
        key = (issue.code, issue.subject_kind, issue.subject_id, issue.message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)
    return tuple(unique)


def _notify(
    callback: ProgressCallback | None,
    step: str,
    status: SectionStatus,
    *,
    agent: AgentName | None = None,
) -> None:
    if callback is not None:
        callback(RunProgress(step=step, status=status, agent=agent))


def _warning(
    code: CheckCode,
    message: str,
    *,
    subject_kind: str,
    subject_id: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code.value,
        message=message,
        severity=IssueSeverity.WARNING,
        subject_kind=subject_kind,
        subject_id=subject_id,
        requires_human_review=True,
    )


def _model_name(client: LLMCompleter, synthesis: SynthesisResult) -> str | None:
    if synthesis.model:
        return synthesis.model
    name = getattr(client, "model_name", None)
    if isinstance(name, str) and name.strip():
        return name
    return None


def _errors(result: ValidationResult) -> str:
    messages = [issue.message for issue in result.errors]
    if not messages:
        return "no errors reported"
    shown = "; ".join(messages[:5])
    if len(messages) > 5:
        shown += f"; and {len(messages) - 5} more"
    return shown


def _describe(exc: BaseException | None) -> str:
    if exc is None:
        return "unknown failure"
    text = " ".join(str(exc).split())
    if not text:
        return type(exc).__name__
    described = f"{type(exc).__name__}: {text}"
    return described[:500]
