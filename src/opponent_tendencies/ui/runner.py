"""Side-effect-free helpers the Streamlit app uses to build a report.

The UI never computes statistics, filters sources, or checks citations itself.
It loads the stored data, constructs the pinned embedder/client, and calls
:func:`opponent_tendencies.orchestrator.run_report`. Everything in this module
is injectable so Streamlit AppTest runs stay offline:

* ``build_report`` accepts pre-loaded data, an index, and an LLM client.
* ``follow_up`` accepts any object with ``complete_structured``.

Production callers use the small default factories at the bottom; tests inject
fixtures and an ``httpx.MockTransport`` client.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from opponent_tendencies.agents.common import LLMCompleter
from opponent_tendencies.cache import ReportCache, cache_key
from opponent_tendencies.config import Settings, get_settings
from opponent_tendencies.data.index import MediaIndex
from opponent_tendencies.data.media_store import MediaData, MediaStore
from opponent_tendencies.data.play_store import SeasonData, SeasonStore
from opponent_tendencies.llm.client import ArcLLMClient, TokenUsage
from opponent_tendencies.llm.embeddings import ArcEmbedder
from opponent_tendencies.llm.prompts import (
    PROMPT_VERSION,
    get_prompt,
    render_report_context,
)
from opponent_tendencies.llm.schemas import FollowUpResponse
from opponent_tendencies.models import Report, Team
from opponent_tendencies.orchestrator import (
    DEFAULT_AGENT_TIMEOUT_SECONDS,
    DEFAULT_DEADLINE_SECONDS,
    DEFAULT_MAX_LLM_CALLS,
    DEFAULT_TOKEN_BUDGET,
    ReportRun,
    RunProgress,
    data_revision,
    run_report,
)
from opponent_tendencies.stats.engine import DEFAULT_MIN_SAMPLE

__all__ = [
    "DEFAULT_SEASON",
    "FollowUpAnswer",
    "FollowUpTurn",
    "answer_follow_up",
    "build_report",
    "create_index",
    "load_media",
    "load_season",
    "summarize_conversation",
]

DEFAULT_SEASON = 2026

# Keep the follow-up prompt bounded no matter how long the conversation gets.
MAX_FULL_TURNS = 3
MAX_SUMMARY_CHARS = 2000


def load_season(season: int = DEFAULT_SEASON, data_dir: Path | None = None) -> SeasonData:
    """Load one season's stored games/plays without requiring LLM credentials."""

    return SeasonStore(season, _season_path(season, data_dir)).load()


def load_media(season: int = DEFAULT_SEASON, data_dir: Path | None = None) -> MediaData:
    """Load one season's stored media/quotes without requiring LLM credentials."""

    return MediaStore(season, _media_path(season, data_dir)).load()


def create_index(
    season: int = DEFAULT_SEASON,
    *,
    chroma_dir: Path | None = None,
    settings: Settings | None = None,
) -> MediaIndex:
    """Return the season's persistent media index using the pinned embedder."""

    resolved = settings if settings is not None else get_settings()
    embedder = ArcEmbedder(
        resolved.vt_arc_base_url,
        resolved.vt_arc_api_key.get_secret_value(),
        resolved.embedding_model,
    )
    directory = chroma_dir if chroma_dir is not None else resolved.chroma_dir
    return MediaIndex(directory, embedder, collection_name=f"media_{int(season)}")


def build_report(
    *,
    opponent: Team,
    weeks: Sequence[int],
    season: int = DEFAULT_SEASON,
    data_dir: Path | None = None,
    chroma_dir: Path | None = None,
    cache_dir: Path | None = None,
    cache: ReportCache | None = None,
    season_data: SeasonData | None = None,
    media_data: MediaData | None = None,
    index: MediaIndex | None = None,
    client: LLMCompleter | None = None,
    on_progress: Callable[[RunProgress], None] | None = None,
    prompt_version: str = PROMPT_VERSION,
    min_sample: int = DEFAULT_MIN_SAMPLE,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
    agent_timeout_seconds: float = DEFAULT_AGENT_TIMEOUT_SECONDS,
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    use_cache: bool = True,
) -> ReportRun:
    """Load what was not injected and run the Stage 9 orchestrator.

    A finished report is reused from :class:`~opponent_tendencies.cache.ReportCache`
    only when the opponent, weeks, prompt version, and combined data revision
    match. Errors from missing configuration propagate to the caller (the app
    renders them as a run failure); deterministic validation failures come back
    as a :class:`ReportRun` with ``report is None``. ``use_cache=False`` forces a
    fresh run for CLI/evaluation callers without deleting stored entries.
    """

    state = season_data if season_data is not None else load_season(season, data_dir)
    media = media_data if media_data is not None else load_media(season, data_dir)
    settings: Settings | None = None
    if client is None:
        settings = get_settings()
        llm: LLMCompleter = ArcLLMClient.from_settings(settings)
    else:
        llm = client
    media_index = index
    if media_index is None and media.media and settings is not None:
        media_index = create_index(season, chroma_dir=chroma_dir, settings=settings)

    revision = data_revision(state, media)
    report_cache = cache if use_cache else None
    if report_cache is None and use_cache:
        resolved_cache_dir = cache_dir
        if resolved_cache_dir is None and data_dir is not None:
            resolved_cache_dir = Path(data_dir) / "reports"
        elif resolved_cache_dir is None and settings is not None:
            resolved_cache_dir = settings.data_dir / "reports"
        if resolved_cache_dir is not None:
            report_cache = ReportCache(resolved_cache_dir)

    if report_cache is not None:
        key = cache_key(opponent, weeks, prompt_version, revision)
        cached = report_cache.load(key)
        if cached is not None:
            return ReportRun(status=cached.status, report=cached, cached=True)

    run = run_report(
        opponent=opponent,
        weeks=weeks,
        season_data=state,
        media_data=media,
        index=media_index,
        client=llm,
        prompt_version=prompt_version,
        min_sample=min_sample,
        deadline_seconds=deadline_seconds,
        agent_timeout_seconds=agent_timeout_seconds,
        max_llm_calls=max_llm_calls,
        token_budget=token_budget,
        on_progress=on_progress,
    )
    if run.report is not None and report_cache is not None:
        report_cache.store(run.report)
    return run


@dataclass(frozen=True)
class FollowUpTurn:
    """One stored question/answer pair; the UI keeps only this much state."""

    question: str
    answer: str
    supported: bool
    evidence_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class FollowUpAnswer:
    """A validated follow-up answer over an already-built report."""

    answer: str
    supported: bool
    evidence_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    usage: TokenUsage = field(default_factory=TokenUsage)
    llm_calls: int = 0
    latency_seconds: float = 0.0
    prompt_version: str = PROMPT_VERSION


def answer_follow_up(
    *,
    client: LLMCompleter,
    report: Report,
    question: str,
    turns: Sequence[FollowUpTurn] = (),
) -> FollowUpAnswer:
    """Answer a follow-up from the stored report and validate its citations.

    The model may only cite IDs already on ``report.evidence``. Cited IDs that
    are unknown are removed, ``supported`` is forced to ``False``, and a
    deterministic limitation explains what happened. No new data is loaded and
    no agent is re-run.
    """

    question = question.strip()
    if not question:
        raise ValueError("question must not be blank")

    prompt = get_prompt("follow_up")
    rendered = prompt.render(
        opponent=report.opponent.value,
        season=str(report.season),
        weeks=", ".join(str(week) for week in report.weeks),
        report_context=render_report_context(report),
        conversation_summary=summarize_conversation(turns),
        question=question,
    )
    result = client.complete_structured(rendered.messages, "follow_up_response")
    response = cast(FollowUpResponse, result.parsed)

    known = {item.evidence_id for item in report.evidence}
    cited = tuple(sorted({evidence_id for evidence_id in response.evidence_ids if evidence_id}))
    unknown = tuple(sorted(set(cited) - known))
    valid = tuple(item for item in cited if item in known)

    supported = bool(response.supported and valid)
    limitations = list(response.limitations)
    if unknown:
        limitations.append(
            "The answer cited evidence that is not present in the stored report: "
            + ", ".join(unknown)
        )
    if response.supported and not valid:
        limitations.append(
            "The answer could not be tied to stored report evidence and is shown as unsupported."
        )

    return FollowUpAnswer(
        answer=response.answer,
        supported=supported,
        evidence_ids=valid,
        limitations=tuple(_dedupe(limitations)),
        usage=result.usage,
        llm_calls=result.llm_calls,
        latency_seconds=result.latency_seconds,
        prompt_version=rendered.version,
    )


def summarize_conversation(
    turns: Sequence[FollowUpTurn],
    *,
    max_full_turns: int = MAX_FULL_TURNS,
    max_chars: int = MAX_SUMMARY_CHARS,
) -> str:
    """Render a deterministic, bounded summary of earlier questions.

    The most recent ``max_full_turns`` are kept in full; older turns collapse
    to their questions so context stays bounded without another model call.
    """

    if not turns:
        return "(no earlier questions)"
    if max_full_turns < 1:
        raise ValueError("max_full_turns must be at least 1")
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    older = turns[:-max_full_turns]
    recent = turns[-max_full_turns:]
    lines: list[str] = []
    if older:
        topics = "; ".join(_clip(turn.question, 80) for turn in older)
        lines.append(f"Earlier questions ({len(older)}): {topics}")
    for index, turn in enumerate(recent, start=1):
        lines.append(f"Turn {index} question: {_clip(turn.question, 240)}")
        lines.append(f"Turn {index} answer: {_clip(turn.answer, 480)}")
        lines.append(
            f"Turn {index} supported: {'yes' if turn.supported else 'no'}"
            + (f" (evidence: {', '.join(turn.evidence_ids)})" if turn.evidence_ids else "")
        )
        for limitation in turn.limitations:
            lines.append(f"Turn {index} limitation: {_clip(limitation, 240)}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + "\n[...truncated]"


def _season_path(season: int, data_dir: Path | None) -> Path | None:
    if data_dir is None:
        return None
    return Path(data_dir) / "seasons" / f"{int(season)}.json"


def _media_path(season: int, data_dir: Path | None) -> Path | None:
    if data_dir is None:
        return None
    return Path(data_dir) / "media" / f"{int(season)}.json"


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
