"""Small, valid contract fixtures shared by unit tests."""

from __future__ import annotations

from datetime import UTC, date, datetime

from pydantic import HttpUrl

from opponent_tendencies.models import (
    AgentName,
    AgentOutput,
    Agreement,
    Confidence,
    DataUsage,
    EvidenceRef,
    Game,
    Gap,
    GapReason,
    MediaItem,
    MediaType,
    Observation,
    ObservationCategory,
    Play,
    PlayType,
    Quote,
    Report,
    RunState,
    RunStatus,
    SectionStatus,
    SpeakerRole,
    SplitDimension,
    StatMetric,
    StatRow,
    StatTable,
    StatUnit,
    Team,
    Tendency,
    TendencyCategory,
    TendencyDirection,
    WeakEvidence,
    WeakEvidenceReason,
)

SEASON = 2026
WEEK = 9
GAME_ID = "2026_09_BUF_KC"
PLAY_ID = 42
NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
GAME_DATE = date(2026, 11, 8)


def make_game() -> Game:
    return Game(
        game_id=GAME_ID,
        season=SEASON,
        week=WEEK,
        game_date=GAME_DATE,
        home_team=Team.KC,
        away_team=Team.BUF,
        home_score=24,
        away_score=20,
    )


def make_play() -> Play:
    return Play(
        game_id=GAME_ID,
        play_id=PLAY_ID,
        season=SEASON,
        week=WEEK,
        posteam=Team.BUF,
        defteam=Team.KC,
        down=1,
        yards_to_go=10,
        yardline_100=75,
        play_type=PlayType.RUN,
        yards_gained=4,
        first_down=False,
    )


def make_media(body: str = "Buffalo leans on outside-zone runs on early downs.") -> MediaItem:
    return MediaItem(
        source_id="kansas-city-star",
        source_type=MediaType.ARTICLE,
        title="Film breakdown: Buffalo's early-down plan",
        author="A. Reporter",
        publisher="Kansas City Star",
        url=HttpUrl("https://example.com/film-breakdown"),
        published_on=date(2026, 11, 9),
        season=SEASON,
        week=WEEK,
        game_id=GAME_ID,
        opponent=Team.BUF,
        body=body,
        retrieved_at=NOW,
    )


QUOTE_TEXT = "We have to run the ball better on first down."


def make_quote_media() -> MediaItem:
    return MediaItem(
        source_id="buffalo-news",
        source_type=MediaType.COACH_QUOTE,
        title="McDermott post-game press conference",
        publisher="Buffalo News",
        url=HttpUrl("https://example.com/mcdermott-presser"),
        published_on=date(2026, 11, 8),
        season=SEASON,
        week=WEEK,
        game_id=GAME_ID,
        opponent=Team.BUF,
        body=f"{QUOTE_TEXT} We will keep working at it.",
        retrieved_at=NOW,
    )


def make_quote(media: MediaItem) -> Quote:
    return Quote(
        media_id=media.media_id,
        speaker="Sean McDermott",
        speaker_role=SpeakerRole.HEAD_COACH,
        team=Team.BUF,
        season=SEASON,
        week=WEEK,
        game_id=GAME_ID,
        spoken_on=GAME_DATE,
        publisher=media.publisher,
        url=media.url,
        text=QUOTE_TEXT,
        char_start=0,
        char_end=len(QUOTE_TEXT),
    )


def make_stat_table() -> StatTable:
    return StatTable(
        metric=StatMetric.RUN_RATE,
        title="Run rate on all scrimmage plays",
        dimension=SplitDimension.OVERALL,
        rows=[
            StatRow(
                label="All plays",
                value=0.5,
                n=10,
                numerator=5,
                denominator=10,
                unit=StatUnit.RATE,
            )
        ],
        n_plays=10,
        game_ids=[GAME_ID],
        weeks=[WEEK],
        min_sample=5,
    )


def make_evidence() -> tuple[EvidenceRef, EvidenceRef, EvidenceRef, EvidenceRef]:
    play = EvidenceRef.for_play(
        GAME_ID,
        PLAY_ID,
        label="1st & 10 run",
        snippet="4-yard run",
    )
    media = make_media()
    media_ref = EvidenceRef.for_media(
        media.media_id,
        label="Film breakdown",
        chunk_id="chunk-1",
        snippet="outside-zone runs on early downs",
        source_url=media.url,
    )
    quote_media = make_quote_media()
    quote = make_quote(quote_media)
    quote_ref = EvidenceRef.for_quote(
        quote.quote_id,
        label="Post-game quote",
        media_id=quote.media_id,
        snippet=quote.text,
        source_url=quote.url,
    )
    stat_ref = make_stat_table().evidence("All plays")
    return play, media_ref, quote_ref, stat_ref


def make_sta_output(
    *,
    status: SectionStatus = SectionStatus.COMPLETED,
) -> tuple[AgentOutput, Tendency, EvidenceRef, EvidenceRef, StatTable]:
    game = make_game()
    table = make_stat_table()
    play_ev, _, _, stat_ev = make_evidence()
    tendency = Tendency(
        statement="Buffalo runs on half of its scrimmage plays.",
        category=TendencyCategory.RUN_PASS,
        direction=TendencyDirection.STABLE,
        evidence_ids=[play_ev.evidence_id, stat_ev.evidence_id],
        weeks=[WEEK],
        confidence=Confidence.HIGH,
        metric="run_rate",
        value=0.5,
        comparison="weeks 5-9",
    )
    output = AgentOutput(
        agent=AgentName.STATISTICS,
        opponent=Team.BUF,
        games=[game],
        weeks=[WEEK],
        tendencies=[tendency],
        observations=[],
        evidence=sorted([play_ev, stat_ev], key=lambda item: item.evidence_id),
        limitations=[],
        stat_tables=[table],
        status=status,
        prompt_version="test-v1",
        model="vt-arc-llm",
    )
    return output, tendency, play_ev, stat_ev, table


def media_evidence() -> EvidenceRef:
    return make_evidence()[1]


def quote_evidence() -> EvidenceRef:
    return make_evidence()[2]


def make_bra_output(
    *,
    status: SectionStatus = SectionStatus.COMPLETED,
) -> tuple[AgentOutput, Observation]:
    game = make_game()
    media_ref = media_evidence()
    observation = Observation(
        statement="Analysts expect Buffalo to keep leaning on early-down play action.",
        category=ObservationCategory.SCHEME,
        evidence_ids=[media_ref.evidence_id],
        weeks=[WEEK],
        confidence=Confidence.MEDIUM,
    )
    limitations = (
        [] if status is SectionStatus.COMPLETED else ["Beat-reporter sources unavailable."]
    )
    output = AgentOutput(
        agent=AgentName.BEAT_REPORTER,
        opponent=Team.BUF,
        games=[game],
        weeks=[WEEK],
        tendencies=[],
        observations=[] if status is not SectionStatus.COMPLETED else [observation],
        evidence=[] if status is not SectionStatus.COMPLETED else [media_ref],
        limitations=limitations,
        stat_tables=[],
        status=status,
        prompt_version="test-v1",
        model="vt-arc-llm",
    )
    return output, observation


def make_cqa_output(
    *,
    status: SectionStatus = SectionStatus.COMPLETED,
) -> tuple[AgentOutput, Observation]:
    game = make_game()
    quote_ref = quote_evidence()
    observation = Observation(
        statement="McDermott said Buffalo must run the ball better on first down.",
        category=ObservationCategory.COACH_EXPLANATION,
        evidence_ids=[quote_ref.evidence_id],
        weeks=[WEEK],
        confidence=Confidence.MEDIUM,
        treat_as_context=True,
    )
    limitations = [] if status is SectionStatus.COMPLETED else ["Coach quotes unavailable."]
    output = AgentOutput(
        agent=AgentName.COACH_QUOTE,
        opponent=Team.BUF,
        games=[game],
        weeks=[WEEK],
        tendencies=[],
        observations=[] if status is not SectionStatus.COMPLETED else [observation],
        evidence=[] if status is not SectionStatus.COMPLETED else [quote_ref],
        limitations=limitations,
        stat_tables=[],
        status=status,
        prompt_version="test-v1",
        model="vt-arc-llm",
    )
    return output, observation


def make_report(*, degraded: bool = False) -> Report:
    game = make_game()
    sta, tendency, _, stat_ev, table = make_sta_output()
    bra, bra_observation = make_bra_output()
    cqa, cqa_observation = make_cqa_output(
        status=SectionStatus.UNAVAILABLE if degraded else SectionStatus.COMPLETED
    )
    _, media_ref, _, _ = make_evidence()

    observations = [bra_observation]
    if not degraded:
        observations.append(cqa_observation)

    agreement = Agreement(
        topic="Early-down identity",
        statement="Both the play-by-play data and reporting describe an early-down run emphasis.",
        agents=[AgentName.STATISTICS, AgentName.BEAT_REPORTER],
        evidence_ids=sorted([stat_ev.evidence_id, media_ref.evidence_id]),
        weeks=[WEEK],
    )
    gap = Gap(
        topic="Personnel detail",
        description="No source described which personnel grouping drives the run tendency.",
        reason=GapReason.OTHER,
        agents=[AgentName.BEAT_REPORTER],
        weeks=[WEEK],
    )
    weak = WeakEvidence(
        topic="Small single-game sample",
        description="Only one game was selected, so the split should be reviewed by a human.",
        reason=WeakEvidenceReason.SMALL_SAMPLE,
        evidence_ids=[stat_ev.evidence_id],
        weeks=[WEEK],
        agent=AgentName.STATISTICS,
    )

    referenced: set[str] = set()
    for finding in [tendency, *observations]:
        referenced.update(finding.evidence_ids)
    referenced.update(agreement.evidence_ids)
    referenced.update(weak.evidence_ids)
    for output in [sta, bra, cqa]:
        referenced.update(item.evidence_id for item in output.evidence)
    by_id = {item.evidence_id: item for item in [*sta.evidence, *bra.evidence, *cqa.evidence]}
    report_evidence = [by_id[evidence_id] for evidence_id in sorted(referenced)]

    agent_status = {
        AgentName.STATISTICS: SectionStatus.COMPLETED,
        AgentName.BEAT_REPORTER: SectionStatus.COMPLETED,
        AgentName.COACH_QUOTE: (SectionStatus.UNAVAILABLE if degraded else SectionStatus.COMPLETED),
        AgentName.SYNTHESIS: SectionStatus.COMPLETED,
    }
    run_status = RunStatus(
        state=RunState.DEGRADED if degraded else RunState.COMPLETED,
        agent_status=agent_status,
        progress=1.0,
        current_step="complete",
        started_at=datetime(2026, 11, 10, 11, 55, tzinfo=UTC),
        finished_at=NOW,
        deadline_seconds=180.0,
    )

    return Report(
        opponent=Team.BUF,
        season=SEASON,
        weeks=[WEEK],
        games=[game],
        generated_at=NOW,
        data_revision="test-data-r1",
        prompt_version="test-v1",
        model="vt-arc-llm",
        status=run_status,
        overview="Buffalo shows an early-down run emphasis, though the sample is small.",
        data_used=DataUsage(
            games=1,
            plays=10,
            media_items=1 if degraded else 2,
            quotes=0 if degraded else 1,
            source_ids=(
                ["buffalo-news", "kansas-city-star"] if not degraded else ["kansas-city-star"]
            ),
        ),
        stat_tables=[table],
        tendencies=[tendency],
        observations=observations,
        agreements=[agreement],
        conflicts=[],
        gaps=[gap],
        weak_evidence=[weak],
        warnings=[],
        evidence=report_evidence,
        agent_outputs=[sta, bra, cqa],
    )
