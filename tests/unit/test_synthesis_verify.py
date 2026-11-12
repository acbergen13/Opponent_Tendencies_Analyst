"""Stage 8 synthesis and verification: deterministic merging and verdicts.

The synthesis model may merge and phrase; Python enforces citations, specialist
agents, conflict coverage, weak evidence, and unavailable sections. Verification
is a separate call whose verdicts Python applies by removing or correcting
claims, never by trusting a rewritten report.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from opponent_tendencies.agents.synthesis import SYNTHESIS_MAX_TOKENS, run_synthesis_agent
from opponent_tendencies.agents.verify import (
    apply_verification,
    build_verifier_claims,
    run_verifier,
)
from opponent_tendencies.data.media_store import MediaData
from opponent_tendencies.data.play_store import SeasonData
from opponent_tendencies.models import (
    AgentName,
    Observation,
    ObservationCategory,
    SectionStatus,
)
from opponent_tendencies.validation.checks import CheckCode
from tests.agent_fixtures import evidence_ids, json_completion, make_client
from tests.unit.factories import (
    WEEK,
    make_bra_output,
    make_cqa_output,
    make_evidence,
    make_game,
    make_media,
    make_play,
    make_quote,
    make_quote_media,
    make_sta_output,
)


def _synthesis_payload(body: dict[str, Any], *, evidence: list[str] | None = None) -> dict:
    ids = evidence or evidence_ids(body)
    return {
        "overview": "Buffalo leans on early-down runs and adjusts on third down.",
        "tendencies": [
            {
                "statement": "Buffalo runs on early downs.",
                "category": "run_pass",
                "direction": "stable",
                "evidence_ids": [ids[0]],
                "confidence": "high",
                "weeks": [WEEK],
                "metric": "run_rate",
                "value": 0.5,
                "comparison": "weeks 5-9",
            }
        ],
        "observations": [
            {
                "statement": "Reporters note condensed formations on third down.",
                "category": "scheme",
                "evidence_ids": [ids[0]],
                "confidence": "medium",
                "weeks": [WEEK],
                "treat_as_context": False,
            }
        ],
        "agreements": [
            {
                "topic": "Early-down identity",
                "statement": "The table and reporting agree on an early-down run emphasis.",
                "agents": ["STA", "BRA"],
                "evidence_ids": [ids[0]],
                "weeks": [WEEK],
            }
        ],
        "conflicts": [],
        "gaps": [],
        "weak_evidence": [],
    }


def _outputs(*, bra_status: SectionStatus = SectionStatus.COMPLETED) -> tuple[list[Any], Any]:
    sta, _, _, _, table = make_sta_output()
    bra, _ = make_bra_output(status=bra_status)
    cqa, _ = make_cqa_output()
    return [sta, bra, cqa], table


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def test_synthesis_merges_validated_findings_and_stamps_ids() -> None:
    outputs, table = _outputs()

    # The handler receives the prompt only after rendering, so build the payload
    # from the request body instead of a fixed document.
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json=json_completion(_synthesis_payload(body)))

    client, requests = make_client(handler)
    result = run_synthesis_agent(
        client=client,
        opponent=make_game().away_team,
        season=2026,
        weeks=[WEEK],
        games=[make_game()],
        agent_outputs=outputs,
        stat_tables=[table],
    )

    assert requests
    prompt = requests[0]["messages"][1]["content"]
    assert "<agent_findings" in prompt
    assert "<deterministic_statistics>" in prompt
    assert requests[0]["max_tokens"] == SYNTHESIS_MAX_TOKENS
    assert requests[0]["stream"] is True
    assert result.overview
    assert result.tendencies
    assert result.agreements
    assert result.evidence == tuple(sorted(result.evidence, key=lambda item: item.evidence_id))
    assert set(result.agreements[0].agents) == {
        AgentName.STATISTICS,
        AgentName.BEAT_REPORTER,
    }
    assert result.dropped == 0
    assert result.llm_calls == 1


def test_synthesis_drops_invalid_agreements_and_reports_it() -> None:
    outputs, table = _outputs()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payload = _synthesis_payload(body)
        payload["agreements"] = [
            {
                "topic": "Bogus",
                "statement": "Synthesis agrees with itself.",
                "agents": ["SYN", "SYN"],
                "evidence_ids": [evidence_ids(body)[0]],
                "weeks": [WEEK],
            }
        ]
        return httpx.Response(200, json=json_completion(payload))

    client, _ = make_client(handler)
    result = run_synthesis_agent(
        client=client,
        opponent=make_game().away_team,
        season=2026,
        weeks=[WEEK],
        games=[make_game()],
        agent_outputs=outputs,
        stat_tables=[table],
    )
    assert result.agreements == ()
    assert result.dropped == 1
    assert any(
        issue.code == CheckCode.SYNTHESIS_UNGROUNDED_CLAIM.value for issue in result.warnings
    )


def test_synthesis_conflict_positions_must_cite_evidence() -> None:
    outputs, table = _outputs()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payload = _synthesis_payload(body)
        payload["conflicts"] = [
            {
                "topic": "Conflicting emphasis",
                "description": "One agent sees runs, another sees passes.",
                "positions": [
                    {
                        "agent": "STA",
                        "statement": "Runs.",
                        "evidence_ids": ["ev:stat:invented"],
                    },
                    {
                        "agent": "BRA",
                        "statement": "Passes.",
                        "evidence_ids": ["ev:media:invented"],
                    },
                ],
                "evidence_ids": ["ev:stat:invented"],
                "weeks": [WEEK],
                "resolution": None,
            }
        ]
        return httpx.Response(200, json=json_completion(payload))

    client, _ = make_client(handler)
    result = run_synthesis_agent(
        client=client,
        opponent=make_game().away_team,
        season=2026,
        weeks=[WEEK],
        games=[make_game()],
        agent_outputs=outputs,
        stat_tables=[table],
    )
    assert result.conflicts == ()
    assert result.dropped >= 1


def test_synthesis_adds_an_explicit_gap_for_an_unavailable_agent() -> None:
    outputs, table = _outputs(bra_status=SectionStatus.UNAVAILABLE)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payload = _synthesis_payload(body)
        payload["gaps"] = []
        return httpx.Response(200, json=json_completion(payload))

    client, _ = make_client(handler)
    result = run_synthesis_agent(
        client=client,
        opponent=make_game().away_team,
        season=2026,
        weeks=[WEEK],
        games=[make_game()],
        agent_outputs=outputs,
        stat_tables=[table],
    )
    gaps = [gap for gap in result.gaps if AgentName.BEAT_REPORTER in gap.agents]
    assert gaps
    assert gaps[0].reason.value == "agent_unavailable"


def test_synthesis_flags_low_confidence_findings_for_human_review() -> None:
    outputs, table = _outputs()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payload = _synthesis_payload(body)
        payload["tendencies"][0]["confidence"] = "low"
        return httpx.Response(200, json=json_completion(payload))

    client, _ = make_client(handler)
    result = run_synthesis_agent(
        client=client,
        opponent=make_game().away_team,
        season=2026,
        weeks=[WEEK],
        games=[make_game()],
        agent_outputs=outputs,
        stat_tables=[table],
    )
    assert result.tendencies[0].weak_evidence
    assert result.weak_evidence
    assert any(item.reason.value == "low_confidence" for item in result.weak_evidence)


def test_synthesis_rejects_wrong_agent_set() -> None:
    outputs, table = _outputs()
    client, _ = make_client(lambda request: httpx.Response(500))
    with pytest.raises(ValueError, match="STA, BRA, and CQA"):
        run_synthesis_agent(
            client=client,
            opponent=make_game().away_team,
            season=2026,
            weeks=[WEEK],
            games=[make_game()],
            agent_outputs=outputs[:2],
            stat_tables=[table],
        )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _verification_data() -> tuple[SeasonData, MediaData, list, Any]:
    game = make_game()
    play = make_play()
    media = make_media()
    quote_media = make_quote_media()
    quote = make_quote(quote_media)
    season = SeasonData(season=2026, games=(game,), plays=(play,))
    media_data = MediaData(season=2026, media=(media, quote_media), quotes=(quote,))
    _, _, _, stat_ref = make_evidence()
    return season, media_data, [make_sta_output()[0].stat_tables[0]], stat_ref


def test_build_verifier_claims_resolves_exact_snippets() -> None:
    season, media, tables, _ = _verification_data()
    evidence = list(make_evidence())
    sta, tendency, _, _, _ = make_sta_output()

    _, bra_observation = make_bra_output()
    _, cqa_observation = make_cqa_output()
    claims = build_verifier_claims(
        [tendency, bra_observation, cqa_observation],
        evidence,
        season_data=season,
        media_data=media,
        stat_tables=tables,
    )
    tendency_claim, bra_claim, cqa_claim = claims
    assert tendency_claim.claim_id == tendency.finding_id
    play_ref, media_ref, quote_ref, stat_ref = evidence
    tendency_texts = {snippet.evidence_id: snippet.text for snippet in tendency_claim.snippets}
    assert "2026_09_BUF_KC play 42" in tendency_texts[play_ref.evidence_id]
    assert "run_rate" in tendency_texts[stat_ref.evidence_id]
    media_texts = {snippet.evidence_id: snippet.text for snippet in bra_claim.snippets}
    assert media_texts[media_ref.evidence_id] == media_ref.snippet
    quote_texts = {snippet.evidence_id: snippet.text for snippet in cqa_claim.snippets}
    assert quote_texts[quote_ref.evidence_id] == quote_ref.snippet
    assert sta.agent is AgentName.STATISTICS


def _claims_and_verdict(
    *,
    verdict: str,
    correction: str | None = None,
    evidence_override: list[str] | None = None,
) -> tuple[SeasonData, MediaData, list, Any, Any]:
    season, media, tables, _ = _verification_data()
    evidence = list(make_evidence())
    _, tendency, _, _, _ = make_sta_output()
    claim = build_verifier_claims(
        [tendency],
        evidence,
        season_data=season,
        media_data=media,
        stat_tables=tables,
    )[0]
    snippet_id = claim.snippets[0].evidence_id

    document = {
        "claims": [
            {
                "claim_id": claim.claim_id,
                "verdict": verdict,
                "reason": "Checked against the attached snippets.",
                "evidence_ids": evidence_override
                if evidence_override is not None
                else ([snippet_id] if verdict == "supported" else []),
                "correction": correction,
            }
        ]
    }
    return season, media, tables, claim, document


def _run(document: dict[str, Any], claim: Any) -> Any:
    client, _ = make_client(lambda request: httpx.Response(200, json=json_completion(document)))
    return run_verifier(
        client=client,
        opponent=make_game().away_team,
        weeks=[WEEK],
        claims=[claim],
    )


def test_supported_claim_is_kept() -> None:
    _, _, _, claim, document = _claims_and_verdict(verdict="supported")
    _, tendency, _, _, _ = make_sta_output()
    result = _run(document, claim)
    applied = apply_verification(
        tendencies=[tendency],
        observations=[],
        claims=[claim],
        verdicts=result.by_claim_id(),
        weeks=[WEEK],
    )
    assert applied.tendencies == (tendency,)
    assert applied.removed == 0
    assert applied.warnings == ()


def test_unsupported_claim_is_removed_and_flagged() -> None:
    _, _, _, claim, document = _claims_and_verdict(verdict="unsupported")
    _, tendency, _, _, _ = make_sta_output()
    result = _run(document, claim)
    applied = apply_verification(
        tendencies=[tendency],
        observations=[],
        claims=[claim],
        verdicts=result.by_claim_id(),
        weeks=[WEEK],
    )
    assert applied.tendencies == ()
    assert applied.removed == 1
    assert applied.weak_evidence
    assert any(
        issue.code == CheckCode.VERIFICATION_REMOVED_CLAIM.value for issue in applied.warnings
    )


def test_partially_supported_claim_is_corrected() -> None:
    _, _, _, claim, document = _claims_and_verdict(
        verdict="partially_supported",
        correction="Buffalo runs on 45 percent of early downs.",
    )
    _, tendency, _, _, _ = make_sta_output()
    result = _run(document, claim)
    applied = apply_verification(
        tendencies=[tendency],
        observations=[],
        claims=[claim],
        verdicts=result.by_claim_id(),
        weeks=[WEEK],
    )
    assert applied.corrected == 1
    assert applied.tendencies[0].statement == "Buffalo runs on 45 percent of early downs."
    assert applied.tendencies[0].finding_id != tendency.finding_id
    assert any(
        issue.code == CheckCode.VERIFICATION_CORRECTED_CLAIM.value for issue in applied.warnings
    )


def test_missing_verdict_removes_the_claim() -> None:
    _, _, _, claim, _document = _claims_and_verdict(verdict="supported")
    _, tendency, _, _, _ = make_sta_output()
    applied = apply_verification(
        tendencies=[tendency],
        observations=[],
        claims=[claim],
        verdicts={},
        weeks=[WEEK],
    )
    assert applied.tendencies == ()
    assert any(
        issue.code == CheckCode.VERIFICATION_MISSING_VERDICT.value for issue in applied.warnings
    )


def test_supported_verdict_citing_external_evidence_is_removed() -> None:
    _, _, _, claim, document = _claims_and_verdict(
        verdict="supported",
        evidence_override=["ev:stat:not-attached"],
    )
    _, tendency, _, _, _ = make_sta_output()
    result = _run(document, claim)
    applied = apply_verification(
        tendencies=[tendency],
        observations=[],
        claims=[claim],
        verdicts=result.by_claim_id(),
        weeks=[WEEK],
    )
    assert applied.tendencies == ()
    assert applied.removed == 1


def test_verifier_skips_the_model_when_there_are_no_claims() -> None:
    client, requests = make_client(
        lambda request: (_ for _ in ()).throw(AssertionError("must not call the model"))
    )
    result = run_verifier(
        client=client,
        opponent=make_game().away_team,
        weeks=[WEEK],
        claims=[],
    )
    assert requests == []
    assert result.llm_calls == 0
    assert result.claims == ()


def test_apply_verification_keeps_observations_and_tendencies_separate() -> None:
    _, _, _, claim, document = _claims_and_verdict(verdict="supported")
    _, tendency, _, _, _ = make_sta_output()
    observation = Observation(
        statement="A reporter noted the same early-down emphasis.",
        category=ObservationCategory.SCHEME,
        evidence_ids=list(tendency.evidence_ids),
        weeks=[WEEK],
    )
    result = _run(document, claim)
    applied = apply_verification(
        tendencies=[tendency],
        observations=[observation],
        claims=[claim],
        verdicts=result.by_claim_id(),
        weeks=[WEEK],
    )
    assert applied.tendencies == (tendency,)
    assert applied.observations == ()
    assert applied.removed == 1
