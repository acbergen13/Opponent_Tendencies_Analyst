"""Tests for the model-facing JSON schemas and wire contracts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from opponent_tendencies.llm.schemas import (
    STRUCTURED_OUTPUT_MODELS,
    AgentResponse,
    ClaimVerdict,
    FollowUpResponse,
    SynthesisResponse,
    VerifiedClaim,
    json_schema_for,
    structured_output_schemas,
    write_json_schemas,
)


def _object_schemas(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            yield node
        for value in node.values():
            yield from _object_schemas(value)
    elif isinstance(node, list):
        for value in node:
            yield from _object_schemas(value)


@pytest.mark.parametrize("name", sorted(STRUCTURED_OUTPUT_MODELS))
def test_wire_schemas_are_strict(name: str) -> None:
    schema = json_schema_for(name)
    objects = list(_object_schemas(schema))

    assert objects
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])


def test_schema_omits_server_owned_identity() -> None:
    for schema in (
        json_schema_for(AgentResponse),
        json_schema_for(SynthesisResponse),
    ):
        properties = set(schema["properties"])
        assert not {"opponent", "games", "weeks", "prompt_version", "report_id"} & properties


def test_agent_response_round_trip() -> None:
    payload = {
        "tendencies": [
            {
                "statement": "Buffalo runs on half of its scrimmage plays.",
                "category": "run_pass",
                "direction": "stable",
                "evidence_ids": ["ev:stat:stt_1:table"],
                "confidence": "high",
                "weeks": [9],
                "metric": "run_rate",
                "value": 0.5,
                "comparison": "weeks 5-9",
            }
        ],
        "observations": [],
        "limitations": [],
    }

    response = AgentResponse.model_validate(payload)

    assert response.tendencies[0].direction.value == "stable"
    assert AgentResponse.model_validate_json(response.model_dump_json()) == response


def test_agent_response_rejects_identity_and_extra_fields() -> None:
    payload = {
        "tendencies": [],
        "observations": [],
        "limitations": [],
        "opponent": "BUF",
    }

    with pytest.raises(ValidationError, match="extra_forbidden"):
        AgentResponse.model_validate(payload)


def test_verified_claim_support_rules() -> None:
    with pytest.raises(ValidationError, match="supported claim must cite evidence"):
        VerifiedClaim(
            claim_id="fnd_1",
            verdict=ClaimVerdict.SUPPORTED,
            reason="The source states it.",
            evidence_ids=[],
            correction=None,
        )

    with pytest.raises(ValidationError, match="must not carry a correction"):
        VerifiedClaim(
            claim_id="fnd_1",
            verdict=ClaimVerdict.SUPPORTED,
            reason="The source states it.",
            evidence_ids=["ev:1"],
            correction="Change it.",
        )

    with pytest.raises(ValidationError, match="must provide a correction"):
        VerifiedClaim(
            claim_id="fnd_1",
            verdict=ClaimVerdict.PARTIALLY_SUPPORTED,
            reason="Only part is supported.",
            evidence_ids=["ev:1"],
            correction=None,
        )

    claim = VerifiedClaim(
        claim_id="fnd_2",
        verdict=ClaimVerdict.UNSUPPORTED,
        reason="No source supports it.",
        evidence_ids=[],
        correction=None,
    )
    assert claim.verdict is ClaimVerdict.UNSUPPORTED


def test_follow_up_requires_evidence_when_supported() -> None:
    with pytest.raises(ValidationError, match="must cite stored evidence"):
        FollowUpResponse(
            answer="Yes.",
            supported=True,
            evidence_ids=[],
            limitations=[],
        )

    response = FollowUpResponse(
        answer="Yes.",
        supported=True,
        evidence_ids=["ev:1"],
        limitations=[],
    )
    assert response.supported is True


def test_structured_output_schemas_share_the_contract_version() -> None:
    schemas = structured_output_schemas()

    assert set(schemas) == set(STRUCTURED_OUTPUT_MODELS)
    assert {schema["$schema"] for schema in schemas.values()} == {
        "https://json-schema.org/draft/2020-12/schema"
    }


def test_write_json_schemas_is_deterministic(tmp_path: Path) -> None:
    first = write_json_schemas(tmp_path / "first")
    second = write_json_schemas(tmp_path / "second")

    assert {path.name for path in first} == {
        f"{name}.schema.json" for name in STRUCTURED_OUTPUT_MODELS
    }
    for first_path, second_path in zip(first, second, strict=True):
        assert first_path.read_text(encoding="utf-8") == second_path.read_text(encoding="utf-8")
        payload = json.loads(first_path.read_text(encoding="utf-8"))
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_unknown_schema_name_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown structured output"):
        json_schema_for("not_a_model")
