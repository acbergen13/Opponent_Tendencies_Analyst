"""Stage 6 LLM client: mocked contract tests.

Every test runs against ``httpx.MockTransport``; nothing touches VT ARC. The
tests pin the OpenAI-compatible request shape, bounded retries, the per-call
timeout policy, latency/token logging without secrets, and the single
bounded repair attempt for structured outputs.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

from opponent_tendencies.config import Settings
from opponent_tendencies.llm.client import (
    ArcLLMClient,
    LLMConnectionError,
    LLMMessage,
    LLMResponseError,
    LLMTimeoutError,
    StructuredOutputError,
    TokenUsage,
)
from opponent_tendencies.llm.schemas import AgentResponse, WireModel

API_KEY = "secret-test-key"
BASE_URL = "http://vt-arc.test/v1"

VALID_AGENT_PAYLOAD: dict[str, Any] = {
    "tendencies": [],
    "observations": [],
    "limitations": [],
}


def completion(
    content: str,
    *,
    model: str = "test-model",
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
    with_usage: bool = True,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "cmpl-1",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    if with_usage:
        body["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    return body


def make_client(
    handler: Any,
    **kwargs: Any,
) -> tuple[ArcLLMClient, list[dict[str, Any]]]:
    """Build a client whose endpoint is ``handler``; recorded bodies returned."""

    requests: list[dict[str, Any]] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        response = handler(request)
        assert isinstance(response, httpx.Response)
        return response

    client = ArcLLMClient(
        BASE_URL,
        API_KEY,
        "test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(recording_handler)),
        retry_backoff_seconds=0.0,
        **kwargs,
    )
    return client, requests


def ok(content: str) -> httpx.Response:
    return httpx.Response(200, json=completion(content))


# ---------------------------------------------------------------------------
# Construction and request shape
# ---------------------------------------------------------------------------


def test_client_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="http"):
        ArcLLMClient("ftp://example.test", API_KEY, "m")
    with pytest.raises(ValueError, match="blank"):
        ArcLLMClient(BASE_URL, API_KEY, "   ")
    with pytest.raises(ValueError, match="positive"):
        ArcLLMClient(BASE_URL, API_KEY, "m", timeout_seconds=0.0)
    with pytest.raises(ValueError, match="negative"):
        ArcLLMClient(BASE_URL, API_KEY, "m", max_retries=-1)
    with pytest.raises(ValueError, match="negative"):
        ArcLLMClient(BASE_URL, API_KEY, "m", retry_backoff_seconds=-1.0)


def test_from_settings_uses_environment_config() -> None:
    settings = Settings.from_env(
        {
            "VT_ARC_BASE_URL": "http://settings.test/v1/",
            "VT_ARC_API_KEY": "settings-key",
            "VT_ARC_MODEL": "settings-model",
            "EMBEDDING_MODEL": "embed",
        }
    )
    requests: list[dict[str, Any]] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return ok("hi")

    configured = ArcLLMClient.from_settings(
        settings,
        http_client=httpx.Client(transport=httpx.MockTransport(recording_handler)),
        retry_backoff_seconds=0.0,
    )
    assert configured.model_name == "settings-model"

    response = configured.chat((LLMMessage(role="user", content="hello"),))
    assert response.content == "hi"
    (payload,) = requests
    assert payload["model"] == "settings-model"


def test_chat_sends_openai_compatible_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok("hi")

    client, _ = make_client(handler)
    response = client.chat(
        (LLMMessage(role="user", content="hello"),), temperature=0.3, max_tokens=99
    )

    (raw_request,) = seen
    assert str(raw_request.url) == f"{BASE_URL}/chat/completions"
    assert raw_request.headers["Authorization"] == f"Bearer {API_KEY}"
    payload = json.loads(raw_request.content)
    assert payload["model"] == "test-model"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 99
    assert "response_format" not in payload

    assert response.content == "hi"
    assert response.model == "test-model"
    assert response.finish_reason == "stop"
    assert response.usage == TokenUsage(prompt_tokens=11, completion_tokens=7)
    assert response.attempts == 1
    assert response.latency_seconds >= 0.0


def test_chat_validates_messages_and_parameters() -> None:
    client, _ = make_client(lambda request: ok("hi"))
    with pytest.raises(ValueError, match="empty"):
        client.chat(())
    bogus = LLMMessage(role="tool", content="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="role"):
        client.chat((bogus,))
    with pytest.raises(ValueError, match="temperature"):
        client.chat((LLMMessage(role="user", content="x"),), temperature=3.0)
    with pytest.raises(ValueError, match="max_tokens"):
        client.chat((LLMMessage(role="user", content="x"),), max_tokens=0)


def test_chat_omits_max_tokens_when_none() -> None:
    client, requests = make_client(lambda request: ok("hi"))
    client.chat((LLMMessage(role="user", content="x"),), max_tokens=None)
    (payload,) = requests
    assert "max_tokens" not in payload


# ---------------------------------------------------------------------------
# Retries, timeouts, failure paths
# ---------------------------------------------------------------------------


def test_chat_retries_retryable_status_then_succeeds() -> None:
    responses = iter(
        [
            httpx.Response(503, text="overloaded"),
            httpx.Response(429, text="slow down"),
            ok("recovered"),
        ]
    )
    client, requests = make_client(lambda request: next(responses))
    response = client.chat((LLMMessage(role="user", content="x"),))

    assert response.content == "recovered"
    assert response.attempts == 3
    assert len(requests) == 3


def test_chat_fails_fast_on_client_errors() -> None:
    client, requests = make_client(lambda request: httpx.Response(400, text="bad request"))
    with pytest.raises(LLMResponseError, match="HTTP 400"):
        client.chat((LLMMessage(role="user", content="x"),))
    assert len(requests) == 1


def test_chat_gives_up_after_bounded_retries() -> None:
    client, requests = make_client(lambda request: httpx.Response(500, text="boom"), max_retries=1)
    with pytest.raises(LLMConnectionError, match="after 2 attempts: HTTP 500"):
        client.chat((LLMMessage(role="user", content="x"),))
    assert len(requests) == 2


def test_chat_timeout_is_retried_then_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out")

    client, requests = make_client(handler, max_retries=1)
    with pytest.raises(LLMTimeoutError, match="timed out after 2 attempts"):
        client.chat((LLMMessage(role="user", content="x"),))
    assert len(requests) == 2


def test_chat_transport_error_is_retried_then_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client, requests = make_client(handler, max_retries=2)
    with pytest.raises(LLMConnectionError, match="after 3 attempts"):
        client.chat((LLMMessage(role="user", content="x"),))
    assert len(requests) == 3


def test_chat_rejects_malformed_success_bodies() -> None:
    malformed = [
        httpx.Response(200, text="not json at all"),
        httpx.Response(200, json={}),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
        httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]}),
    ]
    for body in malformed:
        client, _ = make_client(lambda request, body=body: body)
        with pytest.raises(LLMResponseError):
            client.chat((LLMMessage(role="user", content="x"),))


def test_chat_defaults_missing_usage_to_zero() -> None:
    client, _ = make_client(
        lambda request: httpx.Response(200, json=completion("hi", with_usage=False))
    )
    response = client.chat((LLMMessage(role="user", content="x"),))
    assert response.usage == TokenUsage(prompt_tokens=0, completion_tokens=0)
    assert response.model == "test-model"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_chat_logs_latency_and_tokens_without_secrets(caplog: pytest.LogCaptureFixture) -> None:
    client, _ = make_client(lambda request: ok("hi"))
    with caplog.at_level(logging.INFO, logger="opponent_tendencies.llm.client"):
        client.chat((LLMMessage(role="user", content="top secret prompt"),))
    text = caplog.text
    assert "chat_completion" in text
    assert "prompt_tokens=11" in text
    assert "completion_tokens=7" in text
    assert "latency_ms=" in text
    assert API_KEY not in text
    assert "top secret prompt" not in text


def test_structured_logs_record_repairs_without_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    responses = iter([ok("garbage"), ok(json.dumps(VALID_AGENT_PAYLOAD))])
    client, _ = make_client(lambda request: next(responses))
    with caplog.at_level(logging.INFO, logger="opponent_tendencies.llm.client"):
        result = client.complete_structured(
            (LLMMessage(role="user", content="classify this"),), "agent_response"
        )
    assert result.repaired is True
    text = caplog.text
    assert "structured_output_repair" in text
    assert "structured_output" in text
    assert API_KEY not in text
    assert "classify this" not in text


# ---------------------------------------------------------------------------
# Structured output and the bounded repair
# ---------------------------------------------------------------------------


def test_complete_structured_parses_first_attempt() -> None:
    client, requests = make_client(lambda request: ok(json.dumps(VALID_AGENT_PAYLOAD)))
    result = client.complete_structured((LLMMessage(role="user", content="go"),), "agent_response")

    assert isinstance(result.parsed, AgentResponse)
    assert result.parsed.model_validate(VALID_AGENT_PAYLOAD) == result.parsed
    assert result.schema_name == "agent_response"
    assert result.repaired is False
    assert result.llm_calls == 1
    assert result.usage == TokenUsage(prompt_tokens=11, completion_tokens=7)
    assert result.model == "test-model"

    (payload,) = requests
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "agent_response"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["type"] == "object"


def test_complete_structured_repairs_invalid_json_once() -> None:
    invalid = "Sorry, I cannot produce JSON."
    responses = iter([ok(invalid), ok(json.dumps(VALID_AGENT_PAYLOAD))])
    client, requests = make_client(lambda request: next(responses))

    result = client.complete_structured((LLMMessage(role="user", content="go"),), "agent_response")

    assert result.repaired is True
    assert result.llm_calls == 2
    assert result.usage == TokenUsage(prompt_tokens=22, completion_tokens=14)
    assert len(requests) == 2

    repair_payload = requests[1]
    assert [message["role"] for message in repair_payload["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert repair_payload["messages"][1]["content"] == invalid
    assert "not a valid response" in repair_payload["messages"][2]["content"]
    assert repair_payload["response_format"]["json_schema"]["name"] == "agent_response"


def test_complete_structured_repairs_schema_violations() -> None:
    bad_payload = {
        "tendencies": [],
        "observations": [],
        "limitations": [],
        "unexpected": "field",
    }
    responses = iter([ok(json.dumps(bad_payload)), ok(json.dumps(VALID_AGENT_PAYLOAD))])
    client, _ = make_client(lambda request: next(responses))

    result = client.complete_structured((LLMMessage(role="user", content="go"),), "agent_response")
    assert result.repaired is True
    assert isinstance(result.parsed, AgentResponse)


def test_complete_structured_repair_is_bounded() -> None:
    responses = iter([ok("nope"), ok("still not json")])
    client, requests = make_client(lambda request: next(responses))

    with pytest.raises(StructuredOutputError, match="agent_response"):
        client.complete_structured((LLMMessage(role="user", content="go"),), "agent_response")
    assert len(requests) == 2


def test_complete_structured_accepts_fenced_json_without_repair() -> None:
    fenced = "```json\n" + json.dumps(VALID_AGENT_PAYLOAD) + "\n```"
    client, requests = make_client(lambda request: ok(fenced))
    result = client.complete_structured((LLMMessage(role="user", content="go"),), "agent_response")
    assert result.repaired is False
    assert len(requests) == 1


def test_complete_structured_accepts_json_with_surrounding_prose() -> None:
    chatty = "Here you go:\n" + json.dumps(VALID_AGENT_PAYLOAD) + "\nHope that helps!"
    client, requests = make_client(lambda request: ok(chatty))
    result = client.complete_structured((LLMMessage(role="user", content="go"),), "agent_response")
    assert result.repaired is False
    assert len(requests) == 1


def test_complete_structured_accepts_model_classes_and_rejects_unknowns() -> None:
    client, _ = make_client(lambda request: ok(json.dumps(VALID_AGENT_PAYLOAD)))
    by_class = client.complete_structured((LLMMessage(role="user", content="go"),), AgentResponse)
    assert by_class.schema_name == "agent_response"

    with pytest.raises(ValueError, match="unknown structured output"):
        client.complete_structured((LLMMessage(role="user", content="go"),), "nope")

    class Unregistered(WireModel):
        flag: bool

    with pytest.raises(ValueError, match="not a registered structured output model"):
        client.complete_structured((LLMMessage(role="user", content="go"),), Unregistered)


def test_token_usage_sums_across_calls() -> None:
    assert TokenUsage(1, 2) + TokenUsage(3, 4) == TokenUsage(prompt_tokens=4, completion_tokens=6)
    assert TokenUsage(1, 2).total_tokens == 3
