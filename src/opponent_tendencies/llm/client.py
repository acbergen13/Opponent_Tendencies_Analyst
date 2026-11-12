"""OpenAI-compatible chat client for the VT ARC endpoint.

The client owns everything the embeddings module deliberately left small
(``llm/embeddings.py``): bounded retries with exponential backoff, a per-call
timeout, latency and token logging, and structured-output parsing with
exactly one bounded repair attempt (requirements 10 and 29 in the design
write-up; the prompts that call through here are Stage 6's constitutional
prompts).

No credentials are logged. Request headers, prompt bodies, and completions
never appear in log records or exception messages -- only model names,
statuses, attempt counts, latencies, token counts, and finish reasons.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from pydantic import ValidationError

from opponent_tendencies.config import Settings
from opponent_tendencies.llm.schemas import (
    STRUCTURED_OUTPUT_MODELS,
    WireModel,
    json_schema_for,
)

logger = logging.getLogger("opponent_tendencies.llm.client")

DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 0.5
DEFAULT_MAX_BACKOFF_SECONDS = 8.0
DEFAULT_TEMPERATURE = 0.1
DEFAULT_STRUCTURED_MAX_TOKENS = 8000
DEFAULT_ENABLE_THINKING = False

# VT ARC's proxy buffers non-streaming replies and refuses ``max_tokens`` above
# this cap (HTTP 413). Any call that needs a larger completion budget must be
# sent with ``stream=true``; ``chat`` switches automatically above the cap.
DEFAULT_BUFFERED_MAX_TOKENS = 8000

# VT ARC serves reasoning models through vLLM. The pinned ``vt-arc-llm`` model
# reasons by default, which can consume the whole completion cap before any
# JSON is emitted. Structured extraction does not need hidden reasoning; send
# the vLLM chat-template switch unless the operator opts back in.
DISABLED_THINKING_PAYLOAD: Mapping[str, Any] = {"enable_thinking": False}

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
VALID_ROLES: frozenset[str] = frozenset({"system", "user", "assistant"})

MessageRole = Literal["system", "user", "assistant"]

_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_MAX_VALIDATION_ERRORS = 5
_ERROR_EXCERPT_CHARS = 200

REPAIR_INSTRUCTION = (
    "Your previous reply was not a valid response: it was not JSON or it failed "
    "schema validation. Error summary: {error_summary} "
    "Return a corrected reply now: exactly one JSON object matching the provided "
    "JSON schema, with no markdown fences and no text outside the JSON object."
)


class LLMClientError(RuntimeError):
    """Base class for chat-client failures."""


class LLMConnectionError(LLMClientError):
    """Transport failures that survived every retry attempt."""


class LLMTimeoutError(LLMConnectionError):
    """Timeouts that survived every retry attempt."""


class LLMResponseError(LLMClientError):
    """The endpoint answered with a non-retryable status or a malformed body."""


class StructuredOutputError(LLMClientError):
    """A structured output failed validation even after the repair attempt."""


class _EmptyCompletionError(LLMResponseError):
    """A 200 response whose message content is empty or whitespace."""

    def __init__(self, message: str, *, finish_reason: str | None) -> None:
        super().__init__(message)
        self.finish_reason = finish_reason


@dataclass(frozen=True)
class LLMMessage:
    """One chat message. Only the three classic roles are allowed."""

    role: MessageRole
    content: str

    def to_payload(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported by the endpoint (zero when the server omits them).

    ``reasoning_tokens`` is a subset of ``completion_tokens`` on providers that
    expose hidden reasoning; it is recorded for diagnostics but never added to
    ``total_tokens`` a second time.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass(frozen=True)
class LLMResponse:
    """One successful chat completion."""

    content: str
    model: str
    finish_reason: str | None
    usage: TokenUsage
    latency_seconds: float
    attempts: int


@dataclass(frozen=True)
class StructuredResult[StructuredT: WireModel]:
    """A parsed structured output plus the call telemetry behind it."""

    parsed: StructuredT
    schema_name: str
    repaired: bool
    usage: TokenUsage
    latency_seconds: float
    llm_calls: int
    model: str


@dataclass(frozen=True)
class _Attempt:
    """One HTTP attempt reduced to the fields the retry loop needs."""

    status: int
    content: str = ""
    model: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    reasoning_present: bool = False
    tool_calls: int = 0
    error_excerpt: str = ""


class ArcLLMClient:
    """VT ARC OpenAI-compatible chat client with bounded retries and repair."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
        enable_thinking: bool = DEFAULT_ENABLE_THINKING,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("chat base_url must be an http(s) URL")
        if not model.strip():
            raise ValueError("chat model must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("chat timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("chat max_retries must not be negative")
        if retry_backoff_seconds < 0:
            raise ValueError("chat retry_backoff_seconds must not be negative")
        if max_backoff_seconds < 0:
            raise ValueError("chat max_backoff_seconds must not be negative")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model.strip()
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._enable_thinking = enable_thinking
        self._http_client = http_client

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
        enable_thinking: bool | None = None,
        http_client: httpx.Client | None = None,
    ) -> ArcLLMClient:
        """Build the client from validated environment settings."""

        return cls(
            settings.vt_arc_base_url,
            settings.vt_arc_api_key.get_secret_value(),
            settings.vt_arc_model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
            enable_thinking=(
                settings.vt_arc_enable_thinking if enable_thinking is None else enable_thinking
            ),
            http_client=http_client,
        )

    @property
    def model_name(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # Raw chat completion
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        stream: bool | None = None,
    ) -> LLMResponse:
        """Send one chat completion, retrying bounded, retryable failures.

        ``stream=None`` (the default) streams only when ``max_tokens`` exceeds
        the endpoint's buffered-mode cap; pass ``True``/``False`` to force it.
        """

        self._validate_messages(messages)
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [message.to_payload() for message in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            if max_tokens <= 0:
                raise ValueError("max_tokens must be positive")
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = dict(response_format)
        if not self._enable_thinking:
            payload["chat_template_kwargs"] = dict(DISABLED_THINKING_PAYLOAD)

        use_stream = (
            max_tokens is not None and max_tokens > DEFAULT_BUFFERED_MAX_TOKENS
            if stream is None
            else stream
        )
        if use_stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

        total_attempts = 1 + self._max_retries
        started = time.monotonic()
        last_failure = "unknown failure"
        last_error: BaseException | None = None

        for attempt in range(1, total_attempts + 1):
            try:
                outcome = self._send(payload, stream=use_stream)
            except httpx.TimeoutException as exc:
                last_failure = f"timeout: {exc}"
                last_error = exc
                logger.warning(
                    "chat_completion_retry model=%s attempt=%d/%d failure=timeout",
                    self._model,
                    attempt,
                    total_attempts,
                )
                self._sleep_backoff(attempt, total_attempts)
                continue
            except httpx.TransportError as exc:
                last_failure = f"transport: {exc}"
                last_error = exc
                logger.warning(
                    "chat_completion_retry model=%s attempt=%d/%d failure=transport",
                    self._model,
                    attempt,
                    total_attempts,
                )
                self._sleep_backoff(attempt, total_attempts)
                continue

            status = outcome.status
            if status == httpx.codes.OK:
                try:
                    content = _require_content(outcome)
                except _EmptyCompletionError as exc:
                    # A truncated empty reply is deterministic: the completion
                    # cap is exhausted, so another attempt repeats the failure.
                    if exc.finish_reason == "length":
                        raise
                    last_failure = str(exc)
                    last_error = exc
                    logger.warning(
                        "chat_completion_retry model=%s attempt=%d/%d "
                        "failure=empty_content finish_reason=%s",
                        self._model,
                        attempt,
                        total_attempts,
                        exc.finish_reason,
                    )
                    self._sleep_backoff(attempt, total_attempts)
                    continue

                latency = time.monotonic() - started
                logger.info(
                    "chat_completion model=%s attempt=%d/%d status=200 "
                    "latency_ms=%.0f prompt_tokens=%d completion_tokens=%d "
                    "reasoning_tokens=%d finish_reason=%s stream=%s",
                    outcome.model or self._model,
                    attempt,
                    total_attempts,
                    latency * 1000,
                    outcome.usage.prompt_tokens,
                    outcome.usage.completion_tokens,
                    outcome.usage.reasoning_tokens,
                    outcome.finish_reason,
                    use_stream,
                )
                return LLMResponse(
                    content=content,
                    model=outcome.model or self._model,
                    finish_reason=outcome.finish_reason,
                    usage=outcome.usage,
                    latency_seconds=latency,
                    attempts=attempt,
                )

            if status in RETRYABLE_STATUS_CODES:
                last_failure = f"HTTP {status}"
                last_error = None
                logger.warning(
                    "chat_completion_retry model=%s attempt=%d/%d status=%d",
                    self._model,
                    attempt,
                    total_attempts,
                    status,
                )
                self._sleep_backoff(attempt, total_attempts)
                continue

            raise LLMResponseError(
                f"chat completion failed with HTTP {status}: {outcome.error_excerpt}"
            )

        if isinstance(last_error, httpx.TimeoutException):
            raise LLMTimeoutError(
                f"chat completion timed out after {total_attempts} attempts"
            ) from last_error
        if isinstance(last_error, _EmptyCompletionError):
            raise LLMResponseError(
                f"chat completion returned no content after {total_attempts} attempts: "
                f"{last_failure}"
            ) from last_error
        raise LLMConnectionError(
            f"chat completion failed after {total_attempts} attempts: {last_failure}"
        ) from last_error

    # ------------------------------------------------------------------
    # Structured output with one bounded repair attempt
    # ------------------------------------------------------------------

    def complete_structured(
        self,
        messages: Sequence[LLMMessage],
        output: type[WireModel] | str,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = DEFAULT_STRUCTURED_MAX_TOKENS,
    ) -> StructuredResult[WireModel]:
        """Request a structured output and parse it with Pydantic.

        An invalid first reply triggers exactly one repair call that quotes the
        validation errors back to the model. A second failure raises
        :class:`StructuredOutputError`; the client never loops on repairs.
        """

        schema_name = self._resolve_schema_name(output)
        model_cls = STRUCTURED_OUTPUT_MODELS[schema_name]
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": json_schema_for(model_cls),
            },
        }

        first = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        try:
            parsed = _parse_structured(first.content, model_cls)
        except (ValueError, ValidationError) as first_error:
            if first.finish_reason == "length":
                raise LLMResponseError(_truncation_message(first, max_tokens)) from first_error
            first_summary = _error_summary(first_error)
            logger.warning(
                "structured_output_repair schema=%s first_error=%s",
                schema_name,
                first_summary,
            )
            repair_messages: list[LLMMessage] = [
                *messages,
                LLMMessage(role="assistant", content=first.content),
                LLMMessage(
                    role="user",
                    content=REPAIR_INSTRUCTION.format(error_summary=first_summary),
                ),
            ]
            repair = self.chat(
                repair_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            try:
                parsed = _parse_structured(repair.content, model_cls)
            except (ValueError, ValidationError) as second_error:
                if repair.finish_reason == "length":
                    raise LLMResponseError(
                        _truncation_message(repair, max_tokens)
                    ) from second_error
                raise StructuredOutputError(
                    f"structured output {schema_name!r} failed validation after one "
                    f"repair attempt; first error: {first_summary}; repair error: "
                    f"{_error_summary(second_error)}"
                ) from second_error
            result = StructuredResult(
                parsed=parsed,
                schema_name=schema_name,
                repaired=True,
                usage=first.usage + repair.usage,
                latency_seconds=first.latency_seconds + repair.latency_seconds,
                llm_calls=2,
                model=repair.model,
            )
            logger.info(
                "structured_output schema=%s repaired=true llm_calls=2 "
                "prompt_tokens=%d completion_tokens=%d reasoning_tokens=%d",
                schema_name,
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
                result.usage.reasoning_tokens,
            )
            return result

        result = StructuredResult(
            parsed=parsed,
            schema_name=schema_name,
            repaired=False,
            usage=first.usage,
            latency_seconds=first.latency_seconds,
            llm_calls=1,
            model=first.model,
        )
        logger.info(
            "structured_output schema=%s repaired=false llm_calls=1 "
            "prompt_tokens=%d completion_tokens=%d reasoning_tokens=%d",
            schema_name,
            result.usage.prompt_tokens,
            result.usage.completion_tokens,
            result.usage.reasoning_tokens,
        )
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_messages(self, messages: Sequence[LLMMessage]) -> None:
        if not messages:
            raise ValueError("messages must not be empty")
        for message in messages:
            if message.role not in VALID_ROLES:
                raise ValueError(
                    f"message role {message.role!r} is not one of {sorted(VALID_ROLES)}"
                )
            if not isinstance(message.content, str):
                raise ValueError("message content must be a string")

    def _resolve_schema_name(self, output: type[WireModel] | str) -> str:
        if isinstance(output, str):
            if output not in STRUCTURED_OUTPUT_MODELS:
                known = ", ".join(sorted(STRUCTURED_OUTPUT_MODELS))
                raise ValueError(f"unknown structured output {output!r}; known: {known}")
            return output
        for name, model_cls in STRUCTURED_OUTPUT_MODELS.items():
            if model_cls is output:
                return name
        raise ValueError(f"{output.__name__!r} is not a registered structured output model")

    def _send(self, payload: Mapping[str, Any], *, stream: bool) -> _Attempt:
        url = f"{self._base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if stream:
            if self._http_client is not None:
                with self._http_client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self._timeout_seconds,
                ) as response:
                    return self._read_stream(response)
            with (
                httpx.Client(timeout=self._timeout_seconds) as client,
                client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self._timeout_seconds,
                ) as response,
            ):
                return self._read_stream(response)

        if self._http_client is not None:
            response = self._http_client.post(
                url, json=payload, headers=headers, timeout=self._timeout_seconds
            )
        else:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.post(url, json=payload, headers=headers)
        if response.status_code != httpx.codes.OK:
            return _Attempt(
                status=response.status_code,
                error_excerpt=_error_excerpt(response),
            )
        return self._parse_completion(response)

    def _read_stream(self, response: httpx.Response) -> _Attempt:
        """Consume a server-sent-events completion into one attempt result."""

        if response.status_code != httpx.codes.OK:
            response.read()
            return _Attempt(
                status=response.status_code,
                error_excerpt=_error_excerpt(response),
            )

        # Some gateways ignore ``stream=true`` and return one buffered JSON body.
        # Accept it instead of reading an empty event stream.
        content_type = response.headers.get("content-type", "").lower()
        if "text/event-stream" not in content_type:
            response.read()
            return self._parse_completion(response)

        model = self._model
        finish_reason: str | None = None
        usage = TokenUsage()
        reasoning_present = False
        parts: list[str] = []
        for line in response.iter_lines():
            data = line.strip()
            if not data.startswith("data:"):
                continue
            chunk_text = data.removeprefix("data:").strip()
            if chunk_text == "[DONE]":
                break
            try:
                chunk = json.loads(chunk_text)
            except json.JSONDecodeError as exc:
                raise LLMResponseError("chat completion stream returned malformed JSON") from exc
            if not isinstance(chunk, dict):
                raise LLMResponseError("chat completion stream chunk must be a JSON object")

            reply_model = chunk.get("model")
            if isinstance(reply_model, str) and reply_model.strip():
                model = reply_model
            chunk_usage = chunk.get("usage")
            if isinstance(chunk_usage, dict):
                usage = _token_usage(chunk_usage)
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            first = choices[0]
            if not isinstance(first, dict):
                continue
            reason = first.get("finish_reason")
            if isinstance(reason, str):
                finish_reason = reason
            delta = first.get("delta")
            if not isinstance(delta, dict):
                continue
            text = delta.get("content")
            if isinstance(text, str):
                parts.append(text)
            reasoning = delta.get("reasoning_content")
            if not isinstance(reasoning, str):
                reasoning = delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                reasoning_present = True

        return _Attempt(
            status=httpx.codes.OK,
            content="".join(parts),
            model=model,
            finish_reason=finish_reason,
            usage=usage,
            reasoning_present=reasoning_present,
        )

    def _sleep_backoff(self, attempt: int, total_attempts: int) -> None:
        if attempt >= total_attempts:
            return
        delay = min(
            self._retry_backoff_seconds * (2 ** (attempt - 1)),
            self._max_backoff_seconds,
        )
        if delay > 0:
            time.sleep(delay)

    def _parse_completion(self, response: httpx.Response) -> _Attempt:
        try:
            document = response.json()
        except ValueError as exc:
            raise LLMResponseError("chat completion returned non-JSON content") from exc
        if not isinstance(document, dict):
            raise LLMResponseError("chat completion response must be a JSON object")

        choices = document.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseError("chat completion response has no choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise LLMResponseError("chat completion choice must be a JSON object")
        message = first.get("message")
        if not isinstance(message, dict):
            raise LLMResponseError("chat completion choice has no message")

        reply_model = document.get("model")
        if not isinstance(reply_model, str) or not reply_model.strip():
            reply_model = self._model
        finish_reason = first.get("finish_reason")
        if not isinstance(finish_reason, str):
            finish_reason = None

        usage = document.get("usage")
        usage_document = usage if isinstance(usage, dict) else {}

        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str):
            reasoning = message.get("reasoning")
        tool_calls = message.get("tool_calls")
        content = message.get("content")
        return _Attempt(
            status=httpx.codes.OK,
            content=content if isinstance(content, str) else "",
            model=reply_model,
            finish_reason=finish_reason,
            usage=_token_usage(usage_document),
            reasoning_present=isinstance(reasoning, str) and bool(reasoning.strip()),
            tool_calls=len(tool_calls) if isinstance(tool_calls, list) else 0,
        )


def _token_usage(usage: Mapping[Any, Any]) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=_int_or_zero(usage.get("prompt_tokens")),
        completion_tokens=_int_or_zero(usage.get("completion_tokens")),
        reasoning_tokens=_reasoning_tokens(usage),
    )


def _require_content(attempt: _Attempt) -> str:
    if attempt.content.strip():
        return attempt.content
    raise _EmptyCompletionError(
        "chat completion message has no content "
        f"(finish_reason={attempt.finish_reason}, "
        f"completion_tokens={attempt.usage.completion_tokens}, "
        f"reasoning_tokens={attempt.usage.reasoning_tokens}, "
        f"reasoning_present={attempt.reasoning_present}, "
        f"tool_calls={attempt.tool_calls})",
        finish_reason=attempt.finish_reason,
    )


def _reasoning_tokens(usage: Mapping[Any, Any]) -> int:
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return 0
    return _int_or_zero(details.get("reasoning_tokens"))


def _int_or_zero(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _error_excerpt(response: httpx.Response) -> str:
    text = response.text.strip().replace("\n", " ")
    return text[:_ERROR_EXCERPT_CHARS] or "<empty body>"


def _truncation_message(response: LLMResponse, max_tokens: int | None) -> str:
    cap = f"max_tokens={max_tokens}" if max_tokens is not None else "no explicit max_tokens"
    return (
        "chat completion response was truncated at the completion limit "
        f"(finish_reason=length, completion_tokens={response.usage.completion_tokens}, "
        f"reasoning_tokens={response.usage.reasoning_tokens}, {cap}); "
        "raise the completion cap or shorten the prompt"
    )


def _parse_structured(content: str, model_cls: type[WireModel]) -> WireModel:
    document = _load_json(content)
    return model_cls.model_validate(document)


def _load_json(content: str) -> object:
    text = content.strip()
    if not text:
        raise ValueError("model returned an empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = _FENCED_JSON.search(text)
    if fenced is not None:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"response is not valid JSON: {text[:120]!r}")


def _error_summary(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts: list[str] = []
        for error in exc.errors(include_url=False)[:_MAX_VALIDATION_ERRORS]:
            location = ".".join(str(part) for part in error["loc"]) or "(root)"
            parts.append(f"{location}: {error['msg']}")
        return "; ".join(parts) or "schema validation failed"
    return str(exc)[:_ERROR_EXCERPT_CHARS]


__all__ = [
    "DEFAULT_BUFFERED_MAX_TOKENS",
    "DEFAULT_ENABLE_THINKING",
    "DEFAULT_MAX_BACKOFF_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_BACKOFF_SECONDS",
    "DEFAULT_STRUCTURED_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_SECONDS",
    "REPAIR_INSTRUCTION",
    "RETRYABLE_STATUS_CODES",
    "VALID_ROLES",
    "ArcLLMClient",
    "LLMClientError",
    "LLMConnectionError",
    "LLMMessage",
    "LLMResponse",
    "LLMResponseError",
    "LLMTimeoutError",
    "MessageRole",
    "StructuredOutputError",
    "StructuredResult",
    "TokenUsage",
]
