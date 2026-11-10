"""Tests for fail-fast configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from opponent_tendencies.config import (
    ConfigurationError,
    Settings,
    get_settings,
)

VALID_ENV = {
    "VT_ARC_BASE_URL": "https://llm-api.arc.vt.edu/api/v1",
    "VT_ARC_API_KEY": "test-key",
    "VT_ARC_MODEL": "vt-arc-llm",
    "EMBEDDING_MODEL": "test-embedding-model",
}


def test_missing_api_key_fails_with_actionable_message() -> None:
    env = {name: value for name, value in VALID_ENV.items() if name != "VT_ARC_API_KEY"}

    with pytest.raises(ConfigurationError) as excinfo:
        Settings.from_env(env)

    message = str(excinfo.value)
    assert "VT_ARC_API_KEY" in message
    assert ".env.example" in message


def test_all_missing_variables_are_reported_together() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        Settings.from_env({})

    message = str(excinfo.value)
    for name in VALID_ENV:
        assert name in message


def test_blank_values_are_treated_as_missing() -> None:
    env = dict(VALID_ENV, VT_ARC_API_KEY="   ")

    with pytest.raises(ConfigurationError, match="VT_ARC_API_KEY"):
        Settings.from_env(env)


def test_valid_environment_builds_immutable_settings() -> None:
    settings = Settings.from_env(VALID_ENV)

    assert settings.vt_arc_base_url == "https://llm-api.arc.vt.edu/api/v1"
    assert settings.vt_arc_model == "vt-arc-llm"
    assert settings.embedding_model == "test-embedding-model"
    assert settings.data_dir == Path("data")
    assert settings.chroma_dir == Path("data/chroma")
    assert settings.vt_arc_api_key.get_secret_value() == "test-key"

    with pytest.raises(ValidationError):
        settings.vt_arc_model = "other"


def test_api_key_is_not_leaked_by_repr() -> None:
    settings = Settings.from_env(VALID_ENV)

    assert "test-key" not in repr(settings)


def test_invalid_base_url_is_rejected() -> None:
    env = dict(VALID_ENV, VT_ARC_BASE_URL="llm-api.arc.vt.edu/api/v1")

    with pytest.raises(ConfigurationError, match="vt_arc_base_url"):
        Settings.from_env(env)


def test_paths_from_environment_are_expanded() -> None:
    env = dict(VALID_ENV, DATA_DIR="~/opponent-tendencies-data", CHROMA_DIR="~/idx")

    settings = Settings.from_env(env)

    assert settings.data_dir == Path.home() / "opponent-tendencies-data"
    assert settings.chroma_dir == Path.home() / "idx"


def test_get_settings_reads_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    for name, value in VALID_ENV.items():
        monkeypatch.setenv(name, value)

    try:
        settings = get_settings()
    finally:
        get_settings.cache_clear()

    assert settings.vt_arc_model == "vt-arc-llm"


def test_resolve_data_dir_needs_no_model_credentials() -> None:
    from opponent_tendencies.config import resolve_data_dir

    assert resolve_data_dir({"DATA_DIR": "~/season-data"}) == Path.home() / "season-data"
    assert resolve_data_dir({}) == Path("data")
