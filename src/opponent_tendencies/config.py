"""Environment-driven configuration.

Configuration is read from environment variables, optionally seeded from a
local ``.env`` file, and validated once before any expensive work starts.
Missing or malformed values raise :class:`ConfigurationError` with an
actionable message; the application never falls back to silent defaults for
credentials.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError, field_validator

REQUIRED_VARS: tuple[str, ...] = (
    "VT_ARC_BASE_URL",
    "VT_ARC_API_KEY",
    "VT_ARC_MODEL",
    "EMBEDDING_MODEL",
)

DEFAULT_DATA_DIR = Path("data")
DEFAULT_CHROMA_DIR = Path("data/chroma")


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _path_setting(raw: Mapping[str, str], name: str, default: Path) -> Path:
    value = raw.get(name, "").strip()
    return Path(value).expanduser() if value else default


def _missing_vars_message(missing: Sequence[str]) -> str:
    names = ", ".join(missing)
    return (
        f"Missing required environment variable(s): {names}. "
        "Copy .env.example to .env and set a value for each one "
        "(see PLAN.md, section 3)."
    )


class Settings(BaseModel):
    """Validated runtime settings.

    The API key is held as a :class:`pydantic.SecretStr` so it is never
    rendered in logs, traces, or ``repr`` output.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    vt_arc_base_url: str
    vt_arc_api_key: SecretStr
    vt_arc_model: str
    embedding_model: str
    data_dir: Path = DEFAULT_DATA_DIR
    chroma_dir: Path = DEFAULT_CHROMA_DIR

    @field_validator("vt_arc_base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("must be an http:// or https:// URL")
        return value.rstrip("/")

    @field_validator("vt_arc_model", "embedding_model")
    @classmethod
    def _validate_model_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("data_dir", "chroma_dir", mode="before")
    @classmethod
    def _expand_path(cls, value: object) -> Path:
        if isinstance(value, Path):
            return value.expanduser()
        if isinstance(value, str):
            return Path(value).expanduser()
        raise ValueError("must be a filesystem path")  # pragma: no cover - pydantic guards

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``env`` (defaults to ``os.environ`` + ``.env``)."""

        raw = _read_env(env)
        missing = [name for name in REQUIRED_VARS if not raw.get(name, "").strip()]
        if missing:
            raise ConfigurationError(_missing_vars_message(missing))

        try:
            return cls(
                vt_arc_base_url=raw["VT_ARC_BASE_URL"].strip(),
                vt_arc_api_key=SecretStr(raw["VT_ARC_API_KEY"].strip()),
                vt_arc_model=raw["VT_ARC_MODEL"].strip(),
                embedding_model=raw["EMBEDDING_MODEL"].strip(),
                data_dir=_path_setting(raw, "DATA_DIR", DEFAULT_DATA_DIR),
                chroma_dir=_path_setting(raw, "CHROMA_DIR", DEFAULT_CHROMA_DIR),
            )
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            )
            raise ConfigurationError(f"Invalid configuration: {details}") from exc


def _read_env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    if env is not None:
        return env
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    return os.environ


def resolve_data_dir(env: Mapping[str, str] | None = None) -> Path:
    """Return the data directory without requiring LLM credentials.

    Data ingestion, index builds, and offline scripts only need ``DATA_DIR``;
    they should not fail because a model key is absent. Full :class:`Settings`
    remains the gate for anything that calls the LLM.
    """

    return _path_setting(_read_env(env), "DATA_DIR", DEFAULT_DATA_DIR)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loading and validating them once."""

    return Settings.from_env()
