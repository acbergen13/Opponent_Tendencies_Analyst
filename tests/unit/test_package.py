"""Tests for the package skeleton and project metadata."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import opponent_tendencies
from opponent_tendencies import agents, data, llm, stats, ui, validation

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_package_exposes_version() -> None:
    assert opponent_tendencies.__version__


@pytest.mark.parametrize("module", [data, stats, validation, llm, agents, ui])
def test_subpackages_import(module: object) -> None:
    assert module is not None


def test_pyproject_version_matches_package_version() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)

    assert pyproject["project"]["version"] == opponent_tendencies.__version__
