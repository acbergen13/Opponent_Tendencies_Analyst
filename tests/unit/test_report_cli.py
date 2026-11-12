"""The full-report CLI: argument handling, summary, and JSON output."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from opponent_tendencies.config import Settings
from opponent_tendencies.data.play_store import SeasonStore
from opponent_tendencies.models import Team
from opponent_tendencies.report_cli import build_parser, main, render_summary, run_payload
from opponent_tendencies.ui.runner import build_report
from tests.agent_fixtures import (
    make_client,
    media_data,
    media_index,
)
from tests.agent_fixtures import (
    season_data as sample_season_data,
)
from tests.unit.test_orchestrator import _routing_handler, _run


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        vt_arc_base_url="http://vt-arc.test/v1",
        vt_arc_api_key=SecretStr("secret-test-key"),
        vt_arc_model="test-model",
        embedding_model="test-embedding",
        data_dir=tmp_path,
        chroma_dir=tmp_path / "chroma",
    )


def test_parser_defaults_and_required_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(["--opponent", "BUF", "--weeks", "5", "6"])

    assert args.season == 2026
    assert args.no_cache is False
    assert args.json is False
    assert args.output is None
    with pytest.raises(SystemExit):
        parser.parse_args(["--weeks", "5"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--opponent", "BUF"])


def test_main_runs_the_pipeline_and_writes_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, _ = _run(tmp_path=tmp_path, handler=_routing_handler())
    captured: dict[str, Any] = {}

    def fake_build_report(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return run

    monkeypatch.setattr("opponent_tendencies.report_cli.get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr("opponent_tendencies.report_cli.build_report", fake_build_report)
    output = tmp_path / "report.json"
    stdout = StringIO()

    code = main(
        [
            "--opponent",
            "buf",
            "--weeks",
            "9",
            "5",
            "6",
            "7",
            "8",
            "8",
            "--output",
            str(output),
            "--no-cache",
            "--quiet",
        ],
        stdout=stdout,
    )

    assert code == 0
    assert captured["opponent"].value == "BUF"
    assert captured["weeks"] == (5, 6, 7, 8, 9)
    assert captured["use_cache"] is False
    assert captured["data_dir"] == tmp_path
    assert captured["chroma_dir"] == tmp_path / "chroma"
    assert captured["agent_timeout_seconds"] > 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert run.report is not None
    assert payload["opponent"] == "BUF"
    assert payload["weeks"] == [5, 6, 7, 8, 9]
    assert payload["status"]["state"] == "completed"
    assert payload["report"]["report_id"] == run.report.report_id
    assert payload["usage"]["total_tokens"] == run.usage.total_tokens
    assert "Status: completed" in stdout.getvalue()
    assert f"Wrote {output}" in stdout.getvalue()


def test_main_returns_nonzero_and_explains_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run, _ = _run(tmp_path=None, handler=_routing_handler(), weeks=[1])
    monkeypatch.setattr("opponent_tendencies.report_cli.get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr("opponent_tendencies.report_cli.build_report", lambda **_: run)
    stdout = StringIO()

    code = main(
        ["--opponent", "BUF", "--weeks", "1", "--quiet"],
        stdout=stdout,
    )

    assert code == 1
    assert "Status: failed" in stdout.getvalue()
    assert "validation_failed" in stdout.getvalue()
    assert "Message:" in stdout.getvalue()


def test_json_mode_prints_the_payload_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run, _ = _run(tmp_path=tmp_path, handler=_routing_handler())
    monkeypatch.setattr("opponent_tendencies.report_cli.get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr("opponent_tendencies.report_cli.build_report", lambda **_: run)
    stdout = StringIO()

    code = main(
        ["--opponent", "BUF", "--weeks", "5", "6", "7", "8", "9", "--json", "--quiet"],
        stdout=stdout,
    )

    assert code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["report"]["opponent"] == "BUF"
    assert "Status:" not in stdout.getvalue()


def test_main_rejects_unknown_teams_and_bad_weeks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("opponent_tendencies.report_cli.get_settings", lambda: _settings(tmp_path))

    with pytest.raises(SystemExit):
        main(["--opponent", "XXX", "--weeks", "5"])
    with pytest.raises(SystemExit):
        main(["--opponent", "BUF", "--weeks", "0"])


def test_summary_and_payload_for_a_failed_run() -> None:
    run, _ = _run(tmp_path=None, handler=_routing_handler(), weeks=[1])

    summary = render_summary(run)
    payload = run_payload(run)

    assert "Status: failed" in summary
    assert "Report cache: miss" in summary
    assert payload["report"] is None
    assert payload["opponent"] is None
    assert payload["weeks"] is None
    status = payload["status"]
    assert isinstance(status, dict)
    assert status["failure_reason"] == "validation_failed"


def test_build_report_passes_budgets_and_bypasses_the_cache(tmp_path: Path) -> None:
    data = sample_season_data()
    media = media_data(tmp_path)
    index = media_index(media)
    client, _ = make_client(_routing_handler())
    steps: list[str] = []
    cache_dir = tmp_path / "reports"

    run = build_report(
        opponent=Team.BUF,
        weeks=[5, 6, 7, 8, 9],
        season_data=data,
        media_data=media,
        index=index,
        client=client,
        cache_dir=cache_dir,
        use_cache=False,
        min_sample=7,
        deadline_seconds=30.0,
        agent_timeout_seconds=10.0,
        max_llm_calls=10,
        token_budget=100_000,
        on_progress=lambda event: steps.append(event.step),
    )

    assert run.report is not None
    assert run.cached is False
    assert run.report.stat_tables
    assert all(table.min_sample == 7 for table in run.report.stat_tables)
    assert steps
    assert not cache_dir.exists()


def test_stats_only_runs_without_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = sample_season_data()
    SeasonStore(2026, tmp_path / "seasons" / "2026.json").refresh(data.games, data.plays)
    monkeypatch.setattr("opponent_tendencies.report_cli.resolve_data_dir", lambda: tmp_path)

    def fail_settings() -> Settings:
        raise AssertionError("--stats-only must not read VT ARC settings")

    monkeypatch.setattr("opponent_tendencies.report_cli.get_settings", fail_settings)
    stdout = StringIO()

    code = main(
        ["--opponent", "BUF", "--weeks", "5", "6", "7", "8", "9", "--stats-only", "--json"],
        stdout=stdout,
    )

    assert code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["opponent"] == "BUF"
    assert payload["weeks"] == [5, 6, 7, 8, 9]
    assert payload["tables"]
    assert payload["quality"]["offensive_plays"] > 0


def test_stats_only_reports_an_unplayable_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("opponent_tendencies.report_cli.resolve_data_dir", lambda: tmp_path)
    stdout = StringIO()
    stderr = StringIO()

    code = main(
        ["--opponent", "BUF", "--weeks", "5", "--stats-only"],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert "not playable" in stderr.getvalue()
    assert stdout.getvalue() == ""
