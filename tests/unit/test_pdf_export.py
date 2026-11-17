"""PDF export: bytes are a real PDF and the UI button only appears with a report."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from opponent_tendencies.ui.pdf_export import build_report_pdf, report_pdf_filename
from tests.unit.factories import make_report


def _app() -> AppTest:
    return AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)


@pytest.mark.parametrize("degraded", [False, True])
def test_build_report_pdf_returns_pdf_bytes(degraded: bool) -> None:
    data = build_report_pdf(make_report(degraded=degraded))

    assert data.startswith(b"%PDF-")
    assert data.rstrip().endswith(b"%%EOF")
    assert len(data) > 2_000


def test_filename_names_opponent_season_and_weeks() -> None:
    report = make_report()
    name = report_pdf_filename(report)

    assert name.startswith(f"{report.opponent.value}_{report.season}_weeks_")
    assert name.endswith(".pdf")


def test_export_button_hidden_until_a_report_exists() -> None:
    app = _app()
    app.run()

    assert not app.exception
    assert not any(item.label == "Export to PDF" for item in app.get("download_button"))


def test_export_button_shown_with_a_report() -> None:
    app = _app()
    app.session_state["report"] = make_report()
    app.run()

    assert not app.exception
    assert any(item.label == "Export to PDF" for item in app.get("download_button"))
