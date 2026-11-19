"""Team colors and the Phosphor helmet/football graphics shared by the UI and PDF."""

from __future__ import annotations

import base64
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from streamlit.testing.v1 import AppTest

from opponent_tendencies.models import Team
from opponent_tendencies.ui.graphics import (
    NEUTRAL_COLORS,
    TEAM_COLORS,
    football_parts,
    football_svg,
    helmet_parts,
    helmet_svg,
    team_accent,
    team_colors,
    team_theme,
    team_theme_css,
)
from tests.unit.factories import make_report

_HEX = re.compile(r"#[0-9A-Fa-f]{6}")


def _luma(hex_color: str) -> float:
    red, green, blue = (int(hex_color[start : start + 2], 16) for start in (1, 3, 5))
    return 0.299 * red + 0.587 * green + 0.114 * blue


def _contrast_ratio(one: str, two: str) -> float:
    def linear(value: int) -> float:
        ratio = value / 255
        return ratio / 12.92 if ratio <= 0.04045 else ((ratio + 0.055) / 1.055) ** 2.4

    def luminance(hex_color: str) -> float:
        red, green, blue = (int(hex_color[start : start + 2], 16) for start in (1, 3, 5))
        return 0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)

    first, second = luminance(one), luminance(two)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def _image_texts(app: AppTest) -> list[str]:
    """Decode the app's images, which Streamlit stores as base64 data URIs."""

    texts: list[str] = []
    for item in app.image:
        urls = item.value if isinstance(item.value, list) else [item.value]
        for url in urls:
            text = str(url)
            if "base64," in text:
                texts.append(base64.b64decode(text.split("base64,", 1)[1]).decode())
    return texts


def test_every_team_has_two_distinct_hex_colors() -> None:
    assert set(TEAM_COLORS) == set(Team)
    for primary, secondary in TEAM_COLORS.values():
        assert _HEX.fullmatch(primary)
        assert _HEX.fullmatch(secondary)
        assert primary != secondary


def test_team_accent_never_picks_the_lighter_color() -> None:
    for team in Team:
        primary, secondary = team_colors(team)
        accent = team_accent(team)
        chosen, other = (primary, secondary) if accent == primary else (secondary, primary)
        assert _luma(chosen) <= _luma(other)


def test_team_accent_avoids_silver_text() -> None:
    # Las Vegas' primary is silver, so the accent must fall back to black.
    assert team_accent(Team.LV) == TEAM_COLORS[Team.LV][1]
    assert team_accent(Team.BUF) == TEAM_COLORS[Team.BUF][0]


def test_helmet_svg_is_well_formed_and_tinted_by_team() -> None:
    svg = helmet_svg(Team.BUF)

    assert svg.startswith("<svg")
    assert ET.fromstring(svg).tag.endswith("}svg")
    primary, secondary = team_colors(Team.BUF)
    assert primary in svg
    assert secondary in svg
    assert "Buffalo Bills" in svg
    assert helmet_svg(Team.KC) != svg


def _bezier_point(
    start: tuple[float, ...],
    first: tuple[float, ...],
    second: tuple[float, ...],
    end: tuple[float, ...],
    t: float,
) -> tuple[float, float]:
    rest = 1.0 - t
    weights = (rest**3, 3.0 * rest**2 * t, 3.0 * rest * t**2, t**3)
    points = (start, first, second, end)
    return (
        sum(weight * point[0] for weight, point in zip(weights, points, strict=True)),
        sum(weight * point[1] for weight, point in zip(weights, points, strict=True)),
    )


def test_helmet_parts_are_the_phosphor_shell_and_outline() -> None:
    parts = helmet_parts(Team.SEA)
    primary, secondary = team_colors(Team.SEA)

    assert len(parts) == 2
    assert parts[0].fill == primary
    assert parts[1].fill == secondary
    assert all(part.stroke is None for part in parts)
    assert all(part.commands for part in parts)
    assert all(part.commands[0][0] == "M" for part in parts)
    assert {op for part in parts for op, _ in part.commands} == {"M", "L", "C", "Z"}
    # The shell is one subpath; the outline adds the ear hole, jaw, and mask windows.
    assert [sum(op == "M" for op, _ in part.commands) for part in parts] == [1, 5]


def test_helmet_parts_cover_the_phosphor_canvas() -> None:
    xs: list[float] = []
    ys: list[float] = []
    for part in helmet_parts(Team.KC):
        for _, coords in part.commands:
            xs += coords[::2]
            ys += coords[1::2]

    assert 23.0 <= min(xs) <= 25.0
    assert 231.0 <= max(xs) <= 233.0
    assert 23.0 <= min(ys) <= 25.0
    assert 223.0 <= max(ys) <= 225.0


def test_helmet_arcs_stay_on_their_circles() -> None:
    # The outline's first subpath is the ear hole: a 12-point circle at (84, 164).
    outline = helmet_parts(Team.SEA)[1].commands
    ear_hole: list[tuple[str, tuple[float, ...]]] = []
    for op, coords in outline:
        if op == "M" and ear_hole:
            break
        ear_hole.append((op, coords))

    assert ear_hole[0] == ("M", (96.0, 164.0))
    current: tuple[float, ...] = ear_hole[0][1]
    for op, coords in ear_hole:
        if op == "M":
            current = coords
        elif op == "C":
            for t in (0.25, 0.5, 0.75):
                point = _bezier_point(current, coords[:2], coords[2:4], coords[4:], t)
                radius = math.hypot(point[0] - 84.0, point[1] - 164.0)
                assert abs(radius - 12.0) < 0.05
            current = coords[4:]


def test_helmet_svg_inlines_the_asset_paths_recolored() -> None:
    asset_path = Path(__file__).parents[2] / "assets" / "football_helmet.svg"
    asset = ET.fromstring(asset_path.read_text(encoding="utf-8"))
    original = [element for element in asset.iter() if element.tag.endswith("}path")]

    svg = helmet_svg(Team.BUF)
    rendered = ET.fromstring(svg)
    recolored = [element for element in rendered.iter() if element.tag.endswith("}path")]

    assert len(original) == len(recolored) == 2
    assert [element.get("d") for element in recolored] == [element.get("d") for element in original]
    primary, secondary = team_colors(Team.BUF)
    assert [element.get("fill") for element in recolored] == [primary, secondary]
    assert 'viewBox="0 0 256 256"' in svg
    assert 'width="120"' in svg and 'height="120"' in svg
    assert "opacity" not in svg
    assert "currentColor" not in svg


def test_neutral_helmet_renders_without_a_selection() -> None:
    svg = helmet_svg(None)

    assert ET.fromstring(svg).tag.endswith("}svg")
    assert NEUTRAL_COLORS[0] in svg
    assert NEUTRAL_COLORS[1] in svg
    assert 'aria-label="Generic football helmet"' in svg
    assert helmet_svg(Team.BUF) != svg


def test_football_svg_is_well_formed_and_generic() -> None:
    svg = football_svg()

    assert svg.startswith("<svg")
    assert ET.fromstring(svg).tag.endswith("}svg")
    assert football_svg(width=48) != svg
    parts = football_parts()
    assert len(parts) == 7  # ball, two end stripes, four lace strokes
    assert all(part.commands for part in parts)


def test_app_renders_helmet_and_football_images() -> None:
    app = AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)
    app.session_state["report"] = make_report()
    app.run()

    assert not app.exception
    texts = _image_texts(app)
    # Title helmet and football, selection helmet, masthead helmet, and finding bullets.
    assert len(texts) >= 6
    assert any("helmet" in text for text in texts)
    assert any('aria-label="Football"' in text for text in texts)


def test_theme_colors_stay_readable_for_every_team() -> None:
    for team in Team:
        theme = team_theme(team)
        assert _contrast_ratio(theme.accent, "#FFFFFF") >= 3.0
        assert _contrast_ratio(theme.rule, "#FFFFFF") >= 3.0
        assert _contrast_ratio(theme.button, "#FFFFFF") >= 4.5
        assert _contrast_ratio(theme.hover, "#FFFFFF") >= 4.5
        assert theme.hover != theme.button


def test_theme_rule_prefers_the_secondary_when_visible_on_white() -> None:
    # Buffalo's red secondary is dark enough for borders; the Jets' white is not.
    assert team_theme(Team.BUF).rule == TEAM_COLORS[Team.BUF][1]
    assert team_theme(Team.NYJ).rule == team_accent(Team.NYJ)


def test_theme_css_styles_heading_button_and_borders() -> None:
    css = team_theme_css(Team.BUF)
    theme = team_theme(Team.BUF)

    assert css.startswith("<style>")
    assert css.endswith("</style>")
    assert f"color: {theme.accent} !important;" in css
    assert f"background-color: {theme.button} !important;" in css
    assert f"border-bottom: 4px solid {theme.rule};" in css
    assert 'button[data-testid="stBaseButton-primary"]' in css
    assert '[data-testid="stMarkdownContainer"] hr' in css


def test_app_themes_the_header_once_an_opponent_is_selected() -> None:
    app = AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)
    app.run()
    assert not any(str(item.value).startswith("<style>") for item in app.markdown)

    app.selectbox[0].set_value(Team.BUF.value)
    app.run()

    assert not app.exception
    primary, _ = team_colors(Team.BUF)
    texts = _image_texts(app)
    assert any("Generic football helmet in Buffalo Bills colors" in text for text in texts)
    assert any(primary in text for text in texts)
    themed = [str(item.value) for item in app.markdown if str(item.value).startswith("<style>")]
    assert len(themed) == 1
    assert "stBaseButton-primary" in themed[0]
    assert team_theme(Team.BUF).accent in themed[0]
