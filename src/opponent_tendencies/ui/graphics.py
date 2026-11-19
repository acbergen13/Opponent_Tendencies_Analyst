"""Team colors, app theming, and helmet/football vector graphics.

Colors are approximate official team colors used only to tint *generic* vector
shapes and app chrome: no team logos, wordmarks, or other protected marks are
drawn. Geometry lives in SVG-style view coordinates (y grows downward) as
:class:`VectorPart` data, so the Streamlit app (SVG strings) and the PDF export
(canvas paths) render identical graphics. The football is hand-authored; the
helmet is the MIT-licensed Phosphor Icons duotone football helmet
(https://phosphoricons.com), stored verbatim at ``assets/football_helmet.svg``
and read locally — the page inlines its markup recolored per team, and the PDF
draws the same paths compiled to :class:`VectorPart` commands.
"""

from __future__ import annotations

import copy
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from opponent_tendencies.models import TEAM_DISPLAY_NAMES, Team

__all__ = [
    "FOOTBALL_VIEWBOX",
    "HELMET_VIEWBOX",
    "NEUTRAL_COLORS",
    "TEAM_COLORS",
    "TeamTheme",
    "VectorPart",
    "football_parts",
    "football_svg",
    "helmet_parts",
    "helmet_svg",
    "team_accent",
    "team_colors",
    "team_theme",
    "team_theme_css",
]

type Command = tuple[str, tuple[float, ...]]


@dataclass(frozen=True)
class VectorPart:
    """One filled and/or stroked path in view coordinates (y grows downward)."""

    commands: tuple[Command, ...]
    fill: str | None = None
    stroke: str | None = None
    stroke_width: float = 0.0


TEAM_COLORS: dict[Team, tuple[str, str]] = {
    Team.ARI: ("#97233F", "#FFB612"),
    Team.ATL: ("#A71930", "#101820"),
    Team.BAL: ("#241773", "#9E7C0C"),
    Team.BUF: ("#00338D", "#C60C30"),
    Team.CAR: ("#101820", "#0085CA"),
    Team.CHI: ("#0B162A", "#C83803"),
    Team.CIN: ("#FB4F14", "#101820"),
    Team.CLE: ("#311D00", "#FF3C00"),
    Team.DAL: ("#041E42", "#869397"),
    Team.DEN: ("#002244", "#FB4F14"),
    Team.DET: ("#0076B6", "#B0B7BC"),
    Team.GB: ("#203731", "#FFB612"),
    Team.HOU: ("#03202F", "#A71930"),
    Team.IND: ("#002C5F", "#A2AAAD"),
    Team.JAX: ("#006778", "#D7A22A"),
    Team.KC: ("#E31837", "#FFB81C"),
    Team.LA: ("#003594", "#FFD100"),
    Team.LAC: ("#0080C6", "#FFC20E"),
    Team.LV: ("#A5ACAF", "#101820"),
    Team.MIA: ("#008E97", "#FC4C02"),
    Team.MIN: ("#4F2683", "#FFC62F"),
    Team.NE: ("#002244", "#C60C30"),
    Team.NO: ("#D3BC8D", "#101820"),
    Team.NYG: ("#0B2265", "#A71930"),
    Team.NYJ: ("#125740", "#FFFFFF"),
    Team.PHI: ("#004C54", "#A5ACAF"),
    Team.PIT: ("#101820", "#FFB612"),
    Team.SEA: ("#002244", "#69BE28"),
    Team.SF: ("#AA0000", "#B3995D"),
    Team.TB: ("#D50A0A", "#464545"),
    Team.TEN: ("#0C2340", "#4B92DB"),
    Team.WSH: ("#5A1414", "#FFB612"),
}


NEUTRAL_COLORS: tuple[str, str] = ("#454B54", "#C2C8CF")


def team_colors(team: Team | None) -> tuple[str, str]:
    """Return the approximate official (primary, secondary) colors for ``team``.

    ``None`` returns a neutral charcoal/silver pair used before any opponent is
    selected.
    """

    if team is None:
        return NEUTRAL_COLORS
    return TEAM_COLORS[team]


def _luma(hex_color: str) -> float:
    red, green, blue = (int(hex_color[start : start + 2], 16) for start in (1, 3, 5))
    return 0.299 * red + 0.587 * green + 0.114 * blue


def team_accent(team: Team) -> str:
    """Return the darker of a team's colors, readable as text or a rule on white."""

    primary, secondary = TEAM_COLORS[team]
    return secondary if _luma(secondary) < _luma(primary) else primary


_WHITE = "#FFFFFF"
_BLACK = "#000000"


def _channels(hex_color: str) -> tuple[int, int, int]:
    return (int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16))


def _relative_luminance(hex_color: str) -> float:
    def linear(value: int) -> float:
        ratio = value / 255
        return ratio / 12.92 if ratio <= 0.04045 else ((ratio + 0.055) / 1.055) ** 2.4

    red, green, blue = _channels(hex_color)
    return 0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)


def _contrast_ratio(one: str, two: str) -> float:
    """Return the WCAG contrast ratio between two hex colors."""

    first = _relative_luminance(one)
    second = _relative_luminance(two)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def _mix(hex_color: str, other: str, amount: float) -> str:
    """Blend ``amount`` of ``other`` into ``hex_color`` (0.0 keeps it unchanged)."""

    blended = [
        round(a + (b - a) * amount)
        for a, b in zip(_channels(hex_color), _channels(other), strict=True)
    ]
    return f"#{blended[0]:02X}{blended[1]:02X}{blended[2]:02X}"


def _readable_on_white(hex_color: str, target: float = 4.5) -> str:
    """Darken ``hex_color`` just enough for white text on it to reach ``target``."""

    color = hex_color
    for _ in range(10):
        if _contrast_ratio(color, _WHITE) >= target:
            break
        color = _mix(color, _BLACK, 0.08)
    return color


@dataclass(frozen=True)
class TeamTheme:
    """Chrome colors derived from one team, all readable on a light background."""

    accent: str
    rule: str
    button: str
    hover: str


def team_theme(team: Team) -> TeamTheme:
    """Derive heading, border, and button colors for ``team``.

    The accent keeps the team's darker color (large heading text needs only
    3:1); rules use the secondary when it is visible on white; the button
    background darkens the accent until its white label reaches 4.5:1.
    """

    accent = team_accent(team)
    secondary = team_colors(team)[1]
    rule = secondary if _contrast_ratio(secondary, _WHITE) >= 3.0 else accent
    button = _readable_on_white(accent)
    return TeamTheme(accent=accent, rule=rule, button=button, hover=_mix(button, _BLACK, 0.12))


def team_theme_css(team: Team) -> str:
    """Return a ``<style>`` block tinting the heading, primary button, and borders.

    Selectors target Streamlit's stable ``stMarkdownContainer`` and
    ``stBaseButton-primary`` test ids, so only the main heading, the Build
    Report button, dividers, and the heading's accent underline change.
    """

    theme = team_theme(team)
    return f"""<style>
[data-testid="stMarkdownContainer"] h1 {{
  color: {theme.accent} !important;
}}
[data-testid="stMarkdownContainer"] h1::after {{
  content: "";
  display: block;
  width: 96px;
  border-bottom: 4px solid {theme.rule};
  margin: 4px 0 2px;
}}
[data-testid="stMarkdownContainer"] hr {{
  border-top-color: {theme.rule} !important;
}}
button[data-testid="stBaseButton-primary"] {{
  background-color: {theme.button} !important;
  border-color: {theme.button} !important;
  color: {_WHITE} !important;
}}
button[data-testid="stBaseButton-primary"]:hover {{
  background-color: {theme.hover} !important;
  border-color: {theme.hover} !important;
  color: {_WHITE} !important;
}}
</style>"""


# --- shared geometry ------------------------------------------------------------

_KAPPA = 0.5522847498


def _ellipse_commands(cx: float, cy: float, rx: float, ry: float) -> tuple[Command, ...]:
    """Approximate an ellipse with four cubic segments, portable to SVG and canvas."""

    dx, dy = _KAPPA * rx, _KAPPA * ry
    return (
        ("M", (cx + rx, cy)),
        ("C", (cx + rx, cy + dy, cx + dx, cy + ry, cx, cy + ry)),
        ("C", (cx - dx, cy + ry, cx - rx, cy + dy, cx - rx, cy)),
        ("C", (cx - rx, cy - dy, cx - dx, cy - ry, cx, cy - ry)),
        ("C", (cx + dx, cy - ry, cx + rx, cy - dy, cx + rx, cy)),
        ("Z", ()),
    )


# --- SVG path compilation ---------------------------------------------------------

_PATH_TOKEN = re.compile(r"[MmLlHhVvCcAaZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_PATH_COMMAND_SIZES: dict[str, int] = {
    "M": 2,
    "L": 2,
    "H": 1,
    "V": 1,
    "C": 6,
    "A": 7,
    "Z": 0,
}
_PATH_SEPARATORS = " ,\t\n\r\f"


def _tokenize_path(data: str) -> list[tuple[str, list[float]]]:
    """Split SVG path data into command letters, each with its coordinate groups."""

    tokens: list[tuple[str, list[float]]] = []
    letter = ""
    numbers: list[float] = []
    position = 0
    for match in _PATH_TOKEN.finditer(data):
        if data[position : match.start()].strip(_PATH_SEPARATORS):
            raise ValueError(f"unexpected text in SVG path data: {data!r}")
        position = match.end()
        token = match.group(0)
        if token.isalpha():
            if letter:
                tokens.append((letter, numbers))
            letter, numbers = token, []
        else:
            if not letter:
                raise ValueError(f"SVG path data starts with a coordinate: {data!r}")
            numbers.append(float(token))
    if data[position:].strip(_PATH_SEPARATORS):
        raise ValueError(f"unexpected text in SVG path data: {data!r}")
    if not letter:
        raise ValueError("empty SVG path data")
    tokens.append((letter, numbers))
    return tokens


def _parse_svg_path(data: str) -> tuple[Command, ...]:
    """Compile SVG path data into absolute M/L/C/Z commands in view coordinates.

    Relative commands, H/V segments, and arcs are expanded, and the coordinates
    of every arc are approximated with cubic segments, so the result draws the
    same geometry with the command set shared by the SVG and canvas renderers.
    """

    commands: list[Command] = []
    x = y = 0.0
    start_x = start_y = 0.0
    for letter, numbers in _tokenize_path(data):
        upper = letter.upper()
        if upper == "Z":
            if numbers:
                raise ValueError("an SVG closepath takes no coordinates")
            commands.append(("Z", ()))
            x, y = start_x, start_y
            continue
        size = _PATH_COMMAND_SIZES[upper]
        if not numbers or len(numbers) % size:
            raise ValueError(f"SVG path command {letter} needs {size} coordinates per point")
        for offset in range(0, len(numbers), size):
            params = numbers[offset : offset + size]
            op = letter
            if offset and upper == "M":
                op = "l" if letter == "m" else "L"
            if op in ("M", "m"):
                x, y = (x + params[0], y + params[1]) if op == "m" else (params[0], params[1])
                start_x, start_y = x, y
                commands.append(("M", (x, y)))
            elif op in ("L", "l"):
                x, y = (x + params[0], y + params[1]) if op == "l" else (params[0], params[1])
                commands.append(("L", (x, y)))
            elif op in ("H", "h"):
                x = x + params[0] if op == "h" else params[0]
                commands.append(("L", (x, y)))
            elif op in ("V", "v"):
                y = y + params[0] if op == "v" else params[0]
                commands.append(("L", (x, y)))
            elif op in ("C", "c"):
                coords = list(params)
                if op == "c":
                    coords = [
                        value + (x if index % 2 == 0 else y) for index, value in enumerate(coords)
                    ]
                commands.append(("C", tuple(coords)))
                x, y = coords[4], coords[5]
            else:
                rx, ry, rotation, large_arc, sweep, end_x, end_y = params
                target = (x + end_x, y + end_y) if op == "a" else (end_x, end_y)
                commands.extend(_arc_commands(x, y, rx, ry, rotation, large_arc, sweep, *target))
                x, y = target
    return tuple(commands)


def _arc_commands(
    x1: float,
    y1: float,
    rx: float,
    ry: float,
    rotation: float,
    large_arc: float,
    sweep: float,
    x2: float,
    y2: float,
) -> tuple[Command, ...]:
    """Approximate one SVG elliptical arc with cubic bezier segments.

    This is the endpoint-to-center conversion from the SVG specification,
    followed by the standard at-most-quarter-circle cubic split.
    """

    if (x1, y1) == (x2, y2):
        return ()
    rx, ry = abs(rx), abs(ry)
    if rx == 0.0 or ry == 0.0:
        return (("L", (x2, y2)),)
    phi = math.radians(rotation)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    half_dx, half_dy = (x1 - x2) / 2, (y1 - y2) / 2
    x1p = cos_phi * half_dx + sin_phi * half_dy
    y1p = -sin_phi * half_dx + cos_phi * half_dy
    ratio = x1p * x1p / (rx * rx) + y1p * y1p / (ry * ry)
    if ratio > 1.0:
        scale = math.sqrt(ratio)
        rx, ry = rx * scale, ry * scale
    numerator = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    denominator = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    factor = math.sqrt(max(numerator, 0.0) / denominator)
    if large_arc == sweep:
        factor = -factor
    cxp = factor * rx * y1p / ry
    cyp = -factor * ry * x1p / rx
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2

    def arc_angle(ux: float, uy: float, vx: float, vy: float) -> float:
        cosine = max(
            -1.0, min(1.0, (ux * vx + uy * vy) / (math.hypot(ux, uy) * math.hypot(vx, vy)))
        )
        sign = 1.0 if ux * vy - uy * vx >= 0.0 else -1.0
        return sign * math.acos(cosine)

    theta = arc_angle(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    extent = arc_angle((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if sweep == 0.0 and extent > 0.0:
        extent -= 2.0 * math.pi
    elif sweep != 0.0 and extent < 0.0:
        extent += 2.0 * math.pi
    segments = max(1, math.ceil(abs(extent) / (math.pi / 2.0)))
    step = extent / segments
    handle = 4.0 / 3.0 * math.tan(step / 4.0)
    commands: list[Command] = []

    def placed(px: float, py: float) -> tuple[float, float]:
        return (cx + cos_phi * px - sin_phi * py, cy + sin_phi * px + cos_phi * py)

    angle = theta
    for _ in range(segments):
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        angle += step
        cos_b, sin_b = math.cos(angle), math.sin(angle)
        first = placed(rx * (cos_a - handle * sin_a), ry * (sin_a + handle * cos_a))
        second = placed(rx * (cos_b + handle * sin_b), ry * (sin_b - handle * cos_b))
        end = placed(rx * cos_b, ry * sin_b)
        commands.append(("C", (*first, *second, *end)))
    return tuple(commands)


# --- Phosphor helmet asset --------------------------------------------------------

HELMET_VIEWBOX: tuple[float, float] = (256.0, 256.0)

_HELMET_ASSET_PATH = Path(__file__).resolve().parents[3] / "assets" / "football_helmet.svg"
_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
ET.register_namespace("", _SVG_NAMESPACE)


@lru_cache(maxsize=1)
def _helmet_tree() -> ET.Element:
    """Return the parsed Phosphor helmet asset; callers must not mutate it."""

    root = ET.fromstring(_HELMET_ASSET_PATH.read_text(encoding="utf-8"))
    view_box = [float(value) for value in root.get("viewBox", "").replace(",", " ").split()]
    paths = list(root.iter(f"{{{_SVG_NAMESPACE}}}path"))
    if (
        not root.tag.endswith("}svg")
        or view_box != [0.0, 0.0, *HELMET_VIEWBOX]
        or len(paths) != 2
        or sum("opacity" in path.attrib for path in paths) != 1
        or any(not path.get("d") for path in paths)
    ):
        raise ValueError(
            f"{_HELMET_ASSET_PATH} is not the Phosphor duotone football helmet "
            "(expected one shaded shell path, one outline path, and a "
            f"0 0 {HELMET_VIEWBOX[0]:g} {HELMET_VIEWBOX[1]:g} viewBox)"
        )
    return root


def _helmet_elements(root: ET.Element) -> tuple[ET.Element, ET.Element]:
    """Return the asset's shaded shell path and its outline path from ``root``."""

    paths = list(root.iter(f"{{{_SVG_NAMESPACE}}}path"))
    shell = next((path for path in paths if "opacity" in path.attrib), paths[0])
    outline = next((path for path in paths if "opacity" not in path.attrib), paths[1])
    return shell, outline


@lru_cache(maxsize=1)
def _helmet_geometry() -> tuple[tuple[Command, ...], tuple[Command, ...]]:
    """Return the shell and outline path data compiled to absolute commands."""

    shell, outline = _helmet_elements(_helmet_tree())
    return _parse_svg_path(shell.get("d", "")), _parse_svg_path(outline.get("d", ""))


def helmet_parts(team: Team | None = None) -> tuple[VectorPart, ...]:
    """Return the Phosphor helmet parts: the shell and the outline with facemask.

    Without a team the shell and outline use the neutral charcoal/silver pair.
    """

    primary, secondary = team_colors(team)
    shell, outline = _helmet_geometry()
    return (VectorPart(shell, fill=primary), VectorPart(outline, fill=secondary))


FOOTBALL_VIEWBOX: tuple[float, float] = (48.0, 30.0)

_LEATHER = "#8A4B2A"
_LEATHER_DARK = "#5A331A"
_WHITE = "#FFFFFF"

_BALL: tuple[float, float, float, float] = (24.0, 15.0, 22.0, 12.5)

_STRIPES: tuple[tuple[Command, ...], ...] = (
    (("M", (10.0, 7.5)), ("C", (7.2, 11.0, 7.2, 19.0, 10.0, 22.5))),
    (("M", (38.0, 7.5)), ("C", (40.8, 11.0, 40.8, 19.0, 38.0, 22.5))),
)

_LACES: tuple[tuple[Command, ...], ...] = (
    (("M", (17.0, 15.0)), ("L", (31.0, 15.0))),
    (("M", (20.0, 12.2)), ("L", (20.0, 17.8))),
    (("M", (24.0, 12.0)), ("L", (24.0, 18.0))),
    (("M", (28.0, 12.2)), ("L", (28.0, 17.8))),
)


def football_parts() -> tuple[VectorPart, ...]:
    """Return the generic football parts: ball, end stripes, and laces."""

    parts = [
        VectorPart(
            _ellipse_commands(*_BALL),
            fill=_LEATHER,
            stroke=_LEATHER_DARK,
            stroke_width=1.5,
        )
    ]
    parts.extend(VectorPart(stripe, stroke=_WHITE, stroke_width=3.0) for stripe in _STRIPES)
    parts.extend(VectorPart(lace, stroke=_WHITE, stroke_width=2.0) for lace in _LACES)
    return tuple(parts)


# --- SVG rendering ----------------------------------------------------------------


def _path_d(commands: Sequence[Command]) -> str:
    return " ".join(" ".join([op, *(f"{value:g}" for value in coords)]) for op, coords in commands)


def _part_markup(part: VectorPart) -> str:
    attributes = [f'd="{_path_d(part.commands)}"']
    attributes.append(f'fill="{part.fill}"' if part.fill is not None else 'fill="none"')
    if part.stroke is not None:
        attributes.append(f'stroke="{part.stroke}"')
        attributes.append(f'stroke-width="{part.stroke_width:g}"')
        attributes.append('stroke-linecap="round"')
        attributes.append('stroke-linejoin="round"')
    return f"<path {' '.join(attributes)}/>"


def _graphic_svg(
    parts: Sequence[VectorPart],
    viewbox: tuple[float, float],
    width: int,
    label: str,
) -> str:
    height = round(width * viewbox[1] / viewbox[0])
    body = "".join(_part_markup(part) for part in parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {viewbox[0]:g} {viewbox[1]:g}" '
        f'width="{width}" height="{height}" role="img" aria-label="{label}">{body}</svg>'
    )


def helmet_svg(team: Team | None = None, *, width: int = 120) -> str:
    """Return the Phosphor football helmet as an SVG string tinted with ``team``'s colors.

    The asset's markup is inlined with the original path data and proportions
    untouched: only the colors change, with the shell filled from the primary
    color and the outline and facemask from the secondary color.
    """

    primary, secondary = team_colors(team)
    label = "Generic football helmet"
    if team is not None:
        label = f"{label} in {TEAM_DISPLAY_NAMES[team]} colors"
    root = copy.deepcopy(_helmet_tree())
    root.attrib.pop("fill", None)
    height = round(width * HELMET_VIEWBOX[1] / HELMET_VIEWBOX[0])
    root.set("width", str(width))
    root.set("height", str(height))
    root.set("role", "img")
    root.set("aria-label", label)
    for element, color in zip(_helmet_elements(root), (primary, secondary), strict=True):
        element.set("fill", color)
        element.attrib.pop("opacity", None)
    return ET.tostring(root, encoding="unicode")


def football_svg(*, width: int = 26) -> str:
    """Return a generic football as an SVG string."""

    return _graphic_svg(football_parts(), FOOTBALL_VIEWBOX, width, "Football")
