"""Render a finished :class:`Report` as a readable PDF.

Like ``ui/render.py`` this module only places already-validated contract data on
a page. It never recomputes a statistic, reorders evidence, or hides a warning.
Evidence IDs are opaque hashes, so the body cites them as numbered markers
(``[1]``) that point at the evidence appendix at the end of the document.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import FILL_NON_ZERO, Canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

from opponent_tendencies.models import (
    TEAM_DISPLAY_NAMES,
    Agreement,
    Conflict,
    EvidenceRef,
    Gap,
    Report,
    StatTable,
    Tendency,
    ValidationIssue,
)
from opponent_tendencies.ui.graphics import (
    FOOTBALL_VIEWBOX,
    HELMET_VIEWBOX,
    VectorPart,
    football_parts,
    helmet_parts,
    team_accent,
)
from opponent_tendencies.ui.render import agent_label, analyzed_scrimmage_plays, format_stat_value

__all__ = ["build_report_pdf", "report_pdf_filename"]

_INK = colors.HexColor("#1f2933")
_MUTED = colors.HexColor("#616e7c")
_ACCENT = colors.HexColor("#1d4e89")
_RULE = colors.HexColor("#cbd2d9")
_BAND = colors.HexColor("#f0f4f8")
_WARN_BG = colors.HexColor("#fff4d6")
_WARN_EDGE = colors.HexColor("#d9a400")

_MARGIN = 0.8 * inch


def report_pdf_filename(report: Report) -> str:
    weeks = "-".join(str(week) for week in report.weeks)
    return f"{report.opponent.value}_{report.season}_weeks_{weeks}_scouting_report.pdf"


def build_report_pdf(report: Report) -> bytes:
    """Return the PDF bytes for ``report``."""

    buffer = BytesIO()
    team_name = TEAM_DISPLAY_NAMES[report.opponent]
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN,
        title=f"{team_name} Opponent Scouting Report",
        author="Opponent Tendencies Analyst",
    )
    accent = colors.HexColor(team_accent(report.opponent))
    styles = _styles(accent)
    refs = {item.evidence_id: number for number, item in enumerate(report.evidence, start=1)}

    story: list[Flowable] = []
    story += _header(report, styles)
    story += _overview(report, styles)
    story += _tendencies(report, styles, refs)
    story += _observations(report, styles, refs)
    story += _synthesis(report, styles)
    story += _statistics(report, styles, accent)
    story += _data_used(report, styles)
    story += _warnings(report, styles)
    story += _evidence_appendix(report, styles)

    footer = _footer(f"{report.opponent.value} · {report.season} · report {report.report_id[:12]}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


# --- styles ----------------------------------------------------------------


def _styles(accent: colors.Color = _ACCENT) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["Normal"]
    body = ParagraphStyle(
        "body", parent=base, fontName="Helvetica", fontSize=10, leading=14.5, textColor=_INK
    )
    return {
        "body": body,
        "title": ParagraphStyle(
            "title",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=accent,
            spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=body, fontSize=11, leading=15, textColor=_MUTED, spaceAfter=10
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=accent,
            spaceBefore=16,
            spaceAfter=5,
            keepWithNext=1,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            spaceBefore=8,
            spaceAfter=3,
            keepWithNext=1,
        ),
        "item": ParagraphStyle("item", parent=body, spaceAfter=1),
        "meta": ParagraphStyle(
            "meta", parent=body, fontSize=8.5, leading=11.5, textColor=_MUTED, spaceAfter=6
        ),
        "cell": ParagraphStyle("cell", parent=body, fontSize=9, leading=12),
        "cellhead": ParagraphStyle(
            "cellhead",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=colors.white,
        ),
        "url": ParagraphStyle(
            "url",
            parent=body,
            fontSize=8,
            leading=10.5,
            textColor=_MUTED,
            wordWrap="CJK",
            alignment=TA_LEFT,
        ),
    }


def _footer(label: str) -> Callable[[Canvas, SimpleDocTemplate], None]:
    def draw(canvas: Canvas, doc: SimpleDocTemplate) -> None:
        canvas.saveState()
        canvas.setStrokeColor(_RULE)
        canvas.line(_MARGIN, 0.6 * inch, LETTER[0] - _MARGIN, 0.6 * inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(_MUTED)
        canvas.drawString(_MARGIN, 0.42 * inch, _clean(label))
        canvas.drawRightString(LETTER[0] - _MARGIN, 0.42 * inch, f"Page {doc.page}")
        canvas.restoreState()

    return draw


# --- text helpers ------------------------------------------------------------


def _clean(text: str) -> str:
    """Replace characters the built-in Helvetica (cp1252) cannot draw."""

    return text.encode("cp1252", "replace").decode("cp1252")


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(_clean(text)), style)


def _rich(markup: str, style: ParagraphStyle) -> Paragraph:
    """Paragraph from markup the caller has already escaped."""

    return Paragraph(markup, style)


def _words(value: str) -> str:
    return value.replace("_", " ")


def _citations(ids: Sequence[str], refs: dict[str, int]) -> str:
    numbers = sorted(refs[item] for item in ids if item in refs)
    if not numbers:
        return "no evidence attached"
    return " ".join(f"[{number}]" for number in numbers)


def _bullets(items: Sequence[Paragraph], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [ListItem(item, leftIndent=14, bulletColor=_ACCENT) for item in items],
        bulletType="bullet",
        bulletFontSize=7,
        leftIndent=14,
        start="•",
    )


def _callout(text: str, style: ParagraphStyle) -> Table:
    table = Table([[_p(text, style)]], colWidths=[LETTER[0] - 2 * _MARGIN])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _WARN_BG),
                ("LINEBEFORE", (0, 0), (0, -1), 3, _WARN_EDGE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


# --- vector graphics -----------------------------------------------------------


class _VectorGraphic(Flowable):
    """Draw :class:`VectorPart` geometry (y-down view coordinates) on the canvas.

    The geometry is shared with the Streamlit SVG rendering, so both surfaces
    draw the same helmet and football shapes. Filling uses SVG's nonzero
    winding rule, which the helmet outline needs for its enclosed holes.
    """

    def __init__(
        self,
        parts: Sequence[VectorPart],
        viewbox: tuple[float, float],
        width: float,
    ) -> None:
        super().__init__()
        self._parts = tuple(parts)
        self._viewbox = viewbox
        self._width = width
        self._height = width * viewbox[1] / viewbox[0]
        self.width = width
        self.height = self._height

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        return (self._width, self._height)

    def draw(self) -> None:
        canvas = self.canv
        scale = self._width / self._viewbox[0]
        canvas.saveState()
        for part in self._parts:
            path = canvas.beginPath()
            for op, coords in part.commands:
                points = [
                    value * scale if index % 2 == 0 else (self._viewbox[1] - value) * scale
                    for index, value in enumerate(coords)
                ]
                if op == "M":
                    path.moveTo(*points)
                elif op == "L":
                    path.lineTo(*points)
                elif op == "C":
                    path.curveTo(*points)
                elif op == "Z":
                    path.close()
            if part.fill is not None:
                canvas.setFillColor(colors.HexColor(part.fill))
            if part.stroke is not None:
                canvas.setStrokeColor(colors.HexColor(part.stroke))
                canvas.setLineWidth(part.stroke_width * scale)
                canvas.setLineCap(1)
                canvas.setLineJoin(1)
            canvas.drawPath(
                path,
                stroke=part.stroke is not None,
                fill=part.fill is not None,
                fillMode=FILL_NON_ZERO,
            )
        canvas.restoreState()


def _crest(report: Report) -> Flowable:
    """Stack the opponent's helmet over a football as the report's graphic mark."""

    helmet = _VectorGraphic(helmet_parts(report.opponent), HELMET_VIEWBOX, 0.78 * inch)
    ball = _VectorGraphic(football_parts(), FOOTBALL_VIEWBOX, 0.42 * inch)
    stack = Table([[helmet], [ball]], colWidths=[1.05 * inch])
    stack.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return stack


# --- sections ------------------------------------------------------------------


def _header(report: Report, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    weeks = ", ".join(str(week) for week in report.weeks)
    games = report.games
    accent = colors.HexColor(team_accent(report.opponent))
    titles = [
        _p(f"{TEAM_DISPLAY_NAMES[report.opponent]}", styles["title"]),
        _p("Opponent Scouting Report", styles["subtitle"]),
    ]
    masthead = Table(
        [[_crest(report), titles]],
        colWidths=[1.1 * inch, LETTER[0] - 2 * _MARGIN - 1.1 * inch],
    )
    masthead.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    masthead.hAlign = "LEFT"
    flow: list[Flowable] = [
        masthead,
        HRFlowable(width="100%", thickness=1.2, color=accent, spaceAfter=8),
    ]
    facts = [
        ["Season", str(report.season)],
        ["Weeks analyzed", weeks],
        ["Games", str(len(games))],
        ["Generated", report.generated_at.strftime("%B %d, %Y")],
    ]
    table = Table(
        [[_p(label, styles["meta"]), _p(value, styles["cell"])] for label, value in facts],
        colWidths=[1.3 * inch, 4.5 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    table.hAlign = "LEFT"
    flow.append(table)
    flow.append(Spacer(1, 8))
    if report.is_partial:
        names = ", ".join(agent_label(agent) for agent in report.unavailable_sections)
        flow.append(
            _callout(
                f"Partial report: {names} unavailable. Statistics and the completed sections "
                "are shown; review the warnings before relying on it.",
                styles["body"],
            )
        )
    return flow


def _overview(report: Report, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    flow: list[Flowable] = [_p("Overview", styles["h1"]), _p(report.overview, styles["body"])]
    if report.weak_evidence:
        flow.append(Spacer(1, 6))
        topics = "; ".join(item.topic for item in report.weak_evidence)
        flow.append(_callout(f"Weak evidence, human review recommended: {topics}", styles["body"]))
    return flow


def _tendencies(
    report: Report, styles: dict[str, ParagraphStyle], refs: dict[str, int]
) -> list[Flowable]:
    if not report.tendencies:
        return []
    flow: list[Flowable] = [_p("Key Tendencies", styles["h1"])]
    for tendency in report.tendencies:
        flow.append(KeepTogether([_finding_paragraph(tendency, styles, refs)]))
    return flow


def _finding_paragraph(
    finding: Tendency, styles: dict[str, ParagraphStyle], refs: dict[str, int]
) -> Flowable:
    detail = f"{finding.confidence.value.lower()} confidence · {_words(finding.direction.value)}"
    markup = (
        f"{escape(_clean(finding.statement))}<br/>"
        f'<font size="8.5" color="#616e7c">{escape(detail)} · weeks '
        f"{', '.join(str(week) for week in finding.weeks)} · evidence "
        f"{escape(_citations(finding.evidence_ids, refs))}</font>"
    )
    return _bullets([_rich(markup, styles["item"])], styles["item"])


def _observations(
    report: Report, styles: dict[str, ParagraphStyle], refs: dict[str, int]
) -> list[Flowable]:
    if not report.observations:
        return []
    flow: list[Flowable] = [_p("Reporting and Coach Context", styles["h1"])]
    for observation in report.observations:
        context = " · coach context, not proof" if observation.treat_as_context else ""
        detail = (
            f"{observation.confidence.value.lower()} confidence · "
            f"{_words(observation.category.value)}{context} · evidence "
            f"{_citations(observation.evidence_ids, refs)}"
        )
        markup = (
            f"{escape(_clean(observation.statement))}<br/>"
            f'<font size="8.5" color="#616e7c">{escape(detail)}</font>'
        )
        flow.append(KeepTogether([_bullets([_rich(markup, styles["item"])], styles["item"])]))
    return flow


def _synthesis(report: Report, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    flow: list[Flowable] = []
    if report.agreements:
        flow.append(_p("Where Sources Agree", styles["h1"]))
        flow += [_agreement(item, styles) for item in report.agreements]
    if report.conflicts:
        flow.append(_p("Conflicts", styles["h1"]))
        flow += [_conflict(item, styles) for item in report.conflicts]
    if report.gaps:
        flow.append(_p("Gaps", styles["h1"]))
        flow += [_gap(item, styles) for item in report.gaps]
    return flow


def _agreement(item: Agreement, styles: dict[str, ParagraphStyle]) -> Flowable:
    sources = ", ".join(agent_label(agent) for agent in item.agents)
    markup = (
        f"<b>{escape(_clean(item.topic))}.</b> {escape(_clean(item.statement))} "
        f'<font size="8.5" color="#616e7c">({escape(sources)})</font>'
    )
    return _bullets([_rich(markup, styles["item"])], styles["item"])


def _conflict(item: Conflict, styles: dict[str, ParagraphStyle]) -> Flowable:
    parts: list[Flowable] = [
        _bullets(
            [
                _rich(
                    f"<b>{escape(_clean(item.topic))}.</b> {escape(_clean(item.description))}",
                    styles["item"],
                )
            ],
            styles["item"],
        )
    ]
    positions = [
        _rich(
            f"<b>{escape(agent_label(pos.agent))}:</b> {escape(_clean(pos.statement))}",
            styles["cell"],
        )
        for pos in item.positions
    ]
    if item.resolution:
        positions.append(_p(f"Resolution: {item.resolution}", styles["cell"]))
    indent = Table([[_bullets(positions, styles["cell"])]], colWidths=[LETTER[0] - 2 * _MARGIN])
    indent.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 24)]))
    parts.append(indent)
    return KeepTogether(parts)


def _gap(item: Gap, styles: dict[str, ParagraphStyle]) -> Flowable:
    markup = (
        f"<b>{escape(_clean(item.topic))}.</b> {escape(_clean(item.description))} "
        f'<font size="8.5" color="#616e7c">({escape(_words(item.reason.value))})</font>'
    )
    return _bullets([_rich(markup, styles["item"])], styles["item"])


def _statistics(
    report: Report, styles: dict[str, ParagraphStyle], accent: colors.Color
) -> list[Flowable]:
    if not report.stat_tables:
        return []
    flow: list[Flowable] = [
        PageBreak(),
        _p("Statistics", styles["h1"]),
        _p(
            "Computed deterministically from play-by-play data; the model only interprets "
            "these figures. Rows below the minimum sample size are flagged.",
            styles["meta"],
        ),
    ]
    flow += [_stat_table(table, styles, accent) for table in report.stat_tables]
    return flow


def _stat_table(
    table: StatTable, styles: dict[str, ParagraphStyle], accent: colors.Color
) -> Flowable:
    title = table.title
    if table.dimension_value:
        title = f"{title}: {table.dimension_value}"
    heading = _p(f"{title}  (n = {table.n_plays})", styles["h2"])

    rows: list[list[Paragraph]] = [
        [_p(name, styles["cellhead"]) for name in ("Label", "Value", "n", "Notes")]
    ]
    for row in table.rows:
        note = "; ".join(row.notes) or ("suppressed" if row.suppressed else "")
        rows.append(
            [
                _p(row.label, styles["cell"]),
                _p(format_stat_value(row), styles["cell"]),
                _p(str(row.n), styles["cell"]),
                _p(note, styles["cell"]),
            ]
        )
    grid = Table(
        rows,
        colWidths=[2.5 * inch, 0.9 * inch, 0.6 * inch, 2.9 * inch],
        repeatRows=1,
    )
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), accent),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, _RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    style += [("BACKGROUND", (0, i), (-1, i), _BAND) for i in range(2, len(rows), 2)]
    grid.setStyle(TableStyle(style))

    extra: list[Flowable] = []
    if table.suppressed:
        extra.append(_p("Suppressed: below the minimum sample size.", styles["meta"]))
    extra += [_p(note, styles["meta"]) for note in table.notes]
    return KeepTogether([heading, grid, *extra, Spacer(1, 6)])


def _data_used(report: Report, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    usage = report.data_used
    figures: list[tuple[str, int]] = [
        ("Games", usage.games),
        ("Play-by-play records", usage.plays),
        ("Media items", usage.media_items),
        ("Quotes", usage.quotes),
        ("Sources", len(usage.source_ids)),
    ]
    scrimmage = analyzed_scrimmage_plays(report)
    if scrimmage is not None:
        figures.insert(2, ("Analyzed scrimmage plays", scrimmage))
    grid = Table(
        [
            [_p(str(value), styles["title"]) for _, value in figures],
            [_p(label, styles["meta"]) for label, _ in figures],
        ],
        colWidths=[(LETTER[0] - 2 * _MARGIN) / len(figures)] * len(figures),
    )
    grid.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _BAND)]))
    flow: list[Flowable] = [_p("Data Used", styles["h1"]), grid]
    if usage.source_ids:
        flow.append(Spacer(1, 4))
        flow.append(_p("Sources: " + ", ".join(usage.source_ids), styles["url"]))
    return flow


def _warnings(report: Report, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    flow: list[Flowable] = []
    if report.warnings:
        flow.append(_p("Warnings", styles["h1"]))
        flow += [_warning_line(issue, styles) for issue in report.warnings]
    unavailable = [output for output in report.agent_outputs if not output.is_available]
    if unavailable:
        flow.append(_p("Unavailable Sections", styles["h1"]))
        for output in unavailable:
            lines = [agent_label(output.agent), *output.limitations]
            flow.append(
                _bullets(
                    [_rich(f"<b>{escape(lines[0])}</b>", styles["item"])]
                    + [_p(line, styles["item"]) for line in lines[1:]],
                    styles["item"],
                )
            )
    return flow


def _warning_line(issue: ValidationIssue, styles: dict[str, ParagraphStyle]) -> Flowable:
    review = " · human review" if issue.requires_human_review else ""
    markup = (
        f"{escape(_clean(issue.message))} "
        f'<font size="8.5" color="#616e7c">({escape(issue.code)}, '
        f"{escape(issue.severity.value)}{review})</font>"
    )
    return _bullets([_rich(markup, styles["item"])], styles["item"])


def _evidence_appendix(report: Report, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    if not report.evidence:
        return []
    flow: list[Flowable] = [
        PageBreak(),
        _p("Evidence Appendix", styles["h1"]),
        _p("Numbers match the bracketed citations in the report body.", styles["meta"]),
    ]
    for number, item in enumerate(report.evidence, start=1):
        flow.append(KeepTogether(_evidence_entry(number, item, styles)))
    return flow


def _evidence_entry(
    number: int, item: EvidenceRef, styles: dict[str, ParagraphStyle]
) -> list[Flowable]:
    entry: list[Flowable] = [
        _rich(
            f"<b>[{number}]</b> {escape(_clean(item.label))} "
            f'<font size="8.5" color="#616e7c">({escape(item.kind.value)})</font>',
            styles["item"],
        )
    ]
    if item.source_url is not None:
        entry.append(_p(str(item.source_url), styles["url"]))
    entry.append(Spacer(1, 4))
    return entry
