"""Generate a sample document collection into data/docs/ (run from /home/claude).

    python -m docintel.make_sample_docs

Numbers are shared across documents via the FACTS dict so cross-document questions work.
"""
import os
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
                                PageBreak, BaseDocTemplate, PageTemplate, Frame, FrameBreak,
                                KeepTogether)

from docintel.schemas import DOCS_DIR

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
EFFICIENCY = {"Q1": 78, "Q2": 92, "Q3": 85, "Q4": 81}          # percent, overall equipment effectiveness
OUTPUT = {"Q1": 118000, "Q2": 139000, "Q3": 128000, "Q4": 121000}  # units
SCRAP = {"Q1": 3.1, "Q2": 2.2, "Q3": 2.6, "Q4": 3.0}              # percent
DOWNTIME = {  # hours by cause per quarter
    "Mechanical failure": [42, 28, 55, 71],
    "Electrical fault": [18, 14, 20, 22],
    "Material shortage": [12, 9, 31, 38],
    "Changeover": [30, 28, 31, 33],
    "Planned maintenance": [40, 40, 40, 40],
}
DOWNTIME_TOTAL = [sum(v[i] for v in DOWNTIME.values()) for i in range(4)]  # 142, 119, 177, 204

_ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=_ss["Heading1"], fontName="Helvetica-Bold", fontSize=20, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=_ss["Heading2"], fontName="Helvetica-Bold", fontSize=14, spaceBefore=10, spaceAfter=6)
BODY = ParagraphStyle("B", parent=_ss["BodyText"], fontName="Helvetica", fontSize=10, leading=14)
CAP = ParagraphStyle("C", parent=BODY, fontName="Helvetica-Oblique", fontSize=9, textColor=colors.HexColor("#444444"))
BODY_SM = ParagraphStyle("BS", parent=BODY, fontSize=9, leading=12)


def _table(data, widths=None):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE6F0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    return t


def _chart_png(kind):
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    if kind == "efficiency":
        vals = [EFFICIENCY[q] for q in QUARTERS]
        bars = ax.bar(QUARTERS, vals, color="#3B7DD8")
        ax.plot(QUARTERS, vals, color="#D8503B", marker="o", label="Trend")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v}%", ha="center", fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Production efficiency (%)")
        ax.set_xlabel("Quarter, 2025")
        ax.set_title("Production Efficiency by Quarter, 2025")
        ax.legend(loc="lower right")
    else:
        bottom = np.zeros(4)
        cmap = ["#3B7DD8", "#D8503B", "#E0A526", "#5BA55B", "#8C6BB1"]
        for (cause, v), c in zip(DOWNTIME.items(), cmap):
            ax.bar(QUARTERS, v, bottom=bottom, label=cause, color=c)
            bottom += np.array(v)
        for i, tot in enumerate(DOWNTIME_TOTAL):
            ax.text(i, tot + 3, str(tot), ha="center", fontsize=9)
        ax.set_ylim(0, 240)
        ax.set_ylabel("Downtime (hours)")
        ax.set_xlabel("Quarter, 2025")
        ax.set_title("Downtime Hours by Cause, 2025")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def make_factory_report(path):
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2.2 * cm, bottomMargin=2.2 * cm, title="Factory Report 2025")
    P = lambda t, s=BODY: Paragraph(t, s)
    s = []
    s += [P("Factory Operations Report 2025", H1),
          P("1. Executive Summary", H2),
          P("This report reviews the 2025 operating performance of the Northfield assembly plant. "
            "Overall production efficiency, measured as overall equipment effectiveness (OEE), peaked "
            f"at {EFFICIENCY['Q2']}% in Q2 and ended the year at {EFFICIENCY['Q4']}% in Q4. Total output "
            f"for the year was {sum(OUTPUT.values()):,} units. Management remains committed to the "
            "annual target of 85% average efficiency."),
          P("The plant operates three production lines across two shifts. Line 3, which produces the "
            "high-volume gearbox housing, accounts for roughly 45% of total output and therefore "
            "dominates the plant-level results. Safety performance was stable with no lost-time "
            "incidents recorded in the year."),
          P("2. Plant Overview", H2),
          P("Northfield employs 240 people, of whom 168 work directly on the production lines. "
            "Investment in 2025 focused on automation of the packaging area and on a new quality "
            "inspection station at the end of Line 2. The product mix was unchanged from 2024, so "
            "differences between quarters are attributable to operating conditions rather than mix."),
          PageBreak(),
          P("3. Quarterly Production Efficiency", H2),
          P("Table 1 gives the quarterly efficiency, output and scrap rate. Efficiency is the product of "
            "availability, performance and quality as defined in the plant KPI handbook."),
          Spacer(1, 6)]
    rows = [["Quarter", "Efficiency (%)", "Output (units)", "Scrap rate (%)"]]
    for q in QUARTERS:
        rows.append([q, str(EFFICIENCY[q]), f"{OUTPUT[q]:,}", f"{SCRAP[q]}"])
    s += [_table(rows, [3 * cm, 4 * cm, 4 * cm, 4 * cm]), Spacer(1, 4),
          P("Table 1: Quarterly production efficiency, output and scrap rate, 2025.", CAP), Spacer(1, 12),
          Image(_chart_png("efficiency"), width=14 * cm, height=14 * cm * 3.2 / 6),
          Spacer(1, 4),
          P("Figure 1: Production efficiency by quarter (%), bar chart with trend line.", CAP),
          P("The chart shows efficiency rising from 78% in Q1 to a peak of 92% in Q2, followed by a "
            "steady decline through Q3 (85%) and Q4 (81%)."),
          PageBreak(),
          P("4. Analysis of the Q2 to Q4 Decline", H2),
          P(f"Efficiency fell by {EFFICIENCY['Q2'] - EFFICIENCY['Q4']} percentage points between Q2 "
            f"({EFFICIENCY['Q2']}%) and Q4 ({EFFICIENCY['Q4']}%), a relative decline of about 12.0%. "
            "Three main reasons explain the change:"),
          P("<b>Reason 1 - Rising mechanical failures on Line 3.</b> Ageing spindle bearings caused "
            "repeated unplanned stops. Mechanical failure downtime grew from 28 hours in Q2 to 71 hours "
            "in Q4 (see the Maintenance Summary 2025)."),
          P("<b>Reason 2 - Raw material shortages.</b> A supplier delay for cast aluminium blanks "
            "starved the lines in the second half. Material shortage downtime rose from 9 hours in Q2 "
            "to 38 hours in Q4."),
          P("<b>Reason 3 - Staffing turnover and training.</b> Eleven experienced operators left in "
            "Q3 and were replaced by trainees, which lowered line speed and raised the scrap rate from "
            "2.2% in Q2 to 3.0% in Q4."),
          P("5. Outlook and Recommendations", H2),
          P("For 2026 the plant will replace the Line 3 spindle bearings in the Q1 planned shutdown, "
            "dual-source the aluminium blanks, and introduce a structured four-week onboarding program. "
            "These measures target a return to at least 88% efficiency by Q2 2026.")]
    doc.build(s)


def make_maintenance_summary(path):
    W, H = A4
    m = 1.8 * cm
    gap = 0.8 * cm
    colw = (W - 2 * m - gap) / 2
    top = H - m
    full = Frame(m, top - 4.2 * cm, W - 2 * m, 4.2 * cm, id="title", leftPadding=0, rightPadding=0)
    body_h = H - 2 * m - 4.2 * cm
    l1 = Frame(m, m, colw, body_h, id="l1", leftPadding=0, rightPadding=4)
    r1 = Frame(m + colw + gap, m, colw, body_h, id="r1", leftPadding=4, rightPadding=0)
    l2 = Frame(m, m, colw, H - 2 * m, id="l2", leftPadding=0, rightPadding=4)
    r2 = Frame(m + colw + gap, m, colw, H - 2 * m, id="r2", leftPadding=4, rightPadding=0)
    doc = BaseDocTemplate(path, pagesize=A4, title="Maintenance Summary 2025")
    doc.addPageTemplates([PageTemplate("first", [full, l1, r1], autoNextPageTemplate="rest"),
                          PageTemplate("rest", [l2, r2])])
    P = lambda t, s=BODY_SM: Paragraph(t, s)
    s = [P("Maintenance Summary 2025", H1),
         P("Northfield assembly plant - Maintenance Department annual review", BODY), FrameBreak(),
         P("1. Scope", H2),
         P("This summary records equipment downtime at Northfield for 2025, grouped by cause and "
           "quarter. It complements the Factory Operations Report 2025, which discusses the effect of "
           "downtime on production efficiency."),
         P("Downtime hours are counted across all three production lines. Planned maintenance is a "
           "fixed 40 hours per quarter and is included in the totals."),
         P("2. Downtime by Cause", H2)]
    rows = [["Cause"] + QUARTERS]
    for cause, v in DOWNTIME.items():
        rows.append([cause] + [str(x) for x in v])
    rows.append(["Total"] + [str(x) for x in DOWNTIME_TOTAL])
    s += [_table(rows, [3.4 * cm] + [1.35 * cm] * 4), Spacer(1, 3),
          P("Table 2: Downtime hours by cause and quarter, 2025.", CAP),
          Spacer(1, 6),
          P("Total downtime was lowest in Q2 at "
            f"{DOWNTIME_TOTAL[1]} hours and highest in Q4 at {DOWNTIME_TOTAL[3]} hours, an increase of "
            f"{DOWNTIME_TOTAL[3] - DOWNTIME_TOTAL[1]} hours."),
         P("3. Findings", H2),
         P("Mechanical failure is the largest single cause of unplanned downtime, increasing from "
           "28 hours in Q2 to 71 hours in Q4. Root-cause analysis traced most events to worn spindle "
           "bearings on Line 3."),
         FrameBreak(),
         Image(_chart_png("downtime"), width=colw - 6, height=(colw - 6) * 3.2 / 6),
         Spacer(1, 3),
         P("Figure 2: Downtime hours by cause per quarter, stacked bars.", CAP),
         P("4. Actions Taken", H2),
         P("Bearing inspections were moved from quarterly to monthly in Q3. A spare spindle assembly "
           "was purchased in Q4 to shorten repair time. Electrical faults stayed roughly flat at "
           "14 to 22 hours per quarter."),
         P("5. Recommendations", H2),
         P("Replace all Line 3 spindle bearings during the planned Q1 2026 shutdown, hold a minimum "
           "safety stock of cast aluminium blanks, and review the planned maintenance allowance of 40 "
           "hours per quarter once bearing replacement is complete.")]
    doc.build(s)


def make_scanned_audit(path, tmp_png):
    import pymupdf as fitz
    from PIL import Image as PI, ImageFilter
    clean = tmp_png + ".clean.pdf"
    doc = SimpleDocTemplate(clean, pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm,
                            topMargin=2.5 * cm, bottomMargin=2.5 * cm)
    P = lambda t, s=BODY: Paragraph(t, s)
    rows = [["Line", "Defect rate (%)", "Result"], ["Line 1", "1.4", "Pass"],
            ["Line 2", "1.9", "Pass"], ["Line 3", "3.0", "Fail"]]
    doc.build([P("Quality Audit Report - Northfield Plant", H1),
               P("Audit date: 14 November 2025. Auditor: R. Mehta.", BODY),
               P("1. Audit Findings", H2),
               P("The Q4 2025 internal quality audit found an overall defect rate of 3.0% on Line 3, "
                 "above the 2.5% acceptance limit. Lines 1 and 2 met the limit."),
               _table(rows, [3 * cm, 4 * cm, 3 * cm]), Spacer(1, 10),
               P("2. Corrective Actions", H2),
               P("The audit raised three non-conformances: worn spindle bearings on Line 3, "
                 "incomplete operator training records for new hires, and missing calibration labels "
                 "on two gauges. Corrective actions are due by 31 January 2026.")])
    src = fitz.open(clean)
    pix = src[0].get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), alpha=False)
    img = PI.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
    img = img.filter(ImageFilter.GaussianBlur(1.4)).rotate(2.5, expand=True, fillcolor=255, resample=PI.BICUBIC)
    rng = np.random.default_rng(0)
    arr = np.asarray(img).astype(float) + rng.normal(0, 6, (img.height, img.width))
    img = PI.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    img.save(tmp_png)
    out = fitz.open()
    pg = out.new_page(width=595, height=842)
    pg.insert_image(pg.rect, filename=tmp_png)
    out.save(path)
    os.remove(tmp_png)
    os.remove(clean)


FACTS_MD = f"""# Sample document ground truth

Generated by `docintel/make_sample_docs.py`. All numbers below are authoritative.

## factory_report_2025.pdf (3 pages)

| Fact | Value | Location |
|---|---|---|
| Efficiency Q1 / Q2 / Q3 / Q4 | 78 / 92 / 85 / 81 (%) | p2, Table 1 and Figure 1 (section 3 Quarterly Production Efficiency) |
| Output Q1..Q4 (units) | 118,000 / 139,000 / 128,000 / 121,000 | p2, Table 1 |
| Scrap rate Q1..Q4 (%) | 3.1 / 2.2 / 2.6 / 3.0 | p2, Table 1 |
| Total annual output | 506,000 units | p1, section 1 Executive Summary (computed from table) |
| Annual efficiency target | 85% average | p1, Executive Summary |
| Peak efficiency | 92% in Q2 | p1 and p2 |
| Q2 to Q4 decline | 11 percentage points (92 to 81), about 12.0% relative (11/92 = 11.96%) | p3, section 4 |
| Three reasons for decline | (1) rising mechanical failures on Line 3 (spindle bearings), (2) raw material shortages (aluminium blanks), (3) staffing turnover/training (11 operators left in Q3) | p3, section 4 Analysis of the Q2 to Q4 Decline |
| Employees | 240 total, 168 on lines | p1, section 2 |
| Line 3 share of output | about 45% | p1, section 1 |
| Target for 2026 | at least 88% by Q2 2026 | p3, section 5 |

## maintenance_summary_2025.pdf (1 page, two-column layout)

Downtime hours by cause (p1, Table 2, section 2 Downtime by Cause; Figure 2 chart in the right column):

| Cause | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| Mechanical failure | 42 | 28 | 55 | 71 |
| Electrical fault | 18 | 14 | 20 | 22 |
| Material shortage | 12 | 9 | 31 | 38 |
| Changeover | 30 | 28 | 31 | 33 |
| Planned maintenance | 40 | 40 | 40 | 40 |
| Total | {DOWNTIME_TOTAL[0]} | {DOWNTIME_TOTAL[1]} | {DOWNTIME_TOTAL[2]} | {DOWNTIME_TOTAL[3]} |

- Lowest total downtime: Q2 ({DOWNTIME_TOTAL[1]} h); highest: Q4 ({DOWNTIME_TOTAL[3]} h); increase {DOWNTIME_TOTAL[3] - DOWNTIME_TOTAL[1]} h.
- Mechanical failure Q2 to Q4: 28 to 71 h (+43 h, +153.6%). Material shortage Q2 to Q4: 9 to 38 h.
- Planned maintenance is fixed at 40 h per quarter.
- Section 3 Findings: worn spindle bearings on Line 3.

## scanned_quality_audit.pdf (1 page, image-only, blurred and rotated; needs OCR)

- Audit date 14 November 2025, auditor R. Mehta.
- Defect rate Line 1 1.4%, Line 2 1.9%, Line 3 3.0% (Fail; limit 2.5%).
- Three non-conformances: worn spindle bearings on Line 3, incomplete operator training records, missing gauge calibration labels. Corrective actions due 31 January 2026.

## Cross-document questions

- Q4 mechanical failure downtime (71 h, maintenance summary) explains reason 1 of the Q2-Q4 efficiency decline (factory report p3).
- Q4 scrap rate 3.0% (factory report Table 1) equals the Line 3 audit defect rate 3.0% (scanned audit).
- Efficiency drop Q2 to Q4 is 11 points while total downtime rose {DOWNTIME_TOTAL[3] - DOWNTIME_TOTAL[1]} h.
- Unanswerable (should be refused): any 2024 figures, Q1 2026 results, or the plant's revenue.
"""


def main(docs_dir=DOCS_DIR):
    os.makedirs(docs_dir, exist_ok=True)
    make_factory_report(os.path.join(docs_dir, "factory_report_2025.pdf"))
    make_maintenance_summary(os.path.join(docs_dir, "maintenance_summary_2025.pdf"))
    make_scanned_audit(os.path.join(docs_dir, "scanned_quality_audit.pdf"),
                       os.path.join(docs_dir, "_scan_tmp.png"))
    with open(os.path.join(docs_dir, "README_sample_facts.md"), "w") as f:
        f.write(FACTS_MD)
    print("Wrote sample docs to", docs_dir)


if __name__ == "__main__":
    main()
