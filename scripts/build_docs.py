"""Build the submission documents in docs/.

  docs/architecture.pdf             vector architecture diagram (also figures/architecture.png)
  docs/figures/dashboard_mockup.png conceptual clinician dashboard, drawn from real twin output
  docs/presentation.pptx            project presentation (export a PDF with scripts/export_pdf.ps1)

  uv sync --all-extras
  uv run python scripts/build_docs.py

Every number in the deck is read from results/strip_budget/summary.json, so the slides cannot drift
away from the experiment. The dashboard mock-up runs the real twin engine on a synthetic patient.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIGS = DOCS / "figures"
RESULTS = ROOT / "results" / "strip_budget"

REPO_URL = "github.com/SriRamkunamsetty/SITARAM_QISCET"
TEAM = "SITARAM"
MEMBERS = [
    ("Mohan Sriram Kunamsetty", "Team Leader", "QIS College of Engineering and Technology"),
    (
        "Mohana Lakshmi Srija Kunamsetty",
        "Team Member",
        "RISE Krishna Sai College of Engineering and Technology",
    ),
]

C = {
    "navy": "#0B2545",
    "teal": "#0F8B8D",
    "amber": "#E08E0B",
    "red": "#C8394B",
    "green": "#2E7D4F",
    "text": "#1F2D3D",
    "muted": "#5B6B7F",
    "bg": "#F4F7FA",
    "white": "#FFFFFF",
    "light_teal": "#DDF1F1",
    "light_amber": "#FCEFD6",
    "light_navy": "#E3E9F2",
}


# ====================================================================== architecture diagram
def _wrap(text: str, width_units: float, chars_per_unit: float = 1.3) -> str:
    return textwrap.fill(text, width=max(int((width_units - 2.4) * chars_per_unit), 12))


def _box(ax, x, y, w, h, title, body, edge, face="#FFFFFF", size=9.6):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.0,rounding_size=1.2",
            linewidth=1.9,
            edgecolor=edge,
            facecolor=face,
        )
    )
    ax.text(x + 1.3, y + h - 1.5, title, fontsize=11.5, fontweight="bold", color=edge, va="top")
    ax.text(
        x + 1.3, y + h - 4.7, _wrap(body, w, 0.98), fontsize=size, color=C["text"], va="top", linespacing=1.4
    )


def _arrow(ax, p0, p1, color, style="-|>", ls="-", lw=1.9):
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops={
            "arrowstyle": style,
            "color": color,
            "lw": lw,
            "linestyle": ls,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 14,
        },
    )


def build_architecture() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(C["white"])
    ax.set_xlim(0, 162)
    ax.set_ylim(0, 90)
    ax.axis("off")

    ax.text(3, 87.2, "Pratibimb: system architecture", fontsize=21, fontweight="bold", color=C["navy"])
    ax.text(
        3,
        83.4,
        "EHR + sparse fingersticks (or CGM) + wearables  ->  personalised physiological twin  ->  "
        "forecast, risk, and when to test",
        fontsize=11.5,
        color=C["muted"],
    )
    for x, label in [(3, "1  DATA IN"), (48, "2  TWIN CORE"), (120, "3  DECISIONS AND OUTPUTS")]:
        ax.text(x, 79.6, label, fontsize=10.5, fontweight="bold", color=C["teal"])
        ax.plot([x, x + (37 if x != 48 else 64)], [78.6, 78.6], color=C["teal"], lw=1.6)

    navy, teal, amber = C["navy"], C["teal"], C["amber"]
    # column 1: inputs
    col1 = [
        (64, "EHR summary", "Age, BMI, waist, HbA1c, fasting glucose, regimen (metformin, sulfonylurea)"),
        (48, "Wearable summary", "Resting heart rate and daily steps (optional)"),
        (32, "Meal, dose, activity log", "Carbohydrate, glycaemic index, food order, drug doses, exercise"),
        (16, "Glucose readings", "Fingerstick (SMBG) 1-4 a day, or CGM every 15 minutes"),
    ]
    for y, t, b in col1:
        _box(ax, 3, y, 37, 13, t, b, navy, C["light_navy"])

    # column 2: core
    _box(
        ax, 48, 62, 64, 15, "Prior builder",
        "Soft assignment to the 4 INSPIRED Indian T2D endotypes (SIDD, IROD, CIRDD, MARD) gives a "
        "log-normal prior on 6 patient parameters; wearables nudge insulin sensitivity.",
        teal, C["light_teal"],
    )  # fmt: skip
    _box(
        ax, 48, 43, 64, 15, "Mechanistic physiology model",
        "Extended Bergman minimal model: 10 states, 6 patient parameters. Gut absorption, "
        "sulfonylurea PK/PD, metformin, exercise, circadian rhythm, dawn effect.",
        teal, C["light_teal"],
    )  # fmt: skip
    _box(
        ax, 48, 24, 64, 15, "Data assimilation: particle-filter twin",
        "Joint state + parameter filter, 1,000 particles. Student-t likelihood, Liu-West resampling, "
        "slow drift of insulin sensitivity, basal glucose and SU response.",
        teal, C["light_teal"],
    )  # fmt: skip
    _box(
        ax, 48, 10, 64, 10, "Event-sourced audit log",
        "Every reading, meal and dose is an immutable event; late data triggers deterministic replay.",
        navy, C["light_navy"], size=9.4,
    )  # fmt: skip

    # column 3: outputs
    _box(
        ax, 120, 62, 37, 15, "Forecast with uncertainty",
        "Glucose bands (10-90% and 2.5-97.5%) and risk: P(<70), P(<54), P(>250 mg/dL).",
        navy, C["light_navy"],
    )  # fmt: skip
    _box(
        ax, 120, 43, 37, 15, "Active-sensing planner",
        "Picks the reading times with the most information about hypoglycaemia: 'test at 16:30'.",
        amber, C["light_amber"],
    )  # fmt: skip
    _box(
        ax, 120, 24, 37, 15, "Twin state vs prior",
        "Insulin sensitivity, basal glucose and SU response with posterior intervals.",
        navy, C["light_navy"],
    )  # fmt: skip
    _box(
        ax, 120, 10, 37, 10, "FastAPI service /v1",
        "API-key auth, audit export, erasure endpoint.",
        navy, C["light_navy"], size=9.4,
    )  # fmt: skip

    # arrows: inputs -> core
    _arrow(ax, (40, 70.5), (48, 72), navy)
    _arrow(ax, (40, 54.5), (48, 66), navy)
    _arrow(ax, (40, 38.5), (48, 34), navy)
    _arrow(ax, (40, 22.5), (48, 28.5), navy)
    # inside core
    _arrow(ax, (80, 62), (80, 58), teal)
    ax.text(81.5, 59.2, "prior over parameters", fontsize=8.2, style="italic", color=C["muted"], va="center")
    _arrow(ax, (80, 43), (80, 39), teal, style="<|-|>")
    ax.text(81.5, 40.7, "propagate states", fontsize=8.2, style="italic", color=C["muted"], va="center")
    _arrow(ax, (80, 24), (80, 20), navy, style="<|-|>")
    ax.text(81.5, 21.9, "replay", fontsize=8.2, style="italic", color=C["muted"], va="center")
    # core -> outputs via a bus
    ax.plot([112, 116], [31.5, 31.5], color=teal, lw=1.9)
    ax.plot([116, 116], [15, 69.5], color=teal, lw=1.9)
    for y in (69.5, 50.5, 31.5, 15):
        _arrow(ax, (116, y), (120, y), teal)
    # two-way loop back to the patient's readings
    amber_c = amber
    ax.plot([157, 160], [50.5, 50.5], color=amber_c, lw=2.2, ls="--")
    ax.plot([160, 160], [50.5, 4.5], color=amber_c, lw=2.2, ls="--")
    ax.plot([160, 21.5], [4.5, 4.5], color=amber_c, lw=2.2, ls="--")
    _arrow(ax, (21.5, 4.5), (21.5, 16), amber_c, ls="--", lw=2.2)
    ax.text(
        90, 5.7,
        "Two-way loop: the twin asks the patient to test at the chosen time; the new reading updates the twin",
        fontsize=9.6, color=amber_c, fontweight="bold", ha="center", va="bottom",
    )  # fmt: skip
    ax.text(
        3, 0.8,
        f"Team {TEAM}  |  Digital Twin Challenge 2026  |  Research prototype, not a medical device  |  Apache-2.0",
        fontsize=8.6, color=C["muted"],
    )  # fmt: skip

    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS / "architecture.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote docs/architecture.pdf and docs/figures/architecture.png")


# ====================================================================== dashboard mock-up
IST = timezone(timedelta(hours=5, minutes=30))
DAY0 = datetime(2026, 10, 12, 0, 0, tzinfo=IST)


def _iso(t_min: float) -> str:
    return (DAY0 + timedelta(minutes=float(t_min))).isoformat()


def _example_twin() -> dict:
    """Run the real twin on a synthetic patient and collect what a dashboard would show."""
    from pratibimb.cohort.virtual import (
        SMBG_SENSOR,
        generate_day,
        sample_virtual_patient,
        sensor_reading,
        simulate_truth,
    )
    from pratibimb.twin.engine import DigitalTwin

    rng = np.random.default_rng(7)
    vp = sample_virtual_patient(rng, "vp001", endotype="MARD", force_sulfonylurea=True)
    days = [generate_day(vp, d, rng) for d in range(3)]
    truth = simulate_truth(vp, days, rng)
    twin = DigitalTwin(vp.profile, DAY0, seed=1, n_particles=500)
    for day in days[:2]:
        for m in day.logged_events.meals:
            twin.add_meal(DAY0 + timedelta(minutes=m.t_min), m.carbs_g, m.glycemic_index, label=m.label)
        for d in day.logged_events.medications:
            twin.add_medication(DAY0 + timedelta(minutes=d.t_min), d.drug, d.dose_mg)

    readings: list[tuple[float, float]] = []
    now = 1440 + 390.0
    first = twin.plan_fingersticks(DAY0 + timedelta(minutes=now), 3)["recommended_times"][0]["time"]
    t_first = (datetime.fromisoformat(first) - DAY0).total_seconds() / 60
    value = sensor_reading(truth.at(t_first), SMBG_SENSOR, rng)
    twin.add_glucose(DAY0 + timedelta(minutes=t_first), round(value))
    readings.append((t_first, round(value)))

    now_dt = DAY0 + timedelta(minutes=t_first + 5)
    horizon_min = 16 * 60
    return {
        "profile": vp.profile,
        "endotype": vp.endotype,
        "readings": readings,
        "now": now_dt,
        "horizon_h": horizon_min // 60,
        "state": twin.state(now_dt),
        "forecast": twin.forecast(now_dt, horizon_min=horizon_min),
        "plan": twin.plan_fingersticks(now_dt, strips=2),
    }


def _hours(iso: str) -> float:
    return (datetime.fromisoformat(iso) - DAY0).total_seconds() / 3600.0 - 24.0


def build_dashboard() -> None:
    ex = _example_twin()
    st, fc, plan, prof = ex["state"], ex["forecast"], ex["plan"], ex["profile"]

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(C["bg"])

    hdr = fig.add_axes([0.0, 0.92, 1.0, 0.08])
    hdr.set_facecolor(C["navy"])
    hdr.set_xticks([])
    hdr.set_yticks([])
    for s in hdr.spines.values():
        s.set_visible(False)
    hdr.text(
        0.02, 0.5, "Pratibimb  |  clinician view", color="white", fontsize=17, fontweight="bold", va="center"
    )
    hdr.text(0.98, 0.5, "CONCEPTUAL MOCK-UP  |  synthetic patient, real twin output", color="#F6C56B",
             fontsize=11.5, fontweight="bold", va="center", ha="right")  # fmt: skip

    # left: patient and twin vitals
    left = fig.add_axes([0.02, 0.07, 0.22, 0.82])
    left.axis("off")
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.add_patch(FancyBboxPatch((0, 0.72), 1, 0.28, boxstyle="round,pad=0,rounding_size=0.02",
                                  fc="white", ec=C["navy"], lw=1.6))  # fmt: skip
    probs = st["endotype_probabilities"]
    regimen = "; ".join(f"{r.drug} {r.dose_mg:g} mg" for r in prof.regimens)
    left.text(
        0.05, 0.965, "Patient VP-001 (synthetic)", fontsize=12.5, fontweight="bold", color=C["navy"], va="top"
    )
    left.text(0.05, 0.915, textwrap.fill(f"Age {prof.age_years:.0f}, BMI {prof.bmi:.1f}, HbA1c {prof.hba1c_pct}%", 34),
              fontsize=10, color=C["text"], va="top")  # fmt: skip
    left.text(0.05, 0.865, f"Endotype: {ex['endotype']} (p = {probs[ex['endotype']]:.2f})", fontsize=10,
              color=C["text"], va="top")  # fmt: skip
    left.text(0.05, 0.82, textwrap.fill(f"Regimen: {regimen}", 34), fontsize=10, color=C["text"], va="top")
    left.text(
        0.0,
        0.69,
        "Twin vitals (median, 90% interval)",
        fontsize=11.5,
        fontweight="bold",
        color=C["teal"],
        va="top",
    )
    params = st["parameters"]
    cards = [
        ("si", "Insulin sensitivity", "{:.2e}"),
        ("gb", "Basal glucose (untreated)", "{:.0f} mg/dL"),
        ("ksu", "Sulfonylurea response", "{:.2f}"),
    ]
    for i, (key, label, fmt) in enumerate(cards):
        y0 = 0.45 - i * 0.215
        p = params[key]
        left.add_patch(FancyBboxPatch((0, y0), 1, 0.19, boxstyle="round,pad=0,rounding_size=0.02",
                                      fc="white", ec=C["teal"], lw=1.4))  # fmt: skip
        left.text(0.05, y0 + 0.165, label, fontsize=10, color=C["muted"], va="top")
        left.text(
            0.05,
            y0 + 0.125,
            fmt.format(p["median"]),
            fontsize=16,
            fontweight="bold",
            color=C["navy"],
            va="top",
        )
        left.text(0.05, y0 + 0.045, f"{fmt.format(p['p05'])} to {fmt.format(p['p95'])}", fontsize=9.2,
                  color=C["muted"], va="bottom")  # fmt: skip
        left.text(0.95, y0 + 0.125, f"{p['change_vs_prior_pct']:+.0f}% vs prior", fontsize=10, ha="right",
                  color=C["teal"], fontweight="bold", va="top")  # fmt: skip

    # centre: forecast chart
    ch = fig.add_axes([0.285, 0.17, 0.45, 0.67])
    ch.set_facecolor("white")
    t = np.array([_hours(x) for x in fc["times"]])
    ch.fill_between(t, fc["p2_5"], fc["p97_5"], color=C["teal"], alpha=0.16, label="95% band", lw=0)
    ch.fill_between(t, fc["p10"], fc["p90"], color=C["teal"], alpha=0.34, label="80% band", lw=0)
    ch.plot(t, fc["median"], color=C["navy"], lw=2.6, label="Twin forecast (median)")
    rx = [(tm - 1440) / 60 for tm, _ in ex["readings"]]
    ry = [v for _, v in ex["readings"]]
    ch.scatter(rx, ry, s=95, color=C["red"], zorder=5, label="Fingerstick today", edgecolor="white", lw=1.4)
    ch.axhline(70, color=C["amber"], lw=1.8, ls="--")
    ch.text(t[0] + 0.1, 72, "70 mg/dL", color=C["amber"], fontsize=10, va="bottom")
    ch.axhline(54, color=C["red"], lw=1.8, ls="--")
    ch.text(t[0] + 0.1, 56, "54 mg/dL", color=C["red"], fontsize=10, va="bottom")
    ch.set_ylim(30, max(max(fc["p97_5"]) * 1.05, 260))
    for r in plan["recommended_times"]:
        h = _hours(r["time"])
        ch.axvline(h, color=C["amber"], lw=3.0, alpha=0.9)
        ch.text(h + 0.12, ch.get_ylim()[1] * 0.97, "test here", color=C["amber"], fontsize=10.5,
                fontweight="bold", va="top", rotation=90)  # fmt: skip
    ch.axvline(t[0], color=C["muted"], lw=1.2, ls=":")
    ch.text(t[0] - 0.1, 32, "now", color=C["muted"], ha="right", fontsize=10)
    ch.set_xlim(6.5, t[-1] + 0.2)
    ticks = np.arange(7, t[-1], 2)
    ch.set_xticks(ticks)
    ch.set_xticklabels([f"{int(x) % 24:02d}:00" for x in ticks])
    ch.set_ylabel("Glucose (mg/dL)")
    ch.set_title(
        f"Glucose forecast for the next {ex['horizon_h']} hours",
        fontsize=13,
        fontweight="bold",
        color=C["navy"],
        loc="left",
    )
    ch.grid(alpha=0.25)
    ch.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=9.5, frameon=False)
    for s in ("top", "right"):
        ch.spines[s].set_visible(False)

    # right: risk and strip plan
    rt = fig.add_axes([0.765, 0.07, 0.215, 0.82])
    rt.axis("off")
    rt.set_xlim(0, 1)
    rt.set_ylim(0, 1)
    rt.text(
        0,
        0.985,
        f"Risk in the next {ex['horizon_h']} hours",
        fontsize=12,
        fontweight="bold",
        color=C["teal"],
        va="top",
    )
    risks = [("Glucose < 70 for 15+ min", fc["risk"]["hypo_lt70_15min"], C["amber"]),
             ("Glucose < 54 for 15+ min", fc["risk"]["hypo_lt54_15min"], C["red"]),
             ("Glucose > 250", fc["risk"]["hyper_gt250"], C["navy"])]  # fmt: skip
    for i, (label, p, col) in enumerate(risks):
        y0 = 0.86 - i * 0.125
        rt.text(0, y0, label, fontsize=10, color=C["text"], va="top")
        rt.add_patch(FancyBboxPatch((0, y0 - 0.065), 1, 0.03, boxstyle="round,pad=0,rounding_size=0.01",
                                    fc="#E3E9F2", ec="none"))  # fmt: skip
        rt.add_patch(FancyBboxPatch((0, y0 - 0.065), max(p, 0.012), 0.03, boxstyle="round,pad=0,rounding_size=0.01",
                                    fc=col, ec="none"))  # fmt: skip
        rt.text(1.0, y0, f"{p:.0%}", fontsize=11.5, fontweight="bold", color=col, ha="right", va="top")
    rt.text(0, 0.44, "Recommended strips", fontsize=12, fontweight="bold", color=C["teal"], va="top")
    rt.text(0, 0.405, "ranked by information about low-glucose risk", fontsize=9, color=C["muted"], va="top",
            style="italic")  # fmt: skip
    recs = plan["recommended_times"]
    if recs:
        for i, r in enumerate(recs):
            y0 = 0.35 - i * 0.1
            rt.add_patch(FancyBboxPatch((0, y0 - 0.075), 1, 0.085, boxstyle="round,pad=0,rounding_size=0.015",
                                        fc=C["light_amber"], ec=C["amber"], lw=1.3))  # fmt: skip
            rt.text(
                0.05,
                y0 - 0.032,
                f"{r['time'][11:16]}",
                fontsize=15,
                fontweight="bold",
                color=C["navy"],
                va="center",
            )
            rt.text(0.95, y0 - 0.032, f"{r['information_gain_bits']:.2f} bits", fontsize=10, color=C["muted"],
                    ha="right", va="center")  # fmt: skip
    else:
        rt.text(0, 0.35, textwrap.fill("No further strips needed today: another reading would not change the "
                                       "risk estimate.", 30), fontsize=10, color=C["text"], va="top")  # fmt: skip
    rt.text(0, 0.04, textwrap.fill("Research prototype. Not a medical device. No dosing advice.", 32),
            fontsize=8.8, color=C["muted"], va="bottom", style="italic")  # fmt: skip

    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "dashboard_mockup.png", dpi=170)
    plt.close(fig)
    print("wrote docs/figures/dashboard_mockup.png")


# ====================================================================== presentation
SW, SH = 13.333, 7.5


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color.lstrip("#").upper())


def _text(slide, x, y, w, h, paragraphs, size=18, color="text", bold=False, align=PP_ALIGN.LEFT,
          anchor=MSO_ANCHOR.TOP, font="Calibri", space_after=6):  # fmt: skip
    """Add a text box. paragraphs: str, or list of str / (str, {overrides})."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    items = [paragraphs] if isinstance(paragraphs, str) else paragraphs
    for i, item in enumerate(items):
        text, o = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = o.get("align", align)
        p.space_after = Pt(o.get("space_after", space_after))
        run = p.add_run()
        run.text = text
        run.font.name = o.get("font", font)
        run.font.size = Pt(o.get("size", size))
        run.font.bold = o.get("bold", bold)
        run.font.color.rgb = rgb(C.get(o.get("color", color), o.get("color", color)))
    return box


def _rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(C.get(fill, fill))
    if line:
        s.line.color.rgb = rgb(C.get(line, line))
        s.line.width = Pt(1.5)
    else:
        s.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.shadow.inherit = False
    return s


def _table(slide, x, y, w, rows, col_w, size=13, header_fill="navy", row_h=0.42):
    shape = slide.shapes.add_table(
        len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows))
    )
    tbl = shape.table
    for i, cw in enumerate(col_w):
        tbl.columns[i].width = Inches(cw)
    for r, row in enumerate(rows):
        tbl.rows[r].height = Inches(row_h)
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(C[header_fill] if r == 0 else (C["white"] if r % 2 else C["bg"]))
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = str(val)
            run.font.size = Pt(size)
            run.font.name = "Calibri"
            run.font.bold = r == 0 or c == 0
            run.font.color.rgb = rgb("#FFFFFF" if r == 0 else C["text"])
    return shape


class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width = Inches(SW)
        self.prs.slide_height = Inches(SH)
        self.n = 0

    def slide(self, title: str | None, notes: str = ""):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.n += 1
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb(C["white"])
        if title:
            _rect(s, 0, 0, SW, 1.05, "navy", shape=MSO_SHAPE.RECTANGLE)
            _rect(s, 0, 1.05, SW, 0.06, "teal", shape=MSO_SHAPE.RECTANGLE)
            _text(
                s,
                0.55,
                0.12,
                SW - 1.1,
                0.85,
                title,
                size=30,
                color="white",
                bold=True,
                anchor=MSO_ANCHOR.MIDDLE,
            )
            _text(
                s,
                0.55,
                7.08,
                9,
                0.3,
                f"Team {TEAM}  |  Pratibimb  |  Digital Twin Challenge 2026",
                size=10,
                color="muted",
            )
            _text(s, SW - 1.4, 7.08, 0.9, 0.3, str(self.n), size=10, color="muted", align=PP_ALIGN.RIGHT)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    def save(self, path: Path) -> None:
        self.prs.save(path)


def _load_results() -> dict:
    return json.loads((RESULTS / "summary.json").read_text())["summary"]


def build_presentation() -> None:
    res = _load_results()
    g = res.get
    d = Deck()

    # 1. Title -------------------------------------------------------------------------------
    s = d.slide(
        None,
        "Introduce the team, the challenge and the one-line idea: a digital twin that decides when to test.",
    )
    _rect(s, 0, 0, SW, SH, "navy", shape=MSO_SHAPE.RECTANGLE)
    _rect(s, 0, 5.05, SW, 0.07, "amber", shape=MSO_SHAPE.RECTANGLE)
    _text(s, 0.8, 0.9, 11.5, 1.2, "Pratibimb", size=66, color="white", bold=True)
    _text(s, 0.8, 2.0, 11.5, 0.7, "प्रतिबिंब  (reflection)", size=26, color="#9FD8D9", font="Nirmala UI")
    _text(s, 0.8, 2.8, 11.9, 1.0, "A metabolic digital twin for Type 2 Diabetes that decides when to test",
          size=27, color="white")  # fmt: skip
    _text(s, 0.8, 4.05, 11.7, 0.6, "India-calibrated  |  Digital Twin Challenge 2026  |  Happiest Health",
          size=17, color="#9FD8D9")  # fmt: skip
    _text(s, 0.8, 5.35, 6, 0.5, f"Team {TEAM}", size=24, color="#F6C56B", bold=True)
    y = 5.9
    for name, role, college in MEMBERS:
        _text(s, 0.8, y, 12, 0.45, [(f"{name}  ({role}), {college}", {})], size=15, color="white")
        y += 0.42
    _text(s, 0.8, 6.85, 12, 0.4, REPO_URL, size=13, color="#9FD8D9")

    # 2. Problem -----------------------------------------------------------------------------
    s = d.slide(
        "The problem: silent lows, and CGM is out of reach",
        "Walk through the three numbers, then pose the question: if a patient can afford only a few strips a day, when should they test?",
    )
    stats = [
        ("~101 million", "people with diabetes in India (ICMR-INDIAB-17, Lancet D&E 2023)"),
        ("67% / 73%", "In sulfonylurea-treated adults on CGM, lows were typically nocturnal (67%) and asymptomatic (73%)  (Middleton, Diabetes Care 2017)"),
        ("~Rs 9,000 / month", "cost of continuous CGM in India: Rs 4,200-5,250 per 14-day sensor (2026 retail prices)"),
    ]  # fmt: skip
    for i, (big, small) in enumerate(stats):
        x = 0.6 + i * 4.15
        _rect(s, x, 1.55, 3.95, 2.9, "light_navy", line="navy")
        _text(s, x + 0.15, 1.7, 3.65, 0.9, big, size=32, color="navy", bold=True)
        _text(s, x + 0.15, 2.6, 3.65, 1.8, small, size=15, color="text")
    _rect(s, 0.6, 4.85, 12.15, 1.65, "light_amber", line="amber")
    _text(s, 0.85, 4.95, 11.7, 0.5, "The question we answer", size=15, color="amber", bold=True)
    _text(s, 0.85, 5.4, 11.7, 1.1, "With only 1-4 test strips a day, WHEN should a patient test so that dangerous lows are predicted before they happen?",
          size=23, color="navy", bold=True)  # fmt: skip

    # 3. Idea --------------------------------------------------------------------------------
    s = d.slide("Our idea: a twin that learns, predicts, and decides when to test",
                "Three verbs: learn, predict, decide. The third is what other teams do not have.")  # fmt: skip
    cards = [
        ("1  Learn", "teal", "light_teal", "A personal physiological model starts from the EHR and an Indian-specific prior, then learns the patient's insulin sensitivity, basal glucose and drug response from every reading."),
        ("2  Predict", "navy", "light_navy", "Forecast glucose with honest uncertainty bands, and the probability of hypoglycaemia (below 70 and below 54 mg/dL) in the coming hours and overnight."),
        ("3  Decide", "amber", "light_amber", "Choose the testing times that tell the twin the most about upcoming lows, and re-plan after every reading. Fewer strips, better warnings."),
    ]  # fmt: skip
    for i, (t, edge, fill, body) in enumerate(cards):
        x = 0.6 + i * 4.15
        _rect(s, x, 1.6, 3.95, 3.9, fill, line=edge)
        _text(s, x + 0.2, 1.75, 3.55, 0.7, t, size=28, color=edge, bold=True)
        _text(s, x + 0.2, 2.65, 3.55, 2.8, body, size=18, color="text")
    _text(s, 0.6, 5.85, 12.1, 0.5, "A classifier gives a number once. A digital twin keeps learning the patient and tells you what to do next.",
          size=16, color="muted", bold=True)  # fmt: skip

    # 4. Use case ----------------------------------------------------------------------------
    s = d.slide("Healthcare use case: one patient, one day",
                "Take the five steps in order. Emphasise that the doctor stays in the loop and every number is auditable.")  # fmt: skip
    steps = [
        ("Enrol", "Doctor creates the twin from the EHR: age, BMI, waist, HbA1c, fasting glucose, regimen."),
        ("Personalise", "Indian endotype prior, then the twin learns from each reading plus the meal and dose log."),
        ("Plan", "Each morning: the most informative testing times for the patient's strip budget."),
        ("Warn", "At bedtime: overnight hypoglycaemia risk with an uncertainty band. Doctor reviews the dose."),
        ("Audit", "Every number traces back to the exact readings and events behind it."),
    ]  # fmt: skip
    for i, (t, body) in enumerate(steps):
        x = 0.45 + i * 2.5
        chev = _rect(s, x, 1.7, 2.45, 0.9, "teal" if i != 2 else "amber", shape=MSO_SHAPE.PENTAGON)
        chev.text_frame.text = ""
        _text(
            s,
            x + 0.1,
            1.72,
            2.1,
            0.85,
            t,
            size=21,
            color="white",
            bold=True,
            anchor=MSO_ANCHOR.MIDDLE,
            align=PP_ALIGN.CENTER,
        )
        _text(s, x + 0.05, 2.85, 2.3, 2.6, body, size=15, color="text")
    _rect(s, 0.6, 5.0, 12.15, 1.35, "light_navy", line="navy")
    _text(s, 0.85, 5.1, 11.7, 1.2, [("Who benefits", {"bold": True, "color": "navy", "size": 16}),
                                     "Patients on sulfonylureas without a CGM, and the diabetologists who must decide who needs review tonight, not next month."],
          size=16)  # fmt: skip

    # 5. NASEM -------------------------------------------------------------------------------
    s = d.slide("Why this is a digital twin, not just a prediction model",
                "Read the NASEM definition, then walk down the table. Each row is something a plain classifier cannot do.")  # fmt: skip
    _text(s, 0.6, 1.3, 12.1, 0.7, "NASEM (2023): a virtual representation that is dynamically updated from its physical twin, has predictive capability, and informs decisions.",
          size=16, color="muted")  # fmt: skip
    _table(s, 0.6, 2.1, 12.1, [
        ["NASEM element", "Pratibimb"],
        ["Virtual representation of a specific patient", "10-state glucose, insulin, gut and drug model with 6 patient-specific parameters"],
        ["Dynamically updated from the physical twin", "Particle filter assimilates every reading (fingerstick or CGM)"],
        ["Predictive capability", "Posterior-predictive glucose trajectories and event probabilities"],
        ["Informs decisions", "Active-sensing planner (when to test), overnight risk at bedtime"],
        ["Two-way data flow", "The twin asks the patient to test at 16:30 and updates on the answer"],
        ["Verification, validation, uncertainty", "29 unit tests, known-truth evaluation, bootstrap CIs, full audit log"],
    ], [4.3, 7.8], size=15, row_h=0.62)  # fmt: skip

    # 6. Architecture ------------------------------------------------------------------------
    s = d.slide(
        "System architecture",
        "Left to right: data in, twin core, decisions out. Point out the amber loop: the twin asks the patient for data.",
    )
    s.shapes.add_picture(str(FIGS / "architecture.png"), Inches(1.3), Inches(1.2), height=Inches(5.8))

    # 7. AI/ML -------------------------------------------------------------------------------
    s = d.slide("AI / ML methods: hybrid physiology + probabilistic inference",
                "Explain that this is deliberately not a black box. Each component is interpretable and carries uncertainty.")  # fmt: skip
    _table(s, 0.45, 1.3, 12.45, [
        ["Component", "Method", "Why"],
        ["Mechanistic core", "Extended Bergman minimal model: 10 states, 6 patient parameters, RK4 solver", "Meaningful parameters; valid what-if simulation"],
        ["Personalised prior", "Soft assignment to 4 INSPIRED endotypes (n = 19,084), log-normal prior; wearables shift insulin sensitivity", "Sensible twin on day 1, specific to Indian patients"],
        ["Learning", "Joint state + parameter particle filter, Student-t likelihood, Liu-West resampling, parameter drift", "Learns online from sparse noisy data, with full posterior"],
        ["Prediction", "Posterior-predictive ensembles; P(<70), P(<54), P(>250 mg/dL)", "Risk with uncertainty, not a point guess"],
        ["Decision (active sensing)", "Bayesian experimental design: greedy mutual information between readings and hypoglycaemia", "Picks WHEN to test for the most information per strip"],
    ], [2.35, 6.0, 4.1], size=13, row_h=0.74)  # fmt: skip
    _text(s, 0.45, 6.4, 12.4, 0.5, "Stack: Python, NumPy (no GPU needed), FastAPI, Pydantic, Docker, GitHub Actions CI (Python 3.11 and 3.12).",
          size=15, color="muted")  # fmt: skip

    # 8. Evaluation design -------------------------------------------------------------------
    s = d.slide(
        "How we tested it",
        "Stress that the truth is known, so we can score predictions exactly. Be upfront that this is synthetic data.",
    )
    _text(s, 0.6, 1.45, 6.2, 5.3, [
        ("Virtual Indian cohort with known ground truth", {"bold": True, "color": "navy", "size": 19}),
        "40 sulfonylurea-treated patients drawn from the INSPIRED endotype mix; 7 evaluation days each; 1,120 risk windows (morning, afternoon, evening, night).",
        ("Honest mismatch", {"bold": True, "color": "navy", "size": 19}),
        "The simulator differs from the twin: perturbed constants, daily parameter drift, carbohydrate-counting error, unlogged snacks, missed doses, device noise.",
        ("Fair comparison", {"bold": True, "color": "navy", "size": 19}),
        "Same truth, logs, meter noise and seed for every policy; patient-level paired bootstrap.",
    ], size=16, space_after=8)  # fmt: skip
    _rect(s, 7.2, 1.45, 5.5, 5.2, "light_teal", line="teal")
    _text(s, 7.4, 1.55, 5.1, 0.5, "Testing policies compared", size=19, color="teal", bold=True)
    _text(s, 7.4, 2.15, 5.1, 4.4, [
        "No readings: prior + logs only",
        "Fixed: routine fasting / post-meal schedule",
        "Random: random waking times",
        "Twin-scheduled: information-maximising times, re-planned after every reading",
        "CGM: reading every 15 minutes (reference)",
        ("Metric: AUROC and Brier score of P(low) at the start of each window.", {"bold": True, "color": "navy"}),
    ], size=16, space_after=9)  # fmt: skip

    # 9. Results -----------------------------------------------------------------------------
    f1, t1, f2, t2 = g("fixed-1"), g("twin-1"), g("fixed-2"), g("twin-2")
    cgm, none = g("cgm"), g("none")

    def row(label, m):
        return [
            label,
            f"{m['readings_per_day']:.1f}",
            f"{m['auroc']:.3f}",
            f"{m['brier']:.4f}",
            f"{m['sensitivity@0.2']:.2f}",
        ]

    s = d.slide("Results: twin-chosen testing predicts lows better with fewer strips",
                "Read the table, not just the chart. At one or two strips the twin beats the fixed schedule and uses fewer strips. At three and four strips a random schedule catches up; say so.")  # fmt: skip
    s.shapes.add_picture(str(RESULTS / "strip_budget.png"), Inches(0.35), Inches(1.4), width=Inches(7.3))
    _table(s, 7.85, 1.4, 5.15, [
        ["Policy", "Strips/day", "AUROC", "Brier", "Sens."],
        row("Fixed, budget 1", f1),
        row("Twin, budget 1", t1),
        row("Fixed, budget 2", f2),
        row("Twin, budget 2", t2),
        row("CGM (reference)", cgm),
        row("No readings", none),
    ], [1.75, 0.9, 0.85, 0.85, 0.8], size=12, row_h=0.46)  # fmt: skip
    _text(s, 7.85, 4.75, 5.15, 2.2, [
        ("Brier improvement vs fixed schedule excludes zero at every budget (at 1 strip: +0.008, 95% CI +0.002 to +0.016).", {"size": 14}),
        ("AUROC / Brier: higher AUROC and lower Brier are better. Sens. = sensitivity at P >= 0.2.", {"size": 12, "color": "muted"}),
    ], size=14, space_after=8)  # fmt: skip
    _text(
        s,
        0.45,
        4.3,
        7.2,
        0.8,
        "Chart: x-axis is the strip budget; the twin used fewer strips than the budget because it skips readings that would not change the risk estimate.",
        size=12,
        color="muted",
    )
    _rect(s, 0.45, 5.2, 7.2, 1.65, "light_amber", line="amber")
    _text(s, 0.65, 5.28, 6.8, 1.55, [
        ("Where it does not help", {"bold": True, "color": "amber", "size": 15}),
        ("At 3-4 strips a day a random schedule performs about as well as the twin (no significant difference), and general glucose tracking is worse than the fixed schedule because the planner optimises for lows.", {"size": 13}),
    ], space_after=4)  # fmt: skip

    # 10. Limitations ------------------------------------------------------------------------
    s = d.slide(
        "What we claim, and what we do not",
        "This slide protects our credibility. Read the limitations aloud: few events, synthetic data, and the tracking trade-off.",
    )
    _rect(s, 0.6, 1.5, 5.95, 4.1, "light_teal", line="teal")
    _text(s, 0.8, 1.6, 5.6, 0.5, "What the evidence supports", size=21, color="teal", bold=True)
    _text(s, 0.8, 2.25, 5.6, 3.3, [
        "At 1-2 strips a day, twin-chosen times predict hypoglycaemia better than the routine fasting / post-meal schedule.",
        "The twin uses fewer strips: it skips readings that would not change the risk estimate.",
        "With under one strip a day its risk ranking comes close to a CGM's (AUROC 0.985 vs 0.990).",
    ], size=16, space_after=10)  # fmt: skip
    _rect(s, 6.8, 1.5, 5.95, 4.1, "light_amber", line="amber")
    _text(s, 7.0, 1.6, 5.6, 0.5, "Limitations we state openly", size=21, color="amber", bold=True)
    _text(s, 7.0, 2.25, 5.6, 3.3, [
        "Only 5 of 40 virtual patients had any low (47 windows): small evidence base.",
        "Synthetic data only; the twin and simulator share model structure. Real-data validation is next.",
        "At 3-4 strips a random schedule matches the twin.",
        "General glucose tracking is worse than the fixed schedule (14.1 vs 10.4 mg/dL at budget 3): the planner optimises for lows.",
    ], size=16, space_after=10)  # fmt: skip

    _rect(s, 0.6, 5.85, 12.15, 0.95, "navy")
    _text(s, 0.85, 5.9, 11.7, 0.85, "Bottom line: a promising result on synthetic patients. Validation on real patient data decides whether it holds.",
          size=18, color="white", bold=True, anchor=MSO_ANCHOR.MIDDLE)  # fmt: skip

    # 11. Dashboard --------------------------------------------------------------------------
    s = d.slide(
        "Conceptual clinician dashboard",
        "Walk the doctor's eye path: patient on the left, forecast in the middle with the amber 'test here' markers, risks and strips on the right.",
    )
    s.shapes.add_picture(str(FIGS / "dashboard_mockup.png"), Inches(1.3), Inches(1.2), height=Inches(5.8))

    # 12. Safety -----------------------------------------------------------------------------
    s = d.slide("Safety, privacy and regulation by design",
                "Say plainly that this is a research prototype and gives no dosing advice. Mention the DPDP and CDSCO points as design direction, not compliance claims.")  # fmt: skip
    pts = [
        ("No dosing advice", "Outputs are forecasts and risk. Every API response carries a research-prototype disclaimer."),
        ("Only synthetic and open data", "No patient data in the repository; dataset licences documented in DATA.md."),
        ("Privacy by design (DPDP Act 2023 / Rules 2025)", "No names or identifiers in the profile, erasure endpoint, full audit trail. Consent management is future work."),
        ("Regulatory awareness", "CDSCO draft guidance on medical device software (21 Oct 2025) would treat a clinical version as SaMD; we make no clinical claims."),
        ("Auditability", "Event-sourced: replaying the log reproduces every number the twin ever reported."),
    ]  # fmt: skip
    for i, (t, body) in enumerate(pts):
        y = 1.45 + i * 1.07
        _rect(s, 0.6, y, 0.12, 0.9, "teal", shape=MSO_SHAPE.RECTANGLE)
        _text(s, 0.9, y - 0.02, 11.8, 0.4, t, size=18, color="navy", bold=True)
        _text(s, 0.9, y + 0.4, 11.8, 0.55, body, size=15, color="text")

    # 13. Roadmap ----------------------------------------------------------------------------
    s = d.slide(
        "Next steps and how to run it", "Close with the roadmap and the repository link. Invite questions."
    )
    _text(s, 0.6, 1.4, 6.2, 5.4, [
        ("Roadmap", {"bold": True, "color": "navy", "size": 22}),
        "Validate on real open data: ShanghaiT2DM, then CGMacros",
        "Conformal calibration of forecast bands",
        "Interactive clinician dashboard (this mock-up is conceptual)",
        "DPP-4 and SGLT2 inhibitor mechanisms; fasting-day (vrat / Ramadan) scenario",
        "ABDM FHIR R4 import of the EHR profile",
    ], size=17, space_after=9)  # fmt: skip
    _rect(s, 7.1, 1.4, 5.6, 3.15, "light_navy", line="navy")
    _text(s, 7.3, 1.5, 5.2, 0.5, "Run it", size=20, color="navy", bold=True)
    _text(s, 7.3, 2.1, 5.2, 2.4, [
        ("uv sync --all-extras", {"font": "Consolas"}),
        ("uv run pytest", {"font": "Consolas"}),
        ("uv run python scripts/demo_day.py", {"font": "Consolas"}),
        ("uv run uvicorn pratibimb.api.app:app", {"font": "Consolas"}),
    ], size=14, space_after=6)  # fmt: skip
    _rect(s, 7.1, 4.8, 5.6, 1.95, "navy")
    _text(s, 7.3, 4.9, 5.2, 0.5, f"Team {TEAM}", size=21, color="#F6C56B", bold=True)
    _text(
        s,
        7.3,
        5.4,
        5.2,
        1.3,
        ["Apache-2.0", REPO_URL, "Thank you. Questions?"],
        size=15,
        color="white",
        space_after=6,
    )

    DOCS.mkdir(parents=True, exist_ok=True)
    d.save(DOCS / "presentation.pptx")
    print(f"wrote docs/presentation.pptx ({d.n} slides)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["architecture", "dashboard", "presentation"])
    args = ap.parse_args()
    if args.only in (None, "architecture"):
        build_architecture()
    if args.only in (None, "dashboard"):
        build_dashboard()
    if args.only in (None, "presentation"):
        build_presentation()


if __name__ == "__main__":
    main()
