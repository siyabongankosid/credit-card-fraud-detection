"""Sentinel Fraud Analytics — local Streamlit dashboard.

A user-friendly front end for the trained credit-card fraud model.  The app
never retrains, replaces or fabricates the saved artifacts; every metric and
prediction shown here comes from the project's genuine files.

Run locally with:
    .venv\\Scripts\\python.exe -m streamlit run app.py
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import fraud_ml as fm  # noqa: E402

st.set_page_config(
    page_title="Sentinel Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Batch scoring keeps its original limit.  Raising the Dataset Lab limit must
# not silently raise the limit for this separate uploader.
BATCH_UPLOAD_MAX_MB = 200

# --------------------------------------------------------------------------- #
# Design system
# --------------------------------------------------------------------------- #
_CSS = """
:root {
    --bg: #f4f6fa;
    --card: #ffffff;
    --border: #e5eaf3;
    --border-strong: #d7dfeb;
    --text: #0e1f3c;
    --muted: #54657e;
    --faint: #8794ab;
    --accent: #2563eb;
    --accent-dark: #1d4ed8;
    --accent-soft: #eaf1fe;
    --fraud: #dc2626;
    --fraud-soft: #fef2f2;
    --fraud-border: #fecaca;
    --legit: #15803d;
    --legit-soft: #f0fdf4;
    --legit-border: #bbf7d0;
    --amber: #b45309;
    --amber-soft: #fffbeb;
    --amber-border: #fde68a;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #f4f6fa !important;
    color: #0e1f3c !important;
}
[data-testid="stHeader"] { background: rgba(244, 246, 250, 0.9) !important; }
[data-testid="stAppDeployButton"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stMainBlockContainer"] {
    padding: 2.2rem 2.8rem 5rem !important;
    max-width: 1360px !important;
    margin: 0 auto;
}
@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] { padding: 1.3rem 1rem 3rem !important; }
}

html, body, [data-testid="stApp"], .stMarkdown, input, textarea, button, select, label {
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
}
h1, h2, h3, h4, h5 { color: #0e1f3c !important; letter-spacing: -0.012em; }
.stMarkdown p, .stMarkdown li { color: #43536b; line-height: 1.65; font-size: 0.93rem; }
a { color: #2563eb; }

/* ---------- Page header ---------- */
.page-header {
    display: flex; justify-content: space-between; align-items: flex-start;
    gap: 16px; margin: 0 0 22px; flex-wrap: wrap;
}
.page-header h1 { font-size: 1.9rem; font-weight: 800; margin: 0; line-height: 1.18; }
.page-subtitle {
    color: #54657e; margin: 7px 0 0; font-size: 0.94rem;
    line-height: 1.6; max-width: 800px;
}
.pill {
    display: inline-block; background: #eaf1fe; color: #1d4ed8;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; padding: 6px 12px; border-radius: 999px;
    white-space: nowrap; border: 1px solid #dbe7ff;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5eaf3 !important;
}
section[data-testid="stSidebar"] > div { background: transparent !important; }
section[data-testid="stSidebar"] .stMarkdown p { color: #54657e !important; }
section[data-testid="stSidebar"] .stButton > button {
    justify-content: flex-start;
    background: transparent !important;
    color: #43536b !important;
    border: 1px solid transparent !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #f2f6fc !important; color: #0e1f3c !important;
}
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
    background: #eaf1fe !important; color: #1d4ed8 !important;
    border-color: #dbe7ff !important;
}
section[data-testid="stSidebar"] .stButton > button p { color: inherit !important; }

.brand { display: flex; align-items: center; gap: 12px; padding: 10px 4px 20px; }
.brand-mark {
    width: 40px; height: 40px; border-radius: 12px; background: #2563eb;
    color: #ffffff; display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 1.08rem;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.28);
}
.brand-name { font-size: 1.02rem; font-weight: 800; color: #0e1f3c; line-height: 1.2; }
.brand-sub {
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #8794ab; margin-top: 2px;
}
.sidebar-card {
    margin-top: 18px; padding: 14px 15px; border: 1px solid #e5eaf3;
    border-radius: 12px; background: #f8fafd;
}
.sidebar-card .sc-eyebrow {
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #8794ab;
}
.sidebar-card .sc-title { font-size: 0.88rem; font-weight: 700; color: #0e1f3c; margin-top: 5px; }
.sidebar-card .sc-sub { font-size: 0.76rem; color: #54657e; margin-top: 4px; line-height: 1.5; }
.sidebar-note {
    margin-top: 12px; padding: 0 2px; font-size: 0.74rem;
    color: #8794ab; line-height: 1.55;
}

/* ---------- Cards (st.container(border=True)) ---------- */
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
    background: #ffffff !important;
    border: 1px solid #e5eaf3 !important;
    border-radius: 16px !important;
    padding: 20px 22px !important;
    box-shadow: 0 1px 2px rgba(16, 35, 63, 0.04);
}
.card-title { font-size: 1.02rem; font-weight: 700; color: #0e1f3c; margin: 0 0 4px; }
.card-subtitle { font-size: 0.84rem; color: #54657e; margin: 0 0 14px; line-height: 1.55; }

/* ---------- KPI grid ---------- */
.kpi-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 14px; margin: 2px 0 20px;
}
.kpi-card {
    background: #ffffff; border: 1px solid #e5eaf3; border-radius: 14px;
    padding: 16px 18px; box-shadow: 0 1px 2px rgba(16, 35, 63, 0.04);
}
.kpi-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: #6b7a92;
}
.kpi-value {
    font-size: 1.62rem; font-weight: 800; color: #0e1f3c;
    margin-top: 7px; letter-spacing: -0.025em; line-height: 1.2;
}
.kpi-delta { font-size: 0.79rem; font-weight: 600; margin-top: 6px; }
.kpi-delta.good { color: #15803d; }
.kpi-delta.bad { color: #dc2626; }
.kpi-delta.neutral { color: #2563eb; }
.kpi-foot { font-size: 0.74rem; color: #8794ab; margin-top: 4px; line-height: 1.45; }

/* ---------- Callouts, banners ---------- */
.callout {
    background: #f8fafd; border: 1px solid #e5eaf3; border-radius: 12px;
    padding: 13px 16px; font-size: 0.87rem; color: #43536b; line-height: 1.6;
}
.callout b { color: #0e1f3c; }
.callout.warn { background: #fffbeb; border-color: #fde68a; color: #92400e; }
.callout.warn b { color: #78350f; }
.callout.danger { background: #fef2f2; border-color: #fecaca; color: #991b1b; }
.callout.danger b { color: #7f1d1d; }
.callout.ok { background: #f0fdf4; border-color: #bbf7d0; color: #166534; }
.callout.ok b { color: #14532d; }
.callout.info { background: #eff6ff; border-color: #dbeafe; color: #1e40af; }
.callout.info b { color: #1e3a8a; }

/* ---------- First-run hero ---------- */
.hero {
    background: #f7faff; border: 1px solid #dbe7ff; border-radius: 18px;
    padding: 22px 24px; margin-bottom: 18px;
}
.hero .eyebrow {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: #2563eb; margin-bottom: 6px;
}
.hero h3 { margin: 0 0 6px; font-size: 1.28rem; font-weight: 800; }
.hero p { margin: 0; color: #43536b; font-size: 0.92rem; line-height: 1.6; max-width: 860px; }
.hero-steps {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px; margin-top: 16px;
}
.hero-step { background: #ffffff; border: 1px solid #e5eaf3; border-radius: 12px; padding: 12px 14px; }
.hero-step .n {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 999px; background: #eaf1fe;
    color: #1d4ed8; font-size: 0.72rem; font-weight: 800; margin-bottom: 7px;
}
.hero-step .t { font-size: 0.84rem; font-weight: 700; color: #0e1f3c; }
.hero-step .d { font-size: 0.78rem; color: #54657e; margin-top: 3px; line-height: 1.5; }

/* ---------- Verdict panel ---------- */
.verdict {
    border-radius: 18px; padding: 22px 24px; border: 1px solid #e5eaf3;
    background: #ffffff;
}
.verdict.fraud { border-color: #fecaca; background: #fffafa; }
.verdict.legit { border-color: #bbf7d0; background: #fbfefc; }
.verdict-head { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
.verdict-title { font-size: 1.34rem; font-weight: 800; color: #0e1f3c; }
.verdict.fraud .verdict-title { color: #b91c1c; }
.verdict.legit .verdict-title { color: #15803d; }
.verdict-prob {
    font-size: 2.55rem; font-weight: 800; letter-spacing: -0.035em;
    color: #0e1f3c; margin-top: 14px; line-height: 1.1;
}
.verdict-caption { font-size: 0.78rem; color: #6b7a92; margin-top: 6px; }
.prob-track {
    height: 10px; background: #eef1f7; border-radius: 999px;
    overflow: hidden; margin-top: 16px;
}
.prob-fill { height: 100%; border-radius: 999px; background: #2563eb; }
.verdict.fraud .prob-fill { background: #dc2626; }
.verdict.legit .prob-fill { background: #15803d; }
.risk-chip {
    display: inline-block; font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase;
    padding: 5px 11px; border-radius: 999px; border: 1px solid;
}
.risk-chip.high { background: #fef2f2; color: #b91c1c; border-color: #fecaca; }
.risk-chip.medium { background: #fffbeb; color: #b45309; border-color: #fde68a; }
.risk-chip.low { background: #f0fdf4; color: #15803d; border-color: #bbf7d0; }

.match {
    margin-top: 16px; padding: 13px 15px; border-radius: 12px;
    font-size: 0.86rem; line-height: 1.6; border: 1px solid;
}
.match.ok { background: #f0fdf4; border-color: #bbf7d0; color: #166534; }
.match.bad { background: #fef2f2; border-color: #fecaca; color: #991b1b; }
.match b { color: inherit; }

/* ---------- Status table ---------- */
.report-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.86rem; }
.report-table th {
    text-align: left; font-size: 0.68rem; letter-spacing: 0.07em;
    text-transform: uppercase; color: #6b7a92; font-weight: 700;
    padding: 0 12px 10px; border-bottom: 1px solid #eef1f7;
}
.report-table td {
    padding: 12px; border-bottom: 1px solid #f2f5fa;
    vertical-align: top; color: #43536b; line-height: 1.55;
}
.report-table tr:last-child td { border-bottom: none; }
.report-table td:first-child { color: #0e1f3c; font-weight: 600; }
.status {
    display: inline-block; padding: 4px 10px; border-radius: 999px;
    font-size: 0.66rem; font-weight: 800; letter-spacing: 0.05em;
    text-transform: uppercase; white-space: nowrap; border: 1px solid;
}
.status.pass { background: #f0fdf4; color: #15803d; border-color: #bbf7d0; }
.status.fail { background: #fef2f2; color: #b91c1c; border-color: #fecaca; }
.status.warn { background: #fffbeb; color: #b45309; border-color: #fde68a; }

/* ---------- Summary grids ---------- */
.summary-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px; margin: 4px 0 0;
}
.summary-item {
    background: #f8fafd; border: 1px solid #eef1f7;
    border-radius: 12px; padding: 12px 14px;
}
.summary-item .k {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.07em;
    color: #8794ab; font-weight: 700;
}
.summary-item .v { font-size: 1rem; font-weight: 700; color: #0e1f3c; margin-top: 4px; }

.eyebrow {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: #8794ab; margin-bottom: 6px;
}
.def-term { font-size: 0.88rem; font-weight: 800; color: #0e1f3c; margin-bottom: 5px; }
.def-body { font-size: 0.81rem; color: #54657e; line-height: 1.6; }

.empty {
    text-align: center; padding: 32px 22px; border: 1px dashed #d7dfeb;
    border-radius: 14px; background: #fbfcfe;
}
.empty .title { font-weight: 700; color: #0e1f3c; font-size: 0.95rem; }
.empty .body { color: #6b7a92; font-size: 0.85rem; margin-top: 6px; line-height: 1.6; }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button {
    border-radius: 10px !important; font-weight: 600 !important;
    min-height: 42px; border: 1px solid #d7dfeb !important;
    background: #ffffff !important; color: #0e1f3c !important;
}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    border-color: #2563eb !important; color: #2563eb !important;
}
button[data-testid="stBaseButton-primary"],
[data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primary"] {
    background: #2563eb !important; border-color: #2563eb !important;
    color: #ffffff !important; box-shadow: 0 3px 10px rgba(37, 99, 235, 0.22);
}
button[data-testid="stBaseButton-primary"]:hover,
[data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primary"]:hover {
    background: #1d4ed8 !important; border-color: #1d4ed8 !important;
    color: #ffffff !important;
}

/* ---------- Tabs ---------- */
[data-testid="stTabs"] [role="tablist"] {
    gap: 4px; border-bottom: 1px solid #e5eaf3;
}
[data-testid="stTabs"] [role="tab"] {
    padding: 10px 14px; font-weight: 600; color: #54657e !important;
    border-radius: 10px 10px 0 0; cursor: pointer;
}
[data-testid="stTabs"] [role="tab"]:hover { color: #0e1f3c !important; background: #f8fafd; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #2563eb !important;
    box-shadow: inset 0 -2px 0 #2563eb;
}

/* ---------- Expander ---------- */
[data-testid="stExpander"] {
    background: #fbfcfe !important;
    border: 1px solid #e5eaf3 !important;
    border-radius: 12px !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    padding: 12px 16px !important; font-weight: 600 !important; color: #0e1f3c !important;
}
[data-testid="stExpander"] summary p { color: #0e1f3c !important; font-size: 0.86rem !important; }

/* ---------- Inputs / tables ---------- */
[data-testid="stDataFrame"] { border: 1px solid #e5eaf3; border-radius: 12px; overflow: hidden; }
[data-baseweb="input"], [data-baseweb="select"] > div { border-radius: 10px !important; }
.stProgress > div > div > div { background: #2563eb !important; }
[data-testid="stFileUploaderDropzone"] {
    background: #fbfcfe !important; border: 1px dashed #d7dfeb !important;
    border-radius: 12px !important;
}
.stAlert { border-radius: 12px; }

/* ---------- Charts ---------- */
.js-plotly-plot .plotly .modebar { opacity: 0.45; }
@media (max-width: 1100px) {
    .page-header h1 { font-size: 1.55rem; }
    .kpi-grid { grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
}
"""

st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def page_header(title: str, subtitle: str, badge: str | None = None) -> None:
    badge_html = f'<span class="pill">{escape(badge)}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="page-header">
            <div>
                <h1>{escape(title)}</h1>
                <p class="page-subtitle">{escape(subtitle)}</p>
            </div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title: str | None = None, subtitle: str | None = None):
    container = st.container(border=True)
    if title:
        container.markdown(f'<div class="card-title">{escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        container.markdown(f'<div class="card-subtitle">{escape(subtitle)}</div>', unsafe_allow_html=True)
    return container


def kpi_grid(items: list[dict]) -> None:
    cards = []
    for item in items:
        delta = item.get("delta", "")
        delta_html = (
            f'<div class="kpi-delta {item.get("tone", "neutral")}">{escape(str(delta))}</div>'
            if delta
            else ""
        )
        foot = item.get("foot", "")
        foot_html = f'<div class="kpi-foot">{escape(str(foot))}</div>' if foot else ""
        cards.append(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{escape(str(item["label"]))}</div>'
            f'<div class="kpi-value">{item["value"]}</div>'
            f"{delta_html}{foot_html}"
            f"</div>"
        )
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def callout(text: str, tone: str = "") -> None:
    klass = f"callout {tone}".strip()
    st.markdown(f'<div class="{klass}">{text}</div>', unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f'<div class="empty"><div class="title">{escape(title)}</div>'
        f'<div class="body">{escape(body)}</div></div>',
        unsafe_allow_html=True,
    )


def definition(term: str, body: str) -> None:
    st.markdown(
        f'<div class="def-term">{escape(term)}</div><div class="def-body">{escape(body)}</div>',
        unsafe_allow_html=True,
    )


def fmt_int(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def fmt_money(value) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def fmt_pct(value, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_metric(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def style_figure(fig: go.Figure, height: int = 320, legend: bool = False) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='Inter, "Segoe UI", system-ui, sans-serif', color="#0e1f3c", size=12),
        margin=dict(l=6, r=8, t=12, b=6),
        height=height,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color="#54657e", size=11)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e5eaf3", font=dict(color="#0e1f3c")),
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, linecolor="#e5eaf3",
        tickfont=dict(color="#54657e", size=11),
        title_font=dict(color="#54657e", size=12),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#eef1f7", zeroline=False,
        tickfont=dict(color="#54657e", size=11),
        title_font=dict(color="#54657e", size=12),
    )
    return fig


def fix_mojibake(text: str) -> str:
    """Make CSV model names safe to display (the saved CSV is UTF-8)."""
    return str(text).replace("\u2014", "-").strip()


def model_available(quiet: bool = False) -> bool:
    if fm.artifact_status()["model"]:
        return True
    if not quiet:
        empty_state(
            "Saved model not found",
            f"Expected {fm.MODEL_PATH}. Restore the saved artifact or run the training notebook; "
            "the dashboard will not substitute a different model.",
        )
    return False


def status_label(status: str) -> str:
    return {"pass": "Passed", "fail": "Failed", "warn": "Warning"}.get(status, status.title())


def report_table(checks: list[dict]) -> None:
    rows = []
    for check in checks:
        rows.append(
            "<tr>"
            f"<td>{escape(check['check'])}</td>"
            f'<td><span class="status {check["status"]}">{status_label(check["status"])}</span></td>'
            f"<td>{escape(check['detail'])}</td>"
            "</tr>"
        )
    st.markdown(
        '<table class="report-table"><thead><tr>'
        "<th>Check</th><th>Result</th><th>Details</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>",
        unsafe_allow_html=True,
    )


def render_verdict(prob: float, threshold: float) -> None:
    risk = fm.risk_category(prob)
    label = fm.decision_label(prob, threshold)
    tone = "fraud" if label == "Potential fraud" else "legit"
    pct = max(0.0, min(100.0, float(prob) * 100))
    st.markdown(
        f"""
        <div class="verdict {tone}">
            <div class="verdict-head">
                <span class="verdict-title">{label}</span>
                <span class="risk-chip {risk.lower()}">{risk} risk</span>
            </div>
            <div class="verdict-prob">{pct:.1f}%</div>
            <div class="verdict-caption">Estimated fraud probability &middot;
                decision threshold {threshold * 100:.0f}% &middot; risk bands: Low &lt; 10%,
                Medium 10–30%, High &ge; 30%</div>
            <div class="prob-track"><div class="prob-fill" style="width:{pct:.2f}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_explanations(hypothetical: bool = False) -> None:
    note = (
        "These numbers are hypothetical input, not a real transaction."
        if hypothetical
        else "An estimated probability is not proof of fraud."
    )
    st.markdown(
        f"""
        <div class="callout" style="margin-top:14px;">
            <b>How to read this:</b> {note} The percentage is the model's estimated
            score based on patterns learned from historical data. A <b>false alarm</b>
            (false positive) is a legitimate transaction flagged as fraud; a
            <b>missed fraud</b> (false negative) is a fraudulent transaction that was
            not flagged. Use the score together with human review.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_match(actual: int, prob: float, threshold: float) -> None:
    predicted_fraud = float(prob) >= float(threshold)
    actual_fraud = int(actual) == fm.CLASS_FRAUD
    match = predicted_fraud == actual_fraud
    actual_text = "Fraud" if actual_fraud else "Legitimate"
    predicted_text = "Potential fraud" if predicted_fraud else "Likely legitimate"

    if match and actual_fraud:
        explanation = "Both the historical label and the model say fraud. The model caught this historical case."
    elif match:
        explanation = "Both the historical label and the model say legitimate. The model correctly left this transaction alone."
    elif actual_fraud:
        explanation = (
            "The historical label is fraud, but the model did not flag it. This is a missed fraud "
            "(false negative) at this threshold — a limitation of the saved model."
        )
    else:
        explanation = (
            "The historical label is legitimate, but the model flagged it. This is a false alarm "
            "(false positive) at this threshold — a limitation of the saved model."
        )

    st.markdown(
        f"""
        <div class="match {'ok' if match else 'bad'}">
            <b>Historical label:</b> {actual_text} &nbsp;&middot;&nbsp;
            <b>Model prediction:</b> {predicted_text}<br>
            {escape(explanation)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_uploaded_frame(uploaded, key: str, *, max_mb: int | None = None) -> pd.DataFrame:
    """Cache an uploaded file in session state so interactions do not re-read it.

    The uploaded bytes are parsed once per upload (identified by Streamlit's
    ``file_id`` when available).  Raises :class:`src.fraud_ml.UploadError` with a
    user-facing message for unsupported, oversized, empty or malformed files.
    """
    signature = fm.upload_signature(uploaded)
    state_key = f"_uploaded_df_{key}"
    sig_key = f"_uploaded_sig_{key}"
    if st.session_state.get(sig_key) != signature:
        frame = fm.read_csv_upload(uploaded, max_mb=max_mb)
        # Replace the cached frame only after a successful read, so a failed
        # upload never leaves stale results attached to the new file name.
        st.session_state[state_key] = frame
        st.session_state[sig_key] = signature
    return st.session_state[state_key]


def build_result_frame(original: pd.DataFrame, probas: np.ndarray, threshold: float) -> pd.DataFrame:
    result = original.copy()
    result.columns = [str(c) for c in result.columns]
    result["Predicted Class"] = np.where(np.asarray(probas) >= float(threshold), "Fraud", "Legitimate")
    result["Fraud Probability"] = np.round(np.asarray(probas, dtype=float), 6)
    result["Risk Category"] = [fm.risk_category(p) for p in probas]
    if "Class" in result.columns:
        result = result.rename(columns={"Class": "Actual Class (provided)"})
        result["Actual Class (provided)"] = result["Actual Class (provided)"].replace(
            {0: "Legitimate", 1: "Fraud"}
        )
    return result


def render_scored_payload(payload: dict) -> None:
    """Render prediction results for a batch/lab scoring payload."""
    result = payload["result"]
    threshold = float(payload["threshold"])
    probas = result["Fraud Probability"].to_numpy(dtype=float)

    n_total = len(result)
    n_fraud = int((result["Predicted Class"] == "Fraud").sum())
    mean_prob = float(np.mean(probas)) if n_total else 0.0
    n_high = int((result["Risk Category"] == "High").sum())

    kpi_grid(
        [
            {"label": "Transactions scored", "value": f"{n_total:,}", "foot": "Rows processed by the saved model."},
            {
                "label": "Predicted fraud",
                "value": f"{n_fraud:,}",
                "delta": f"{n_fraud / max(n_total, 1) * 100:.2f}% of batch",
                "tone": "bad" if n_fraud else "neutral",
                "foot": f"At a {threshold * 100:.0f}% decision threshold.",
            },
            {
                "label": "Mean fraud probability",
                "value": f"{mean_prob * 100:.1f}%",
                "foot": "Average estimated score across the batch.",
            },
            {
                "label": "High-risk transactions",
                "value": f"{n_high:,}",
                "foot": "Estimated probability of 30% or more.",
            },
        ]
    )

    chart_col, risk_col = st.columns([3, 2], gap="large")
    with chart_col:
        with card("Score distribution", "How the model's estimated fraud probabilities are spread across this batch."):
            fig = go.Figure(
                go.Histogram(
                    x=probas * 100,
                    nbinsx=30,
                    marker=dict(color="#2563eb", line=dict(color="#ffffff", width=0.6)),
                    opacity=0.9,
                    hovertemplate="Estimated probability %{x:.1f}%<br>Transactions: %{y}<extra></extra>",
                )
            )
            fig.add_vline(
                x=threshold * 100,
                line_dash="dash",
                line_color="#dc2626",
                annotation_text=f"Threshold {threshold * 100:.0f}%",
                annotation_position="top right",
                annotation_font_color="#dc2626",
            )
            fig.update_xaxes(title="Estimated fraud probability (%)")
            fig.update_yaxes(title="Transactions")
            st.plotly_chart(style_figure(fig, height=300))
    with risk_col:
        with card("Risk levels", "Risk bands are based on estimated probability, independent of the threshold."):
            order = ["Low", "Medium", "High"]
            counts = [int((result["Risk Category"] == level).sum()) for level in order]
            fig = go.Figure(
                go.Bar(
                    x=order,
                    y=counts,
                    marker_color=["#15803d", "#d97706", "#dc2626"],
                    text=counts,
                    textposition="outside",
                    hovertemplate="%{x} risk<br>Transactions: %{y}<extra></extra>",
                )
            )
            fig.update_xaxes(title=None)
            fig.update_yaxes(title="Transactions")
            st.plotly_chart(style_figure(fig, height=300))

    preview_cols = [
        "Time",
        "Amount",
        "Actual Class (provided)",
        "Predicted Class",
        "Fraud Probability",
        "Risk Category",
    ]
    preview_cols = [c for c in preview_cols if c in result.columns]
    # Slice the first 200 rows before selecting columns/copying so a large
    # result frame is not copied in full just to render the preview.
    preview = result.head(200).loc[:, preview_cols].copy()
    if "Time" in preview:
        preview["Time"] = preview["Time"].map(fm.format_time_offset)
    if "Fraud Probability" in preview:
        preview["Fraud Probability"] = preview["Fraud Probability"].map(lambda p: f"{float(p) * 100:.2f}%")
    st.markdown('<div class="eyebrow">Prediction results</div>', unsafe_allow_html=True)
    st.dataframe(preview, hide_index=True, height=320)
    st.caption("Showing up to 200 rows. The download contains every row, with all original columns preserved.")

    try:
        csv_bytes = result.to_csv(index=False).encode("utf-8")
    except MemoryError:
        st.warning(
            "Not enough memory to prepare the CSV download. The results shown above are still valid; "
            "score a smaller file to download predictions."
        )
        return
    st.download_button(
        "Download predictions (CSV)",
        data=csv_bytes,
        file_name="sentinel_predictions.csv",
        mime="text/csv",
        width="stretch",
    )
    if "Actual Class (provided)" in result.columns:
        callout(
            "<b>Labels vs predictions:</b> 'Actual Class (provided)' was copied from your uploaded "
            "Class column. 'Predicted Class' is the saved model's output. They are kept separate and "
            "are not the same measurement.",
        )


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def _start_demo() -> None:
    st.session_state["page"] = "Fraud Detection"
    st.session_state["sample_loaded"] = True
    st.session_state["sample_seed"] = int(st.session_state.get("sample_seed", 10)) + 1
    st.session_state["sample_filter"] = "Any transaction"
    st.session_state["demo_banner"] = True
    st.rerun()


def page_overview() -> None:
    status = fm.artifact_status()
    page_header(
        "Overview",
        "A plain-language view of the transaction data, fraud prevalence and the saved model's held-out test performance.",
        badge="Dataset & model snapshot",
    )

    clean = fm.load_clean_dataset()
    clean_total = int(len(clean))
    clean_fraud = int(clean["Class"].sum()) if "Class" in clean.columns else 0
    clean_legit = clean_total - clean_fraud
    clean_fraud_pct = clean_fraud / clean_total * 100 if clean_total else 0.0

    summary = fm.load_dataset_summary()
    total = int(summary.get("total_transactions") or len(clean))
    fraud_total = int(summary.get("fraudulent_transactions") or clean_fraud)
    fraud_pct = float(summary.get("fraud_percentage") or clean_fraud_pct)
    dup_removed = int(summary.get("duplicate_rows") or 0)

    final = fm.load_final_test_results()
    final_row = final.iloc[0] if not final.empty else None
    final_recall = float(final_row.get("Recall")) if final_row is not None else None
    final_ap = float(final_row.get("Average Precision")) if final_row is not None else None

    # ---------- First-run hero ----------
    hero_col, action_col = st.columns([3, 1], gap="large")
    with hero_col:
        st.markdown(
            """
            <div class="hero">
                <div class="eyebrow">Start here</div>
                <h3>Understand a fraud check in under a minute</h3>
                <p>This dashboard uses a real saved machine-learning model. You do not need to know
                Python or machine learning to use it. Start with a historical sample, then try your own CSV.</p>
                <div class="hero-steps">
                    <div class="hero-step"><div class="n">1</div><div class="t">Try a real sample</div>
                        <div class="d">Load a transaction from the local dataset and compare the model's result with its historical label.</div></div>
                    <div class="hero-step"><div class="n">2</div><div class="t">Score a batch</div>
                        <div class="d">Upload a CSV of transactions. The app checks the columns and reports exactly what it found.</div></div>
                    <div class="hero-step"><div class="n">3</div><div class="t">Check another dataset</div>
                        <div class="d">Use the Dataset Lab to see whether a new dataset fits the saved model or needs its own workflow.</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with action_col:
        with card("Quick actions", "Everything stays on this machine."):
            if st.button("Try a demo", type="primary", width="stretch", key="overview_demo"):
                _start_demo()
            if st.button("Open Fraud Detection", width="stretch", key="overview_detection"):
                st.session_state["page"] = "Fraud Detection"
                st.rerun()
            if st.button("Open Dataset Lab", width="stretch", key="overview_lab"):
                st.session_state["page"] = "Dataset Lab"
                st.rerun()

    kpi_grid(
        [
            {
                "label": "Transactions analysed",
                "value": fmt_int(total),
                "delta": f"{fmt_int(clean_total)} after de-duplication",
                "foot": "Source: data/creditcard.csv (original dataset).",
            },
            {
                "label": "Fraudulent transactions",
                "value": fmt_int(fraud_total),
                "delta": f"{fraud_pct:.2f}% fraud rate",
                "tone": "bad",
                "foot": "Historical labels in the original dataset (Class = 1).",
            },
            {
                "label": "Cleaned transactions",
                "value": fmt_int(clean_total),
                "delta": f"{fmt_int(clean_fraud)} fraudulent",
                "foot": f"{fmt_int(dup_removed)} duplicate row(s) removed before model work.",
            },
            {
                "label": "Final test recall",
                "value": f"{final_recall:.3f}" if final_recall is not None else "Not available",
                "delta": f"Average precision {final_ap:.3f}" if final_ap is not None else "",
                "tone": "good",
                "foot": "Held-out test split, saved in results/final_test_results.csv.",
            },
        ]
    )

    donut_col, amount_col = st.columns([2, 3], gap="large")

    with donut_col:
        with card("Class balance", "Actual historical labels in the cleaned dataset."):
            fig = go.Figure(
                go.Pie(
                    labels=["Legitimate", "Fraudulent"],
                    values=[clean_legit, clean_fraud],
                    hole=0.64,
                    marker=dict(colors=["#15803d", "#dc2626"], line=dict(color="#ffffff", width=2)),
                    textinfo="none",
                    hovertemplate="%{label}<br>%{value:,} transactions<br>%{percent}<extra></extra>",
                )
            )
            fig.update_layout(
                annotations=[
                    dict(
                        text=f"<b>{clean_fraud_pct:.2f}%</b><br><span style='font-size:11px;color:#54657e'>fraud</span>",
                        showarrow=False,
                        font=dict(size=17, color="#0e1f3c"),
                    )
                ]
            )
            st.plotly_chart(style_figure(fig, height=300, legend=True))
            st.caption(
                f"{clean_fraud:,} fraudulent · {clean_legit:,} legitimate transactions after de-duplication."
            )

    with amount_col:
        with card("Transaction amounts by class", "Share of each class across amount ranges (log scale)."):
            amounts = clean["Amount"].to_numpy(dtype=float)
            legit_amounts = clean.loc[clean["Class"] == fm.CLASS_LEGITIMATE, "Amount"].to_numpy(dtype=float)
            fraud_amounts = clean.loc[clean["Class"] == fm.CLASS_FRAUD, "Amount"].to_numpy(dtype=float)
            positive = amounts[amounts > 0]
            low = float(positive.min()) if positive.size else 0.01
            high = max(float(amounts.max()), low * 10)
            bins = np.logspace(np.log10(low), np.log10(high), 34)
            legit_counts, edges = np.histogram(legit_amounts, bins=bins)
            fraud_counts, _ = np.histogram(fraud_amounts, bins=bins)
            centers = np.sqrt(edges[:-1] * edges[1:])
            widths = np.diff(edges) * 0.86
            legit_share = legit_counts / max(legit_counts.sum(), 1) * 100
            fraud_share = fraud_counts / max(fraud_counts.sum(), 1) * 100
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=centers, y=legit_share, width=widths, name="Legitimate",
                    marker_color="#15803d", opacity=0.55,
                    hovertemplate="Amount ≈ $%{x:,.2f}<br>Share of legitimate: %{y:.2f}%<extra></extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=centers, y=fraud_share, width=widths, name="Fraudulent",
                    marker_color="#dc2626", opacity=0.8,
                    hovertemplate="Amount ≈ $%{x:,.2f}<br>Share of fraud: %{y:.2f}%<extra></extra>",
                )
            )
            fig.update_layout(barmode="overlay")
            fig.update_xaxes(type="log", title="Transaction amount (log scale)")
            fig.update_yaxes(title="Share within class (%)")
            st.plotly_chart(style_figure(fig, height=300, legend=True))

    recent_col, snapshot_col = st.columns([3, 2], gap="large")
    with recent_col:
        with card(
            "Recent transactions",
            "Latest rows by Time. The column shown is the historical label, not a model prediction.",
        ):
            recent = clean.sort_values("Time", ascending=False).head(8).copy()
            recent["Time"] = recent["Time"].map(fm.format_time_offset)
            recent["Amount"] = recent["Amount"].map(fmt_money)
            recent["Actual Class"] = recent["Class"].map({0: "Legitimate", 1: "Fraud"})
            st.dataframe(recent[["Time", "Amount", "Actual Class"]], hide_index=True, height=320)
    with snapshot_col:
        with card("Saved model", "Loaded locally; no retraining happens in this dashboard."):
            model_name = "Unavailable"
            if status["model"]:
                try:
                    model_name = fm.model_display_name(fm.load_model())
                except Exception:
                    model_name = "Unavailable"
            scaler_note = "already includes scaling" if status["model"] else "not loaded"
            try:
                if status["model"] and fm.pipeline_has_scaler(fm.load_model()):
                    scaler_note = "includes a fitted scaler"
                elif status["model"]:
                    scaler_note = "no separate scaler step"
            except Exception:
                scaler_note = "status unknown"
            st.markdown(
                f"""
                <div class="summary-grid">
                    <div class="summary-item"><div class="k">Model</div><div class="v">{escape(model_name)}</div></div>
                    <div class="summary-item"><div class="k">Input features</div><div class="v">30</div></div>
                    <div class="summary-item"><div class="k">Preprocessing</div><div class="v">{escape(scaler_note)}</div></div>
                    <div class="summary-item"><div class="k">Decision threshold</div><div class="v">50% default</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "The 50% threshold is a configurable default, not a claim that it is optimal. "
                "Change it on the Fraud Detection page."
            )


# --------------------------------------------------------------------------- #
# Fraud Detection
# --------------------------------------------------------------------------- #
def _sample_mode(threshold: float) -> None:
    if "sample_loaded" not in st.session_state:
        st.session_state["sample_loaded"] = True
        st.session_state["sample_seed"] = 3
        st.session_state["sample_filter"] = "Any transaction"
        st.session_state["sample_filter_used"] = "Any transaction"

    with card(
        "Try a sample transaction",
        "Recommended for a first look. The app draws a real row from the local dataset, "
        "scores its 30 model features, and compares the result with the historical label.",
    ):
        col_filter, col_load, col_another = st.columns([2, 1, 1], vertical_alignment="bottom")
        with col_filter:
            selected_filter = st.selectbox(
                "Example type",
                list(fm.SAMPLE_CLASS_OPTIONS.keys()),
                key="sample_filter",
                help="Choose whether to draw from legitimate rows, fraud rows, or any row.",
            )
        with col_load:
            load_clicked = st.button("Load sample transaction", type="primary", width="stretch", key="sample_load")
        with col_another:
            another_clicked = st.button("Show another sample", width="stretch", key="sample_other")

    if load_clicked or another_clicked:
        st.session_state["sample_seed"] = int(st.session_state.get("sample_seed", 0)) + 1
        st.session_state["sample_loaded"] = True
        st.session_state["sample_filter_used"] = selected_filter
    elif (
        st.session_state.get("sample_loaded")
        and st.session_state.get("sample_filter_used") != selected_filter
    ):
        st.session_state["sample_seed"] = int(st.session_state.get("sample_seed", 0)) + 1
        st.session_state["sample_filter_used"] = selected_filter

    if st.session_state.get("demo_banner"):
        callout(
            "<b>Demo loaded.</b> A real historical transaction will be scored below. "
            "Historical results are for demonstration and are not guarantees of real-world performance.",
            "info",
        )
        st.session_state["demo_banner"] = False

    if not st.session_state.get("sample_loaded"):
        empty_state(
            "No sample transaction loaded yet",
            "Choose an example type and click 'Load sample transaction'. You can keep clicking "
            "'Show another sample' to try different historical rows.",
        )
        return

    seed = int(st.session_state.get("sample_seed", 1))
    class_filter = st.session_state.get("sample_filter", "Any transaction")

    try:
        row, row_index = fm.draw_sample(seed, class_filter)
    except Exception as exc:
        st.error(f"Could not draw a sample transaction: {exc}")
        return

    actual_class = int(row["Class"])
    actual_text = "Fraud" if actual_class == fm.CLASS_FRAUD else "Legitimate"
    actual_chip = "high" if actual_class == fm.CLASS_FRAUD else "low"

    with card("Sample transaction summary", "Values shown here come from the local dataset row."):
        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-item"><div class="k">Amount</div><div class="v">{fmt_money(row["Amount"])}</div></div>
                <div class="summary-item"><div class="k">Time</div><div class="v">{escape(fm.format_time_offset(row["Time"]))}</div></div>
                <div class="summary-item"><div class="k">Dataset row</div><div class="v">#{row_index:,}</div></div>
                <div class="summary-item"><div class="k">Historical label</div>
                    <div class="v"><span class="risk-chip {actual_chip}">{actual_text}</span></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Only the 30 model features are sent to the model. The historical Class label is shown "
            "for comparison and is never used as an input."
        )

    features = pd.DataFrame([row.loc[fm.EXPECTED_FEATURES].astype(float)])
    model = fm.load_model()
    scored, error = fm.score_transactions(model, features, threshold)
    if error or scored is None:
        st.error(error or "Prediction failed.")
        return

    prob = float(scored.iloc[0]["Fraud Probability"])
    render_verdict(prob, threshold)
    render_match(actual_class, prob, threshold)
    render_result_explanations()

    with st.expander("View the raw anonymized feature values"):
        st.markdown(
            "<p style='font-size:0.85rem;color:#54657e;'>V1–V28 are anonymized components produced by PCA "
            "in the original dataset. They are not merchant names, locations, card numbers or other "
            "business fields, so they are difficult for a person to interpret.</p>",
            unsafe_allow_html=True,
        )
        raw = row.loc[fm.EXPECTED_FEATURES].to_frame().T
        raw.columns = fm.EXPECTED_FEATURES
        st.dataframe(raw, hide_index=True)

    callout(
        "<b>Demo data notice:</b> this is a historical sample from the local dataset. Results are for "
        "demonstration only and are not guarantees of real-world model performance.",
        "info",
    )


def _manual_mode(threshold: float) -> None:
    callout(
        "<b>Advanced mode.</b> V1–V28 are anonymized PCA components from the original dataset — not "
        "merchant, location or card fields. Most people cannot supply meaningful values for them. "
        "Use the sample mode or a CSV batch instead unless you already have valid feature values.",
        "warn",
    )

    with st.form("manual_transaction_form"):
        st.markdown('<div class="eyebrow">Basic fields</div>', unsafe_allow_html=True)
        col_amount, col_time = st.columns(2)
        with col_amount:
            amount_value = st.number_input(
                "Transaction amount",
                min_value=0.0,
                step=0.01,
                value=None,
                placeholder="e.g. 149.62",
                format="%.2f",
                key="manual_amount",
                help="The amount charged on the transaction.",
            )
        with col_time:
            time_value = st.number_input(
                "Time offset (seconds)",
                min_value=0.0,
                step=1.0,
                value=None,
                placeholder="e.g. 0",
                key="manual_time",
                help="Seconds after the first transaction in the dataset. This is not a calendar date.",
            )

        with st.expander("Advanced anonymized features (V1–V28)", expanded=False):
            st.markdown(
                "<p style='font-size:0.84rem;color:#54657e;'>These 28 values are anonymized PCA components. "
                "They cannot be looked up from a receipt or a bank statement. They are required by the "
                "saved model because that is how the original dataset was built.</p>",
                unsafe_allow_html=True,
            )
            values: dict[str, float | None] = {}
            v_features = [f"V{i}" for i in range(1, 29)]
            for start in range(0, len(v_features), 4):
                cols = st.columns(4)
                for offset, col in enumerate(cols):
                    idx = start + offset
                    if idx >= len(v_features):
                        break
                    feature = v_features[idx]
                    with col:
                        values[feature] = st.number_input(
                            feature,
                            step=0.01,
                            value=None,
                            format="%.4f",
                            placeholder="—",
                            key=f"manual_{feature}",
                        )

        acknowledged = st.checkbox(
            "I understand this input is hypothetical and is not a real transaction record.",
            key="manual_ack",
        )
        submitted = st.form_submit_button("Run fraud prediction", type="primary", width="stretch")

    if not submitted:
        return

    if not acknowledged:
        st.warning(
            "Please confirm that the values are hypothetical before running a prediction. "
            "Manual mode is not a way to submit a real-world transaction record."
        )
        return

    collected = {"Time": time_value, "Amount": amount_value, **values}
    missing_values = [name for name in fm.EXPECTED_FEATURES if collected.get(name) is None]
    if missing_values:
        st.warning(
            f"{len(missing_values)} of the 30 required model values are empty. "
            "Most users cannot supply V1–V28 meaningfully — use the sample mode or upload a CSV batch instead."
        )
        return

    frame = pd.DataFrame([collected], columns=list(fm.EXPECTED_FEATURES)).astype(float)
    model = fm.load_model()
    scored, error = fm.score_transactions(model, frame, threshold)
    if error or scored is None:
        st.error(error or "Prediction failed.")
        return

    prob = float(scored.iloc[0]["Fraud Probability"])
    st.markdown('<div class="eyebrow" style="margin-top:20px;">Result for the hypothetical input</div>', unsafe_allow_html=True)
    render_verdict(prob, threshold)
    render_result_explanations(hypothetical=True)


def _batch_mode(threshold: float) -> None:
    with card(
        "Upload multiple transactions",
        "Use a CSV with the 30 model features. Extra columns are ignored, and a Class column is kept "
        "separate from predictions. Your uploaded file is never modified.",
    ):
        st.download_button(
            "Download CSV template",
            data=fm.build_template_csv(),
            file_name="sentinel_input_template.csv",
            mime="text/csv",
            width="stretch",
            key="batch_template",
        )
        st.caption(
            "The template contains the correct 30 columns and one real row from the local dataset as a "
            "structural example — not a claim about a new transaction."
        )
        uploaded = st.file_uploader(
            "Upload transactions CSV",
            type=["csv"],
            key="batch_file",
            max_upload_size=BATCH_UPLOAD_MAX_MB,
            help=f"CSV only, up to {BATCH_UPLOAD_MAX_MB} MB.",
        )

    if uploaded is None:
        for stale_key in ("_uploaded_df_batch", "_uploaded_sig_batch", "batch_payload"):
            st.session_state.pop(stale_key, None)
        empty_state(
            "No file uploaded yet",
            "Upload a CSV containing Time, V1–V28 and Amount. You can download the template above to "
            "start from the correct column layout.",
        )
        return

    if st.session_state.get("_uploaded_sig_batch") != fm.upload_signature(uploaded):
        # A new upload must not show results from the previous batch.
        st.session_state.pop("batch_payload", None)

    try:
        df = get_uploaded_frame(uploaded, "batch", max_mb=BATCH_UPLOAD_MAX_MB)
    except fm.UploadError as exc:
        st.error(str(exc))
        st.caption(
            "Upload a comma-separated CSV containing the 30 model features. "
            "No data was scored and nothing was saved."
        )
        return
    except MemoryError:
        st.error("Not enough memory to read this file. Try a smaller CSV or close other applications.")
        return
    except Exception as exc:
        st.error(f"Could not read the uploaded file as CSV: {exc}")
        st.caption("Check that the file is comma-separated and saved with UTF-8 encoding.")
        return

    with card("File preview", f"{uploaded.name} · {len(df):,} rows · {df.shape[1]} columns"):
        st.dataframe(df.head(10), hide_index=True, height=260)
        st.caption("Showing the first 10 rows exactly as uploaded. Nothing has been scored yet.")

    # Validation-only pass: no second full feature frame is built for a large
    # upload.  Scoring converts one chunk at a time below.
    report = fm.inspect_frame(df, build_features=False)
    checks = [
        {
            "check": "File contains transaction rows",
            "status": "pass" if report["rows"] > 0 else "fail",
            "detail": f"{report['rows']:,} row(s) parsed." if report["rows"] else "No rows were parsed.",
        },
        {
            "check": "All 30 model features present",
            "status": "fail" if report["missing"] else "pass",
            "detail": (
                "Missing: " + ", ".join(report["missing"])
                if report["missing"]
                else "Time, V1–V28 and Amount were all found."
            ),
        },
        {
            "check": "Required values are numeric",
            "status": "fail" if report["non_numeric"] else "pass",
            "detail": (
                "Non-numeric cells in: "
                + ", ".join(f"{c} ({n})" for c, n in report["non_numeric"].items())
                if report["non_numeric"]
                else "All required feature values are numeric."
            ),
        },
        {
            "check": "No missing feature values",
            "status": "fail" if report["missing_values"] else "pass",
            "detail": (
                "Blanks in: " + ", ".join(f"{c} ({n})" for c, n in report["missing_values"].items())
                if report["missing_values"]
                else "No blank values in the 30 model features."
            ),
        },
        {
            "check": "No infinite values",
            "status": "fail" if report["infinite_values"] else "pass",
            "detail": (
                "Infinite values in: " + ", ".join(f"{c} ({n})" for c, n in report["infinite_values"].items())
                if report["infinite_values"]
                else "All required values are finite."
            ),
        },
        {
            "check": "Extra columns are ignored",
            "status": "warn" if report["unexpected"] else "pass",
            "detail": (
                "Ignored, not passed to the model: " + ", ".join(report["unexpected"])
                if report["unexpected"]
                else "No unexpected columns."
            ),
        },
        {
            "check": "Class column kept separate",
            "status": "pass",
            "detail": (
                "'Class' is present and will be shown as a historical label only."
                if report["class_present"]
                else "No Class column was supplied."
            ),
        },
    ]
    with card("Validation report", "Every check runs on the uploaded file before the model is called."):
        report_table(checks)

    if not report["ok"]:
        st.error("This file cannot be scored yet. Fix the failed checks above and upload it again.")
        for message in report["errors"]:
            st.markdown(f"- {message}")
        return

    signature = f"{fm.upload_signature(uploaded)}:{threshold:.4f}"
    expected_n = report["rows"]
    if st.button(
        f"Run prediction on {expected_n:,} transactions",
        type="primary",
        key="batch_run",
        width="stretch",
    ):
        progress = st.progress(0.0, text="Starting batch scoring…")

        def update(done: int, total: int) -> None:
            progress.progress(min(done / max(total, 1), 1.0), text=f"Scored {done:,} of {total:,} transactions…")

        probas, error = fm.score_transactions_chunked(
            fm.load_model(), df, threshold, progress_callback=update
        )
        progress.empty()
        if error or probas is None:
            st.error(error or "Batch prediction failed.")
            return
        st.session_state["batch_payload"] = {
            "signature": signature,
            "threshold": threshold,
            "result": build_result_frame(df, probas, threshold),
        }

    payload = st.session_state.get("batch_payload")
    if payload and payload.get("signature") == signature:
        st.markdown('<div class="eyebrow" style="margin-top:20px;">Batch results</div>', unsafe_allow_html=True)
        render_scored_payload(payload)
    elif payload:
        callout(
            "The decision threshold or uploaded file changed. Run the prediction again to refresh the results.",
            "warn",
        )


def page_detection() -> None:
    page_header(
        "Fraud Detection",
        "Check a transaction in three ways: a real historical sample (recommended), manual entry for "
        "advanced users, or a CSV batch of many transactions.",
        badge="Model inference",
    )

    if not model_available():
        return

    with card(
        "Detection settings",
        "The decision threshold converts the estimated probability into a fraud / legitimate decision.",
    ):
        threshold = st.slider(
            "Decision threshold",
            min_value=0.05,
            max_value=0.95,
            value=fm.DEFAULT_DECISION_THRESHOLD,
            step=0.05,
            format="%.2f",
            key="decision_threshold",
            help="Lower thresholds catch more fraud but create more false alarms. "
            "The default 50% is a common starting point, not a proven optimum.",
        )
        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-item"><div class="k">Current threshold</div><div class="v">{threshold * 100:.0f}%</div></div>
                <div class="summary-item"><div class="k">Low risk</div><div class="v">&lt; 10%</div></div>
                <div class="summary-item"><div class="k">Medium risk</div><div class="v">10% – 30%</div></div>
                <div class="summary-item"><div class="k">High risk</div><div class="v">&ge; 30%</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Risk bands are shown for context. The threshold decides 'Potential fraud' vs "
            "'Likely legitimate'; it does not silently change the saved model."
        )

    tab_sample, tab_manual, tab_batch = st.tabs(
        ["Sample transaction (recommended)", "Manual entry (advanced)", "Batch CSV"]
    )
    with tab_sample:
        _sample_mode(threshold)
    with tab_manual:
        _manual_mode(threshold)
    with tab_batch:
        _batch_mode(threshold)


# --------------------------------------------------------------------------- #
# Model Performance
# --------------------------------------------------------------------------- #
def page_performance() -> None:
    page_header(
        "Model Performance",
        "Every number on this page is read from the project's saved results or reproduced from the "
        "saved model and the local held-out test labels.",
        badge="Saved results",
    )

    final = fm.load_final_test_results()
    if final.empty:
        empty_state(
            "Final test results not found",
            f"Expected {fm.FINAL_TEST_RESULTS_PATH}. Restore the saved results file; no metrics are invented.",
        )
        return

    final_row = final.iloc[0]
    final_name = fix_mojibake(final_row.get("Model", "Saved model"))
    precision = float(final_row.get("Precision", 0.0))
    recall = float(final_row.get("Recall", 0.0))
    f1 = float(final_row.get("F1-score", 0.0))
    ap = float(final_row.get("Average Precision", 0.0))

    kpi_grid(
        [
            {
                "label": "Precision",
                "value": fmt_metric(precision),
                "delta": f"{precision * 100:.1f}% of fraud alerts were correct",
                "tone": "good",
                "foot": "Final held-out test split.",
            },
            {
                "label": "Recall",
                "value": fmt_metric(recall),
                "delta": f"{recall * 100:.1f}% of real fraud was caught",
                "tone": "good",
                "foot": "Final held-out test split.",
            },
            {
                "label": "F1-score",
                "value": fmt_metric(f1),
                "foot": "Balance of precision and recall.",
            },
            {
                "label": "Average precision",
                "value": fmt_metric(ap),
                "foot": "Ranking quality across all thresholds.",
            },
        ]
    )

    callout(
        "<b>Final test vs validation:</b> validation numbers were used to compare and choose a model. "
        "The final test numbers were measured once on data held out until after model selection, so they "
        "are the more realistic estimate. They are different measurements and should not be mixed.",
        "info",
    )

    with card("What the numbers mean", "Short definitions for readers who do not work with machine learning."):
        left_col, right_col = st.columns(2, gap="large")
        with left_col:
            definition(
                "Precision",
                "Of all transactions the model flagged as fraud, the share that really were fraud. "
                "High precision means fewer false alarms.",
            )
            st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
            definition(
                "F1-score",
                "A single balance between precision and recall. Useful when fraud is rare and both "
                "error types matter.",
            )
        with right_col:
            definition(
                "Recall",
                "Of all real fraud cases, the share the model caught. High recall means fewer missed frauds.",
            )
            st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
            definition(
                "Average precision",
                "Summarises the precision/recall trade-off across every possible threshold. "
                "Higher is better for rare fraud.",
            )

    comparison = fm.load_model_comparison()
    if not comparison.empty:
        with card(
            "Validation comparison",
            "Both models measured on the validation split (results/model_comparison.csv). "
            "This is how the saved model was selected — not the final test score.",
        ):
            display_cmp = comparison.copy()
            display_cmp["Model"] = display_cmp["Model"].map(fix_mojibake)
            for metric in ["Precision", "Recall", "F1-score", "Average Precision"]:
                if metric in display_cmp.columns:
                    display_cmp[metric] = display_cmp[metric].map(fmt_metric)
            st.dataframe(display_cmp, hide_index=True)

            metric_names = ["Precision", "Recall", "F1-score", "Average Precision"]
            colors = {"Random Forest": "#2563eb", "Logistic Regression": "#0e1f3c"}
            fig = go.Figure()
            for _, row in comparison.iterrows():
                model_name = fix_mojibake(row["Model"])
                fig.add_trace(
                    go.Bar(
                        name=model_name,
                        x=metric_names,
                        y=[float(row[m]) for m in metric_names],
                        marker_color=colors.get(model_name, "#64748b"),
                        text=[f"{float(row[m]):.3f}" for m in metric_names],
                        textposition="outside",
                    )
                )
            fig.update_layout(barmode="group")
            fig.update_yaxes(range=[0, 1.08], title="Score")
            st.plotly_chart(style_figure(fig, height=330, legend=True))

    try:
        cm = fm.compute_final_test_confusion()
    except Exception as exc:
        callout(f"Confusion matrix could not be reproduced: {escape(str(exc))}", "danger")
        cm = None

    if cm is not None:
        with card(
            "Confusion matrix",
            f"Genuine counts reproduced from the saved {fm.MODEL_DISPLAY_NAMES.get(cm['model'], cm['model'])} "
            f"on {cm['n_test']:,} held-out test rows at a {cm['threshold'] * 100:.0f}% threshold. "
            "Rows are historical labels; columns are model predictions.",
        ):
            z = [[1, 2], [3, 4]]  # tn, fp / fn, tp — categorical colours, not magnitude
            counts = [
                [f"{cm['tn']:,}", f"{cm['fp']:,}"],
                [f"{cm['fn']:,}", f"{cm['tp']:,}"],
            ]
            fig = go.Figure(
                data=go.Heatmap(
                    z=z,
                    x=["Predicted legitimate", "Predicted fraud"],
                    y=["Actual legitimate", "Actual fraud"],
                    zmin=1,
                    zmax=4,
                    colorscale=[
                        [0.0, "#dbeafe"],
                        [0.1666, "#dbeafe"],
                        [0.1667, "#fef3c7"],
                        [0.4999, "#fef3c7"],
                        [0.5, "#fee2e2"],
                        [0.8332, "#fee2e2"],
                        [0.8333, "#dcfce7"],
                        [1.0, "#dcfce7"],
                    ],
                    text=counts,
                    texttemplate="%{text}",
                    textfont={"size": 17, "color": "#0e1f3c"},
                    showscale=False,
                    hovertemplate="%{y}<br>%{x}<br>Count: %{text}<extra></extra>",
                )
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style_figure(fig, height=300))

            q_cols = st.columns(4)
            quadrants = [
                ("True negative", cm["tn"], "Legitimate transactions correctly left alone."),
                ("False positive", cm["fp"], "False alarms: legitimate transactions flagged as fraud."),
                ("False negative", cm["fn"], "Missed fraud: fraudulent transactions not flagged."),
                ("True positive", cm["tp"], "Fraudulent transactions correctly flagged."),
            ]
            for col, (label, value, explanation) in zip(q_cols, quadrants):
                with col:
                    st.markdown(
                        f'<div class="summary-item"><div class="k">{escape(label)}</div>'
                        f'<div class="v">{value:,}</div>'
                        f'<div class="kpi-foot" style="color:#54657e;margin-top:6px;">{escape(explanation)}</div></div>',
                        unsafe_allow_html=True,
                    )

    # Precision-recall curve + threshold trade-off, both on the validation split.
    try:
        data = fm.validation_probabilities()
        tradeoff = fm.threshold_tradeoff()
    except Exception as exc:
        callout(f"Validation detail could not be reproduced: {escape(str(exc))}", "danger")
        return

    curve_col, threshold_col = st.columns([3, 2], gap="large")
    with curve_col:
        with card(
            "Precision–recall curve",
            f"Validation split ({data['n']:,} rows). Shows the trade-off as the decision threshold moves.",
        ):
            precision_curve, recall_curve, _ = precision_recall_curve(data["y"], data["proba"])
            curve_ap = average_precision_score(data["y"], data["proba"])
            predicted = data["proba"] >= fm.DEFAULT_DECISION_THRESHOLD
            tp = int(((predicted == 1) & (data["y"] == 1)).sum())
            fp = int(((predicted == 1) & (data["y"] == 0)).sum())
            fn = int(((predicted == 0) & (data["y"] == 1)).sum())
            p_at = tp / (tp + fp) if (tp + fp) else 0.0
            r_at = tp / (tp + fn) if (tp + fn) else 0.0

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=recall_curve,
                    y=precision_curve,
                    mode="lines",
                    name="Saved model",
                    line=dict(color="#2563eb", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(37,99,235,0.08)",
                    hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[r_at],
                    y=[p_at],
                    mode="markers",
                    name="At 50% threshold",
                    marker=dict(size=10, color="#dc2626", line=dict(color="#ffffff", width=2)),
                    hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>",
                )
            )
            fig.update_layout(
                annotations=[
                    dict(
                        x=0.98,
                        y=0.82,
                        xref="paper",
                        yref="paper",
                        xanchor="right",
                        text=f"Average precision = {curve_ap:.3f}",
                        showarrow=False,
                        font=dict(size=11, color="#54657e"),
                    )
                ]
            )
            fig.update_xaxes(title="Recall (share of fraud caught)", range=[0, 1.02])
            fig.update_yaxes(title="Precision (share of alerts that are correct)", range=[0, 1.05])
            st.plotly_chart(style_figure(fig, height=340, legend=True))
            st.caption("Validation split — this curve is not the final test measurement.")

    with threshold_col:
        with card(
            "Threshold trade-off",
            "Validation split only. The 50% default is highlighted; it is not assumed to be optimal.",
        ):
            visible_columns = ["Threshold", "Precision", "Recall", "F1-score"]
            styled = tradeoff[visible_columns].style.format(
                {
                    "Threshold": lambda v: f"{float(v) * 100:.0f}%",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1-score": "{:.3f}",
                }
            ).apply(
                lambda row: [
                    "background-color: #eaf1fe; font-weight: 700"
                    if abs(float(row["Threshold"]) - fm.DEFAULT_DECISION_THRESHOLD) < 1e-9
                    else ""
                    for _ in row
                ],
                axis=1,
            )
            st.dataframe(styled, hide_index=True, height=300)
            row_50 = tradeoff[
                (tradeoff["Threshold"] - fm.DEFAULT_DECISION_THRESHOLD).abs() < 1e-9
            ]
            if not row_50.empty:
                flagged = int(row_50.iloc[0]["Predicted fraud"])
                missed = int(row_50.iloc[0]["Missed fraud"])
                st.caption(
                    f"At the 50% default: {flagged} transactions flagged and {missed} fraud cases missed "
                    "on the validation split."
                )
            st.caption(
                "Lower thresholds flag more transactions (higher recall, lower precision). "
                "Choose based on the real cost of false alarms versus missed fraud."
            )


# --------------------------------------------------------------------------- #
# Transaction Explorer
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=None)
def get_historical_scores() -> pd.DataFrame:
    model = fm.load_model()
    clean = fm.load_clean_dataset()
    probas, error = fm.score_transactions_chunked(model, clean.loc[:, fm.EXPECTED_FEATURES], fm.DEFAULT_DECISION_THRESHOLD)
    if error or probas is None:
        return pd.DataFrame()
    out = clean.copy()
    out["Fraud Probability"] = probas
    out["Risk Category"] = [fm.risk_category(p) for p in probas]
    return out


def page_explorer() -> None:
    page_header(
        "Transaction Explorer",
        "Browse historical transactions with the saved model's estimated risk scores. "
        "Historical labels and model predictions are shown in separate columns.",
        badge="Exploratory",
    )

    if not model_available():
        return

    with st.spinner("Scoring historical transactions — this runs once per session…"):
        hist = get_historical_scores()
    if hist.empty:
        st.error("Historical predictions could not be computed with the saved model.")
        return

    with card("Filters", "The table and charts below update to match these filters."):
        col_pred, col_risk, col_threshold = st.columns(3)
        with col_pred:
            pred_filter = st.selectbox(
                "Predicted class", ["All transactions", "Predicted fraud", "Predicted legitimate"], key="explorer_pred"
            )
        with col_risk:
            risk_filter = st.selectbox(
                "Risk category", ["All risk levels", "Low", "Medium", "High"], key="explorer_risk"
            )
        with col_threshold:
            explorer_threshold = st.slider(
                "Decision threshold", 0.05, 0.95, fm.DEFAULT_DECISION_THRESHOLD, 0.05,
                format="%.2f", key="explorer_threshold",
            )
        col_min, col_max, col_sort = st.columns(3)
        amount_max = float(hist["Amount"].max()) if len(hist) else 0.0
        with col_min:
            amount_min = st.number_input(
                "Minimum amount", min_value=0.0, max_value=max(amount_max, 1.0),
                value=0.0, step=1.0, key="explorer_min",
            )
        with col_max:
            amount_max_input = st.number_input(
                "Maximum amount", min_value=0.0, max_value=max(amount_max, 1.0),
                value=amount_max, step=1.0, key="explorer_max",
            )
        with col_sort:
            sort_by = st.selectbox("Sort by", ["Fraud probability", "Amount", "Time"], key="explorer_sort")

    filtered = hist.copy()
    filtered["Predicted Class"] = (filtered["Fraud Probability"] >= float(explorer_threshold)).astype(int)

    if pred_filter == "Predicted fraud":
        filtered = filtered[filtered["Predicted Class"] == fm.CLASS_FRAUD]
    elif pred_filter == "Predicted legitimate":
        filtered = filtered[filtered["Predicted Class"] == fm.CLASS_LEGITIMATE]
    if risk_filter != "All risk levels":
        filtered = filtered[filtered["Risk Category"] == risk_filter]
    if float(amount_min) > 0:
        filtered = filtered[filtered["Amount"] >= float(amount_min)]
    filtered = filtered[filtered["Amount"] <= float(amount_max_input)]

    sort_col = {"Fraud probability": "Fraud Probability", "Amount": "Amount", "Time": "Time"}[sort_by]
    filtered = filtered.sort_values(by=sort_col, ascending=sort_by in ("Amount", "Time")).reset_index(drop=True)

    n_match = len(filtered)
    n_pred_fraud = int((filtered["Predicted Class"] == fm.CLASS_FRAUD).sum())
    n_actual_fraud = int((filtered["Class"] == fm.CLASS_FRAUD).sum()) if "Class" in filtered else 0
    mean_prob = float(filtered["Fraud Probability"].mean()) if n_match else 0.0

    kpi_grid(
        [
            {"label": "Matching transactions", "value": f"{n_match:,}", "foot": "After all filters."},
            {
                "label": "Predicted fraud",
                "value": f"{n_pred_fraud:,}",
                "delta": f"{n_pred_fraud / max(n_match, 1) * 100:.2f}% of matches",
                "tone": "bad" if n_pred_fraud else "neutral",
            },
            {
                "label": "Historical fraud",
                "value": f"{n_actual_fraud:,}",
                "foot": "Actual labels, not predictions.",
            },
            {"label": "Mean estimated probability", "value": f"{mean_prob * 100:.1f}%"},
        ]
    )

    chart_col, risk_col = st.columns([3, 2], gap="large")
    with chart_col:
        with card(
            "Amount vs model score",
            "A sample of up to 3,000 matching transactions. Colour shows the historical label; "
            "the dashed line is the current threshold.",
        ):
            if n_match:
                sample = filtered.sample(min(3000, n_match), random_state=42)
                fig = go.Figure()
                for class_value, name, color in [
                    (fm.CLASS_LEGITIMATE, "Legitimate (historical)", "#15803d"),
                    (fm.CLASS_FRAUD, "Fraud (historical)", "#dc2626"),
                ]:
                    subset = sample[sample["Class"] == class_value]
                    if subset.empty:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=subset["Amount"].clip(lower=0.01),
                            y=subset["Fraud Probability"],
                            mode="markers",
                            name=name,
                            marker=dict(size=6, color=color, opacity=0.55, line=dict(width=0)),
                            hovertemplate="Amount $%{x:,.2f}<br>Estimated probability %{y:.1%}<extra></extra>",
                        )
                    )
                fig.add_hline(
                    y=float(explorer_threshold),
                    line_dash="dash",
                    line_color="#2563eb",
                    annotation_text=f"Threshold {explorer_threshold * 100:.0f}%",
                    annotation_position="top left",
                    annotation_font_color="#2563eb",
                )
                fig.update_xaxes(type="log", title="Amount (log scale)")
                fig.update_yaxes(title="Estimated fraud probability", tickformat=".0%", range=[-0.02, 1.02])
                st.plotly_chart(style_figure(fig, height=340, legend=True))
            else:
                empty_state("No matching transactions", "Widen the filters to see results.")
    with risk_col:
        with card("Risk levels in the filtered set", "Share of matching transactions in each estimated-probability band."):
            order = ["Low", "Medium", "High"]
            palette = {"Low": "#15803d", "Medium": "#d97706", "High": "#dc2626"}
            counts = [int((filtered["Risk Category"] == level).sum()) for level in order]
            total = max(n_match, 1)
            fig = go.Figure()
            for level, count in zip(order, counts):
                share = count / total * 100
                fig.add_trace(
                    go.Bar(
                        y=["Risk mix"],
                        x=[share],
                        orientation="h",
                        name=level,
                        marker_color=palette[level],
                        text=[f"{count:,}" if share >= 4 else ""],
                        textposition="inside",
                        insidetextanchor="middle",
                        hovertemplate=f"{level} risk<br>{count:,} transactions ({share:.2f}%)<extra></extra>",
                    )
                )
            fig.update_layout(barmode="stack")
            fig.update_xaxes(range=[0, 100], title="Share of matching transactions (%)", ticksuffix="%")
            fig.update_yaxes(showticklabels=False)
            st.plotly_chart(style_figure(fig, height=260, legend=True))
            summary = " · ".join(
                f"{level}: {count:,} ({count / total * 100:.2f}%)" for level, count in zip(order, counts)
            )
            st.caption(summary)

    with card(
        "Historical transactions",
        "Historical labels are in 'Actual Class'; model output is in 'Predicted Class'. "
        "Showing up to 300 rows; the download contains every matching row.",
    ):
        display = filtered.head(300).copy()
        display["Time"] = display["Time"].map(fm.format_time_offset)
        display["Actual Class"] = display["Class"].map({0: "Legitimate", 1: "Fraud"})
        display["Predicted Class"] = display["Predicted Class"].map({0: "Legitimate", 1: "Fraud"})
        display["Fraud Probability"] = display["Fraud Probability"].map(lambda p: f"{float(p) * 100:.2f}%")
        table_cols = ["Time", "Amount", "Actual Class", "Predicted Class", "Fraud Probability", "Risk Category"]
        st.dataframe(display[table_cols], hide_index=True, height=360)

        export = filtered.copy()
        export["Time"] = export["Time"].map(fm.format_time_offset)
        st.download_button(
            "Download filtered transactions (CSV)",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name="sentinel_transactions.csv",
            mime="text/csv",
            width="stretch",
        )


# --------------------------------------------------------------------------- #
# Dataset Lab
# --------------------------------------------------------------------------- #
KAGGLE_2023_URL = "https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023"
PAYSIM_URL = "https://www.kaggle.com/datasets/ealaxi/paysim1"


def _exploratory_tab(df: pd.DataFrame, profile: dict) -> None:
    callout(
        "<b>Exploratory analysis only.</b> These statistics describe the uploaded file. "
        "The saved model has not been used on this dataset.",
        "info",
    )

    kpi_grid(
        [
            {"label": "Rows", "value": fmt_int(profile["rows"])},
            {"label": "Columns", "value": fmt_int(profile["columns"]), "foot": f"{profile['numeric_columns']} numeric."},
            {
                "label": "Missing cells",
                "value": fmt_int(profile["missing_total"]),
                "tone": "good" if profile["missing_total"] == 0 else "bad",
            },
            {
                "label": "Duplicate rows",
                "value": fmt_int(profile["duplicate_rows"]),
                "foot": "Exact duplicate rows in the file.",
            },
        ]
    )

    left, right = st.columns(2, gap="large")
    with left:
        with card("Class distribution", "Historical label counts in the uploaded dataset, if a label column exists."):
            if profile["class_counts"]:
                labels = list(profile["class_counts"].keys())
                values = list(profile["class_counts"].values())
                target = profile["target"] or "Class"
                fig = go.Figure(
                    go.Bar(
                        x=labels,
                        y=values,
                        marker_color="#2563eb",
                        text=values,
                        textposition="outside",
                    )
                )
                fig.update_xaxes(title=f"{target} value")
                fig.update_yaxes(title="Rows")
                st.plotly_chart(style_figure(fig, height=300))
                percentages = profile["class_percentages"] or {}
                summary = " · ".join(
                    f"{label}: {percentages.get(label, 0):.2f}%"
                    for label in labels
                )
                st.caption(summary)
                if profile.get("class_counts_truncated"):
                    st.caption(
                        f"Showing the 10 largest of {profile['class_count_total']} distinct classes. "
                        "Percentages are of all labelled rows."
                    )
            else:
                empty_state("No label column detected", "No Class / isFraud / label column was found in this file.")
    with right:
        with card("Missing values", "The 12 columns with the most missing cells."):
            if profile["missing_by_column"]:
                cols = list(reversed(list(profile["missing_by_column"].keys())))
                counts = [profile["missing_by_column"][c] for c in cols]
                fig = go.Figure(go.Bar(x=counts, y=cols, orientation="h", marker_color="#d97706"))
                fig.update_xaxes(title="Missing cells")
                fig.update_yaxes(title=None)
                st.plotly_chart(style_figure(fig, height=300))
            else:
                empty_state("No missing values", "Every cell in this uploaded file is populated.")

    with card("Column schema", "Column types and fill status exactly as uploaded."):
        column_names = [str(c) for c in df.columns]
        schema = pd.DataFrame(
            {
                "Column": column_names,
                "Type": [profile["dtypes"].get(c, "—") for c in column_names],
                "Missing": [profile["missing_counts"].get(c, 0) for c in column_names],
                "Unique values": [profile["unique_by_column"].get(c, "—") for c in column_names],
            }
        )
        st.dataframe(schema, hide_index=True, height=320)


def _score_tab(df: pd.DataFrame, signature: str) -> None:
    callout(
        "<b>This will use the saved model.</b> It sends only Time, V1–V28 and Amount to the model. "
        "No retraining or model replacement happens. Predictions are kept separate from any labels.",
        "warn",
    )
    # Validation-only: features are converted chunk by chunk during scoring.
    report = fm.inspect_frame(df, build_features=False)
    if not report["ok"]:
        st.error("This dataset cannot be scored by the saved model.")
        for message in report["errors"]:
            st.markdown(f"- {message}")
        return

    if st.button(
        f"Score {report['rows']:,} transactions with the saved model",
        type="primary",
        key="lab_score",
        width="stretch",
    ):
        progress = st.progress(0.0, text="Starting…")

        def update(done: int, total: int) -> None:
            progress.progress(min(done / max(total, 1), 1.0), text=f"Scored {done:,} of {total:,} transactions…")

        try:
            probas, error = fm.score_transactions_chunked(
                fm.load_model(), df, fm.DEFAULT_DECISION_THRESHOLD, progress_callback=update
            )
        finally:
            progress.empty()
        if error or probas is None:
            st.error(error or "Scoring failed.")
            return
        try:
            st.session_state["lab_payload"] = {
                "signature": signature,
                "threshold": fm.DEFAULT_DECISION_THRESHOLD,
                "result": build_result_frame(df, probas, fm.DEFAULT_DECISION_THRESHOLD),
            }
        except MemoryError:
            st.error(
                "The model scored every row, but there is not enough memory to build the results table. "
                "Upload a smaller file to see and download predictions."
            )
            return

    payload = st.session_state.get("lab_payload")
    if payload and payload.get("signature") != signature:
        payload = None
    if payload:
        render_scored_payload(payload)
    else:
        empty_state(
            "Ready to score",
            "Click the button above to run the saved model on this dataset. "
            "Nothing is scored automatically.",
        )


def _training_tab(df: pd.DataFrame) -> None:
    callout(
        "<b>Separate model, separate files.</b> This option prepares a standalone training script for "
        "this dataset's own schema. It does not run now, does not retrain the saved model and writes only "
        "to models/experiments/ and results/experiments/ when you run it.",
        "info",
    )
    target_guess = fm.detect_target_column(df)
    options = [str(c) for c in df.columns]
    default_index = options.index(target_guess) if target_guess in options else 0
    target = st.selectbox(
        "Label column to predict",
        options,
        index=default_index,
        key="lab_target",
        help="Choose the column that contains the fraud / legitimate label for this dataset.",
    )
    script = fm.prepare_training_script(target)
    st.download_button(
        "Download separate training script",
        data=script.encode("utf-8"),
        file_name="train_new_dataset.py",
        mime="text/x-python",
        width="stretch",
        key="lab_download_script",
    )
    st.caption(
        "Edit DATA_PATH at the top of the downloaded script so it points at your CSV, then run it. "
        "It prints validation and test metrics and saves new artifacts without touching the existing ones."
    )
    with st.expander("Preview the generated script"):
        st.code(script, language="python")


def page_dataset_lab() -> None:
    page_header(
        "Dataset Lab",
        "Inspect another public fraud dataset, check whether it fits the saved model, and keep "
        "exploration separate from prediction.",
        badge="Compatibility & exploration",
    )

    source_left, source_right = st.columns(2, gap="large")
    with source_left:
        with card("Kaggle Credit Card Fraud 2023", "V1–V28, Amount, Class and an id column — but no Time."):
            st.markdown(
                "<p style='font-size:0.85rem;color:#54657e;line-height:1.6;'>This dataset has the same "
                "anonymized feature names as the original data, but it does not contain the <b>Time</b> "
                "column the saved model was trained with. The <b>id</b> column is an identifier and must "
                "not be substituted for Time.</p>",
                unsafe_allow_html=True,
            )
            st.link_button("Open dataset on Kaggle", KAGGLE_2023_URL, width="stretch")
            st.caption("Check the dataset page for its licence and usage terms before downloading.")
    with source_right:
        with card("PaySim (optional future dataset)", "A mobile-money simulation with a completely different schema."):
            st.markdown(
                "<p style='font-size:0.85rem;color:#54657e;line-height:1.6;'>PaySim uses columns such as "
                "<b>step</b>, <b>type</b>, <b>amount</b>, account balances and <b>isFraud</b>. It is "
                "<b>not compatible</b> with the credit-card model and cannot be scored by it — it needs "
                "its own training workflow.</p>",
                unsafe_allow_html=True,
            )
            st.link_button("Open PaySim on Kaggle", PAYSIM_URL, width="stretch")
            st.caption("Optional reference only. No PaySim file is bundled with this project.")

    upload_limit_mb = fm.dataset_upload_limit_mb()
    with card(
        "Upload a dataset to inspect",
        f"CSV only, up to {upload_limit_mb} MB. The file is read locally and never uploaded anywhere. "
        "Compatibility is checked without calling the model.",
    ):
        callout(
            f"<b>Upload limit: {upload_limit_mb} MB (CSV only).</b> The whole file is loaded into "
            "memory for analysis, so a dataset close to the limit may require several gigabytes of RAM. "
            "If analysis fails for lack of memory, close other applications or upload a smaller file.",
            "info",
        )
        uploaded = st.file_uploader(
            "Upload dataset CSV",
            type=["csv"],
            key="lab_file",
            max_upload_size=upload_limit_mb,
            help=f"CSV only. Maximum {upload_limit_mb} MB. Large files can require significant RAM.",
        )

    if uploaded is None:
        # Release any parsed frame/statistics when the upload is removed.
        for stale_key in ("_uploaded_df_lab", "_uploaded_sig_lab", "_lab_analysis", "lab_payload"):
            st.session_state.pop(stale_key, None)
        empty_state(
            "No dataset uploaded yet",
            "Download the 2023 dataset from Kaggle (or another CSV), then upload it here. "
            "You will get a compatibility report and exploratory statistics before anything is scored.",
        )
        return

    signature = fm.upload_signature(uploaded)
    if st.session_state.get("_uploaded_sig_lab") != signature:
        # Clear the previous dataset's results before attempting to parse the
        # new file, so a failed upload cannot leave stale statistics visible.
        st.session_state.pop("lab_payload", None)
        st.session_state.pop("_lab_analysis", None)

    try:
        df = get_uploaded_frame(uploaded, "lab", max_mb=upload_limit_mb)
    except fm.UploadError as exc:
        st.error(str(exc))
        st.caption(
            f"Only CSV files up to {upload_limit_mb} MB are accepted, and the whole file must be "
            "readable. Nothing was analysed or scored."
        )
        return
    except MemoryError:
        st.error(
            "Not enough memory to read this file. Close other applications, or upload a smaller CSV. "
            "No partial statistics are shown."
        )
        return
    except Exception as exc:
        st.error(f"Could not read the uploaded file as CSV: {exc}")
        st.caption("Check that the file is comma-separated and saved with UTF-8 encoding.")
        return

    with card(
        "File preview",
        f"{uploaded.name} · showing first {min(len(df), fm.DATASET_PREVIEW_ROWS):,} of {len(df):,} rows",
    ):
        # Only the first 100 rows are rendered; statistics below use every row.
        st.dataframe(df.head(fm.DATASET_PREVIEW_ROWS), hide_index=True, height=320)
        if len(df) > fm.DATASET_PREVIEW_ROWS:
            st.caption(
                f"Only the first {fm.DATASET_PREVIEW_ROWS:,} rows are displayed. Every statistic and "
                f"compatibility check below is computed from all {len(df):,} rows — nothing is truncated."
            )
        else:
            st.caption(f"Showing all {len(df):,} rows. Every statistic below uses the whole file.")

    cached = st.session_state.get("_lab_analysis")
    if not cached or cached.get("signature") != signature:
        # A new upload invalidates results computed for the previous dataset.
        st.session_state.pop("lab_payload", None)
        try:
            profile = fm.dataset_profile(df)
            compatibility = fm.check_compatibility(df)
        except MemoryError:
            st.session_state.pop("_lab_analysis", None)
            st.error(
                "Not enough memory to analyse this dataset. Close other applications, or upload a "
                "smaller CSV. No statistics are shown because the full file could not be analysed."
            )
            return
        cached = {
            "signature": signature,
            "profile": profile,
            "compatibility": compatibility,
        }
        st.session_state["_lab_analysis"] = cached

    profile = cached["profile"]
    compatibility = cached["compatibility"]

    with card("Dataset profile", f"{uploaded.name} · {len(df):,} rows · {df.shape[1]} columns"):
        class_value = "—"
        if profile["class_counts"]:
            target = profile["target"] or "Class"
            percentages = profile["class_percentages"] or {}
            n_classes = int(profile.get("class_count_total") or len(profile["class_counts"]))
            if n_classes == 2:
                fraud_like = [k for k in profile["class_counts"] if str(k) in {"1", "1.0", "True", "Fraud"}]
                share = percentages.get(fraud_like[0], 0.0) if fraud_like else None
                class_value = f"{share:.2f}% positive" if share is not None else "2 classes"
            elif n_classes == 1:
                class_value = f"Only class {next(iter(profile['class_counts']))}"
            else:
                class_value = f"{n_classes} classes"
            label_foot = f"Detected label column: {target}"
        else:
            label_foot = "No label column detected."
        kpi_grid(
            [
                {"label": "Rows", "value": fmt_int(profile["rows"])},
                {"label": "Columns", "value": fmt_int(profile["columns"])},
                {
                    "label": "Missing cells",
                    "value": fmt_int(profile["missing_total"]),
                    "tone": "good" if profile["missing_total"] == 0 else "bad",
                },
                {"label": "Class balance", "value": class_value, "foot": label_foot},
            ]
        )

    with card("Compatibility report", "Checks compare this file with the saved model's 30-feature schema."):
        if compatibility["compatible"]:
            callout(
                "<b>Compatible.</b> This file has the 30 features the saved model needs. You can score it "
                "explicitly below, and the exploratory statistics stay separate from predictions.",
                "ok",
            )
        else:
            reasons = [
                c["detail"] for c in compatibility["checks"] if c["status"] == "fail"
            ]
            summary = " ".join(reasons[:2]) if reasons else "Required model features are missing."
            callout(
                "<b>Not compatible with the saved model.</b> " + escape(summary)
                + " The saved model was not used on this dataset.",
                "danger",
            )
        report_table(compatibility["checks"])

    if compatibility["compatible"]:
        tab_score, tab_explore, tab_train = st.tabs(
            ["Score with saved model", "Exploratory analysis", "Prepare separate model"]
        )
        with tab_score:
            _score_tab(df, signature)
    else:
        tab_explore, tab_train = st.tabs(["Exploratory analysis", "Prepare separate model"])
    with tab_explore:
        _exploratory_tab(df, profile)
    with tab_train:
        _training_tab(df)


# --------------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------------- #
PAGES = ["Overview", "Fraud Detection", "Model Performance", "Transaction Explorer", "Dataset Lab"]

if "page" not in st.session_state or st.session_state["page"] not in PAGES:
    st.session_state["page"] = "Overview"

page = st.session_state["page"]

st.sidebar.markdown(
    """
    <div class="brand">
        <div class="brand-mark">S</div>
        <div>
            <div class="brand-name">Sentinel</div>
            <div class="brand-sub">Fraud Analytics</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

for page_name in PAGES:
    if st.sidebar.button(
        page_name,
        key=f"nav_{page_name}",
        type="primary" if page_name == page else "secondary",
        width="stretch",
    ):
        st.session_state["page"] = page_name
        st.rerun()

status = fm.artifact_status()
if status["model"]:
    try:
        sidebar_model = fm.model_display_name(fm.load_model())
        sidebar_note = "Saved pipeline loaded locally"
    except Exception:
        sidebar_model = "Model unavailable"
        sidebar_note = "Artifact exists but could not be loaded"
else:
    sidebar_model = "Model not found"
    sidebar_note = "Restore models/fraud_detection_model.joblib"

st.sidebar.markdown(
    f"""
    <div class="sidebar-card">
        <div class="sc-eyebrow">Saved model</div>
        <div class="sc-title">{escape(sidebar_model)}</div>
        <div class="sc-sub">{escape(sidebar_note)}</div>
        <div class="sc-sub" style="margin-top:8px;">No retraining on this page</div>
    </div>
    <div class="sidebar-note">
        Local-only dashboard. Transaction data never leaves this machine.
        Predictions are estimates, not proof of fraud.
    </div>
    """,
    unsafe_allow_html=True,
)

if not status["feature_names"]:
    st.sidebar.warning("Feature-name file missing; using the documented 30-feature schema.")

try:
    if page == "Overview":
        page_overview()
    elif page == "Fraud Detection":
        page_detection()
    elif page == "Model Performance":
        page_performance()
    elif page == "Transaction Explorer":
        page_explorer()
    elif page == "Dataset Lab":
        page_dataset_lab()
except FileNotFoundError as exc:
    st.error(str(exc))
except Exception as exc:  # pragma: no cover - final safety net
    st.error(f"Something went wrong while rendering this page: {exc}")
    st.caption("No model or result files were modified. Check the terminal for the full traceback.")
