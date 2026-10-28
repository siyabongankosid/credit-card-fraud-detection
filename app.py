import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import fraud_ml as fm

st.set_page_config(
    page_title="Sentinel Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
:root {
    --bg: #f4f6fa;
    --card: #ffffff;
    --border: #e5eaf3;
    --border-strong: #d8e0ec;
    --text: #0e1f3c;
    --muted: #54657e;
    --faint: #8b99b1;
    --accent: #2563eb;
    --accent-dark: #1e4fd8;
    --accent-soft: #eaf1fe;
    --fraud: #dc2626;
    --fraud-soft: #fef2f2;
    --legit: #15803d;
    --legit-soft: #f0fdf4;
    --amber: #b45309;
    --radius: 16px;
}

body { background-color: #f4f6fa !important; color: #0e1f3c !important; }
h1, h2, h3, h4 { color: #0e1f3c !important; }

.stMarkdown p, .stMarkdown li { color: #43536b !important; line-height: 1.65; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #0e1f3c !important; }

.page-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 26px; flex-wrap: wrap; }
.page-header h1 { font-size: 1.85rem; font-weight: 700; margin: 0; }
.page-subtitle { color: #54657e; margin: 8px 0 0; font-size: 0.95rem; line-height: 1.55; max-width: 780px; }
.pill {
    display: inline-block;
    background: #eaf1fe;
    color: #1e4fd8;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 6px 12px;
    border-radius: 999px;
    white-space: nowrap;
}

section[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e5eaf3 !important; }
section[data-testid="stSidebar"] > div { background: transparent !important; }
section[data-testid="stSidebar"] .stMarkdown p { color: #54657e !important; }

.stButton > button { border-radius: 10px !important; font-weight: 500; min-height: 42px; }
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #43536b !important;
    border: 1px solid transparent !important;
}
section[data-testid="stSidebar"] .stButton > button:hover { background: #f2f6fc; color: #0e1f3c; }
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] { background: #2563eb; color: #ffffff; }
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-tertiary"] { background: transparent; }

.brand { display: flex; align-items: center; gap: 12px; padding: 6px 4px 18px; }
.brand-mark {
    width: 38px; height: 38px; border-radius: 12px; background: #2563eb; color: #ffffff;
    display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.05rem;
}
.brand-name { font-size: 1.02rem; font-weight: 700; color: #0e1f3c; line-height: 1.2; }
.brand-sub { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: #54657e; }

.sidebar-footer { margin-top: 26px; padding: 14px; border: 1px solid #e5eaf3; border-radius: 12px; background: #f8fafd; }
.sidebar-footer .sf-title { font-size: 0.8rem; font-weight: 700; color: #0e1f3c; }
.sidebar-footer .sf-sub { font-size: 0.75rem; color: #54657e; margin-top: 3px; line-height: 1.5; }

.callout { background: #f8fafd; border: 1px solid #e5eaf3; border-radius: 12px; padding: 14px 16px; font-size: 0.88rem; color: #43536b; line-height: 1.6; }
.callout b { color: #0e1f3c; }

.explain p { color: #43536b; line-height: 1.65; margin: 0 0 10px; font-size: 0.92rem; }
.explain p:last-child { margin-bottom: 0; }

.result-hero { text-align: center; padding: 8px 0 6px; }
.result-hero .big { font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em; }
.result-hero .label { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; color: #54657e; }

/* ---------- Cards ---------- */
[data-testid="stVerticalBlock"] { background: #ffffff !important; border: 1px solid #e5eaf3 !important; border-radius: 16px !important; }
.card-title { font-size: 1.05rem; font-weight: 700; color: #0e1f3c; margin: 0 0 4px; }
.card-subtitle { font-size: 0.85rem; color: #54657e; margin: 0 0 14px; line-height: 1.5; }

/* ---------- KPI metrics ---------- */
[data-testid="stMetric"] { background: #ffffff !important; border: 1px solid #e5eaf3 !important; border-radius: 16px !important; padding: 2px; }

@media (max-width: 1100px) { .page-header h1 { font-size: 1.5rem; } }
"""

st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def page_header(title: str, subtitle: str, badge: str = None) -> None:
    badge_html = f'<span class="pill">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="page-header">
            <div>
                <h1>{title}</h1>
                <p class="page-subtitle">{subtitle}</p>
            </div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, subtitle: str = None):
    c = st.container(border=True)
    c.markdown(f"<div class='card-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        c.markdown(f"<div class='card-subtitle'>{subtitle}</div>", unsafe_allow_html=True)
    return c


def fmt_pct(prob) -> str:
    return f"{float(prob) * 100:.2f}%"


def fraud_word(cls) -> str:
    return "Fraud" if int(cls) == fm.CLASS_FRAUD else "Legitimate"


def get_final_test_row():
    final_results = fm.load_final_test_results()
    if final_results is None or final_results.empty:
        return None
    return final_results.iloc[0]


V_FEATURE_GROUPS = [
    ["V1", "V2", "V3", "V4", "V5"],
    ["V6", "V7", "V8", "V9", "V10"],
    ["V11", "V12", "V13", "V14", "V15"],
    ["V16", "V17", "V18", "V19", "V20"],
    ["V21", "V22", "V23", "V24", "V25"],
    ["V26", "V27", "V28"],
]


# --------------------------------------------------------------------------- #
# Page: Overview
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    page_header(
        "Overview",
        "Portfolio-level view of transaction volume, fraud prevalence and the saved model's final held-out test performance.",
        badge="Dataset & model snapshot",
    )

    summary = fm.load_dataset_summary()
    total = int(summary.get("total_transactions") or 0)
    fraud_total = int(summary.get("fraudulent_transactions") or 0)
    legit_total = int(summary.get("legitimate_transactions") or 0)
    fraud_pct = float(summary.get("fraud_percentage") or 0.0)
    dup_removed = int(summary.get("duplicate_rows") or 0)

    clean = fm.load_clean_dataset()
    clean_total = int(len(clean))
    clean_fraud = int(clean["Class"].sum())
    clean_legit = int(clean_total - clean_fraud)

    final_row = get_final_test_row()
    final_model_name = "Random Forest"
    final_metrics = {"Precision": 0.0, "Recall": 0.0, "F1-score": 0.0, "Average Precision": 0.0}
    if final_row is not None:
        final_model_name = str(final_row.get("Model") or final_model_name)
        for key in final_metrics:
            final_metrics[key] = float(final_row.get(key) or 0.0)
    final_recall = final_metrics["Recall"]
    final_ap = final_metrics["Average Precision"]

    kpi_specs = [
        {
            "label": "Transactions analysed",
            "value": f"{total:,}",
            "delta": f"{clean_total:,} after de-duplication",
            "sub": "Source: original dataset (data/creditcard.csv)",
        },
        {
            "label": "Fraudulent transactions",
            "value": f"{fraud_total:,}",
            "delta": f"{fraud_pct:.2f}% fraud rate",
            "sub": "Original dataset labels (Class = 1)",
        },
        {
            "label": "Cleaned transactions",
            "value": f"{clean_total:,}",
            "delta": f"{clean_fraud} fraudulent",
            "sub": f"Cleaned dataset · {dup_removed:,} duplicates removed",
        },
        {
            "label": "Final test recall (Random Forest)",
            "value": f"{final_recall:.2f}",
            "delta": f"Average precision {final_ap:.2f}",
            "sub": "Final held-out test evaluation",
        },
    ]

    kpi_cols = st.columns(4)
    for col, spec in zip(kpi_cols, kpi_specs):
        with col:
            c = col.container(border=True)
            c.metric(spec["label"], spec["value"], delta=spec["delta"])
            c.caption(spec["sub"])

    chart_col, context_col = st.columns([3, 2], gap="large")

    with chart_col:
        c = card("Transaction distribution", "Number of transactions by actual class (cleaned dataset).")
        counts = pd.DataFrame(
            {
                "Class": ["Legitimate", "Fraudulent"],
                "Count": [int(clean_legit), int(clean_fraud)],
            }
        )
        fig = px.bar(
            counts,
            x="Class",
            y="Count",
            color="Class",
            color_discrete_map={"Legitimate": "#15803d", "Fraudulent": "#dc2626"},
            text="Count",
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="#ffffff",
            xaxis=dict(showgrid=False, color="#8b99b1"),
            yaxis=dict(showgrid=True, gridcolor="#e5eaf3", color="#54657e"),
            font=dict(color="#0e1f3c"),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig)

    with context_col:
        c = card("Imbalance context", "Why fraud prevalence matters for this model.")
        st.markdown(
            f"""
            <div class="explain">
            <p>The dataset is heavily imbalanced: only <b>{fraud_pct:.2f}%</b> of transactions are fraudulent
            ({fraud_total:,} fraud cases out of {total:,} transactions).</p>
            <p>With such low fraud prevalence, plain accuracy is misleading. The trained model uses a balanced
            strategy internally and is evaluated with precision, recall and average precision.</p>
            <p>On the final held-out test set, the saved <b>{final_model_name}</b> reached a recall of
            <b>{final_recall:.2f}</b> ({final_recall * 100:.0f}% of fraud cases correctly flagged) and an average
            precision of <b>{final_ap:.2f}</b>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c = card("Recent transactions", "Most recent rows of the cleaned dataset (by Transaction Time). Actual labels shown — not predictions.")
    recent = clean.sort_values("Time", ascending=False).head(10)
    st.dataframe(recent[["Time", "Amount", "Class"]].rename(columns={"Class": "Actual Class"}))
    fraud_rate_clean = clean_fraud / clean_total * 100 if clean_total else 0.0
    st.caption(f"Source: cleaned dataset (duplicates removed). Fraud rate after cleaning: {fraud_rate_clean:.2f}%.")


# --------------------------------------------------------------------------- #
# Page: Fraud Detection
# --------------------------------------------------------------------------- #
def render_batch_results(scored: pd.DataFrame) -> None:
    n_scored = len(scored)
    n_fraud = int(scored["Predicted Class"].sum())
    mean_prob = float(scored["Fraud Probability"].mean())

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Transactions scored", f"{n_scored:,}")
    with m2:
        st.metric("Predicted fraud", f"{n_fraud:,}", delta=f"{n_fraud / max(n_scored, 1):.2%} of batch")
    with m3:
        st.metric("Mean fraud probability", f"{mean_prob:.3f}")

    fig = px.histogram(scored, x="Fraud Probability", nbins=20, template="plotly_white")
    fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="#ffffff",
        xaxis=dict(title="Fraud probability", color="#54657e"),
        yaxis=dict(title="Transactions", color="#54657e"),
        font=dict(color="#0e1f3c"),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig)

    csv_bytes = scored.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions (CSV)",
        data=csv_bytes,
        file_name="sentinel_predictions.csv",
        mime="text/csv",
    )

    display_df = scored[["Time", "Amount", "Predicted Class", "Fraud Probability", "Risk Category"]].head(500)
    st.dataframe(display_df)


def page_detection() -> None:
    page_header(
        "Fraud Detection",
        "Score uploaded transaction batches or a single transaction with the saved fraud model.",
        badge="Model inference",
    )

    tab_batch, tab_single = st.tabs(["Batch Upload", "Single Transaction"])

    with tab_batch:
        st.markdown(
            "Upload a CSV containing the model's 30 expected features: `Time`, `V1`–`V28` and `Amount`. "
            "A `Class` column (the historical label) is ignored. Missing or unexpected columns are rejected with guidance."
        )
        uploaded = st.file_uploader("Upload transactions CSV", type=["csv"], key="batch_transactions")
        if uploaded is not None:
            try:
                uploaded_df = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Could not read the uploaded file as CSV: {exc}")
            else:
                features, errors = fm.validate_frame(uploaded_df)
                if features is None:
                    st.error("The uploaded file could not be used for prediction.")
                    for e in errors:
                        st.markdown(f"- {e}")
                else:
                    model = fm.load_model()
                    scored, error = fm.score_transactions(model, features)
                    if error is not None or scored is None:
                        st.error(error or "Prediction failed.")
                    else:
                        render_batch_results(scored)

    with tab_single:
        st.markdown(
            "<div class='callout'>Enter the actual transaction feature values for the single transaction you want to score. "
            "Fields left at their default placeholder values are not real transaction data.</div>",
            unsafe_allow_html=True,
        )
        with st.form("single_transaction_form") as form:
            col_time, col_amount = st.columns(2)
            with col_time:
                time_value = st.number_input("Time (seconds)", min_value=0.0, max_value=172800.0, value=0.0, step=1.0, key="single_time")
            with col_amount:
                amount_value = st.number_input("Amount", min_value=0.0, max_value=1000000.0, value=0.0, step=0.01, key="single_amount")
            values = {"Time": time_value, "Amount": amount_value}
            for group in V_FEATURE_GROUPS:
                group_cols = st.columns(len(group))
                for feat, col in zip(group, group_cols):
                    with col:
                        values[feat] = st.number_input(feat, min_value=-50.0, max_value=50.0, value=0.0, step=0.01, key=f"single_{feat}")
            submitted = st.form_submit_button("Run fraud prediction")

        if submitted:
            row = {feat: float(values[feat]) for feat in fm.EXPECTED_FEATURES}
            row_df = pd.DataFrame([row], columns=list(fm.EXPECTED_FEATURES))
            model = fm.load_model()
            scored, error = fm.score_transactions(model, row_df)
            if scored is None:
                st.error(error or "Prediction failed.")
            else:
                prob = float(scored.iloc[0]["Fraud Probability"])
                pred = int(scored.iloc[0]["Predicted Class"])
                risk = str(scored.iloc[0]["Risk Category"])
                st.markdown(
                    f"""
                    <div class="result-hero">
                        <div class="big" style="color: {'#dc2626' if risk == 'High' else '#0e1f3c'};">{fraud_word(pred)}</div>
                        <div class="label">Predicted class</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Fraud probability", f"{prob:.3f}", delta=fmt_pct(prob))
                with m2:
                    st.metric("Risk category", risk)
                with m3:
                    st.metric("Decision threshold", "0.50")
                st.markdown(
                    "<div class='callout'>The probability above is the model's estimated fraud score for this transaction. "
                    "It is not a guarantee that fraud occurred — it reflects the model's confidence learned from historical data.</div>",
                    unsafe_allow_html=True,
                )


# --------------------------------------------------------------------------- #
# Page: Model Performance
# --------------------------------------------------------------------------- #
def page_performance() -> None:
    page_header(
        "Model Performance",
        "Metrics for the saved fraud model, read from the project's saved results and reproduced from the held-out test split.",
        badge="Saved results",
    )

    final_row = get_final_test_row()
    final_metrics_df = pd.DataFrame(
        [
            {
                "Model": str(final_row.get("Model")) if final_row is not None else "Unknown",
                "Precision": float(final_row.get("Precision") or 0.0) if final_row is not None else 0.0,
                "Recall": float(final_row.get("Recall") or 0.0) if final_row is not None else 0.0,
                "F1-score": float(final_row.get("F1-score") or 0.0) if final_row is not None else 0.0,
                "Average Precision": float(final_row.get("Average Precision") or 0.0) if final_row is not None else 0.0,
            }
        ]
    )

    c = card("Final held-out test results", "Result reported in results/final_test_results.csv for the selected best model.")
    st.dataframe(final_metrics_df)

    c = card("Validation comparison", "Both models evaluated on the validation split (results/model_comparison.csv).")
    comparison = fm.load_model_comparison()
    if comparison is not None and not comparison.empty:
        display_cmp = comparison.copy()
        for key in ["Precision", "Recall", "F1-score", "Average Precision"]:
            if key in display_cmp.columns:
                display_cmp[key] = display_cmp[key].map(lambda v: f"{float(v):.4f}")
        st.dataframe(display_cmp)

        fig = px.bar(
            comparison,
            x="Model",
            y="Average Precision",
            color="Model",
            color_discrete_map={"Logistic Regression": "#2563eb", "Random Forest": "#0e1f3c"},
            text="Average Precision",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="#ffffff",
            yaxis=dict(range=[0, 1], color="#54657e"),
            xaxis=dict(color="#0e1f3c"),
            font=dict(color="#0e1f3c"),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig)

    cm_info = fm.compute_final_test_confusion()
    c = card(
        "Confusion matrix (reproduced)",
        f"Held-out test set ({cm_info['n_test']:,} rows) scored with the saved {cm_info['model']} using the same deterministic split procedure as the notebook.",
    )
    tn, fp, fn, tp = cm_info["tn"], cm_info["fp"], cm_info["fn"], cm_info["tp"]
    z = [[tn, fp], [fn, tp]]
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted Legitimate", "Predicted Fraud"],
            y=["Actual Legitimate", "Actual Fraud"],
            zmin=0,
            colorscale=[[0, "#f8fafc"], [0.55, "#bfdbfe"], [1, "#1e40af"]],
            text=z,
            texttemplate="%{text:d}",
            textfont={"size": 14},
            colorbar=dict(title="Count"),
        )
    )
    fig_cm.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_cm)

    left, right = st.columns(2, gap="large")
    with left:
        c = card("False positive", "A transaction predicted as fraud was actually legitimate.")
        st.markdown(
            f"""
            <div class="explain">
            <p>In plain language: the model raised a fraud alert for a transaction that was later confirmed legitimate.</p>
            <p><b>Cost:</b> genuine customers may be blocked or asked to verify; manual review effort increases.</p>
            <p><b>Final test count:</b> {fp} false positive(s).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        c = card("False negative", "A transaction that was actually fraud was predicted as legitimate.")
        st.markdown(
            f"""
            <div class="explain">
            <p>In plain language: the model missed a fraudulent transaction and let it through.</p>
            <p><b>Cost:</b> the financial loss from the fraud was not prevented.</p>
            <p><b>Final test count:</b> {fn} false negative(s).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    X = fm.load_clean_dataset().drop(columns=["Class", "Amount_log"], errors="ignore")
    y = fm.load_clean_dataset()["Class"].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    val_proba = fm.predict_probabilities(fm.load_model(), X_val)
    precision, recall, _ = precision_recall_curve(y_val.to_numpy(), val_proba)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines+markers", name="Random Forest"))
    fig_pr.update_layout(
        title="Precision-Recall curve (validation set)",
        xaxis_title="Recall",
        yaxis_title="Precision",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_pr)


# --------------------------------------------------------------------------- #
# Page: Transaction Explorer
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=None)
def get_historical_predictions() -> pd.DataFrame:
    model = fm.load_model()
    clean = fm.load_clean_dataset()
    features = clean.loc[:, list(fm.EXPECTED_FEATURES)].copy()
    scored, error = fm.score_transactions(model, features)
    if error is not None or scored is None:
        return pd.DataFrame()
    out = scored.loc[:, list(fm.EXPECTED_FEATURES) + ["Predicted Class", "Fraud Probability", "Risk Category"]].copy()
    out["Actual Class"] = clean["Class"].to_numpy()
    return out.reset_index(drop=True)


def page_explorer() -> None:
    page_header(
        "Transaction Explorer",
        "Browse historical transactions with model risk scores, and filter by predicted class, amount and risk score.",
        badge="Exploratory",
    )
    hist = get_historical_predictions()
    if hist is None or hist.empty:
        st.error("Unable to compute historical predictions right now.")
        return

    pred_filter = st.selectbox("Predicted class", ["All transactions", "Predicted fraud", "Predicted legitimate"], index=0)
    risk_filter = st.selectbox("Risk category", ["All risk levels", "Low", "Medium", "High"], index=0)
    amount_min = st.number_input("Minimum amount", min_value=0.0, max_value=1000000.0, value=0.0, step=0.01, key="explorer_amount_min")
    amount_max = st.number_input("Maximum amount", min_value=0.0, max_value=1000000.0, value=float(hist["Amount"].max()), step=0.01, key="explorer_amount_max")
    sort_by = st.selectbox("Sort by", ["Fraud probability", "Amount", "Time"], index=0)

    filtered = hist.copy()
    if pred_filter == "Predicted fraud":
        filtered = filtered[filtered["Predicted Class"] == fm.CLASS_FRAUD]
    elif pred_filter == "Predicted legitimate":
        filtered = filtered[filtered["Predicted Class"] == fm.CLASS_LEGITIMATE]
    if risk_filter != "All risk levels":
        filtered = filtered[filtered["Risk Category"] == risk_filter]
    filtered = filtered[filtered["Amount"] <= float(amount_max)]
    if float(amount_min) > 0:
        filtered = filtered[filtered["Amount"] >= float(amount_min)]

    ascending = True if sort_by in ("Amount", "Time") else False
    sort_col = {"Fraud probability": "Fraud Probability", "Amount": "Amount", "Time": "Time"}[sort_by]
    filtered = filtered.sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)

    display_cols = ["Time", "Amount", "Actual Class", "Predicted Class", "Fraud Probability", "Risk Category"]
    display = filtered[display_cols].copy()
    display["Predicted Class"] = display["Predicted Class"].map({0: "Legitimate", 1: "Fraud"})
    display["Actual Class"] = display["Actual Class"].map({0: "Legitimate", 1: "Fraud"})
    display["Fraud Probability"] = display["Fraud Probability"].map(lambda p: f"{float(p) * 100:.2f}%")

    st.markdown(f"**{len(filtered):,}** transactions match the current filters.")
    st.dataframe(display)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered transactions (CSV)", data=csv_bytes, file_name="sentinel_transactions.csv", mime="text/csv")


# --------------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------------- #
PAGES = ["Overview", "Fraud Detection", "Model Performance", "Transaction Explorer"]

if "page" not in st.session_state:
    st.session_state["page"] = "Overview"

page = st.session_state.get("page", "Overview")

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

for p in PAGES:
    if st.sidebar.button(p, key=f"nav_{p}", type="primary" if p == page else "secondary"):
        st.session_state["page"] = p

st.sidebar.markdown(
    """
    <div class="sidebar-footer">
        <div class="sf-title">Saved model</div>
        <div class="sf-sub">Random Forest · fraud_detection_model.joblib</div>
        <div class="sf-sub" style="margin-top: 8px;">Local dashboard · no data leaves this machine</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Overview":
    page_overview()
elif page == "Fraud Detection":
    page_detection()
elif page == "Model Performance":
    page_performance()
elif page == "Transaction Explorer":
    page_explorer()
else:
    st.session_state["page"] = "Overview"
    page_overview()
