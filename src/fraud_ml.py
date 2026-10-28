"""Core fraud-detection helpers for the Sentinel Fraud Analytics dashboard.

Loads the trained model artifacts, validates transaction frames and runs
fraud predictions. All heavy operations are cached by Streamlit so the app
stays responsive during local use.

The saved pipeline (``models/fraud_detection_model.joblib``) is used directly;
no second scaling step is applied because the saved pipeline already contains
the fitted preprocessing it needs.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.joblib"
FEATURE_NAMES_PATH = BASE_DIR / "models" / "feature_names.joblib"
MODEL_COMPARISON_PATH = BASE_DIR / "results" / "model_comparison.csv"
FINAL_TEST_RESULTS_PATH = BASE_DIR / "results" / "final_test_results.csv"
DATASET_PATH = BASE_DIR / "data" / "creditcard.csv"
DATASET_SUMMARY_PATH = BASE_DIR / "dataset_summary.csv"

# The model was trained on these 30 features, in this order:
# Time, V1..V28, Amount
EXPECTED_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

OPTIONAL_COLUMNS = {"Class"}  # historical label column, ignored at prediction time

CLASS_LEGITIMATE = 0
CLASS_FRAUD = 1

FRAUD_PROBABILITY_MEDIUM_THRESHOLD = 0.10
FRAUD_PROBABILITY_HIGH_THRESHOLD = 0.30


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
@st.cache_resource(ttl=None)
def load_model():
    """Load the trained fraud detection pipeline saved with joblib."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run the notebook (fraud_detection.ipynb) end-to-end to generate it."
        )
    return joblib.load(MODEL_PATH)


@st.cache_resource(ttl=None)
def load_feature_names() -> list:
    """Load the feature names the model was trained with."""
    if not FEATURE_NAMES_PATH.exists():
        return list(EXPECTED_FEATURES)
    return [str(name) for name in joblib.load(FEATURE_NAMES_PATH)]


def model_family(model) -> str:
    """Human readable name of the underlying classifier."""
    clf = model
    if hasattr(model, "steps") and len(model.steps) > 0:
        clf = model.steps[-1][1]
    return type(clf).__name__


@st.cache_data(ttl=None)
def load_dataset() -> pd.DataFrame:
    """Raw credit card transactions (original, with duplicates)."""
    return pd.read_csv(DATASET_PATH)


@st.cache_data(ttl=None)
def load_clean_dataset() -> pd.DataFrame:
    """Dataset after the notebook's de-duplication step."""
    return pd.read_csv(DATASET_PATH).drop_duplicates().copy()


@st.cache_data(ttl=None)
def load_dataset_summary() -> dict:
    """Key scalar statistics from dataset_summary.csv."""
    if not DATASET_SUMMARY_PATH.exists():
        return {}
    summary = pd.read_csv(DATASET_SUMMARY_PATH)
    if summary.empty:
        return {}
    row = summary.iloc[0].to_dict()
    return {
        "total_transactions": int(row.get("total_transactions") or 0),
        "legitimate_transactions": int(row.get("legitimate_transactions") or 0),
        "fraudulent_transactions": int(row.get("fraudulent_transactions") or 0),
        "fraud_percentage": float(row.get("fraud_percentage") or 0.0),
        "missing_values": int(row.get("missing_values") or 0),
        "duplicate_rows": int(row.get("duplicate_rows") or 0),
    }


@st.cache_data(ttl=None)
def load_model_comparison() -> pd.DataFrame:
    """Model comparison on the validation set (results/model_comparison.csv)."""
    return pd.read_csv(MODEL_COMPARISON_PATH)


@st.cache_data(ttl=None)
def load_final_test_results() -> pd.DataFrame:
    """Final test results (results/final_test_results.csv)."""
    return pd.read_csv(FINAL_TEST_RESULTS_PATH)


# --------------------------------------------------------------------------- #
# Validation + prediction
# --------------------------------------------------------------------------- #
def validate_frame(df: pd.DataFrame):
    """Validate an uploaded transaction frame against the model expectations.

    Returns ``(frame, errors)`` where ``frame`` is a numeric DataFrame with the
    exact expected columns (and order) or ``None`` when validation failed.
    """
    errors: list[str] = []

    columns = [str(c) for c in df.columns]
    missing = [c for c in EXPECTED_FEATURES if c not in columns]
    unexpected = [c for c in columns if c not in EXPECTED_FEATURES and c != "Class"]

    if missing:
        errors.append(
            "Missing required column(s): " + ", ".join(missing) + ". "
            "Expected columns: " + ", ".join(EXPECTED_FEATURES) + "."
        )
    if unexpected:
        errors.append(
            "Unexpected column(s) found: " + ", ".join(unexpected) + ". "
            "The model only accepts the features "
            + ", ".join(EXPECTED_FEATURES)
            + " (a 'Class' column, if present, is ignored)."
        )

    frame = df.copy()
    if "Class" in frame.columns:
        frame = frame.drop(columns=["Class"])

    if missing or unexpected:
        return None, errors

    features = frame.loc[:, EXPECTED_FEATURES].copy()
    for col in EXPECTED_FEATURES:
        features[col] = pd.to_numeric(frame[col], errors="coerce")
        bad = features[col].isna().to_numpy()
        if bad.any():
            errors.append(
                f"Column '{col}' contains missing or non-numeric values. "
                "Every cell must be a valid number."
            )

    if features.empty:
        errors.append("The uploaded file contains no transaction rows to score.")
        return None, errors

    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        errors.append(
            "Invalid input: non-finite values (NaN or infinity) were detected. "
            "Check the numeric entries and re-upload the file."
        )

    if errors:
        return None, errors

    return features.astype(float), errors


def predict_probabilities(model, frame: pd.DataFrame) -> np.ndarray:
    """Return fraud probability (class-1 score) for each row of ``frame``."""
    proba = np.asarray(model.predict_proba(frame))[:, 1]
    return np.asarray(proba, dtype=float)


def predict_class_from_probability(prob: float) -> int:
    return int(prob >= 0.5)


def risk_category(prob: float) -> str:
    """Map a fraud probability to a simple risk category."""
    p = float(prob)
    if p >= FRAUD_PROBABILITY_HIGH_THRESHOLD:
        return "High"
    if p >= FRAUD_PROBABILITY_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def compute_final_test_confusion() -> dict:
    """Reproduce the final held-out test confusion matrix with the saved model.

    Uses the same deterministic split procedure as the notebook:
    clean -> 70/30 split -> 50/50 split of the remainder (random_state=42, stratified).
    """
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    from sklearn.model_selection import train_test_split

    df_clean = load_clean_dataset()
    X = df_clean.drop(columns=["Class", "Amount_log"], errors="ignore")
    y = df_clean["Class"].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    model = load_model()
    proba = predict_probabilities(model, X_test)
    pred = np.array([int(p >= 0.5) for p in proba])

    cm = sk_confusion_matrix(y_test.to_numpy(), pred, labels=[CLASS_LEGITIMATE, CLASS_FRAUD])
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "model": model_family(model), "n_test": int(len(y_test))}


def score_transactions(model, frame: pd.DataFrame):
    """Score a validated frame; returns (frame_with_predictions, error)."""
    try:
        probas = predict_probabilities(model, frame)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Model prediction failed: {exc}"
    preds = [int(p >= 0.5) for p in probas]
    out = frame.copy()
    out["Predicted Class"] = preds
    out["Fraud Probability"] = probas
    out["Risk Category"] = [risk_category(p) for p in probas]
    return out, None
