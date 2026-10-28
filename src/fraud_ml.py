"""Core helpers for the Sentinel Fraud Analytics dashboard.

This module is the single place that knows how to:

* load the genuine saved model artifacts and project datasets,
* validate transaction frames before they touch the model,
* score transactions with a configurable decision threshold,
* draw reproducible sample transactions from the local dataset,
* inspect third-party datasets and report compatibility with the saved model,
* read and reproduce the project's saved evaluation results.

The saved pipeline (``models/fraud_detection_model.joblib``) is used exactly as
it was trained.  It currently contains only a fitted
``RandomForestClassifier`` (no scaler), so the dashboard never scales the data
a second time.
"""

from __future__ import annotations

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

# Dataset Lab upload settings.
# `.streamlit/config.toml` -> [server] maxUploadSize is the single source of
# truth for the Dataset Lab limit.  The fallback below is only used if the
# Streamlit configuration cannot be read (for example in bare test runs).
FALLBACK_DATASET_UPLOAD_MB = 400
DATASET_PREVIEW_ROWS = 100
SUPPORTED_UPLOAD_SUFFIX = ".csv"

# The saved model expects these 30 features, in this exact order.
EXPECTED_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Historical label column.  It may be present in uploaded files for
# exploration/evaluation, but it is NEVER passed to the model.
OPTIONAL_COLUMNS = {"Class"}

CLASS_LEGITIMATE = 0
CLASS_FRAUD = 1

DEFAULT_DECISION_THRESHOLD = 0.50

# Risk categories are based on the estimated fraud probability and are
# independent of the decision threshold.  They are documented in the app.
RISK_MEDIUM_THRESHOLD = 0.10
RISK_HIGH_THRESHOLD = 0.30

THRESHOLD_OPTIONS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)


# --------------------------------------------------------------------------- #
# Uploaded files
# --------------------------------------------------------------------------- #
class UploadError(Exception):
    """A user-facing problem with an uploaded file (type, size or contents)."""


def dataset_upload_limit_mb() -> int:
    """Return the Dataset Lab upload limit in MB.

    ``.streamlit/config.toml`` -> ``[server] maxUploadSize`` is the single
    source of truth.  The fallback is only used when Streamlit's configuration
    cannot be read (for example a bare script/test import).
    """
    try:
        configured = st.get_option("server.maxUploadSize")
        limit = int(configured)
    except Exception:
        # Bare script/test runs have no Streamlit runtime; use the documented
        # fallback rather than failing the whole page.
        limit = FALLBACK_DATASET_UPLOAD_MB
    if limit <= 0:
        limit = FALLBACK_DATASET_UPLOAD_MB
    return limit


def upload_signature(uploaded) -> str:
    """Identity for an uploaded file that does not read its contents.

    ``UploadedFile.file_id`` is stable for the lifetime of one upload, so the
    app can cache parsed frames and full-dataset statistics without re-reading
    the file on every widget interaction.
    """
    file_id = getattr(uploaded, "file_id", None)
    if file_id:
        return f"id:{file_id}"
    return f"{getattr(uploaded, 'name', '')}:{getattr(uploaded, 'size', 0)}"


def validate_uploaded_file(uploaded, *, max_mb: int | None = None) -> None:
    """Check the file name, extension and size before any parsing happens."""
    name = str(getattr(uploaded, "name", "") or "")
    suffix = Path(name).suffix.lower()
    if suffix != SUPPORTED_UPLOAD_SUFFIX:
        shown = suffix or "no extension"
        raise UploadError(
            f"Unsupported file type ({shown}). Only CSV files ({SUPPORTED_UPLOAD_SUFFIX}) are accepted."
        )

    limit = int(max_mb) if max_mb else dataset_upload_limit_mb()
    size = getattr(uploaded, "size", None)
    if size is None:
        return
    try:
        size = int(size)
    except (TypeError, ValueError):
        return
    if size <= 0:
        raise UploadError("The uploaded file is empty.")
    if size > limit * 1024 * 1024:
        actual_mb = size / (1024 * 1024)
        raise UploadError(
            f"The file is {actual_mb:,.1f} MB, which exceeds the {limit} MB upload limit."
        )


def read_csv_upload(uploaded, *, max_mb: int | None = None) -> pd.DataFrame:
    """Validate and parse an uploaded CSV exactly once.

    Raises :class:`UploadError` with a user-facing message for unsupported
    types, oversized/empty files, malformed CSV content and memory failures.
    """
    validate_uploaded_file(uploaded, max_mb=max_mb)

    try:
        uploaded.seek(0)
    except (AttributeError, OSError, ValueError):
        pass

    try:
        frame = pd.read_csv(uploaded)
    except MemoryError as exc:
        raise UploadError(
            "Not enough memory to read this file. Close other applications, "
            "or upload a smaller CSV file, then try again."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise UploadError("The uploaded CSV contains no data.") from exc
    except pd.errors.ParserError as exc:
        raise UploadError(
            f"The file could not be parsed as CSV: {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise UploadError(
            "The file is not valid UTF-8 text. Save the CSV with UTF-8 encoding and try again."
        ) from exc
    except (ValueError, TypeError, OSError) as exc:
        raise UploadError(f"The file could not be read as CSV: {exc}") from exc

    if frame.shape[1] == 0:
        raise UploadError("The uploaded CSV has no columns.")
    return frame


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def artifact_status() -> dict[str, bool]:
    """Return which optional/required project artifacts exist on disk."""
    return {
        "model": MODEL_PATH.exists(),
        "feature_names": FEATURE_NAMES_PATH.exists(),
        "dataset": DATASET_PATH.exists(),
        "dataset_summary": DATASET_SUMMARY_PATH.exists(),
        "model_comparison": MODEL_COMPARISON_PATH.exists(),
        "final_test_results": FINAL_TEST_RESULTS_PATH.exists(),
    }


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
def load_feature_names() -> list[str]:
    """Load the feature names the model was trained with (fallback: schema)."""
    if not FEATURE_NAMES_PATH.exists():
        return list(EXPECTED_FEATURES)
    try:
        names = [str(name) for name in joblib.load(FEATURE_NAMES_PATH)]
    except Exception:
        return list(EXPECTED_FEATURES)
    return names or list(EXPECTED_FEATURES)


def model_family(model) -> str:
    """Human readable name of the underlying classifier."""
    clf = model
    if hasattr(model, "steps") and len(model.steps) > 0:
        clf = model.steps[-1][1]
    return type(clf).__name__


MODEL_DISPLAY_NAMES = {
    "RandomForestClassifier": "Random Forest",
    "LogisticRegression": "Logistic Regression",
    "GradientBoostingClassifier": "Gradient Boosting",
    "HistGradientBoostingClassifier": "Histogram Gradient Boosting",
}


def model_display_name(model) -> str:
    family = model_family(model)
    return MODEL_DISPLAY_NAMES.get(family, family)


def pipeline_step_names(model) -> list[str]:
    if hasattr(model, "steps"):
        return [name for name, _ in model.steps]
    return [type(model).__name__]


def pipeline_has_scaler(model) -> bool:
    """Best-effort check whether the saved pipeline already scales features."""
    scaler_names = {"scaler", "standardscaler", "minmaxscaler", "robustscaler", "normalizer"}
    for name, step in getattr(model, "steps", []):
        if name.lower() in scaler_names or type(step).__name__.lower() in scaler_names:
            return True
    return False


# --------------------------------------------------------------------------- #
# Project data + saved results
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=None)
def load_dataset() -> pd.DataFrame:
    """Raw credit-card transactions (original file, with duplicates)."""
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
    if not MODEL_COMPARISON_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MODEL_COMPARISON_PATH)


@st.cache_data(ttl=None)
def load_final_test_results() -> pd.DataFrame:
    """Final test results (results/final_test_results.csv)."""
    if not FINAL_TEST_RESULTS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FINAL_TEST_RESULTS_PATH)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def inspect_frame(df: pd.DataFrame, *, build_features: bool = True) -> dict:
    """Inspect an input frame without calling the model.

    Returns a report describing structural and numeric problems.  When
    ``build_features`` is true (the default) and the frame is usable, the report
    also contains a ``features`` frame with the exact expected columns/order.
    Only the 30 model features are ever returned — the ``Class`` column (and any
    extra columns) are excluded from model input.

    Pass ``build_features=False`` for a validation-only pass on a large file:
    it avoids materialising a second full feature frame, while still checking
    every required column and value.
    """
    report: dict = {
        "rows": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "missing": [],
        "unexpected": [],
        "duplicate_columns": [],
        "non_numeric": {},
        "missing_values": {},
        "infinite_values": {},
        "class_present": False,
        "ok": False,
        "errors": [],
        "warnings": [],
        "features": None,
    }

    columns = [str(c) for c in df.columns]
    duplicate_columns = sorted({c for c in columns if columns.count(c) > 1})
    report["duplicate_columns"] = duplicate_columns
    if duplicate_columns:
        report["errors"].append(
            "Duplicate column name(s): " + ", ".join(duplicate_columns)
            + ". Rename or remove the duplicate columns, then try again."
        )
        return report

    if len(df) == 0:
        report["errors"].append(
            "The file contains no transaction rows. Add at least one row of numeric feature values."
        )
        return report

    missing = [c for c in EXPECTED_FEATURES if c not in columns]
    unexpected = [c for c in columns if c not in EXPECTED_FEATURES and c not in OPTIONAL_COLUMNS]
    report["missing"] = missing
    report["unexpected"] = unexpected
    report["class_present"] = "Class" in columns

    frame = df
    if list(df.columns) != columns:
        # Column labels need normalising to strings; a shallow copy is enough
        # because only the labels change, not the underlying data.
        frame = df.copy(deep=False)
        frame.columns = columns

    if missing:
        report["errors"].append(
            "Missing required column(s): " + ", ".join(missing)
            + ". The saved model needs all 30 features: "
            + ", ".join(EXPECTED_FEATURES) + "."
        )

    if unexpected:
        report["warnings"].append(
            "Extra column(s) will be ignored: " + ", ".join(unexpected)
            + ". They are not passed to the model."
        )
    if report["class_present"]:
        report["warnings"].append(
            "'Class' will be treated as the historical label and kept separate from predictions. "
            "It is never passed to the model."
        )

    numeric_cols: dict[str, pd.Series] = {}
    for col in EXPECTED_FEATURES:
        if col not in columns:
            continue
        raw = frame[col]
        converted = pd.to_numeric(raw, errors="coerce")

        raw_missing = raw.isna()
        non_numeric_mask = converted.isna() & ~raw_missing
        if non_numeric_mask.any():
            report["non_numeric"][col] = int(non_numeric_mask.sum())
        if raw_missing.any():
            report["missing_values"][col] = int(raw_missing.sum())
        infinite_mask = np.isinf(converted.to_numpy(dtype=float, na_value=np.nan))
        if infinite_mask.any():
            report["infinite_values"][col] = int(infinite_mask.sum())
        if build_features:
            numeric_cols[col] = converted

    if report["non_numeric"]:
        detail = ", ".join(f"{c} ({n} cell(s))" for c, n in report["non_numeric"].items())
        report["errors"].append(
            "Non-numeric values found in required feature(s): " + detail
            + ". Every model feature must contain numbers only."
        )
    if report["missing_values"]:
        detail = ", ".join(f"{c} ({n} blank)" for c, n in report["missing_values"].items())
        report["errors"].append(
            "Missing values found in required feature(s): " + detail
            + ". Fill or remove these rows before scoring."
        )
    if report["infinite_values"]:
        detail = ", ".join(f"{c} ({n})" for c, n in report["infinite_values"].items())
        report["errors"].append(
            "Infinite values (inf / -inf) found in: " + detail + ". Replace them with finite numbers."
        )

    if report["errors"]:
        return report

    if build_features:
        features = pd.DataFrame({c: numeric_cols[c] for c in EXPECTED_FEATURES})
        report["features"] = features.astype(float)
    report["ok"] = True
    return report


def validate_frame(df: pd.DataFrame):
    """Backwards-compatible wrapper: return ``(features, errors)``."""
    report = inspect_frame(df)
    return report["features"], report["errors"]


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_probabilities(model, frame: pd.DataFrame) -> np.ndarray:
    """Return fraud probability (class-1 score) for each row of ``frame``."""
    proba = np.asarray(model.predict_proba(frame))[:, 1]
    return np.asarray(proba, dtype=float)


def risk_category(prob: float) -> str:
    """Map a fraud probability to a documented, threshold-independent risk band."""
    p = float(prob)
    if p >= RISK_HIGH_THRESHOLD:
        return "High"
    if p >= RISK_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def decision_label(prob: float, threshold: float = DEFAULT_DECISION_THRESHOLD) -> str:
    """Human-friendly prediction label for a probability and threshold."""
    return "Potential fraud" if float(prob) >= float(threshold) else "Likely legitimate"


def score_transactions(model, frame: pd.DataFrame, threshold: float = DEFAULT_DECISION_THRESHOLD):
    """Score a validated frame; returns ``(frame_with_predictions, error)``."""
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        return None, "The decision threshold must be between 0 and 1."

    try:
        # One float projection, reused for both prediction and output. Adding
        # columns to a projected frame does not mutate the caller's frame.
        out = frame.loc[:, EXPECTED_FEATURES].astype(float)
        probas = predict_probabilities(model, out)
    except MemoryError:
        return None, "Not enough memory to score these transactions. Try a smaller file."
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Model prediction failed: {exc}"

    out["Predicted Class"] = (probas >= threshold).astype(int)
    out["Fraud Probability"] = probas
    out["Risk Category"] = [risk_category(p) for p in probas]
    return out, None


def score_transactions_chunked(
    model,
    frame: pd.DataFrame,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
    chunk_size: int = 25_000,
    progress_callback=None,
):
    """Score large frames in chunks so the UI can show progress.

    Returns ``(probabilities, error)`` as a numpy array aligned with ``frame``.
    """
    n = len(frame)
    probas = np.empty(n, dtype=float)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        try:
            # Convert only the current chunk, so peak memory stays close to the
            # size of the uploaded frame plus one chunk instead of two full
            # float copies of the dataset.
            chunk = frame.iloc[start:stop].loc[:, EXPECTED_FEATURES].astype(float)
            probas[start:stop] = predict_probabilities(model, chunk)
        except MemoryError:
            return None, "Not enough memory to score this file. Try a smaller file or close other applications."
        except Exception as exc:  # pragma: no cover - defensive
            return None, f"Model prediction failed: {exc}"
        if progress_callback is not None:
            progress_callback(stop, n)
    return probas, None


# --------------------------------------------------------------------------- #
# Sample transactions
# --------------------------------------------------------------------------- #
SAMPLE_CLASS_OPTIONS = {
    "Any transaction": None,
    "Legitimate example": CLASS_LEGITIMATE,
    "Fraud example": CLASS_FRAUD,
}


def draw_sample(seed: int, class_filter: str = "Any transaction"):
    """Draw a reproducible transaction from the local dataset.

    Returns ``(row, row_index)`` where ``row`` is a pandas Series containing
    the original columns (including the historical ``Class`` label).  The
    label is shown to the user but never passed to the model.
    """
    if class_filter not in SAMPLE_CLASS_OPTIONS:
        raise ValueError(f"Unknown sample filter: {class_filter}")

    dataset = load_dataset()
    class_value = SAMPLE_CLASS_OPTIONS[class_filter]
    if class_value is None:
        pool = np.arange(len(dataset))
    else:
        pool = np.flatnonzero(dataset["Class"].to_numpy() == class_value)

    if pool.size == 0:
        raise ValueError(f"No '{class_filter}' rows are available in the local dataset.")

    rng = np.random.default_rng(int(seed))
    index = int(rng.choice(pool))
    return dataset.iloc[index], index


def format_time_offset(seconds: float) -> str:
    """Format the dataset's Time column as days + clock time."""
    total = int(round(float(seconds)))
    days, remainder = divmod(total, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    return f"Day {days} · {hours:02d}:{minutes:02d}:{secs:02d}"


def build_template_csv() -> bytes:
    """Build a CSV template with the 30 required columns and one real example row.

    The example row is copied from the local credit-card dataset so it has
    genuine numeric values and the correct structure.  It is a structural
    template only — not a claim about a new transaction.
    """
    dataset = load_dataset()
    example = dataset.iloc[[0]].loc[:, EXPECTED_FEATURES]
    return example.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------- #
# Dataset Lab: profiling + compatibility
# --------------------------------------------------------------------------- #
def detect_target_column(df: pd.DataFrame) -> str | None:
    for candidate in ("Class", "isFraud", "fraud", "label", "target"):
        if candidate in df.columns:
            return candidate
    return None


def dataset_profile(df: pd.DataFrame, *, include_uniques: bool = True) -> dict:
    """Plain-language dataset statistics used by the Dataset Lab.

    Every statistic is computed over the whole frame.  The result is intended
    to be cached by the caller per upload, because the passes over the file are
    expensive for large datasets.
    """
    missing = df.isna().sum()
    missing_counts = {str(col): int(count) for col, count in missing.items()}
    missing_by_column = {
        col: count
        for col, count in sorted(missing_counts.items(), key=lambda item: item[1], reverse=True)
        if count > 0
    }
    missing_by_column = dict(list(missing_by_column.items())[:12])

    # Checking the dtype directly avoids the full DataFrame copy that
    # ``select_dtypes`` would create for a large upload.
    numeric_columns = int(
        sum(1 for dtype in df.dtypes if pd.api.types.is_numeric_dtype(dtype))
    )

    target = detect_target_column(df)
    class_counts = None
    class_percentages = None
    class_count_total = 0
    class_counts_truncated = False
    if target is not None:
        all_counts = df[target].value_counts(dropna=False)
        class_count_total = int(len(all_counts))
        counts = all_counts.head(10)
        class_counts = {str(k): int(v) for k, v in counts.items()}
        total = max(int(df[target].notna().sum()), 1)
        class_percentages = {str(k): float(v) / total * 100 for k, v in class_counts.items()}
        class_counts_truncated = class_count_total > len(class_counts)

    unique_by_column = (
        {str(col): int(df[col].nunique(dropna=True)) for col in df.columns}
        if include_uniques
        else {}
    )

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_columns,
        "missing_total": int(missing.sum()),
        "missing_counts": missing_counts,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_by_column": missing_by_column,
        "unique_by_column": unique_by_column,
        "target": target,
        "class_count_total": class_count_total,
        "class_counts_truncated": class_counts_truncated,
        "class_counts": class_counts,
        "class_percentages": class_percentages,
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
    }


def _status_row(name: str, status: str, detail: str) -> dict:
    return {"check": name, "status": status, "detail": detail}


def check_compatibility(df: pd.DataFrame) -> dict:
    """Check whether a dataset can be scored by the saved model.

    This function never calls the model.  It only compares the dataset schema
    and values with the 30 features the saved model expects.
    """
    checks: list[dict] = []
    columns = [str(c) for c in df.columns]
    missing = [c for c in EXPECTED_FEATURES if c not in columns]
    unexpected = [c for c in columns if c not in EXPECTED_FEATURES and c not in OPTIONAL_COLUMNS]
    has_class = "Class" in columns

    # 1. Rows
    if len(df) > 0:
        checks.append(_status_row("Contains transaction rows", "pass", f"{len(df):,} row(s) found."))
    else:
        checks.append(_status_row("Contains transaction rows", "fail", "The dataset has no rows."))

    # 2. Required feature columns
    if missing:
        checks.append(
            _status_row(
                "All 30 model features present",
                "fail",
                "Missing: " + ", ".join(missing) + ". The saved model needs Time, V1–V28 and Amount.",
            )
        )
    else:
        checks.append(
            _status_row(
                "All 30 model features present",
                "pass",
                "Time, V1–V28 and Amount were all found.",
            )
        )

    # 3. Time specifically (never substitute another column)
    if "Time" in columns:
        checks.append(
            _status_row(
                "Time feature available",
                "pass",
                "A 'Time' column exists. It will be used exactly as supplied.",
            )
        )
    else:
        detail = (
            "No 'Time' column was found. The saved model was trained with Time "
            "and there is no safe substitute. An 'id' or index column is NOT a "
            "replacement for Time, and no Time values will be invented."
        )
        if "id" in columns:
            detail += " This dataset looks like the Kaggle 2023 credit-card dataset (V1–V28, Amount, Class, id)."
        checks.append(_status_row("Time feature available", "fail", detail))

    # 4. Numeric feature values
    non_numeric: dict[str, int] = {}
    missing_values: dict[str, int] = {}
    infinite_values: dict[str, int] = {}
    for col in EXPECTED_FEATURES:
        if col not in columns:
            continue
        raw = df[col]
        converted = pd.to_numeric(raw, errors="coerce")
        raw_missing = raw.isna()
        bad = converted.isna() & ~raw_missing
        if bad.any():
            non_numeric[col] = int(bad.sum())
        if raw_missing.any():
            missing_values[col] = int(raw_missing.sum())
        inf_mask = np.isinf(converted.to_numpy(dtype=float, na_value=np.nan))
        if inf_mask.any():
            infinite_values[col] = int(inf_mask.sum())

    if non_numeric:
        detail = ", ".join(f"{c} ({n})" for c, n in non_numeric.items())
        checks.append(_status_row("Required features are numeric", "fail", "Non-numeric values in: " + detail + "."))
    elif missing:
        checks.append(
            _status_row(
                "Required features are numeric",
                "warn",
                "Not fully checked because required columns are missing.",
            )
        )
    else:
        checks.append(_status_row("Required features are numeric", "pass", "All model features are numeric."))

    # 5. Missing values
    if missing_values:
        detail = ", ".join(f"{c} ({n})" for c, n in missing_values.items())
        checks.append(_status_row("No missing model features", "fail", "Missing values in: " + detail + "."))
    elif missing:
        checks.append(_status_row("No missing model features", "warn", "Not fully checked because required columns are missing."))
    else:
        checks.append(_status_row("No missing model features", "pass", "No blanks in the 30 model features."))

    # 6. Infinite values
    if infinite_values:
        detail = ", ".join(f"{c} ({n})" for c, n in infinite_values.items())
        checks.append(_status_row("No infinite model values", "fail", "Infinite values in: " + detail + "."))
    elif missing:
        checks.append(_status_row("No infinite model values", "warn", "Not fully checked because required columns are missing."))
    else:
        checks.append(_status_row("No infinite model values", "pass", "No infinite values in the model features."))

    # 7. Label column
    if has_class:
        checks.append(
            _status_row(
                "Historical label column",
                "pass",
                "'Class' is present. It will be kept separate from predictions and never passed to the model.",
            )
        )
    else:
        checks.append(
            _status_row(
                "Historical label column",
                "warn",
                "No 'Class' column. Predictions are still possible, but accuracy/evaluation cannot be measured here.",
            )
        )

    # 8. Extra columns
    if unexpected:
        checks.append(
            _status_row(
                "Extra columns",
                "warn",
                "Ignored for model input: " + ", ".join(unexpected) + ". They cannot replace a missing model feature.",
            )
        )
    else:
        checks.append(_status_row("Extra columns", "pass", "No unexpected columns."))

    # 9. Column order / alignment
    if not missing:
        checks.append(
            _status_row(
                "Column order can be aligned",
                "pass",
                "Input columns will be reordered to the saved model order before scoring.",
            )
        )
    else:
        checks.append(
            _status_row(
                "Column order can be aligned",
                "fail",
                "Cannot align columns because required features are missing.",
            )
        )

    # 10. Known dataset layouts
    lower = {c.lower() for c in columns}
    if {"step", "type", "nameorig", "oldbalanceorg"}.issubset(lower):
        checks.append(
            _status_row(
                "Known dataset layout",
                "fail",
                "This looks like the PaySim mobile-money dataset. Its schema (step, type, account balances, …) "
                "is different from the credit-card model and cannot be scored by it.",
            )
        )
    elif "id" in lower and "time" not in lower:
        checks.append(
            _status_row(
                "Known dataset layout",
                "fail",
                "This matches the Kaggle 2023 credit-card dataset layout (id + V1–V28 + Amount, no Time). "
                "It is not compatible with the saved model as-is.",
            )
        )
    else:
        checks.append(_status_row("Known dataset layout", "pass", "No known incompatible schema detected."))

    failed = [c for c in checks if c["status"] == "fail"]
    compatible = len(failed) == 0
    return {
        "compatible": compatible,
        "checks": checks,
        "missing": missing,
        "unexpected": unexpected,
        "has_class": has_class,
        "rows": int(len(df)),
        "columns": columns,
    }


# --------------------------------------------------------------------------- #
# Non-destructive training workflow for another dataset
# --------------------------------------------------------------------------- #
def prepare_training_script(target_column: str) -> str:
    """Return a standalone training script for a *different* dataset schema.

    The script trains and evaluates a new Random Forest, writing to
    ``models/experiments/`` and ``results/experiments/`` so the existing saved
    model and results can never be overwritten.  It is provided for the user to
    run manually; the dashboard does not run it.
    """
    target = str(target_column).strip() or "Class"
    return f'''"""Separate training workflow generated by Sentinel Fraud Analytics.

This script is intentionally independent from the existing credit-card model.
It trains and evaluates a NEW Random Forest on a dataset with its own schema,
and writes only to models/experiments/ and results/experiments/.  It never
overwrites models/fraud_detection_model.joblib or the saved credit-card results.

Usage (Windows PowerShell):
    .venv\\Scripts\\python.exe train_new_dataset.py

Edit DATA_PATH below so it points at your CSV file before running.
"""

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_PATH = Path("data/REPLACE_WITH_YOUR_FILE.csv")
TARGET = {target!r}
RANDOM_STATE = 42

MODEL_DIR = Path("models/experiments")
RESULT_DIR = Path("results/experiments")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise SystemExit(f"Dataset not found: {{DATA_PATH}}. Update DATA_PATH first.")

df = pd.read_csv(DATA_PATH)
if TARGET not in df.columns:
    raise SystemExit(f"Target column {{TARGET!r}} not found. Available: {{list(df.columns)}}")

# Use the dataset's own numeric schema (drop the target and non-numeric columns).
X = df.drop(columns=[TARGET]).select_dtypes(include=[np.number])
y = df[TARGET]

if len(X.columns) == 0:
    raise SystemExit("No numeric feature columns remain after dropping the target.")
if y.nunique() < 2:
    raise SystemExit(f"Target {{TARGET!r}} needs at least two classes to train a classifier.")

print(f"Training on {{len(df):,}} rows and {{X.shape[1]}} numeric features.")
print(f"Target: {{TARGET}} | classes: {{sorted(y.dropna().unique().tolist())}}")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X_train, y_train)

def evaluate(name, X_part, y_part):
    proba = model.predict_proba(X_part)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {{
        "split": name,
        "Precision": precision_score(y_part, pred, zero_division=0),
        "Recall": recall_score(y_part, pred, zero_division=0),
        "F1-score": f1_score(y_part, pred, zero_division=0),
        "Average Precision": average_precision_score(y_part, proba),
    }}
    print(name, metrics)
    print("confusion matrix [[tn, fp], [fn, tp]]:", confusion_matrix(y_part, pred).tolist())
    return metrics

metrics = pd.DataFrame([evaluate("validation", X_val, y_val), evaluate("test", X_test, y_test)])

# File names are derived from the target column; nothing existing is replaced.
slug = re.sub(r"[^a-z0-9_]+", "_", TARGET.lower()).strip("_") or "dataset"
model_path = MODEL_DIR / f"{{slug}}_random_forest.joblib"
features_path = MODEL_DIR / f"{{slug}}_features.joblib"
metrics_path = RESULT_DIR / f"{{slug}}_metrics.csv"

joblib.dump(model, model_path)
joblib.dump(list(X.columns), features_path)
metrics.to_csv(metrics_path, index=False)

print(f"Saved new model: {{model_path}}")
print(f"Saved feature names: {{features_path}}")
print(f"Saved metrics: {{metrics_path}}")
print("Existing credit-card artifacts were not modified.")
'''


# --------------------------------------------------------------------------- #
# Saved evaluation results (genuine values only)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=None)
def compute_final_test_confusion() -> dict:
    """Reproduce the final held-out test confusion matrix with the saved model.

    Uses the same deterministic split procedure as the notebook:
    clean -> 70/30 split -> 50/50 split of the remainder (random_state=42,
    stratified).  Only genuine labels from the local dataset are used.
    """
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    from sklearn.model_selection import train_test_split

    df_clean = load_clean_dataset()
    X = df_clean.loc[:, EXPECTED_FEATURES]
    y = df_clean["Class"].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    model = load_model()
    proba = predict_probabilities(model, X_test)
    pred = (proba >= DEFAULT_DECISION_THRESHOLD).astype(int)

    cm = sk_confusion_matrix(y_test.to_numpy(), pred, labels=[CLASS_LEGITIMATE, CLASS_FRAUD])
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "model": model_family(model),
        "n_test": int(len(y_test)),
        "threshold": DEFAULT_DECISION_THRESHOLD,
    }


@st.cache_data(ttl=None)
def validation_probabilities() -> dict:
    """Validation-split labels and saved-model probabilities (genuine data)."""
    from sklearn.model_selection import train_test_split

    df_clean = load_clean_dataset()
    X = df_clean.loc[:, EXPECTED_FEATURES]
    y = df_clean["Class"].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    model = load_model()
    proba = predict_probabilities(model, X_val)
    return {
        "y": np.asarray(y_val, dtype=int),
        "proba": np.asarray(proba, dtype=float),
        "n": int(len(y_val)),
    }


@st.cache_data(ttl=None)
def threshold_tradeoff(thresholds: tuple = THRESHOLD_OPTIONS) -> pd.DataFrame:
    """Precision/recall/F1 on the validation split for several cut-offs."""
    data = validation_probabilities()
    y = data["y"]
    proba = data["proba"]

    rows = []
    for threshold in thresholds:
        pred = (proba >= float(threshold)).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "Threshold": float(threshold),
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
                "Predicted fraud": tp + fp,
                "Missed fraud": fn,
            }
        )
    return pd.DataFrame(rows)
