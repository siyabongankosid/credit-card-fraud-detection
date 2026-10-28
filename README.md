# Sentinel Fraud Analytics

A local, interactive fraud analytics dashboard for the **Credit Card Fraud Detection Using Machine Learning**
project. Built with **Streamlit** + **Plotly**, backed by the trained model and real dataset artifacts.

## What it does

1. **Overview** — transaction volume, fraud prevalence, model final-test performance, imbalance context and recent transactions.
2. **Fraud Detection** — batch CSV upload scoring + single-transaction prediction through the saved pipeline.
3. **Model Performance** — saved validation/final test metrics, reproduced confusion matrix, precision-recall curve, false positive / false negative explanation.
4. **Transaction Explorer** — browse historical transactions with model risk scores and filters.

## Project structure

```
credit-card-fraud-detection/
├── app.py                     # Streamlit dashboard entry point
├── requirements.txt           # Python dependencies
├── README.md
├── data/
│   └── creditcard.csv        # original transaction dataset
├── models/
│   ├── fraud_detection_model.joblib   # saved trained pipeline
│   └── feature_names.joblib            # saved feature names
├── results/
│   ├── model_comparison.csv           # validation metrics (Logistic Regression vs Random Forest)
│   └── final_test_results.csv        # final held-out test metrics (selected best model)
├── src/
│   ├── __init__.py
│   └── fraud_ml.py            # model loading, validation, prediction helpers
├── fraud_detection.ipynb     # original training notebook (preserved)
└── dataset_summary.csv       # dataset-level statistics
```

## Run locally

```powershell
cd "C:\Users\siyab\OneDrive\Desktop\School work\ISI S4\Machine Learning\project\credit-card-fraud-detection"
.venv\Scripts\Activate.ps1          # activate the project virtual environment
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open <http://localhost:8501>.

## Notes

- The saved model is loaded with `joblib`; no retraining is performed by the dashboard.
- The saved pipeline is used as-is (it already contains any fitted preprocessing).
- Predictions use the expected 30 features in order: `Time`, `V1`–`V28`, `Amount`. A `Class` column in uploaded files is ignored.
- Probabilities are model-estimated fraud scores, not guarantees.
