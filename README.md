# Sentinel Fraud Analytics

A local, plain-language fraud analytics dashboard for the **Credit Card Fraud Detection Using Machine Learning** project. Built with **Python, Streamlit, Plotly, pandas, and scikit-learn**, using a saved machine learning model and genuine evaluation results.

The dashboard runs entirely on your computer. It does not require cloud services, an external API server, a database, or deployment configuration.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset Download](#dataset-download)
- [Installation](#installation)
- [Running the Dashboard](#running-the-dashboard)
- [Dashboard Pages](#dashboard-pages)
- [Understanding the Model Features](#understanding-the-model-features)
- [Sample Transactions](#sample-transactions)
- [Decision Threshold and Risk Levels](#decision-threshold-and-risk-levels)
- [Using Other Datasets](#using-other-datasets)
- [Dataset Lab Upload Limits](#dataset-lab-upload-limits)
- [Model Evaluation Results](#model-evaluation-results)
- [Limitations](#limitations)
- [Technologies Used](#technologies-used)

## Features

- Interactive dashboard with charts and key dataset statistics.
- Fraud detection using the saved machine learning model.
- Sample transaction predictions using real dataset records.
- Advanced manual transaction entry.
- Batch predictions from compatible CSV files.
- Model performance analysis using saved evaluation results.
- Historical transaction exploration with risk scores.
- Dataset Lab for exploring other datasets and checking compatibility.
- Downloadable prediction results where supported by the application.
- Local processing without sending transaction data to a cloud service.

## Project Structure

The project uses the following structure:

```text
credit-card-fraud-detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── creditcard.csv
├── models/
│   ├── fraud_detection_model.joblib
│   └── feature_names.joblib
├── results/
│   ├── model_comparison.csv
│   └── final_test_results.csv
├── src/
│   ├── __init__.py
│   └── fraud_ml.py
├── fraud_detection.ipynb
└── dataset_summary.csv
```

**Important:** The `data/creditcard.csv` dataset is not included in this GitHub repository. You must download it separately by following the instructions below.

The model files and evaluation results are also required for the corresponding dashboard features. Make sure any required project artifacts are included in the repository or generated through the documented training workflow.

## Dataset Download

### Original Credit Card Fraud Detection Dataset

This project uses the Credit Card Fraud Detection dataset provided on Kaggle.

**Dataset link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains European cardholder transactions made over two days in September 2013. It includes anonymized transaction features and a `Class` column identifying whether a transaction was fraudulent.

The original dataset contains 284,807 transactions before duplicate removal.

### How to download the dataset

1. Open the [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Sign in to your Kaggle account.
3. Click the **Download** button.
4. Extract the downloaded ZIP file.
5. Locate the file named `creditcard.csv`.
6. Copy that file into the project's `data` folder.

After downloading, confirm that the file is located here:

```text
credit-card-fraud-detection/data/creditcard.csv
```

Make sure the filename is exactly `creditcard.csv`. Do not leave the dataset inside the ZIP archive.

### Why is the dataset not on GitHub?

The dataset is excluded from the repository to avoid committing a large data file. Each person who clones or downloads the project must obtain the dataset separately from Kaggle.

The repository can include a `data/README.md` file with these instructions because Git does not track empty folders.

If the dataset is missing, features that depend on loading the original data—such as sample transactions and historical transaction exploration—will not work.

## Installation

### Requirements

You will need:

- Windows, macOS, or Linux.
- Python installed on your computer.
- Git, if cloning the repository.
- The project files and required model artifacts.
- The original dataset downloaded from Kaggle.

The commands below use Windows PowerShell and assume that a `.venv` virtual environment already exists in the project directory.

### 1. Open the project directory

Open PowerShell and navigate to the project folder:

```powershell
cd "your folder directory"
```

Replace `your folder directory` with the actual path to your project.

### 2. Create a virtual environment if needed

If you do not already have a `.venv` folder, run:

```powershell
python -m venv .venv
```

### 3. Install the dependencies

Run:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This installs the Python packages listed in `requirements.txt`.

### 4. Verify the required files

Before starting the dashboard, check that these files exist:

- `data/creditcard.csv`
- `models/fraud_detection_model.joblib`
- `models/feature_names.joblib`
- `results/model_comparison.csv`
- `results/final_test_results.csv`

If any required artifact is missing, the features that depend on it may not work. Download the dataset or follow the project's training workflow to regenerate artifacts where appropriate.

## Running the Dashboard

From the project directory, run:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will start the local application. Open the following address in your browser:

http://localhost:8501

To stop the dashboard, return to the PowerShell terminal and press `Ctrl+C`.

### Restarting after configuration changes

If you modify `app.py` or `.streamlit/config.toml`, stop the running process and start it again using the command above. A full restart is required for changes to Streamlit server configuration to take effect.

## Dashboard Pages

### 1. Overview

Provides a summary of the dataset, including transaction counts, fraud prevalence, class distribution, and the saved model's final test performance.

It also includes a **Try a demo** option for users who want to explore the dashboard quickly.

### 2. Fraud Detection

Provides three transaction analysis modes:

- **Sample transaction:** Run the model on an actual transaction selected from the original dataset.
- **Manual entry:** Enter feature values manually. Advanced anonymized features are hidden in a collapsible section.
- **Batch CSV:** Upload a CSV containing transactions with the required model features.

The dashboard presents predictions and estimated fraud probabilities. Batch results can be downloaded where supported by the application.

### 3. Model Performance

Displays the saved model evaluation results, including validation and final test metrics.

Depending on the available evaluation artifacts and implementation, this page can include a confusion matrix, precision–recall curve, threshold trade-off analysis, and plain-English explanations of evaluation metrics.

### 4. Transaction Explorer

Allows users to inspect historical transactions and their model risk scores.

Historical fraud labels and model predictions are displayed separately so users can distinguish the known dataset label from the model's prediction.

### 5. Dataset Lab

Allows users to inspect another public dataset, explore its structure, and check whether its columns and values are compatible with the saved model.

Incompatible datasets can be explored without scoring or used as the basis for a separate model-training workflow.

## Understanding the Model Features

The saved model expects 30 input features in this exact order:

```text
Time, V1, V2, V3, ..., V28, Amount
```

The `Class` column is the target label and must not be included as a prediction input.

### What are V1–V28?

The columns `V1` through `V28` are anonymized features produced using Principal Component Analysis (PCA) in the original dataset.

They do not directly represent understandable information such as a merchant name, card number, transaction location, or cardholder identity.

Because their original meanings are not provided, most users cannot enter meaningful values for these features manually.

For this reason, **sample mode and compatible CSV batch predictions are recommended** unless you already have valid values for all required features.

### Important preprocessing note

The saved model pipeline must be used as provided. Do not apply additional scaling or transformations unless the model's original training workflow requires them.

The current project documentation describes the saved artifact as a classifier without an included scaler. Features are therefore passed through unchanged. If the model artifacts are regenerated, ensure the training and prediction preprocessing remain consistent.

## Sample Transactions

Sample mode selects a real row from `data/creditcard.csv` using a reproducible seed.

The dashboard uses the row's 30 model features to generate a prediction. If the row contains a historical `Class` label, the label is displayed separately from the prediction.

The application can then indicate whether the prediction matches the known label.

Sample predictions are demonstrations of model behavior. They are not guarantees that the model will correctly identify future fraud.

## Decision Threshold and Risk Levels

The default decision threshold is **50%**, and the application allows it to be configured.

A lower threshold may identify more fraudulent transactions but can also increase false alarms. A higher threshold may reduce false alarms while missing more fraudulent transactions.

The documented risk bands are:

| Risk level | Estimated fraud probability |
|---|---:|
| Low | Below 10% |
| Medium | 10% to below 30% |
| High | 30% or above |

Risk levels are separate from the decision threshold. A risk category describes the estimated probability band, while the threshold determines how the application converts the model's probability into a fraud or legitimate prediction.

**A model probability is an estimate, not proof of fraud.** These results should not be used as the sole basis for real financial decisions.

## Using Other Datasets

The Dataset Lab can help explore other datasets, but matching column names alone does not guarantee compatibility.

### Credit Card Fraud Detection Dataset 2023

Dataset link: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

This dataset includes `V1`–`V28`, `Amount`, `Class`, and an `id` column, but it does not contain the original dataset's `Time` feature.

The saved model requires `Time`. The `id` column is not a valid substitute for `Time`, so this dataset must not be passed directly to the existing model as though it were compatible.

Available approaches include:

1. Explore the dataset without generating predictions from the existing model.
2. Build and evaluate a separate model trained on this dataset's own schema.

Check the dataset's Kaggle page for its licence and usage terms before redistributing or using it.

### PaySim Synthetic Mobile Money Dataset

Dataset link: https://www.kaggle.com/datasets/ealaxi/paysim1

PaySim contains mobile-money transaction data with a different schema, including fields such as `step`, `type`, account balances, and `isFraud`.

It is **not compatible** with the existing credit-card fraud model. Using it for fraud prediction requires a separate preprocessing and model-training workflow.

## Dataset Lab Upload Limits

The Dataset Lab accepts CSV uploads up to **400 MB**.

The Streamlit server limit is configured in `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 400
```

Change this setting in the configuration file rather than hard-coding a different server limit in the app. Restart Streamlit after modifying it.

### Memory considerations

The selected CSV is parsed into a pandas DataFrame in memory. Large files can require several times their file size in RAM, especially when analysis creates additional intermediate data.

A CSV approaching 400 MB may therefore require several gigabytes of available memory. If the dashboard runs out of memory, close other applications or try a smaller file.

### Preview and validation

- The interface previews only the first 100 rows.
- Statistics and compatibility checks must use the complete dataset, not just the preview.
- The application should not silently truncate the dataset or present sample-only statistics as full-dataset results.
- Empty, malformed, oversized, or unsupported files should produce clear error messages.
- Required model columns and values must be validated before scoring.
- The application may cache parsed data and statistics for the current session to avoid unnecessary repeated reads, depending on the implementation.

The **Fraud Detection → Batch CSV** uploader retains its separate 200 MB limit. Increasing the Dataset Lab limit does not automatically increase the batch prediction limit.

## Model Evaluation Results

The dashboard uses the project's saved evaluation artifacts.

### `results/model_comparison.csv`

Contains the validation comparison between Logistic Regression and Random Forest. These results explain how the candidate models were compared and how the final model was selected.

### `results/final_test_results.csv`

Contains the final held-out test metrics for the selected model. These are the headline performance results for the project.

### Confusion matrix

The confusion matrix should be based on genuine held-out labels and predictions generated using the saved model and the documented deterministic data split.

The project uses a stratified 70/15/15 train, validation, and test split with `random_state=42` after duplicate removal. Evaluation results should only be reported when supported by the saved artifacts or reproducible calculations.

### Why accuracy is not enough

Fraud detection is an imbalanced classification problem: fraudulent transactions make up a very small proportion of the original dataset.

A model can achieve high accuracy simply by predicting that nearly every transaction is legitimate.

For this reason, metrics such as **precision, recall, F1-score, average precision, and the confusion matrix** provide important additional information.

- **Precision:** Of the transactions predicted as fraud, how many were actually fraudulent?
- **Recall:** Of the actual fraudulent transactions, how many did the model detect?
- **F1-score:** A combined measure of precision and recall.
- **Average precision:** Summarizes precision–recall performance across different thresholds.
- **Confusion matrix:** Shows true positives, false positives, true negatives, and false negatives.

All displayed values should come from actual model evaluation results rather than invented or hard-coded performance numbers.

## Limitations

- The saved model is trained for the original credit-card dataset's feature schema.
- The anonymized PCA features are difficult to interpret or enter manually.
- Predictions depend on the quality and compatibility of the input data.
- A predicted fraud probability is not a guarantee of fraud.
- Other datasets generally require their own preprocessing and training pipelines.
- Large CSV uploads can consume significant memory.
- The dashboard is intended for educational and analytical use, not as a production banking fraud prevention system.
- Local processing does not eliminate the need to protect sensitive data stored on your computer.

## Technologies Used

- **Python** — core programming language.
- **Streamlit** — local interactive dashboard.
- **Plotly** — interactive charts and visualizations.
- **pandas** — dataset loading and analysis.
- **NumPy** — numerical operations.
- **scikit-learn** — machine learning and evaluation.
- **joblib** — loading and saving model artifacts.
- **Jupyter Notebook** — exploratory data analysis and model development.

## Disclaimer

This project was developed for educational and machine learning demonstration purposes. It does not guarantee the detection of every fraudulent transaction and should not be used as a standalone financial security system.

The original dataset is provided through Kaggle. Refer to the dataset page for its terms, attribution requirements, and usage conditions.

---

**Sentinel Fraud Analytics** — making machine learning fraud analysis easier to explore and understand.