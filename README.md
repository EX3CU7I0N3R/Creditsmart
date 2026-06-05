# CreditSmart

CreditSmart is a Streamlit dashboard for credit risk modeling. It trains baseline and ensemble classifiers on loan data, evaluates default-risk performance, and writes model diagnostics and explainability plots.

## Features

- Upload a CSV dataset from the dashboard
- Train logistic regression, random forest, XGBoost, and stacked ensemble models
- Tune decision thresholds from validation data
- Generate ROC, precision-recall, confusion matrix, SHAP, and permutation-importance artifacts
- Download the model metrics table

## Dataset

The pipeline expects a CSV with a binary `not.fully.paid` target column. A sample LendingClub-style dataset is included as `sample_loan_data.csv`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Dashboard

```powershell
streamlit run app.py
```

Upload `sample_loan_data.csv`, then click **Execute Model Pipeline**.

## Run the Pipeline Directly

```powershell
python creditsmart_pipeline.py --data sample_loan_data.csv
```

Outputs are written to `artifacts/`.

## Notes

CreditSmart is intended for model exploration and portfolio-risk analysis. It should not be used as the sole basis for lending decisions without model governance, fairness review, monitoring, and validation against production data.
