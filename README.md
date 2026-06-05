# CreditSmart

CreditSmart is an end-to-end credit risk modeling application that combines data ingestion, feature engineering, ensemble learning, threshold tuning, and explainability into a single Streamlit-powered workflow.

## Project Summary

This project demonstrates a production-relevant credit risk pipeline built for loan default prediction. It includes:

- A Streamlit front-end for uploading loan data and launching analysis
- A reusable Python pipeline for data preparation, model training, scoring, and diagnostics
- Four classifiers: Logistic Regression, Random Forest, XGBoost, and a stacked ensemble
- Validation-driven threshold optimization to maximize F1 performance on imbalanced credit data
- Explainable model artifacts including ROC, precision-recall, confusion matrix, SHAP, and permutation importance

## Resume-Worthy Highlights

- Designed and implemented an end-to-end machine learning workflow for credit risk assessment
- Engineered derived features such as credit utilization rate, income-to-installment ratio, FICO buckets, and interaction terms
- Built and evaluated baseline and ensemble classifiers with class balancing using SMOTE
- Tuned decision thresholds using a validation split to improve recall/precision tradeoffs for default risk
- Integrated model diagnostics and visual explainability into a user-facing dashboard

## Technical Stack

- Python 3
- Streamlit for interactive UI
- pandas, NumPy for data manipulation
- scikit-learn for feature scaling, modeling, evaluation, and pipeline management
- XGBoost for gradient-boosted tree modeling
- SHAP for explainability and feature impact visualization
- imbalanced-learn for SMOTE-based oversampling
- Matplotlib for plotting model diagnostics

## Architecture & Pipeline

1. **Data ingestion**: Reads a CSV dataset with a binary `not.fully.paid` target column.
2. **Feature engineering**: Encodes categorical fields, derives credit and income ratios, buckets FICO scores, and constructs interaction terms.
3. **Train / validation / test split**: Splits data using stratified sampling to preserve class balance.
4. **Model pipelines**: Builds a preprocessing pipeline with standard scaling + SMOTE oversampling + classifier.
5. **Model training**: Trains Logistic Regression, Random Forest, XGBoost, and a StackingClassifier ensemble.
6. **Threshold optimization**: Selects the best classification threshold from validation probabilities using F1 score.
7. **Evaluation**: Computes accuracy, precision, recall, F1-score, and ROC-AUC on the held-out test set.
8. **Diagnostics**: Produces ROC and precision-recall curves, confusion matrix, SHAP explainability plots, and permutation feature importance.

## Dashboard Experience

The Streamlit app (`app.py`) enables users to:

- Upload a CSV file and preview loan records
- Execute the model pipeline from the dashboard
- View aggregated model performance metrics
- Inspect diagnostic plots for the stacked ensemble and explainability reports
- Download the model metrics table for offline review

## Dataset Requirements

The pipeline expects a CSV file with the following characteristics:

- A binary target column named `not.fully.paid`
- Loan-level attributes such as `fico`, `int.rate`, `installment`, `revol.bal`, `days.with.cr.line`, `log.annual.inc`, `purpose`, and `dti`

A sample LendingClub-style dataset is provided as `sample_loan_data.csv`.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the App

```powershell
streamlit run app.py
```

Upload `sample_loan_data.csv` and click **Execute Model Pipeline**.

## Running the Pipeline Directly

```powershell
python creditsmart_pipeline.py --data sample_loan_data.csv
```

By default, the pipeline writes outputs to `artifacts/`.

## Output Artifacts

The pipeline writes the following artifacts to the configured `artifacts` directory:

- `metrics_table.csv` — tabular model performance summary
- `roc_curve.png` — ROC curve for the stacked ensemble
- `pr_curve.png` — precision-recall curve
- `confusion_matrix.png` — confusion matrix at the tuned threshold
- `shap_summary.png` — SHAP summary plot for XGBoost
- `shap_beeswarm.png` — SHAP beeswarm plot for XGBoost
- `pfi_importance.png` — permutation feature importance fallback

## Notes

CreditSmart is intended for exploratory risk modeling and educational use. It is not intended to replace production lending systems or formal credit risk governance.