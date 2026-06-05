import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

TARGET_COLUMN = "not.fully.paid"


def engineer_features(loan_records: pd.DataFrame) -> pd.DataFrame:
    loans = loan_records.copy()

    if "purpose" in loans.columns and loans["purpose"].dtype == "object":
        loans["purpose"] = LabelEncoder().fit_transform(loans["purpose"])

    if {"revol.bal", "days.with.cr.line"}.issubset(loans.columns):
        loans["credit_utilization_rate"] = loans["revol.bal"] / (loans["days.with.cr.line"] + 1.0)

    if {"log.annual.inc", "installment"}.issubset(loans.columns):
        loans["income_to_installment"] = loans["log.annual.inc"] / (loans["installment"] + 1.0)

    if "fico" in loans.columns:
        loans["fico_bucket"] = pd.cut(
            loans["fico"],
            bins=[-np.inf, 600, 650, 700, 750, 800, np.inf],
            labels=[0, 1, 2, 3, 4, 5],
        ).astype(int)

    if {"int.rate", "fico"}.issubset(loans.columns):
        loans["rate_fico_interact"] = loans["int.rate"] * (850 - loans["fico"])

    if {"dti", "log.annual.inc"}.issubset(loans.columns):
        loans["dti_per_income"] = loans["dti"] / (loans["log.annual.inc"] + 1.0)

    return loans


def build_models() -> dict[str, object]:
    logistic_regression = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    random_forest = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    xgboost = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stacked_ensemble = StackingClassifier(
        estimators=[
            ("logistic_regression", logistic_regression),
            ("random_forest", random_forest),
            ("xgboost", xgboost),
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        cv=cross_validator,
        n_jobs=-1,
    )

    return {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest,
        "XGBoost": xgboost,
        "CreditSmart (Stacked Ensemble)": stacked_ensemble,
    }


def build_pipeline(estimator: object, feature_columns: list[str]) -> ImbalancedPipeline:
    preprocessor = ColumnTransformer(
        [("numeric", StandardScaler(), feature_columns)],
        remainder="drop",
    )
    return ImbalancedPipeline(
        [
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", clone(estimator)),
        ]
    )


def threshold_from_validation(target: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    f1_scores = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
    best_index = int(np.nanargmax(f1_scores))

    if best_index < 1 or len(thresholds) == 0:
        return 0.5

    return float(thresholds[best_index - 1])


def evaluate_model(name: str, target: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float | str]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "Model": name,
        "Accuracy": accuracy_score(target, predictions),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1-Score": f1_score(target, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(target, probabilities),
    }


def plot_curves(target: pd.Series, probabilities: np.ndarray, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    false_positive_rate, true_positive_rate, _ = roc_curve(target, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(false_positive_rate, true_positive_rate, label=f"AUC={roc_auc_score(target, probabilities):.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / "roc_curve.png", dpi=200)
    plt.close()

    precision, recall, _ = precision_recall_curve(target, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(artifacts_dir / "pr_curve.png", dpi=200)
    plt.close()


def plot_confusion_matrix(target: pd.Series, probabilities: np.ndarray, threshold: float, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(target, predictions)

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, interpolation="nearest")
    plt.title(f"Confusion Matrix @ threshold={threshold:.2f}")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["Good (0)", "Risk (1)"])
    plt.yticks(ticks, ["Good (0)", "Risk (1)"])

    midpoint = matrix.max() / 2
    for row_index in range(2):
        for column_index in range(2):
            color = "white" if matrix[row_index, column_index] > midpoint else "black"
            plt.text(column_index, row_index, format(matrix[row_index, column_index], "d"), ha="center", color=color)

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(artifacts_dir / "confusion_matrix.png", dpi=200)
    plt.close()


def plot_shap_explanations(model: XGBClassifier, fitted_features: np.ndarray, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    background = shap.sample(fitted_features, 5000, random_state=42) if fitted_features.shape[0] > 5000 else fitted_features
    shap_values = shap.TreeExplainer(model)(background)

    plt.figure(figsize=(8, 6))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(artifacts_dir / "shap_summary.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(artifacts_dir / "shap_beeswarm.png", dpi=200)
    plt.close()

    print(f"[INFO] SHAP plots saved to {artifacts_dir}")


def plot_permutation_importance(
    model: XGBClassifier,
    fitted_features: np.ndarray,
    target: pd.Series,
    artifacts_dir: Path,
) -> None:
    print("[INFO] Using permutation importance fallback")
    importance = permutation_importance(model, fitted_features, target, n_repeats=10, random_state=42)
    ranked_features = pd.DataFrame(
        {
            "Feature Index": np.arange(fitted_features.shape[1]),
            "Importance": importance.importances_mean,
        }
    ).sort_values(by="Importance", ascending=False)

    top_features = ranked_features.head(20)
    plt.figure(figsize=(8, 6))
    plt.barh(top_features["Feature Index"].astype(str), top_features["Importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.ylabel("Feature Index")
    plt.title("Permutation Feature Importance (Top 20)")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(artifacts_dir / "pfi_importance.png", dpi=200)
    plt.close()
    print(f"[INFO] Permutation importance plot saved to {artifacts_dir}")


def run_pipeline(data_path: Path, artifacts_dir: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    loan_records = pd.read_csv(data_path)
    if TARGET_COLUMN not in loan_records.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing")

    loans = engineer_features(loan_records)
    target = loans[TARGET_COLUMN].astype(int)
    features = loans.drop(columns=[TARGET_COLUMN])
    feature_columns = features.columns.tolist()

    train_features_full, test_features, train_target_full, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )
    train_features, validation_features, train_target, validation_target = train_test_split(
        train_features_full,
        train_target_full,
        test_size=0.2,
        stratify=train_target_full,
        random_state=42,
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, float | str]] = []
    thresholds: dict[str, float] = {}

    for name, estimator in build_models().items():
        print(f"[TRAIN] {name}")
        model_pipeline = build_pipeline(estimator, feature_columns)
        model_pipeline.fit(train_features, train_target)

        validation_probabilities = model_pipeline.predict_proba(validation_features)[:, 1]
        threshold = threshold_from_validation(validation_target, validation_probabilities)
        thresholds[name] = threshold

        test_probabilities = model_pipeline.predict_proba(test_features)[:, 1]
        metrics.append(evaluate_model(name, test_target, test_probabilities, threshold))

        if name == "CreditSmart (Stacked Ensemble)":
            plot_curves(test_target, test_probabilities, artifacts_dir)
            plot_confusion_matrix(test_target, test_probabilities, threshold, artifacts_dir)

        if name == "XGBoost":
            classifier = model_pipeline.named_steps["classifier"]
            fitted_features = model_pipeline.named_steps["preprocessor"].transform(train_features)
            try:
                plot_shap_explanations(classifier, fitted_features, artifacts_dir)
            except Exception as exc:
                print(f"[SHAP] Failed: {exc}")
                plot_permutation_importance(classifier, fitted_features, train_target, artifacts_dir)

    metrics_table = pd.DataFrame(metrics).sort_values(by="ROC-AUC", ascending=False)
    metrics_table.to_csv(artifacts_dir / "metrics_table.csv", index=False)

    print("\n=== CreditSmart Test Metrics (F1-tuned threshold) ===")
    print(metrics_table.to_string(index=False))
    print("\nThresholds:", {name: round(threshold, 3) for name, threshold in thresholds.items()})
    print(f"\nArtifacts saved to: {artifacts_dir.resolve()}")
    print("Done.")

    return metrics_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CreditSmart risk models and generate diagnostics.")
    parser.add_argument("--data", type=Path, default=Path("sample_loan_data.csv"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_pipeline(args.data, args.artifacts_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
