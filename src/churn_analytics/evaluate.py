
from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)

from train import (
    DATA_PATH,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    build_preprocessor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"

THRESHOLD_PATH = REPORT_DIR / "threshold_selection.json"
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
REPORT_PATH = REPORT_DIR / "model_evaluation.md"

MODEL_VERSION = "logistic_regression_v1"


def evaluate_final_model():

    print("\n=== FINAL HELD-OUT MODEL EVALUATION ===")

    # Prevent accidental repeated test evaluation.
    output_files = [
        MODEL_PATH,
        METADATA_PATH,
        REPORT_PATH,
    ]

    existing = [str(p) for p in output_files if p.exists()]

    if existing:
        raise FileExistsError(
            "Final evaluation outputs already exist. "
            "Refusing to overwrite them:\n" +
            "\n".join(existing)
        )

    # Load previously selected validation threshold.
    selection = json.loads(
        THRESHOLD_PATH.read_text(encoding="utf-8")
    )

    if selection["model"] != "Logistic Regression":
        raise ValueError("Unexpected selected model.")

    if selection["random_state"] != RANDOM_STATE:
        raise ValueError("Random seed mismatch.")

    threshold = float(selection["selected_threshold"])

    if not 0 < threshold < 1:
        raise ValueError("Invalid selected threshold.")

    print("Selected model:", MODEL_VERSION)
    print("Validation-selected threshold:", threshold)

    # Load cleaned customer snapshot.
    df = pd.read_csv(DATA_PATH)

    if df["customerID"].isna().any():
        raise ValueError("Missing customer identifiers.")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customer identifiers.")

    X = df[FEATURE_COLUMNS].copy()
    y = df["churn_flag"].astype(int)

    if not y.isin([0, 1]).all():
        raise ValueError("Invalid churn labels.")

    # Reconstruct the exact original data split.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    if (
        set(X_train.index) & set(X_val.index)
        or set(X_train.index) & set(X_test.index)
        or set(X_val.index) & set(X_test.index)
    ):
        raise AssertionError("Overlapping dataset splits.")

    print("\nTraining records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Test records:", len(X_test))

    # Train on training data only, maintaining
    # the exact selected model configuration.
    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])

    print("\nFitting selected model...")
    model.fit(X_train, y_train)

    # First final evaluation on held-out test records.
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(
            y_test, probabilities
        ),
        "brier_score": brier_score_loss(
            y_test, probabilities
        ),
    }

    cm = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    print("\n=== FINAL TEST METRICS ===")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n=== CONFUSION MATRIX ===")
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)

    report = classification_report(
        y_test,
        predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    )

    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

    # Create output directories.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save the complete fitted preprocessing + ML pipeline.
    joblib.dump(model, MODEL_PATH)

    evaluated_at = datetime.now(
        timezone.utc
    ).isoformat()

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "LogisticRegression",
        "threshold": threshold,
        "threshold_source": "validation_f1",
        "random_state": RANDOM_STATE,
        "training_records": len(X_train),
        "validation_records": len(X_val),
        "test_records": len(X_test),
        "feature_columns": FEATURE_COLUMNS,
        "test_metrics": {
            key: float(value)
            for key, value in metrics.items()
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "evaluated_at_utc": evaluated_at,
        "limitations": [
            "Fictional cross-sectional IBM telecom dataset.",
            "Random holdout does not establish future forecasting performance.",
            "Probabilities are not guaranteed to be calibrated.",
            "No observed retention campaign outcomes.",
        ],
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Write a readable report for GitHub documentation.
    lines = [
        "# Customer Churn Model Evaluation",
        "",
        f"Model: {MODEL_VERSION}",
        "",
        f"Classification threshold: {threshold:.2f}",
        "",
        "## Held-out test metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    for name, value in metrics.items():
        lines.append(f"| {name} | {value:.4f} |")

    lines.extend([
        "",
        "## Confusion Matrix",
        "",
        f"- True negatives: {tn}",
        f"- False positives: {fp}",
        f"- False negatives: {fn}",
        f"- True positives: {tp}",
        "",
        "## Classification Report",
        "",
        "```text",
        report,
        "```",
        "",
        "## Limitations",
        "",
        "This is a fictional historical customer snapshot.",
        "A stratified random holdout does not prove future",
        "churn forecasting performance or business uplift.",
        "Retention outcomes are not observed in this dataset.",
        "",
    ])

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n=== SAVED FILES ===")
    print("Model:", MODEL_PATH)
    print("Metadata:", METADATA_PATH)
    print("Report:", REPORT_PATH)

    print("\n=== FINAL EVALUATION COMPLETE ===")


if __name__ == "__main__":
    evaluate_final_model()
