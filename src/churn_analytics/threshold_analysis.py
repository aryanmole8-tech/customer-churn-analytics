
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Reuse the exact feature definitions and preprocessing
# from our existing training script.
from train import (
    DATA_PATH,
    RANDOM_STATE,
    FEATURE_COLUMNS,
    build_preprocessor
)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports"


def analyze_thresholds():

    print("\n=== VALIDATION THRESHOLD ANALYSIS ===")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS].copy()
    y = df["churn_flag"].astype(int)

    # Same reproducible 60/20/20 split.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    # Check split integrity.
    assert X_train.index.is_unique
    assert X_val.index.is_unique
    assert X_test.index.is_unique

    assert not set(X_train.index) & set(X_val.index)
    assert not set(X_train.index) & set(X_test.index)
    assert not set(X_val.index) & set(X_test.index)

    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ])

    model.fit(X_train, y_train)

    # Validation probabilities only.
    probabilities = model.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.20, 0.71, 0.05)

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1]
        ).ravel()

        results.append({
            "threshold": round(float(threshold), 2),
            "precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_val, predictions, zero_division=0
            ),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn)
        })

    results_df = pd.DataFrame(results)

    # Select threshold using validation F1 only.
    best_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    selected_threshold = float(best_row["threshold"])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = REPORT_DIR / "validation_threshold_analysis.csv"

    results_df.to_csv(csv_path, index=False)

    selection = {
        "model": "Logistic Regression",
        "selection_metric": "validation_f1",
        "selected_threshold": selected_threshold,
        "random_state": RANDOM_STATE,
        "test_set_evaluated": False
    }

    json_path = REPORT_DIR / "threshold_selection.json"

    json_path.write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8"
    )

    print("\n=== THRESHOLD COMPARISON ===")
    print(results_df.round(4).to_string(index=False))

    print("\n=== SELECTED THRESHOLD ===")
    print("Threshold:", selected_threshold)
    print("Validation F1:", round(float(best_row["f1"]), 4))
    print("Validation Recall:", round(float(best_row["recall"]), 4))

    print("\nSaved:", csv_path)
    print("Saved:", json_path)

    print("\nTest labels remain unused.")
    print("\n=== THRESHOLD ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    analyze_thresholds()
