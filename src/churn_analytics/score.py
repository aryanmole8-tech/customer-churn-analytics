
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from train import (
    DATA_PATH,
    FEATURE_COLUMNS,
    RANDOM_STATE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "exports" /
    "heldout_customer_risk_scores.csv"
)


def score_customers():

    print("\n=== CUSTOMER CHURN RISK SCORING ===")

    # --------------------------------------------------
    # 1. LOAD SAVED MODEL AND METADATA
    # --------------------------------------------------

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            "Scoring output already exists. "
            "Refusing to overwrite it."
        )

    model = joblib.load(MODEL_PATH)

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    if metadata["random_state"] != RANDOM_STATE:
        raise ValueError("Random state mismatch.")

    if metadata["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Model feature mismatch.")

    threshold = float(metadata["threshold"])

    if not 0 < threshold < 1:
        raise ValueError("Invalid threshold.")

    model_version = metadata["model_version"]

    print("Loaded model:", model_version)
    print("Classification threshold:", threshold)

    # --------------------------------------------------
    # 2. LOAD CLEANED DATA
    # --------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    if df["customerID"].isna().any():
        raise ValueError("Missing customer identifiers.")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customer identifiers.")

    if len(df) != sum([
        metadata["training_records"],
        metadata["validation_records"],
        metadata["test_records"],
    ]):
        raise ValueError("Unexpected dataset row count.")

    X = df[FEATURE_COLUMNS].copy()

    # Target used only to reconstruct the original
    # stratified split, never as an input to the model.
    y = df["churn_flag"].astype(int)

    # --------------------------------------------------
    # 3. RECONSTRUCT ORIGINAL DATA SPLIT
    # --------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    if len(X_test) != metadata["test_records"]:
        raise ValueError("Unexpected test partition size.")

    if (
        set(X_train.index) & set(X_test.index)
        or set(X_val.index) & set(X_test.index)
    ):
        raise AssertionError("Data split overlap detected.")

    # --------------------------------------------------
    # 4. GENERATE HELD-OUT PREDICTIONS
    # --------------------------------------------------

    probabilities = model.predict_proba(X_test)[:, 1]

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite churn probabilities.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities outside [0, 1].")

    result = pd.DataFrame({
        "customer_id": df.loc[X_test.index, "customerID"].values,
        "model_version": model_version,
        "data_split": "test",
        "churn_probability": probabilities,
        "MonthlyCharges": df.loc[
            X_test.index, "MonthlyCharges"
        ].values,
    })

    # Use the validation-selected classification threshold.
    # These are threshold flags, not independently
    # validated high/medium/low risk bands.
    result["risk_segment"] = np.where(
        result["churn_probability"] >= threshold,
        "Flagged",
        "Below threshold",
    )

    # Illustrative monthly charge exposure proxy.
    # Not observed lost revenue or campaign benefit.
    result["expected_monthly_charge_exposure_cu"] = (
        result["churn_probability"]
        * result["MonthlyCharges"]
    )

    # --------------------------------------------------
    # 5. VALIDATE AND EXPORT
    # --------------------------------------------------

    if result["customer_id"].duplicated().any():
        raise AssertionError("Duplicate scored customers.")

    if len(result) != len(X_test):
        raise AssertionError("Scoring row count mismatch.")

    if result.isna().any().any():
        raise AssertionError("Missing values in scoring output.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT_PATH, index=False)

    print("\n=== SCORING SUMMARY ===")
    print("Scored customers:", len(result))
    print("Model version:", model_version)

    print("\nThreshold segments:")
    print(result["risk_segment"].value_counts())

    print("\nProbability summary:")
    print(result["churn_probability"].describe().round(4))

    print("\nIllustrative probability-weighted")
    print("monthly charge exposure (CU):")
    print(
        round(
            result[
                "expected_monthly_charge_exposure_cu"
            ].sum(),
            2,
        )
    )

    print("\nSaved:", OUTPUT_PATH)

    print("\n=== SCORING COMPLETE ===")


if __name__ == "__main__":
    score_customers()
