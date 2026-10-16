
from pathlib import Path
from datetime import datetime, timezone
import json
import sqlite3

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = PROJECT_ROOT / "database" / "customer_churn.db"
SCORES_PATH = (
    PROJECT_ROOT / "data" / "exports" /
    "heldout_customer_risk_scores.csv"
)
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


def load_predictions():
    print("\n=== LOAD ML RESULTS INTO SQLITE ===")

    # Load outputs from our completed ML pipeline
    scores = pd.read_csv(SCORES_PATH)
    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    model_version = metadata["model_version"]
    test_metrics = metadata["test_metrics"]

    required = {
        "customer_id",
        "model_version",
        "data_split",
        "churn_probability",
        "risk_segment",
    }

    missing = required - set(scores.columns)

    if missing:
        raise ValueError(f"Missing scoring columns: {missing}")

    if len(scores) != metadata["test_records"]:
        raise ValueError("Scored row count mismatch.")

    if scores["customer_id"].duplicated().any():
        raise ValueError("Duplicate scored customer IDs.")

    if scores[list(required)].isna().any().any():
        raise ValueError("Missing prediction values.")

    if not scores["churn_probability"].between(0, 1).all():
        raise ValueError("Invalid churn probability detected.")

    if not scores["model_version"].eq(model_version).all():
        raise ValueError("Model version mismatch.")

    if not scores["data_split"].eq("test").all():
        raise ValueError("Unexpected scoring split.")

    scored_at = datetime.now(timezone.utc).isoformat()

    prediction_rows = [
        (
            row.customer_id,
            row.model_version,
            row.data_split,
            float(row.churn_probability),
            row.risk_segment,
            scored_at,
        )
        for row in scores[
            list([
                "customer_id",
                "model_version",
                "data_split",
                "churn_probability",
                "risk_segment",
            ])
        ].itertuples(index=False)
    ]

    metric_rows = [
        (
            model_version,
            "test",
            metric_name,
            float(metric_value),
            metadata["evaluated_at_utc"],
        )
        for metric_name, metric_value in test_metrics.items()
    ]

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        # Avoid overwriting previously loaded ML results.
        existing_predictions = conn.execute(
            """
            SELECT COUNT(*)
            FROM model_predictions
            WHERE model_version = ? AND data_split = 'test'
            """,
            (model_version,),
        ).fetchone()[0]

        if existing_predictions:
            raise RuntimeError(
                "Predictions for this model are already loaded. "
                "Existing records were not modified."
            )

        # Ensure all scored customers exist in customers_clean.
        customer_ids = set(
            row[0] for row in conn.execute(
                "SELECT customer_id FROM customers_clean"
            )
        )

        if not set(scores["customer_id"]).issubset(customer_ids):
            raise ValueError(
                "Some scored customer IDs are missing from SQLite."
            )

        conn.executemany(
            """
            INSERT INTO model_predictions (
                customer_id,
                model_version,
                data_split,
                churn_probability,
                risk_segment,
                scored_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            prediction_rows,
        )

        conn.executemany(
            """
            INSERT INTO model_evaluation (
                model_version,
                data_split,
                metric_name,
                metric_value,
                evaluated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            metric_rows,
        )

        loaded_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM model_predictions
            WHERE model_version = ? AND data_split = 'test'
            """,
            (model_version,),
        ).fetchone()[0]

        metric_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM model_evaluation
            WHERE model_version = ? AND data_split = 'test'
            """,
            (model_version,),
        ).fetchone()[0]

        join_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM model_predictions p
            JOIN customers_clean c
                ON p.customer_id = c.customer_id
            WHERE p.model_version = ?
              AND p.data_split = 'test'
            """,
            (model_version,),
        ).fetchone()[0]

        foreign_key_issues = conn.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

        if loaded_count != len(scores):
            raise AssertionError("Prediction count mismatch.")

        if join_count != loaded_count:
            raise AssertionError("Prediction join count mismatch.")

        if metric_count != len(metric_rows):
            raise AssertionError("Evaluation metric count mismatch.")

        if foreign_key_issues:
            raise AssertionError("Foreign key violations detected.")

        print("Model version:", model_version)
        print("Predictions loaded:", loaded_count)
        print("Evaluation metrics loaded:", metric_count)
        print("Customer join count:", join_count)
        print("Foreign key issues:", len(foreign_key_issues))

    print("\n=== ML DATABASE INTEGRATION COMPLETE ===")


if __name__ == "__main__":
    load_predictions()
