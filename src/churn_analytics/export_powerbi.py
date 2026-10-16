
from pathlib import Path
import json
import sqlite3

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = PROJECT_ROOT / "database" / "customer_churn.db"
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


def export_powerbi():

    print("\n=== POWER BI DATA EXPORT ===")

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    model_version = metadata["model_version"]
    threshold = float(metadata["threshold"])

    if not 0 < threshold < 1:
        raise ValueError("Invalid model threshold.")

    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("PRAGMA foreign_keys = ON")

        # -----------------------------------------
        # 1. HISTORICAL CUSTOMER DATA
        # -----------------------------------------

        customers = pd.read_sql_query(
            """
            SELECT *
            FROM customers_clean
            ORDER BY customer_id
            """,
            conn
        )

        # -----------------------------------------
        # 2. HELD-OUT CUSTOMER RISK
        # -----------------------------------------

        risk = pd.read_sql_query(
            """
            SELECT
                p.customer_id,
                p.model_version,
                p.data_split,
                p.churn_probability,
                p.risk_segment,
                p.scored_at,
                c.Contract,
                c.tenure,
                c.tenure_band,
                c.InternetService,
                c.PaymentMethod,
                c.MonthlyCharges,
                c.churn_flag AS actual_churn_flag,
                p.churn_probability * c.MonthlyCharges
                    AS expected_monthly_charge_exposure_cu
            FROM model_predictions p
            INNER JOIN customers_clean c
                ON p.customer_id = c.customer_id
            WHERE p.model_version = ?
              AND p.data_split = 'test'
            ORDER BY p.customer_id
            """,
            conn,
            params=(model_version,)
        )

        # -----------------------------------------
        # 3. MODEL EVALUATION METRICS
        # -----------------------------------------

        evaluation = pd.read_sql_query(
            """
            SELECT
                model_version,
                data_split,
                metric_name,
                metric_value,
                evaluated_at
            FROM model_evaluation
            WHERE model_version = ?
              AND data_split = 'test'
            ORDER BY metric_name
            """,
            conn,
            params=(model_version,)
        )

        # -----------------------------------------
        # 4. SQL CUSTOMER SEGMENT SUMMARY
        # -----------------------------------------

        segments = pd.read_sql_query(
            """
            SELECT
                Contract,
                tenure_band,
                COUNT(*) AS total_customers,
                SUM(churn_flag) AS churned_customers,
                100.0 * AVG(churn_flag) AS churn_rate_pct,
                AVG(MonthlyCharges) AS avg_monthly_charges_cu
            FROM customers_clean
            GROUP BY Contract, tenure_band
            ORDER BY Contract, tenure_band
            """,
            conn
        )

        foreign_key_issues = conn.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

    # -----------------------------------------
    # 5. DATA VALIDATION
    # -----------------------------------------

    expected_customers = 7043
    expected_predictions = metadata["test_records"]

    if len(customers) != expected_customers:
        raise AssertionError("Customer count mismatch.")

    if customers["customer_id"].duplicated().any():
        raise AssertionError("Duplicate customer IDs.")

    if len(risk) != expected_predictions:
        raise AssertionError("Prediction count mismatch.")

    if risk["customer_id"].duplicated().any():
        raise AssertionError("Duplicate scored customers.")

    if len(evaluation) != len(metadata["test_metrics"]):
        raise AssertionError("Evaluation metric count mismatch.")

    if foreign_key_issues:
        raise AssertionError("Foreign key violations detected.")

    if not risk["churn_probability"].between(0, 1).all():
        raise AssertionError("Invalid churn probabilities.")

    if risk.isna().any().any():
        raise AssertionError("Null values in risk export.")

    if not risk["data_split"].eq("test").all():
        raise AssertionError("Unexpected dataset split.")

    # Check risk classification against saved threshold
    expected_flags = np.where(
        risk["churn_probability"] >= threshold,
        "Flagged",
        "Below threshold"
    )

    if not np.array_equal(
        risk["risk_segment"].to_numpy(),
        expected_flags
    ):
        raise AssertionError("Risk threshold mismatch.")

    if segments["total_customers"].sum() != len(customers):
        raise AssertionError("Segment population mismatch.")

    if segments["churned_customers"].sum() != int(
        customers["churn_flag"].sum()
    ):
        raise AssertionError("Segment churn count mismatch.")

    # -----------------------------------------
    # 6. EXPORT TO CSV
    # -----------------------------------------

    exports = {
        "powerbi_customers.csv": customers,
        "powerbi_customer_risk.csv": risk,
        "powerbi_model_evaluation.csv": evaluation,
        "powerbi_customer_segments.csv": segments,
    }

    for filename, dataframe in exports.items():

        output_path = EXPORT_DIR / filename

        dataframe.to_csv(output_path, index=False)

        print(f"Exported: {filename}")
        print(f"Rows: {len(dataframe)}\n")

    # -----------------------------------------
    # 7. FINAL RECONCILIATION
    # -----------------------------------------

    print("=== POWER BI EXPORT VERIFICATION ===")
    print("Customers:", len(customers))
    print("Observed churned:", int(customers["churn_flag"].sum()))
    print("Scored customers:", len(risk))

    flagged = int(
        risk["risk_segment"].eq("Flagged").sum()
    )

    exposure = risk[
        "expected_monthly_charge_exposure_cu"
    ].sum()

    print("Flagged customers:", flagged)
    print("Model evaluation metrics:", len(evaluation))
    print("Customer segments:", len(segments))
    print("Weighted monthly exposure (CU):", round(exposure, 2))
    print("Foreign key issues:", len(foreign_key_issues))

    print("\nAll reporting exports were generated from SQLite.")
    print("Export directory:", EXPORT_DIR)
    print("\n=== POWER BI EXPORT COMPLETE ===")


if __name__ == "__main__":
    export_powerbi()
