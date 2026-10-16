
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
OUTPUT_DATA = PROJECT_ROOT / "data" / "processed" / "customers_clean.csv"

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def preprocess_data():
    print("\n=== CUSTOMER CHURN DATA PREPROCESSING ===")

    df = pd.read_csv(RAW_DATA, dtype={"TotalCharges": "string"})

    original_rows = len(df)
    print("Original rows:", original_rows)

    # 1. Validate unique customer identifiers
    if df["customerID"].isna().any():
        raise ValueError("Missing customer IDs detected.")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")

    # 2. Validate target labels
    if df["Churn"].isna().any():
        raise ValueError("Missing churn labels detected.")

    if not df["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Unexpected churn labels detected.")

    # 3. Convert TotalCharges to numeric
    raw_charges = df["TotalCharges"].str.strip()
    blank_charges = raw_charges.isna() | raw_charges.eq("")

    numeric_charges = pd.to_numeric(
        raw_charges.mask(blank_charges),
        errors="coerce"
    )

    invalid_charges = numeric_charges.isna() & ~blank_charges

    if invalid_charges.any():
        raise ValueError("Invalid non-blank TotalCharges detected.")

    # Zero total charges for zero-tenure customers only
    if (df.loc[blank_charges, "tenure"] != 0).any():
        raise ValueError(
            "Blank TotalCharges found for customers with tenure > 0."
        )

    df["TotalCharges"] = numeric_charges.fillna(0.0).astype(float)

    # 4. Validate numerical fields
    for column in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if df[column].isna().any():
            raise ValueError(f"Missing values in {column}")

        if (df[column] < 0).any():
            raise ValueError(f"Negative values in {column}")

    if not df["tenure"].between(0, 72).all():
        raise ValueError("Tenure outside expected range 0-72.")

    # 5. Historical target for analytics
    df["churn_flag"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    # 6. Tenure segmentation
    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"]
    ).astype("string")

    # 7. Monthly charge segmentation
    df["monthly_charge_band"] = pd.cut(
        df["MonthlyCharges"],
        bins=[-np.inf, 35, 70, 100, np.inf],
        labels=["Low", "Medium", "High", "Very High"]
    ).astype("string")

    # 8. Count subscribed services
    # Each "Yes" = one service.
    # DSL/Fiber optic = one internet service.
    # "No internet service" and "No phone service" = zero.
    service_count = pd.Series(0, index=df.index, dtype="int64")

    for column in SERVICE_COLUMNS:
        if column == "InternetService":
            service_count += df[column].isin(
                ["DSL", "Fiber optic"]
            ).astype(int)
        else:
            service_count += df[column].eq("Yes").astype(int)

    df["service_count"] = service_count

    # 9. Final quality checks
    if len(df) != original_rows:
        raise AssertionError("Customer row count changed.")

    if df["customerID"].nunique() != original_rows:
        raise AssertionError("Customer IDs are not unique.")

    if df["TotalCharges"].isna().any():
        raise AssertionError("TotalCharges still contains nulls.")

    # 10. Export cleaned data
    OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DATA, index=False)

    print("\n=== PREPROCESSING SUMMARY ===")
    print("Original records:", original_rows)
    print("Cleaned records:", len(df))
    print("Blank TotalCharges handled:", int(blank_charges.sum()))
    print("Total columns:", df.shape[1])
    print("Duplicate IDs:", int(df["customerID"].duplicated().sum()))

    print("\nTenure bands:")
    print(df["tenure_band"].value_counts().sort_index())

    print("\nChurn rate:")
    print(f"{df['churn_flag'].mean() * 100:.2f}%")

    print("\nSaved cleaned data to:")
    print(OUTPUT_DATA)

    print("\n=== PREPROCESSING COMPLETE ===")


if __name__ == "__main__":
    preprocess_data()
