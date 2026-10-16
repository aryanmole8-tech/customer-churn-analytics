from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" /
    "Telco-Customer-Churn.csv"
)


def validate_dataset():
    df = pd.read_csv(DATA_PATH, dtype={"TotalCharges": "string"})

    print("\n========== DATA QUALITY REPORT ==========")

    # 1. Dataset dimensions
    print("\n1. DATASET SHAPE")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    # 2. Data types
    print("\n2. COLUMN DATA TYPES")
    print(df.dtypes.to_string())

    # 3. Missing values
    print("\n3. MISSING VALUES")
    print(df.isna().sum().to_string())

    # 4. Blank TotalCharges
    print("\n4. BLANK TOTAL CHARGES")
    blank_charges = (
        df["TotalCharges"].isna()
        | df["TotalCharges"].str.strip().eq("")
    ).fillna(False)

    print("Blank TotalCharges:", int(blank_charges.sum()))

    print("\nRecords with blank TotalCharges:")
    print(
        df.loc[
            blank_charges,
            ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]
        ].to_string(index=False)
    )

    # 5. Duplicate customer IDs
    print("\n5. DUPLICATE CUSTOMER IDS")
    print(df["customerID"].duplicated().sum())

    # 6. Churn distribution
    print("\n6. CHURN DISTRIBUTION")
    print(df["Churn"].value_counts(dropna=False).to_string())

    print("\nChurn Percentage:")
    print(
        (
            df["Churn"].value_counts(normalize=True, dropna=False)
            * 100
        ).round(2).to_string()
    )

    # 7. Numerical summary
    print("\n7. NUMERICAL SUMMARY")
    print(
        df[["tenure", "MonthlyCharges", "SeniorCitizen"]]
        .describe()
        .to_string()
    )

    # 8. Invalid data checks
    print("\n8. INVALID DATA CHECKS")
    print("Negative tenure:", int((df["tenure"] < 0).sum()))
    print(
        "Negative monthly charges:",
        int((df["MonthlyCharges"] < 0).sum())
    )
    print("Missing customer IDs:", int(df["customerID"].isna().sum()))

    total_numeric = pd.to_numeric(
        df["TotalCharges"].str.strip(),
        errors="coerce"
    )

    invalid_total = (
        total_numeric.isna() & ~blank_charges
    )

    print(
        "Invalid non-blank TotalCharges:",
        int(invalid_total.sum())
    )
    print(
        "Negative TotalCharges:",
        int((total_numeric < 0).sum())
    )

    print("\n========== VALIDATION COMPLETE ==========")


if __name__ == "__main__":
    validate_dataset()