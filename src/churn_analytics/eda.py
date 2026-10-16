
from pathlib import Path
import matplotlib

# Save figures without requiring an interactive window
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "customers_clean.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

sns.set_theme(style="whitegrid")


def save_chart(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def generate_eda():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    required = {
        "customerID", "Churn", "churn_flag", "Contract",
        "tenure_band", "PaymentMethod", "MonthlyCharges"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("\n=== CUSTOMER CHURN EDA ===")
    print("Customers:", len(df))

    # Chart 1: Customer churn distribution
    counts = df["Churn"].value_counts().reindex(["No", "Yes"])

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette="Set2",
        legend=False
    )
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Observed Churn")
    ax.set_ylabel("Number of Customers")

    for i, count in enumerate(counts.values):
        ax.text(i, count + 40, str(count), ha="center")

    save_chart("01_churn_distribution.png")

    # Chart 2: Churn rate by contract type
    contract = (
        df.groupby("Contract")["churn_flag"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        x=contract.index,
        y=contract["mean"] * 100,
        hue=contract.index,
        palette="Blues_d",
        legend=False
    )
    ax.set_title("Observed Churn Rate by Contract Type")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, max(contract["mean"] * 100) * 1.2)

    for i, value in enumerate(contract["mean"] * 100):
        ax.text(i, value + 1, f"{value:.1f}%", ha="center")

    save_chart("02_churn_by_contract.png")

    # Chart 3: Churn rate by tenure band
    tenure_order = ["0-12", "13-24", "25-48", "49-72"]

    tenure = (
        df.groupby("tenure_band")["churn_flag"]
        .mean()
        .reindex(tenure_order) * 100
    )

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x=tenure.index,
        y=tenure.values,
        hue=tenure.index,
        palette="viridis",
        legend=False
    )
    ax.set_title("Observed Churn Rate by Customer Tenure")
    ax.set_xlabel("Tenure Band (Months)")
    ax.set_ylabel("Churn Rate (%)")

    save_chart("03_churn_by_tenure.png")

    # Chart 4: Churn by payment method
    payment = (
        df.groupby("PaymentMethod")["churn_flag"]
        .mean()
        .sort_values(ascending=False) * 100
    )

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        x=payment.values,
        y=payment.index,
        hue=payment.index,
        palette="mako",
        legend=False
    )
    ax.set_title("Observed Churn Rate by Payment Method")
    ax.set_xlabel("Churn Rate (%)")
    ax.set_ylabel("Payment Method")

    save_chart("04_churn_by_payment.png")

    # Chart 5: Monthly charges and churn
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges",
        hue="Churn",
        palette="Set2",
        legend=False,
        order=["No", "Yes"]
    )
    plt.title("Monthly Charges by Observed Churn")
    plt.xlabel("Observed Churn")
    plt.ylabel("Monthly Charges (CU)")

    save_chart("05_monthly_charges_by_churn.png")

    print("\n=== EDA COMPLETE ===")
    print("Figures location:", OUTPUT_DIR)


if __name__ == "__main__":
    generate_eda()
