
from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_DATA = PROJECT_ROOT / "data" / "processed" / "customers_clean.csv"
DATABASE_PATH = PROJECT_ROOT / "database" / "customer_churn.db"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


def create_database():
    print("\n=== SQLITE DATABASE SETUP ===")

    df = pd.read_csv(CLEAN_DATA)

    if df["customerID"].isna().any():
        raise ValueError("Missing customer IDs detected.")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")

    df = df.rename(columns={"customerID": "customer_id"})

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        # Create all tables from the version-controlled schema.
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

        # Refuse to overwrite an existing customer snapshot.
        existing_count = conn.execute(
            "SELECT COUNT(*) FROM customers_clean"
        ).fetchone()[0]

        if existing_count != 0:
            raise RuntimeError(
                "customers_clean already contains records. "
                "Existing data was not modified."
            )

        # Load cleaned customer data.
        df.to_sql(
            "customers_clean",
            conn,
            if_exists="append",
            index=False
        )

        # Reconcile source and database counts.
        db_count = conn.execute(
            "SELECT COUNT(*) FROM customers_clean"
        ).fetchone()[0]

        db_unique = conn.execute(
            "SELECT COUNT(DISTINCT customer_id) FROM customers_clean"
        ).fetchone()[0]

        db_churned = conn.execute(
            "SELECT SUM(churn_flag) FROM customers_clean"
        ).fetchone()[0]

        fk_issues = conn.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

        if db_count != len(df):
            raise AssertionError("Database row count mismatch.")

        if db_unique != db_count:
            raise AssertionError("Customer ID uniqueness failure.")

        if db_churned != int(df["churn_flag"].sum()):
            raise AssertionError("Churn count mismatch.")

        if fk_issues:
            raise AssertionError("Foreign key problems detected.")

        print("Database:", DATABASE_PATH)
        print("CSV rows:", len(df))
        print("Database rows:", db_count)
        print("Unique customers:", db_unique)
        print("Observed churned customers:", db_churned)
        print("Foreign key issues:", len(fk_issues))

        print("\nTables:")
        tables = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()

        for table in tables:
            print("-", table[0])

    print("\n=== DATABASE SETUP COMPLETE ===")


if __name__ == "__main__":
    create_database()
