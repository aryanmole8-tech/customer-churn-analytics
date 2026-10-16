
from pathlib import Path
import json

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "customers_clean.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42

# ============================================================
# 2. MODEL INPUT FEATURES
# ============================================================

# Numeric customer attributes
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]

# Categorical customer attributes
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# 3. BUILD LEAKAGE-SAFE PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor():

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        ),
    ])

    return preprocessor


# ============================================================
# 4. MODEL EVALUATION FUNCTION
# ============================================================

def calculate_metrics(model, X, y):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y, predictions
        ),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y, probabilities
        ),
        "pr_auc": average_precision_score(
            y, probabilities
        ),
    }

    return metrics


# ============================================================
# 5. MAIN TRAINING FUNCTION
# ============================================================

def train_models():

    print("\n=== CUSTOMER CHURN ML TRAINING ===")

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully!")
    print("Total customer records:", len(df))

    if df["customerID"].isna().any():
        raise ValueError("Missing customer IDs detected.")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")

    if df[FEATURE_COLUMNS].isna().all(axis=1).any():
        raise ValueError(
            "Records with entirely missing model features detected."
        )

    if df["churn_flag"].isna().any():
        raise ValueError("Missing churn labels detected.")

    if not df["churn_flag"].isin([0, 1]).all():
        raise ValueError("Invalid churn labels detected.")

    # --------------------------------------------------------
    # DEFINE FEATURES AND TARGET
    # --------------------------------------------------------

    X = df[FEATURE_COLUMNS].copy()

    y = df["churn_flag"].astype(int)

    print("\nTotal input features:", len(FEATURE_COLUMNS))
    print("Target column: churn_flag")

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    # --------------------------------------------------------
    # STRATIFIED TRAIN / VALIDATION / TEST SPLIT
    # --------------------------------------------------------

    # 60% training and 40% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Split temporary data into 20% validation and 20% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print("\n=== DATASET SPLIT ===")

    print("Training records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Test records (untouched):", len(X_test))

    # --------------------------------------------------------
    # MODEL 1: DUMMY CLASSIFIER
    # --------------------------------------------------------

    dummy_model = Pipeline([
        (
            "preprocessor",
            build_preprocessor()
        ),
        (
            "classifier",
            DummyClassifier(
                strategy="most_frequent"
            )
        ),
    ])

    # --------------------------------------------------------
    # MODEL 2: LOGISTIC REGRESSION
    # --------------------------------------------------------

    logistic_model = Pipeline([
        (
            "preprocessor",
            build_preprocessor()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        ),
    ])

    # --------------------------------------------------------
    # MODEL 3: RANDOM FOREST
    # --------------------------------------------------------

    random_forest_model = Pipeline([
        (
            "preprocessor",
            build_preprocessor()
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        ),
    ])

    # --------------------------------------------------------
    # MODEL COLLECTION
    # --------------------------------------------------------

    models = {
        "Dummy Classifier": dummy_model,
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model,
    }

    results = []

    # --------------------------------------------------------
    # TRAIN AND VALIDATE EACH MODEL
    # --------------------------------------------------------

    for name, model in models.items():

        print(f"\nTraining: {name}")

        model.fit(X_train, y_train)

        print(f"Training completed: {name}")

        # Validation set only.
        # Test set remains untouched.
        metrics = calculate_metrics(
            model,
            X_val,
            y_val
        )

        results.append({
            "model": name,
            **metrics
        })

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n=== VALIDATION MODEL COMPARISON ===")

    print(
        results_df.round(4).to_string(index=False)
    )

    # --------------------------------------------------------
    # SAVE VALIDATION RESULTS
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison_path = (
        REPORT_DIR / "validation_model_comparison.csv"
    )

    results_df.to_csv(
        comparison_path,
        index=False
    )

    # --------------------------------------------------------
    # SAVE SPLIT METADATA
    # --------------------------------------------------------

    split_metadata = {
        "random_state": RANDOM_STATE,
        "split_method": "stratified_random",
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
        "feature_columns": FEATURE_COLUMNS,
        "trained_models": list(models.keys()),
        "test_set_evaluated": False
    }

    metadata_path = (
        REPORT_DIR / "split_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            split_metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n=== OUTPUT FILES ===")
    print("Model comparison:", comparison_path)
    print("Split metadata:", metadata_path)

    print("\nTest set remains untouched.")
    print("No final model has been selected yet.")

    print("\n=== TRAINING COMPLETE ===")


# ============================================================
# 6. EXECUTE
# ============================================================

if __name__ == "__main__":
    train_models()
