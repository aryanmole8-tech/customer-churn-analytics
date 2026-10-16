
CREATE TABLE IF NOT EXISTS customers_clean (
    customer_id TEXT PRIMARY KEY,
    gender TEXT NOT NULL,
    SeniorCitizen INTEGER NOT NULL,
    Partner TEXT NOT NULL,
    Dependents TEXT NOT NULL,
    tenure INTEGER NOT NULL,
    PhoneService TEXT NOT NULL,
    MultipleLines TEXT NOT NULL,
    InternetService TEXT NOT NULL,
    OnlineSecurity TEXT NOT NULL,
    OnlineBackup TEXT NOT NULL,
    DeviceProtection TEXT NOT NULL,
    TechSupport TEXT NOT NULL,
    StreamingTV TEXT NOT NULL,
    StreamingMovies TEXT NOT NULL,
    Contract TEXT NOT NULL,
    PaperlessBilling TEXT NOT NULL,
    PaymentMethod TEXT NOT NULL,
    MonthlyCharges REAL NOT NULL,
    TotalCharges REAL NOT NULL,
    Churn TEXT NOT NULL,
    churn_flag INTEGER NOT NULL CHECK (churn_flag IN (0, 1)),
    tenure_band TEXT NOT NULL,
    monthly_charge_band TEXT NOT NULL,
    service_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS model_predictions (
    customer_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    data_split TEXT NOT NULL,
    churn_probability REAL NOT NULL
        CHECK (churn_probability BETWEEN 0 AND 1),
    risk_segment TEXT NOT NULL,
    scored_at TEXT NOT NULL,
    PRIMARY KEY (customer_id, model_version, data_split),
    FOREIGN KEY (customer_id)
        REFERENCES customers_clean(customer_id)
);

CREATE TABLE IF NOT EXISTS model_evaluation (
    model_version TEXT NOT NULL,
    data_split TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    evaluated_at TEXT NOT NULL,
    PRIMARY KEY (model_version, data_split, metric_name)
);

CREATE TABLE IF NOT EXISTS data_quality_audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    check_name TEXT NOT NULL,
    fail_count INTEGER NOT NULL,
    status TEXT NOT NULL,
    notes TEXT
);
