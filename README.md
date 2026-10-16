# Customer Churn Prediction & Revenue Retention Analytics

An end-to-end data science and business analytics project that analyzes customer churn, predicts customer churn risk using machine learning, and presents actionable retention insights through interactive Power BI dashboards.

## Project Objectives

- Analyze customer characteristics and historical churn patterns.
- Develop a machine learning model to estimate customer churn probability.
- Identify customers with elevated churn risk.
- Explore monthly charge exposure associated with predicted churn.
- Present customer analytics and retention priorities through Power BI.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analytics | Pandas, NumPy |
| Machine Learning | Scikit-learn, Logistic Regression |
| Database | SQLite, SQL |
| Data Visualization | Power BI, Matplotlib, Seaborn |
| Model Serialization | Joblib |
| Development | VS Code, Git, GitHub |

## Dataset

**IBM Telco Customer Churn sample dataset**

- Total customer records: 7,043
- Machine learning training records: 4,225
- Validation records: 1,409
- Test records: 1,409
- Input features: 19

The dataset includes demographic, subscription, service usage, contract, tenure, and billing information.

## Project Workflow

1. Dataset acquisition and validation
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. SQLite database integration and analytical SQL
5. Feature engineering and machine learning model development
6. Model evaluation and churn risk scoring
7. Power BI data exports and dashboard development
8. Customer retention prioritization and business interpretation

## Machine Learning Model

The project uses a Logistic Regression classification model to estimate customer churn probabilities.

**Model configuration**

- Model version: `logistic_regression_v1`
- Selected classification threshold: 0.40
- Threshold selection: Validation F1-score
- Random state: 42

### Test-Set Performance

| Metric | Result |
|---|---:|
| Accuracy | 77.29% |
| Precision | 56.11% |
| Recall | 66.31% |
| F1-score | 60.78% |
| ROC-AUC | 0.8326 |
| PR-AUC | 0.6224 |
| Brier Score | 0.1414 |

The model correctly identified 248 customers who churned in the held-out test set, with 194 false positives and 126 false negatives.

These results describe evaluation on a random holdout of the sample dataset, not guaranteed real-world forecasting performance.

## Power BI Dashboards

### Dashboard 1 — Customer Overview

Analyzes historical customer churn through KPI cards, customer segmentation, contract and tenure analysis, and interactive filters.

**Key results**

- Total customers: 7,043
- Observed churned customers: 1,869
- Observed churn rate: 26.54%

### Dashboard 2 — Predictive Churn & Revenue Risk

Presents the trained model's churn predictions, probability distribution, risk segments, and estimated monthly charge exposure.

**Key results**

- Scored customers: 1,409
- Flagged customers: 442
- Flagged customer rate: 31.37%
- Expected monthly charge exposure: 28,280.18 CU
- Model ROC-AUC: 0.8326

### Dashboard 3 — Customer Retention & Action Insights

Supports retention planning by highlighting customers flagged for churn risk and examining monthly charge exposure across contracts and payment methods.

Includes a high-priority customer table sorted by churn probability.

## Customer Retention Recommendations

The analysis can support the following proposed business actions:

- Prioritize model-flagged customers for further review and targeted engagement.
- Review month-to-month contract customers for suitable loyalty or renewal offers.
- Investigate churn risk and monthly charge exposure across payment methods.
- Use customer-level risk probability and charge exposure to help allocate retention resources.
- Evaluate retention campaigns using controlled experiments before claiming measurable financial benefits.

These actions are recommendations, not demonstrated campaign outcomes.

## Project Structure

- `data/` — Raw, processed, and exported datasets
- `database/` — SQLite database
- `models/` — Trained model pipeline and metadata
- `notebooks/` — Notebook workspace
- `powerbi/` — Power BI report
- `reports/` — Model evaluation reports and EDA figures
- `sql/` — Database schema, analytical queries, and reporting views
- `src/churn_analytics/` — Python pipeline modules
- `tests/` — Testing directory

## Installation

Clone the repository and navigate to the project folder.

Create and activate a compatible Python virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The project uses Python scripts under `src/churn_analytics/`. Review the relevant module and its command-line arguments before executing pipeline stages. The existing trained model and export files are included in the local project structure.

## Project Limitations

- The IBM telecom dataset is a sample dataset rather than a live customer system.
- Random train-validation-test splitting does not establish future forecasting accuracy.
- Predicted probabilities are not guaranteed to be perfectly calibrated.
- Monthly charge exposure is an analytical estimate, not confirmed lost revenue.
- No actual retention campaign outcomes or savings have been observed.

## Author

Data Science and Data Analytics Portfolio Project

## Project Status

Python analysis, SQLite database, model training and evaluation, and three Power BI report pages developed. Final repository publication and reproducibility checks are in progress.