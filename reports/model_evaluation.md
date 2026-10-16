# Customer Churn Model Evaluation

Model: logistic_regression_v1

Classification threshold: 0.40

## Held-out test metrics

| Metric | Value |
|---|---:|
| accuracy | 0.7729 |
| precision | 0.5611 |
| recall | 0.6631 |
| f1 | 0.6078 |
| roc_auc | 0.8326 |
| pr_auc | 0.6224 |
| brier_score | 0.1414 |

## Confusion Matrix

- True negatives: 841
- False positives: 194
- False negatives: 126
- True positives: 248

## Classification Report

```text
              precision    recall  f1-score   support

    No Churn       0.87      0.81      0.84      1035
       Churn       0.56      0.66      0.61       374

    accuracy                           0.77      1409
   macro avg       0.72      0.74      0.72      1409
weighted avg       0.79      0.77      0.78      1409

```

## Limitations

This is a fictional historical customer snapshot.
A stratified random holdout does not prove future
churn forecasting performance or business uplift.
Retention outcomes are not observed in this dataset.
