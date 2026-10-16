
-- =====================================================
-- CUSTOMER CHURN ANALYTICS
-- Database: customer_churn.db
-- All churn metrics describe the historical snapshot.
-- Monetary values are in unspecified currency units.
-- =====================================================

-- QUERY 1: Overall customer churn
SELECT
    COUNT(*) AS total_customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
FROM customers_clean;


-- QUERY 2: Churn by contract type
SELECT
    Contract,
    COUNT(*) AS total_customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
FROM customers_clean
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- QUERY 3: Churn by tenure band
SELECT
    tenure_band,
    COUNT(*) AS total_customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
FROM customers_clean
GROUP BY tenure_band
ORDER BY
    CASE tenure_band
        WHEN '0-12' THEN 1
        WHEN '13-24' THEN 2
        WHEN '25-48' THEN 3
        WHEN '49-72' THEN 4
    END;


-- QUERY 4: Churn by payment method
SELECT
    PaymentMethod,
    COUNT(*) AS total_customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
FROM customers_clean
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;


-- QUERY 5: Churn by internet service
SELECT
    InternetService,
    COUNT(*) AS total_customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
FROM customers_clean
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;


-- QUERY 6: Churn by technical support
SELECT
    TechSupport,
    COUNT(*) AS total_customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
FROM customers_clean
GROUP BY TechSupport
ORDER BY churn_rate_pct DESC;


-- QUERY 7: Average monthly charges by churn status
SELECT
    Churn,
    COUNT(*) AS customers,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges_cu,
    ROUND(SUM(MonthlyCharges), 2) AS group_monthly_charges_cu
FROM customers_clean
GROUP BY Churn;


-- QUERY 8: Monthly charge exposure of churned customers
SELECT
    COUNT(*) AS churned_customers,
    ROUND(SUM(MonthlyCharges), 2)
        AS observed_churned_group_monthly_charges_cu
FROM customers_clean
WHERE churn_flag = 1;


-- QUERY 9: Customer service count segmentation
SELECT
    service_count,
    COUNT(*) AS total_customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
FROM customers_clean
GROUP BY service_count
ORDER BY service_count;


-- QUERY 10: Database quality verification
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT customer_id) AS unique_customers,
    SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END)
        AS missing_customer_ids,
    SUM(CASE WHEN MonthlyCharges < 0 THEN 1 ELSE 0 END)
        AS negative_monthly_charges,
    SUM(CASE WHEN TotalCharges < 0 THEN 1 ELSE 0 END)
        AS negative_total_charges
FROM customers_clean;
