-- Feature Extraction Query for PostgreSQL
WITH service_counts AS (
    SELECT 
        customer_id,
        (CASE WHEN phone_service = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN multiple_lines = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN internet_service IN ('DSL', 'Fiber optic') THEN 1 ELSE 0 END +
         CASE WHEN online_security = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN online_backup = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN device_protection = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN tech_support = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN streaming_tv = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN streaming_movies = 'Yes' THEN 1 ELSE 0 END) AS active_service_count
    FROM customer_services
)

SELECT 
    c.customer_id,
    c.gender,
    c.senior_citizen,
    c.partner,
    c.dependents,
    
    a.tenure_months,
    a.contract_type,
    a.paperless_billing,
    a.payment_method,
    
    s.phone_service,
    s.multiple_lines,
    s.internet_service,
    s.online_security,
    s.online_backup,
    s.device_protection,
    s.tech_support,
    s.streaming_tv,
    s.streaming_movies,
    sc.active_service_count,
    
    b.monthly_charges,
    COALESCE(b.total_charges, b.monthly_charges * a.tenure_months) AS total_charges,
    ROUND(b.monthly_charges / GREATEST(a.tenure_months, 1), 2) AS charge_per_tenure,
    
    -- Binary Target Variable
    CASE WHEN a.churn_status = 'Yes' THEN 1 ELSE 0 END AS churn

FROM customers c
INNER JOIN customer_accounts a ON c.customer_id = a.customer_id
LEFT JOIN customer_services s ON c.customer_id = s.customer_id
LEFT JOIN service_counts sc ON c.customer_id = sc.customer_id
LEFT JOIN billing_history b ON c.customer_id = b.customer_id;