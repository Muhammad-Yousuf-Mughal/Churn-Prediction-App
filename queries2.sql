DROP TABLE IF EXISTS billing_history;
DROP TABLE IF EXISTS customer_services;
DROP TABLE IF EXISTS customer_accounts;
DROP TABLE IF EXISTS customers;

-- 1. Core Demographics
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(10) NOT NULL,
    senior_citizen INT CHECK (senior_citizen IN (0, 1)),
    partner VARCHAR(5) CHECK (partner IN ('Yes', 'No')),
    dependents VARCHAR(5) CHECK (dependents IN ('Yes', 'No'))
);

-- 2. Account & Contract Information
CREATE TABLE customer_accounts (
    account_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    tenure_months INT NOT NULL CHECK (tenure_months >= 0),
    contract_type VARCHAR(30) NOT NULL,
    paperless_billing VARCHAR(5) CHECK (paperless_billing IN ('Yes', 'No')),
    payment_method VARCHAR(50) NOT NULL,
    churn_status VARCHAR(5) NOT NULL CHECK (churn_status IN ('Yes', 'No'))
);

-- 3. Subscribed Telco Services
CREATE TABLE customer_services (
    service_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    phone_service VARCHAR(20),
    multiple_lines VARCHAR(20),
    internet_service VARCHAR(20),
    online_security VARCHAR(20),
    online_backup VARCHAR(20),
    device_protection VARCHAR(20),
    tech_support VARCHAR(20),
    streaming_tv VARCHAR(20),
    streaming_movies VARCHAR(20)
);

-- 4. Financial & Billing History
CREATE TABLE billing_history (
    billing_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    monthly_charges DECIMAL(10, 2) NOT NULL,
    total_charges DECIMAL(10, 2)
);

-- Indexes for Speed Optimization
CREATE INDEX idx_customers_id ON customers(customer_id);
CREATE INDEX idx_accounts_customer ON customer_accounts(customer_id);
CREATE INDEX idx_billing_customer ON billing_history(customer_id);