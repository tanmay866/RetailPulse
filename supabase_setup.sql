-- RetailPulse — Supabase schema + seed
-- Paste this entire file into Supabase → SQL Editor → Run

-- ── Tables ────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS retail_clean (
    invoice         TEXT,
    stock_code      TEXT,
    description     TEXT,
    quantity        INTEGER,
    invoice_date    TIMESTAMPTZ,
    price           NUMERIC(10,4),
    customer_id     BIGINT,
    country         TEXT,
    revenue         NUMERIC(12,4)
);

CREATE TABLE IF NOT EXISTS daily_revenue_rolling (
    date                DATE PRIMARY KEY,
    revenue             NUMERIC(14,4),
    rolling_7d_mean     NUMERIC(14,4),
    rolling_7d_std      NUMERIC(14,4),
    rolling_30d_mean    NUMERIC(14,4),
    rolling_30d_std     NUMERIC(14,4)
);

CREATE TABLE IF NOT EXISTS rfm_scores (
    customer_id     BIGINT PRIMARY KEY,
    recency         INTEGER,
    frequency       INTEGER,
    monetary        NUMERIC(14,4),
    r_score         INTEGER,
    f_score         INTEGER,
    m_score         INTEGER,
    rfm_score       INTEGER,
    rfm_total       INTEGER,
    segment         TEXT
);

CREATE TABLE IF NOT EXISTS customer_segments (
    customer_id     BIGINT PRIMARY KEY,
    recency         INTEGER,
    frequency       INTEGER,
    monetary        NUMERIC(14,4),
    kmeans_cluster  INTEGER,
    dbscan_cluster  INTEGER,
    business_label  TEXT
);

CREATE TABLE IF NOT EXISTS churn_predictions (
    customer_id         BIGINT PRIMARY KEY,
    churn_probability   NUMERIC(8,6),
    predicted_churn     INTEGER,
    actual_churn        INTEGER
);

CREATE TABLE IF NOT EXISTS daily_revenue_ts (
    date                DATE PRIMARY KEY,
    revenue             NUMERIC(14,4),
    quantity            NUMERIC(14,4),
    log_revenue         NUMERIC(14,8),
    log_quantity        NUMERIC(14,8),
    day_of_week         INTEGER,
    month               INTEGER,
    quarter             INTEGER,
    is_weekend          BOOLEAN,
    is_month_end        BOOLEAN,
    lag_1               NUMERIC(14,4),
    lag_7               NUMERIC(14,4),
    lag_30              NUMERIC(14,4),
    rolling_7_mean      NUMERIC(14,4),
    rolling_30_mean     NUMERIC(14,4),
    rolling_7_std       NUMERIC(14,4)
);

CREATE TABLE IF NOT EXISTS inventory_recommendations (
    store_id            TEXT,
    product_id          TEXT,
    category            TEXT,
    region              TEXT,
    current_inventory   NUMERIC(14,4),
    mean_daily_demand   NUMERIC(14,6),
    safety_stock        NUMERIC(14,4),
    rop                 NUMERIC(14,4),
    eoq                 NUMERIC(14,4),
    days_of_stock       NUMERIC(14,4),
    status              TEXT,
    units_to_order      NUMERIC(14,4)
);

CREATE TABLE IF NOT EXISTS clv_predictions (
    customer_id             BIGINT PRIMARY KEY,
    frequency               NUMERIC(10,4),
    recency                 NUMERIC(10,4),
    t                       NUMERIC(10,4),
    monetary_value          NUMERIC(14,6),
    prob_alive              NUMERIC(8,6),
    pred_purchases_90d      NUMERIC(10,6),
    pred_purchases_180d     NUMERIC(10,6),
    pred_purchases_365d     NUMERIC(10,6),
    clv_12m                 NUMERIC(14,6),
    clv_segment             TEXT
);

CREATE TABLE IF NOT EXISTS users (
    username        TEXT PRIMARY KEY,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS email_history (
    id              SERIAL PRIMARY KEY,
    sent_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sent_by         TEXT NOT NULL,
    recipient       TEXT NOT NULL,
    subject         TEXT NOT NULL,
    churn_high_risk INTEGER,
    stockout_skus   INTEGER,
    revenue_change  TEXT,
    status          TEXT NOT NULL,
    error_msg       TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          SERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    username    TEXT NOT NULL,
    action      TEXT NOT NULL,
    resource    TEXT NOT NULL,
    details     TEXT
);

-- ── Indexes ───────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_retail_cust    ON retail_clean (customer_id);
CREATE INDEX IF NOT EXISTS idx_retail_date    ON retail_clean (invoice_date);
CREATE INDEX IF NOT EXISTS idx_churn_cust     ON churn_predictions (customer_id);
CREATE INDEX IF NOT EXISTS idx_email_sent_at  ON email_history (sent_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_ts       ON audit_log (ts DESC);

-- ── Demo users ────────────────────────────────────────────────────────────────
-- Passwords: Admin@2026 / Analyst@2026 / Guest@2026 / Science@2026
-- Hashes are SHA-256 of the plaintext passwords.

INSERT INTO users (username, password_hash, role) VALUES
    ('retailpulse.admin',
     'a36aef5a11c4073fbe60314fc9df530a9d5f986533594d1f5190742ff9e0e408',
     'admin'),
    ('retail.analyst',
     '345982ba4e71cf6789b88de67e9b5f769ff011065010a273bae02fee9ccead97',
     'analyst'),
    ('guest',
     '87718cfb6bea18c8e1c85e10abddd16930e507e63ca8c53f09fb1602a78a3765',
     'viewer'),
    ('retail.scientist',
     '4640f615c539c2b2d857a8c122d715069946604e39d85a4879a266d25edf1258',
     'data_scientist')
ON CONFLICT (username) DO NOTHING;
