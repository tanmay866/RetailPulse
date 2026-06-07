"""
Initialise the RetailPulse PostgreSQL schema and load processed CSV data.

Usage:
    python scripts/db_init.py                   # create tables + load all CSVs
    python scripts/db_init.py --schema-only     # create tables only
    python scripts/db_init.py --flush-redis     # invalidate Redis cache after load
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_PROCESSED = ROOT / "data" / "processed"
DATABASE_URL   = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres123@localhost:15432/postgres",
)

# ── DDL ───────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
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

CREATE INDEX IF NOT EXISTS idx_email_history_sent_at ON email_history (sent_at DESC);

CREATE TABLE IF NOT EXISTS audit_log (
    id          SERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    username    TEXT NOT NULL,
    action      TEXT NOT NULL,
    resource    TEXT NOT NULL,
    details     TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_ts       ON audit_log (ts DESC);
CREATE INDEX IF NOT EXISTS idx_retail_cust    ON retail_clean (customer_id);
CREATE INDEX IF NOT EXISTS idx_retail_date    ON retail_clean (invoice_date);
CREATE INDEX IF NOT EXISTS idx_churn_cust     ON churn_predictions (customer_id);
"""

# ── CSV → table mapping ───────────────────────────────────────────────────────

def _norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names, replace spaces with underscores."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _load_retail_clean(conn) -> None:
    df = pd.read_csv(DATA_PROCESSED / "retail_clean.csv", parse_dates=["InvoiceDate"])
    df = _norm_columns(df)
    # _norm_columns: "StockCode"→"stockcode", "InvoiceDate"→"invoicedate"
    # DDL uses snake_case: stock_code, invoice_date
    df = df.rename(columns={"stockcode": "stock_code", "invoicedate": "invoice_date"})
    _upsert(conn, "retail_clean", df, conflict_col=None)


def _load_daily_revenue_rolling(conn) -> None:
    df = pd.read_csv(DATA_PROCESSED / "daily_revenue_rolling.csv", parse_dates=["Date"])
    df = _norm_columns(df)
    _upsert(conn, "daily_revenue_rolling", df, conflict_col="date")


def _load_rfm_scores(conn) -> None:
    df = pd.read_csv(DATA_PROCESSED / "rfm_scores.csv")
    df = _norm_columns(df)
    _upsert(conn, "rfm_scores", df, conflict_col="customer_id")


def _load_customer_segments(conn) -> None:
    df = pd.read_csv(DATA_PROCESSED / "customer_segments.csv")
    df = _norm_columns(df)
    _upsert(conn, "customer_segments", df, conflict_col="customer_id")


def _load_churn_predictions(conn) -> None:
    df = pd.read_csv(DATA_PROCESSED / "churn_predictions.csv")
    df = _norm_columns(df)
    df = df.rename(columns={"customer_id": "customer_id"})
    _upsert(conn, "churn_predictions", df, conflict_col="customer_id")


def _load_daily_revenue_ts(conn) -> None:
    df = pd.read_csv(DATA_PROCESSED / "daily_revenue_ts.csv")
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = _norm_columns(df)
    # convert boolean-like columns
    for col in ("is_weekend", "is_month_end"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})
    _upsert(conn, "daily_revenue_ts", df, conflict_col="date")


def _load_inventory_recommendations(conn) -> None:
    df = pd.read_csv(DATA_PROCESSED / "inventory_recommendations.csv")
    df = _norm_columns(df)
    _upsert(conn, "inventory_recommendations", df, conflict_col=None)


def _load_clv_predictions(conn) -> None:
    df = pd.read_csv(DATA_PROCESSED / "clv_predictions.csv")
    df = _norm_columns(df)
    _upsert(conn, "clv_predictions", df, conflict_col="customer_id")


# ── Bulk upsert helper ────────────────────────────────────────────────────────

def _upsert(conn, table: str, df: pd.DataFrame, conflict_col: str | None) -> None:
    cols    = list(df.columns)
    rows    = [tuple(r) for r in df.itertuples(index=False, name=None)]
    col_sql = ", ".join(cols)
    tmpl    = "(" + ", ".join(["%s"] * len(cols)) + ")"

    if conflict_col:
        update_cols = [c for c in cols if c != conflict_col]
        update_sql  = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        sql = (
            f"INSERT INTO {table} ({col_sql}) VALUES %s "
            f"ON CONFLICT ({conflict_col}) DO UPDATE SET {update_sql}"
        )
    else:
        sql = f"INSERT INTO {table} ({col_sql}) VALUES %s"

    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {table}" if conflict_col is None else "SELECT 1")
        execute_values(cur, sql, rows, template=tmpl, page_size=500)
    print(f"  [OK]{table}: {len(rows):,} rows")


# ── Main ──────────────────────────────────────────────────────────────────────

def _seed_users(conn) -> None:
    """Insert demo users; skip rows that already exist (ON CONFLICT DO NOTHING)."""
    import hashlib

    def _sha(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    users = [
        ("retailpulse.admin", _sha("Admin@2026"),   "admin"),
        ("retail.analyst",    _sha("Analyst@2026"), "analyst"),
        ("guest",             _sha("Guest@2026"),   "viewer"),
        ("retail.scientist",  _sha("Science@2026"), "data_scientist"),
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO users (username, password_hash, role) VALUES %s "
            "ON CONFLICT (username) DO NOTHING",
            users,
        )
    print(f"  [OK] users: {len(users)} demo accounts seeded")


_LOADERS = [
    ("retail_clean",              _load_retail_clean),
    ("daily_revenue_rolling",     _load_daily_revenue_rolling),
    ("rfm_scores",                _load_rfm_scores),
    ("customer_segments",         _load_customer_segments),
    ("churn_predictions",         _load_churn_predictions),
    ("daily_revenue_ts",          _load_daily_revenue_ts),
    ("inventory_recommendations", _load_inventory_recommendations),
    ("clv_predictions",           _load_clv_predictions),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise RetailPulse database")
    parser.add_argument("--schema-only", action="store_true", help="Create tables only, skip data load")
    parser.add_argument("--flush-redis", action="store_true", help="Invalidate Redis cache after load")
    args = parser.parse_args()

    print(f"Connecting to: {DATABASE_URL.split('@')[-1]}")
    try:
        conn = psycopg2.connect(DATABASE_URL)
    except Exception as exc:
        print(f"ERROR: cannot connect to PostgreSQL — {exc}")
        sys.exit(1)

    # Create schema
    print("\nCreating schema …")
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()
    print("  [OK]Schema ready")

    # Always seed users (safe — ON CONFLICT DO NOTHING)
    print("\nSeeding users …")
    try:
        _seed_users(conn)
        conn.commit()
    except Exception as exc:
        conn.rollback()
        print(f"  [FAIL] users: {exc}")

    if not args.schema_only:
        print("\nLoading CSV data …")
        for name, loader in _LOADERS:
            try:
                loader(conn)
                conn.commit()
            except Exception as exc:
                conn.rollback()
                print(f"  [FAIL]{name}: {exc}")

    conn.close()
    print("\nDatabase initialisation complete.")

    if args.flush_redis:
        print("Flushing Redis cache …")
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis
            r = redis.Redis.from_url(redis_url)
            keys = r.keys("rp:*")
            if keys:
                r.delete(*keys)
            print(f"  [OK]Removed {len(keys)} cached keys")
        except Exception as exc:
            print(f"  [FAIL]Redis flush failed: {exc}")


if __name__ == "__main__":
    main()
