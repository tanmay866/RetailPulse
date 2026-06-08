"""Prometheus metrics for RetailPulse.

Call start_metrics_server() once at application startup (guarded by a
threading.Event so repeated Streamlit re-runs are safe).
"""
import threading

import prometheus_client as prom

# ── Metric registry ────────────────────────────────────────────────────────────

PAGE_VIEWS = prom.Counter(
    "retailpulse_page_views_total",
    "Number of Streamlit page renders",
    ["page"],
)

PAGE_LOAD_SECONDS = prom.Histogram(
    "retailpulse_page_load_seconds",
    "Page render duration in seconds",
    ["page"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

CHURN_HIGH_RISK = prom.Gauge(
    "retailpulse_churn_high_risk_customers",
    "Number of customers currently predicted as high-risk churn",
)

FORECAST_REQUESTS = prom.Counter(
    "retailpulse_forecast_requests_total",
    "Number of demand forecast page loads",
)

ALERTS_FIRED = prom.Counter(
    "retailpulse_alerts_fired_total",
    "Number of real-time alerts triggered",
    ["severity"],
)

# ── Server lifecycle ───────────────────────────────────────────────────────────

_started = threading.Event()


def start_metrics_server(port: int = 8000) -> None:
    """Start Prometheus HTTP exposition server on *port* — idempotent."""
    if not _started.is_set():
        prom.start_http_server(port)
        _started.set()
