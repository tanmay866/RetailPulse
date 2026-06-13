"""Locust load test for the RetailPulse Streamlit app.

Streamlit serves its UI over a websocket after the initial page load, so HTTP
load testing targets the endpoints that report capacity and health rather than
simulating full user sessions: the Streamlit health probe, the initial page
render, and the Prometheus metrics exposition endpoint.

Run against a locally running stack (docker compose up):

    # headless, 50 users, spawn 5/s, 1 minute, write an HTML report
    locust -f tests/load/locustfile.py --host http://localhost:8501 \
        --headless -u 50 -r 5 -t 1m --html reports/load_test.html

    # interactive web UI at http://localhost:8089
    locust -f tests/load/locustfile.py --host http://localhost:8501

The metrics endpoint runs on a separate port (8000); override it with
RETAILPULSE_METRICS_URL if it is not http://localhost:8000.
"""
import os

from locust import HttpUser, between, task

METRICS_URL = os.getenv("RETAILPULSE_METRICS_URL", "http://localhost:8000/metrics")


class StreamlitUser(HttpUser):
    """Simulates a viewer loading the dashboard and the monitoring scrape."""

    # Think time between requests — mimics a human navigating, not a tight loop.
    wait_time = between(1, 3)

    @task(5)
    def health(self):
        """Streamlit's internal health probe — the cheapest liveness check."""
        self.client.get("/_stcore/health", name="streamlit health")

    @task(3)
    def home(self):
        """Initial HTML render of the dashboard shell."""
        self.client.get("/", name="dashboard home")

    @task(1)
    def metrics(self):
        """Prometheus metrics exposition — same request the scraper makes."""
        # Absolute URL because metrics are served on a different port than --host.
        self.client.get(METRICS_URL, name="prometheus metrics")
