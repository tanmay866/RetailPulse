"""Shared pytest configuration and fixtures for the RetailPulse test suite."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so all test modules can do
# `from src.xxx import ...` without each file patching sys.path manually.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW       = ROOT / "data" / "raw"


# ── Custom mark registration ──────────────────────────────────────────────────
# Prevents PytestUnknownMarkWarning in CI when tests use @needs_data.

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "needs_data: skip test when required CSV data files are not present",
    )


# ── Shared path fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repository root."""
    return ROOT


@pytest.fixture(scope="session")
def data_processed() -> Path:
    """Path to data/processed/ — pre-computed CSVs used by most tests."""
    return DATA_PROCESSED


@pytest.fixture(scope="session")
def data_raw() -> Path:
    """Path to data/raw/ — source datasets (may not exist in CI)."""
    return DATA_RAW


# ── Output sandbox fixture ─────────────────────────────────────────────────────

@pytest.fixture()
def tmp_output(tmp_path: Path) -> Path:
    """Isolated temp directory pre-created with figures/ and processed/ sub-dirs.

    Use this in any test that writes PNGs, CSVs, or model files so the output
    never lands in the real reports/ or data/ directories during a test run.
    """
    (tmp_path / "figures").mkdir()
    (tmp_path / "processed").mkdir()
    (tmp_path / "models").mkdir()
    return tmp_path
