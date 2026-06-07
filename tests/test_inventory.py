import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inventory_optimizer import (
    STATUS_OPTIMAL,
    STATUS_OVERSTOCK,
    STATUS_STOCKOUT,
    InventoryOptimizer,
)

INVENTORY_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "retail_store_inventory.csv"
needs_data = pytest.mark.skipif(not INVENTORY_CSV.exists(), reason="retail_store_inventory.csv not available")


# ---------------------------------------------------------------------------
# Helpers — build a minimal synthetic DataFrame for formula tests
# ---------------------------------------------------------------------------

def _make_df(units_sold, demand_forecast, inventory_level, price=10.0, n_dates=30):
    """Return a minimal inventory DataFrame with one SKU and n_dates rows."""
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="D")
    return pd.DataFrame({
        "Date":            dates,
        "Store ID":        "S001",
        "Product ID":      "P0001",
        "Category":        "Test",
        "Region":          "North",
        "Units Sold":      units_sold,
        "Demand Forecast": demand_forecast,
        "Inventory Level": inventory_level,
        "Units Ordered":   50,
        "Price":           price,
        "Discount":        0,
    })


def _optimizer_from_df(df, **kwargs) -> InventoryOptimizer:
    opt = InventoryOptimizer(**kwargs)
    opt._df = df
    return opt


# ---------------------------------------------------------------------------
# 1. Safety stock formula
# ---------------------------------------------------------------------------

def test_safety_stock_formula():
    from scipy.stats import norm
    df = _make_df(units_sold=20, demand_forecast=20, inventory_level=500)
    opt = _optimizer_from_df(df, service_level=0.95, lead_time_days=7)
    opt.compute_demand_stats()
    opt.compute_safety_stock()

    assert opt._stats is not None
    z = norm.ppf(0.95)
    std = df["Units Sold"].std()
    expected = round(z * std * np.sqrt(7), 2)
    actual   = float(opt._stats["safety_stock"].iloc[0])
    assert abs(actual - expected) < 0.01, f"Expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# 2. Reorder point formula
# ---------------------------------------------------------------------------

def test_reorder_point_formula():
    df = _make_df(units_sold=20, demand_forecast=25, inventory_level=500)
    opt = _optimizer_from_df(df, service_level=0.95, lead_time_days=7)
    opt.compute_demand_stats()
    opt.compute_safety_stock()
    opt.compute_reorder_point()

    assert opt._stats is not None
    s = opt._stats.iloc[0]
    expected = round(s["mean_daily_demand"] * 7 + s["safety_stock"], 2)
    actual   = round(float(s["rop"]), 2)
    assert abs(actual - expected) < 0.01


# ---------------------------------------------------------------------------
# 3. EOQ formula
# ---------------------------------------------------------------------------

def test_eoq_formula():
    df = _make_df(units_sold=20, demand_forecast=20, inventory_level=500, price=10.0)
    opt = _optimizer_from_df(df, ordering_cost=50.0, holding_cost_rate=0.25)
    opt.compute_demand_stats()
    opt.compute_safety_stock()
    opt.compute_reorder_point()
    opt.compute_eoq()

    assert opt._stats is not None
    mean_demand   = float(opt._stats["mean_daily_demand"].iloc[0])
    annual_demand = mean_demand * 365
    holding_cost  = 0.25 * 10.0
    expected      = round(np.sqrt(2 * annual_demand * 50.0 / holding_cost), 2)
    actual        = round(float(opt._stats["eoq"].iloc[0]), 2)
    assert abs(actual - expected) < 0.01


# ---------------------------------------------------------------------------
# 4. Status — STOCKOUT_RISK when inventory < ROP
# ---------------------------------------------------------------------------

def test_status_stockout():
    df = _make_df(units_sold=50, demand_forecast=50, inventory_level=10)
    opt = _optimizer_from_df(df, lead_time_days=7, service_level=0.95)
    opt.compute_demand_stats()
    opt.compute_safety_stock()
    opt.compute_reorder_point()
    opt.compute_eoq()
    opt.evaluate_current_stock()
    opt.generate_recommendations()

    assert opt._recommendations is not None
    status = opt._recommendations["status"].iloc[0]
    assert status == STATUS_STOCKOUT, f"Expected STOCKOUT_RISK, got {status}"


# ---------------------------------------------------------------------------
# 5. Status — OVERSTOCK when inventory > ROP + EOQ
# ---------------------------------------------------------------------------

def test_status_overstock():
    df = _make_df(units_sold=5, demand_forecast=5, inventory_level=100_000)
    opt = _optimizer_from_df(df, lead_time_days=1, service_level=0.90)
    opt.compute_demand_stats()
    opt.compute_safety_stock()
    opt.compute_reorder_point()
    opt.compute_eoq()
    opt.evaluate_current_stock()
    opt.generate_recommendations()

    assert opt._recommendations is not None
    status = opt._recommendations["status"].iloc[0]
    assert status == STATUS_OVERSTOCK, f"Expected OVERSTOCK, got {status}"


# ---------------------------------------------------------------------------
# 6. Zero demand — days_of_stock should be inf, not raise
# ---------------------------------------------------------------------------

def test_zero_demand_no_error():
    df = _make_df(units_sold=0, demand_forecast=0, inventory_level=100)
    opt = _optimizer_from_df(df)
    opt.compute_demand_stats()
    opt.compute_safety_stock()
    opt.compute_reorder_point()
    opt.compute_eoq()
    opt.evaluate_current_stock()
    opt.generate_recommendations()

    assert opt._recommendations is not None
    days = opt._recommendations["days_of_stock"].iloc[0]
    assert days == np.inf


# ---------------------------------------------------------------------------
# 7. Negative Demand Forecast clipped to 0
# ---------------------------------------------------------------------------

def test_negative_forecast_clipped():
    df = _make_df(units_sold=10, demand_forecast=-5, inventory_level=200)
    opt = _optimizer_from_df(df)
    opt.compute_demand_stats()

    assert opt._stats is not None
    mean = float(opt._stats["mean_daily_demand"].iloc[0])
    assert mean == 0.0, f"Negative forecast should clip to 0, got {mean}"


# ---------------------------------------------------------------------------
# 8. generate_recommendations() output shape and required columns
# ---------------------------------------------------------------------------

def test_recommendations_columns():
    df = _make_df(units_sold=20, demand_forecast=20, inventory_level=300)
    opt = _optimizer_from_df(df)
    opt.compute_demand_stats()
    opt.compute_safety_stock()
    opt.compute_reorder_point()
    opt.compute_eoq()
    opt.evaluate_current_stock()
    rec = opt.generate_recommendations()

    required = {
        "store_id", "product_id", "category", "region",
        "current_inventory", "mean_daily_demand", "safety_stock",
        "rop", "eoq", "days_of_stock", "status", "units_to_order",
    }
    assert required.issubset(set(rec.columns)), f"Missing columns: {required - set(rec.columns)}"
    assert len(rec) == 1


# ---------------------------------------------------------------------------
# 9. Full pipeline on real data (skipped if file absent)
# ---------------------------------------------------------------------------

@needs_data
def test_full_pipeline_runs():
    opt = InventoryOptimizer(service_level=0.95, lead_time_days=7)
    opt.load()
    opt.compute_demand_stats()
    opt.compute_safety_stock()
    opt.compute_reorder_point()
    opt.compute_eoq()
    opt.evaluate_current_stock()
    rec = opt.generate_recommendations()

    assert len(rec) > 0
    assert set(rec["status"].unique()).issubset({STATUS_STOCKOUT, STATUS_OVERSTOCK, STATUS_OPTIMAL})
    assert (rec["days_of_stock"] >= 0).all()
    assert (rec["units_to_order"] >= 0).all()
