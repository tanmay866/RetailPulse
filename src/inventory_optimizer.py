"""Inventory optimization using forecasted demand from retail_store_inventory.csv.

Computes safety stock, reorder point (ROP), and EOQ per (Store ID, Product ID).
Flags each SKU as STOCKOUT_RISK, OVERSTOCK, or OPTIMAL based on current inventory.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT           = Path(__file__).resolve().parents[1]
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES_DIR    = ROOT / "reports" / "figures"

INVENTORY_FILE = DATA_RAW / "retail_store_inventory.csv"

STATUS_STOCKOUT  = "STOCKOUT_RISK"
STATUS_OVERSTOCK = "OVERSTOCK"
STATUS_OPTIMAL   = "OPTIMAL"


class InventoryOptimizer:
    """Compute inventory optimization metrics per SKU.

    Parameters
    ----------
    service_level:     Target in-stock probability (e.g. 0.95).
    lead_time_days:    Days between placing and receiving an order.
    ordering_cost:     Fixed cost per purchase order (currency units).
    holding_cost_rate: Annual holding cost as a fraction of unit price.
    """

    def __init__(
        self,
        service_level: float = 0.95,
        lead_time_days: int = 1,
        ordering_cost: float = 50.0,
        holding_cost_rate: float = 0.25,
    ) -> None:
        self.service_level     = service_level
        self.lead_time_days    = lead_time_days
        self.ordering_cost     = ordering_cost
        self.holding_cost_rate = holding_cost_rate

        self._z = float(norm.ppf(service_level))

        self._df: pd.DataFrame | None = None
        self._stats: pd.DataFrame | None = None
        self._recommendations: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # 1. Data loading
    # ------------------------------------------------------------------

    def load(self, path: Path = INVENTORY_FILE) -> "InventoryOptimizer":
        df = pd.read_csv(path, parse_dates=["Date"])
        df.columns = df.columns.str.strip()
        self._df = df
        return self

    # ------------------------------------------------------------------
    # 2. Demand statistics — use Units Sold for variability,
    #    Demand Forecast (clipped ≥ 0) for forward-looking mean.
    # ------------------------------------------------------------------

    def compute_demand_stats(self) -> pd.DataFrame:
        df = self._df
        grp = df.groupby(["Store ID", "Product ID", "Category", "Region"])

        mean_demand = (
            grp["Demand Forecast"]
            .mean()
            .clip(lower=0)
            .rename("mean_daily_demand")
        )

        # Std from actual sales — captures true demand variability
        std_demand = (
            grp["Units Sold"]
            .std()
            .fillna(0)
            .clip(lower=0)
            .rename("std_daily_demand")
        )

        avg_price = grp["Price"].mean().rename("avg_price")

        stats = pd.concat([mean_demand, std_demand, avg_price], axis=1).reset_index()
        stats["cv"] = np.where(
            stats["mean_daily_demand"] > 0,
            stats["std_daily_demand"] / stats["mean_daily_demand"],
            0.0,
        )
        self._stats = stats
        return stats

    # ------------------------------------------------------------------
    # 3. Safety stock
    # ------------------------------------------------------------------

    def compute_safety_stock(self) -> pd.DataFrame:
        s = self._stats.copy()
        # Safety stock = Z × σ_demand × √(lead_time)
        s["safety_stock"] = (
            self._z * s["std_daily_demand"] * np.sqrt(self.lead_time_days)
        ).round(2)
        self._stats = s
        return s

    # ------------------------------------------------------------------
    # 4. Reorder point
    # ------------------------------------------------------------------

    def compute_reorder_point(self) -> pd.DataFrame:
        s = self._stats.copy()
        s["rop"] = (
            s["mean_daily_demand"] * self.lead_time_days + s["safety_stock"]
        ).round(2)
        self._stats = s
        return s

    # ------------------------------------------------------------------
    # 5. Economic Order Quantity
    # ------------------------------------------------------------------

    def compute_eoq(self) -> pd.DataFrame:
        s = self._stats.copy()
        annual_demand   = s["mean_daily_demand"] * 365
        holding_cost    = self.holding_cost_rate * s["avg_price"]
        holding_cost    = holding_cost.clip(lower=0.01)   # avoid div/0

        s["eoq"] = np.where(
            annual_demand > 0,
            np.sqrt(2 * annual_demand * self.ordering_cost / holding_cost),
            0.0,
        ).round(2)
        self._stats = s
        return s

    # ------------------------------------------------------------------
    # 6. Current inventory status (snapshot at global max date)
    # ------------------------------------------------------------------

    def evaluate_current_stock(self) -> pd.DataFrame:
        df = self._df
        latest_date = df["Date"].max()
        df_latest   = df[df["Date"] == latest_date].copy()

        merged = df_latest.merge(
            self._stats[["Store ID", "Product ID", "mean_daily_demand",
                          "safety_stock", "rop", "eoq"]],
            on=["Store ID", "Product ID"],
            how="left",
        )

        # Days of stock remaining — guard against zero demand
        merged["days_of_stock"] = np.where(
            merged["mean_daily_demand"] > 0,
            (merged["Inventory Level"] / merged["mean_daily_demand"]).round(1),
            np.inf,
        )

        def _classify(row: pd.Series) -> str:
            inv = row["Inventory Level"]
            if inv < row["rop"]:
                return STATUS_STOCKOUT
            if inv > row["rop"] + row["eoq"]:
                return STATUS_OVERSTOCK
            return STATUS_OPTIMAL

        merged["status"] = merged.apply(_classify, axis=1)

        # Units to order: only for STOCKOUT_RISK SKUs
        merged["units_to_order"] = np.where(
            merged["status"] == STATUS_STOCKOUT,
            (merged["rop"] + merged["eoq"] - merged["Inventory Level"]).clip(lower=0).round(0),
            0.0,
        )

        self._snapshot = merged
        return merged

    # ------------------------------------------------------------------
    # 7. Final recommendations output
    # ------------------------------------------------------------------

    def generate_recommendations(self) -> pd.DataFrame:
        snap = self._snapshot.copy()

        cols = {
            "Store ID":        "store_id",
            "Product ID":      "product_id",
            "Category":        "category",
            "Region":          "region",
            "Inventory Level": "current_inventory",
            "mean_daily_demand": "mean_daily_demand",
            "safety_stock":    "safety_stock",
            "rop":             "rop",
            "eoq":             "eoq",
            "days_of_stock":   "days_of_stock",
            "status":          "status",
            "units_to_order":  "units_to_order",
        }
        rec = snap[list(cols)].rename(columns=cols)
        self._recommendations = rec
        return rec

    # ------------------------------------------------------------------
    # 8. MLflow logging
    # ------------------------------------------------------------------

    def log_to_mlflow(self, run_name: str = "inventory_optimizer_v1") -> None:
        rec = self._recommendations
        total = len(rec)

        pct_stockout  = round((rec["status"] == STATUS_STOCKOUT).sum()  / total * 100, 2)
        pct_overstock = round((rec["status"] == STATUS_OVERSTOCK).sum() / total * 100, 2)
        pct_optimal   = round((rec["status"] == STATUS_OPTIMAL).sum()   / total * 100, 2)

        finite_days = rec["days_of_stock"].replace(np.inf, np.nan)
        avg_days    = round(float(finite_days.mean()), 2)

        total_units_to_order = int(rec["units_to_order"].sum())

        mlflow.set_experiment("inventory_optimization")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "service_level":     self.service_level,
                "lead_time_days":    self.lead_time_days,
                "ordering_cost":     self.ordering_cost,
                "holding_cost_rate": self.holding_cost_rate,
            })
            mlflow.log_metrics({
                "pct_stockout_risk":    pct_stockout,
                "pct_overstock":        pct_overstock,
                "pct_optimal":          pct_optimal,
                "avg_days_of_stock":    avg_days,
                "total_units_to_order": total_units_to_order,
            })

            out_path = DATA_PROCESSED / "inventory_recommendations.csv"
            rec.to_csv(out_path, index=False)
            mlflow.log_artifact(str(out_path))

            fig_paths = [
                FIGURES_DIR / "inventory_stockout_heatmap.png",
                FIGURES_DIR / "inventory_eoq_vs_ordered.png",
                FIGURES_DIR / "inventory_safety_stock_dist.png",
                FIGURES_DIR / "inventory_days_of_stock.png",
                FIGURES_DIR / "inventory_status_breakdown.png",
            ]
            for p in fig_paths:
                if p.exists():
                    mlflow.log_artifact(str(p))

        print(f"MLflow run logged: {run_name}")
        print(f"  STOCKOUT_RISK : {pct_stockout}%")
        print(f"  OVERSTOCK     : {pct_overstock}%")
        print(f"  OPTIMAL       : {pct_optimal}%")
        print(f"  Avg days stock: {avg_days}")
        print(f"  Units to order: {total_units_to_order:,}")

    # ------------------------------------------------------------------
    # 9. Visualizations
    # ------------------------------------------------------------------

    def plot_all(self) -> None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        self._plot_stockout_heatmap()
        self._plot_eoq_vs_ordered()
        self._plot_safety_stock_dist()
        self._plot_days_of_stock()
        self._plot_status_breakdown()

    def _plot_stockout_heatmap(self) -> None:
        rec = self._recommendations
        pivot = (
            rec[rec["status"] == STATUS_STOCKOUT]
            .groupby(["store_id", "category"])
            .size()
            .unstack(fill_value=0)
        )
        total_pivot = rec.groupby(["store_id", "category"]).size().unstack(fill_value=0)
        pct_pivot   = (pivot / total_pivot.replace(0, np.nan) * 100).fillna(0)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pct_pivot.values, aspect="auto", cmap="Reds")
        ax.set_xticks(range(len(pct_pivot.columns)))
        ax.set_yticks(range(len(pct_pivot.index)))
        ax.set_xticklabels(pct_pivot.columns, rotation=45, ha="right")
        ax.set_yticklabels(pct_pivot.index)
        plt.colorbar(im, ax=ax, label="% SKUs at Stockout Risk")
        ax.set_title("Stockout Risk Heatmap (Store × Category)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "inventory_stockout_heatmap.png", dpi=150)
        plt.close()
        print("saved inventory_stockout_heatmap.png")

    def _plot_eoq_vs_ordered(self) -> None:
        snap = self._snapshot.copy()
        snap = snap[snap["eoq"] > 0]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(snap["eoq"], snap["Units Ordered"], alpha=0.3, s=10, color="steelblue")
        max_val = max(snap["eoq"].max(), snap["Units Ordered"].max())
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="EOQ = Ordered")
        ax.set_xlabel("EOQ (optimal order qty)")
        ax.set_ylabel("Actual Units Ordered")
        ax.set_title("EOQ vs Actual Units Ordered")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "inventory_eoq_vs_ordered.png", dpi=150)
        plt.close()
        print("saved inventory_eoq_vs_ordered.png")

    def _plot_safety_stock_dist(self) -> None:
        stats = self._stats.copy()
        categories = stats["Category"].unique()

        fig, ax = plt.subplots(figsize=(10, 5))
        for cat in sorted(categories):
            subset = stats[stats["Category"] == cat]["safety_stock"]
            ax.hist(subset, bins=30, alpha=0.5, label=cat)
        ax.set_xlabel("Safety Stock (units)")
        ax.set_ylabel("SKU count")
        ax.set_title("Safety Stock Distribution by Category")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "inventory_safety_stock_dist.png", dpi=150)
        plt.close()
        print("saved inventory_safety_stock_dist.png")

    def _plot_days_of_stock(self) -> None:
        rec = self._recommendations.copy()
        rec["days_of_stock_plot"] = rec["days_of_stock"].replace(np.inf, np.nan)

        categories = sorted(rec["category"].unique())
        data_by_cat = [
            rec[rec["category"] == c]["days_of_stock_plot"].dropna().values
            for c in categories
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(data_by_cat, labels=categories, patch_artist=True)
        ax.axhline(self.lead_time_days, color="red", linestyle="--",
                   linewidth=1.2, label=f"Lead time ({self.lead_time_days}d)")
        ax.set_xlabel("Category")
        ax.set_ylabel("Days of Stock Remaining")
        ax.set_title("Days of Stock Remaining by Category")
        ax.legend()
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "inventory_days_of_stock.png", dpi=150)
        plt.close()
        print("saved inventory_days_of_stock.png")

    def _plot_status_breakdown(self) -> None:
        rec = self._recommendations.copy()
        breakdown = (
            rec.groupby(["category", "status"])
            .size()
            .unstack(fill_value=0)
        )
        for col in [STATUS_STOCKOUT, STATUS_OVERSTOCK, STATUS_OPTIMAL]:
            if col not in breakdown.columns:
                breakdown[col] = 0
        breakdown = breakdown[[STATUS_OPTIMAL, STATUS_OVERSTOCK, STATUS_STOCKOUT]]

        fig, ax = plt.subplots(figsize=(10, 5))
        breakdown.plot(kind="bar", stacked=True, ax=ax,
                       color=["#4caf50", "#ff9800", "#f44336"])
        ax.set_xlabel("Category")
        ax.set_ylabel("SKU count")
        ax.set_title("Inventory Status Breakdown by Category")
        ax.legend(title="Status", bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "inventory_status_breakdown.png", dpi=150)
        plt.close()
        print("saved inventory_status_breakdown.png")

    # ------------------------------------------------------------------
    # Convenience: run full pipeline in one call
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        self.load()
        self.compute_demand_stats()
        self.compute_safety_stock()
        self.compute_reorder_point()
        self.compute_eoq()
        self.evaluate_current_stock()
        self.generate_recommendations()
        self.plot_all()
        self.log_to_mlflow()
        return self._recommendations
