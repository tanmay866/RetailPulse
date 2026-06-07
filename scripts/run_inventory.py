import argparse

from src.inventory_optimizer import InventoryOptimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inventory optimization using forecasted demand.")
    parser.add_argument("--service-level",     type=float, default=0.95, help="Target service level (default: 0.95)")
    parser.add_argument("--lead-time",         type=int,   default=1,    help="Lead time in days (default: 1)")
    parser.add_argument("--ordering-cost",     type=float, default=50.0, help="Fixed cost per order (default: 50.0)")
    parser.add_argument("--holding-cost-rate", type=float, default=0.25, help="Annual holding cost as fraction of unit price (default: 0.25)")
    args = parser.parse_args()

    optimizer = InventoryOptimizer(
        service_level     = args.service_level,
        lead_time_days    = args.lead_time,
        ordering_cost     = args.ordering_cost,
        holding_cost_rate = args.holding_cost_rate,
    )
    optimizer.run()


if __name__ == "__main__":
    main()
