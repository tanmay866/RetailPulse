import argparse

from src.churn import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the churn prediction model.")
    parser.add_argument("--n-trials", type=int, default=20, help="Optuna tuning trials (default: 20)")
    args = parser.parse_args()
    run(n_trials=args.n_trials)


if __name__ == "__main__":
    main()
