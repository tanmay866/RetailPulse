"""CLI entry point for feature importance analysis and multi-model Optuna tuning."""
import argparse

from src.model_tuner import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature importance analysis + multi-model Optuna tuning")
    parser.add_argument("--n-trials", type=int, default=75, help="Optuna trials per model (default: 75)")
    args = parser.parse_args()
    run(n_trials=args.n_trials)
