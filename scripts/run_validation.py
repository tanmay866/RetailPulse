import argparse
import sys

from src.validation import format_summary, run_validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final accuracy validation: recompute model metrics and check acceptance gates."
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="Print the summary without writing reports/accuracy_validation.json",
    )
    args = parser.parse_args()

    report = run_validation(write=not args.no_write)
    print(format_summary(report))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
