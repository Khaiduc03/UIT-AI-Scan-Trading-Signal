import json
import logging
from pathlib import Path

import pandas as pd
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def validate_data(df: pd.DataFrame, expected_columns: list) -> dict:
    """Validate OHLCV data against specific rules."""
    report = {
        "total_rows": len(df),
        "is_sorted": True,
        "has_duplicates": False,
        "missing_columns": [],
        "has_nans": False,
        "errors": [],
    }

    # 1. Check columns
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        report["missing_columns"] = missing_cols
        report["errors"].append(f"Missing columns: {missing_cols}")

    # 2. Check NaNs
    if df.isnull().values.any():
        report["has_nans"] = True
        report["errors"].append("Data contains NaN values.")

        # Log NaN details
        nan_counts = df.isnull().sum()
        nan_cols = nan_counts[nan_counts > 0].to_dict()
        logger.warning(f"NaNs found in columns: {nan_cols}")

    # 3. Check chronological order (is sorted)
    if "time" in df.columns:
        # We need to make sure time is a proper numeric type or comparable string
        # Our download script saves ISO 8601 strings, which are lexicographically sortable
        is_sorted = df["time"].is_monotonic_increasing
        if not is_sorted:
            report["is_sorted"] = False
            report["errors"].append("Data is not sorted chronologically.")

        # 4. Check for duplicates
        if df["time"].duplicated().any():
            report["has_duplicates"] = True
            report["errors"].append("Duplicate timestamps found.")

            duplicate_count = df["time"].duplicated().sum()
            logger.warning(f"Found {duplicate_count} duplicate timestamps.")

    return report


def main():
    config = load_config()
    output_csv = config.get("data", {}).get("output_csv", "artifacts/raw/BTCUSDT_15m.csv")

    # We can hardcode report path or read from config if it existed, we'll put it in reports dir
    report_path = "artifacts/reports/data_quality.json"

    logger.info(f"Loading data from {output_csv} for validation...")

    try:
        df = pd.read_csv(output_csv)
    except FileNotFoundError:
        logger.error(f"File {output_csv} not found. Ensure download step completed successfully.")
        return

    expected_columns = ["time", "open", "high", "low", "close", "volume"]

    logger.info("Starting validation...")
    report = validate_data(df, expected_columns)

    # Check overall status
    is_valid = len(report["errors"]) == 0
    report["is_valid"] = is_valid

    # Print summary
    if is_valid:
        logger.info(f"Data validation PASSED. Checked {report['total_rows']} rows.")
    else:
        logger.error(f"Data validation FAILED with {len(report['errors'])} errors.")
        for err in report["errors"]:
            logger.error(f"  - {err}")

    # Save report
    out_dir = Path(report_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Validation report saved to {report_path}")


if __name__ == "__main__":
    main()
