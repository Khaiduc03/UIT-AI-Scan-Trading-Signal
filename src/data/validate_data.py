import json
import logging
from pathlib import Path
from time import perf_counter

import pandas as pd
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Đọc config chung để lấy đường dẫn dữ liệu đầu vào.
def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Kiểm tra chất lượng dữ liệu thô trước khi đi vào feature engineering.
# Các check chính:
# 1) Thiếu cột bắt buộc
# 2) NaN
# 3) Thứ tự thời gian tăng dần
# 4) Trùng timestamp
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

    # 1) Kiểm tra cột bắt buộc.
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        report["missing_columns"] = missing_cols
        report["errors"].append(f"Missing columns: {missing_cols}")

    # 2) Kiểm tra giá trị NaN.
    if df.isnull().values.any():
        report["has_nans"] = True
        report["errors"].append("Data contains NaN values.")

        # Log NaN details
        nan_counts = df.isnull().sum()
        nan_cols = nan_counts[nan_counts > 0].to_dict()
        logger.warning(f"NaNs found in columns: {nan_cols}")

    # 3) Kiểm tra dữ liệu đã sort theo thời gian hay chưa.
    if "time" in df.columns:
        # time đang ở dạng ISO8601 string, vẫn so sánh tăng dần được.
        is_sorted = df["time"].is_monotonic_increasing
        if not is_sorted:
            report["is_sorted"] = False
            report["errors"].append("Data is not sorted chronologically.")

        # 4) Kiểm tra timestamp trùng lặp.
        if df["time"].duplicated().any():
            report["has_duplicates"] = True
            report["errors"].append("Duplicate timestamps found.")

            duplicate_count = df["time"].duplicated().sum()
            logger.warning(f"Found {duplicate_count} duplicate timestamps.")

    return report


def main():
    start_time = perf_counter()

    # 1) Lấy đường dẫn input từ config.
    config = load_config()
    output_csv = config.get("data", {}).get("output_csv", "artifacts/raw/BTCUSDT_15m.csv")

    # Báo cáo validation ghi cố định vào artifacts/reports.
    report_path = "artifacts/reports/data_quality.json"

    logger.info(f"Loading data from {output_csv} for validation...")

    # 2) Đọc dữ liệu thô.
    try:
        df = pd.read_csv(output_csv)
    except FileNotFoundError:
        logger.error(f"File {output_csv} not found. Ensure download step completed successfully.")
        return

    # 3) Thực thi các check chất lượng dữ liệu.
    expected_columns = ["time", "open", "high", "low", "close", "volume"]

    logger.info("Starting validation...")
    report = validate_data(df, expected_columns)

    # 4) Tổng hợp trạng thái pass/fail.
    is_valid = len(report["errors"]) == 0
    report["is_valid"] = is_valid

    # 5) In log tóm tắt.
    if is_valid:
        logger.info(f"Data validation PASSED. Checked {report['total_rows']} rows.")
    else:
        logger.error(f"Data validation FAILED with {len(report['errors'])} errors.")
        for err in report["errors"]:
            logger.error(f"  - {err}")

    # 6) Lưu report JSON để các phase sau kiểm tra nhanh.
    out_dir = Path(report_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Validation report saved to {report_path}")
    logger.info(
        "Validation summary | status=%s | rows=%s | errors=%s | elapsed=%.2fs",
        "PASS" if is_valid else "FAIL",
        report["total_rows"],
        len(report["errors"]),
        perf_counter() - start_time,
    )


if __name__ == "__main__":
    main()
