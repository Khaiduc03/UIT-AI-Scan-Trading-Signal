import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml

LEAKAGE_COLS = {"future_range", "strongmove_threshold"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def check_feature_leakage(feature_columns: list[str]) -> tuple[bool, list[str]]:
    leaked = sorted(list(set(feature_columns).intersection(LEAKAGE_COLS)))
    return len(leaked) == 0, leaked


def check_threshold_alignment(
    report_thresholds: list[dict],
    config_thresholds: list[float],
) -> tuple[bool, list[float], list[float]]:
    report_vals = sorted([float(x["threshold"]) for x in report_thresholds])
    cfg_vals = sorted([float(x) for x in config_thresholds])
    return report_vals == cfg_vals, report_vals, cfg_vals


def check_hotzones_reasonable(total_zones: int, total_bars: int) -> tuple[bool, str]:
    if total_zones == 0:
        return False, "no zones found"
    if total_zones == 1 and total_bars > 0:
        return False, "single zone may indicate over-grouping"
    return True, "zones count looks reasonable"


def _check_hotzone_membership(
    zone_df: pd.DataFrame,
    zones: list[dict],
    hot_threshold: float,
) -> tuple[bool, int]:
    violations = 0
    risks = zone_df["zoneRisk"].to_numpy(dtype=float)
    for z in zones:
        start_i = int(z["from_index"])
        end_i = int(z["to_index"])
        hot_count = int((risks[start_i : end_i + 1] >= hot_threshold).sum())
        if hot_count != int(z["count_hot_bars"]):
            violations += 1
    return violations == 0, violations


def build_markdown_report(result: dict) -> str:
    lines = [
        "# Leakage Checks (Phase 6)",
        "",
        f"- Overall status: **{result['overall_status']}**",
        "",
        "## Scope",
        "- Verify feature set excludes leakage columns.",
        "- Verify scanner artifacts are consistent across files.",
        "- Confirm threshold definitions align with config.",
        "- Summarize residual risks.",
        "",
        "## Checks",
    ]
    for item in result["checks"]:
        lines.append(f"- [{item['status']}] {item['name']}: {item['detail']}")

    lines.extend(
        [
            "",
            "## Residual Risks",
            (
                "- Logic checks are static/offline and do not prove causal safety "
                "under all market regimes."
            ),
            (
                "- High class imbalance and regime shifts can mimic leakage-like "
                "behavior in threshold metrics."
            ),
            "",
            "## Conclusion",
            f"**{result['overall_status']}**",
            "",
        ]
    )
    return "\n".join(lines)


def run_checks() -> dict:
    cfg = load_config()
    split_cfg = cfg.get("split", {})
    scanner_cfg = cfg.get("scanner", {})
    label_cfg = cfg.get("label", {})
    model_cfg = cfg.get("model1", {})

    processed_dir = Path(split_cfg.get("output_dir", "artifacts/processed"))
    test_path = Path(split_cfg.get("out_test", processed_dir / "test.parquet"))
    model_path = Path(model_cfg.get("output_path", "artifacts/models/model1.pkl"))
    zone_path = Path(
        scanner_cfg.get("out_zoneRisk_test", "artifacts/reports/zoneRisk_test.parquet")
    )
    threshold_path = Path(
        scanner_cfg.get("out_threshold_report", "artifacts/reports/scanner_threshold_report.json")
    )
    hotzones_path = Path(scanner_cfg.get("out_hotzones", "artifacts/reports/hotzones_test.json"))
    leakage_md_path = Path(
        scanner_cfg.get("out_leakage_report", "artifacts/reports/leakage_checks.md")
    )

    required = [test_path, model_path, zone_path, threshold_path, hotzones_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files for leakage checks: {missing}")

    test_df = pd.read_parquet(test_path)
    zone_df = pd.read_parquet(zone_path)
    model_artifact = joblib.load(model_path)
    threshold_report = json.load(open(threshold_path))
    hotzones = json.load(open(hotzones_path))

    checks = []

    ok_no_leak, leaked = check_feature_leakage(model_artifact["feature_columns"])
    checks.append(
        {
            "name": "Feature leakage columns",
            "status": "PASS" if ok_no_leak else "FAIL",
            "detail": "No forbidden leakage columns in model features"
            if ok_no_leak
            else f"Found leakage columns: {leaked}",
        }
    )

    rows_ok = len(test_df) == len(zone_df)
    checks.append(
        {
            "name": "zoneRisk row alignment",
            "status": "PASS" if rows_ok else "FAIL",
            "detail": f"test rows={len(test_df)} zoneRisk rows={len(zone_df)}",
        }
    )

    times_ok = test_df["time"].reset_index(drop=True).equals(zone_df["time"].reset_index(drop=True))
    checks.append(
        {
            "name": "zoneRisk time alignment",
            "status": "PASS" if times_ok else "FAIL",
            "detail": "time column aligned between test and zoneRisk artifacts"
            if times_ok
            else "time mismatch between test and zoneRisk artifacts",
        }
    )

    thr_ok, thr_report, thr_cfg = check_threshold_alignment(
        threshold_report["thresholds"],
        scanner_cfg.get("report_thresholds", [0.60, 0.70, 0.75, 0.80]),
    )
    checks.append(
        {
            "name": "Threshold list alignment",
            "status": "PASS" if thr_ok else "FAIL",
            "detail": f"report={thr_report} config={thr_cfg}",
        }
    )

    hot_reasonable, hot_msg = check_hotzones_reasonable(
        int(hotzones.get("total_zones", 0)),
        int(len(zone_df)),
    )
    checks.append(
        {
            "name": "Hotzone count sanity",
            "status": "PASS" if hot_reasonable else "WARN",
            "detail": hot_msg,
        }
    )

    membership_ok, violations = _check_hotzone_membership(
        zone_df=zone_df,
        zones=hotzones.get("zones", []),
        hot_threshold=float(hotzones.get("hot_threshold", scanner_cfg.get("hot_threshold", 0.75))),
    )
    checks.append(
        {
            "name": "Hotzone bar-count consistency",
            "status": "PASS" if membership_ok else "FAIL",
            "detail": "zone count_hot_bars matches recomputed values"
            if membership_ok
            else f"violations={violations}",
        }
    )

    horizon_ok = int(label_cfg.get("horizon_k", 0)) > 0
    checks.append(
        {
            "name": "Label future horizon configured",
            "status": "PASS" if horizon_ok else "FAIL",
            "detail": f"horizon_k={label_cfg.get('horizon_k')}",
        }
    )

    # Overall status: any FAIL => FAIL; else any WARN => WARN; else PASS.
    statuses = [c["status"] for c in checks]
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "WARN" in statuses:
        overall = "WARN"
    else:
        overall = "PASS"

    result = {
        "overall_status": overall,
        "checks": checks,
        "artifacts": {
            "test_path": str(test_path),
            "model_path": str(model_path),
            "zone_path": str(zone_path),
            "threshold_path": str(threshold_path),
            "hotzones_path": str(hotzones_path),
            "leakage_report_path": str(leakage_md_path),
        },
    }

    leakage_md_path.parent.mkdir(parents=True, exist_ok=True)
    leakage_md_path.write_text(build_markdown_report(result), encoding="utf-8")
    return result


def main():
    result = run_checks()
    logger.info("Leakage checks status: %s", result["overall_status"])
    logger.info("Leakage report saved: %s", result["artifacts"]["leakage_report_path"])


if __name__ == "__main__":
    main()
