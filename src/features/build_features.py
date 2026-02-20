import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_raw_data(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    expected_cols = ["time", "open", "high", "low", "close", "volume"]
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def compute_atr(df: pd.DataFrame, atr_period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=atr_period, min_periods=atr_period).mean()


def build_core_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    feat_cfg = cfg.get("features", {})
    atr_period = int(feat_cfg.get("atr_period", 14))
    vol_sma_period = int(feat_cfg.get("vol_sma_period", 50))
    returns_periods = set(int(p) for p in feat_cfg.get("returns_periods", [1, 3, 6]))
    returns_periods.update({1, 3, 6})
    returns_format = feat_cfg.get("returns_format", "ratio")

    out = pd.DataFrame({"time": df["time"]})
    out["atr14"] = compute_atr(df, atr_period=atr_period)
    out["atr_pct"] = out["atr14"] / df["close"]
    out["range1"] = df["high"] - df["low"]

    for p in sorted(returns_periods):
        col = f"ret{p}"
        out[col] = df["close"].pct_change(periods=int(p))
        if returns_format == "percent":
            out[col] = out[col] * 100.0

    out["abs_ret1"] = out["ret1"].abs()
    out["vol_sma50"] = (
        df["volume"]
        .rolling(
            window=vol_sma_period,
            min_periods=vol_sma_period,
        )
        .mean()
    )
    out["vol_ratio"] = df["volume"] / out["vol_sma50"]

    return out


def detect_confirmed_swings(df: pd.DataFrame, swing_size: int) -> tuple[pd.Series, pd.Series]:
    n = len(df)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    confirmed_swing_high = np.full(n, np.nan)
    confirmed_swing_low = np.full(n, np.nan)

    # Pivot at i is only known at i + swing_size (confirmation delay).
    for i in range(swing_size, n - swing_size):
        left_high = highs[i - swing_size : i]
        right_high = highs[i + 1 : i + swing_size + 1]
        if highs[i] > left_high.max() and highs[i] >= right_high.max():
            confirm_idx = i + swing_size
            if confirm_idx < n:
                confirmed_swing_high[confirm_idx] = highs[i]

        left_low = lows[i - swing_size : i]
        right_low = lows[i + 1 : i + swing_size + 1]
        if lows[i] < left_low.min() and lows[i] <= right_low.min():
            confirm_idx = i + swing_size
            if confirm_idx < n:
                confirmed_swing_low[confirm_idx] = lows[i]

    return pd.Series(confirmed_swing_high), pd.Series(confirmed_swing_low)


def build_structure_features(df: pd.DataFrame, core_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    swing_size = int(cfg.get("features", {}).get("swing_size", 5))

    confirmed_high, confirmed_low = detect_confirmed_swings(df, swing_size=swing_size)

    out = pd.DataFrame({"time": df["time"]})
    out["last_swing_high"] = confirmed_high.ffill()
    out["last_swing_low"] = confirmed_low.ffill()

    atr = core_df["atr14"].replace(0, np.nan)
    close = df["close"]
    out["dist_high_atr"] = (out["last_swing_high"] - close) / atr
    out["dist_low_atr"] = (close - out["last_swing_low"]) / atr
    out["near_structure"] = pd.concat(
        [out["dist_high_atr"], out["dist_low_atr"]],
        axis=1,
    ).min(axis=1)

    return out


def apply_warmup(df: pd.DataFrame, warmup_bars: int) -> pd.DataFrame:
    if warmup_bars <= 0:
        return df
    return df.iloc[warmup_bars:].reset_index(drop=True)


def main():
    cfg = load_config("configs/config.yaml")

    input_csv = cfg.get("data", {}).get("output_csv", "artifacts/raw/BTCUSDT_15m.csv")
    processed_dir = Path(cfg.get("split", {}).get("output_dir", "artifacts/processed"))
    warmup_bars = int(cfg.get("features", {}).get("warmup_bars", 300))

    core_path = processed_dir / "features_core.parquet"
    structure_path = processed_dir / "features_structure.parquet"
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw data from {input_csv}...")
    raw_df = load_raw_data(input_csv)

    logger.info("Building core features...")
    core_raw = build_core_features(raw_df, cfg)
    core_df = apply_warmup(core_raw, warmup_bars=warmup_bars)
    core_required = [
        "atr14",
        "atr_pct",
        "range1",
        "ret1",
        "ret3",
        "ret6",
        "abs_ret1",
        "vol_sma50",
        "vol_ratio",
    ]
    core_df = core_df.dropna(subset=core_required)

    logger.info("Building structure features...")
    structure_df = build_structure_features(raw_df, core_raw, cfg)
    structure_df = apply_warmup(structure_df, warmup_bars=warmup_bars)

    # Keep rows where at least one swing side is known, but avoid dropping all data if sparse.
    structure_df = structure_df.dropna(
        subset=["dist_high_atr", "dist_low_atr", "near_structure"],
        how="all",
    )

    logger.info(f"Saving core features to {core_path}...")
    core_df.to_parquet(core_path, index=False)

    logger.info(f"Saving structure features to {structure_path}...")
    structure_df.to_parquet(structure_path, index=False)

    logger.info(f"Done. core_rows={len(core_df)}, structure_rows={len(structure_df)}")


if __name__ == "__main__":
    main()
