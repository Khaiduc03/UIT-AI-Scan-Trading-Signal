import pandas as pd

from src.features.build_features import build_core_features, build_structure_features


def _sample_ohlcv(n: int = 80) -> pd.DataFrame:
    base = pd.Series(range(n), dtype="float64")
    close = 40000 + base * 5
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC").strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "open": close - 2,
            "high": close + 10,
            "low": close - 10,
            "close": close,
            "volume": 1000 + base * 3,
        }
    )


def test_core_and_structure_feature_columns_exist():
    cfg = {
        "features": {
            "atr_period": 14,
            "vol_sma_period": 50,
            "returns_periods": [1, 3, 6],
            "returns_format": "ratio",
            "swing_size": 5,
        }
    }
    df = _sample_ohlcv()

    core = build_core_features(df, cfg)
    structure = build_structure_features(df, core, cfg)

    core_expected = {
        "time",
        "atr14",
        "atr_pct",
        "range1",
        "ret1",
        "ret3",
        "ret6",
        "abs_ret1",
        "vol_sma50",
        "vol_ratio",
    }
    structure_expected = {
        "time",
        "last_swing_high",
        "last_swing_low",
        "dist_high_atr",
        "dist_low_atr",
        "near_structure",
    }

    assert core_expected.issubset(set(core.columns))
    assert structure_expected.issubset(set(structure.columns))
