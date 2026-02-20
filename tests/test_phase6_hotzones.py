import pandas as pd

from src.evaluation.extract_hotzones import build_hotzones


def _zone_df(risks: list[float]) -> pd.DataFrame:
    times = (
        pd.date_range("2024-01-01", periods=len(risks), freq="15min", tz="UTC")
        .strftime("%Y-%m-%dT%H:%M:%SZ")
        .tolist()
    )
    return pd.DataFrame({"time": times, "zoneRisk": risks})


def test_hotzone_uses_gte_threshold():
    df = _zone_df([0.75, 0.74, 0.75])
    zones = build_hotzones(df, hot_threshold=0.75, min_zone_bars=1, max_gap_bars=0)
    # indexes 0 and 2 are hot when using >=
    assert len(zones) == 2
    assert zones[0]["from_index"] == 0
    assert zones[1]["from_index"] == 2


def test_gap_merge_with_max_gap_1():
    # hot at 0, 2, 4 => with max_gap=1 all merge into single zone 0..4
    df = _zone_df([0.8, 0.1, 0.85, 0.2, 0.9])
    zones = build_hotzones(df, hot_threshold=0.75, min_zone_bars=1, max_gap_bars=1)
    assert len(zones) == 1
    z = zones[0]
    assert z["from_index"] == 0
    assert z["to_index"] == 4
    assert z["count_hot_bars"] == 3
    assert z["count_bars_total"] == 5


def test_gap_not_merge_when_exceeds():
    # hot at 0 and 3 => gap=2 > max_gap=1 => two zones
    df = _zone_df([0.8, 0.1, 0.1, 0.9])
    zones = build_hotzones(df, hot_threshold=0.75, min_zone_bars=1, max_gap_bars=1)
    assert len(zones) == 2


def test_min_zone_bars_filters_by_hot_count():
    df = _zone_df([0.8, 0.1, 0.85, 0.1, 0.2])
    zones = build_hotzones(df, hot_threshold=0.75, min_zone_bars=3, max_gap_bars=1)
    assert zones == []
