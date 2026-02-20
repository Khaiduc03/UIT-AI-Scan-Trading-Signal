from src.evaluation.leakage_checks_phase6 import (
    check_feature_leakage,
    check_hotzones_reasonable,
    check_threshold_alignment,
)


def test_check_feature_leakage():
    ok, leaked = check_feature_leakage(["atr14", "ret1", "near_structure"])
    assert ok is True
    assert leaked == []

    ok2, leaked2 = check_feature_leakage(["atr14", "future_range", "strongmove_threshold"])
    assert ok2 is False
    assert leaked2 == ["future_range", "strongmove_threshold"]


def test_check_threshold_alignment():
    report = [{"threshold": 0.6}, {"threshold": 0.7}, {"threshold": 0.75}, {"threshold": 0.8}]
    cfg = [0.8, 0.75, 0.7, 0.6]
    ok, rep, conf = check_threshold_alignment(report, cfg)
    assert ok is True
    assert rep == [0.6, 0.7, 0.75, 0.8]
    assert conf == [0.6, 0.7, 0.75, 0.8]


def test_check_hotzones_reasonable():
    ok, _ = check_hotzones_reasonable(10, 1000)
    assert ok is True

    ok_zero, msg_zero = check_hotzones_reasonable(0, 1000)
    assert ok_zero is False
    assert "no zones" in msg_zero

    ok_one, msg_one = check_hotzones_reasonable(1, 1000)
    assert ok_one is False
    assert "single zone" in msg_one
