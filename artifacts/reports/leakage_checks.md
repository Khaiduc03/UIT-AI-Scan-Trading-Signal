# Leakage Checks (Phase 6)

- Overall status: **PASS**

## Scope
- Verify feature set excludes leakage columns.
- Verify scanner artifacts are consistent across files.
- Confirm threshold definitions align with config.
- Summarize residual risks.

## Checks
- [PASS] Feature leakage columns: No forbidden leakage columns in model features
- [PASS] zoneRisk row alignment: test rows=10927 zoneRisk rows=10927
- [PASS] zoneRisk time alignment: time column aligned between test and zoneRisk artifacts
- [PASS] Threshold list alignment: report=[0.6, 0.7, 0.75, 0.8] config=[0.6, 0.7, 0.75, 0.8]
- [PASS] Hotzone count sanity: zones count looks reasonable
- [PASS] Hotzone bar-count consistency: zone count_hot_bars matches recomputed values
- [PASS] Label future horizon configured: horizon_k=12

## Residual Risks
- Logic checks are static/offline and do not prove causal safety under all market regimes.
- High class imbalance and regime shifts can mimic leakage-like behavior in threshold metrics.

## Conclusion
**PASS**
