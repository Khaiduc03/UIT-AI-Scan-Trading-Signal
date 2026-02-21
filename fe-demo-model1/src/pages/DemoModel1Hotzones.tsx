import { useEffect, useMemo, useState } from 'react';
import { ChartCanvas } from '../components/ChartCanvas';
import { ControlsBar } from '../components/ControlsBar';
import { ZoneSidebar } from '../components/ZoneSidebar';
import { loadCandles, loadHotzones, loadZoneRiskPoints } from '../lib/loaders';
import type {
  Candle,
  Hotzone,
  HotzonesPayload,
  ZoneRiskPoint,
  ZoneTooltipState,
} from '../types/data';

export function DemoModel1Hotzones() {
  const [candles, setCandles] = useState<Candle[]>([]);
  const [zonesPayload, setZonesPayload] = useState<HotzonesPayload | null>(null);
  const [zoneRiskPoints, setZoneRiskPoints] = useState<ZoneRiskPoint[]>([]);
  const [showHotzones, setShowHotzones] = useState(true);
  const [showZoneRisk, setShowZoneRisk] = useState(true);
  const [opacity, setOpacity] = useState(0.24);
  const [selectedZone, setSelectedZone] = useState<Hotzone | null>(null);
  const [tooltip, setTooltip] = useState<ZoneTooltipState>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [candlesRes, zonesRes, riskRes] = await Promise.all([
          loadCandles(),
          loadHotzones(),
          loadZoneRiskPoints(),
        ]);
        if (!mounted) {
          return;
        }
        setCandles(candlesRes.candles);
        setZonesPayload(zonesRes);
        setZoneRiskPoints(riskRes.points);
      } catch (e) {
        if (!mounted) {
          return;
        }
        setError(e instanceof Error ? e.message : 'Failed to load demo data');
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    }
    load();
    return () => {
      mounted = false;
    };
  }, []);

  const zones = zonesPayload?.zones ?? [];
  const timeframe = zonesPayload?.timeframe ?? '15m';
  const selectedZoneId = selectedZone?.zone_id ?? null;

  const summaryText = useMemo(() => {
    if (!zonesPayload) {
      return 'Loading summary...';
    }
    return `${zonesPayload.total_zones} zones | ${zonesPayload.test_range.rows} bars | ${timeframe}`;
  }, [zonesPayload, timeframe]);

  return (
    <div className="demo-layout">
      <header className="top-header">
        <div>
          <h1>Model 1 Hot Zones + ZoneRisk Overlay</h1>
          <p className="muted">
            FE demo visualizes frozen scanner outputs only. Zone count depends on scanner params.
          </p>
        </div>
        <div className="summary-chip">{summaryText}</div>
      </header>

      <ControlsBar
        showHotzones={showHotzones}
        showZoneRisk={showZoneRisk}
        opacity={opacity}
        onToggleHotzones={setShowHotzones}
        onToggleZoneRisk={setShowZoneRisk}
        onOpacityChange={setOpacity}
      />

      {loading && <div className="panel">Loading demo data...</div>}
      {error && <div className="panel error">{error}</div>}

      {!loading && !error && zonesPayload && (
        <div className="content-grid">
          <div className="chart-panel" onMouseLeave={() => setTooltip(null)}>
            <ChartCanvas
              candles={candles}
              zones={zones}
              zoneRiskPoints={zoneRiskPoints}
              showHotzones={showHotzones}
              showZoneRisk={showZoneRisk}
              opacity={opacity}
              selectedZoneId={selectedZoneId}
              onSelectZone={setSelectedZone}
              onHoverZone={setTooltip}
            />
            {tooltip && (
              <div
                className="zone-tooltip"
                style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}
                role="tooltip"
              >
                <div className="tooltip-title">Zone #{tooltip.zone.zone_id}</div>
                <div>Max risk: {tooltip.zone.max_risk.toFixed(4)}</div>
                <div>Avg risk: {tooltip.zone.avg_risk.toFixed(4)}</div>
                <div>From: {tooltip.zone.from_time}</div>
                <div>To: {tooltip.zone.to_time}</div>
              </div>
            )}
          </div>

          <ZoneSidebar selectedZone={selectedZone} totalZones={zones.length} timeframe={timeframe} />
        </div>
      )}
    </div>
  );
}
