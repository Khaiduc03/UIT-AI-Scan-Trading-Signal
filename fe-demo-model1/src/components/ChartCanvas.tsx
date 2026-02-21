import {
  CandlestickSeries,
  type CandlestickData,
  ColorType,
  createChart,
  type IChartApi,
  type ISeriesApi,
  LineSeries,
  type LineData,
  type Time,
} from 'lightweight-charts';
import { useEffect, useMemo, useRef, useState } from 'react';
import { parseIsoToUnixSeconds } from '../lib/time';
import { getZoneLogicalRange } from '../lib/zoneMath';
import type { Candle, Hotzone, RectGeometry, ZoneRiskPoint, ZoneTooltipState } from '../types/data';

type ChartCanvasProps = {
  candles: Candle[];
  zones: Hotzone[];
  zoneRiskPoints: ZoneRiskPoint[];
  showHotzones: boolean;
  showZoneRisk: boolean;
  opacity: number;
  selectedZoneId: number | null;
  onSelectZone: (zone: Hotzone) => void;
  onHoverZone: (tooltip: ZoneTooltipState) => void;
};

const CHART_OPTIONS = {
  layout: {
    background: { type: ColorType.Solid, color: '#081320' },
    textColor: '#d5deea',
  },
  grid: {
    vertLines: { color: 'rgba(255,255,255,0.05)' },
    horzLines: { color: 'rgba(255,255,255,0.05)' },
  },
  rightPriceScale: {
    borderVisible: false,
  },
  timeScale: {
    borderVisible: false,
    timeVisible: true,
    secondsVisible: false,
  },
  crosshair: {
    vertLine: { color: 'rgba(159, 207, 255, 0.5)' },
    horzLine: { color: 'rgba(159, 207, 255, 0.5)' },
  },
};

export function ChartCanvas({
  candles,
  zones,
  zoneRiskPoints,
  showHotzones,
  showZoneRisk,
  opacity,
  selectedZoneId,
  onSelectZone,
  onHoverZone,
}: ChartCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const riskSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const [renderTick, setRenderTick] = useState(0);
  const rafRef = useRef<number | null>(null);

  const candlesData = useMemo<CandlestickData<Time>[]>(() => {
    return candles.map((c) => ({
      time: parseIsoToUnixSeconds(c.time) as Time,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
  }, [candles]);

  const zoneRiskData = useMemo<LineData<Time>[]>(() => {
    return zoneRiskPoints.map((p) => ({
      time: parseIsoToUnixSeconds(p.time) as Time,
      value: p.zoneRisk,
    }));
  }, [zoneRiskPoints]);

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }
    const chart = createChart(containerRef.current, {
      ...CHART_OPTIONS,
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
    });
    chartRef.current = chart;

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#24c883',
      downColor: '#f25f5c',
      wickUpColor: '#24c883',
      wickDownColor: '#f25f5c',
      borderVisible: false,
    });
    candleSeriesRef.current = candleSeries;

    const riskSeries = chart.addSeries(LineSeries, {
      color: '#ffd166',
      lineWidth: 2,
      priceScaleId: 'risk',
      visible: showZoneRisk,
      crosshairMarkerVisible: false,
    });
    riskSeriesRef.current = riskSeries;
    chart.priceScale('risk').applyOptions({
      visible: showZoneRisk,
      scaleMargins: { top: 0.85, bottom: 0.0 },
      autoScale: true,
      borderVisible: false,
    });

    const queueRepaint = () => {
      if (rafRef.current !== null) {
        return;
      }
      rafRef.current = window.requestAnimationFrame(() => {
        rafRef.current = null;
        setRenderTick((x) => x + 1);
      });
    };

    const handleRangeChange = () => queueRepaint();
    const handleLogicalRangeChange = () => queueRepaint();
    chart.timeScale().subscribeVisibleTimeRangeChange(handleRangeChange);
    chart.timeScale().subscribeVisibleLogicalRangeChange(handleLogicalRangeChange);

    const ro = new ResizeObserver(() => {
      if (!containerRef.current || !chartRef.current) {
        return;
      }
      chartRef.current.applyOptions({
        width: containerRef.current.clientWidth,
        height: containerRef.current.clientHeight,
      });
      queueRepaint();
    });
    ro.observe(containerRef.current);

    const el = containerRef.current;
    const handleWheel = () => queueRepaint();
    const handleTouchMove = () => queueRepaint();
    el.addEventListener('wheel', handleWheel, { passive: true });
    el.addEventListener('touchmove', handleTouchMove, { passive: true });

    return () => {
      chart.timeScale().unsubscribeVisibleTimeRangeChange(handleRangeChange);
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(handleLogicalRangeChange);
      ro.disconnect();
      el.removeEventListener('wheel', handleWheel);
      el.removeEventListener('touchmove', handleTouchMove);
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      riskSeriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!candleSeriesRef.current || candlesData.length === 0) {
      return;
    }
    candleSeriesRef.current.setData(candlesData);
    chartRef.current?.timeScale().fitContent();
    setRenderTick((x) => x + 1);
  }, [candlesData]);

  useEffect(() => {
    if (!riskSeriesRef.current || !chartRef.current) {
      return;
    }
    riskSeriesRef.current.setData(zoneRiskData);
    riskSeriesRef.current.applyOptions({ visible: showZoneRisk });
    chartRef.current.priceScale('risk').applyOptions({ visible: showZoneRisk });
    setRenderTick((x) => x + 1);
  }, [zoneRiskData, showZoneRisk]);

  useEffect(() => {
    setRenderTick((x) => x + 1);
  }, [opacity, showHotzones]);

  const rects = useMemo<RectGeometry[]>(() => {
    const chart = chartRef.current;
    const candleSeries = candleSeriesRef.current;
    if (!chart || !candleSeries || !showHotzones) {
      return [];
    }

    const output: RectGeometry[] = [];
    for (const zone of zones) {
      const x1 = chart.timeScale().timeToCoordinate(parseIsoToUnixSeconds(zone.from_time) as Time);
      const x2 = chart.timeScale().timeToCoordinate(parseIsoToUnixSeconds(zone.to_time) as Time);
      const yTop = candleSeries.priceToCoordinate(zone.top_price);
      const yBottom = candleSeries.priceToCoordinate(zone.bottom_price);

      if ([x1, x2, yTop, yBottom].some((v) => v === null || !Number.isFinite(v))) {
        continue;
      }

      const left = Math.min(x1 as number, x2 as number);
      const right = Math.max(x1 as number, x2 as number);
      const top = Math.min(yTop as number, yBottom as number);
      const bottom = Math.max(yTop as number, yBottom as number);

      output.push({
        zone,
        left,
        top,
        width: Math.max(4, right - left),
        height: Math.max(3, bottom - top),
      });
    }
    return output;
  }, [zones, showHotzones, renderTick]);

  const candleIndexByTime = useMemo(() => {
    const m = new Map<string, number>();
    for (let i = 0; i < candles.length; i += 1) {
      m.set(candles[i].time, i);
    }
    return m;
  }, [candles]);

  useEffect(() => {
    if (!chartRef.current || candles.length === 0 || zones.length === 0 || selectedZoneId !== null) {
      return;
    }
    const first = zones[0];
    const fromIdx = candleIndexByTime.get(first.from_time) ?? first.from_index;
    const toIdx = candleIndexByTime.get(first.to_time) ?? first.to_index;
    const range = getZoneLogicalRange(fromIdx, toIdx, candles.length);
    chartRef.current.timeScale().setVisibleLogicalRange(range);
  }, [candles.length, zones, selectedZoneId, candleIndexByTime]);

  function handleSelectZone(zone: Hotzone) {
    onSelectZone(zone);
    const chart = chartRef.current;
    if (!chart || candles.length === 0) {
      return;
    }

    const fromIdx = candleIndexByTime.get(zone.from_time) ?? zone.from_index;
    const toIdx = candleIndexByTime.get(zone.to_time) ?? zone.to_index;
    const range = getZoneLogicalRange(fromIdx, toIdx, candles.length);
    chart.timeScale().setVisibleLogicalRange(range);
  }

  return (
    <div className="chart-shell">
      <div ref={containerRef} className="chart-root" />
      {showHotzones && (
        <div className="zone-overlay" aria-hidden>
          {rects.map((r) => {
            const selected = selectedZoneId === r.zone.zone_id;
            return (
              <div
                key={r.zone.zone_id}
                className={`zone-rect ${selected ? 'selected' : ''}`}
                style={{
                  left: r.left,
                  top: r.top,
                  width: r.width,
                  height: r.height,
                  opacity: selected ? Math.min(1, opacity + 0.2) : opacity,
                }}
                onMouseEnter={(e) =>
                  onHoverZone({ zone: r.zone, x: e.clientX, y: e.clientY })
                }
                onMouseMove={(e) =>
                  onHoverZone({ zone: r.zone, x: e.clientX, y: e.clientY })
                }
                onMouseLeave={() => onHoverZone(null)}
                onClick={() => handleSelectZone(r.zone)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    handleSelectZone(r.zone);
                  }
                }}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
