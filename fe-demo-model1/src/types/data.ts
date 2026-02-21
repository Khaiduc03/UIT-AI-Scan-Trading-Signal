export type Candle = {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type CandlesPayload = {
  timeframe: string;
  rows: number;
  candles: Candle[];
};

export type Hotzone = {
  zone_id: number;
  from_time: string;
  to_time: string;
  from_index: number;
  to_index: number;
  top_price: number;
  bottom_price: number;
  mid_price: number;
  max_risk: number;
  avg_risk: number;
  count_hot_bars: number;
  count_bars_total: number;
};

export type HotzonesPayload = {
  timeframe: string;
  params: {
    hot_threshold: number;
    min_zone_bars: number;
    max_gap_bars: number;
  };
  test_range: {
    start_time: string;
    end_time: string;
    rows: number;
  };
  total_zones: number;
  zones: Hotzone[];
};

export type ZoneRiskPoint = {
  time: string;
  zoneRisk: number;
};

export type ZoneRiskPointsPayload = {
  timeframe: string;
  rows: number;
  points: ZoneRiskPoint[];
};

export type ZoneTooltipState = {
  zone: Hotzone;
  x: number;
  y: number;
} | null;

export type RectGeometry = {
  zone: Hotzone;
  left: number;
  top: number;
  width: number;
  height: number;
};
