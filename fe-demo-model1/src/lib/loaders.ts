import type { CandlesPayload, HotzonesPayload, ZoneRiskPointsPayload } from '../types/data';

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }
  return (await response.json()) as T;
}

export function loadCandles(): Promise<CandlesPayload> {
  return fetchJson<CandlesPayload>('/data/candles_test.json');
}

export function loadHotzones(): Promise<HotzonesPayload> {
  return fetchJson<HotzonesPayload>('/data/hotzones_ui.json');
}

export function loadZoneRiskPoints(): Promise<ZoneRiskPointsPayload> {
  return fetchJson<ZoneRiskPointsPayload>('/data/zoneRisk_points.json');
}
