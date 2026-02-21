import { formatIsoShort } from '../lib/time';
import type { Hotzone } from '../types/data';

type ZoneSidebarProps = {
  selectedZone: Hotzone | null;
  totalZones: number;
  timeframe: string;
};

export function ZoneSidebar({ selectedZone, totalZones, timeframe }: ZoneSidebarProps) {
  return (
    <aside className="zone-sidebar">
      <h2>Model 1 Demo</h2>
      <p className="muted">Timeframe: {timeframe}</p>
      <p className="muted">Total zones: {totalZones}</p>

      {!selectedZone ? (
        <div className="empty-state">Click a zone to inspect details.</div>
      ) : (
        <div className="zone-details">
          <h3>Zone #{selectedZone.zone_id}</h3>
          <p>
            <strong>From:</strong> {formatIsoShort(selectedZone.from_time)}
          </p>
          <p>
            <strong>To:</strong> {formatIsoShort(selectedZone.to_time)}
          </p>
          <p>
            <strong>Top / Bottom:</strong> {selectedZone.top_price.toFixed(2)} /{' '}
            {selectedZone.bottom_price.toFixed(2)}
          </p>
          <p>
            <strong>Mid:</strong> {selectedZone.mid_price.toFixed(2)}
          </p>
          <p>
            <strong>Risk (max / avg):</strong> {selectedZone.max_risk.toFixed(4)} /{' '}
            {selectedZone.avg_risk.toFixed(4)}
          </p>
          <p>
            <strong>Bars (hot / total):</strong> {selectedZone.count_hot_bars} /{' '}
            {selectedZone.count_bars_total}
          </p>
          <p>
            <strong>Index range:</strong> {selectedZone.from_index} - {selectedZone.to_index}
          </p>
        </div>
      )}
    </aside>
  );
}
