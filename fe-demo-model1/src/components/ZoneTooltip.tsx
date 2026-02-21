import { formatIsoShort } from '../lib/time';
import type { ZoneTooltipState } from '../types/data';

type ZoneTooltipProps = {
  tooltip: ZoneTooltipState;
};

export function ZoneTooltip({ tooltip }: ZoneTooltipProps) {
  if (!tooltip) {
    return null;
  }

  return (
    <div
      className="zone-tooltip"
      style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}
      role="tooltip"
    >
      <div className="tooltip-title">Zone #{tooltip.zone.zone_id}</div>
      <div>Max risk: {tooltip.zone.max_risk.toFixed(4)}</div>
      <div>Avg risk: {tooltip.zone.avg_risk.toFixed(4)}</div>
      <div>From: {formatIsoShort(tooltip.zone.from_time)}</div>
      <div>To: {formatIsoShort(tooltip.zone.to_time)}</div>
    </div>
  );
}
