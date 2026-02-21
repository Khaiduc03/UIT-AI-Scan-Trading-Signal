type ControlsBarProps = {
  showHotzones: boolean;
  showZoneRisk: boolean;
  opacity: number;
  onToggleHotzones: (next: boolean) => void;
  onToggleZoneRisk: (next: boolean) => void;
  onOpacityChange: (next: number) => void;
};

export function ControlsBar({
  showHotzones,
  showZoneRisk,
  opacity,
  onToggleHotzones,
  onToggleZoneRisk,
  onOpacityChange,
}: ControlsBarProps) {
  return (
    <div className="controls-bar">
      <label className="control-item">
        <input
          type="checkbox"
          checked={showHotzones}
          onChange={(e) => onToggleHotzones(e.target.checked)}
        />
        <span>Show Hotzones</span>
      </label>

      <label className="control-item">
        <input
          type="checkbox"
          checked={showZoneRisk}
          onChange={(e) => onToggleZoneRisk(e.target.checked)}
        />
        <span>Show ZoneRisk</span>
      </label>

      <label className="control-item slider-item">
        <span>Opacity: {opacity.toFixed(2)}</span>
        <input
          type="range"
          min={0.1}
          max={0.6}
          step={0.01}
          value={opacity}
          onChange={(e) => onOpacityChange(Number(e.target.value))}
        />
      </label>
    </div>
  );
}
