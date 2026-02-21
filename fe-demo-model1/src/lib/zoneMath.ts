export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function getZoneLogicalRange(
  fromIndex: number,
  toIndex: number,
  totalRows: number,
): { from: number; to: number } {
  const zoneLengthBars = Math.max(1, toIndex - fromIndex + 1);
  const paddingBars = clamp(Math.max(8, Math.round(zoneLengthBars * 0.5)), 8, 60);

  const minIndex = 0;
  const maxIndex = Math.max(0, totalRows - 1);
  const from = clamp(fromIndex - paddingBars, minIndex, maxIndex);
  const to = clamp(toIndex + paddingBars, minIndex, maxIndex);
  return { from, to };
}
