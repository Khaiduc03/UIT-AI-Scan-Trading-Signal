export function parseIsoToUnixSeconds(iso: string): number {
  const ms = Date.parse(iso);
  if (Number.isNaN(ms)) {
    throw new Error(`Invalid ISO time: ${iso}`);
  }
  return Math.floor(ms / 1000);
}

export function formatIsoShort(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) {
    return iso;
  }
  return d.toISOString().replace('T', ' ').replace('.000Z', ' UTC');
}
