export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function improvementLabel(value: number): string {
  if (Math.abs(value) < 0.05) return "No material change";
  return value > 0 ? `${value.toFixed(1)}% better` : `${Math.abs(value).toFixed(1)}% worse`;
}
