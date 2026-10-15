const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatOneInN(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "\u2014";
  return `1 in ${Math.round(1 / value)}`;
}

export function formatProbability(value: number, mode: "percent" | "oneInN"): string {
  return mode === "oneInN" ? formatOneInN(value) : formatPercent(value);
}

export function formatIsoDate(value: string): string {
  const date = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return value;
  return `${date.getUTCDate()} ${MONTHS[date.getUTCMonth()]} ${date.getUTCFullYear()}`;
}

export function addDaysToDate(value: string, days: number): string {
  const date = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return value;
  date.setUTCDate(date.getUTCDate() + days);
  return date.toISOString().slice(0, 10);
}

export function formatIntlDate(value: string, locale: "en" | "sq"): string {
  const date = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(locale === "sq" ? "sq-AL" : "en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(date);
}

export function cutoffDateOnly(value: string): string {
  const match = value.match(/^(\d{4}-\d{2}-\d{2})/);
  return match ? match[1] : value;
}
export function formatScoreline(home: number, away: number): string {
  return `${home}\u2013${away}`;
}
