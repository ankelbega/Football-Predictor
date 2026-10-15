interface StatRowProps {
  label: string;
  value: string;
  hint?: string;
}

export function StatRow({ label, value, hint }: StatRowProps) {
  return (
    <div className="stat-row">
      <span className="stat-row__label">{label}</span>
      <span className="stat-row__value">{value}</span>
      {hint ? <span className="stat-row__hint">{hint}</span> : null}
    </div>
  );
}
