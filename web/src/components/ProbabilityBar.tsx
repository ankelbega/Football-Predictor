import styles from "./ProbabilityBar.module.css";

interface Segment {
  label: string;
  value: number;
  color: string;
}

interface ProbabilityBarProps {
  segments: Segment[];
}

export function ProbabilityBar({ segments }: ProbabilityBarProps) {
  const total = segments.reduce((sum, segment) => sum + segment.value, 0) || 1;
  return (
    <div className={styles.root} role="img" aria-label={segments.map((s) => `${s.label} ${Math.round(s.value * 100)}%`).join(", ")}>
      <div className={styles.bar}>
        {segments.map((segment) => (
          <div
            key={segment.label}
            className={styles.segment}
            style={{ width: `${(segment.value / total) * 100}%`, background: segment.color }}
          />
        ))}
      </div>
      <div className={styles.labels}>
        {segments.map((segment) => (
          <span key={segment.label} className={styles.label}>
            {segment.label} {Math.round(segment.value * 100)}%
          </span>
        ))}
      </div>
    </div>
  );
}
