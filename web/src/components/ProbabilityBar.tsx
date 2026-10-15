import { formatProbability } from "../lib/format";
import styles from "./ProbabilityBar.module.css";

interface Segment {
  label: string;
  value: number;
  color: string;
}

interface ProbabilityBarProps {
  segments: Segment[];
  displayMode?: "percent" | "oneInN";
}

export function ProbabilityBar({ segments, displayMode = "percent" }: ProbabilityBarProps) {
  const total = segments.reduce((sum, segment) => sum + segment.value, 0) || 1;
  return (
    <div className={styles.root} data-testid="outlook-bar">
      <div className={styles.stats}>
        {segments.map((segment) => (
          <div key={segment.label} className={styles.stat}>
            <div className={styles.statLabel}>
              <span className={styles.key} style={{ background: segment.color }} aria-hidden="true" />
              <span className={styles.labelText}>{segment.label}</span>
            </div>
            <div className={styles.statValue}>{formatProbability(segment.value, displayMode)}</div>
          </div>
        ))}
      </div>
      <div className={styles.bar} role="img" aria-label={segments.map((segment) => `${segment.label} ${formatProbability(segment.value, displayMode)}`).join(", ")}>
        {segments.map((segment) => (
          <div
            key={segment.label}
            className={styles.segment}
            style={{ width: `${(segment.value / total) * 100}%`, background: segment.color }}
          />
        ))}
      </div>
    </div>
  );
}
