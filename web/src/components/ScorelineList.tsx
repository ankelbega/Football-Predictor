import styles from "./ScorelineList.module.css";
import { formatScoreline } from "../lib/format";

interface ScorelineItem {
  home_goals: number;
  away_goals: number;
  probability: number;
}

interface ScorelineListProps {
  items: ScorelineItem[];
  displayMode: "percent" | "oneInN";
}

export function ScorelineList({ items, displayMode }: ScorelineListProps) {
  const maximum = items.reduce(
    (currentMaximum, item) => Math.max(currentMaximum, item.probability),
    0,
  ) || 1;

  return (
    <ol className={styles.list}>
      {items.map((item) => (
        <li key={`${item.home_goals}-${item.away_goals}`} className={styles.row}>
          <span className={styles.score}>{formatScoreline(item.home_goals, item.away_goals)}</span>
          <span className={styles.barTrack}>
            <span
              className={styles.bar}
              style={{ width: `${(item.probability / maximum) * 100}%` }}
            />
          </span>
          <span className={styles.probability}>
            {(item.probability * 100).toFixed(1)}%
            {displayMode === "oneInN" ? ` (${Math.max(1, Math.round(1 / item.probability))})` : ""}
          </span>
        </li>
      ))}
    </ol>
  );
}
