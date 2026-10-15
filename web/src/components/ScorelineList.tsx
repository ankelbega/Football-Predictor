import styles from "./ScorelineList.module.css";

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
  return (
    <ol className={styles.list}>
      {items.map((item) => (
        <li key={`${item.home_goals}-${item.away_goals}`} className={styles.item}>
          <span className={styles.score}>{item.home_goals}&ndash;{item.away_goals}</span>
          <span className={styles.probability}>
            {(item.probability * 100).toFixed(1)}%
            {displayMode === "oneInN" ? ` (\u2248 1 in ${Math.max(1, Math.round(1 / item.probability))})` : ""}
          </span>
        </li>
      ))}
    </ol>
  );
}
