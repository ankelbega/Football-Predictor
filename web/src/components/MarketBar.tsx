import styles from "./MarketBar.module.css";

interface MarketBarProps {
  label: string;
  firstLabel: string;
  firstValue: number;
  secondLabel: string;
  secondValue: number;
}

export function MarketBar({ label, firstLabel, firstValue, secondLabel, secondValue }: MarketBarProps) {
  const total = firstValue + secondValue || 1;
  return (
    <div className={styles.root}>
      <div className={styles.labels}>
        <span>{firstLabel}</span>
        <span>{secondLabel}</span>
      </div>
      <div className={styles.bar} role="img" aria-label={label}>
        <span className={styles.first} style={{ width: `${(firstValue / total) * 100}%` }} />
        <span className={styles.second} style={{ width: `${(secondValue / total) * 100}%` }} />
      </div>
      <div className={styles.values}>
        <span>{(firstValue * 100).toFixed(1)}%</span>
        <span>{(secondValue * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}
