import styles from "./ExplanatoryFeatureList.module.css";
import type { ExplanatoryFeature } from "../lib/types";

interface ExplanatoryFeatureListProps {
  model: "lightgbm" | "dixon_coles";
  items: ExplanatoryFeature[];
}

export function ExplanatoryFeatureList({ model, items }: ExplanatoryFeatureListProps) {
  if (model !== "lightgbm") {
    return (
      <ul className={styles.list}>
        {items.map((item) => (
          <li key={item.feature} className={styles.dixonItem}>
            <span className={styles.feature}>{item.feature}</span>
            <span className={styles.value}>{valueLabel(item.value)}</span>
          </li>
        ))}
      </ul>
    );
  }

  const maxAbs = Math.max(...items.map((item) => Math.abs(item.shap_value ?? 0)), 0.001);

  return (
    <ul className={styles.list}>
      {items.map((item) => {
        const shap = item.shap_value ?? 0;
        return (
          <li key={item.feature} className={styles.item}>
            <div className={styles.topline}>
              <span className={styles.feature}>{item.label ?? item.feature}</span>
              <span className={styles.contribution}>{shap >= 0 ? "+" : "\u2212"}{Math.abs(shap).toFixed(2)}</span>
            </div>
            <div className={styles.barTrack}>
              <span className={styles.bar} style={{ width: `${Math.min(100, (Math.abs(shap) / maxAbs) * 100)}%` }} />
            </div>
            {item.interpretation ? <p className={styles.detail}>{item.interpretation}</p> : null}
          </li>
        );
      })}
    </ul>
  );
}

function valueLabel(value: number | null | undefined): string {
  if (value === null || value === undefined) return "\u2014";
  return value.toFixed(2);
}
