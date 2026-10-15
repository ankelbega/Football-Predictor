import styles from "./ExplanatoryFeatureList.module.css";
import type { ExplanatoryFeature } from "../lib/types";
import { formatExplanationLabel } from "../lib/explainLabels";
import { useI18n } from "../i18n/useI18n";

interface ExplanatoryFeatureListProps {
  model: "lightgbm" | "dixon_coles";
  items: ExplanatoryFeature[];
  explainedOutcome: string;
}

export function ExplanatoryFeatureList({
  model,
  items,
  explainedOutcome,
}: ExplanatoryFeatureListProps) {
  const { t } = useI18n();
  const directionalItems = model === "lightgbm" ? items.slice(0, 5) : items;
  const maximum = Math.max(
    0.001,
    ...directionalItems.map((item) => Math.abs(item.shap_value ?? 0)),
  );

  if (model !== "lightgbm") {
    return (
      <ul className={styles.list}>
        {items.map((item) => (
          <li key={item.feature} className={styles.dixonItem}>
            <span className={styles.feature}>{formatExplanationLabel(item.feature)}</span>
            <span className={styles.value}>{formatValue(item.value ?? null)}</span>
          </li>
        ))}
      </ul>
    );
  }

  return (
    <section className={styles.root}>
      <div className={styles.topline}>
        <h3 className={styles.heading}>
          {t("predict.explanation.lightgbm")}{" "}
          <span className={styles.modelName}>{t("predict.modelLightgbm")}</span>
        </h3>
        <p className={styles.outcome}>
          {t("predict.explaining")} {explainedOutcome}
        </p>
      </div>
      <ul className={styles.list}>
        {directionalItems.map((item) => {
          const contribution = item.shap_value ?? 0;
          const raising = contribution >= 0;
          return (
            <li key={item.feature} className={styles.item}>
              <div className={styles.barsTop}>
                <span className={styles.feature}>{formatExplanationLabel(item.feature)}</span>
                <span className={styles.contribution}>{formatSigned(contribution)}</span>
              </div>
              <div className={styles.track}>
                <span className={styles.zeroLine} aria-hidden="true" />
                <span
                  className={raising ? styles.barRaising : styles.barLowering}
                  style={{ width: `${(Math.abs(contribution) / maximum) * 50}%` }}
                />
              </div>
              <dl className={styles.detail}>
                <div>
                  <dt>{t("predict.explanation.value").replace("{value}", formatValue(item.feature_value ?? null))}</dt>
                  <dd>{raising ? t("predict.explanation.raises") : t("predict.explanation.lowers")}</dd>
                </div>
              </dl>
            </li>
          );
        })}
      </ul>
      <p className={styles.barNote}>{t("predict.explanation.barNote")}</p>
    </section>
  );
}

function formatSigned(value: number): string {
  return `${value >= 0 ? "+" : "\u2212"}${Math.abs(value).toFixed(2)}`;
}

function formatValue(value: number | null): string {
  if (value === null) return "\u2014";
  if (!Number.isFinite(value)) return "\u2014";
  if (Math.abs(value - Math.trunc(value)) < 0.001) return Math.trunc(value).toString();
  if (Math.abs(value) >= 100) return value.toFixed(1);
  return value.toFixed(2);
}
