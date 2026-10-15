import data from "../data/performance.json";
import { useI18n } from "../i18n/useI18n";
import styles from "./MethodView.module.css";

type ShapRow = (typeof data.global_shap_importance)[number];

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function topFeatures(n: number): ShapRow[] {
  return (data.global_shap_importance as ShapRow[])
    .filter((row) => isNumber(row.mean_abs_shap))
    .sort((left, right) => (right.mean_abs_shap ?? 0) - (left.mean_abs_shap ?? 0))
    .slice(0, n);
}

export function MethodView() {
  const { t } = useI18n();

  return (
    <section className="container view">
      <h1>{t("method.title")}</h1>
      <p>{t("method.intro")}</p>

      <h2>{t("method.dataTitle")}</h2>
      <p>{t("method.dataBody")}</p>
      <p>{t("method.dataBody2")}</p>

      <h2>{t("method.modelsTitle")}</h2>
      <ul>
        <li>{t("method.modelClassPrior")}</li>
        <li>{t("method.modelElo")}</li>
        <li>{t("method.modelDixonColes")}</li>
        <li>{t("method.modelLightGbm")}</li>
        <li>{t("method.modelPlatt")}</li>
        <li>{t("method.modelEnsemble")}</li>
      </ul>

      <h2>{t("method.outcomeMetricsTitle")}</h2>
      <p>{t("method.outcomeMetricsBody")}</p>
      <p>{t("method.bttsNote")}</p>

      <h2>{t("method.leakageTitle")}</h2>
      <p>{t("method.leakageBody")}</p>

      <h2>{t("method.calibrationTitle")}</h2>
      <p>{t("method.calibrationBody")}</p>

      <h2>{t("method.explanationTitle")}</h2>
      <p>{t("method.explanationIntro")}</p>
      <h3>{t("method.largestContributors")}</h3>
      <ul className={styles.longList}>
        {topFeatures(10).map((row) => (
          <li key={String(row.feature)}>
            {String(row.feature)}: {(row.mean_abs_shap as number).toFixed(4)}
          </li>
        ))}
      </ul>

      <h2>{t("method.limitationsTitle")}</h2>
      <ul>
        <li>{t("method.limitation1")}</li>
        <li>{t("method.limitation2")}</li>
        <li>{t("method.limitation3")}</li>
        <li>{t("method.limitation4")}</li>
      </ul>

      <p>{t("method.closing")}</p>
    </section>
  );
}
