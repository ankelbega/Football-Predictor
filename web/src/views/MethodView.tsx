import data from "../data/performance.json";
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
  return (
    <section className="container view">
      <h1>Method</h1>
      <p>
        This project forecasts football matches from public historical match data. Statistics are real,
        held-out results; estimates remain uncertain rather than promises.
      </p>

      <h2>Data and scope</h2>
      <p>
        Coverage runs from 16 August 2014 through 24 May 2026 across 4,560 matches and 35 clubs. All
        prediction requests after that date are out-of-sample relative to the fitted bundle.
      </p>
      <p>
        Results are produced with an expanding window: 2023/24 is validation, with 2024/25 and
        2025/26 tested one season at a time. There are 380 matches in each test season.
      </p>

      <h2>Models compared</h2>
      <ul>
        <li>Class prior: a baseline using season-level class frequencies.</li>
        <li>Elo: rating-based forecasts, with no match odds.</li>
        <li>Dixon-Coles: a goals-model with shared home advantage and attack/defence parameters.</li>
        <li>LightGBM: gradient boosting on pre-match rolling features.</li>
        <li>LightGBM with one-vs-rest Platt scaling, chosen on validation loss only.</li>
        <li>A weighted ensemble from the same candidate set.</li>
      </ul>

      <h2>Markets and metrics</h2>
      <p>
        Models are scored on log loss, Brier score, ranked probability score, accuracy, and
        top-label expected calibration error where relevant. Bookmaker probabilities are used as a
        benchmark, though de-noised prices are not independent observations and availability varies by row.
      </p>
      <p>
        Over 2.5 is compared with markets; BTTS odds are absent from the source files, so BTTS is
        compared against class priors rather than misleadingly against unrelated prices.
      </p>

      <h2>Leakage controls</h2>
      <p>
        Features are built from matches with dates strictly before the fixture being predicted. The
        API rejects anything on or before the training cutoff and rejects home against the same club.
        No bookmaker odds are used as model training features.
      </p>

      <h2>Calibration</h2>
      <p>
        Raw, temperature-scaled, one-vs-rest Platt and class-wise isotonic calibrations were compared.
        Platt scaling was retained for 1X2 based on validation log loss; class-wise isotonic was retained
        as a diagnostic only due to overfit risk. Isotonic is not presented as the final calibrated model.
      </p>

      <h2>Explanation scale</h2>
      <p>
        LightGBM explanations use SHAP contributions. These describe the change in class raw score
        for that fixture and are not probability-point changes.
      </p>
      <h3>Largest global contributors</h3>
      <ul className={styles.longList}>
        {topFeatures(10).map((row) => (
          <li key={String(row.feature)}>
            {String(row.feature)}: {String(row.mean_abs_shap)}
          </li>
        ))}
      </ul>

      <h2>Limitations</h2>
      <ul>
        <li>Two held-out seasons of 380 matches each is a small sample.</li>
        <li>Rankings and bootstrap intervals retain substantial uncertainty.</li>
        <li>Prices may not be true closing odds; price availability, limits and costs are omitted.</li>
        <li>Historical performance does not guarantee future behaviour.</li>
      </ul>

      <p>
        Data source: football-data.co.uk. Code is released under the MIT licence. Estimates only and not
        betting advice.
      </p>
    </section>
  );
}
