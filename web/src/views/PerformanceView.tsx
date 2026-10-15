import data from "../data/performance.json";
import "./PerformanceView.css";

type Phase4Row = (typeof data.phase4_model_comparison)[number];

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function byMarketAndSeason(market: string) {
  return (data.phase4_model_comparison as Phase4Row[])
    .filter((row) => row.partition === "test" && row.market === market)
    .sort((left, right) =>
      String(left.season).localeCompare(String(right.season)),
    );
}

function bestModelFor(rows: Phase4Row[], season: string): string | undefined {
  const seasonRows = rows.filter(
    (row) => String(row.season) === season && isNumber(row.log_loss),
  );
  if (seasonRows.length === 0) return undefined;
  return seasonRows.reduce((best, row) =>
    isNumber(row.log_loss) &&
    (best.log_loss === null || row.log_loss < (best.log_loss ?? Infinity))
      ? row
      : best,
  ).model as string | undefined;
}

function MetricTable({ title, market }: { title: string; market: string }) {
  const rows = byMarketAndSeason(market);
  const seasons = [...new Set(rows.map((row) => String(row.season)))];
  return (
    <section className="card-block">
      <h2>{title}</h2>
      <p className="card-block__note">
        Test partitions only. Lower log loss is better; the lowest value for each season is underlined.
      </p>
      {seasons.map((season) => {
        const seasonRows = rows.filter((row) => String(row.season) === season);
        const best = bestModelFor(rows, season);
        return (
          <div key={season} className="season-block">
            <h3>Season {season}</h3>
            <table className="metric-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Matches</th>
                  <th>Log loss</th>
                  <th>Brier score</th>
                  <th>Accuracy</th>
                  {market === "1x2" ? <th>RPS</th> : null}
                </tr>
              </thead>
              <tbody>
                {seasonRows.map((row) => (
                  <tr key={`${row.model}-${row.season}`}>
                    <td>{formatModel(String(row.model))}</td>
                    <td className="num">{row.matches ?? "\u2014"}</td>
                    <td className={`num ${row.model === best ? "is-best" : ""}`}>
                      {formatMetric(row.log_loss)}
                    </td>
                    <td className="num">{formatMetric(row.brier_score)}</td>
                    <td className="num">{formatMetric(row.accuracy)}</td>
                    {market === "1x2" ? (
                      <td className="num">{formatMetric(row.rps)}</td>
                    ) : null}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })}
    </section>
  );
}

function formatModel(model: string): string {
  return model
    .replaceAll("_", " ")
    .replace("lightgbm calibrated platt", "LightGBM (Platt)")
    .replace("lightgbm calibrated raw", "LightGBM (raw)")
    .replace("lightgbm", "LightGBM")
    .replace("dixon coles", "Dixon-Coles")
    .replace("class prior", "Class prior")
    .replace("bookmaker", "Bookmaker");
}

function formatMetric(value: unknown): string {
  return isNumber(value) ? value.toFixed(4) : "\u2014";
}

export function PerformanceView() {
  const walkForward = data.walk_forward_test_results as Phase4Row[];

  return (
    <section className="container view">
      <h1>Performance</h1>
      <p>
        Hold-out results come from committed evaluation tables. Estimates only and not a guarantee of future
        outcomes.
      </p>

      <MetricTable title="1X2" market="1x2" />
      <MetricTable title="Over 2.5" market="over_2_5" />
      <MetricTable title="Both teams to score" market="btts" />

      <section className="card-block">
        <h2>Walk-forward expansion</h2>
        <img
          src="/performance/walk-forward-log-loss.png"
          alt="Walk-forward log-loss comparison across seasons 2024/25 and 2025/26."
          loading="lazy"
        />
        <p className="card-block__note">
          Each season is evaluated one at a time on 380 held-out matches. Details are in the table below.
        </p>
        <table className="metric-table">
          <thead>
            <tr>
              <th>Season</th>
              <th>Model</th>
              <th>Matches</th>
              <th>Log loss</th>
              <th>Brier score</th>
              <th>Accuracy</th>
            </tr>
          </thead>
          <tbody>
            {walkForward.map((row) => (
              <tr key={`${row.season}-${row.model}`}>
                <td>{row.season}</td>
                <td>{formatModel(String(row.model))}</td>
                <td className="num">{row.matches ?? "\u2014"}</td>
                <td className="num">{formatMetric(row.log_loss)}</td>
                <td className="num">{formatMetric(row.brier_score)}</td>
                <td className="num">{formatMetric(row.accuracy)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="card-block">
        <h2>Reliability</h2>
        <img
          src="/performance/reliability-validation.png"
          alt="Validation reliability diagram comparing predicted and observed frequencies."
          loading="lazy"
        />
        <p className="card-block__note">
          Validation-only isotonic diagnostics are retained as diagnostics rather than claimed test-set results.
        </p>
      </section>

      <p className="card-block__note">
        Calibration notes: Platt scaling improved 1X2 validation loss; class-wise isotonic was diagnostic only.
      </p>
      <p className="card-block__note">
        Small samples: 380 matches per test season. Intervals are wide and rankings are uncertain.
      </p>
    </section>
  );
}
