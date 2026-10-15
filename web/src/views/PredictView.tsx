import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Combobox } from "../components/Combobox";
import { Disclaimer } from "../components/Disclaimer";
import { ExplanatoryFeatureList } from "../components/ExplanatoryFeatureList";
import { MarketBar } from "../components/MarketBar";
import { ProbabilityBar } from "../components/ProbabilityBar";
import { ScorelineList } from "../components/ScorelineList";
import { useI18n } from "../i18n/useI18n";
import { useHealth } from "../hooks/useHealth";
import { useTeams } from "../hooks/useTeams";
import { usePrediction } from "../hooks/usePrediction";
import { addDaysToDate, cutoffDateOnly, formatIntlDate, formatIsoDate } from "../lib/format";
import type { ExplanatoryFeature, PredictionResponse } from "../lib/types";

const MODELS = ["lightgbm", "dixon_coles"] as const;

export function PredictView() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [displayMode, setDisplayMode] = useState<"percent" | "oneInN">("percent");
  const home = searchParams.get("home") ?? "";
  const away = searchParams.get("away") ?? "";
  const date = searchParams.get("date") ?? "";
  const rawModel = searchParams.get("model") ?? "lightgbm";
  const model = rawModel === "dixon_coles" ? "dixon_coles" : "lightgbm";
  const { t } = useI18n();

  const { data: teams, loading: teamsLoading, error: teamsError } = useTeams();
  const { data: health } = useHealth();
  const cutoff = health?.training_data_cutoff ?? "2026-05-24T00:00:00";
  const minimumDate = addDaysToDate(cutoffDateOnly(cutoff), 1);

  const sameTeam = home !== "" && home === away;
  const dateError =
    date !== "" && date < minimumDate
      ? t("predict.dateTooEarly").replace("{date}", formatIsoDate(minimumDate))
      : "";
  const formIsValid =
    home !== "" && away !== "" && date !== "" && !sameTeam && dateError === "";

  const predictionState = usePrediction(
    { home, away, date, model },
    formIsValid,
  );
  const prediction = predictionState.result;
  const predictionLoading = predictionState.loading;
  const predictionError = predictionState.error;

  const updateParam = (field: "home" | "away" | "date" | "model", value: string) => {
    const next = new URLSearchParams(searchParams);
    if (value) {
      next.set(field, value);
    } else {
      next.delete(field);
    }
    setSearchParams(next, { replace: true });
  };

  useEffect(() => {
    // Seed the first visit with a valid date after loading the API cutoff.
    if (!date && !home && !away) {
      updateParam("date", minimumDate);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [minimumDate]);

  const teamOptions = teams ?? [];

  return (
    <section className="predict-view">
      <div className="predict-view__intro">
        <h1>{t("predict.title")}</h1>
        <p>{t("predict.lead")}</p>
      </div>

      <div className="predict-controls">
        <Combobox
          id="prediction-home"
          label={t("predict.homeTeam")}
          options={teamOptions}
          value={home}
          onChange={(value) => updateParam("home", value)}
          placeholder={teamsLoading ? t("predict.loadingClubs") : t("predict.chooseClub")}
          disabled={teamsLoading || teamOptions.length === 0}
        />
        <Combobox
          id="prediction-away"
          label={t("predict.awayTeam")}
          options={teamOptions}
          value={away}
          onChange={(value) => updateParam("away", value)}
          placeholder={teamsLoading ? t("predict.loadingClubs") : t("predict.chooseClub")}
          disabled={teamsLoading || teamOptions.length === 0}
        />
        <label className="predict-field" htmlFor="prediction-date">
          <span className="predict-field__label">{t("predict.fixtureDate")}</span>
          <input
            id="prediction-date"
            type="date"
            min={minimumDate}
            value={date}
            onChange={(event) => updateParam("date", event.target.value)}
          />
        </label>
        <label className="predict-field" htmlFor="prediction-model">
          <span className="predict-field__label">{t("predict.model")}</span>
          <select
            id="prediction-model"
            value={model}
            onChange={(event) => updateParam("model", event.target.value)}
          >
            {MODELS.map((entry) => (
              <option key={entry} value={entry}>
                {entry === "lightgbm" ? "LightGBM" : "Dixon-Coles"}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="predict-meta">
        <div className="predict-display-toggle" role="group" aria-label={t("predict.displayPercent")}>
          <button
            type="button"
            className={displayMode === "percent" ? "text-button is-selected" : "text-button"}
            onClick={() => setDisplayMode("percent")}
          >
            {t("predict.displayPercent")}
          </button>
          <button
            type="button"
            className={displayMode === "oneInN" ? "text-button is-selected" : "text-button"}
            onClick={() => setDisplayMode("oneInN")}
          >
            {t("predict.displayOneInN")}
          </button>
        </div>
      </div>

      {teamsError ? <p role="alert">{teamsError}</p> : null}
      {sameTeam ? <p role="alert">{t("predict.sameTeam")}</p> : null}
      {dateError ? <p role="alert">{dateError}</p> : null}
      {!teamsLoading && teamOptions.length === 0 ? (
        <p role="status">{t("predict.noClubList")}</p>
      ) : null}

      <div className={`prediction-output ${predictionState.stale ? "is-stale" : ""}`} aria-busy={predictionLoading}>
        {predictionLoading && !prediction ? <p>{t("predict.submitting")}</p> : null}
        {predictionError ? (
          <p role="alert">
            {predictionError}{" "}
            <a href="https://github.com/ankelbega/Football-Predictor" rel="noreferrer" target="_blank">
              GitHub
            </a>
          </p>
        ) : null}

        {prediction ? <PredictionResult prediction={prediction} displayMode={displayMode} /> : null}

        {!formIsValid && !predictionLoading ? (
          <p className="prediction-empty">{t("predict.idle")}</p>
        ) : null}
      </div>

      <Disclaimer text={t("footer.disclaimerEstimates")} />
    </section>
  );
}

function PredictionResult({ prediction, displayMode }: { prediction: PredictionResponse; displayMode: "percent" | "oneInN" }) {
  const { t, locale } = useI18n();
  const ideals = prediction.explanatory_features.map((item) => ({
    ...item,
    direction: item.direction ?? "raised",
  }));

  return (
    <>
      <div className="section-header">
        <h2>{prediction.home_team} vs {prediction.away_team}</h2>
        <p>
          {formatIntlDate(prediction.match_date, locale)} | {t("predict.model")}: {prediction.model}
        </p>
      </div>

      <h3>{t("predict.outlookHeading")}</h3>
      <ProbabilityBar
        segments={[
          { label: `${prediction.home_team} ${t("predict.winWord")}`, value: prediction.probabilities_1x2.home_win, color: "var(--color-home)" },
          { label: t("predict.drawOutcome"), value: prediction.probabilities_1x2.draw, color: "var(--color-draw)" },
          { label: `${prediction.away_team} ${t("predict.winWord")}`, value: prediction.probabilities_1x2.away_win, color: "var(--color-away)" },
        ]}
        displayMode={displayMode}
      />

      <div className="predict-columns">
        <section className="predict-column">
          <h3>{t("predict.goalsHeading")}</h3>
          <div className="goals-markets">
            <MarketBar
              label="over/under"
              firstLabel={t("predict.overLabel").replace("{threshold}", "2.5")}
              firstValue={prediction.over_2_5}
              secondLabel={t("predict.underLabel").replace("{threshold}", "2.5")}
              secondValue={prediction.under_2_5}
            />
            <MarketBar
              label="both teams to score"
              firstLabel={t("predict.bttsYes")}
              firstValue={prediction.btts_yes}
              secondLabel={t("predict.bttsNo")}
              secondValue={prediction.btts_no}
            />
          </div>
        </section>
        <section className="predict-column">
          <h3>{t("predict.scorelinesHeading")}</h3>
          <ScorelineList items={prediction.most_likely_scorelines} displayMode={displayMode} />
        </section>
      </div>

      {prediction.explanatory_features.length > 0 ? (
        <>
          <h3>
            {prediction.model === "lightgbm"
              ? t("predict.explanation.lightgbm")
              : t("predict.explanation.dixonColes")}
          </h3>
          {prediction.model === "lightgbm" ? (
            <ExplanatoryFeatureList model={prediction.model} items={ideals} />
          ) : (
            <DixonRatings homeTeam={prediction.home_team} awayTeam={prediction.away_team} features={prediction.explanatory_features} />
          )}
          {prediction.model === "lightgbm" ? (
            <p className="explanatory-scale">{t("predict.contributionRawScore")}</p>
          ) : null}
        </>
      ) : null}

      <p className="prediction-footnote">
        {t("predict.modelLabel")
          .replace("{model}", prediction.model === "lightgbm" ? "LightGBM" : "Dixon-Coles")
          .replace("{date}", formatIntlDate(cutoffDateOnly(prediction.training_data_cutoff), locale))}
      </p>
    </>
  );
}

function ratingValue(features: ExplanatoryFeature[], key: string): number | null {
  const item = features.find((feature) => feature.feature === key);
  return typeof item?.value === "number" ? item.value : null;
}

function displayRating(value: number | null): string {
  return value === null ? "—" : value.toFixed(2);
}

function DixonRatings({
  homeTeam,
  awayTeam,
  features,
}: {
  homeTeam: string;
  awayTeam: string;
  features: ExplanatoryFeature[];
}) {
const homeAttack = ratingValue(features, "home_attack");
  const awayAttack = ratingValue(features, "away_attack");
  const homeDefence = ratingValue(features, "home_defence");
  const awayDefence = ratingValue(features, "away_defence");
  const homeAdvantage = ratingValue(features, "home_advantage_log_rate");

  return (
    <div className="ratings-table">
      <table>
        <thead>
          <tr>
            <th scope="col">Rating</th>
            <th scope="col" className="num">{homeTeam}</th>
            <th scope="col" className="num">{awayTeam}</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">Attack</th>
            <td className="num">{displayRating(homeAttack)}</td>
            <td className="num">{displayRating(awayAttack)}</td>
          </tr>
          <tr>
            <th scope="row">Defence</th>
            <td className="num">{displayRating(homeDefence)}</td>
            <td className="num">{displayRating(awayDefence)}</td>
          </tr>
          <tr>
            <th scope="row">Home advantage</th>
            <td className="num" colSpan={2}>{displayRating(homeAdvantage)}</td>
          </tr>
        </tbody>
      </table>
      <p className="ratings-note">
        Attack and defence are centred log-strengths (zero is league average); home advantage is a log-rate increment.
      </p>
    </div>
  );
}

