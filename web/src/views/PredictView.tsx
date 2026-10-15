import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Combobox } from "../components/Combobox";
import { Disclaimer } from "../components/Disclaimer";
import { ExplanatoryFeatureList } from "../components/ExplanatoryFeatureList";
import { MarketBar } from "../components/MarketBar";
import { ProbabilityBar } from "../components/ProbabilityBar";
import { ScorelineList } from "../components/ScorelineList";
import { useHealth } from "../hooks/useHealth";
import { useTeams } from "../hooks/useTeams";
import { usePrediction } from "../hooks/usePrediction";
import { addDaysToDate, cutoffDateOnly, formatIsoDate, formatProbability } from "../lib/format";
import type { PredictionResponse } from "../lib/types";

const MODELS = ["lightgbm", "dixon_coles"] as const;

export function PredictView() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [displayMode, setDisplayMode] = useState<"percent" | "oneInN">("percent");
  const home = searchParams.get("home") ?? "";
  const away = searchParams.get("away") ?? "";
  const date = searchParams.get("date") ?? "";
  const rawModel = searchParams.get("model") ?? "lightgbm";
  const model = rawModel === "dixon_coles" ? "dixon_coles" : "lightgbm";

  const { data: teams, loading: teamsLoading, error: teamsError } = useTeams();
  const { data: health } = useHealth();
  const cutoff = health?.training_data_cutoff ?? "2026-05-24T00:00:00";
  const minimumDate = addDaysToDate(cutoffDateOnly(cutoff), 1);

  const sameTeam = home !== "" && home === away;
  const dateError =
    date !== "" && date < minimumDate ? `Prediction date must be after ${formatIsoDate(minimumDate)}.` : "";
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
        <h1>Predict a match to come</h1>
        <p>Pick two clubs and a fixture date after the model&apos;s training cutoff.</p>
      </div>

      <div className="predict-controls">
        <Combobox
          id="prediction-home"
          label="Home team"
          options={teamOptions}
          value={home}
          onChange={(value) => updateParam("home", value)}
          placeholder={teamsLoading ? "Loading clubs..." : "Choose a home club"}
          disabled={teamsLoading || teamOptions.length === 0}
        />
        <Combobox
          id="prediction-away"
          label="Away team"
          options={teamOptions}
          value={away}
          onChange={(value) => updateParam("away", value)}
          placeholder={teamsLoading ? "Loading clubs..." : "Choose an away club"}
          disabled={teamsLoading || teamOptions.length === 0}
        />
        <label className="predict-field" htmlFor="prediction-date">
          <span className="predict-field__label">Fixture date</span>
          <input
            id="prediction-date"
            type="date"
            min={minimumDate}
            value={date}
            onChange={(event) => updateParam("date", event.target.value)}
          />
        </label>
        <label className="predict-field" htmlFor="prediction-model">
          <span className="predict-field__label">Model</span>
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

        <div className="predict-display-toggle" role="group" aria-label="Display format">
          <button
            type="button"
            className={displayMode === "percent" ? "text-button is-selected" : "text-button"}
            onClick={() => setDisplayMode("percent")}
          >
            Percent
          </button>
          <button
            type="button"
            className={displayMode === "oneInN" ? "text-button is-selected" : "text-button"}
            onClick={() => setDisplayMode("oneInN")}
          >
            1 in N
          </button>
        </div>
      </div>

      {teamsError ? <p role="alert">{teamsError}</p> : null}
      {sameTeam ? <p role="alert">Choose two different teams.</p> : null}
      {dateError ? <p role="alert">{dateError}</p> : null}
      {!teamsLoading && teamOptions.length === 0 ? (
        <p role="status">Could not load the club list.</p>
      ) : null}

      <div className={`prediction-output ${predictionState.stale ? "is-stale" : ""}`} aria-busy={predictionLoading}>
        {predictionLoading && !prediction ? <p>Calculating...</p> : null}
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
          <p className="prediction-empty">Pick two teams to see the outlook.</p>
        ) : null}
      </div>

      <Disclaimer text="Estimates only. Not betting advice and not a guarantee of the result." />
    </section>
  );
}

function PredictionResult({ prediction, displayMode }: { prediction: PredictionResponse; displayMode: "percent" | "oneInN" }) {
  const ideals = prediction.explanatory_features.map((item) => ({
    ...item,
    direction: item.direction ?? "raised",
  }));

  return (
    <>
      <div className="section-header">
        <h2>{prediction.home_team} vs {prediction.away_team}</h2>
        <p>
          {formatIsoDate(prediction.match_date)} | Model: {prediction.model}
        </p>
      </div>

      <h3>Match outlook</h3>
      <p>
        Home win {formatProbability(prediction.probabilities_1x2.home_win, displayMode)} |{" "}
        Draw {formatProbability(prediction.probabilities_1x2.draw, displayMode)} |{" "}
        Away win {formatProbability(prediction.probabilities_1x2.away_win, displayMode)}
      </p>
      <ProbabilityBar
        segments={[
          { label: "Home", value: prediction.probabilities_1x2.home_win, color: "var(--color-home)" },
          { label: "Draw", value: prediction.probabilities_1x2.draw, color: "var(--color-draw)" },
          { label: "Away", value: prediction.probabilities_1x2.away_win, color: "var(--color-away)" },
        ]}
      />

      <h3>Goals</h3>
      <MarketBar label="over/under" firstLabel="Over 2.5" firstValue={prediction.over_2_5} secondLabel="Under 2.5" secondValue={prediction.under_2_5} />
      <MarketBar label="both teams to score" firstLabel="BTTS yes" firstValue={prediction.btts_yes} secondLabel="No" secondValue={prediction.btts_no} />

      <h3>Most likely scorelines</h3>
      <ScorelineList items={prediction.most_likely_scorelines} displayMode={displayMode} />

      {prediction.explanatory_features.length > 0 ? (
        <>
          <h3>
            {prediction.model === "lightgbm"
              ? "Why the model leans this way (LightGBM)"
              : "Team ratings used by Dixon-Coles"}
          </h3>
          <ExplanatoryFeatureList model={prediction.model} items={ideals} />
          {prediction.model === "lightgbm" ? (
            <p className="explanatory-scale">
              Contributions are to the model&apos;s raw score, not percentage points.
            </p>
          ) : null}
        </>
      ) : null}

      <p className="prediction-footnote">
        Model: {prediction.model}. Trained on matches up to {formatIsoDate(cutoffDateOnly(prediction.training_data_cutoff))}.
      </p>
    </>
  );
}
