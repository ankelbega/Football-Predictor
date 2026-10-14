"""FastAPI serving layer; model bundles are loaded once at app construction."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from footballpred.config import load_config
from footballpred.explain.text import explain_match
from footballpred.features.builder import build_features
from footballpred.pipeline import load_bundle


class MarketProbabilities(BaseModel):
    home_win: float = Field(ge=0, le=1)
    draw: float = Field(ge=0, le=1)
    away_win: float = Field(ge=0, le=1)


class Scoreline(BaseModel):
    home_goals: int = Field(ge=0)
    away_goals: int = Field(ge=0)
    probability: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    home_team: str
    away_team: str
    match_date: date
    model: str
    training_data_cutoff: str
    probabilities_1x2: MarketProbabilities
    over_2_5: float = Field(ge=0, le=1)
    under_2_5: float = Field(ge=0, le=1)
    btts_yes: float = Field(ge=0, le=1)
    btts_no: float = Field(ge=0, le=1)
    most_likely_scorelines: list[Scoreline]
    explanatory_features: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    training_data_cutoff: str


def create_app(bundle_path: str | Path | None = None) -> FastAPI:
    """Create API app and load models one time at startup/construction."""
    if bundle_path is None:
        settings = load_config()
        model_path = settings.data.model_path
    else:
        model_path = Path(bundle_path)
    bundle = load_bundle(model_path)
    models = bundle["models"]
    teams = list(bundle["teams"])
    training_cutoff = str(bundle["training_cutoff"])
    app = FastAPI(
        title="Football Predictor API",
        description="Educational match probability forecasts; not betting advice.",
        version="0.1.0",
    )
    app.state.bundle = bundle

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok", model_loaded=True, training_data_cutoff=training_cutoff
        )

    @app.get("/teams", response_model=list[str])
    def list_teams() -> list[str]:
        return teams

    @app.get("/predict", response_model=PredictionResponse)
    def predict(
        home: str = Query(min_length=1),
        away: str = Query(min_length=1),
        model: Literal["lightgbm", "dixon_coles"] = "lightgbm",
        match_date: date = Query(..., alias="date"),
    ) -> PredictionResponse:
        home_name = _resolve_team(home, teams)
        away_name = _resolve_team(away, teams)
        if pd.Timestamp(match_date) <= pd.to_datetime(bundle["history"]["date"]).max():
            raise HTTPException(
                status_code=422,
                detail="Prediction date must be after the model training-data cutoff.",
            )
        if home_name == away_name:
            raise HTTPException(status_code=422, detail="Home and away teams must be different")
        if model not in models:
            raise HTTPException(status_code=400, detail=f"Model '{model}' is not available")
        # The serve-time feature history is built from stored match history. The initial bundle
        # stores the complete training data, so the queried fixture is appended as a label-free row.
        history = bundle["history"].copy()
        fixture = _fixture_stub(match_date, home_name, away_name)
        combined = pd.concat([history, fixture], ignore_index=True, sort=False)
        feature_rows = build_features(combined)
        fixture_features = feature_rows.iloc[[-1]].reset_index(drop=True)
        if model == "lightgbm":
            output_model = models["lightgbm"]
            probabilities = output_model.predict_proba(fixture_features)[0]
            market_models = {key: models[key] for key in ("over_2_5", "btts")}
            over = float(market_models["over_2_5"].predict_proba(fixture_features)[0])
            btts = float(market_models["btts"].predict_proba(fixture_features)[0])
            explanation = explain_match(output_model, fixture_features, home_name, away_name)
            explanatory_features = explanation["top_features"]
            market_label = "lightgbm"
            dc_markets = models["dixon_coles"].predict_markets(fixture)
            scorelines = dc_markets[0]["scorelines"]
        else:
            dc_model = models["dixon_coles"]
            dc_markets = dc_model.predict_markets(fixture)
            market = dc_markets[0]
            probabilities = market["1x2"]
            over = float(market["over_2_5"])
            btts = float(market["btts_yes"])
            scorelines = market["scorelines"]
            explanatory_features = [
                {"feature": "home_attack", "value": dc_model.attack_.get(home_name, 0.0)},
                {"feature": "away_attack", "value": dc_model.attack_.get(away_name, 0.0)},
                {"feature": "home_defence", "value": dc_model.defence_.get(home_name, 0.0)},
                {"feature": "away_defence", "value": dc_model.defence_.get(away_name, 0.0)},
                {"feature": "home_advantage_log_rate", "value": dc_model.home_advantage_},
            ]
            market_label = "dixon_coles"
        return PredictionResponse(
            home_team=home_name,
            away_team=away_name,
            match_date=match_date,
            model=market_label,
            training_data_cutoff=training_cutoff,
            probabilities_1x2=MarketProbabilities(
                home_win=float(probabilities[0]),
                draw=float(probabilities[1]),
                away_win=float(probabilities[2]),
            ),
            over_2_5=over,
            under_2_5=1.0 - over,
            btts_yes=btts,
            btts_no=1.0 - btts,
            most_likely_scorelines=[Scoreline(**row) for row in scorelines[:5]],
            explanatory_features=explanatory_features[:5],
        )

    return app


def _resolve_team(name: str, teams: list[str]) -> str:
    normalized = " ".join(name.strip().split()).casefold()
    candidates = {team.casefold(): team for team in teams}
    if normalized not in candidates:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown team '{name}'. Use GET /teams to list available teams.",
        )
    return candidates[normalized]


def _fixture_stub(match_date: date, home: str, away: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "date": pd.Timestamp(match_date),
            "season": str(match_date.year),
            "home_team": home,
            "away_team": away,
            "home_goals": 0,
            "away_goals": 0,
            "result": "H",
        }]
    )


def _load_default_app() -> FastAPI:
    """Load default model artifact when launching `uvicorn footballpred.api.app:app`."""
    return create_app()


try:
    app: FastAPI | None = _load_default_app()
except (FileNotFoundError, ValueError):
    app = None
