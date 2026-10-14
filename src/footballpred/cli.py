"""Command-line entry point for data setup and inspection."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from footballpred.config import load_config
from footballpred.evaluate.report import (
    evaluate_phase2,
    evaluate_phase3,
    evaluate_phase4,
)
from footballpred.features.builder import build_features
from footballpred.ingest.download import ingest_dataset
from footballpred.ingest.validate import write_summary, write_validation_report
from footballpred.pipeline import fit_deployable_models, save_bundle

app = typer.Typer(help="Reproducible football forecasting research pipeline.")


def _load_processed(path: Path) -> pd.DataFrame:
    if not path.exists():
        message = f"Processed dataset not found at {path}; run 'footballpred download' first"
        raise typer.BadParameter(message)
    return pd.read_csv(path, parse_dates=["date"])


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Prepare and inspect open football match data."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


@app.command()
def download(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    refresh: bool = typer.Option(False, "--refresh", help="Redownload files, bypassing the cache."),
) -> None:
    """Download/cache and normalize the configured seasons and leagues."""
    settings = load_config(config)
    dataset = ingest_dataset(settings.data, refresh=refresh)
    typer.echo(f"Saved {len(dataset):,} matches to {settings.data.processed_path}")


@app.command()
def validate(config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c")) -> None:
    """Validate the processed dataset and write its data-quality report and summary."""
    settings = load_config(config)
    dataset = _load_processed(settings.data.processed_path)
    report = write_validation_report(dataset, settings.data.validation_report_path)
    summary = write_summary(dataset, settings.data.stats_path)
    typer.echo(f"Validation report: {settings.data.validation_report_path}")
    typer.echo(f"Data summary: {settings.data.stats_path}")
    typer.echo(
        f"Rows: {report['rows']:,}; warnings: {len(report['warnings'])}; "
        f"draw rate: {summary['draw_rate']:.3f}"
    )


@app.command()
def evaluate(config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c")) -> None:
    """Evaluate baselines, Elo, and Dixon-Coles using configured temporal seasons."""
    settings = load_config(config)
    dataset = _load_processed(settings.data.processed_path)
    section = settings.model_dump()["evaluation"]
    results = evaluate_phase2(
        dataset,
        str(section["validation_season"]),
        [str(season) for season in section["test_seasons"]],
        section["elo"],
        settings.data.results_path,
    )
    typer.echo(f"Saved {len(results)} metric rows to {settings.data.results_path}")
    phase3_results = evaluate_phase3(
        dataset,
        str(section["validation_season"]),
        [str(season) for season in section["test_seasons"]],
        section,
        settings.data.results_path,
    )
    typer.echo(
        f"Saved {len(phase3_results)} market metric rows to "
        "results/phase3-market-comparison.csv"
    )
    phase4_results = evaluate_phase4(
        dataset,
        str(section["validation_season"]),
        [str(season) for season in section["test_seasons"]],
        section,
        settings.data.results_path,
    )
    typer.echo(
        f"Saved {len(phase4_results)} LightGBM metric rows to "
        "results/phase4-model-comparison.csv"
    )


@app.command()
def train(config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c")) -> None:
    """Fit final deployable models on all downloaded completed-season matches."""
    settings = load_config(config)
    matches = _load_processed(settings.data.processed_path)
    bundle = fit_deployable_models(matches, settings.evaluation)
    artifact = save_bundle(bundle, settings.data.model_path)
    typer.echo(
        f"Saved model bundle to {artifact}; training cutoff {bundle['training_cutoff']} "
        f"({len(bundle['teams'])} teams)"
    )


@app.command("build-features")
def build_features_command(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    output: Path = typer.Option(Path("data/processed/features.csv"), "--output", "-o"),
) -> None:
    """Build and save causal pre-match features for configured processed data."""
    settings = load_config(config)
    matches = _load_processed(settings.data.processed_path)
    features = build_features(matches)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    typer.echo(
        f"Saved {features.shape[0]} feature rows and {features.shape[1]} columns "
        f"to {output}"
    )


@app.command()
def predict(
    home: str = typer.Option(..., "--home", help="Canonical home team name."),
    away: str = typer.Option(..., "--away", help="Canonical away team name."),
    match_date: str = typer.Option(..., "--date", help="Fixture date in YYYY-MM-DD format."),
    model: str = typer.Option("lightgbm", "--model", help="lightgbm or dixon_coles."),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
) -> None:
    """Predict a fixture using the trained bundle (without fitting on request)."""
    settings = load_config(config)
    from footballpred.pipeline import load_bundle

    bundle = load_bundle(settings.data.model_path)
    teams = bundle["teams"]
    matches = bundle["history"]
    def resolve(name: str) -> str:
        found = next((team for team in teams if team.casefold() == name.strip().casefold()), None)
        if found is None:
            raise typer.BadParameter(
                f"Unknown team '{name}'. Use 'footballpred teams' to list teams."
            )
        return found

    home_team, away_team = resolve(home), resolve(away)
    from datetime import date as date_type

    fixture_date = date_type.fromisoformat(match_date)
    fixture = pd.DataFrame(
        [{"date": pd.Timestamp(fixture_date), "season": str(fixture_date.year),
          "home_team": home_team, "away_team": away_team, "home_goals": 0,
          "away_goals": 0, "result": "H"}]
    )
    if pd.Timestamp(fixture_date) <= pd.to_datetime(matches["date"]).max():
        raise typer.BadParameter(
            "Prediction date must be after the bundle training cutoff; use only future fixtures."
        )
    if model == "dixon_coles":
        market_output = bundle["models"]["dixon_coles"].predict_markets(fixture)[0]
        probabilities = market_output["1x2"]
        over = market_output["over_2_5"]
        btts = market_output["btts_yes"]
        scorelines = market_output["scorelines"]
        explanations = []
    elif model == "lightgbm":
        from footballpred.explain.text import explain_match

        feature_frame = build_features(pd.concat([matches, fixture], ignore_index=True, sort=False))
        feature_row = feature_frame.iloc[[-1]].reset_index(drop=True)
        lgbm_model = bundle["models"]["lightgbm"]
        probabilities = lgbm_model.predict_proba(feature_row)[0]
        over = float(bundle["models"]["over_2_5"].predict_proba(feature_row)[0])
        btts = float(bundle["models"]["btts"].predict_proba(feature_row)[0])
        dc_output = bundle["models"]["dixon_coles"].predict_markets(fixture)[0]
        scorelines = dc_output["scorelines"]
        explanations = explain_match(lgbm_model, feature_row, home_team, away_team)[
            "top_features"
        ]
    else:
        raise typer.BadParameter("model must be 'lightgbm' or 'dixon_coles'")
    payload = {
        "home_team": home_team,
        "away_team": away_team,
        "date": fixture_date.isoformat(),
        "model": model,
        "training_data_cutoff": bundle["training_cutoff"],
        "probabilities_1x2": {
            "home_win": float(probabilities[0]),
            "draw": float(probabilities[1]),
            "away_win": float(probabilities[2]),
        },
        "over_2_5": float(over),
        "under_2_5": float(1 - over),
        "btts_yes": float(btts),
        "btts_no": float(1 - btts),
        "most_likely_scorelines": scorelines,
        "explanatory_features": explanations,
    }
    import json

    typer.echo(json.dumps(payload, indent=2))


@app.command()
def teams(config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c")) -> None:
    """List canonical teams from the trained model bundle."""
    from footballpred.pipeline import load_bundle

    settings = load_config(config)
    bundle = load_bundle(settings.data.model_path)
    typer.echo("\n".join(bundle["teams"]))


if __name__ == "__main__":
    app()
