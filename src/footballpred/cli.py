"""Command-line entry point for data setup and inspection."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from footballpred.config import load_config
from footballpred.evaluate.report import evaluate_phase2
from footballpred.ingest.download import ingest_dataset
from footballpred.ingest.validate import write_summary, write_validation_report

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
    """Evaluate Phase 2 baselines and Elo using configured temporal seasons."""
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


if __name__ == "__main__":
    app()
