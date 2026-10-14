"""Load the YAML application configuration into typed settings.

This setup module is used by the CLI and API before ingest, evaluation, model training, or
serving. Pydantic rejects unknown data keys so misspelled configuration fails early.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    """Paths, source templates, aliases and timeouts used by data ingestion."""

    model_config = ConfigDict(extra="forbid")

    source_url: str
    leagues: list[str] = Field(min_length=1)
    seasons: list[str] = Field(min_length=1)
    cache_dir: Path
    processed_path: Path
    validation_report_path: Path
    stats_path: Path
    results_path: Path = Path("results/model-comparison.csv")
    model_path: Path = Path("models/football-model.joblib")
    request_timeout_seconds: int = Field(default=30, gt=0)
    team_aliases: dict[str, str] = Field(default_factory=dict)


class AppConfig(BaseModel):
    """Top-level validated data and evaluation configuration."""

    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    evaluation: dict[str, Any] = Field(default_factory=dict)


def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    """Read a YAML file and validate its keys and values.

    Args:
        path: Configuration path; defaults to the repository's standard YAML file.

    Returns:
        Typed AppConfig object with paths converted to Path values.
    """
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        payload: Any = yaml.safe_load(stream)
    return AppConfig.model_validate(payload)
