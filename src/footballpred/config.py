"""Configuration loading and validation."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    """Settings for source files and dataset outputs."""

    model_config = ConfigDict(extra="forbid")

    source_url: str
    leagues: list[str] = Field(min_length=1)
    seasons: list[str] = Field(min_length=1)
    cache_dir: Path
    processed_path: Path
    validation_report_path: Path
    stats_path: Path
    results_path: Path = Path("results/model-comparison.csv")
    request_timeout_seconds: int = Field(default=30, gt=0)
    team_aliases: dict[str, str] = Field(default_factory=dict)


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    evaluation: dict[str, Any] = Field(default_factory=dict)


def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    """Read and validate a YAML configuration file."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        payload: Any = yaml.safe_load(stream)
    return AppConfig.model_validate(payload)
