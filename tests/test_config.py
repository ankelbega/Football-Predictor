"""Validate configuration schema and fail-fast behavior."""


from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from footballpred.config import load_config


def write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def base_payload() -> dict:
    return {
        "data": {
            "source_url": "https://example.test/{league}_{season}.csv",
            "leagues": ["E0"],
            "seasons": ["2122"],
            "cache_dir": "data/raw",
            "processed_path": "data/processed/matches.csv",
            "validation_report_path": "results/data-validation.json",
            "stats_path": "results/data-summary.json",
        }
    }


def test_load_config_accepts_minimal_valid_configuration(tmp_path: Path) -> None:
    """Risk: valid minimal configurations must still load with defaults intact."""
    config = load_config(write_config(tmp_path, base_payload()))
    assert config.data.leagues == ["E0"]
    assert config.data.request_timeout_seconds == 30
    assert config.evaluation == {}


def test_load_config_rejects_unknown_data_keys(tmp_path: Path) -> None:
    """Risk: misspelled configuration keys must fail early rather than being ignored."""
    payload = base_payload()
    payload["data"]["leaguez"] = ["E0"]
    with pytest.raises(ValidationError):
        load_config(write_config(tmp_path, payload))


def test_load_config_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    """Risk: unknown top-level sections must not be silently accepted."""
    payload = base_payload()
    payload["evaluationn"] = {}
    with pytest.raises(ValidationError):
        load_config(write_config(tmp_path, payload))


def test_load_config_requires_non_empty_leagues_and_seasons(tmp_path: Path) -> None:
    """Risk: running ingest without configured league/season data should be impossible."""
    payload = base_payload()
    payload["data"]["leagues"] = []
    with pytest.raises(ValidationError):
        load_config(write_config(tmp_path, payload))
    payload = base_payload()
    payload["data"]["seasons"] = []
    with pytest.raises(ValidationError):
        load_config(write_config(tmp_path, payload))


def test_load_config_requires_positive_request_timeout(tmp_path: Path) -> None:
    """Risk: non-positive HTTP timeouts could hang ingestion indefinitely."""
    payload = base_payload()
    payload["data"]["request_timeout_seconds"] = 0
    with pytest.raises(ValidationError):
        load_config(write_config(tmp_path, payload))


def test_load_config_converts_paths(tmp_path: Path) -> None:
    """Risk: configured string paths must be usable as Path objects downstream."""
    config = load_config(write_config(tmp_path, base_payload()))
    assert isinstance(config.data.cache_dir, Path)
    assert isinstance(config.data.processed_path, Path)
