"""Download and cache public football-data.co.uk match files."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from footballpred.config import DataConfig
from footballpred.ingest.clean import clean_source_file

LOGGER = logging.getLogger(__name__)


def download_source(config: DataConfig, refresh: bool = False) -> list[Path]:
    """Download configured CSVs, retaining successful files in the local cache."""
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for league in config.leagues:
        for season in config.seasons:
            path = config.cache_dir / f"{league}_{season}.csv"
            if path.exists() and not refresh:
                LOGGER.info("Using cached %s", path)
                paths.append(path)
                continue
            url = config.source_url.format(season=season, league=league)
            try:
                request = Request(
                    url, headers={"User-Agent": "football-predictor/0.1 research project"}
                )
                with urlopen(request, timeout=config.request_timeout_seconds) as response:
                    content = response.read()
                if not content.strip():
                    raise ValueError("source returned an empty response")
                path.write_bytes(content)
                LOGGER.info("Downloaded %s", url)
                paths.append(path)
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                LOGGER.warning("Could not download %s (%s): %s", url, type(exc).__name__, exc)
    if not paths:
        raise RuntimeError("No source files were downloaded or found in cache")
    return paths


def ingest_dataset(config: DataConfig, refresh: bool = False) -> pd.DataFrame:
    """Download/cache configured sources, normalize, combine, and persist matches."""
    files = download_source(config, refresh=refresh)
    frames: list[pd.DataFrame] = []
    for path in files:
        league, season = path.stem.rsplit("_", maxsplit=1)
        try:
            frames.append(clean_source_file(path, league, season, config.team_aliases))
        except (UnicodeDecodeError, ValueError) as exc:
            LOGGER.warning("Skipping unusable source %s: %s", path, exc)
    if not frames:
        raise RuntimeError(
            "No usable match rows found; inspect source files and configured columns"
        )
    dataset = pd.concat(frames, ignore_index=True)
    config.processed_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(config.processed_path, index=False, date_format="%Y-%m-%d")
    return dataset
