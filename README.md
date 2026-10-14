# football-predictor

An educational, reproducible project for forecasting football match outcome probabilities and evaluating them honestly against bookmaker probabilities. The current development phase builds and validates the source-data pipeline; modeling and measured results will be added in later phases.

## Current status

Phase 1 implements cached downloads of football-data.co.uk CSVs, stable team labels, schema-aware cleaning, a data-quality report, and basic descriptive statistics. No model performance claims are made yet.

## Quickstart

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
footballpred download
footballpred validate
```

Use `--config path/to/config.yaml` to select another configuration, and `footballpred download --refresh` to bypass cached downloads.

## Data source and caveats

Match results and bookmaker odds come from the publicly accessible CSVs at [football-data.co.uk](https://www.football-data.co.uk/). The project downloads only configured files and caches them under `data/raw/`; processed data is written under `data/processed/`. Data files are excluded from version control. Check the source's current terms and attribution requirements before redistribution; availability does not imply that source data has been relicensed by this project.

The source schema varies between seasons. The ingestion code validates required columns at runtime and retains available odds fields. Validation highlights duplicate match keys, missing essentials, implausible scores/results, and suspicious decimal odds; it does not silently repair questionable source data.

## Project status

The remaining phases add time-aware baselines and models, leakage-free features, proper scoring rules, calibration, explainability, walk-forward evaluation, CLI/API functionality, and results generated from reproducible code. This is an educational project, not betting advice.
