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

The project now includes time-aware baselines and models, leakage-free features, proper scoring rules, calibration, explainability, walk-forward evaluation, and CLI/API functionality. This is an educational project, not betting advice.

## CLI and API

```bash
footballpred download
footballpred validate
footballpred build-features
footballpred evaluate
footballpred train
footballpred predict --home "Arsenal" --away "Chelsea" --date 2025-03-01
uvicorn footballpred.api.app:app --host 127.0.0.1 --port 8000
```

The API exposes `/health`, `/teams`, and `/predict`; interactive OpenAPI docs are at `/docs`.
The service loads `models/football-model.joblib` once at startup and does not retrain per request.
That artifact is gitignored and is generated with `footballpred train`.

Docker requires the processed dataset and a locally trained model artifact:

```bash
docker build -t football-predictor .
docker run --rm -p 8000:8000 -v "${PWD}/models:/app/models" football-predictor
```

`make all` reproduces the current pipeline from download through evaluation and training.
