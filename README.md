# football-predictor

An educational, reproducible machine-learning project for forecasting **football match outcome probabilities** (home win, draw, away win), goals markets, and both-teams-to-score (BTTS). It compares its forecasts with no-vig bookmaker probabilities using chronological evaluation and proper scoring rules. The goal is to demonstrate sound forecasting methodology—including leakage prevention, time-aware testing, and honest reporting—not to encourage betting or promise an advantage over the market.

> **Status:** Phases 1–6 are implemented. Results below are generated from the checked-in evaluation commands on football-data.co.uk data for Premier League seasons 2014/15–2025/26. The tested future folds are 2024/25 and 2025/26. They show that the bookmaker is a strong benchmark and that model results are mixed; the project makes no claim of consistent market superiority.

## What this is—and is not

- **Is:** a local research and portfolio project that produces probability distributions, interpretable ratings, feature explanations, and time-aware evaluation tables.
- **Is not:** a betting product, financial advice, a guarantee of match outcomes, or evidence that a model can beat bookmaker prices after real-world execution costs.
- No bets are placed. The optional historical-simulation work is not included.
- A model score is an estimate from historical data. It does not incorporate injuries, confirmed lineups, transfers, tactical changes, or live information.

## Quickstart

Requires Python 3.11+ and `pip`. From a clone of this repository:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
footballpred download
footballpred validate
footballpred evaluate
footballpred train
```

The download is cached under `data/raw/`; normalized data is stored in `data/processed/`. Both are gitignored. Subsequent `footballpred download` runs use the cache; use `footballpred download --refresh` to fetch again. Configuration is in `configs/default.yaml` and CLI commands accept `--config PATH`.

`make all` runs download, validation, feature generation, evaluation, and training end to end. It needs access to the free source website on first download. To run only tests and lint:

```bash
pytest -q
ruff check src tests
```

## Results

The default dataset contained **4,560 completed Premier League matches**, from 2014-08-16 through 2026-05-24. The observed home-win, draw, and away-win rates were 44.4%, 23.8%, and 31.8%, respectively. Mean goals were 1.54 at home and 1.26 away. The quality report found zero duplicate match keys, impossible negative scores, invalid result codes, and missing required match fields. It flagged some suspicious odds values (at or below 1.0/non-finite); odds are validated per fixture and invalid prices are not used as market probabilities.

### Held-out 1X2 log loss

Lower is better. Each test season is forecast in an expanding-window fold, trained only on matches strictly before the test season. Odds are proportionally normalized to remove overround, preferring Pinnacle closing prices when complete, then available alternatives. The `difference` column is model log loss minus bookmaker log loss; positive means the model scored worse. Confidence intervals are paired 95% bootstrap intervals over test-match log-loss differences. They quantify sampling uncertainty within these seasons, not future-market uncertainty.

| Test season | Model | Log loss | Difference vs bookmaker (95% paired bootstrap CI) |
|---|---|---:|---:|
| 2024/25 | Bookmaker | 0.9664 | — |
| 2024/25 | LightGBM | 1.0020 | +0.0356 (+0.0135, +0.0557) |
| 2024/25 | LightGBM + Platt | 0.9934 | +0.0270 (+0.0059, +0.0462) |
| 2024/25 | Dixon–Coles | 1.0204 | +0.0540 (+0.0267, +0.0805) |
| 2024/25 | Elo | 1.0220 | +0.0556 (+0.0272, +0.0826) |
| 2024/25 | Class prior | 1.0812 | +0.1148 (+0.0733, +0.1544) |
| 2025/26 | Bookmaker | 1.0166 | — |
| 2025/26 | LightGBM | 1.0473 | +0.0307 (+0.0093, +0.0539) |
| 2025/26 | LightGBM + Platt | 1.0477 | +0.0311 (+0.0099, +0.0522) |
| 2025/26 | Dixon–Coles | 1.0562 | +0.0397 (+0.0196, +0.0605) |
| 2025/26 | Elo | 1.0434 | +0.0268 (-0.0015, +0.0530) |
| 2025/26 | Class prior | 1.0835 | +0.0669 (+0.0323, +0.1001) |

Across both test seasons, the bookmaker baseline had lower 1X2 log loss than LightGBM, calibrated LightGBM, Dixon–Coles, and the validation-weighted ensemble. Elo was close in 2025/26, but not consistently better. This is an honest negative result: the implemented models have not demonstrated reliable market superiority.

### Other markets

For Over 2.5 goals, the de-vigged Bet365 baseline had log loss 0.6764 in 2024/25 and 0.6887 in 2025/26. Dixon–Coles recorded 0.6784 and 0.6844, respectively; paired 95% bootstrap intervals for the model-minus-market loss differences include zero in both folds. Results therefore do not establish a reliable difference from the market. football-data.co.uk provides no independent BTTS odds series in the retained files, so BTTS is compared with class priors and not misleadingly compared to O/U prices.

### Reproduce results and figures

The command below regenerates the CSV/Markdown score tables, validation tuning tables, reliability figure, SHAP importance figure, sample explanations, and Dixon–Coles ratings:

```bash
footballpred evaluate
```

Checked-in outputs include:

- `results/model-comparison.csv` and `.md` — Phase 2 baselines and Elo.
- `results/phase3-market-comparison.csv` and `.md` — Dixon–Coles plus 1X2, Over 2.5, and BTTS.
- `results/phase4-model-comparison.csv` and `.md` — baselines, LightGBM, calibration, and ensemble across the markets.
- `results/walk-forward-test-results.csv` — compact season-wise Phase 2 holdout table.
- `results/elo-validation-tuning.csv`, `results/dixon-coles-validation-tuning.csv`, `results/lightgbm-validation-tuning.csv`, and `results/lightgbm-calibration-validation.csv` — validation-only selection records.
- `docs/reliability-validation.png`, `docs/shap-global-importance.png`, `docs/sample-explanations.md`, and `docs/dixon-coles-team-strengths.md`.
- `docs/walk-forward-log-loss.png` — held-out season log-loss plot, with bookmaker reference.

## Data source and licensing notes

Results and bookmaker prices are sourced from public CSV downloads at [football-data.co.uk](https://www.football-data.co.uk/). The URL template is configurable; the default league is the English Premier League (`E0`), and seasons are configured in `configs/default.yaml`. The current default ends with the completed 2025/26 season. Source headers vary across years; required columns are checked when ingesting, and optional odds/shots fields are retained if present. Team aliases are configured in YAML.

The software is licensed under the MIT License (`LICENSE`). This does **not** relicense football-data.co.uk data. Review the data provider’s current terms, attribution rules, and redistribution conditions before sharing downloaded data. Raw and processed source data are not committed.

## Methodology

### Time splits and walk-forward evaluation

- Default: train on seasons before 2023/24, use 2023/24 for validation/tuning, then report held-out 2024/25 and 2025/26 season folds.
- Test folds use an expanding window: each season’s models are fit only on data dated before that season’s first test fixture. No random split is used.
- Validation tunes Elo, Dixon–Coles time decay, LightGBM hyperparameters, and calibration choices. Test scores do not determine settings.
- Same-day fixtures do not update each other’s features. A match’s features are assembled from earlier dates only; its outcome is appended after its date’s feature snapshot.

### Models and probabilities

- **Class prior:** training-set outcome frequencies.
- **Bookmaker:** closing odds when available, otherwise a complete fallback price triplet; inverse decimal odds are proportionally normalized to remove overround.
- **Elo:** zero-sum rating updates, home advantage, and a draw-aware probability mapping fit on training data.
- **Dixon–Coles:** time-decayed Poisson score model fit by maximum likelihood. Attack and defence log strengths are centered across training teams for identifiability. Unknown teams receive neutral strength. The scoreline matrix is truncated at ten goals per side and renormalized; 1X2, O/U 2.5, and BTTS probabilities are all derived from that same matrix.
- **LightGBM:** multiclass 1X2 plus separate binary Over 2.5 and BTTS classifiers. Inputs include pre-match Elo, rolling last-5/last-10 overall and venue-specific form, rest, seasonal match counts/progress, and available historical shots. Bookmaker odds are excluded from model features.
- **Calibration:** raw probabilities are compared with temperature and one-vs-rest Platt; isotonic is retained as a diagnostic but not selected on its own in-sample score because that encourages overfit. Parametric calibration is used only if validation log loss improves.
- **Ensemble:** optional convex combination of Dixon–Coles and LightGBM; weight is selected on validation 1X2 log loss. On the test folds here, it did not improve on bookmaker odds.

Primary metrics are log loss, multiclass Brier score, and ranked probability score (RPS). Accuracy is secondary. Calibration uses class reliability plots and top-label ECE. Binary markets use binary log loss/Brier. Bootstrap intervals compare paired per-match log losses against the market where corresponding odds are available.

### Cold starts and limits

New teams receive the Elo starting rating and neutral Dixon–Coles strengths; early rolling form is unavailable and is imputed by the LightGBM pipeline. This is simple and transparent, but promoted-team strength can be poorly estimated. Date-only data cannot distinguish same-day kickoff order. Match-level model updates happen at the season evaluation fold boundary rather than being retrained before every individual fixture. The default data are one league; cross-league team strength is not inferred.

Bookmaker prices may not be truly synchronized closing prices for every match; fallback prices vary by row and source. Overround normalization is proportional rather than a claim about the bookmaker’s underlying probability. Results cover two test seasons, so intervals and rankings remain uncertain. The backtest omits factors such as price availability, transaction costs, limits, and market movement; no wagering recommendation is made.

## Explainability

`docs/shap-global-importance.png` gives mean absolute SHAP contributions for the LightGBM class raw scores. `docs/sample-explanations.md` provides three examples. Contributions are on the model raw-score scale—not percentage-point changes in the normalized probability. Dixon–Coles attack/defence strengths and shared home-advantage parameter are saved in `results/dixon-coles-team-strengths.csv` and summarized in `docs/dixon-coles-team-strengths.md`.

## CLI and API

```bash
footballpred download
footballpred validate
footballpred build-features
footballpred evaluate
footballpred train
footballpred teams
footballpred predict --home "Arsenal" --away "Chelsea" --date 2026-10-15
uvicorn footballpred.api.app:app --host 127.0.0.1 --port 8000
```

The API has `GET /health`, `GET /teams`, and `GET /predict?home=...&away=...&model=...&date=...`; interactive OpenAPI docs are at `http://127.0.0.1:8000/docs`. Train first. The API loads `models/football-model.joblib` once and does not retrain per request. Predictions dated at or before the training cutoff are rejected to avoid historical leakage. The artifact is gitignored and is produced locally by `footballpred train`.

Docker is optional. When Docker Desktop/WSL is available, build and run with a locally trained bundle:

```bash
docker build -t football-predictor .
docker run --rm -p 8000:8000 -v "${PWD}/models:/app/models" football-predictor
```

## Project structure

```text
configs/                  YAML data/model/evaluation configuration
data/                     ignored source cache and processed dataset
docs/                     committed figures and explanatory examples
results/                  committed, code-generated metric tables
src/footballpred/ingest/  downloading, cleaning, normalization, validation
src/footballpred/features/ causal Elo and rolling form feature builder
src/footballpred/models/  baselines, Elo, Dixon–Coles, LightGBM, calibration, ensemble
src/footballpred/evaluate/ splits, proper metrics, plots, walk-forward reports
src/footballpred/explain/ SHAP and plain-language summaries
src/footballpred/api/     FastAPI endpoints and response schemas
tests/                    unit and API tests
```

## Future work

Additional leagues through adapters; more robust promoted-team priors; xG, injuries, and lineups where licensed/available; true per-matchday refitting; temporal calibration subfolds; additional bookmaker closing-price audits; richer API explanations; and a small browser UI.

## Disclaimer

This is an educational software engineering and forecasting project. It is **not betting advice**. Probability estimates are uncertain, historical performance does not guarantee future performance, and the project does not claim to beat bookmakers.
