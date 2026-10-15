# Model card

## Model details

This project contains several forecasting models, used together:

- `ClassPriorModel`: a baseline that returns the training-season outcome rates.
- `BookmakerModel` / `bookmaker_probabilities`: no-vig probabilities derived from decimal betting odds.
- `EloModel`: a draw-aware Elo rating updated after each match.
- `DixonColesModel`: a time-decayed goal-scoring model producing a score matrix, 1X2, Over/Under 2.5, and BTTS probabilities.
- `LightGBM1X2` (`lgbm.py`): a multiclass gradient-boosting classifier for home win / draw / away win.
- `LightGBMBinary`: binary LightGBM classifiers for Over 2.5 and BTTS.
- Calibration wrappers: `TemperatureScaler`, `PlattScaler`, and an isotonic diagnostic (`IsotonicMulticlassScaler`, `IsotonicBinaryScaler`).
- `combine_probabilities` / `tune_ensemble_weight`: validation-weighted blend of Dixon-Coles and LightGBM probabilities.

Key library versions verified in the environment used for this card: Python 3.13.2 (package supports Python 3.11 or newer), scikit-learn 1.9.1, LightGBM 4.7.0, SciPy 1.18.1, NumPy 2.5.4, pandas 2.3.3. The model bundle is serialized with joblib and stored at `models/football-model.joblib` (gitignored because it contains large binary weights).

## Intended use

- Pre-match probability forecasts for English Premier League matches from historical tabular match data.
- Reproducible research and teaching files: causal features, walk-forward evaluation, calibration and interpretability.
- API or CLI use by the owner after running `footballpred train`.

## Out-of-scope uses

- Betting advice, staking plans or financial decisions of any kind.
- Live or in-play forecasting.
- Player-level, injury-aware, lineup-aware or xG-based forecasting.
- Any claim that these forecasts are more accurate than the closing bookmaker.

## Training data

- Source: football-data.co.uk CSVs, configured in `configs/default.yaml` (`data.source_url` points at `https://www.football-data.co.uk/mmz4281/{season}/{league}.csv`).
- League: E0 (Premier League) only.
- Seasons: 1415 through 2526 inclusive, 380 matches per season, 4560 match rows in total, 35 distinct teams.
- Date range: 2014-08-16 to 2026-05-24.
- Training uses all rows before the selected cutoff. The committed bundle's cutoff is 2026-05-24.

## Evaluation data

- Validation season: 2324 (380 matches).
- Test seasons: 2425 and 2526 (380 matches each).
- All model parameters (Elo K-factor and home advantage, Dixon-Coles xi, LightGBM trees and leaves, calibration choice, ensemble weight) are selected on the validation season only. Test seasons are reported separately.

## Split strategy

- `season_split` in `evaluate/splits.py` builds training, validation and test sets by season label.
- Training contains only complete seasons before validation. Validation is one season. Each test season is scored in its own expanding-window fold, refitting on all rows strictly before that season's first match date.
- Date-level guards reject overlapping or non-chronological partitions.

## Features

- The feature builder in `features/builder.py` produces 48 numeric pre-match features for each row.
- Features include home/away days rest, matches played this season, rolling form over 5 and 10 matches (points, goals for, goals against, shots, shots on target), overall and venue-specific form, pre-match Elo ratings and difference, and a bounded season-progress indicator.
- Shot columns are converted from optional source fields; earlier rows can be NaN until a team has history.
- Features are calculated in date order with a same-date snapshot, so the result of another match on the same day cannot enter a fixture's features.
- Bookmaker odds are **not** used as features in the main models. They are used only to build the separate `BookmakerModel` baseline and the binary bookmaker references for Over 2.5.

## Metrics and results

Numbers below were generated programmatically from `results/phase4-model-comparison.csv` and `results/walk-forward-test-results.csv`.

1X2 log loss on held-out test seasons (lower is better):

| model | season 2425 | season 2526 |
|---|---:|---:|
| bookmaker | 0.9664 | 1.0166 |
| lightgbm + Platt | 0.9934 | 1.0477 |
| lightgbm | 1.0020 | 1.0473 |
| dixon_coles | 1.0204 | 1.0562 |
| ensemble | 1.0096 | 1.0479 |
| elo | 1.0220 | 1.0434 |
| class_prior | 1.0812 | 1.0835 |

The difference `test log loss minus bookmaker log loss` with its 95% paired bootstrap interval is positive in every retained comparison, so the models are worse than the closing bookmaker on these seasons and the difference is not explained by sampling noise at the reported precision. Example: LightGBM + Platt on season 2425 has difference 0.0270 with interval 0.0059 to 0.0462.

Over 2.5 binary log loss on held-out seasons:

| model | season 2425 | season 2526 |
|---|---:|---:|
| bookmaker | 0.6764 | 0.6887 |
| dixon_coles | 0.6784 | 0.6844 |
| lightgbm | 0.6863 | 0.6902 |
| class_prior | 0.6865 | 0.6885 |

BTTS has no published bookmaker odds in this source, so only class prior, Dixon-Coles and LightGBM are compared there.

## Calibration findings

- `select_multiclass_calibrator` compares raw, temperature, Platt and isotonic fits on validation predictions. Isotonic has the lowest validation score but is deliberately not promoted as a serving choice because it is fit on the same calibration sample and overfits easily. Platt scaling is the selected 1X2 calibrator because it improves on raw without that risk.
- Binary calibration defaults to raw for Over 2.5 and BTTS; isotonic is reported as a diagnostic only.

## Known limitations and failure modes

- The models do not beat the closing bookmaker on the reported seasons. This is expected in a public-data, no-feature-engineering-beyond-form setting and is reported as a result rather than hidden.
- Only one league, one market family (1X2, Over 2.5, BTTS), no player data, no injuries, no expected-goals inputs.
- Class imbalance and small validation samples make isotonic calibration risky; hence it is treated as diagnostic.
- Some seasons have fewer columns (historical odds formats differ), so odds-based baselines are only scored where complete odds exist.
- The bundle loads the tuning CSV from a fixed relative path `results/lightgbm-validation-tuning.csv` if present; otherwise it uses the first configured candidate.
- Evaluation writes some artifacts to relative paths under the current working directory.

## Ethical considerations

- This is a forecasting research project, not a tool for gambling. Probability is not certainty; treat outputs as research signals.
- Published models can encourage gambling. Do not use them for financial decisions or staking plans.
- The data is public football results and derived statistics; check the source's terms of use before commercial reuse (see `docs/DATA_CARD.md`).

## Reproduction

    python -m pip install -e ".[dev]"
    footballpred download
    footballpred validate
    footballpred evaluate
    footballpred train
    footballpred --help

The full test suite is `python -m pytest -q` and lint is `ruff check src tests`.
