# Interview notes

## 60-second summary

Football Predictor is a Python pipeline that forecasts the outcome of a Premier League match before kickoff. It downloads and cleans public match data, builds features that only use information available before the match, fits a sequence of increasingly complex models (class prior, bookmaker odds, Elo, Dixon-Coles, gradient-boosting, calibration, ensemble), compares them against the closing bookmaker in walk-forward evaluation, and serves one fixture through a CLI and a FastAPI endpoint. The main honest result is that the models do not beat the bookmaker on the held-out seasons, which is the expected outcome for a public-data model without market or player information. The project is a research tool, not a betting tool, and the code is organised to prevent leakage rather than to chase accuracy.

## Questions an interviewer might ask

1. **What is data leakage in this context?**
   Leakage is using information that would not have been available at prediction time. Here it would mean using the final score or same-day results to predict that match. It is prevented in `features/builder.py`, which snapshots all matches on a date before appending their scores, and by tests such as `test_feature_pipeline_is_invariant_to_future_results`.

2. **How is the evaluation split designed?**
   `evaluate/splits.py` splits by season, not random rows. Training uses full seasons before the validation season, validation is one season, and each later season is a separate test fold. Date ranges are checked for overlap. This gives walk-forward numbers in `results/walk-forward-test-results.csv`.

3. **Why log loss and not accuracy?**
   Log loss is a proper scoring rule: it rewards both the chosen outcome and the confidence. Accuracy ignores confidence and can be maximised by a model that is always uncertain or only checks the argmax. The repository reports log loss, Brier score, RPS and accuracy together (`evaluate/metrics.py`).

4. **Did the model beat the bookmaker?**
   No. On season 2425, LightGBM with Platt calibration has log loss 0.9934 against the bookmaker's 0.9664; the paired difference is 0.0270 with a 95% bootstrap interval of 0.0059 to 0.0462. The same holds on season 2526. See `results/phase4-model-comparison.csv`. The positive difference and its interval say the bookmaker is genuinely better here.

5. **Why can a public-data model not beat the market?**
   The closing bookmaker price aggregates more information (lineups, injuries, market movement, late information) and a very large amount of historical signal. This model uses only published result history and simple form. Not beating the market with these inputs is the expected result, not a failure of the code. The honest claim is that the pipeline measures the gap, not that it closes it.

6. **What does calibration mean here?**
   Calibration means predicted probabilities match observed frequencies. Temperature scaling, Platt scaling and an isotonic diagnostic are compared in `models/calibration.py` and the validation scores are in `results/lightgbm-calibration-validation.csv`. Platt scaling was selected for 1X2 because it improved on raw; isotonic had the lowest score but was kept as a diagnostic because it overfits small calibration sets. Binary markets keep the raw forecast.

7. **Why walk-forward evaluation rather than one train/test split?**
   A single split can reward a model that happens to fit one season. Walk-forward re-fits on expanding history and reports each later season separately, which tests whether performance is stable across seasons. See `evaluate/report.py` and `walk-forward-test-results.csv`.

8. **What is the Dixon-Coles score matrix?**
   It models the home and away goal rate, adds a small correction for low scores, and tabulates the probability of every (home goals, away goals) pair up to `max_goals`. Derived markets (1X2, Over 2.5, BTTS) are summed from that matrix, so the main market and score markets come from one coherent model. Code is in `models/dixon_coles.py`.

9. **How are bookmaker odds converted to a forecast?**
   Decimal odds are turned into implied probabilities (`1 / odds`) and then divided by their row sum to remove the overround. Missing odds are handled per market; if no complete three-way triplet exists for a row, the baseline cannot be scored there. See `models/baselines.py`.

10. **What features are used, and deliberately not used?**
    48 pre-match features from `features/builder.py`: rest, season progress, rolling form (5 and 10 matches, overall and venue), and Elo. Bookmaker odds are **not** used as features; they are only the baseline. This keeps the comparison against the market honest.

11. **How is determinism guaranteed?**
    Fixed random seeds (LightGBM `random_state=2026` and others), ordered data processing, stable sorts, and tests that re-run fitting and compare outputs (`tests/test_determinism.py`). Evaluation bootstrap intervals use explicit RNG seeds.

12. **What would you do with more time or data?**
    Add expected goals and shot-quality inputs, Elo variants, a larger league set, player/injury data, a separate calibration period that is not also used for selection, and a proper temporal calibration split rather than the current validation predictions.

## What I would NOT claim

- Do not claim the model beats the market. The reported differences are positive in the bookmaker's favour.
- Do not collect or quote accuracy alone as if it summarises performance.
- Do not call probabilities certainties, and do not convert them into staking advice.
- Do not claim official endorsement of this code by the data provider.
- Do not describe the project as a commercial product; it is a reproducible research and learning repository.
- Do not inflate the owner's ability: the work is a single-domain forecasting pipeline with honest negative results, not a published system.

## Facts cited (files and results used)

- `results/phase4-model-comparison.csv`, `results/walk-forward-test-results.csv`
- `results/lightgbm-calibration-validation.csv`
- `src/footballpred/evaluate/splits.py`, `features/builder.py`, `models/*`, `evaluate/metrics.py`
- `tests/test_*` for the behaviour claims
