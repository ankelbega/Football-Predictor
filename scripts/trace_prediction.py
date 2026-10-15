"""Print one concrete match through the pipeline stages.

Run from the repository root with:

    python scripts/trace_prediction.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from footballpred.config import load_config
from footballpred.features.builder import build_features
from footballpred.ingest.clean import clean_source_file
from footballpred.models.dixon_coles import DixonColesConfig, DixonColesModel
from footballpred.models.lgbm import LightGBM1X2, LightGBMConfig


def main() -> None:
    settings = load_config(ROOT / "configs" / "default.yaml")
    processed_path = ROOT / settings.data.processed_path
    matches = pd.read_csv(processed_path, parse_dates=["date"])

    selected = matches.loc[matches["season"].astype(str).eq("2425")].sort_values("date").iloc[0]
    selected_date = pd.Timestamp(selected["date"])
    training = matches.loc[matches["date"] < selected_date].copy()

    raw_path = ROOT / settings.data.cache_dir / "E0_2425.csv"
    raw_source = pd.read_csv(raw_path, encoding="utf-8-sig", on_bad_lines="skip")
    source_match = raw_source[
        raw_source["HomeTeam"].astype(str).str.strip().eq(str(selected["home_team"]))
        & raw_source["AwayTeam"].astype(str).str.strip().eq(str(selected["away_team"]))
    ]
    raw_row = source_match.iloc[0] if not source_match.empty else None

    fixture = selected.to_frame().T.copy()
    fixture_frame = pd.concat([training, fixture], ignore_index=True, sort=False)
    features = build_features(fixture_frame)
    feature_row = features.iloc[[-1]].reset_index(drop=True)

    dixon_config = DixonColesConfig(xi=0.0018, max_goals=10, maxiter=200)
    dixon_model = DixonColesModel(dixon_config).fit(training)
    dixon_markets = dixon_model.predict_markets(fixture)[0]

    lightgbm_config = LightGBMConfig(
        n_estimators=120,
        num_leaves=15,
        learning_rate=0.035,
        min_child_samples=25,
        reg_lambda=2.0,
        random_state=2026,
    )
    lightgbm_model = LightGBM1X2(lightgbm_config).fit(features.iloc[:-1], training)
    lightgbm_probabilities = lightgbm_model.predict_proba(feature_row)[0]

    print("Selected match")
    print(f"  season={selected['season']}, date={selected['date'].date()}, "
          f"home={selected['home_team']}, away={selected['away_team']}, "
          f"score={selected['home_goals']}-{selected['away_goals']}, result={selected['result']}")
    print(f"Training rows available before the match: {len(training)}")
    print()

    print("Raw source row from E0_2425.csv")
    if raw_row is None:
        print("  none found")
    else:
        print(raw_row.to_string())
    print()

    print("Cleaned row from data/processed/matches.csv")
    print(selected.to_string())
    print()

    print("Leakage-free features for this match")
    print(feature_row.to_string(index=False))
    print()

    print("Pre-match Elo ratings from the feature row")
    print(f"  elo_home={feature_row.loc[0, 'elo_home']:.2f}")
    print(f"  elo_away={feature_row.loc[0, 'elo_away']:.2f}")
    print(f"  elo_difference={feature_row.loc[0, 'elo_difference']:.2f}")
    print(f"  home_days_rest={feature_row.loc[0, 'home_days_rest']}")
    print(f"  away_days_rest={feature_row.loc[0, 'away_days_rest']}")
    print()

    print("Dixon-Coles result for this match")
    print(f"  1X2 probabilities: {[round(float(value), 4) for value in dixon_markets['1x2']]}")
    print(f"  over_2.5 probability: {dixon_markets['over_2_5']:.4f}")
    print(f"  btts probability: {dixon_markets['btts_yes']:.4f}")
    print("  top scorelines:")
    for row in dixon_markets["scorelines"][:5]:
        print(f"    {row['home_goals']}-{row['away_goals']}: {row['probability']:.4f}")
    print()

    print("LightGBM result for this match")
    print(f"  1X2 probabilities: {[round(float(value), 4) for value in lightgbm_probabilities]}")
    print()

    bundle_path = ROOT / "models" / "football-model.joblib"
    if bundle_path.exists():
        bundle = joblib.load(bundle_path)
        print(f"Serving bundle cutoff: {bundle['training_cutoff']}")
    else:
        print("Serving bundle cutoff: not built")


if __name__ == "__main__":
    main()
