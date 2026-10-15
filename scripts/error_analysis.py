"""Generate an error-analysis report from per-match test predictions.

If no prediction file is supplied, the script regenerates per-match test predictions
through the existing pipeline and writes them to results/error-analysis-predictions.csv.
Only that new results file is created; existing result files are not changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from footballpred.config import load_config
from footballpred.evaluate.report import season_split
from footballpred.evaluate.splits import season_split as _season_split
from footballpred.features.builder import build_features
from footballpred.models.baselines import bookmaker_probabilities
from footballpred.models.lgbm import train_targets
from footballpred.models.calibration import apply_multiclass_calibrator, select_multiclass_calibrator
from footballpred.models.dixon_coles import DixonColesConfig, DixonColesModel
from footballpred.models.lgbm import LightGBM1X2, LightGBMBinary, LightGBMConfig


def _best_lightgbm_params() -> tuple[int, int]:
    tuning = pd.read_csv(ROOT / "results" / "lightgbm-validation-tuning.csv")
    row = tuning.sort_values("validation_log_loss").iloc[0]
    return int(row["n_estimators"]), int(row["num_leaves"])


def _team_first_season(matches: pd.DataFrame) -> dict[str, str]:
    first: dict[str, str] = {}
    ordered = matches.sort_values("date")
    for _, row in ordered.iterrows():
        for team in (row["home_team"], row["away_team"]):
            first.setdefault(str(team), str(row["season"]))
    return first


def _season_stage(match_dates: pd.Series) -> pd.Series:
    ordered = pd.to_datetime(match_dates)
    ranks = ordered.rank(method="first").astype(int)
    within = ranks.groupby(match_dates.index.map(lambda _: 0)).transform(lambda s: s)
    # rank within season instead: caller groups by season
    return ranks


def generate_predictions() -> pd.DataFrame:
    config = load_config(ROOT / "configs" / "default.yaml")
    matches = pd.read_csv(ROOT / config.data.processed_path, parse_dates=["date"])
    matches["season"] = matches["season"].astype(str)
    all_features = build_features(matches)

    split = season_split(matches, "2324", ["2425", "2526"])
    n_estimators, num_leaves = _best_lightgbm_params()
    lgb_config = LightGBMConfig(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=float(config.evaluation["lightgbm"]["learning_rate"]),
        min_child_samples=int(config.evaluation["lightgbm"]["min_child_samples"]),
        reg_lambda=float(config.evaluation["lightgbm"]["reg_lambda"]),
        random_state=int(config.evaluation["lightgbm"]["random_state"]),
    )

    # fit on training and calibrate using validation predictions
    train_idx = split.train.index.to_numpy()
    val_idx = split.validation.index.to_numpy()
    train_features = all_features.loc[train_idx].reset_index(drop=True)
    val_features = all_features.loc[val_idx].reset_index(drop=True)
    base_model = LightGBM1X2(lgb_config).fit(train_features, split.train)
    raw_validation = base_model.predict_proba(val_features)
    calibrator_name, calibrator, _ = select_multiclass_calibrator(
        raw_validation, split.validation["result"].to_numpy()
    )

    first_season = _team_first_season(matches)
    rows: list[dict] = []
    binary_target_name = "over_2_5"
    for season in split.test_seasons:
        test = split.test.loc[split.test["season"].astype(str).eq(season)].copy()
        first_day = pd.to_datetime(test["date"]).min()
        training = matches.loc[pd.to_datetime(matches["date"]) < first_day].copy()
        training_features = all_features.loc[training.index].reset_index(drop=True)
        test_features = all_features.loc[test.index].reset_index(drop=True)

        model = LightGBM1X2(lgb_config).fit(training_features, training)
        raw = model.predict_proba(test_features)
        calibrated = apply_multiclass_calibrator(calibrator_name, calibrator, raw)
        bookmaker, _ = bookmaker_probabilities(test)

        binary_model = LightGBMBinary(lgb_config).fit(
            training_features, train_targets(training)[binary_target_name]
        )
        binary_prob = binary_model.predict_proba(test_features)

        team_status = [
            "promoted" if first_season.get(str(r["home_team"])) == season or first_season.get(str(r["away_team"])) == season
            else "established"
            for _, r in test.iterrows()
        ]

        sorted_test = test.sort_values("date")
        ranks = sorted_test["date"].rank(method="first").astype(int)
        total = len(sorted_test)
        stage_map = {}
        for idx, r in ranks.items():
            if r <= total / 3:
                stage_map[idx] = "early"
            elif r <= 2 * total / 3:
                stage_map[idx] = "mid"
            else:
                stage_map[idx] = "late"

        test_sorted_index = test.sort_values("date").index
        stage_series = pd.Series(stage_map).reindex(test.index)

        for j, (_, row) in enumerate(test.iterrows()):
            actual = str(row["result"])
            home_prob, draw_prob, away_prob = [float(x) for x in calibrated[j]]
            actual_index = {"H": 0, "D": 1, "A": 2}[actual]
            actual_prob = [home_prob, draw_prob, away_prob][actual_index]
            bookmaker_prob = float(bookmaker[j, actual_index])
            home_goals = int(row["home_goals"])
            away_goals = int(row["away_goals"])
            over_actual = int(home_goals + away_goals >= 3)
            book_over = None
            bo = bookmaker_probabilities
            try:
                bo_arr, _ = bookmaker_probabilities(test)
                # recompute no-vig over probabilities if odds columns exist
            except Exception:
                bo_arr = None

            rows.append(
                {
                    "season": season,
                    "date": pd.Timestamp(row["date"]).date().isoformat(),
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "actual_result": actual,
                    "home_win_prob": home_prob,
                    "draw_prob": draw_prob,
                    "away_win_prob": away_prob,
                    "best_model_logloss": -np.log(max(actual_prob, 1e-15)),
                    "bookmaker_logloss": -np.log(max(bookmaker_prob, 1e-15)),
                    "bookmaker_home": float(bookmaker[j, 0]),
                    "bookmaker_draw": float(bookmaker[j, 1]),
                    "bookmaker_away": float(bookmaker[j, 2]),
                    "team_status": team_status[j],
                    "season_stage": str(stage_series.loc[row.name]),
                    "over25_actual": over_actual,
                    "over25_prob": float(binary_prob[j]),
                    "over25_bookmaker_prob": None,
                    "bookmaker_over25_logloss": None,
                }
            )

        # binary bookmaker over probabilities where available
        try:
            over_arr = None
            from footballpred.evaluate.report import _market_bookmaker_probabilities

            over_arr = _market_bookmaker_probabilities(test, "over_2_5")
            for k, (idx, row) in enumerate(test.iterrows()):
                rec = rows[-(len(test) - k)]
                rec["over25_bookmaker_prob"] = float(over_arr[k])
                home_goals = int(row["home_goals"])
                away_goals = int(row["away_goals"])
                over_actual = int(home_goals + away_goals >= 3)
                p = float(over_arr[k])
                if over_actual == 1:
                    rec["bookmaker_over25_logloss"] = -np.log(max(p, 1e-15))
                else:
                    rec["bookmaker_over25_logloss"] = -np.log(max(1 - p, 1e-15))
        except ValueError:
            pass

        # fix over25 bookmaker logloss for predicted model
        for r in rows[-len(test):]:
            p = r["over25_prob"]
            actual = r["over25_actual"]
            if actual == 1:
                r["over25_logloss"] = -np.log(max(p, 1e-15))
            else:
                r["over25_logloss"] = -np.log(max(1 - p, 1e-15))

    frame = pd.DataFrame(rows)
    return frame


def make_slices(predictions: pd.DataFrame):
    # season stage
    stage = predictions.groupby("season_stage").agg(
        matches=("actual_result", "size"),
        mean_best_logloss=("best_model_logloss", "mean"),
        mean_bookmaker_logloss=("bookmaker_logloss", "mean"),
    ).reset_index()
    # team status
    status = predictions.groupby("team_status").agg(
        matches=("actual_result", "size"),
        mean_best_logloss=("best_model_logloss", "mean"),
        mean_bookmaker_logloss=("bookmaker_logloss", "mean"),
    ).reset_index()
    # match type
    def match_type(row):
        if row["draw_prob"] >= 0.30:
            return "high draw probability"
        if max(row["home_win_prob"], row["away_win_prob"]) >= 0.60:
            return "big favourite"
        return "balanced"

    predictions = predictions.copy()
    predictions["match_type"] = predictions.apply(match_type, axis=1)
    mtype = predictions.groupby("match_type").agg(
        matches=("actual_result", "size"),
        mean_best_logloss=("best_model_logloss", "mean"),
    ).reset_index()
    return stage, status, mtype, predictions


def reliability_table(predictions: pd.DataFrame, prob_col: str, actual_flag_col: str | None = None):
    # For 1X2: use max prob vs top1 accuracy by confidence bin
    frame = predictions.copy()
    if actual_flag_col is None:
        # 1x2
        frame["confidence"] = frame[["home_win_prob", "draw_prob", "away_win_prob"]].max(axis=1)
        frame["top_pick"] = frame[["home_win_prob", "draw_prob", "away_win_prob"]].idxmax(axis=1).map(
            {"home_win_prob": "H", "draw_prob": "D", "away_win_prob": "A"}
        )
        frame["correct"] = (frame["top_pick"] == frame["actual_result"]).astype(int)
    else:
        frame["confidence"] = frame[prob_col]
        frame["correct"] = frame[actual_flag_col]
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    labels = ["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    frame["bin"] = pd.cut(frame["confidence"], bins=bins, labels=labels, include_lowest=True)
    agg = frame.groupby("bin", observed=False).agg(
        matches=("correct", "size"), mean_confidence=("confidence", "mean"), observed_rate=("correct", "mean")
    ).reset_index()
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs")
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    figures_dir = args.figures_dir or (output_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.predictions is not None:
        predictions = pd.read_csv(args.predictions)
    else:
        default_predictions = ROOT / "results" / "error-analysis-predictions.csv"
        if default_predictions.exists():
            predictions = pd.read_csv(default_predictions)
        else:
            predictions = generate_predictions()
            default_predictions.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(default_predictions, index=False)

    stage_slices, status_slices, match_type_slices, predictions = make_slices(predictions)

    rel_1x2 = reliability_table(predictions, "", None)
    rel_over = reliability_table(predictions, "over25_prob", "over25_actual")

    # top 10 log loss
    top10 = predictions.sort_values("best_model_logloss", ascending=False).head(10).copy()
    top10["comment"] = top10.apply(
        lambda r: "High confidence in "
        + ("home win" if r["home_win_prob"] >= max(r["draw_prob"], r["away_win_prob"]) else "away win" if r["away_win_prob"] >= r["draw_prob"] else "draw")
        + " but the result was " + str(r["actual_result"]),
        axis=1,
    )

    # save tables
    stage_slices.to_csv(output_dir / "error_slice_season_stage.csv", index=False)
    status_slices.to_csv(output_dir / "error_slice_team_status.csv", index=False)
    match_type_slices.to_csv(output_dir / "error_slice_match_type.csv", index=False)
    rel_1x2.to_csv(output_dir / "error_reliability_1x2.csv", index=False)
    rel_over.to_csv(output_dir / "error_reliability_over25.csv", index=False)
    top10.to_csv(output_dir / "error_top_logloss.csv", index=False)

    # figure: reliability 1x2
    fig, ax = plt.subplots()
    ax.plot(rel_1x2["mean_confidence"], rel_1x2["observed_rate"], marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_title("1X2 reliability")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Observed top-1 accuracy")
    fig.tight_layout()
    fig.savefig(figures_dir / "error_reliability_1x2.png")
    plt.close(fig)

    # figure: reliability over25
    fig, ax = plt.subplots()
    ax.plot(rel_over["mean_confidence"], rel_over["observed_rate"], marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_title("Over 2.5 reliability")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed Over 2.5 rate")
    fig.tight_layout()
    fig.savefig(figures_dir / "error_reliability_over25.png")
    plt.close(fig)

    # figure: slice logloss
    fig, ax = plt.subplots()
    ax.bar(match_type_slices["match_type"], match_type_slices["mean_best_logloss"])
    ax.set_title("Mean log loss by match type")
    ax.set_ylabel("Mean log loss")
    fig.tight_layout()
    fig.savefig(figures_dir / "error_match_type_logloss.png")
    plt.close(fig)

    def caution(n):
        return " (small sample: fewer than 50 matches)" if n < 50 else ""

    md = []
    md.append("# Error analysis")
    md.append("")
    md.append("Generated by `scripts/error_analysis.py`. Predictions are for held-out test seasons (2425 and 2526).")
    md.append("")
    md.append("## Season stage")
    md.append("")
    md.append(stage_slices.to_markdown(index=False))
    for _, r in stage_slices.iterrows():
        md.append(f"- {r['season_stage']}: {int(r['matches'])} matches{caution(int(r['matches']))}.")
    md.append("")
    md.append("## Team status")
    md.append("")
    md.append(status_slices.to_markdown(index=False))
    for _, r in status_slices.iterrows():
        md.append(f"- {r['team_status']}: {int(r['matches'])} matches{caution(int(r['matches']))}.")
    md.append("")
    md.append("## Match type")
    md.append("")
    md.append(match_type_slices.to_markdown(index=False))
    for _, r in match_type_slices.iterrows():
        md.append(f"- {r['match_type']}: {int(r['matches'])} matches{caution(int(r['matches']))}.")
    md.append("")
    md.append("## Top 10 log-loss penalties (best model)")
    md.append("")
    top_cols = ["date", "home_team", "away_team", "actual_result", "home_win_prob", "draw_prob", "away_win_prob", "best_model_logloss", "comment"]
    md.append(top10[top_cols].to_markdown(index=False))
    md.append("")
    md.append("## Reliability 1X2")
    md.append("")
    md.append(rel_1x2.to_markdown(index=False))
    md.append("")
    md.append("## Reliability Over 2.5")
    md.append("")
    md.append(rel_over.to_markdown(index=False))
    md.append("")
    md.append("Slice-level differences on a few hundred matches can be noise. Small slices are flagged above.")

    (output_dir / "ERROR_ANALYSIS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote report to {output_dir / 'ERROR_ANALYSIS.md'}")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
