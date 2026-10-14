"""Fit a time-decayed score model and derive all markets from its score matrix.

This model-stage module is called by evaluation, ensemble prediction, explanations, and API
serving. Attack and defence strengths are centered to identify the model; unseen teams use
neutral strengths so promoted clubs can still receive a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from footballpred.models.base import ProbabilityModel, validate_probabilities
from footballpred.models.baselines import OUTCOMES


@dataclass(frozen=True)
class DixonColesConfig:
    """Controls for time decay, score truncation, optimization, and shrinkage.

    Args:
        xi: Exponential decay per day; larger values give recent matches more relative weight.
        max_goals: Highest home or away score retained in the truncated matrix.
        maxiter: Maximum L-BFGS iterations for the likelihood fit.
        regularization: Small quadratic penalty on centered team strengths.
    """

    xi: float = 0.0018
    max_goals: int = 10
    maxiter: int = 400
    regularization: float = 1e-5


def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    home_rate: float,
    away_rate: float,
    rho: float,
) -> float:
    """Return the Dixon–Coles dependence multiplier for low score pairs.

    Args:
        home_goals: Observed/predicted home goals.
        away_goals: Observed/predicted away goals.
        home_rate: Expected home goals under Poisson component.
        away_rate: Expected away goals under Poisson component.
        rho: Fitted low-score dependence parameter.

    Returns:
        Tau correction. Only 0-0, 0-1, 1-0 and 1-1 are adjusted; other scores use one.
    """
    # rho redistributes probability among the four common low scores to model score dependence.
    if home_goals == 0 and away_goals == 0:
        return 1.0 - home_rate * away_rate * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + home_rate * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + away_rate * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def poisson_score_matrix(
    home_rate: float, away_rate: float, rho: float = 0.0, max_goals: int = 10
) -> np.ndarray:
    """Build and normalize the truncated joint home/away score probability matrix.

    Args:
        home_rate: Expected home goals, strictly positive.
        away_rate: Expected away goals, strictly positive.
        rho: Dixon–Coles low-score adjustment; zero gives independent Poisson scores.
        max_goals: Last integer goal count on each matrix axis.

    Returns:
        Square matrix where cell ``[h, a]`` is probability of score h-a. Tail mass beyond the
        truncation is dropped and the retained matrix is renormalized.
    """
    # Poisson pmf uses exp(k*log(rate) - rate - log(k!)); gammaln keeps factorials stable.
    if home_rate <= 0 or away_rate <= 0:
        raise ValueError("Poisson goal rates must be positive")
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1")
    goals = np.arange(max_goals + 1)
    home_pmf = np.exp(goals * np.log(home_rate) - home_rate - gammaln(goals + 1))
    away_pmf = np.exp(goals * np.log(away_rate) - away_rate - gammaln(goals + 1))
    matrix = np.outer(home_pmf, away_pmf)
    # Dixon–Coles changes low-score dependence, leaving higher scoreline ratios unchanged.
    matrix[0, 0] *= dixon_coles_tau(0, 0, home_rate, away_rate, rho)
    if max_goals >= 1:
        # The common default is 10; still handle 0 safely for callers requesting tiny matrices.
        matrix[0, 1] *= dixon_coles_tau(0, 1, home_rate, away_rate, rho)
        matrix[1, 0] *= dixon_coles_tau(1, 0, home_rate, away_rate, rho)
        matrix[1, 1] *= dixon_coles_tau(1, 1, home_rate, away_rate, rho)
    if not np.isfinite(matrix).all() or (matrix < 0).any() or matrix.sum() <= 0:
        raise ValueError("Dixon-Coles parameters yielded an invalid score matrix")
    return matrix / matrix.sum()


def markets_from_score_matrix(matrix: np.ndarray) -> dict[str, object]:
    """Sum a joint score matrix into 1X2, goals, BTTS and top scoreline probabilities.

    Args:
        matrix: Square array with home goals on rows and away goals on columns.

    Returns:
        Mapping of 1X2 vector, Over/Under 2.5, BTTS yes/no and five highest scorelines.
        Each market is a sum over score cells, so the reported markets are internally consistent.
    """
    score = np.asarray(matrix, dtype=float)
    if score.ndim != 2 or score.shape[0] != score.shape[1]:
        raise ValueError("Score matrix must be a square (home goals, away goals) matrix")
    if (score < 0).any() or not np.isfinite(score).all() or score.sum() <= 0:
        raise ValueError("Score matrix must be finite, non-negative, and have positive mass")
    score = score / score.sum()
    # A home win is below the diagonal (home-goal row index greater than away-goal column).
    home = float(np.tril(score, k=-1).sum())
    draw = float(np.trace(score))
    away = float(np.triu(score, k=1).sum())
    goal_counts = np.add.outer(np.arange(score.shape[0]), np.arange(score.shape[1]))
    # Over 2.5 means at least three combined goals; under is its complement.
    over = float(score[goal_counts >= 3].sum())
    both_score = np.ones_like(score, dtype=bool)
    both_score[0, :] = False
    both_score[:, 0] = False
    btts = float(score[both_score].sum())
    flat = np.argsort(score.ravel())[::-1][:5]
    scorelines = [
        {"home_goals": int(index // score.shape[1]), "away_goals": int(index % score.shape[1]),
         "probability": float(score.ravel()[index])}
        for index in flat
    ]
    return {
        "1x2": validate_probabilities(np.array([[home, draw, away]]))[0],
        "over_2_5": over,
        "under_2_5": 1.0 - over,
        "btts_yes": btts,
        "btts_no": 1.0 - btts,
        "scorelines": scorelines,
    }


class DixonColesModel(ProbabilityModel):
    """Estimate team attack/defence log strengths and predict joint score probabilities.

    Attack and defence are zero-centered over observed training teams to resolve the
    intercept/strength identifiability issue. The shared home-advantage term is a log-rate
    increment. Recent matches receive weight ``exp(-xi * age_days)``.
    """

    name = "dixon_coles"
    classes_ = np.asarray(OUTCOMES)

    def __init__(self, config: DixonColesConfig | None = None) -> None:
        """Create an unfitted score model.

        Args:
            config: Fit and score-matrix settings; defaults to ``DixonColesConfig``.
        """
        self.config = config or DixonColesConfig()
        self.teams_: list[str] = []
        self.attack_: dict[str, float] = {}
        self.defence_: dict[str, float] = {}
        self.home_advantage_: float = 0.0
        self.intercept_: float = np.log(1.35)
        self.rho_: float = 0.0
        self.training_cutoff_: pd.Timestamp | None = None
        self.optimization_success_: bool = False

    def fit(self, matches: pd.DataFrame) -> DixonColesModel:
        """Fit attack, defence, home advantage and rho using weighted Poisson likelihood.

        Args:
            matches: Rows with date, home/away teams, and non-negative final goal counts.

        Returns:
            This fitted model instance.

        Assumption:
            Each match's home and away goals have Poisson marginals; rho corrects only the
            four low scorelines. The model does not represent lineup or within-match dynamics.
        """
        required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(matches.columns)
        if missing:
            raise ValueError(f"Dixon-Coles input missing columns: {sorted(missing)}")
        if matches.empty:
            raise ValueError("Cannot fit Dixon-Coles on an empty training dataset")
        data = matches.copy()
        data["date"] = pd.to_datetime(data["date"], errors="raise")
        data["home_goals"] = pd.to_numeric(data["home_goals"], errors="raise").astype(int)
        data["away_goals"] = pd.to_numeric(data["away_goals"], errors="raise").astype(int)
        if (data[["home_goals", "away_goals"]] < 0).any().any():
            raise ValueError("Goal counts must be non-negative")
        self.training_cutoff_ = data["date"].max()
        self.teams_ = sorted(set(data["home_team"].astype(str)) | set(data["away_team"].astype(str)))
        team_index = {team: i for i, team in enumerate(self.teams_)}
        home_idx = data["home_team"].astype(str).map(team_index).to_numpy(dtype=int)
        away_idx = data["away_team"].astype(str).map(team_index).to_numpy(dtype=int)
        home_goals = data["home_goals"].to_numpy(dtype=int)
        away_goals = data["away_goals"].to_numpy(dtype=int)
        # NORMALIZATION: express match age in days because xi is configured per day.
        age_days = (self.training_cutoff_ - data["date"]).dt.total_seconds().to_numpy() / 86400.0
        # Recent matches get larger likelihood weight; xi is per day, age_days is day units.
        weights = np.exp(-self.config.xi * age_days)
        n_teams = len(self.teams_)

        # Parameter vector: intercept, home advantage, n attack, n defence, rho.
        # Centering attacks/defences inside likelihood fixes the otherwise free offsets.
        initial = np.zeros(2 + 2 * n_teams + 1, dtype=float)
        initial[0] = np.log(max(float(data[["home_goals", "away_goals"]].to_numpy().mean()), 0.2))
        initial[1] = 0.15
        # Bounds prevent optimizer excursions into implausible goal rates or invalid rho terms.
        lower = np.concatenate(([-1.5, -0.8], np.full(2 * n_teams, -2.5), [-0.2]))
        upper = np.concatenate(([1.5, 0.8], np.full(2 * n_teams, 2.5), [0.2]))

        def likelihood(parameters: np.ndarray) -> float:
            """Use age weights as recency multipliers in the negative log likelihood."""
            """Negative weighted log likelihood, with a penalty to limit extreme strengths."""
            intercept, home_advantage = parameters[:2]
            raw_attack = parameters[2 : 2 + n_teams]
            raw_defence = parameters[2 + n_teams : 2 + 2 * n_teams]
            attack = raw_attack - raw_attack.mean()
            defence = raw_defence - raw_defence.mean()
            rho = parameters[-1]
            home_log_rate = intercept + home_advantage + attack[home_idx] - defence[away_idx]
            away_log_rate = intercept + attack[away_idx] - defence[home_idx]
            home_rate = np.exp(np.clip(home_log_rate, -4, 4))
            away_rate = np.exp(np.clip(away_log_rate, -4, 4))
            tau = np.array(
                [
                    dixon_coles_tau(int(hg), int(ag), float(lh), float(la), float(rho))
                    for hg, ag, lh, la in zip(
                        home_goals, away_goals, home_rate, away_rate, strict=True
                    )
                ]
            )
            if (tau <= 0).any():
                return 1e12
            log_likelihood = (
                home_goals * np.log(home_rate)
                - home_rate
                - gammaln(home_goals + 1)
                + away_goals * np.log(away_rate)
                - away_rate
                - gammaln(away_goals + 1)
                + np.log(tau)
            )
            # Small ridge penalty discourages extreme strengths in sparse seasons.
            # The quadratic penalty keeps sparse-team attack/defence estimates near league mean.
            regularization = self.config.regularization * (
                np.sum(attack**2) + np.sum(defence**2)
            )
            return float(-np.sum(weights * log_likelihood) + regularization)

        result = minimize(
            likelihood,
            initial,
            method="L-BFGS-B",
            bounds=list(zip(lower, upper, strict=True)),
            options={"maxiter": self.config.maxiter, "ftol": 1e-8},
        )
        if not np.isfinite(result.fun):
            raise RuntimeError("Dixon-Coles optimization returned a non-finite objective")
        parameters = result.x
        raw_attack = parameters[2 : 2 + n_teams]
        raw_defence = parameters[2 + n_teams : 2 + 2 * n_teams]
        attack = raw_attack - raw_attack.mean()
        defence = raw_defence - raw_defence.mean()
        self.attack_ = dict(zip(self.teams_, attack, strict=True))
        self.defence_ = dict(zip(self.teams_, defence, strict=True))
        self.intercept_ = float(parameters[0])
        self.home_advantage_ = float(parameters[1])
        self.rho_ = float(parameters[-1])
        self.optimization_success_ = bool(result.success)
        return self

    def goal_rates(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Compute expected home and away goals for a named fixture.

        Args:
            home_team: Home club name.
            away_team: Away club name.

        Returns:
            Positive expected-goal rates. Unknown clubs have zero attack/defence log strength.
        """
        home_attack = self.attack_.get(str(home_team), 0.0)
        away_attack = self.attack_.get(str(away_team), 0.0)
        home_defence = self.defence_.get(str(home_team), 0.0)
        away_defence = self.defence_.get(str(away_team), 0.0)
        home_rate = float(np.exp(self.intercept_ + self.home_advantage_ + home_attack - away_defence))
        away_rate = float(np.exp(self.intercept_ + away_attack - home_defence))
        return home_rate, away_rate

    def team_strengths(self) -> pd.DataFrame:
        """Return fitted centered attack/defence values as a reviewable table.

        Returns:
            Columns are team, centered attack and defence log strengths, and shared home
            advantage. Positive defence means a higher conceded-goal rate (weaker defence).
        """
        return pd.DataFrame(
            [
                {
                    "team": team,
                    "attack_log_strength": self.attack_[team],
                    "defence_log_strength": self.defence_[team],
                    "home_advantage_log_rate": self.home_advantage_,
                }
                for team in self.teams_
            ]
        ).sort_values("attack_log_strength", ascending=False, ignore_index=True)

    def predict_score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Predict one fixture's normalized score distribution.

        Args:
            home_team: Home club name.
            away_team: Away club name.

        Returns:
            Square score probability matrix truncated according to ``config.max_goals``.
        """
        home_rate, away_rate = self.goal_rates(home_team, away_team)
        return poisson_score_matrix(home_rate, away_rate, self.rho_, self.config.max_goals)

    def predict_markets(self, matches: pd.DataFrame) -> list[dict[str, object]]:
        """Return each fixture's markets, all summed from its own score matrix.

        Args:
            matches: DataFrame containing ``home_team`` and ``away_team`` columns.

        Returns:
            One dictionary per row, with 1X2, Over/Under, BTTS and top scorelines.
        """
        required = {"home_team", "away_team"}
        missing = required - set(matches.columns)
        if missing:
            raise ValueError(f"Dixon-Coles prediction input missing columns: {sorted(missing)}")
        return [
            markets_from_score_matrix(self.predict_score_matrix(row.home_team, row.away_team))
            for row in matches.itertuples(index=False)
        ]

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        """Return 1X2 probabilities with columns ordered home win, draw, away win.

        Args:
            matches: Fixture dataframe with home_team and away_team.

        Returns:
            Valid ``(n_matches, 3)`` probability matrix.
        """
        markets = self.predict_markets(matches)
        if not markets:
            return np.empty((0, 3), dtype=float)
        return validate_probabilities(np.stack([market["1x2"] for market in markets]))
