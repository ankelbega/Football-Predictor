const EXPLANATION_LABELS: Record<string, string> = {
  elo_difference: "Elo rating gap (home minus away)",
  elo_home: "Home team Elo rating",
  elo_away: "Away team Elo rating",
  home_overall_last5_points_per_match: "Home team recent points per match",
  away_overall_last5_points_per_match: "Away team recent points per match",
  home_home_last5_points_per_match: "Home team home points per match",
  away_away_last5_points_per_match: "Away team away points per match",
  home_overall_last5_goals_for_per_match: "Home team recent goals scored per match",
  away_overall_last5_goals_for_per_match: "Away team recent goals scored per match",
  home_overall_last5_goals_against_per_match: "Home team recent goals conceded per match",
  away_overall_last5_goals_against_per_match: "Away team recent goals conceded per match",
  home_days_rest: "Home team days of rest",
  away_days_rest: "Away team days of rest",
  season_progress: "Season progress",
};

export function formatExplanationLabel(key: string): string {
  const mapped = EXPLANATION_LABELS[key];
  if (mapped) return mapped;
  const cleaned = key.replace(/_/g, " ").trim();
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
}
