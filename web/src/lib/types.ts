export interface MarketProbabilities {
  home_win: number;
  draw: number;
  away_win: number;
}

export interface Scoreline {
  home_goals: number;
  away_goals: number;
  probability: number;
}

export interface ExplanatoryFeature {
  feature: string;
  label?: string;
  feature_value?: number | null;
  shap_value?: number;
  direction?: "raised" | "lowered" | string;
  interpretation?: string;
  value?: number | null;
}

export interface ApiError extends Error {
  status: number;
  detail: string;
}

export interface PredictionResponse {
  home_team: string;
  away_team: string;
  match_date: string;
  model: "lightgbm" | "dixon_coles";
  training_data_cutoff: string;
  probabilities_1x2: MarketProbabilities;
  over_2_5: number;
  under_2_5: number;
  btts_yes: number;
  btts_no: number;
  most_likely_scorelines: Scoreline[];
  explanatory_features: ExplanatoryFeature[];
}

export interface HealthResponse {
  status: "ok";
  model_loaded: boolean;
  training_data_cutoff: string;
}

export type DisplayMode = "percent" | "oneInN";
