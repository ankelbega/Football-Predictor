import type { HealthResponse, PredictionResponse } from "./types";

const BASE = "/api";

export class RequestError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "RequestError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${BASE}${path}`, { signal });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = (await response.json()) as { detail?: unknown };
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      // Keep the status text when the response is not JSON.
    }
    throw new RequestError(response.status, detail);
  }
  return (await response.json()) as T;
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>("/health", signal);
}

export async function fetchTeams(signal?: AbortSignal): Promise<string[]> {
  return request<string[]>("/teams", signal);
}

export interface PredictionQuery {
  home: string;
  away: string;
  date: string;
  model: string;
}

export async function fetchPrediction(
  query: PredictionQuery,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  const params = new URLSearchParams({
    home: query.home,
    away: query.away,
    date: query.date,
    model: query.model,
  });
  return request<PredictionResponse>(`/predict?${params.toString()}`, signal);
}
