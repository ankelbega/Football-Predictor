import { useEffect, useRef, useState } from "react";
import { fetchPrediction } from "../lib/api";
import type { PredictionResponse } from "../lib/types";

interface PredictionInput {
  home: string;
  away: string;
  date: string;
  model: string;
}

interface PredictionState {
  result: PredictionResponse | null;
  loading: boolean;
  error: string | null;
  stale: boolean;
}

const MISSING: PredictionState = {
  result: null,
  loading: false,
  error: null,
  stale: false,
};

function isAbortError(cause: unknown): boolean {
  return cause instanceof DOMException && cause.name === "AbortError";
}

function messageForPrediction(cause: unknown): string {
  if (cause instanceof Error) {
    const status = (cause as { status?: unknown }).status;
    if (typeof status === "number" && status >= 500) {
      return "Couldn't reach the prediction service. Start it with `uvicorn footballpred.api.app:app --host 127.0.0.1 --port 8000` and refresh.";
    }
    return cause.message;
  }
  return "Unexpected error.";
}

export function usePrediction(input: PredictionInput, enabled: boolean): PredictionState {
  const [state, setState] = useState<PredictionState>(MISSING);
  const previousKeyRef = useRef<string | null>(null);
  const previousResultRef = useRef<PredictionResponse | null>(null);

  useEffect(() => {
    if (!enabled || !input.home || !input.away || !input.date) {
      previousKeyRef.current = null;
      previousResultRef.current = null;
      setState(MISSING);
      return;
    }
    const key = `${input.home}|${input.away}|${input.date}|${input.model}`;
    const abortController = new AbortController();
    const becomingStale =
      previousKeyRef.current !== null && previousKeyRef.current !== key && previousResultRef.current !== null;
    setState({
      result: previousResultRef.current,
      loading: true,
      error: null,
      stale: becomingStale,
    });
    const home = input.home;
    const away = input.away;
    const date = input.date;
    const model = input.model;
    fetchPrediction({ home, away, date, model }, abortController.signal)
      .then((result) => {
        setState({ result, loading: false, error: null, stale: false });
        previousKeyRef.current = key;
        previousResultRef.current = result;
      })
      .catch((cause) => {
        if (isAbortError(cause)) return;
        setState({
          result: previousResultRef.current,
          loading: false,
          error: messageForPrediction(cause),
          stale: false,
        });
      });
    return () => abortController.abort();
  }, [input.home, input.away, input.date, input.model, enabled]);

  return state;
}
