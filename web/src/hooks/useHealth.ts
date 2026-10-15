import { useEffect, useState } from "react";
import { fetchHealth } from "../lib/api";
import type { HealthResponse } from "../lib/types";

interface ResourceState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

export function useHealth(): ResourceState<HealthResponse> {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetchHealth(controller.signal)
      .then((value) => setData(value))
      .catch((cause: unknown) => {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        setError(messageForError(cause));
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, []);

  return { data, loading, error };
}

export function messageForError(cause: unknown): string {
  if (cause instanceof Error) return cause.message;
  return "Unexpected error.";
}

void useHealth;
