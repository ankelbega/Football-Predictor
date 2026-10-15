import { useEffect, useState } from "react";
import { fetchTeams } from "../lib/api";

interface ResourceState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

export function useTeams(): ResourceState<string[]> {
  const [data, setData] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetchTeams(controller.signal)
      .then((teams) => setData(teams.slice().sort((left, right) => left.localeCompare(right))))
      .catch((cause: unknown) => {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        setError(messageForFetch(cause));
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, []);

  return { data, loading, error };
}

function messageForFetch(cause: unknown): string {
  if (cause instanceof Error) return cause.message;
  return "Could not load team list.";
}
