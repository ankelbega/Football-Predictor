import { describe, expect, it } from "vitest";
import { formatExplanationLabel } from "../lib/explainLabels";

describe("formatExplanationLabel", () => {
  it("maps known keys to readable labels", () => {
    expect(formatExplanationLabel("elo_difference")).toBe("Elo rating gap (home minus away)");
    expect(formatExplanationLabel("elo_home")).toBe("Home team Elo rating");
  });

  it("falls back to a sentence-cased key", () => {
    expect(formatExplanationLabel("unknown_feature_key")).toBe("Unknown feature key");
    expect(formatExplanationLabel("unlikelyIdentifier")).toBe("UnlikelyIdentifier");
    expect(formatExplanationLabel("unknown_feature_key").includes("_")).toBe(false);
  });
});
