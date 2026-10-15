import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ExplanatoryFeatureList } from "./ExplanatoryFeatureList";

vi.mock("../i18n/useI18n", () => ({
  useI18n: () => ({
    t: (key: string) => (key === "predict.explanation.value" ? "Value: {value}" : key),
    locale: "en",
  }),
}));

describe("ExplanatoryFeatureList", () => {
  it("uses real labels and direction classes", () => {
    const container = render(
      <ExplanatoryFeatureList
        model="lightgbm"
        explainedOutcome="Liverpool win"
        items={[
          {
            feature: "elo_difference",
            feature_value: 135,
            shap_value: 0.33,
            direction: "raised",
          },
          {
            feature: "elo_away",
            feature_value: 1440,
            shap_value: -0.12,
            direction: "lowered",
          },
          {
            feature: "unknown_key",
            feature_value: 12,
            shap_value: 0.05,
            direction: "raised",
          },
        ]}
      />,
    ).container;
    const text = container.textContent ?? "";
    expect(text).toContain("Elo rating gap (home minus away)");
    expect(text).toContain("Unknown key");
    expect(text).toContain("Value: 135");
    expect(text).not.toContain("elo-class");
    expect(container.querySelectorAll("[class*=\"barRaising\"]").length).toBe(2);
    expect(container.querySelectorAll("[class*=\"barLowering\"]").length).toBe(1);
    expect(screen.getByText("+0.33")).toBeInTheDocument();
  });
});
