import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ScorelineList } from "./ScorelineList";

describe("ScorelineList", () => {
  it("renders en dashes as text", () => {
    render(
      <ScorelineList
        items={[
          { home_goals: 1, away_goals: 1, probability: 0.129 },
          { home_goals: 10, away_goals: 10, probability: 0.05 },
        ]}
        displayMode="percent"
      />,
    );
    expect(screen.getByText("1\u20131")).toBeInTheDocument();
    expect(screen.getByText("10\u201310")).toBeInTheDocument();
    const text = document.body.textContent ?? "";
    expect(text).not.toMatch(/\\u[0-9a-fA-F]{4}/);
    expect(text).not.toContain("\\");
  });
});
