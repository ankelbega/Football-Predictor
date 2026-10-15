import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ProbabilityBar } from "./ProbabilityBar";

describe("ProbabilityBar", () => {
  it("renders direct labels", () => {
    render(
      <ProbabilityBar
        segments={[
          { label: "Home", value: 0.54, color: "#1d4e6b" },
          { label: "Draw", value: 0.24, color: "#a39e93" },
          { label: "Away", value: 0.22, color: "#c9822b" },
        ]}
      />,
    );
    expect(screen.getByText("Home 54%")).toBeInTheDocument();
  });
});
