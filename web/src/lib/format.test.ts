import { describe, expect, it } from "vitest";
import {
  addDaysToDate,
  formatIsoDate,
  formatProbability,
} from "./format";

describe("format", () => {
  it("formats percentages and one-in-n odds", () => {
    expect(formatProbability(0.542, "percent")).toBe("54.2%");
    expect(formatProbability(0.542, "oneInN")).toBe("1 in 2");
  });

  it("formats ISO dates and date offsets", () => {
    expect(formatIsoDate("2026-05-24")).toBe("24 May 2026");
    expect(addDaysToDate("2026-05-24", 1)).toBe("2026-05-25");
  });
});
