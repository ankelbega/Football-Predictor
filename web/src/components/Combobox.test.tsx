import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Combobox } from "./Combobox";

describe("Combobox", () => {
  it("updates its value through keyboard navigation", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <Combobox
        id="team"
        label="Team"
        options={["Arsenal", "Chelsea", "Liverpool"]}
        value=""
        onChange={onChange}
      />,
    );

    const input = screen.getByRole("combobox");
    await user.click(input);
    await user.keyboard("Ar{Enter}");
    expect(onChange).toHaveBeenCalledWith("Arsenal");
  });
});
