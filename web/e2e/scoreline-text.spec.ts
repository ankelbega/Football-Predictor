import { expect, test } from "@playwright/test";

const fixture = {
  home_team: "Liverpool",
  away_team: "Arsenal",
  match_date: "2026-05-25",
  model: "lightgbm",
  training_data_cutoff: "2026-05-24T00:00:00",
  probabilities_1x2: { home_win: 0.301, draw: 0.271, away_win: 0.428 },
  over_2_5: 0.52,
  under_2_5: 0.48,
  btts_yes: 0.564,
  btts_no: 0.436,
  most_likely_scorelines: [
    { home_goals: 1, away_goals: 1, probability: 0.129 },
    { home_goals: 2, away_goals: 1, probability: 0.091 },
    { home_goals: 0, away_goals: 0, probability: 0.087 },
  ],
  explanatory_features: [
    {
      feature: "Elo difference",
      label: "Elo home advantage over away",
      shap_value: 0.52,
      direction: "raised",
      interpretation: "Elo advantage raised the home score.",
    },
  ],
};

for (const theme of ["light", "dark"] as const) {
  test(`predict raw text is clean in ${theme} theme`, async ({ page }) => {
    await page.addInitScript((chosenTheme) => localStorage.setItem("theme", chosenTheme), theme);
    await page.setViewportSize({ width: 1900, height: 1000 });
    await page.route("**/api/predict*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(fixture),
      });
    });
    await page.goto("/?home=Liverpool&away=Arsenal&date=2026-05-25&model=lightgbm");
    await page.waitForLoadState("networkidle");
    await expect(page.getByTestId("outlook-bar")).toBeVisible();
    const text = await page.locator("body").innerText();
    expect(text).not.toMatch(/\\u[0-9a-fA-F]{4}/);
    expect(text).not.toMatch(/[a-z]+_[a-z]+(?:_[a-z]+)*/);
  });
}
