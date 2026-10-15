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
    { home_goals: 1, away_goals: 2, probability: 0.091 },
    { home_goals: 0, away_goals: 1, probability: 0.087 },
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

async function openResult(page: import("@playwright/test").Page) {
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
}

test.describe("Predict layout", () => {
  for (const width of [1440, 1024, 768, 390]) {
    test(`alignment at ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await openResult(page);

      const metrics = await page.evaluate(() => {
        const rect = (selector: string) => {
          const element = document.querySelector(selector) as HTMLElement | null;
          if (!element) return null;
          const box = element.getBoundingClientRect();
          return { left: box.left, right: box.right, top: box.top, bottom: box.bottom };
        };
        const selectors = [
          ".wordmark",
          ".predict-view__intro h1",
          ".predict-controls > *:first-child",
          "h3",
          ".site-footer p",
        ];
        const lefts = selectors.map((selector) => rect(selector)).
          map((box) => (box ? Math.round(box.left) : null));
        const overlapping = (() => {
          const elements = Array.from(
            document.querySelectorAll(".predict-controls > *"),
          ) as HTMLElement[];
          for (let i = 0; i < elements.length; i += 1) {
            const current = elements[i]!.getBoundingClientRect();
            for (let j = i + 1; j < elements.length; j += 1) {
              const next = elements[j]!.getBoundingClientRect();
              if (current.left < next.right && next.left < current.right && current.top < next.bottom && next.top < current.bottom) {
                return true;
              }
            }
          }
          return false;
        })();
        return { lefts, overlapping };
      });

      for (const left of metrics.lefts) {
        expect(left).not.toBeNull();
      }
      const baseLeft = metrics.lefts[0] as number;
      for (const left of metrics.lefts) {
        expect(Math.abs((left as number) - baseLeft)).toBeLessThanOrEqual(1);
      }

      const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
      expect(scrollWidth).toBeLessThanOrEqual(width);

      const resultText = await page.locator(".prediction-output").innerText();
      expect(resultText).not.toMatch(/[a-z]+_[a-z]+(?:_[a-z]+)*/);

      if (width >= 1024) {
        const desktopMetrics = await page.evaluate(() => {
          const rect = (selector: string) => {
            const element = document.querySelector(selector) as HTMLElement | null;
            if (!element) return null;
            const box = element.getBoundingClientRect();
            return { right: box.right };
          };
          return {
            lastField: rect(".predict-controls > *:last-child"),
            bar: rect('[data-testid="outlook-bar"]'),
            footer: rect(".site-footer p"),
          };
        });
        expect(desktopMetrics.lastField?.right).toBeDefined();
        expect(desktopMetrics.bar?.right).toBeDefined();
        expect(desktopMetrics.footer?.right).toBeDefined();
        if (desktopMetrics.lastField && desktopMetrics.bar && desktopMetrics.footer) {
          expect(Math.abs(desktopMetrics.lastField.right - desktopMetrics.bar.right)).toBeLessThanOrEqual(1);
          expect(Math.abs(desktopMetrics.bar.right - desktopMetrics.footer.right)).toBeLessThanOrEqual(1);
        }
      }

      expect(metrics.overlapping).toBe(false);
    });
  }
});
