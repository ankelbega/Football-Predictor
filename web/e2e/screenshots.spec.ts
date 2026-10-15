import * as fs from "fs";
import * as path from "path";
import { test } from "@playwright/test";

const routes = ["/", "/performance", "/method", "/styleguide"];
const themes = ["light", "dark"] as const;

for (const theme of themes) {
  for (const route of routes) {
    test(`${route} ${theme}`, async ({ page }, testInfo) => {
      const viewport = testInfo.project.name;
      await page.addInitScript((chosenTheme: string) => {
        window.localStorage.setItem("theme", chosenTheme);
      }, theme);
      await page.goto(route);
      await page.waitForLoadState("networkidle");
      const slug = route === "/" ? "predict" : route.slice(1);
      const directory = path.join(process.cwd(), "screenshots");
      fs.mkdirSync(directory, { recursive: true });
      await page.screenshot({
        path: path.join(directory, `${slug}-${viewport}-${theme}.png`),
        fullPage: true,
      });
    });
  }
}
