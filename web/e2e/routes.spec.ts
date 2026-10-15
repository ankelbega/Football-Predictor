import { test, expect } from "@playwright/test";

test("home renders navigation and main content", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("h1")).toBeVisible();
  await page.goto("/performance");
  await expect(page.locator("h1")).toContainText("Performance");
  await page.goto("/method");
  await expect(page.locator("h1")).toContainText("Method");
  await page.goto("/styleguide");
  await expect(page.locator("h1")).toBeVisible();
});

test("language toggle sets the locale", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => localStorage.removeItem("language"));
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("lang", "en");
  await page.getByRole("button", { name: /toggle language/i }).click();
  await expect(page.locator("html")).toHaveAttribute("lang", "sq");
});
