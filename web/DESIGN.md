# Web design specification

## Discovery: what the API supports today

Source of truth: `src/footballpred/api/app.py` and its Pydantic models.

Endpoints:

- `GET /health` returns `{"status":"ok","model_loaded":true,"training_data_cutoff":"..."}`.
- `GET /teams` returns a list of team name strings.
- `GET /predict` requires `home`, `away`, `date` (ISO `YYYY-MM-DD`) and accepts optional `model` (`lightgbm` or `dixon_coles`; defaults to `lightgbm`). It returns `PredictionResponse`.

`PredictionResponse` fields:

- `home_team`, `away_team` (string)
- `match_date` (date)
- `model` ("lightgbm" or "dixon_coles")
- `training_data_cutoff` (string)
- `probabilities_1x2.home_win`, `.draw`, `.away_win` (floats, 0 to 1)
- `over_2_5`, `under_2_5`, `btts_yes`, `btts_no` (floats, 0 to 1)
- `most_likely_scorelines` (list of `{home_goals, away_goals, probability}`, up to 5)
- `explanatory_features` (list of objects with `feature`, `label`, `shap_value`, `direction`, `interpretation` for the LightGBM path; fixture-specific for Dixon-Coles)

Capabilities and gaps:

- Supported: team list, prediction, scorelines, SHAP explanations, training cutoff, HTTP 422 validation.
- Partially supported: per-match feature contributions exist for LightGBM but are described as raw-score SHAP values, not probability-point changes; they must be labeled that way.
- Not supported: no endpoint exposes a "recommended model" beyond the API default; no feature-contribution endpoint distinct from `/predict`; no prediction history or caching endpoint; no probability uncertainty interval endpoint. Those will stay in the Method view as explanatory text rather than invented widgets.
- Performance and Method figures come from files under `results/` after deterministic export, not from new endpoints.

## Token set

### Color - light (default)

- `--color-bg: #f7f5f0`
- `--color-surface: #fffefb`
- `--color-surface-2: #f1eee7`
- `--color-ink: #1c1b19`
- `--color-ink-muted: #6b675f`
- `--color-ink-subtle: #8a857b`
- `--color-hairline: #dcd8cf`
- `--color-hairline-strong: #bfb8ab`
- `--color-accent: #1d4e6b`
- `--color-accent-hover: #163c52`
- `--color-home: #1d4e6b`
- `--color-draw: #a39e93`
- `--color-away: #c9822b`
- `--color-success: #2f6f54`
- `--color-warning: #9a6a1f`
- `--color-danger: #8a3b2f`

### Color - dark

- `--color-bg: #141412`
- `--color-surface: #1b1b18`
- `--color-surface-2: #23231f`
- `--color-ink: #ecebe6`
- `--color-ink-muted: #a7a39a`
- `--color-ink-subtle: #7f7b72`
- `--color-hairline: #2d2c28`
- `--color-hairline-strong: #434038`
- `--color-accent: #7aa2bf`
- `--color-accent-hover: #96b7ce`
- `--color-home: #7aa2bf`
- `--color-draw: #7c786f`
- `--color-away: #d7a053`
- `--color-success: #62a37d`
- `--color-warning: #c89a4a`
- `--color-danger: #d27b6a`

### Typography

Families:

- `--font-serif: "Newsreader Variable", serif`
- `--font-sans: "Instrument Sans Variable", sans-serif`
- `--font-mono: "IBM Plex Mono", monospace`

Type scale (rem, 16px base):

- `--text-xs: 0.75rem` (12)
- `--text-sm: 0.875rem` (14)
- `--text-base: 1rem` (16)
- `--text-md: 1.125rem` (18)
- `--text-lg: 1.25rem` (20)
- `--text-xl: 1.5rem` (24)
- `--text-2xl: 1.875rem` (30)
- `--text-3xl: 2.375rem` (38)

Weights: 400 body, 500 labels, 600 table headers, 700 display. Only these.

### Spacing, radius, motion

Spacing scale (px): 4, 8, 12, 16, 24, 32, 48, 64, 96.

- `--radius-sm: 2px`
- `--radius-md: 4px`
- `--rule: 1px`
- `--rule-strong: 2px`
- `--motion-fast: 150ms`
- `--motion-base: 250ms`
- `--motion-slow: 300ms`
- `--ease-out: cubic-bezier(0.16, 1, 0.3, 1)`

Motion is width animation on probability bars only. All motion disabled under `prefers-reduced-motion: reduce`.

## Component inventory

Layout:

- `AppShell` (header, main, footer)
- `Header` (wordmark, nav, theme toggle, language toggle)
- `Footer` (source credit, disclaimer, repo link, training cutoff)
- `Container` (max 1200px, responsive gutters)

Controls:

- `Button` (primary, secondary, text; small radius)
- `Combobox` (searchable, keyboard accessible)
- `Select`
- `Toggle` (theme and language)
- `SegmentedControl` (percent vs 1-in-N)

Data display:

- `ProbabilityBar` (segmented 1X2 bar with direct labels)
- `MarketBar` (two-way labelled bar)
- `ScorelineList`
- `ExplanatoryFeatureList` (only when API returns contributions)
- `DataTable` (sortable by column, horizontal scroll on small screens)
- `StatRow`
- `ReliabilityFigure` (hand-drawn SVG)
- `Disclaimer`

States:

- `Skeleton`, `EmptyState`, `ErrorState`, `LoadingInline`

Navigation between Predict, Performance and Method is three short text links; no large nav bar.

## Component microcopy

- Predict idle: "Pick two teams to see the outlook."
- Loading: "Calculating..."
- Error connecting: "Couldn't reach the prediction service. Start it with `uvicorn footballpred.api.app:app --host 127.0.0.1 --port 8000` and refresh."
- Same-team block: "Choose two different teams."
- Invalid date or 422: use the API's detail text, humanized.
- Empty performance: "Results are bundled with the repository. Run `footballpred evaluate` to regenerate them."
- Method closing: "Estimates only. Not betting advice."

## Wireframes

### Predict (desktop, 1280px)

```text
Football Predictor            Predict  Performance  Method          [theme] [EN|SQ]
---------------------------------------------------------------------------
Home team              Away team              Model
[Arsenal      v]       [Chelsea      v]       [LightGBM v]
---------------------------------------------------------------------------
MATCH OUTLOOK
Home win 54.2%     Draw 24.1%     Away win 21.7%
[===================================|==============|==========]
---------------------------------------------------------------------------
GOALS
Over 2.5  56.0%  [=============    ]   Under 2.5  44.0%
BTTS yes  52.3%  [============     ]   BTTS no    47.7%
---------------------------------------------------------------------------
MOST LIKELY SCORELINES
1-1  11.2%  2-1  9.8%  1-0  9.4%  2-0  8.1%  2-2  6.0%
---------------------------------------------------------------------------
WHY THE MODEL LEANS THIS WAY (LightGBM)
Recent form (home)     +4.1  [====]
Elo difference         +3.0  [===]
...
---------------------------------------------------------------------------
Model: lightgbm · trained on matches up to 2026-05-24
Estimates only. Not betting advice.
```

### Predict (mobile, 390px)

Home team and away team stack vertically. Segmented probability bar wraps into three stacked labelled rows. Scorelines wrap. Explanatory features collapse to a single column. Everything remains 44px touch height.

### Performance (desktop)

Title, one sentence of context, then three tables (1X2, Over 2.5, BTTS) with right-aligned monospace numerals, hairline rules, sorted headers, highlighted best value by weight and underline. Walk-forward figure and two calibration figures below, full width but max 800px. Small-sample cautions as plain text beneath tables.

### Method (desktop)

Single text column at roughly 65 characters. Section headings, short paragraphs, one small illustrative SVG reliability curve, limitations list, credits and disclaimer at the end. Same layout on mobile with narrower measure.

## Proposed API additions (for approval only, not implemented)

- A `GET /health`-style field naming the available models and a recommended default would remove the need to read the OpenAPI default, but the front end can already use the API's documented default (`lightgbm`).
- If real SHAP is unavailable from a deployed bundle, the current API already supplies contributions; no addition needed.
- No other endpoint changes are proposed. CORS is not needed because the Vite dev proxy will reach the API directly.<|end|><|start|>write to=assistant<|channel|>analysis<|message|>Created file successfully: web/DESIGN.md
## Layout fix pass notes

- Shell width changed to 1120px with a `Container` used by header, main and footer.
- Header uses a left-aligned wordmark/navigation group and right-aligned controls.
- Predict controls use 4/2/1 columns; goals and scorelines use a 7/5 split desktop.
- Outcome numerals use one source (`ProbabilityBar`) with the same display mode.
- The date input follows the theme color scheme and `prefers-reduced-motion` remains unchanged.
