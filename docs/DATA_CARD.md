# Data card

## Source

- Name: football-data.co.uk (English Premier League match and odds data).
- URL pattern configured in `configs/default.yaml`: `https://www.football-data.co.uk/mmz4281/{season}/{league}.csv`.
- League codes used: `E0`.
- Seasons used: 1415 through 2526 (12 seasons).
- Cached raw files live in `data/raw/E0_<season>.csv` and are reused when rerunning commands.

Check the source's own terms of use for redistribution and commercial use; this repository documents the processing of the files, not a license for them.

## Columns used

Required columns from the source (verified in `clean_source_file`):

- `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`

Optional columns captured when present:

- `B365H`, `B365D`, `B365A` become `b365h`, `b365d`, `b365a`
- `PSCH`, `PSCD`, `PSCA` become `psch`, `pscd`, `psca`
- `B365>2.5`, `B365<2.5`, `Avg>2.5`, `Avg<2.5` become `b365_over_2.5`, `b365_under_2.5`, `avg_over_2.5`, `avg_under_2.5`
- Optional shot and corner fields present in some seasons are parsed when available

The processed table has no rows with missing dates, teams, goals or results.

## Row counts

Total processed rows: 4560, across 35 teams, from 2014-08-16 to 2026-05-24. Each season has 380 matches.

| Season | Matches | First date | Last date |
|---|---:|---|---|
| 1415 | 380 | 2014-08-16 | 2015-05-24 |
| 1516 | 380 | 2015-08-08 | 2016-05-17 |
| 1617 | 380 | 2016-08-13 | 2017-05-21 |
| 1718 | 380 | 2017-08-11 | 2018-05-13 |
| 1819 | 380 | 2018-08-10 | 2019-05-12 |
| 1920 | 380 | 2019-08-09 | 2020-07-26 |
| 2021 | 380 | 2020-09-12 | 2021-05-23 |
| 2122 | 380 | 2021-08-13 | 2022-05-22 |
| 2223 | 380 | 2022-08-05 | 2023-05-28 |
| 2324 | 380 | 2023-08-11 | 2024-05-19 |
| 2425 | 380 | 2024-08-16 | 2025-05-25 |
| 2526 | 380 | 2025-08-15 | 2026-05-24 |

Counts were produced by `scripts/data_stats.py`.

## Missing values

For the 4560 processed rows:

- Core fields (date, league, season, teams, goals, result): 0 missing.
- `b365h/d/a`: 0 missing.
- `psch/pscd/psca`: 170 missing (3.73%).
- `b365_over_2.5` / `b365_under_2.5`: 1901 missing (41.69%).
- `avg_over_2.5` / `avg_under_2.5`: 1900 missing (41.67%).
- Validation found 0 duplicate matches, 0 impossible scores, 0 invalid result codes, and 170 rows with odds columns checked that were suspicious (odd value at or below 1.0 or non-finite). These 170 correspond to the missing Pinnacle odds rows.

## Cleaning steps

`ingest/clean.py` performs the following:

1. Reads the source with several common encodings (UTF-8 with BOM, Windows-1252, Latin-1).
2. Checks the required columns are present; otherwise the file is skipped.
3. Drops fully blank rows.
4. Parses `Date` with day-first parsing for two- or four-digit years.
5. Normalizes team names by collapsing whitespace and applying the alias table.
6. Converts goal counts to integers; rows missing required fields are removed.
7. Normalizes `FTR` (H/D/A) to uppercase.
8. Converts available odds columns to numeric lowercase names.
9. Sorts by date, home team and away team.

`ingest/validate.py` then counts missing values, duplicates, impossible scores, invalid results and suspicious odds, and writes JSON. `summarize_matches` computes outcome rates and goal statistics.

## Team-name aliases

There are roughly 100 alias entries in `configs/default.yaml`. They map source spellings such as `Man Utd`, `Man City`, `Nott'm Forest`, `Sheffield Utd` and `Bournemouth` to canonical names. The alias table also contains many identity mappings. Whitespace is collapsed before alias lookup, so `  Man   Utd ` still maps to `Manchester United`.

## Quirks

- Historical seasons use different column layouts. Optional shot and odds columns appear in some seasons only.
- Pinnacle closing odds (`psch/pscd/psca`) are missing for a minority of rows (3.7%).
- Over/Under 2.5 and BTTS odds are absent for roughly 42% of seasons. Those market evaluations are therefore reported on the rows and seasons where the odds exist, and `evaluate_phase4` handles the failure when no odds are present for Over 2.5.
- The bookmaker probability conversion deduplicates missing odds per fixture and requires a complete three-way (home/draw/away) triplet; per-fixture fallback prefers Pinnacle, then Bet365, then average odds.

## Odds to probabilities

Bookmaker odds are converted proportionally: implied probability is `1 / decimal_odds`, and each row's implied probabilities are divided by their sum so the three outcomes add to one. This removes the overround and gives a no-vig baseline. Binary markets do the same with the two positive/negative odds. Odds at or below 1.0 are invalid and are excluded from the baseline.

## License and terms

The project source code is released under the MIT license (`LICENSE`). For the external CSV data, check the source's terms of use; this repository does not assert a redistribution right for that data.
