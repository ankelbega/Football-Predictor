"""Print basic statistics about the processed match data.

Run from the repository root with:

    python scripts/data_stats.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    settings_path = ROOT / "configs" / "default.yaml"
    import yaml

    with (settings_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    data = config["data"]

    processed = pd.read_csv(ROOT / data["processed_path"], parse_dates=["date"])
    raw_dir = ROOT / data["cache_dir"]

    print("Source URL pattern:", data["source_url"])
    print("Leagues configured:", ", ".join(data["leagues"]))
    print("Seasons configured:", ", ".join(data["seasons"]))
    print("Processed rows:", len(processed))
    print("Teams:", pd.concat([processed["home_team"], processed["away_team"]]).nunique())
    print("Date range:", processed["date"].min().date(), "to", processed["date"].max().date())
    print()

    print("Rows and date range by season:")
    grouped = processed.groupby("season").agg(
        matches=("date", "size"), first_date=("date", "min"), last_date=("date", "max")
    )
    print(grouped.to_string())
    print()

    print("Rows by league:")
    print(processed.groupby("league").size().to_string())
    print()

    print("Missing values (count and rate) for columns present in the processed data:")
    for column in processed.columns:
        missing = int(processed[column].isna().sum())
        rate = missing / len(processed) if len(processed) else 0.0
        print(f"  {column}: {missing} ({rate:.4f})")
    print()

    print("Raw cached source files available:")
    for path in sorted(raw_dir.glob("*.csv")):
        print("  ", path.name, path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
