"""Generate the web Performance page's data file from committed result tables."""

from __future__ import annotations

import csv
import datetime
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "web" / "src" / "data" / "performance.json"

CSV_PATHS = {
    "phase4": RESULTS / "phase4-model-comparison.csv",
    "phase3": RESULTS / "phase3-market-comparison.csv",
    "walk_forward": RESULTS / "walk-forward-test-results.csv",
    "shap": RESULTS / "shap-global-importance.csv",
    "calibration": RESULTS / "lightgbm-calibration-validation.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one committed result table into plain dictionaries."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def clean(value: str | None) -> float | str | None:
    """Convert numeric strings while preserving missing values."""
    if value in (None, "", "nan", "NaN", "None"):
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    if math.isnan(number):
        return None
    return number


def build() -> dict[str, object]:
    """Assemble only the real result rows needed by the Performance view."""
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sources": {name: path.name for name, path in CSV_PATHS.items()},
        "phase4_model_comparison": [
            {key: clean(value) for key, value in row.items()}
            for row in read_csv(CSV_PATHS["phase4"])
        ],
        "phase3_market_comparison": [
            {key: clean(value) for key, value in row.items()}
            for row in read_csv(CSV_PATHS["phase3"])
        ],
        "walk_forward_test_results": [
            {key: clean(value) for key, value in row.items()}
            for row in read_csv(CSV_PATHS["walk_forward"])
        ],
        "global_shap_importance": [
            {key: clean(value) for key, value in row.items()}
            for row in read_csv(CSV_PATHS["shap"])
        ],
        "calibration_validation": [
            {key: clean(value) for key, value in row.items()}
            for row in read_csv(CSV_PATHS["calibration"])
        ],
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build(), indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
