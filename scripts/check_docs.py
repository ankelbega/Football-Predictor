"""Check documentation links, referenced paths and placeholder words.

Run from the repository root with:

    python scripts/check_docs.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md"))


def relative_links(text: str) -> list[str]:
    return re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)


def looks_like_path(token: str) -> bool:
    return (
        "\n" not in token
        and " " not in token
        and re.fullmatch(r"[A-Za-z0-9_./\\-]+\.(?:md|py|csv|png|yaml|yml|json|joblib)", token) is not None
    )


def placeholders(text: str) -> list[str]:
    return [word for word in ("TODO", "TBD", "XXX") if re.search(r"\b" + word + r"\b", text)]


def check_links(path: Path, text: str) -> list[str]:
    errors = []
    for link in relative_links(text):
        if link.startswith("http") or link.startswith("#"):
            continue
        target = (path.parent / link.split("#")[0]).resolve()
        if not target.exists():
            errors.append(f"{path.relative_to(ROOT)}: missing link target {link}")
    return errors


def check_paths(path: Path, text: str) -> list[str]:
    errors = []
    for token in re.findall(r"`([^`]+)`", text):
        if looks_like_path(token):
            candidates = [
                (ROOT / token).resolve(),
                (ROOT / "src" / "footballpred" / token).resolve(),
                (path.parent / token).resolve(),
            ]
            if "/" not in token and "\\" not in token:
                candidates.extend(path.resolve() for path in ROOT.rglob(token))
            if not any(candidate.exists() for candidate in candidates):
                errors.append(f"{path.relative_to(ROOT)}: missing referenced path {token}")
    return errors


def check_commands(text: str) -> list[str]:
    errors = []
    commands = set(re.findall(r"footballpred\s+([a-z][a-z-]*)", text))
    for command in sorted(commands):
        result = subprocess.run(
            ["footballpred", command, "--help"], capture_output=True, text=True
        )
        if result.returncode != 0:
            errors.append(f"footballpred {command} --help failed")
    return errors


def check_placeholders(path: Path, text: str) -> list[str]:
    found = placeholders(text)
    return [f"{path.relative_to(ROOT)}: placeholder {word}" for word in found]


def check_numbers() -> list[str]:
    import pandas as pd

    errors = []
    matches = pd.read_csv(ROOT / "data" / "processed" / "matches.csv", parse_dates=["date"])
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    expected = f"{len(matches):,}"
    if "4,560" in readme and expected != "4,560":
        errors.append(f"README match count {expected} differs from 4,560")
    if "2014-08-16" not in readme or str(matches["date"].min().date()) != "2014-08-16":
        errors.append("README start date does not match processed data")
    if "2026-05-24" not in readme or str(matches["date"].max().date()) != "2026-05-24":
        errors.append("README end date does not match processed data")
    return errors


def main() -> int:
    errors = []
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        errors.extend(check_links(path, text))
        errors.extend(check_paths(path, text))
        errors.extend(check_placeholders(path, text))
        errors.extend(check_commands(text))
    errors.extend(check_numbers())

    if errors:
        for error in errors:
            print(error)
        return 1
    print("All documentation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
