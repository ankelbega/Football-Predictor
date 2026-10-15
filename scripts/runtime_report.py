"""Time the pipeline stages on this machine and write docs/RUNTIME.md.

Run from the repository root with:

    python scripts/runtime_report.py

The script does not overwrite existing result files: temporary directories are used
for download, validate, build-features, train and evaluate outputs.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cpu_model() -> str:
    system = platform.system()
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if system == "Windows":
        try:
            output = subprocess.check_output(
                ["wmic", "cpu", "get", "name"], text=True, stderr=subprocess.DEVNULL
            )
            lines = [line.strip() for line in output.splitlines() if line.strip() and line.strip().lower() != "name"]
            if lines:
                return lines[0]
        except Exception:
            pass
    return platform.processor() or "not available"


def ram_bytes() -> int | None:
    system = platform.system()
    if system == "Windows":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint),
                    ("dwMemoryLoad", ctypes.c_uint),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except Exception:
            return None
    if system == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    value = line.split()[1]
                    return int(value) * 1024
        except Exception:
            return None
    return None


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def run_timed(label: str, command: list[str], cwd: Path | None = None) -> float:
    print(f"Starting timed run: {label}")
    start = time.perf_counter()
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)
    elapsed = time.perf_counter() - start
    print(f"Finished {label}: {elapsed:.2f}s")
    return elapsed


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def time_api_startup(port: int) -> float:
    print("Starting timed run: api startup")
    start = time.perf_counter()
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "footballpred.api.app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.perf_counter() + 120.0
        while time.perf_counter() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2.0) as response:
                    if response.status == 200:
                        return time.perf_counter() - start
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("API did not become healthy within 120 seconds")
    finally:
        process.terminate()
        process.wait(timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "RUNTIME.md")
    args = parser.parse_args()

    print("Starting runtime report. The evaluate stage can take several minutes.")

    with tempfile.TemporaryDirectory(prefix="footballpred-runtime-") as temp_name:
        temp = Path(temp_name)
        (temp / "results").mkdir()
        (temp / "docs").mkdir()
        config_path = temp / "runtime.yaml"
        import yaml

        original = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
        config = yaml.safe_load(original)
        config["data"]["cache_dir"] = str(ROOT / "data" / "raw")
        config["data"]["processed_path"] = str(temp / "matches.csv")
        config["data"]["validation_report_path"] = str(temp / "validation.json")
        config["data"]["stats_path"] = str(temp / "summary.json")
        config["data"]["results_path"] = str(temp / "results" / "model-comparison.csv")
        config["data"]["model_path"] = str(temp / "model.joblib")
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        download_time = run_timed("download", ["footballpred", "download", "--config", str(config_path)])
        # CLI is installed as console script; fall back to module if needed. Use footballpred directly.
        validate_time = run_timed("validate", ["footballpred", "validate", "--config", str(config_path)])
        build_time = run_timed("build-features", ["footballpred", "build-features", "--config", str(config_path), "--output", str(temp / "features.csv")])
        train_time = run_timed("train", ["footballpred", "train", "--config", str(config_path)])
        evaluate_time = run_timed("evaluate", ["footballpred", "evaluate", "--config", str(config_path)])

        port = free_port()
        try:
            predict_time = run_timed(
                "predict", ["footballpred", "predict", "--home", "Arsenal", "--away", "Chelsea", "--date", "2026-08-01", "--config", str(config_path)]
            )
        except subprocess.CalledProcessError:
            predict_time = None

        api_time = time_api_startup(port)

    cores = os.cpu_count()
    memory = ram_bytes()
    memory_text = f"{memory / (1024 ** 3):.2f} GiB" if memory is not None else "not available"

    library_rows = []
    for package in ("pandas", "numpy", "scikit-learn", "lightgbm", "scipy", "fastapi", "typer", "uvicorn", "pydantic", "joblib"):
        library_rows.append((package, package_version(package)))

    lines = []
    lines.append("# Runtime report")
    lines.append("")
    lines.append("Timings were measured on this machine with `time.perf_counter` around the existing CLI/API entry points. Downloads used the local cache, and outputs were redirected to a temporary directory.")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- OS: {platform.platform()}")
    lines.append(f"- Python: {platform.python_version()} ({sys.version.split()[0]})")
    lines.append(f"- CPU model: {cpu_model()}")
    lines.append(f"- CPU cores: {cores}")
    lines.append(f"- RAM: {memory_text}")
    lines.append("")
    lines.append("## Package versions")
    lines.append("")
    lines.append("| package | version |")
    lines.append("|---|---|")
    for name, version in library_rows:
        lines.append(f"| {name} | {version} |")
    lines.append("")
    lines.append("## Timings")
    lines.append("")
    lines.append("| stage | seconds |")
    lines.append("|---|---:|")
    lines.append(f"| download (from cache) | {download_time:.2f} |")
    lines.append(f"| validate | {validate_time:.2f} |")
    lines.append(f"| build-features | {build_time:.2f} |")
    lines.append(f"| train | {train_time:.2f} |")
    lines.append(f"| evaluate | {evaluate_time:.2f} |")
    lines.append("| predict | " + (f"{predict_time:.2f}" if predict_time is not None else "not available") + " |")
    lines.append(f"| api startup to /health | {api_time:.2f} |")
    lines.append("")
    lines.append("Timings depend on the machine, cache state and package build. Reproduce with `make runtime-report` or `python scripts/runtime_report.py`.")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote runtime report to {args.output}")


if __name__ == "__main__":
    main()
