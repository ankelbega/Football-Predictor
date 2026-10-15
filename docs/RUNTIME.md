# Runtime report

Timings were measured on this machine with `time.perf_counter` around the existing CLI/API entry points. Downloads used the local cache, and outputs were redirected to a temporary directory.

## Environment

- OS: Windows-10-10.0.19045-SP0
- Python: 3.13.2 (3.13.2)
- CPU model: Intel(R) Core(TM) i5-8350U CPU @ 1.70GHz
- CPU cores: 8
- RAM: 15.85 GiB

## Package versions

| package | version |
|---|---|
| pandas | 2.3.3 |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| lightgbm | 4.7.0 |
| scipy | 1.18.1 |
| fastapi | 0.143.1 |
| typer | 0.27.3 |
| uvicorn | 0.54.0 |
| pydantic | 2.14.1 |
| joblib | 1.6.0 |

## Timings

| stage | seconds |
|---|---:|
| download (from cache) | 4.02 |
| validate | 2.98 |
| build-features | 6.45 |
| train | 41.78 |
| evaluate | 195.62 |
| predict | 6.91 |
| api startup to /health | 3.12 |

Timings depend on the machine, cache state and package build. Reproduce with `make runtime-report` or `python scripts/runtime_report.py`.
