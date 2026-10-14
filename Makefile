.PHONY: install download validate build-features train evaluate api test lint all

# Install the editable package and developer tools (pytest and Ruff).
install:
	python -m pip install -e ".[dev]"

# Download configured CSVs into the local raw cache and write normalized matches.
download:
	footballpred download

# Write the quality report and basic descriptive data summary.
validate:
	footballpred validate

# Build leakage-safe numeric pre-match features for inspection.
build-features:
	footballpred build-features

# Fit serving models and create the ignored local model bundle.
train:
	footballpred train

# Run validation tuning and held-out walk-forward comparisons.
evaluate:
	footballpred evaluate

# Start the already-trained API; does not retrain the bundle.
api:
	uvicorn footballpred.api.app:app --host 0.0.0.0 --port 8000

# Run the complete pytest suite.
test:
	pytest -q

# Run static lint checks on source and tests.
lint:
	ruff check src tests

# Reproduce the pipeline stages in dependency order from cached/free source data.
all: download validate build-features evaluate train
