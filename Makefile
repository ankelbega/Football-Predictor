.PHONY: install download validate build-features train evaluate api test lint all

install:
	python -m pip install -e ".[dev]"

download:
	footballpred download

validate:
	footballpred validate

build-features:
	footballpred build-features

train:
	footballpred train

evaluate:
	footballpred evaluate

api:
	uvicorn footballpred.api.app:app --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check src tests

all: download validate build-features evaluate train
