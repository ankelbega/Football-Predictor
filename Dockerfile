FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN python -m pip install --upgrade pip && python -m pip install .

# Provide the trained bundle at build/runtime through a mounted ./models directory.
EXPOSE 8000
CMD ["uvicorn", "footballpred.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
