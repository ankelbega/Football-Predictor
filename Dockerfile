# Use Python 3.11 to match the package's supported runtime.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
# Install source and configuration; do not bake raw data or trained weights into the image.
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN python -m pip install --upgrade pip && python -m pip install .

# Mount the local ignored model bundle at runtime; no training happens in the container.
EXPOSE 8000
CMD ["uvicorn", "footballpred.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
