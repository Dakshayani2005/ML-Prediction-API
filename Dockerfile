# ---------- Builder Stage ----------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies (if needed for ML libs like numpy, tensorflow)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt


# ---------- Final Stage ----------
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Copy project files
COPY src/ ./src/
COPY models/ ./models/

ENV MODEL_PATH=/app/models/my_classifier_model.h5

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
