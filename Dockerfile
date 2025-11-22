# RetentionAI Production Dockerfile
# Multi-stage build for optimized production deployment

# Stage 1: Base Python environment with dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r retentionai && useradd -r -g retentionai retentionai

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: Development environment (for local development)
FROM base as development

# Install development dependencies
RUN pip install \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    jupyter \
    notebook

# Copy the application code
COPY . .

# Change ownership to non-root user
RUN chown -R retentionai:retentionai /app

# Switch to non-root user
USER retentionai

# Expose ports
EXPOSE 8501 8502

# Default command for development
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Stage 3: Production environment
FROM base as production

# Install production-only dependencies
RUN pip install \
    gunicorn \
    uvicorn[standard] \
    prometheus-client

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY config.py ./
COPY pyproject.toml ./

# Create necessary directories
RUN mkdir -p /app/logs \
    /app/models \
    /app/data/raw \
    /app/data/processed \
    /app/mlruns \
    /app/artifacts

# Change ownership to non-root user
RUN chown -R retentionai:retentionai /app

# Switch to non-root user
USER retentionai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose ports
EXPOSE 8501

# Production entrypoint
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=true"]

# Stage 4: Training environment (for scheduled training jobs)
FROM base as training

# Install additional ML libraries for training
RUN pip install \
    optuna \
    shap \
    mlflow[extras] \
    hyperopt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY config.py ./

# Create training directories
RUN mkdir -p /app/training_outputs \
    /app/model_artifacts \
    /app/experiment_logs

# Change ownership to non-root user
RUN chown -R retentionai:retentionai /app

# Switch to non-root user
USER retentionai

# Training entrypoint
CMD ["python", "src/train.py"]

# Stage 5: API service (for REST API deployment)
FROM base as api

# Install FastAPI and dependencies
RUN pip install \
    fastapi \
    uvicorn[standard] \
    prometheus-fastapi-instrumentator \
    python-multipart

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY config.py ./

# Create API directories
RUN mkdir -p /app/api_logs

# Change ownership to non-root user
RUN chown -R retentionai:retentionai /app

# Switch to non-root user
USER retentionai

# Expose API port
EXPOSE 8000

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# API entrypoint (when FastAPI is implemented)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]