# Multi-stage Dockerfile for DataPipelineEnv
# Supports both development and production deployment

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Production
FROM python:3.11-slim as production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy project files
COPY src/ ./src/
COPY inference.py .
COPY app.py .
COPY openenv.yaml .
COPY README.md .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

# Stage 3: Development (optional, use with --target development)
FROM production as development

USER root

# Install development dependencies
COPY requirements*.txt ./
RUN if [ -f requirements-dev.txt ]; then pip install --user --no-cache-dir -r requirements-dev.txt; fi

# Install package in editable mode
RUN pip install --user -e .

USER appuser

# Expose port for web interface (if added later)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.openenv_project import DataPipelineEnv; env = DataPipelineEnv(); env.reset(); print('Healthy')" || exit 1