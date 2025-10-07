# Multi-stage Docker build for OpenPostTrainingOptimizations
# Author: Nik Jois

# Build stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./
COPY src ./src
COPY README.md CHANGELOG.md LICENSE ./

# Install uv and build
RUN pip install --no-cache-dir uv hatchling
RUN uv pip install --system -e ".[serve,agents,data]"

# Runtime stage
FROM python:3.11-slim as runtime

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser README.md CHANGELOG.md LICENSE ./

# Create directories
RUN mkdir -p /app/data /app/outputs && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/usr/local/bin:${PATH}"

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9000

# Default command
CMD ["opt", "status"]

# Development stage
FROM runtime as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    vim \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install dev dependencies
COPY --from=builder /build/pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev,serve,agents,data]"

USER appuser

CMD ["/bin/bash"]

