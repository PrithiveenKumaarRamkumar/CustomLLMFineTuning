# Multi-stage build for CustomLLM Inference API
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 apiuser && \
    mkdir -p /app /app/logs && \
    chown -R apiuser:apiuser /app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY serving/ ./serving/
COPY Data-Pipeline/configs/ ./configs/
COPY registry/ ./registry/

# Set ownership
RUN chown -R apiuser:apiuser /app

# Switch to non-root user
USER apiuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python3", "-m", "serving.api.main"]