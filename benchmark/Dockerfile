# Dockerfile
# SwarmResearch Benchmark Reproducibility Container
# Version: 1.0.0

FROM python:3.11.7-slim-bookworm

LABEL maintainer="SwarmResearch Team"
LABEL version="1.0.0"
LABEL description="SwarmResearch Benchmark Reproducibility Container"

# Set environment variables for reproducibility
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42
ENV SWARMRESEARCH_SEED=42
ENV SWARMRESEARCH_BENCHMARK_VERSION=1.0.0
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /benchmark

# Copy requirements first for layer caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip==24.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy benchmark code and configuration
COPY seeds.yaml ./
COPY assets.yaml ./
COPY configs/ ./configs/
COPY reproduce.sh ./

# Create directories for assets
RUN mkdir -p /benchmark/models /benchmark/datasets /benchmark/results

# Set entrypoint
ENTRYPOINT ["python", "-m", "swarmresearch.benchmark"]
CMD ["--help"]
