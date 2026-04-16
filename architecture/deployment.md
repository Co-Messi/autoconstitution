# SwarmResearch Deployment Architecture

## Comprehensive Deployment Strategy for Open Source ML Projects

---

## Executive Summary

This document outlines the complete deployment architecture for SwarmResearch, a massively parallel collaborative AI research system. The deployment strategy follows cloud-native best practices, enabling seamless scaling from development (Mac Mini M4) to production GPU clusters while maintaining simplicity for open source contributors.

---

## 1. Containerization Strategy

### 1.1 Multi-Stage Docker Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTAINERIZATION STRATEGY                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  BASE IMAGES HIERARCHY                                              │   │
│  │                                                                     │   │
│  │  ┌─────────────────┐                                                │   │
│  │  │  python:3.11-slim│  ← Official Python slim image                 │   │
│  │  │  (Security scan)│     ~120MB                                     │   │
│  │  └────────┬────────┘                                                │   │
│  │           │                                                         │   │
│  │           ▼                                                         │   │
│  │  ┌─────────────────┐                                                │   │
│  │  │  swarm-base      │  ← Common dependencies                        │   │
│  │  │  (Build deps)    │     uv, gcc, git, curl                        │   │
│  │  └────────┬────────┘     ~180MB                                     │   │
│  │           │                                                         │   │
│  │     ┌─────┴─────┬─────────────┐                                     │   │
│  │     │           │             │                                     │   │
│  │     ▼           ▼             ▼                                     │   │
│  │  ┌───────┐  ┌───────┐   ┌─────────┐                                │   │
│  │  │swarm- │  │swarm- │   │ swarm-  │                                │   │
│  │  │api    │  │worker │   │ollama   │                                │   │
│  │  │~250MB │  │~300MB │   │~2GB     │                                │   │
│  │  └───┬───┘  └───┬───┘   └────┬────┘                                │   │
│  │      │          │            │                                     │   │
│  │      └──────────┴────────────┘                                     │   │
│  │                 │                                                  │   │
│  │                 ▼                                                  │   │
│  │         ┌─────────────┐                                            │   │
│  │         │swarm-full   │  ← All-in-one for development              │   │
│  │         │~2.5GB       │                                            │   │
│  │         └─────────────┘                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Dockerfiles

#### Base Image (`docker/Dockerfile.base`)

```dockerfile
# syntax=docker/dockerfile:1.6
FROM python:3.11-slim-bookworm AS swarm-base

LABEL org.opencontainers.image.source="https://github.com/swarmresearch/swarm-research"
LABEL org.opencontainers.image.description="SwarmResearch Base Image"
LABEL org.opencontainers.image.licenses="MIT"

# Security: Run as non-root
RUN groupadd -r swarm && useradd -r -g swarm swarm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (layer caching)
RUN uv pip install --system --no-cache -e ".[base]"

# Switch to non-root user
USER swarm

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import swarm_research; print('OK')" || exit 1

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
```

#### API Server Image (`docker/Dockerfile.api`)

```dockerfile
# syntax=docker/dockerfile:1.6
FROM swarmresearch/swarm-base:latest AS swarm-api

LABEL org.opencontainers.image.source="https://github.com/swarmresearch/swarm-research"
LABEL org.opencontainers.image.description="SwarmResearch API Server"

USER root

# Install API-specific dependencies
RUN uv pip install --system --no-cache -e ".[api]"

# Copy application code
COPY --chown=swarm:swarm src/ ./src/
COPY --chown=swarm:swarm config/ ./config/

USER swarm

# Expose API port
EXPOSE 8000

# Metrics port for Prometheus
EXPOSE 9090

# Entrypoint
ENTRYPOINT ["uvicorn"]
CMD ["swarm_research.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### Worker Image (`docker/Dockerfile.worker`)

```dockerfile
# syntax=docker/dockerfile:1.6
FROM swarmresearch/swarm-base:latest AS swarm-worker

LABEL org.opencontainers.image.source="https://github.com/swarmresearch/swarm-research"
LABEL org.opencontainers.image.description="SwarmResearch Worker"

USER root

# Install worker-specific dependencies
RUN uv pip install --system --no-cache -e ".[worker]"

# Copy application code
COPY --chown=swarm:swarm src/ ./src/

USER swarm

# Worker doesn't expose ports (message-driven)
# Metrics port for Prometheus
EXPOSE 9090

# Entrypoint
ENTRYPOINT ["python", "-m", "swarm_research.worker"]
CMD ["--concurrency", "10"]
```

#### Ollama Image (`docker/Dockerfile.ollama`)

```dockerfile
# syntax=docker/dockerfile:1.6
FROM ollama/ollama:latest AS swarm-ollama

LABEL org.opencontainers.image.source="https://github.com/swarmresearch/swarm-research"
LABEL org.opencontainers.image.description="SwarmResearch with Ollama"

# Pre-pull common models for faster startup
RUN ollama serve & \
    sleep 5 && \
    ollama pull mistral:7b && \
    ollama pull nomic-embed-text && \
    pkill ollama

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ollama list || exit 1

EXPOSE 11434

ENTRYPOINT ["ollama"]
CMD ["serve"]
```

#### Development Image (`docker/Dockerfile.dev`)

```dockerfile
# syntax=docker/dockerfile:1.6
FROM swarmresearch/swarm-base:latest AS swarm-dev

LABEL org.opencontainers.image.source="https://github.com/swarmresearch/swarm-research"
LABEL org.opencontainers.image.description="SwarmResearch Development Environment"

USER root

# Install development dependencies
RUN uv pip install --system --no-cache -e ".[dev,all]"

# Install additional dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    htop \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy full source with tests
COPY --chown=swarm:swarm . ./

USER swarm

# Default to shell for development
CMD ["/bin/bash"]
```

### 1.3 Docker Compose Configurations

#### Development (`docker-compose.dev.yml`)

```yaml
version: "3.8"

services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
      target: swarm-api
    ports:
      - "8000:8000"
      - "9090:9090"
    environment:
      - SWARM_ENV=development
      - SWARM_LOG_LEVEL=debug
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://swarm:swarm@postgres:5432/swarm
    volumes:
      - ./src:/app/src:ro
      - ./config:/app/config:ro
    depends_on:
      - redis
      - postgres
      - ollama
    networks:
      - swarm-net

  worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    environment:
      - SWARM_ENV=development
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://swarm:swarm@postgres:5432/swarm
    depends_on:
      - redis
      - postgres
      - ollama
    deploy:
      replicas: 2
    networks:
      - swarm-net

  ollama:
    build:
      context: .
      dockerfile: docker/Dockerfile.ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    networks:
      - swarm-net

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - swarm-net

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=swarm
      - POSTGRES_PASSWORD=swarm
      - POSTGRES_DB=swarm
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - swarm-net

  # Development tools
  jupyter:
    build:
      context: .
      dockerfile: docker/Dockerfile.dev
    ports:
      - "8888:8888"
    volumes:
      - .:/app
      - jupyter-data:/home/swarm/.jupyter
    command: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    networks:
      - swarm-net

volumes:
  ollama-data:
  redis-data:
  postgres-data:
  jupyter-data:

networks:
  swarm-net:
    driver: bridge
```

#### Production (`docker-compose.prod.yml`)

```yaml
version: "3.8"

services:
  api:
    image: swarmresearch/swarm-api:${SWARM_VERSION:-latest}
    ports:
      - "8000:8000"
    environment:
      - SWARM_ENV=production
      - SWARM_LOG_LEVEL=info
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=${DATABASE_URL}
      - SECRET_KEY=${SECRET_KEY}
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
      rollback_config:
        parallelism: 1
        delay: 10s
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - swarm-net

  worker:
    image: swarmresearch/swarm-worker:${SWARM_VERSION:-latest}
    environment:
      - SWARM_ENV=production
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=${DATABASE_URL}
    deploy:
      replicas: 10
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
    depends_on:
      - redis
    networks:
      - swarm-net

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    deploy:
      resources:
        limits:
          memory: 2G
    networks:
      - swarm-net

  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    networks:
      - swarm-net

  # Grafana for dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./config/grafana:/etc/grafana/provisioning:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    networks:
      - swarm-net

volumes:
  redis-data:
  prometheus-data:
  grafana-data:

networks:
  swarm-net:
    driver: overlay
    attachable: true
```

### 1.4 Build Scripts

```bash
#!/bin/bash
# scripts/build-images.sh

set -e

VERSION=${1:-$(git describe --tags --always)}
REGISTRY=${REGISTRY:-swarmresearch}

echo "Building SwarmResearch images version: $VERSION"

# Build base image
docker build \
    -f docker/Dockerfile.base \
    -t $REGISTRY/swarm-base:$VERSION \
    -t $REGISTRY/swarm-base:latest \
    .

# Build API image
docker build \
    -f docker/Dockerfile.api \
    -t $REGISTRY/swarm-api:$VERSION \
    -t $REGISTRY/swarm-api:latest \
    .

# Build worker image
docker build \
    -f docker/Dockerfile.worker \
    -t $REGISTRY/swarm-worker:$VERSION \
    -t $REGISTRY/swarm-worker:latest \
    .

# Build Ollama image
docker build \
    -f docker/Dockerfile.ollama \
    -t $REGISTRY/swarm-ollama:$VERSION \
    -t $REGISTRY/swarm-ollama:latest \
    .

# Build dev image
docker build \
    -f docker/Dockerfile.dev \
    -t $REGISTRY/swarm-dev:$VERSION \
    -t $REGISTRY/swarm-dev:latest \
    .

echo "Build complete. Images tagged with: $VERSION"

# Security scan
echo "Running security scans..."
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image $REGISTRY/swarm-api:$VERSION
```

---

## 2. CI/CD Pipeline

### 2.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CI/CD PIPELINE ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   Code      │───▶│    CI       │───▶│    CD       │───▶│  Deploy     │   │
│  │   Push      │    │  Pipeline   │    │  Pipeline   │    │  Target     │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │  Trigger    │    │  Build &    │    │  Artifact   │    │  Staging/   │   │
│  │  (GitHub    │    │  Test       │    │  Publish    │    │  Production │   │
│  │  Events)    │    │             │    │             │    │             │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                              │
│  Triggers:                    Stages:                                        │
│  • Push to main              ┌─────────────────────────────────────────┐    │
│  • Pull request              │  1. Lint & Format (ruff, mypy)          │    │
│  • Tag push (v*)             │  2. Unit Tests (pytest)                 │    │
│  • Scheduled (nightly)       │  3. Integration Tests                   │    │
│  • Manual dispatch           │  4. Security Scan (bandit, trivy)       │    │
│                              │  5. Build Images                        │    │
│                              │  6. Push to Registry                    │    │
│                              │  7. Deploy to Staging                   │    │
│                              │  8. E2E Tests                           │    │
│                              │  9. Deploy to Production (tagged only)  │    │
│                              └─────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 GitHub Actions Workflows

#### Main CI Workflow (`.github/workflows/ci.yml`)

```yaml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
    tags: ['v*']
  pull_request:
    branches: [main, develop]
  schedule:
    - cron: '0 2 * * *'  # Nightly at 2 AM UTC

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

env:
  PYTHON_VERSION: "3.11"
  UV_VERSION: "0.4.x"

jobs:
  # ============ Stage 1: Lint & Format ============
  lint:
    name: Lint & Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: ${{ env.UV_VERSION }}
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: uv pip install -e ".[dev]"
      
      - name: Run ruff (lint)
        run: ruff check src/ tests/
      
      - name: Run ruff (format check)
        run: ruff format --check src/ tests/
      
      - name: Run mypy
        run: mypy src/
      
      - name: Run bandit (security)
        run: bandit -r src/ -f json -o bandit-report.json || true
      
      - name: Upload security report
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: security-report
          path: bandit-report.json

  # ============ Stage 2: Unit Tests ============
  test:
    name: Unit Tests
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      
      - name: Setup Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: uv pip install -e ".[dev,test]"
      
      - name: Run tests with coverage
        run: |
          pytest tests/unit \
            --cov=swarm_research \
            --cov-report=xml \
            --cov-report=html \
            -v
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
      
      - name: Upload coverage report
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report-${{ matrix.python-version }}
          path: htmlcov/

  # ============ Stage 3: Integration Tests ============
  integration-test:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: [lint, test]
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: swarm
          POSTGRES_PASSWORD: swarm
          POSTGRES_DB: swarm_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: uv pip install -e ".[dev,test,all]"
      
      - name: Run integration tests
        env:
          REDIS_URL: redis://localhost:6379
          DATABASE_URL: postgresql://swarm:swarm@localhost:5432/swarm_test
        run: |
          pytest tests/integration -v --timeout=300

  # ============ Stage 4: Build & Push Images ============
  build-images:
    name: Build Docker Images
    runs-on: ubuntu-latest
    needs: [integration-test]
    if: github.event_name != 'pull_request'
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Login to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: |
            swarmresearch/swarm-api
            ghcr.io/${{ github.repository }}/swarm-api
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix=,suffix=,format=short
      
      - name: Build and push API image
        uses: docker/build-push-action@v6
        with:
          context: .
          file: ./docker/Dockerfile.api
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: swarmresearch/swarm-api:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

  # ============ Stage 5: Deploy to Staging ============
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [build-images]
    if: github.ref == 'refs/heads/develop'
    environment:
      name: staging
      url: https://staging.swarmresearch.io
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to staging
        run: |
          echo "Deploying to staging environment..."
          # kubectl or helm deployment
      
      - name: Run smoke tests
        run: |
          curl -f https://staging.swarmresearch.io/health || exit 1

  # ============ Stage 6: Deploy to Production ============
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [build-images]
    if: startsWith(github.ref, 'refs/tags/v')
    environment:
      name: production
      url: https://api.swarmresearch.io
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to production
        run: |
          echo "Deploying to production environment..."
          # kubectl or helm deployment
      
      - name: Verify deployment
        run: |
          curl -f https://api.swarmresearch.io/health || exit 1
```

#### Release Workflow (`.github/workflows/release.yml`)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: write
  packages: write

jobs:
  # ============ Create GitHub Release ============
  github-release:
    name: Create GitHub Release
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Generate changelog
        id: changelog
        run: |
          echo "changelog<<EOF" >> $GITHUB_OUTPUT
          git log --pretty=format:"- %s" $(git describe --tags --abbrev=0 HEAD~1)..HEAD >> $GITHUB_OUTPUT
          echo "" >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT
      
      - name: Create Release
        uses: softprops/action-gh-release@v2
        with:
          body: |
            ## Changes
            ${{ steps.changelog.outputs.changelog }}
            
            ## Docker Images
            - `swarmresearch/swarm-api:${{ github.ref_name }}`
            - `swarmresearch/swarm-worker:${{ github.ref_name }}`
            
            ## PyPI Package
            ```bash
            pip install swarm-research==${{ github.ref_name }}
            ```
          draft: false
          prerelease: ${{ contains(github.ref_name, 'rc') || contains(github.ref_name, 'beta') || contains(github.ref_name, 'alpha') }}

  # ============ Publish to PyPI ============
  pypi-publish:
    name: Publish to PyPI
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      - name: Build package
        run: |
          uv pip install build twine
          python -m build
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

### 2.3 Kubernetes Deployment

```yaml
# k8s/base/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: swarm-research
  labels:
    app.kubernetes.io/name: swarm-research
    app.kubernetes.io/managed-by: kubectl

---
# k8s/base/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: swarm-config
  namespace: swarm-research
data:
  SWARM_ENV: "production"
  SWARM_LOG_LEVEL: "info"
  REDIS_URL: "redis://redis:6379"

---
# k8s/base/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: swarm-secrets
  namespace: swarm-research
type: Opaque
stringData:
  DATABASE_URL: "postgresql://..."
  SECRET_KEY: "..."
  KIMI_API_KEY: "..."
  ANTHROPIC_API_KEY: "..."
  OPENAI_API_KEY: "..."

---
# k8s/base/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: swarm-api
  namespace: swarm-research
  labels:
    app: swarm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: swarm-api
  template:
    metadata:
      labels:
        app: swarm-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 999
        fsGroup: 999
      containers:
        - name: api
          image: swarmresearch/swarm-api:latest
          ports:
            - containerPort: 8000
              name: http
            - containerPort: 9090
              name: metrics
          envFrom:
            - configMapRef:
                name: swarm-config
            - secretRef:
                name: swarm-secrets
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop:
                - ALL

---
# k8s/base/api-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: swarm-api
  namespace: swarm-research
spec:
  selector:
    app: swarm-api
  ports:
    - port: 80
      targetPort: 8000
      name: http
    - port: 9090
      targetPort: 9090
      name: metrics
  type: ClusterIP

---
# k8s/base/api-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: swarm-api-hpa
  namespace: swarm-research
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: swarm-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60

---
# k8s/base/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: swarm-ingress
  namespace: swarm-research
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
    - hosts:
        - api.swarmresearch.io
      secretName: swarm-tls
  rules:
    - host: api.swarmresearch.io
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: swarm-api
                port:
                  number: 80
```

### 2.4 Helm Chart

```yaml
# helm/swarm-research/Chart.yaml
apiVersion: v2
name: swarm-research
description: A Helm chart for SwarmResearch
type: application
version: 0.1.0
appVersion: "0.1.0"
keywords:
  - ai
  - ml
  - multi-agent
  - research
home: https://swarmresearch.io
sources:
  - https://github.com/swarmresearch/swarm-research
maintainers:
  - name: SwarmResearch Team
    email: team@swarmresearch.io

dependencies:
  - name: redis
    version: 18.x.x
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled
  - name: postgresql
    version: 13.x.x
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled
```

```yaml
# helm/swarm-research/values.yaml
# Default values for swarm-research

replicaCount:
  api: 3
  worker: 10

image:
  repository: swarmresearch/swarm-api
  pullPolicy: IfNotPresent
  tag: ""

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "9090"

podSecurityContext:
  runAsNonRoot: true
  runAsUser: 999
  fsGroup: 999

securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop:
      - ALL

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.swarmresearch.io
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: swarm-tls
      hosts:
        - api.swarmresearch.io

resources:
  api:
    requests:
      memory: 512Mi
      cpu: 500m
    limits:
      memory: 2Gi
      cpu: 2000m
  worker:
    requests:
      memory: 256Mi
      cpu: 250m
    limits:
      memory: 1Gi
      cpu: 1000m

autoscaling:
  enabled: true
  api:
    minReplicas: 3
    maxReplicas: 20
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80
  worker:
    minReplicas: 5
    maxReplicas: 100
    targetCPUUtilizationPercentage: 60
    targetMemoryUtilizationPercentage: 70

nodeSelector: {}

tolerations: []

affinity: {}

# Configuration
config:
  logLevel: info
  maxAgents: 100
  defaultProvider: kimi
  
# Secrets (should be provided separately)
secrets:
  databaseUrl: ""
  secretKey: ""
  kimiApiKey: ""
  anthropicApiKey: ""
  openaiApiKey: ""

# Subcharts
redis:
  enabled: true
  auth:
    enabled: false
  master:
    persistence:
      enabled: true
      size: 8Gi

postgresql:
  enabled: true
  auth:
    username: swarm
    database: swarm
    existingSecret: ""
  primary:
    persistence:
      enabled: true
      size: 10Gi
```

---

## 3. Release Management

### 3.1 Versioning Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SEMANTIC VERSIONING STRATEGY                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Version Format: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]                     │
│                                                                              │
│  Examples:                                                                   │
│  • 0.1.0      - Initial release                                              │
│  • 0.2.0      - New features, backward compatible                            │
│  • 0.2.1      - Bug fix                                                      │
│  • 1.0.0      - Stable API release                                           │
│  • 1.1.0-rc.1 - Release candidate                                            │
│  • 2.0.0-beta.1 - Beta for breaking changes                                  │
│                                                                              │
│  Version Bump Rules:                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  MAJOR ↑  - Breaking API changes, incompatible protocol updates    │    │
│  │  MINOR ↑  - New features, backward compatible                      │    │
│  │  PATCH ↑  - Bug fixes, security patches                            │    │
│  │  PRERELEASE - Alpha, beta, rc for testing                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Branch Strategy:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  main        - Production releases (tagged)                        │    │
│  │  develop     - Integration branch, deploys to staging              │    │
│  │  feature/*   - Feature development                                 │    │
│  │  hotfix/*    - Emergency production fixes                          │    │
│  │  release/*   - Release preparation                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Release Process

```bash
#!/bin/bash
# scripts/release.sh

set -e

VERSION=$1
RELEASE_TYPE=${2:-minor}  # major, minor, patch

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version> [major|minor|patch]"
    echo "Example: $0 0.2.0 minor"
    exit 1
fi

echo "🚀 Preparing release $VERSION"

# 1. Ensure clean working directory
if [ -n "$(git status --porcelain)" ]; then
    echo "❌ Working directory is not clean"
    exit 1
fi

# 2. Create release branch
git checkout -b release/$VERSION

# 3. Update version in files
sed -i "s/version = \"[^\"]*\"/version = \"$VERSION\"/" pyproject.toml
sed -i "s/version: [^\n]*/version: $VERSION/" helm/swarm-research/Chart.yaml
sed -i "s/appVersion: \"[^\"]*\"/appVersion: \"$VERSION\"/" helm/swarm-research/Chart.yaml

# 4. Update CHANGELOG
echo "## [$VERSION] - $(date +%Y-%m-%d)" > CHANGELOG.tmp
git log --pretty=format:"- %s" $(git describe --tags --abbrev=0)..HEAD >> CHANGELOG.tmp
echo "" >> CHANGELOG.tmp
cat CHANGELOG.md >> CHANGELOG.tmp
mv CHANGELOG.tmp CHANGELOG.md

# 5. Commit changes
git add pyproject.toml helm/swarm-research/Chart.yaml CHANGELOG.md
git commit -m "chore(release): prepare $VERSION"

# 6. Run tests
echo "🧪 Running tests..."
pytest tests/ -x -q

# 7. Build and test package
echo "📦 Building package..."
python -m build
twine check dist/*

# 8. Merge to main
git checkout main
git merge --no-ff release/$VERSION -m "Release $VERSION"

# 9. Tag release
git tag -a v$VERSION -m "Release $VERSION"

# 10. Push
git push origin main
git push origin v$VERSION

# 11. Merge back to develop
git checkout develop
git merge main
git push origin develop

# 12. Clean up
git branch -d release/$VERSION

echo "✅ Release $VERSION complete!"
echo ""
echo "Next steps:"
echo "  1. GitHub Actions will build and publish"
echo "  2. Verify deployment at https://api.swarmresearch.io"
echo "  3. Create GitHub release notes"
```

### 3.3 Release Checklist

```markdown
# Release Checklist Template

## Pre-Release
- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in pyproject.toml
- [ ] Version bumped in Chart.yaml
- [ ] Security scan passed
- [ ] Performance benchmarks acceptable

## Release
- [ ] Create release branch
- [ ] Run full test suite
- [ ] Build and verify package
- [ ] Merge to main
- [ ] Tag release
- [ ] Push to origin

## Post-Release
- [ ] Verify PyPI publication
- [ ] Verify Docker Hub images
- [ ] Verify GitHub release created
- [ ] Verify staging deployment
- [ ] Verify production deployment (if applicable)
- [ ] Announce release
- [ ] Monitor error rates
```

---

## 4. Distribution Strategy

### 4.1 PyPI Distribution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PYPI DISTRIBUTION STRATEGY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Package Structure:                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  swarm-research/                                                    │    │
│  │  ├── pyproject.toml        ← Package metadata                      │    │
│  │  ├── README.md             ← PyPI description                      │    │
│  │  ├── LICENSE               ← MIT License                           │    │
│  │  ├── src/                                                         │    │
│  │  │   └── swarm_research/                                          │    │
│  │  │       ├── __init__.py                                          │    │
│  │  │       ├── api.py               ← FastAPI app                   │    │
│  │  │       ├── orchestrator.py      ← Core orchestration            │    │
│  │  │       ├── providers/           ← LLM provider adapters         │    │
│  │  │       ├── agents/              ← Agent implementations         │    │
│  │  │       └── ...                                                  │    │
│  │  └── tests/                                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Installation Methods:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  # Basic installation                                               │    │
│  │  pip install swarm-research                                         │    │
│  │                                                                     │    │
│  │  # With all providers                                               │    │
│  │  pip install swarm-research[all]                                    │    │
│  │                                                                     │    │
│  │  # With specific providers                                          │    │
│  │  pip install swarm-research[openai,anthropic]                       │    │
│  │                                                                     │    │
│  │  # Development installation                                         │    │
│  │  pip install swarm-research[dev]                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Extras Configuration:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  [project.optional-dependencies]                                    │    │
│  │  base = ["httpx>=0.25", "pydantic>=2.0"]                            │    │
│  │  openai = ["openai>=1.0"]                                           │    │
│  │  anthropic = ["anthropic>=0.20"]                                    │    │
│  │  kimi = ["openai>=1.0"]  # Uses OpenAI-compatible API               │    │
│  │  ollama = ["ollama>=0.1"]                                           │    │
│  │  api = ["fastapi>=0.100", "uvicorn[standard]>=0.23"]                │    │
│  │  worker = ["celery>=5.3", "redis>=5.0"]                             │    │
│  │  dev = ["pytest>=7.0", "ruff>=0.1", "mypy>=1.0"]                    │    │
│  │  all = ["swarm-research[openai,anthropic,kimi,ollama,api,worker]"]  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### pyproject.toml Configuration

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "swarm-research"
version = "0.1.0"
description = "Massively parallel collaborative AI research system"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "SwarmResearch Team", email = "team@swarmresearch.io"},
]
keywords = [
    "ai",
    "ml",
    "multi-agent",
    "research",
    "llm",
    "swarm",
    "orchestration",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

dependencies = [
    "pydantic>=2.0",
    "httpx>=0.25",
    "structlog>=23.0",
    "tenacity>=8.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
# Provider adapters
openai = ["openai>=1.0"]
anthropic = ["anthropic>=0.20"]
kimi = ["openai>=1.0"]  # OpenAI-compatible
ollama = ["ollama>=0.1"]

# Server components
api = [
    "fastapi>=0.100",
    "uvicorn[standard]>=0.23",
    "python-multipart>=0.0.6",
]
worker = [
    "celery>=5.3",
    "redis>=5.0",
]

# Storage
postgres = ["asyncpg>=0.29", "psycopg2-binary>=2.9"]
vector = ["chromadb>=0.4", "pgvector>=0.2"]

# Observability
observability = [
    "prometheus-client>=0.19",
    "opentelemetry-api>=1.21",
    "opentelemetry-sdk>=1.21",
]

# Development
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "pytest-timeout>=2.2",
    "ruff>=0.1",
    "mypy>=1.0",
    "bandit>=1.7",
    "pre-commit>=3.5",
]

# All extras
all = [
    "swarm-research[openai,anthropic,kimi,ollama,api,worker,postgres,vector,observability]"
]

[project.urls]
Homepage = "https://swarmresearch.io"
Documentation = "https://docs.swarmresearch.io"
Repository = "https://github.com/swarmresearch/swarm-research"
Issues = "https://github.com/swarmresearch/swarm-research/issues"
Changelog = "https://github.com/swarmresearch/swarm-research/blob/main/CHANGELOG.md"

[project.scripts]
swarm-research = "swarm_research.cli:main"
swarm-api = "swarm_research.api:run"
swarm-worker = "swarm_research.worker:main"

[tool.hatch.build.targets.wheel]
packages = ["src/swarm_research"]

[tool.hatch.build.targets.sdist]
include = [
    "/src",
    "/tests",
    "/docs",
    "/README.md",
    "/LICENSE",
    "/CHANGELOG.md",
]
```

### 4.2 Docker Hub Distribution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DOCKER HUB DISTRIBUTION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Image Repository: swarmresearch/                                            │
│                                                                              │
│  Available Images:                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Image                    Description                    Size        │    │
│  │  ─────────────────────────────────────────────────────────────────  │    │
│  │  swarm-base              Base dependencies              ~180MB      │    │
│  │  swarm-api               API server                     ~250MB      │    │
│  │  swarm-worker            Worker process                 ~300MB      │    │
│  │  swarm-ollama            With Ollama pre-installed      ~2GB        │    │
│  │  swarm-dev               Development environment        ~1GB        │    │
│  │  swarm-full              All-in-one (dev only)          ~2.5GB      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Tags Strategy:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Tag              Description                                       │    │
│  │  ─────────────────────────────────────────────────────────────────  │    │
│  │  latest           Latest stable release                             │    │
│  │  v0.1.0           Specific version                                  │    │
│  │  0.1              Latest patch for minor version                    │    │
│  │  main             Latest commit on main branch                      │    │
│  │  develop          Latest commit on develop branch                   │    │
│  │  sha-abc123       Specific commit                                   │    │
│  │  rc-1             Release candidate                                 │    │
│  │  beta-1           Beta release                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Usage Examples:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  # Pull and run API server                                          │    │
│  │  docker run -p 8000:8000 swarmresearch/swarm-api:latest             │    │
│  │                                                                     │    │
│  │  # Run with environment variables                                   │    │
│  │  docker run -e KIMI_API_KEY=xxx swarmresearch/swarm-api:v0.1.0      │    │
│  │                                                                     │    │
│  │  # Run worker                                                       │    │
│  │  docker run swarmresearch/swarm-worker:latest                       │    │
│  │                                                                     │    │
│  │  # Development environment                                          │    │
│  │  docker run -it swarmresearch/swarm-dev:latest bash                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Multi-Platform Builds

```yaml
# .github/workflows/docker-build.yml
name: Docker Multi-Platform Build

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        image: [api, worker, ollama]
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: swarmresearch/swarm-${{ matrix.image }}
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=ref,event=branch
            type=sha
      
      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          file: ./docker/Dockerfile.${{ matrix.image }}
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## 5. Documentation Deployment

### 5.1 Documentation Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DOCUMENTATION DEPLOYMENT                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Documentation Sources:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  docs/                                                              │    │
│  │  ├── index.md           ← Main documentation                        │    │
│  │  ├── getting-started/   ← Quick start guides                      │    │
│  │  ├── api-reference/     ← Auto-generated API docs                 │    │
│  │  ├── architecture/      ← System architecture docs                │    │
│  │  ├── deployment/        ← Deployment guides                       │    │
│  │  ├── examples/          ← Code examples                           │    │
│  │  └── contributing/      ← Contributor guides                      │    │
│  │                                                                     │    │
│  │  README.md              ← Project README (PyPI)                     │    │
│  │  CHANGELOG.md           ← Release notes                             │    │
│  │  LICENSE                ← MIT License                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Deployment Targets:                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Target              URL                    Purpose                 │    │
│  │  ─────────────────────────────────────────────────────────────────  │    │
│  │  Main Docs          docs.swarmresearch.io   Primary documentation   │    │
│  │  API Reference      api.swarmresearch.io    Auto-generated API docs │    │
│  │  GitHub Pages       swarmresearch.github.io Mirror / fallback       │    │
│  │  ReadTheDocs        swarmresearch.rtfd.io   Alternative hosting     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 MkDocs Configuration

```yaml
# mkdocs.yml
site_name: SwarmResearch Documentation
site_description: Massively parallel collaborative AI research system
site_url: https://docs.swarmresearch.io
copyright: Copyright &copy; 2024 SwarmResearch Team

repo_name: swarmresearch/swarm-research
repo_url: https://github.com/swarmresearch/swarm-research
edit_uri: edit/main/docs/

theme:
  name: material
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - navigation.top
    - search.suggest
    - search.highlight
    - content.tabs.link
    - content.code.annotation
    - content.code.copy

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [src]
          options:
            docstring_style: google
            show_source: true
            show_root_heading: true
  - minify:
      minify_html: true
  - git-revision-date-localized:
      enable_creation_date: true

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.tasklist:
      custom_checkbox: true
  - admonition
  - pymdownx.details
  - attr_list
  - md_in_html
  - toc:
      permalink: true

extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/swarmresearch
    - icon: fontawesome/brands/twitter
      link: https://twitter.com/swarmresearch
    - icon: fontawesome/brands/discord
      link: https://discord.gg/swarmresearch
  analytics:
    provider: google
    property: G-XXXXXXXXXX

nav:
  - Home: index.md
  - Getting Started:
    - getting-started/index.md
    - getting-started/installation.md
    - getting-started/quickstart.md
    - getting-started/configuration.md
  - User Guide:
    - user-guide/index.md
    - user-guide/creating-agents.md
    - user-guide/orchestration.md
    - user-guide/providers.md
    - user-guide/workflows.md
  - API Reference:
    - api-reference/index.md
    - api-reference/orchestrator.md
    - api-reference/agents.md
    - api-reference/providers.md
  - Deployment:
    - deployment/index.md
    - deployment/docker.md
    - deployment/kubernetes.md
    - deployment/cloud.md
  - Architecture:
    - architecture/index.md
    - architecture/overview.md
    - architecture/components.md
    - architecture/scaling.md
  - Contributing:
    - contributing/index.md
    - contributing/development.md
    - contributing/testing.md
    - contributing/documentation.md
  - Changelog: changelog.md
```

### 5.3 Documentation CI/CD

```yaml
# .github/workflows/docs.yml
name: Documentation

on:
  push:
    branches: [main]
    paths:
      - 'docs/**'
      - 'mkdocs.yml'
      - 'src/**/*.py'
  pull_request:
    paths:
      - 'docs/**'
      - 'mkdocs.yml'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: |
          pip install mkdocs-material mkdocstrings[python] \
            mkdocs-minify-plugin mkdocs-git-revision-date-localized-plugin
      
      - name: Build documentation
        run: mkdocs build --strict
      
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: documentation
          path: site/

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: |
          pip install mkdocs-material mkdocstrings[python] \
            mkdocs-minify-plugin mkdocs-git-revision-date-localized-plugin
      
      - name: Deploy to GitHub Pages
        run: mkdocs gh-deploy --force
      
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          working-directory: site
```

### 5.4 API Documentation Generation

```python
# scripts/generate_api_docs.py
"""Generate API documentation from source code."""

import ast
import inspect
from pathlib import Path
from typing import List, Dict, Any

import mkdocs_gen_files

def generate_module_docs(module_path: Path, output_path: str) -> None:
    """Generate documentation for a Python module."""
    
    with mkdocs_gen_files.open(output_path, "w") as f:
        f.write(f"# {module_path.stem}\n\n")
        
        # Add module docstring
        content = module_path.read_text()
        try:
            tree = ast.parse(content)
            docstring = ast.get_docstring(tree)
            if docstring:
                f.write(f"{docstring}\n\n")
        except SyntaxError:
            pass
        
        # Add API reference
        f.write("## API Reference\n\n")
        f.write(f"::: swarm_research.{module_path.stem}\n")
        f.write("    options:\n")
        f.write("      show_source: true\n")
        f.write("      show_root_heading: false\n\n")

def main():
    """Generate all API documentation."""
    src_path = Path("src/swarm_research")
    
    for py_file in src_path.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue
        
        # Calculate output path
        relative = py_file.relative_to(src_path)
        output = f"api-reference/{relative.with_suffix('')}.md"
        
        generate_module_docs(py_file, output)
        
        # Add to navigation
        mkdocs_gen_files.set_edit_path(output, py_file)

if __name__ == "__main__":
    main()
```

---

## 6. Security & Compliance

### 6.1 Security Scanning

```yaml
# .github/workflows/security.yml
name: Security Scan

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  code-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Bandit
        uses: PyCQA/bandit@main
        with:
          args: "-r src/ -f json -o bandit-report.json"
      
      - name: Run Safety
        run: |
          pip install safety
          safety check --json --output safety-report.json
      
      - name: Upload results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: bandit-report.json

  dependency-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Snyk
        uses: snyk/actions/python@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high

  container-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build image
        run: docker build -t swarm-scan -f docker/Dockerfile.api .
      
      - name: Run Trivy
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: swarm-scan
          format: sarif
          output: trivy-results.sarif
      
      - name: Upload results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-results.sarif
```

### 6.2 Secret Management

```yaml
# Kubernetes External Secrets
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: swarm-secrets
  namespace: swarm-research
spec:
  refreshInterval: 1h
  secretStoreRef:
    kind: ClusterSecretStore
    name: aws-secrets-manager
  target:
    name: swarm-secrets
    creationPolicy: Owner
  data:
    - secretKey: DATABASE_URL
      remoteRef:
        key: swarm/production
        property: database_url
    - secretKey: SECRET_KEY
      remoteRef:
        key: swarm/production
        property: secret_key
    - secretKey: KIMI_API_KEY
      remoteRef:
        key: swarm/production
        property: kimi_api_key
```

---

## 7. Monitoring & Observability

### 7.1 Metrics Collection

```yaml
# config/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'swarm-api'
    static_configs:
      - targets: ['swarm-api:9090']
    metrics_path: /metrics
    
  - job_name: 'swarm-worker'
    static_configs:
      - targets: ['swarm-worker:9090']
    metrics_path: /metrics
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

### 7.2 Health Checks

```python
# src/swarm_research/health.py
from fastapi import FastAPI, status
from pydantic import BaseModel
from typing import Dict, Any
import asyncio

app = FastAPI()

class HealthStatus(BaseModel):
    status: str
    version: str
    checks: Dict[str, Any]
    timestamp: str

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Basic health check endpoint."""
    return HealthStatus(
        status="healthy",
        version=__version__,
        checks={},
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes."""
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "providers": await check_providers(),
    }
    
    all_healthy = all(c["status"] == "healthy" for c in checks.values())
    
    status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        content={
            "ready": all_healthy,
            "checks": checks,
        },
        status_code=status_code
    )

@app.get("/live")
async def liveness_check():
    """Liveness probe for Kubernetes."""
    return {"alive": True}
```

---

## 8. Summary

### 8.1 Deployment Checklist

```markdown
# Production Deployment Checklist

## Pre-Deployment
- [ ] All CI checks passing
- [ ] Security scans clear
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Database migrations tested
- [ ] Rollback plan documented

## Deployment
- [ ] Deploy to staging first
- [ ] Run smoke tests on staging
- [ ] Monitor error rates
- [ ] Deploy to production
- [ ] Verify health checks
- [ ] Monitor metrics

## Post-Deployment
- [ ] Verify all services healthy
- [ ] Check error rates < 0.1%
- [ ] Monitor response times
- [ ] Verify auto-scaling working
- [ ] Update status page
- [ ] Announce deployment
```

### 8.2 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Container Base** | python:3.11-slim | Small footprint, security updates |
| **Package Manager** | uv | 10-100x faster than pip |
| **CI/CD** | GitHub Actions | Native integration, free for OSS |
| **Orchestration** | Docker Compose → K8s | Progressive complexity |
| **Docs** | MkDocs Material | Beautiful, searchable, maintainable |
| **Versioning** | Semantic | Clear compatibility signals |
| **Registry** | Docker Hub + GHCR | Redundancy, availability |

### 8.3 Resource Requirements

| Environment | CPU | Memory | Storage | Cost (est.) |
|-------------|-----|--------|---------|-------------|
| Development | 4 cores | 8 GB | 50 GB | $0 (local) |
| Staging | 4 cores | 16 GB | 100 GB | $100/mo |
| Production (Small) | 8 cores | 32 GB | 500 GB | $300/mo |
| Production (Medium) | 32 cores | 128 GB | 2 TB | $1000/mo |
| Production (Large) | 100+ cores | 512 GB | 10 TB | $5000/mo |

---

## Appendix: Quick Reference

### Docker Commands

```bash
# Build all images
./scripts/build-images.sh v0.1.0

# Run development environment
docker-compose -f docker-compose.dev.yml up -d

# Run production environment
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose logs -f api

# Scale workers
docker-compose up -d --scale worker=10
```

### Kubernetes Commands

```bash
# Deploy to cluster
kubectl apply -k k8s/overlays/production

# View pods
kubectl get pods -n swarm-research

# View logs
kubectl logs -f deployment/swarm-api -n swarm-research

# Scale deployment
kubectl scale deployment swarm-worker --replicas=20 -n swarm-research

# Helm install
helm install swarm-research ./helm/swarm-research -f values-production.yaml
```

### PyPI Commands

```bash
# Build package
python -m build

# Test upload
python -m twine upload --repository testpypi dist/*

# Production upload
python -m twine upload dist/*
```
