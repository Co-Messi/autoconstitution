# autoconstitution Benchmark Reproducibility Protocol

## Version 1.0.0 | Last Updated: 2024

---

## Executive Summary

This document establishes the complete reproducibility protocol for the autoconstitution benchmark. Following this protocol ensures that any researcher can reproduce benchmark results with identical outputs, enabling fair comparison across different implementations and hardware configurations.

**Key Principles:**
- **Determinism**: Same inputs produce identical outputs
- **Transparency**: All dependencies and configurations are documented
- **Portability**: Results reproducible across compatible hardware
- **Verification**: Built-in checks validate reproducibility

---

## Table of Contents

1. [Fixed Random Seeds](#1-fixed-random-seeds)
2. [Documented Environment](#2-documented-environment)
3. [Version Pinning](#3-version-pinning)
4. [Containerization Approach](#4-containerization-approach)
5. [One-Command Reproduction](#5-one-command-reproduction)
6. [Verification & Validation](#6-verification--validation)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Fixed Random Seeds

### 1.1 Master Seed Configuration

The benchmark uses a hierarchical seeding system to ensure complete reproducibility:

```yaml
# seeds.yaml - Master seed configuration
benchmark_version: "1.0.0"
master_seed: 42

# Component-specific seeds (derived from master_seed)
seeds:
  dataset_shuffle: 42           # Dataset ordering and splits
  agent_initialization: 123     # Agent personality/role assignment
  synthetic_generation: 456     # Synthetic paper generation
  evaluation_sampling: 789      # Evaluation subset selection
  communication_order: 101112   # Agent communication sequencing
  task_assignment: 131415       # Dynamic task allocation
  noise_injection: 161718       # Stochastic exploration noise
```

### 1.2 Seed Application Protocol

All random number generators must be seeded before benchmark execution:

```python
# reproducibility/seeds.py
"""Centralized seed management for autoconstitution benchmark."""

import random
import numpy as np
import torch
from typing import Dict, Optional

# Master seed configuration
SEED_CONFIG = {
    "master_seed": 42,
    "dataset_shuffle": 42,
    "agent_initialization": 123,
    "synthetic_generation": 456,
    "evaluation_sampling": 789,
    "communication_order": 101112,
    "task_assignment": 131415,
    "noise_injection": 161718,
}

def set_all_seeds(seed: Optional[int] = None) -> Dict[str, int]:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Master seed. If None, uses SEED_CONFIG["master_seed"]
        
    Returns:
        Dictionary of seeds used
    """
    master = seed if seed is not None else SEED_CONFIG["master_seed"]
    
    # Python standard library
    random.seed(master)
    
    # NumPy
    np.random.seed(master)
    
    # PyTorch (if available)
    try:
        torch.manual_seed(master)
        torch.cuda.manual_seed_all(master)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # Environment variable for subprocesses
    import os
    os.environ["SWARMRESEARCH_SEED"] = str(master)
    
    return {
        "master_seed": master,
        **{k: v for k, v in SEED_CONFIG.items() if k != "master_seed"}
    }

def get_derived_seed(base_seed: int, component: str, iteration: int = 0) -> int:
    """
    Generate a deterministic derived seed for a specific component.
    
    Args:
        base_seed: Base seed value
        component: Component identifier string
        iteration: Optional iteration number
        
    Returns:
        Derived seed value
    """
    import hashlib
    seed_string = f"{base_seed}:{component}:{iteration}"
    hash_val = int(hashlib.md5(seed_string.encode()).hexdigest(), 16)
    return hash_val % (2**32)

# Convenience function for component-specific seeding
def seed_for_component(component: str, iteration: int = 0) -> int:
    """Get seed for a specific component."""
    base = SEED_CONFIG.get(component, SEED_CONFIG["master_seed"])
    return get_derived_seed(base, component, iteration)
```

### 1.3 Per-Task Seed Management

Each task category uses dedicated seeds to isolate variability:

```python
# Task-specific seed application
def apply_task_seeds(task_name: str):
    """Apply seeds specific to a task category."""
    task_seed_map = {
        "dls": "dataset_shuffle",      # Distributed Literature Synthesis
        "chg": "agent_initialization", # Collaborative Hypothesis Generation
        "cbd": "communication_order",  # Consensus-Based Decision Making
        "dta": "task_assignment",      # Dynamic Task Allocation
    }
    
    seed_key = task_seed_map.get(task_name, "master_seed")
    seed_value = SEED_CONFIG.get(seed_key, SEED_CONFIG["master_seed"])
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    return seed_value
```

### 1.4 Seed Verification

Verify seed consistency across runs:

```python
def verify_seed_reproducibility() -> bool:
    """Verify that seeds produce consistent results."""
    set_all_seeds(42)
    
    # Generate test sequences
    python_random = [random.random() for _ in range(10)]
    numpy_random = np.random.rand(10).tolist()
    
    # Reset and regenerate
    set_all_seeds(42)
    
    python_random_2 = [random.random() for _ in range(10)]
    numpy_random_2 = np.random.rand(10).tolist()
    
    # Verify consistency
    assert python_random == python_random_2, "Python random not reproducible"
    assert numpy_random == numpy_random_2, "NumPy random not reproducible"
    
    return True
```

---

## 2. Documented Environment

### 2.1 Hardware Requirements

#### Minimum Specification (M4 Baseline)

```yaml
hardware:
  platform: "Apple Silicon"
  chip: "M4"
  minimum_ram: "16GB"
  recommended_ram: "24GB"
  storage: "50GB free space"
  os_version: "macOS 14.0+"
  
  cpu:
    performance_cores: 4
    efficiency_cores: 6
    threads: 8
    
  neural_engine:
    available: true
    cores: 16
```

#### Alternative Configurations

```yaml
hardware_alternatives:
  linux_x86:
    cpu: "Intel/AMD 8+ cores"
    ram: "16GB+"
    gpu: "Optional CUDA-compatible"
    os: "Ubuntu 22.04+"
    
  linux_arm:
    cpu: "ARM64 8+ cores"
    ram: "16GB+"
    os: "Ubuntu 22.04+ ARM64"
```

### 2.2 System Configuration

#### macOS System Settings

```bash
#!/bin/bash
# scripts/configure_macos.sh

echo "Configuring macOS for reproducible benchmark execution..."

# Disable sleep during benchmark
sudo pmset -c sleep 0
sudo pmset -b sleep 0

# Disable low power mode
sudo pmset -c lowpowermode 0

# Set performance mode
sudo pmset -c gpuswitch 2  # Use discrete GPU if available

# Verify M4 detection
CHIP=$(sysctl -n machdep.cpu.brand_string)
if [[ "$CHIP" != *"Apple M4"* ]]; then
    echo "WARNING: Expected Apple M4, found: $CHIP"
fi

# Check available RAM
RAM_GB=$(system_profiler SPHardwareDataType | grep "Memory:" | awk '{print $2}')
if [ "$RAM_GB" -lt 16 ]; then
    echo "WARNING: Minimum 16GB RAM recommended, found: ${RAM_GB}GB"
fi

echo "System configuration complete."
```

#### Linux System Settings

```bash
#!/bin/bash
# scripts/configure_linux.sh

echo "Configuring Linux for reproducible benchmark execution..."

# Set CPU governor to performance
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee $cpu > /dev/null
done

# Disable CPU frequency scaling
sudo systemctl stop ondemand 2>/dev/null || true

# Set process priority
sudo renice -n -10 -p $$

echo "System configuration complete."
```

### 2.3 Environment Variables

```bash
# .env.template - Environment configuration template

# Benchmark Configuration
export SWARMRESEARCH_SEED=42
export SWARMRESEARCH_BENCHMARK_VERSION=1.0.0
export SWARMRESEARCH_OUTPUT_DIR=./results
export SWARMRESEARCH_LOG_LEVEL=INFO

# Hardware Configuration
export SWARMRESEARCH_THREADS=8
export SWARMRESEARCH_BATCH_SIZE=512
export SWARMRESEARCH_GPU_LAYERS=0

# Model Configuration
export SWARMRESEARCH_MODEL_PATH=./models/swarm-agent-small-3B-int8.gguf
export SWARMRESEARCH_MODEL_TEMPERATURE=0.7
export SWARMRESEARCH_MODEL_TOP_P=0.9
export SWARMRESEARCH_MODEL_MAX_TOKENS=1024

# Dataset Configuration
export SWARMRESEARCH_DATASET_PATH=./datasets/autoconstitution_bench_v1
export SWARMRESEARCH_VERIFY_CHECKSUMS=true

# Reproducibility Flags
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # For CUDA deterministic mode
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### 2.4 Environment Documentation Script

```python
# reproducibility/environment.py
"""Environment documentation and verification."""

import platform
import sys
import subprocess
import json
from datetime import datetime
from typing import Dict, Any

def capture_environment() -> Dict[str, Any]:
    """Capture complete environment state for reproducibility."""
    
    env = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "path": sys.path,
        },
        "hardware": capture_hardware_info(),
        "environment_variables": capture_relevant_env_vars(),
    }
    
    return env

def capture_hardware_info() -> Dict[str, Any]:
    """Capture hardware information."""
    hw_info = {}
    
    # macOS specific
    if platform.system() == "Darwin":
        try:
            hw_info["chip"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ).decode().strip()
            
            hw_info["memory"] = subprocess.check_output(
                ["system_profiler", "SPHardwareDataType"]
            ).decode()
        except:
            pass
    
    # Linux specific
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                hw_info["cpuinfo"] = f.read()[:1000]
            
            with open("/proc/meminfo") as f:
                hw_info["meminfo"] = f.read()[:500]
        except:
            pass
    
    return hw_info

def capture_relevant_env_vars() -> Dict[str, str]:
    """Capture autoconstitution-related environment variables."""
    import os
    
    relevant_prefixes = ("SWARMRESEARCH", "PYTHON", "OMP", "MKL", "CUDA")
    
    return {
        k: v for k, v in os.environ.items()
        if any(k.startswith(p) for p in relevant_prefixes)
    }

def save_environment_report(output_path: str = "./environment_report.json"):
    """Save environment report to file."""
    env = capture_environment()
    with open(output_path, 'w') as f:
        json.dump(env, f, indent=2)
    return output_path
```

---

## 3. Version Pinning

### 3.1 Python Dependencies

#### requirements.txt (Exact Versions)

```
# requirements.txt
# autoconstitution Benchmark v1.0.0
# Generated: 2024-XX-XX
# Python: 3.11.x

# Core Framework
numpy==1.26.4
scipy==1.12.0

# Data Processing
pandas==2.2.0
pyarrow==15.0.0

# Machine Learning
torch==2.2.0
--extra-index-url https://download.pytorch.org/whl/cpu

# LLM Inference
llama-cpp-python==0.2.90

# API & Communication
requests==2.31.0
httpx==0.26.0

# Data Validation
pydantic==2.7.0
pydantic-core==2.18.0

# Configuration
pyyaml==6.0.1
python-dotenv==1.0.0

# Logging & Monitoring
structlog==24.1.0
rich==13.7.0

# Testing
pytest==8.0.0
pytest-cov==4.1.0

# Utilities
tqdm==4.66.1
click==8.1.7
joblib==1.3.2

# Benchmark Package
autoconstitution-benchmark==1.0.0
```

#### requirements-dev.txt

```
# Development dependencies
-r requirements.txt

# Code quality
black==24.1.1
flake8==7.0.0
mypy==1.8.0
isort==5.13.2

# Documentation
sphinx==7.2.6
sphinx-rtd-theme==2.0.0

# Jupyter
jupyter==1.0.0
ipython==8.21.0

# Profiling
memory-profiler==0.61.0
line-profiler==4.1.1
```

### 3.2 Conda Environment Specification

```yaml
# environment.yml
name: autoconstitution-benchmark
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11.7
  - pip=24.0
  - numpy=1.26.4
  - scipy=1.12.0
  - pandas=2.2.0
  - pyyaml=6.0.1
  - requests=2.31.0
  - pip:
    - llama-cpp-python==0.2.90
    - pydantic==2.7.0
    - autoconstitution-benchmark==1.0.0
```

### 3.3 Lock File Generation

```bash
#!/bin/bash
# scripts/generate_lockfile.sh

set -e

echo "Generating dependency lock file..."

# Create temporary environment
python3.11 -m venv /tmp/venv_lock
source /tmp/venv_lock/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Generate lock file
pip freeze > requirements.lock

echo "Lock file generated: requirements.lock"

# Cleanup
deactivate
rm -rf /tmp/venv_lock
```

### 3.4 Dependency Verification

```python
# reproducibility/verify_deps.py
"""Dependency version verification."""

import pkg_resources
import sys
from typing import List, Tuple

REQUIRED_VERSIONS = {
    "numpy": "1.26.4",
    "scipy": "1.12.0",
    "pandas": "2.2.0",
    "pydantic": "2.7.0",
    "llama-cpp-python": "0.2.90",
    "pyyaml": "6.0.1",
}

def verify_dependencies() -> Tuple[bool, List[str]]:
    """
    Verify all dependencies match required versions.
    
    Returns:
        (success, list of mismatches)
    """
    mismatches = []
    
    installed = {dist.key: dist.version for dist in pkg_resources.working_set}
    
    for package, required_version in REQUIRED_VERSIONS.items():
        installed_version = installed.get(package)
        
        if installed_version is None:
            mismatches.append(f"{package}: NOT INSTALLED (required {required_version})")
        elif installed_version != required_version:
            mismatches.append(
                f"{package}: {installed_version} (required {required_version})"
            )
    
    return len(mismatches) == 0, mismatches

def check_python_version() -> bool:
    """Verify Python version compatibility."""
    version = sys.version_info
    return version.major == 3 and version.minor == 11

if __name__ == "__main__":
    if not check_python_version():
        print(f"ERROR: Python 3.11 required, found {sys.version}")
        sys.exit(1)
    
    success, mismatches = verify_dependencies()
    
    if not success:
        print("ERROR: Dependency version mismatches found:")
        for m in mismatches:
            print(f"  - {m}")
        sys.exit(1)
    
    print("All dependencies verified successfully.")
```

### 3.5 Model & Dataset Version Pinning

```yaml
# assets.yaml - Asset version specification
assets:
  models:
    swarm-agent-small-3B-int8:
      version: "1.0.0"
      url: "https://models.autoconstitution.org/v1/swarm-agent-small-3B-int8.gguf"
      checksum:
        sha256: "f8e2a9c4d1b7..."  # 64 characters
      size_bytes: 3214572800
      
    swarm-agent-medium-7B-int8:
      version: "1.0.0"
      url: "https://models.autoconstitution.org/v1/swarm-agent-medium-7B-int8.gguf"
      checksum:
        sha256: "a1b2c3d4e5f6..."
      size_bytes: 7500000000
      
  datasets:
    autoconstitution_bench_v1:
      version: "1.0.0"
      url: "https://benchmarks.autoconstitution.org/datasets/v1/autoconstitution_bench_v1.tar.gz"
      checksum:
        sha256: "a3f7c9e2d8b1..."  # 64 characters
      size_bytes: 262144000  # 250 MB
      
  checksum_verification:
    enabled: true
    fail_on_mismatch: true
```

---

## 4. Containerization Approach

### 4.1 Docker Configuration

#### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11.7-slim-bookworm

LABEL maintainer="autoconstitution Team"
LABEL version="1.0.0"
LABEL description="autoconstitution Benchmark Reproducibility Container"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42
ENV SWARMRESEARCH_SEED=42
ENV SWARMRESEARCH_BENCHMARK_VERSION=1.0.0

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
COPY requirements.txt requirements.lock ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip==24.0 && \
    pip install --no-cache-dir -r requirements.lock

# Copy benchmark code
COPY . .

# Install benchmark package
RUN pip install --no-cache-dir -e .

# Create directories for assets
RUN mkdir -p /benchmark/models /benchmark/datasets /benchmark/results

# Download and verify assets
RUN python -m autoconstitution.download --verify

# Set entrypoint
ENTRYPOINT ["python", "-m", "autoconstitution.benchmark"]
CMD ["--help"]
```

#### .dockerignore

```
# .dockerignore
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Git
.git/
.gitignore

# IDE
.vscode/
.idea/
*.swp
*.swo

# Results (mount as volume)
results/
logs/

# Documentation
*.md
docs/

# Tests (optional - include if running tests in container)
tests/
.pytest_cache/

# OS
.DS_Store
Thumbs.db
```

### 4.2 Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  benchmark:
    build:
      context: .
      dockerfile: Dockerfile
    image: autoconstitution/benchmark:1.0.0
    container_name: autoconstitution-benchmark
    
    environment:
      - SWARMRESEARCH_SEED=42
      - SWARMRESEARCH_LOG_LEVEL=INFO
      - PYTHONHASHSEED=42
      
    volumes:
      - ./results:/benchmark/results
      - ./configs:/benchmark/configs:ro
      
    # Resource limits for reproducibility
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
        reservations:
          cpus: '4'
          memory: 8G
    
    # Ensure consistent CPU scheduling
    cpu_count: 8
    cpu_shares: 1024
    
    # Prevent swapping for consistent performance
    mem_swappiness: 0
    
    command: >
      --config /benchmark/configs/m4_baseline.yaml
      --output /benchmark/results/
      --seed 42

  # Validation service
  verify:
    build:
      context: .
      dockerfile: Dockerfile
    image: autoconstitution/benchmark:1.0.0
    container_name: autoconstitution-verify
    command: >
      python -m autoconstitution.verify --full
```

### 4.3 Multi-Platform Build

```bash
#!/bin/bash
# scripts/build_container.sh

set -e

VERSION="1.0.0"
IMAGE_NAME="autoconstitution/benchmark"

echo "Building autoconstitution benchmark container..."

# Build for multiple platforms
docker buildx create --use --name swarmbuilder 2>/dev/null || true

docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag ${IMAGE_NAME}:${VERSION} \
    --tag ${IMAGE_NAME}:latest \
    --push \
    .

echo "Container build complete: ${IMAGE_NAME}:${VERSION}"
```

### 4.4 Container Verification

```python
# reproducibility/verify_container.py
"""Container environment verification."""

import os
import subprocess
import json

def verify_container_environment():
    """Verify container is properly configured."""
    
    checks = {
        "in_container": os.path.exists("/.dockerenv"),
        "python_version": get_python_version(),
        "seed_set": os.environ.get("SWARMRESEARCH_SEED") == "42",
        "assets_present": verify_assets(),
        "resource_limits": check_resource_limits(),
    }
    
    return checks

def get_python_version():
    """Get Python version in container."""
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

def verify_assets():
    """Verify model and dataset are present."""
    model_path = "/benchmark/models/swarm-agent-small-3B-int8.gguf"
    dataset_path = "/benchmark/datasets/autoconstitution_bench_v1"
    
    return {
        "model_present": os.path.exists(model_path),
        "dataset_present": os.path.exists(dataset_path),
    }

def check_resource_limits():
    """Check container resource limits."""
    # Read cgroup limits
    limits = {}
    
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            limits["cpu_quota"] = f.read().strip()
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
            limits["memory_limit"] = f.read().strip()
    except:
        limits["note"] = "Could not read cgroup limits"
    
    return limits

if __name__ == "__main__":
    result = verify_container_environment()
    print(json.dumps(result, indent=2))
```

---

## 5. One-Command Reproduction

### 5.1 Master Reproduction Script

```bash
#!/bin/bash
# reproduce.sh - One-command benchmark reproduction

set -e

# Configuration
BENCHMARK_VERSION="1.0.0"
DEFAULT_SEED=42
RESULTS_DIR="./results"
CONFIG_FILE="./configs/m4_baseline.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
SEED=${1:-$DEFAULT_SEED}
OUTPUT_DIR=${2:-$RESULTS_DIR}

log_info "autoconstitution Benchmark Reproduction"
log_info "Version: $BENCHMARK_VERSION"
log_info "Seed: $SEED"
log_info "Output: $OUTPUT_DIR"
echo ""

# Step 1: Environment Check
log_info "Step 1/7: Checking environment..."
python3 --version || { log_error "Python 3 not found"; exit 1; }

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" != "3.11" ]]; then
    log_warn "Python 3.11 recommended, found $PYTHON_VERSION"
fi

# Step 2: Setup Virtual Environment
log_info "Step 2/7: Setting up virtual environment..."
if [ ! -d "./venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Step 3: Install Dependencies
log_info "Step 3/7: Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.lock

# Verify dependencies
python -m reproducibility.verify_deps || { log_error "Dependency verification failed"; exit 1; }

# Step 4: Download Assets
log_info "Step 4/7: Downloading assets..."
python -m autoconstitution.download \
    --dataset \
    --model \
    --verify || { log_error "Asset download failed"; exit 1; }

# Step 5: Verify Environment
log_info "Step 5/7: Verifying environment..."
python -m autoconstitution.verify --full || { log_error "Environment verification failed"; exit 1; }

# Step 6: Run Benchmark
log_info "Step 6/7: Running benchmark (this may take 20-30 minutes)..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$OUTPUT_DIR/run_${TIMESTAMP}_seed${SEED}.json"

mkdir -p "$OUTPUT_DIR"

python -m autoconstitution.benchmark \
    --config "$CONFIG_FILE" \
    --output "$RESULTS_FILE" \
    --seed "$SEED" \
    --verbose || { log_error "Benchmark execution failed"; exit 1; }

# Step 7: Generate Report
log_info "Step 7/7: Generating report..."
python -m autoconstitution.report \
    --results "$RESULTS_FILE" \
    --output "$OUTPUT_DIR/report_${TIMESTAMP}.html"

echo ""
log_info "Reproduction complete!"
log_info "Results: $RESULTS_FILE"
log_info "Report: $OUTPUT_DIR/report_${TIMESTAMP}.html"

# Display summary
if [ -f "$RESULTS_FILE" ]; then
    echo ""
    python -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)
    print('=== Benchmark Results ===')
    print(f\"Overall Score: {data['results']['overall_score']:.1f}\")
    print(f\"DLS Score: {data['results']['dls']['score']:.1f}\")
    print(f\"CHG Score: {data['results']['chg']['score']:.1f}\")
    print(f\"CBD Score: {data['results']['cbd']['score']:.1f}\")
    print(f\"DTA Score: {data['results']['dta']['score']:.1f}\")
"
fi

echo ""
log_info "To verify reproducibility, run again with the same seed."
log_info "Expected: Identical results (within floating-point tolerance)"
```

### 5.2 Makefile Interface

```makefile
# Makefile - Convenient commands for benchmark reproduction

.PHONY: help setup verify run clean docker-build docker-run report

# Default target
.DEFAULT_GOAL := help

# Variables
SEED ?= 42
OUTPUT_DIR ?= ./results
CONFIG ?= ./configs/m4_baseline.yaml
IMAGE_NAME = autoconstitution/benchmark
VERSION = 1.0.0

help: ## Show this help message
	@echo "autoconstitution Benchmark - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Set up the environment
	@echo "Setting up environment..."
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.lock

download: ## Download models and datasets
	@echo "Downloading assets..."
	./venv/bin/python -m autoconstitution.download --dataset --model --verify

verify: ## Verify environment and dependencies
	@echo "Verifying environment..."
	./venv/bin/python -m reproducibility.verify_deps
	./venv/bin/python -m autoconstitution.verify --full

run: ## Run the benchmark (default seed: 42)
	@echo "Running benchmark with seed $(SEED)..."
	mkdir -p $(OUTPUT_DIR)
	./venv/bin/python -m autoconstitution.benchmark \
		--config $(CONFIG) \
		--output $(OUTPUT_DIR)/run_$$(date +%Y%m%d_%H%M%S)_seed$(SEED).json \
		--seed $(SEED)

reproduce: ## Full reproduction pipeline
	@echo "Starting full reproduction..."
	bash reproduce.sh $(SEED) $(OUTPUT_DIR)

report: ## Generate HTML report from results
	@echo "Generating report..."
	./venv/bin/python -m autoconstitution.report \
		--results $(OUTPUT_DIR)/run_*.json \
		--output $(OUTPUT_DIR)/report.html

clean: ## Clean generated files
	@echo "Cleaning..."
	rm -rf venv/
	rm -rf $(OUTPUT_DIR)/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

docker-build: ## Build Docker container
	@echo "Building Docker container..."
	docker build -t $(IMAGE_NAME):$(VERSION) .

docker-run: ## Run benchmark in Docker container
	@echo "Running benchmark in Docker..."
	docker run --rm \
		-v $(PWD)/results:/benchmark/results \
		-v $(PWD)/configs:/benchmark/configs:ro \
		-e SWARMRESEARCH_SEED=$(SEED) \
		$(IMAGE_NAME):$(VERSION) \
		--config /benchmark/configs/m4_baseline.yaml \
		--output /benchmark/results/ \
		--seed $(SEED)

docker-verify: ## Verify Docker environment
	@echo "Verifying Docker environment..."
	docker run --rm $(IMAGE_NAME):$(VERSION) python -m autoconstitution.verify --full

# CI/CD targets
ci-test: setup verify ## Run CI tests
	./venv/bin/python -m pytest tests/ -v --cov=autoconstitution

ci-benchmark: setup download verify run ## Run CI benchmark
	@echo "CI benchmark complete"
```

### 5.3 Python Reproduction API

```python
# reproducibility/api.py
"""Programmatic reproduction API."""

from pathlib import Path
from typing import Optional, Dict, Any
import json
import subprocess

class ReproductionRunner:
    """One-command benchmark reproduction."""
    
    def __init__(
        self,
        seed: int = 42,
        output_dir: str = "./results",
        config_path: str = "./configs/m4_baseline.yaml"
    ):
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.config_path = Path(config_path)
        self.results: Optional[Dict[str, Any]] = None
        
    def setup(self) -> "ReproductionRunner":
        """Set up the environment."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify Python version
        import sys
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError(f"Python 3.11 required, found {sys.version}")
        
        return self
    
    def download_assets(self) -> "ReproductionRunner":
        """Download required assets."""
        subprocess.run(
            ["python", "-m", "autoconstitution.download", "--dataset", "--model", "--verify"],
            check=True
        )
        return self
    
    def verify(self) -> "ReproductionRunner":
        """Verify environment."""
        subprocess.run(
            ["python", "-m", "autoconstitution.verify", "--full"],
            check=True
        )
        return self
    
    def run(self) -> "ReproductionRunner":
        """Execute the benchmark."""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"run_{timestamp}_seed{self.seed}.json"
        
        subprocess.run([
            "python", "-m", "autoconstitution.benchmark",
            "--config", str(self.config_path),
            "--output", str(results_file),
            "--seed", str(self.seed)
        ], check=True)
        
        # Load results
        with open(results_file) as f:
            self.results = json.load(f)
        
        return self
    
    def generate_report(self) -> "ReproductionRunner":
        """Generate HTML report."""
        if self.results is None:
            raise RuntimeError("No results available. Run benchmark first.")
        
        report_file = self.output_dir / f"report_{self.results['run_timestamp'][:19].replace(':', '')}.html"
        
        subprocess.run([
            "python", "-m", "autoconstitution.report",
            "--results", str(self.output_dir / "run_*.json"),
            "--output", str(report_file)
        ], check=True)
        
        return self
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary."""
        if self.results is None:
            return {"status": "not_run"}
        
        return {
            "status": "complete",
            "overall_score": self.results["results"]["overall_score"],
            "task_scores": {
                "dls": self.results["results"]["dls"]["score"],
                "chg": self.results["results"]["chg"]["score"],
                "cbd": self.results["results"]["cbd"]["score"],
                "dta": self.results["results"]["dta"]["score"],
            },
            "seed": self.results["configuration"]["seed"],
            "timestamp": self.results["run_timestamp"],
        }

# Convenience function
def reproduce(
    seed: int = 42,
    output_dir: str = "./results",
    config_path: str = "./configs/m4_baseline.yaml"
) -> Dict[str, Any]:
    """
    One-command benchmark reproduction.
    
    Args:
        seed: Random seed for reproducibility
        output_dir: Directory for results
        config_path: Path to benchmark configuration
        
    Returns:
        Benchmark summary
    """
    runner = ReproductionRunner(seed, output_dir, config_path)
    
    (runner
        .setup()
        .download_assets()
        .verify()
        .run()
        .generate_report())
    
    return runner.get_summary()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="autoconstitution Benchmark Reproduction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="./results", help="Output directory")
    parser.add_argument("--config", default="./configs/m4_baseline.yaml", help="Config file")
    
    args = parser.parse_args()
    
    summary = reproduce(args.seed, args.output, args.config)
    print(json.dumps(summary, indent=2))
```

---

## 6. Verification & Validation

### 6.1 Reproducibility Check Script

```python
# reproducibility/check_reproducibility.py
"""Verify reproducibility across multiple runs."""

import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

def run_benchmark(seed: int, output_dir: Path) -> Path:
    """Run benchmark with given seed and return results path."""
    result = subprocess.run([
        "python", "-m", "autoconstitution.benchmark",
        "--config", "./configs/m4_baseline.yaml",
        "--output", str(output_dir),
        "--seed", str(seed)
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed: {result.stderr}")
    
    # Find the generated results file
    results_files = list(output_dir.glob(f"*_seed{seed}.json"))
    if not results_files:
        raise RuntimeError("No results file found")
    
    return max(results_files, key=lambda p: p.stat().st_mtime)

def compare_results(path1: Path, path2: Path, tolerance: float = 0.01) -> Tuple[bool, Dict]:
    """
    Compare two benchmark results for reproducibility.
    
    Args:
        path1: First results file
        path2: Second results file
        tolerance: Floating point comparison tolerance
        
    Returns:
        (is_reproducible, comparison_details)
    """
    with open(path1) as f:
        results1 = json.load(f)
    with open(path2) as f:
        results2 = json.load(f)
    
    differences = {}
    
    # Compare overall score
    score1 = results1["results"]["overall_score"]
    score2 = results2["results"]["overall_score"]
    
    if abs(score1 - score2) > tolerance:
        differences["overall_score"] = {"run1": score1, "run2": score2}
    
    # Compare task scores
    for task in ["dls", "chg", "cbd", "dta"]:
        task_score1 = results1["results"][task]["score"]
        task_score2 = results2["results"][task]["score"]
        
        if abs(task_score1 - task_score2) > tolerance:
            differences[f"{task}_score"] = {"run1": task_score1, "run2": task_score2}
    
    return len(differences) == 0, differences

def verify_reproducibility(
    num_runs: int = 3,
    seed: int = 42,
    output_dir: Path = Path("./reproducibility_test")
) -> Dict:
    """
    Verify reproducibility by running benchmark multiple times.
    
    Args:
        num_runs: Number of times to run benchmark
        seed: Seed to use for all runs
        output_dir: Directory for test results
        
    Returns:
        Reproducibility report
    """
    print(f"Running reproducibility test with {num_runs} runs (seed={seed})...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_paths = []
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}...")
        path = run_benchmark(seed, output_dir)
        results_paths.append(path)
    
    # Compare all pairs
    print("Comparing results...")
    all_reproducible = True
    pair_comparisons = []
    
    for i in range(len(results_paths)):
        for j in range(i+1, len(results_paths)):
            is_match, diffs = compare_results(results_paths[i], results_paths[j])
            pair_comparisons.append({
                "pair": (i+1, j+1),
                "match": is_match,
                "differences": diffs
            })
            if not is_match:
                all_reproducible = False
    
    report = {
        "num_runs": num_runs,
        "seed": seed,
        "all_reproducible": all_reproducible,
        "pair_comparisons": pair_comparisons,
        "results_files": [str(p) for p in results_paths]
    }
    
    # Save report
    report_path = output_dir / "reproducibility_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReproducibility test {'PASSED' if all_reproducible else 'FAILED'}")
    print(f"Report saved to: {report_path}")
    
    return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./reproducibility_test")
    
    args = parser.parse_args()
    
    report = verify_reproducibility(args.runs, args.seed, Path(args.output))
    
    if not report["all_reproducible"]:
        exit(1)
```

### 6.2 Continuous Integration Configuration

```yaml
# .github/workflows/reproducibility.yml
name: Reproducibility Check

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    # Run weekly on Sundays at 00:00 UTC
    - cron: '0 0 * * 0'

jobs:
  reproducibility:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        seed: [42, 123, 456]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.lock
      
      - name: Download assets
        run: |
          python -m autoconstitution.download --dataset --model --verify
      
      - name: Run benchmark
        run: |
          python -m autoconstitution.benchmark \
            --config configs/m4_baseline.yaml \
            --output results/ \
            --seed ${{ matrix.seed }}
      
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: results-seed-${{ matrix.seed }}
          path: results/*.json
  
  compare:
    needs: reproducibility
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Download all artifacts
        uses: actions/download-artifact@v4
        with:
          path: results/
          pattern: results-seed-*
      
      - name: Compare results
        run: |
          python reproducibility/compare_ci_results.py results/
```

---

## 7. Troubleshooting

### 7.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Results differ between runs | Non-deterministic operations | Check all seeds are set, verify no parallel processing |
| Checksum mismatch | Corrupted download | Re-download asset with `--force` flag |
| Out of memory | Batch size too large | Reduce `batch_size` in config to 256 |
| Slow inference | Background processes | Close other applications, check CPU usage |
| Import errors | Missing dependencies | Run `pip install -r requirements.lock` |
| Docker build fails | Platform mismatch | Use `--platform linux/amd64` flag |

### 7.2 Debug Mode

```bash
# Enable verbose logging
export SWARMRESEARCH_LOG_LEVEL=DEBUG

# Run with debug output
python -m autoconstitution.benchmark --config configs/m4_baseline.yaml --verbose

# Verify specific component
python -m autoconstitution.verify --component seeds
python -m autoconstitution.verify --component dependencies
python -m autoconstitution.verify --component assets
```

### 7.3 Support Resources

- **Documentation**: https://docs.autoconstitution.org
- **Issue Tracker**: https://github.com/autoconstitution/benchmark/issues
- **Discussions**: https://github.com/autoconstitution/benchmark/discussions
- **Email**: benchmark@autoconstitution.org

---

## Appendix A: Quick Reference

### One-Command Quick Start

```bash
# Native execution
./reproduce.sh

# Docker execution
docker run -v $(PWD)/results:/benchmark/results autoconstitution/benchmark:1.0.0

# Makefile
make reproduce

# Python API
python -c "from reproducibility.api import reproduce; reproduce()"
```

### Verification Checklist

- [ ] Python 3.11 installed
- [ ] All dependencies match requirements.lock
- [ ] Model checksum verified
- [ ] Dataset checksum verified
- [ ] Seed set to 42 (or specified value)
- [ ] No background processes consuming resources
- [ ] Sufficient disk space (50GB+)
- [ ] Results directory writable

### Expected Runtime

| Step | Duration |
|------|----------|
| Environment setup | 2-3 minutes |
| Asset download | 5-10 minutes |
| Benchmark execution | 18-20 minutes |
| Report generation | 30 seconds |
| **Total** | **25-35 minutes** |

---

## Appendix B: Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024 | Initial reproducibility protocol |

---

*End of autoconstitution Benchmark Reproducibility Protocol*
