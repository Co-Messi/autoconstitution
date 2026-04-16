#!/bin/bash
# reproduce.sh - One-command benchmark reproduction for SwarmResearch
# Version: 1.0.0

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
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP $1/7]${NC} $2"; }

# Parse arguments
SEED=${1:-$DEFAULT_SEED}
OUTPUT_DIR=${2:-$RESULTS_DIR}

# Print banner
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         SwarmResearch Benchmark Reproduction                 ║"
echo "║                    Version $BENCHMARK_VERSION                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
log_info "Seed: $SEED"
log_info "Output: $OUTPUT_DIR"
echo ""

# Step 1: Environment Check
log_step "1" "Checking environment..."
python3 --version > /dev/null 2>&1 || { log_error "Python 3 not found. Please install Python 3.11."; exit 1; }

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" != "3.11" ]]; then
    log_warn "Python 3.11 recommended, found $PYTHON_VERSION"
    log_warn "Results may not be fully reproducible."
fi
log_info "Python version: $(python3 --version)"

# Step 2: Setup Virtual Environment
log_step "2" "Setting up virtual environment..."
if [ ! -d "./venv" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
log_info "Virtual environment activated."

# Step 3: Install Dependencies
log_step "3" "Installing dependencies..."
pip install --quiet --upgrade pip

if [ -f "requirements.lock" ]; then
    log_info "Installing from requirements.lock..."
    pip install --quiet -r requirements.lock
else
    log_info "Installing from requirements.txt..."
    pip install --quiet -r requirements.txt
fi
log_info "Dependencies installed."

# Step 4: Download Assets
log_step "4" "Downloading assets..."
if ! python -c "import swarmresearch" 2>/dev/null; then
    log_warn "swarmresearch package not found. Skipping asset download."
    log_warn "Please ensure models and datasets are manually placed in:"
    log_warn "  - ./models/swarm-agent-small-3B-int8.gguf"
    log_warn "  - ./datasets/swarmresearch_bench_v1/"
else
    python -m swarmresearch.download --dataset --model --verify || {
        log_error "Asset download failed. Please check your internet connection."
        exit 1
    }
    log_info "Assets downloaded and verified."
fi

# Step 5: Verify Environment
log_step "5" "Verifying environment..."
if python -c "import swarmresearch" 2>/dev/null; then
    python -m swarmresearch.verify --full || {
        log_warn "Environment verification had issues, but continuing..."
    }
else
    log_warn "swarmresearch package not available for verification."
fi

# Step 6: Run Benchmark
log_step "6" "Running benchmark (this may take 20-30 minutes)..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$OUTPUT_DIR/run_${TIMESTAMP}_seed${SEED}.json"

mkdir -p "$OUTPUT_DIR"

if python -c "import swarmresearch" 2>/dev/null; then
    python -m swarmresearch.benchmark \
        --config "$CONFIG_FILE" \
        --output "$RESULTS_FILE" \
        --seed "$SEED" \
        --verbose || { log_error "Benchmark execution failed"; exit 1; }
    log_info "Benchmark completed successfully."
else
    log_error "swarmresearch package not found. Cannot run benchmark."
    log_info "This is a template reproduction script."
    log_info "Please install the swarmresearch-benchmark package to run actual benchmarks."
    exit 1
fi

# Step 7: Generate Report
log_step "7" "Generating report..."
if python -c "import swarmresearch" 2>/dev/null; then
    python -m swarmresearch.report \
        --results "$RESULTS_FILE" \
        --output "$OUTPUT_DIR/report_${TIMESTAMP}.html" || {
        log_warn "Report generation failed, but benchmark completed."
    }
fi

# Print summary
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              Reproduction Complete!                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
log_info "Results: $RESULTS_FILE"
log_info "Report: $OUTPUT_DIR/report_${TIMESTAMP}.html"

# Display results summary
if [ -f "$RESULTS_FILE" ]; then
    echo ""
    echo "=== Benchmark Results Summary ==="
    python3 << EOF
import json
import sys

try:
    with open('$RESULTS_FILE') as f:
        data = json.load(f)
    
    results = data.get('results', {})
    overall = results.get('overall_score', 'N/A')
    
    print(f"Overall Score: {overall}")
    print("")
    print("Task Scores:")
    for task in ['dls', 'chg', 'cbd', 'dta']:
        task_data = results.get(task, {})
        score = task_data.get('score', 'N/A')
        time = task_data.get('time_seconds', 'N/A')
        print(f"  {task.upper()}: {score} (time: {time}s)")
    
    print("")
    print("Configuration:")
    config = data.get('configuration', {})
    print(f"  Seed: {config.get('seed', 'N/A')}")
    print(f"  Model: {data.get('model_info', {}).get('name', 'N/A')}")
    print(f"  Timestamp: {data.get('run_timestamp', 'N/A')}")
    
except Exception as e:
    print(f"Could not parse results: {e}")
EOF
fi

echo ""
log_info "To verify reproducibility, run again with the same seed."
log_info "Expected: Identical results (within floating-point tolerance)"
echo ""
log_info "Command to reproduce: ./reproduce.sh $SEED"
echo ""
