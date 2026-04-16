#!/bin/bash
# scripts/configure_linux.sh
# Configure Linux for reproducible SwarmResearch benchmark execution
# Version: 1.0.0

set -e

echo "SwarmResearch Benchmark - Linux Configuration"
echo "=============================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    log_error "This script is for Linux only"
    exit 1
fi

# Check for root/sudo
if [[ $EUID -ne 0 ]]; then
    log_warn "Some operations require sudo privileges"
fi

# Get system info
log_info "Detecting system configuration..."

# CPU info
CPU_COUNT=$(nproc 2>/dev/null || echo "Unknown")
log_info "CPU cores: $CPU_COUNT"

# Memory info
if [[ -f /proc/meminfo ]]; then
    MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    MEM_GB=$((MEM_KB / 1024 / 1024))
    log_info "Memory: ${MEM_GB}GB"
    
    if [[ "$MEM_GB" -lt 16 ]]; then
        log_warn "Minimum 16GB RAM recommended"
    fi
fi

# OS info
if [[ -f /etc/os-release ]]; then
    source /etc/os-release
    log_info "OS: $NAME $VERSION_ID"
fi

# Configure CPU governor
echo ""
log_info "Configuring CPU governor..."

if [[ -d /sys/devices/system/cpu/cpu0/cpufreq ]]; then
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        if [[ -w "$cpu" ]]; then
            echo performance | sudo tee "$cpu" > /dev/null 2>&1 || true
        fi
    done
    log_info "CPU governor set to performance mode"
else
    log_warn "CPU frequency scaling not available"
fi

# Disable CPU frequency scaling service
if systemctl is-active --quiet ondemand 2>/dev/null; then
    log_info "Disabling ondemand service..."
    sudo systemctl stop ondemand 2>/dev/null || true
fi

# Set process priority (if possible)
log_info "Setting process priority..."
sudo renice -n -10 -p $$ 2>/dev/null || log_warn "Could not set process priority"

# Configure kernel parameters for reproducibility
echo ""
log_info "Configuring kernel parameters..."

# Disable random address space layout (for reproducibility)
# Note: This is a security trade-off for reproducibility
# echo 0 | sudo tee /proc/sys/kernel/randomize_va_space 2>/dev/null || log_warn "Could not disable ASLR"

# Check Python installation
echo ""
log_info "Checking Python installation..."

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    log_info "Found: $PYTHON_VERSION"
    
    # Check if Python 3.11
    if python3 -c 'import sys; exit(0 if sys.version_info[:2] == (3, 11) else 1)' 2>/dev/null; then
        log_info "Python 3.11 verified"
    else
        log_warn "Python 3.11 recommended for reproducibility"
    fi
else
    log_error "Python 3 not found. Please install Python 3.11"
    exit 1
fi

# Check for required packages
echo ""
log_info "Checking for build dependencies..."

MISSING_DEPS=()

if ! command -v gcc &> /dev/null; then
    MISSING_DEPS+=("build-essential")
fi

if ! command -v curl &> /dev/null; then
    MISSING_DEPS+=("curl")
fi

if ! command -v git &> /dev/null; then
    MISSING_DEPS+=("git")
fi

if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
    log_warn "Missing dependencies: ${MISSING_DEPS[*]}"
    log_warn "Install with: sudo apt-get install ${MISSING_DEPS[*]}"
fi

# Summary
echo ""
echo "=============================================="
log_info "Linux configuration complete!"
echo ""
echo "Next steps:"
echo "  1. Run: make setup"
echo "  2. Run: make reproduce"
echo ""
echo "Note: Results on Linux may differ from M4 baseline"
echo "      due to hardware and software differences."
echo ""
