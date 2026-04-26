#!/bin/bash
# scripts/configure_macos.sh
# Configure macOS for reproducible autoconstitution benchmark execution
# Version: 1.0.0

set -e

echo "autoconstitution Benchmark - macOS Configuration"
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

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    log_error "This script is for macOS only"
    exit 1
fi

# Check for sudo
if [[ $EUID -ne 0 ]]; then
    log_warn "Some operations may require sudo privileges"
fi

# Verify M4 detection
log_info "Checking hardware..."
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
log_info "Detected chip: $CHIP"

if [[ "$CHIP" != *"Apple M4"* ]]; then
    log_warn "Expected Apple M4, found: $CHIP"
    log_warn "Benchmark results may vary on different hardware"
fi

# Check RAM
RAM_GB=$(system_profiler SPHardwareDataType 2>/dev/null | grep "Memory:" | awk '{print $2}' || echo "0")
log_info "Detected RAM: ${RAM_GB}GB"

if [[ "$RAM_GB" -lt 16 ]]; then
    log_warn "Minimum 16GB RAM recommended, found: ${RAM_GB}GB"
fi

# Configure system settings
echo ""
log_info "Configuring system settings..."

# Disable sleep during benchmark (when on AC power)
log_info "Disabling sleep on AC power..."
sudo pmset -c sleep 0 2>/dev/null || log_warn "Could not disable sleep (may need sudo)"

# Disable sleep on battery (optional)
# sudo pmset -b sleep 0 2>/dev/null || true

# Disable low power mode
log_info "Disabling low power mode..."
sudo pmset -c lowpowermode 0 2>/dev/null || log_warn "Could not disable low power mode"

# Set performance mode (if applicable)
log_info "Setting performance mode..."
sudo pmset -c gpuswitch 2 2>/dev/null || true

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

# Check for Homebrew
if command -v brew &> /dev/null; then
    log_info "Homebrew found: $(brew --version | head -1)"
else
    log_warn "Homebrew not found. Some dependencies may need manual installation"
fi

# Summary
echo ""
echo "=============================================="
log_info "macOS configuration complete!"
echo ""
echo "Next steps:"
echo "  1. Run: make setup"
echo "  2. Run: make reproduce"
echo ""
echo "To restore original settings after benchmark:"
echo "  sudo pmset -c sleep 10"
echo "  sudo pmset -c lowpowermode 1"
echo ""
