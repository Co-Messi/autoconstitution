"""
SwarmResearch Benchmark Reproducibility Module

This module provides tools and utilities for ensuring reproducible
benchmark execution across different environments and runs.

Version: 1.0.0
"""

from .seeds import (
    SEED_CONFIG,
    set_all_seeds,
    get_derived_seed,
    seed_for_component,
    verify_seed_reproducibility,
)

from .environment import (
    capture_environment,
    capture_hardware_info,
    capture_relevant_env_vars,
    save_environment_report,
)

from .verify_deps import (
    verify_dependencies,
    check_python_version,
    REQUIRED_VERSIONS,
)

from .api import (
    ReproductionRunner,
    reproduce,
)

__version__ = "1.0.0"

__all__ = [
    # Seeds
    "SEED_CONFIG",
    "set_all_seeds",
    "get_derived_seed",
    "seed_for_component",
    "verify_seed_reproducibility",
    # Environment
    "capture_environment",
    "capture_hardware_info",
    "capture_relevant_env_vars",
    "save_environment_report",
    # Dependencies
    "verify_dependencies",
    "check_python_version",
    "REQUIRED_VERSIONS",
    # API
    "ReproductionRunner",
    "reproduce",
]
