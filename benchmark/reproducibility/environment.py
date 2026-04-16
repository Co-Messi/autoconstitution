"""
Environment documentation and verification for SwarmResearch benchmark.

This module captures and documents the complete environment state
to ensure reproducibility across different systems.

Version: 1.0.0
"""

import platform
import sys
import subprocess
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


def capture_environment() -> Dict[str, Any]:
    """
    Capture complete environment state for reproducibility.
    
    Collects information about:
    - Platform and OS
    - Python environment
    - Hardware configuration
    - Environment variables
    - Installed packages
    
    Returns:
        Dictionary containing complete environment state
        
    Example:
        >>> env = capture_environment()
        >>> print(env["platform"]["system"])
        'Darwin'  # or 'Linux', 'Windows'
    """
    env = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "benchmark_version": "1.0.0",
        "platform": capture_platform_info(),
        "python": capture_python_info(),
        "hardware": capture_hardware_info(),
        "environment_variables": capture_relevant_env_vars(),
        "installed_packages": capture_installed_packages(),
    }
    
    return env


def capture_platform_info() -> Dict[str, str]:
    """
    Capture platform information.
    
    Returns:
        Dictionary with platform details
    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": platform.node(),
    }


def capture_python_info() -> Dict[str, Any]:
    """
    Capture Python environment information.
    
    Returns:
        Dictionary with Python details
    """
    return {
        "version": sys.version,
        "version_info": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
            "releaselevel": sys.version_info.releaselevel,
        },
        "executable": sys.executable,
        "prefix": sys.prefix,
        "path": sys.path,
        "implementation": platform.python_implementation(),
    }


def capture_hardware_info() -> Dict[str, Any]:
    """
    Capture hardware information.
    
    Platform-specific hardware detection for macOS and Linux.
    
    Returns:
        Dictionary with hardware details
    """
    hw_info = {
        "platform": platform.system(),
    }
    
    # macOS specific
    if platform.system() == "Darwin":
        try:
            hw_info["chip"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True
            ).strip()
            
            # Get memory info
            mem_bytes = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True
            ).strip()
            hw_info["memory_bytes"] = int(mem_bytes)
            hw_info["memory_gb"] = int(mem_bytes) / (1024**3)
            
            # Get CPU count
            cpu_count = subprocess.check_output(
                ["sysctl", "-n", "hw.ncpu"],
                text=True
            ).strip()
            hw_info["cpu_count"] = int(cpu_count)
            
        except Exception as e:
            hw_info["error"] = str(e)
    
    # Linux specific
    elif platform.system() == "Linux":
        try:
            # CPU info
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
                hw_info["cpuinfo"] = cpuinfo[:2000]  # First 2000 chars
            
            # Memory info
            with open("/proc/meminfo") as f:
                meminfo = f.read()
                hw_info["meminfo"] = meminfo[:1000]  # First 1000 chars
            
            # CPU count
            cpu_count = subprocess.check_output(
                ["nproc"],
                text=True
            ).strip()
            hw_info["cpu_count"] = int(cpu_count)
            
        except Exception as e:
            hw_info["error"] = str(e)
    
    return hw_info


def capture_relevant_env_vars() -> Dict[str, str]:
    """
    Capture SwarmResearch-related environment variables.
    
    Returns:
        Dictionary of relevant environment variables
    """
    relevant_prefixes = (
        "SWARMRESEARCH",
        "PYTHON",
        "OMP",
        "MKL",
        "CUDA",
        "CUBLAS",
        "TOKENIZERS",
    )
    
    return {
        k: v for k, v in os.environ.items()
        if any(k.startswith(p) for p in relevant_prefixes)
    }


def capture_installed_packages() -> Dict[str, str]:
    """
    Capture installed Python packages.
    
    Returns:
        Dictionary of package names and versions
    """
    try:
        import pkg_resources
        return {
            dist.key: dist.version
            for dist in pkg_resources.working_set
        }
    except Exception:
        return {}


def save_environment_report(output_path: str = "./environment_report.json") -> str:
    """
    Save environment report to file.
    
    Args:
        output_path: Path to save the report
        
    Returns:
        Path to the saved report
        
    Example:
        >>> path = save_environment_report("./my_env.json")
        >>> print(path)
        './my_env.json'
    """
    env = capture_environment()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(env, f, indent=2)
    
    return output_path


def check_m4_compatibility() -> Dict[str, Any]:
    """
    Check if the current system meets M4 baseline requirements.
    
    Returns:
        Dictionary with compatibility check results
    """
    results = {
        "compatible": True,
        "checks": {},
        "warnings": [],
    }
    
    # Check platform
    if platform.system() != "Darwin":
        results["checks"]["platform"] = {
            "passed": False,
            "message": f"Expected macOS, found {platform.system()}"
        }
        results["warnings"].append("Non-macOS platform may produce different results")
    else:
        results["checks"]["platform"] = {"passed": True}
    
    # Check chip
    if platform.system() == "Darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True
            ).strip()
            
            if "M4" not in chip:
                results["checks"]["chip"] = {
                    "passed": False,
                    "message": f"Expected Apple M4, found {chip}"
                }
                results["warnings"].append("Non-M4 chip may produce different performance")
            else:
                results["checks"]["chip"] = {"passed": True, "value": chip}
        except Exception as e:
            results["checks"]["chip"] = {"passed": False, "message": str(e)}
    
    # Check RAM
    try:
        if platform.system() == "Darwin":
            mem_bytes = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True
            ).strip())
            mem_gb = mem_bytes / (1024**3)
            
            if mem_gb < 16:
                results["checks"]["memory"] = {
                    "passed": False,
                    "message": f"Minimum 16GB RAM recommended, found {mem_gb:.1f}GB"
                }
                results["warnings"].append("Insufficient RAM may cause failures")
            else:
                results["checks"]["memory"] = {
                    "passed": True,
                    "value_gb": mem_gb
                }
    except Exception as e:
        results["checks"]["memory"] = {"passed": False, "message": str(e)}
    
    # Check Python version
    if sys.version_info[:2] != (3, 11):
        results["checks"]["python_version"] = {
            "passed": False,
            "message": f"Python 3.11 recommended, found {sys.version_info.major}.{sys.version_info.minor}"
        }
        results["warnings"].append("Non-3.11 Python may affect reproducibility")
    else:
        results["checks"]["python_version"] = {"passed": True}
    
    # Overall compatibility
    results["compatible"] = all(
        check.get("passed", False)
        for check in results["checks"].values()
    )
    
    return results


if __name__ == "__main__":
    # Run environment capture when executed directly
    print("SwarmResearch Benchmark Environment Capture")
    print("=" * 50)
    print()
    
    # Check M4 compatibility
    print("Checking M4 compatibility...")
    compat = check_m4_compatibility()
    
    for check_name, check_result in compat["checks"].items():
        status = "✓" if check_result.get("passed") else "✗"
        print(f"  {status} {check_name}")
        if not check_result.get("passed"):
            print(f"    {check_result.get('message', '')}")
    
    if compat["warnings"]:
        print()
        print("Warnings:")
        for warning in compat["warnings"]:
            print(f"  ! {warning}")
    
    print()
    print(f"Overall compatibility: {'PASSED' if compat['compatible'] else 'FAILED'}")
    print()
    
    # Save environment report
    report_path = save_environment_report()
    print(f"Environment report saved to: {report_path}")
