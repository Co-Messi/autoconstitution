"""
Dependency version verification for SwarmResearch benchmark.

This module verifies that all required dependencies are installed
with the correct versions for reproducible benchmark execution.

Version: 1.0.0
"""

import sys
from typing import List, Tuple, Dict

# Required package versions for reproducibility
REQUIRED_VERSIONS = {
    "numpy": "1.26.4",
    "scipy": "1.12.0",
    "pandas": "2.2.0",
    "pydantic": "2.7.0",
    "pydantic-core": "2.18.0",
    "llama-cpp-python": "0.2.90",
    "pyyaml": "6.0.1",
    "requests": "2.31.0",
    "httpx": "0.26.0",
    "structlog": "24.1.0",
    "rich": "13.7.0",
    "tqdm": "4.66.1",
    "click": "8.1.7",
    "joblib": "1.3.2",
    "pytest": "8.0.0",
}

# Optional packages (warn but don't fail)
OPTIONAL_VERSIONS = {
    "torch": "2.2.0",
    "pyarrow": "15.0.0",
    "python-dotenv": "1.0.0",
}


def check_python_version() -> Tuple[bool, str]:
    """
    Verify Python version compatibility.
    
    Returns:
        Tuple of (is_compatible, message)
        
    Example:
        >>> is_ok, msg = check_python_version()
        >>> print(is_ok)
        True
    """
    version = sys.version_info
    if version.major == 3 and version.minor == 11:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python 3.11 required, found {version.major}.{version.minor}.{version.micro}"


def get_installed_packages() -> Dict[str, str]:
    """
    Get dictionary of installed packages and their versions.
    
    Returns:
        Dictionary mapping package names to versions
    """
    try:
        import pkg_resources
        return {
            dist.key: dist.version
            for dist in pkg_resources.working_set
        }
    except Exception as e:
        print(f"Warning: Could not get installed packages: {e}")
        return {}


def verify_dependencies(
    required: Dict[str, str] = None,
    optional: Dict[str, str] = None
) -> Tuple[bool, List[str], List[str]]:
    """
    Verify all dependencies match required versions.
    
    Args:
        required: Dictionary of required package versions
        optional: Dictionary of optional package versions
        
    Returns:
        Tuple of (all_passed, required_mismatches, optional_mismatches)
        
    Example:
        >>> success, req_err, opt_err = verify_dependencies()
        >>> print(success)
        True
    """
    if required is None:
        required = REQUIRED_VERSIONS
    if optional is None:
        optional = OPTIONAL_VERSIONS
    
    installed = get_installed_packages()
    
    required_mismatches = []
    optional_mismatches = []
    
    # Check required packages
    for package, required_version in required.items():
        installed_version = installed.get(package)
        
        if installed_version is None:
            required_mismatches.append(
                f"{package}: NOT INSTALLED (required {required_version})"
            )
        elif installed_version != required_version:
            required_mismatches.append(
                f"{package}: {installed_version} (required {required_version})"
            )
    
    # Check optional packages
    for package, required_version in optional.items():
        installed_version = installed.get(package)
        
        if installed_version is not None and installed_version != required_version:
            optional_mismatches.append(
                f"{package}: {installed_version} (recommended {required_version})"
            )
    
    all_passed = len(required_mismatches) == 0
    
    return all_passed, required_mismatches, optional_mismatches


def verify_specific_package(package_name: str, required_version: str) -> Tuple[bool, str]:
    """
    Verify a specific package version.
    
    Args:
        package_name: Name of the package to check
        required_version: Required version string
        
    Returns:
        Tuple of (is_correct, message)
    """
    installed = get_installed_packages()
    installed_version = installed.get(package_name)
    
    if installed_version is None:
        return False, f"{package_name}: NOT INSTALLED (required {required_version})"
    elif installed_version != required_version:
        return False, f"{package_name}: {installed_version} (required {required_version})"
    else:
        return True, f"{package_name}: {installed_version} ✓"


def get_dependency_report() -> Dict[str, any]:
    """
    Generate a complete dependency verification report.
    
    Returns:
        Dictionary with complete dependency status
    """
    python_ok, python_msg = check_python_version()
    all_ok, req_mismatches, opt_mismatches = verify_dependencies()
    
    return {
        "python": {
            "compatible": python_ok,
            "message": python_msg,
        },
        "required_packages": {
            "all_match": len(req_mismatches) == 0,
            "mismatches": req_mismatches,
        },
        "optional_packages": {
            "all_match": len(opt_mismatches) == 0,
            "mismatches": opt_mismatches,
        },
        "overall_compatible": python_ok and len(req_mismatches) == 0,
        "installed_packages": get_installed_packages(),
    }


def print_dependency_report():
    """
    Print a formatted dependency verification report.
    """
    report = get_dependency_report()
    
    print("Dependency Verification Report")
    print("=" * 50)
    print()
    
    # Python version
    python_status = "✓" if report["python"]["compatible"] else "✗"
    print(f"{python_status} Python: {report['python']['message']}")
    print()
    
    # Required packages
    print("Required Packages:")
    if report["required_packages"]["all_match"]:
        print("  ✓ All required packages match specified versions")
    else:
        for mismatch in report["required_packages"]["mismatches"]:
            print(f"  ✗ {mismatch}")
    print()
    
    # Optional packages
    if report["optional_packages"]["mismatches"]:
        print("Optional Packages (warnings only):")
        for mismatch in report["optional_packages"]["mismatches"]:
            print(f"  ! {mismatch}")
        print()
    
    # Overall
    if report["overall_compatible"]:
        print("Overall: ✓ All dependencies verified")
    else:
        print("Overall: ✗ Dependency verification failed")


if __name__ == "__main__":
    # Run verification when executed directly
    print_dependency_report()
    
    # Exit with error code if verification failed
    report = get_dependency_report()
    if not report["overall_compatible"]:
        sys.exit(1)
