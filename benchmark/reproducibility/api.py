"""
Programmatic reproduction API for autoconstitution benchmark.

This module provides a high-level API for running the benchmark
reproducibility pipeline programmatically.

Version: 1.0.0
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from .seeds import set_all_seeds
from .verify_deps import verify_dependencies, check_python_version


class ReproductionRunner:
    """
    One-command benchmark reproduction runner.
    
    This class provides a fluent interface for running the complete
    benchmark reproduction pipeline.
    
    Example:
        >>> runner = ReproductionRunner(seed=42)
        >>> (runner
        ...     .setup()
        ...     .download_assets()
        ...     .verify()
        ...     .run()
        ...     .generate_report())
        >>> summary = runner.get_summary()
        >>> print(summary["overall_score"])
    """
    
    def __init__(
        self,
        seed: int = 42,
        output_dir: str = "./results",
        config_path: str = "./configs/m4_baseline.yaml",
        verbose: bool = True
    ):
        """
        Initialize the reproduction runner.
        
        Args:
            seed: Random seed for reproducibility
            output_dir: Directory for results
            config_path: Path to benchmark configuration
            verbose: Whether to print progress messages
        """
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.config_path = Path(config_path)
        self.verbose = verbose
        self.results: Optional[Dict[str, Any]] = None
        self.results_file: Optional[Path] = None
        
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def setup(self) -> "ReproductionRunner":
        """
        Set up the environment.
        
        Creates output directory and verifies Python version.
        
        Returns:
            Self for method chaining
        """
        self._log("Setting up environment...")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify Python version
        python_ok, python_msg = check_python_version()
        if not python_ok:
            raise RuntimeError(f"Python version check failed: {python_msg}")
        
        self._log(f"  Python: {python_msg}")
        
        # Set seeds
        set_all_seeds(self.seed)
        self._log(f"  Seed set to: {self.seed}")
        
        return self
    
    def download_assets(self) -> "ReproductionRunner":
        """
        Download required assets (models and datasets).
        
        Returns:
            Self for method chaining
        """
        self._log("Downloading assets...")
        
        try:
            result = subprocess.run(
                ["python", "-m", "autoconstitution.download", "--dataset", "--model", "--verify"],
                capture_output=True,
                text=True,
                check=True
            )
            self._log("  Assets downloaded successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Asset download failed: {e.stderr}")
        except FileNotFoundError:
            self._log("  Warning: autoconstitution package not found, skipping download")
            self._log("  Please ensure assets are manually placed")
        
        return self
    
    def verify(self) -> "ReproductionRunner":
        """
        Verify environment and dependencies.
        
        Returns:
            Self for method chaining
        """
        self._log("Verifying environment...")
        
        # Verify dependencies
        all_ok, req_mismatches, opt_mismatches = verify_dependencies()
        
        if not all_ok:
            for mismatch in req_mismatches:
                self._log(f"  ✗ {mismatch}")
            raise RuntimeError("Dependency verification failed")
        
        self._log("  Dependencies verified")
        
        # Try to run autoconstitution verify if available
        try:
            subprocess.run(
                ["python", "-m", "autoconstitution.verify", "--full"],
                capture_output=True,
                check=True
            )
            self._log("  autoconstitution verification passed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self._log("  autoconstitution verification skipped")
        
        return self
    
    def run(self) -> "ReproductionRunner":
        """
        Execute the benchmark.
        
        Returns:
            Self for method chaining
            
        Raises:
            RuntimeError: If benchmark execution fails
        """
        self._log("Running benchmark...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.output_dir / f"run_{timestamp}_seed{self.seed}.json"
        
        try:
            result = subprocess.run([
                "python", "-m", "autoconstitution.benchmark",
                "--config", str(self.config_path),
                "--output", str(self.results_file),
                "--seed", str(self.seed)
            ], capture_output=True, text=True, check=True)
            
            self._log(f"  Benchmark completed: {self.results_file}")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Benchmark execution failed: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                "autoconstitution package not found. "
                "Please install autoconstitution-benchmark package."
            )
        
        # Load results
        if self.results_file.exists():
            with open(self.results_file) as f:
                self.results = json.load(f)
        
        return self
    
    def generate_report(self) -> "ReproductionRunner":
        """
        Generate HTML report from results.
        
        Returns:
            Self for method chaining
            
        Raises:
            RuntimeError: If no results are available
        """
        if self.results is None:
            raise RuntimeError("No results available. Run benchmark first.")
        
        self._log("Generating report...")
        
        timestamp = self.results.get("run_timestamp", datetime.now().isoformat())
        report_file = self.output_dir / f"report_{timestamp[:19].replace(':', '')}.html"
        
        try:
            subprocess.run([
                "python", "-m", "autoconstitution.report",
                "--results", str(self.results_file),
                "--output", str(report_file)
            ], capture_output=True, check=True)
            
            self._log(f"  Report generated: {report_file}")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            self._log("  Report generation skipped")
        
        return self
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get benchmark summary.
        
        Returns:
            Dictionary with benchmark summary
        """
        if self.results is None:
            return {
                "status": "not_run",
                "message": "Benchmark has not been run yet"
            }
        
        results = self.results.get("results", {})
        
        return {
            "status": "complete",
            "overall_score": results.get("overall_score"),
            "task_scores": {
                "dls": results.get("dls", {}).get("score"),
                "chg": results.get("chg", {}).get("score"),
                "cbd": results.get("cbd", {}).get("score"),
                "dta": results.get("dta", {}).get("score"),
            },
            "seed": self.results.get("configuration", {}).get("seed"),
            "timestamp": self.results.get("run_timestamp"),
            "results_file": str(self.results_file) if self.results_file else None,
        }
    
    def get_detailed_results(self) -> Optional[Dict[str, Any]]:
        """
        Get complete benchmark results.
        
        Returns:
            Complete results dictionary or None if not run
        """
        return self.results


def reproduce(
    seed: int = 42,
    output_dir: str = "./results",
    config_path: str = "./configs/m4_baseline.yaml",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    One-command benchmark reproduction.
    
    This is a convenience function that runs the complete reproduction
    pipeline and returns a summary.
    
    Args:
        seed: Random seed for reproducibility
        output_dir: Directory for results
        config_path: Path to benchmark configuration
        verbose: Whether to print progress messages
        
    Returns:
        Benchmark summary dictionary
        
    Example:
        >>> summary = reproduce(seed=42)
        >>> print(f"Overall score: {summary['overall_score']}")
        
        # With custom settings
        >>> summary = reproduce(
        ...     seed=123,
        ...     output_dir="./my_results",
        ...     verbose=False
        ... )
    """
    runner = ReproductionRunner(seed, output_dir, config_path, verbose)
    
    (runner
        .setup()
        .download_assets()
        .verify()
        .run()
        .generate_report())
    
    return runner.get_summary()


def run_multiple(
    seeds: List[int],
    output_dir: str = "./results",
    config_path: str = "./configs/m4_baseline.yaml",
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Run benchmark multiple times with different seeds.
    
    Args:
        seeds: List of seeds to use
        output_dir: Directory for results
        config_path: Path to benchmark configuration
        verbose: Whether to print progress messages
        
    Returns:
        List of benchmark summaries
        
    Example:
        >>> summaries = run_multiple([42, 123, 456])
        >>> scores = [s['overall_score'] for s in summaries]
        >>> print(f"Mean score: {sum(scores)/len(scores)}")
    """
    summaries = []
    
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\nRun {i+1}/{len(seeds)} with seed {seed}")
            print("-" * 40)
        
        try:
            summary = reproduce(seed, output_dir, config_path, verbose)
            summaries.append(summary)
        except Exception as e:
            if verbose:
                print(f"Error with seed {seed}: {e}")
            summaries.append({
                "status": "error",
                "seed": seed,
                "error": str(e)
            })
    
    return summaries


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="autoconstitution Benchmark Reproduction API"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output",
        default="./results",
        help="Output directory (default: ./results)"
    )
    parser.add_argument(
        "--config",
        default="./configs/m4_baseline.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    try:
        summary = reproduce(
            seed=args.seed,
            output_dir=args.output,
            config_path=args.config,
            verbose=not args.quiet
        )
        
        print(json.dumps(summary, indent=2))
        
        # Exit with error code if benchmark failed
        if summary.get("status") != "complete":
            exit(1)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        exit(1)
