"""
Reproducibility verification for autoconstitution benchmark.

This module provides tools to verify that benchmark results are
reproducible by running the benchmark multiple times and comparing
results.

Version: 1.0.0
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np


def run_benchmark(seed: int, output_dir: Path, config_path: str = "./configs/m4_baseline.yaml") -> Path:
    """
    Run benchmark with given seed and return results path.
    
    Args:
        seed: Random seed to use
        output_dir: Directory for output
        config_path: Path to benchmark configuration
        
    Returns:
        Path to results file
        
    Raises:
        RuntimeError: If benchmark execution fails
    """
    result = subprocess.run([
        "python", "-m", "autoconstitution.benchmark",
        "--config", config_path,
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


def load_results(path: Path) -> Dict[str, any]:
    """
    Load results from a JSON file.
    
    Args:
        path: Path to results file
        
    Returns:
        Results dictionary
    """
    with open(path) as f:
        return json.load(f)


def extract_scores(results: Dict[str, any]) -> Dict[str, float]:
    """
    Extract relevant scores from results.
    
    Args:
        results: Results dictionary
        
    Returns:
        Dictionary of score names to values
    """
    r = results.get("results", {})
    
    return {
        "overall": r.get("overall_score"),
        "dls": r.get("dls", {}).get("score"),
        "chg": r.get("chg", {}).get("score"),
        "cbd": r.get("cbd", {}).get("score"),
        "dta": r.get("dta", {}).get("score"),
    }


def compare_results(
    path1: Path,
    path2: Path,
    tolerance: float = 0.01
) -> Tuple[bool, Dict[str, any]]:
    """
    Compare two benchmark results for reproducibility.
    
    Args:
        path1: First results file
        path2: Second results file
        tolerance: Floating point comparison tolerance
        
    Returns:
        Tuple of (is_reproducible, comparison_details)
    """
    results1 = load_results(path1)
    results2 = load_results(path2)
    
    scores1 = extract_scores(results1)
    scores2 = extract_scores(results2)
    
    differences = {}
    all_match = True
    
    for key in scores1.keys():
        s1 = scores1[key]
        s2 = scores2[key]
        
        if s1 is None or s2 is None:
            differences[key] = {
                "run1": s1,
                "run2": s2,
                "error": "Missing score"
            }
            all_match = False
        elif abs(s1 - s2) > tolerance:
            differences[key] = {
                "run1": s1,
                "run2": s2,
                "difference": abs(s1 - s2),
                "tolerance": tolerance
            }
            all_match = False
    
    return all_match, {
        "scores_match": all_match,
        "differences": differences,
        "tolerance": tolerance,
    }


def verify_reproducibility(
    num_runs: int = 3,
    seed: int = 42,
    output_dir: Path = Path("./reproducibility_test"),
    tolerance: float = 0.01,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Verify reproducibility by running benchmark multiple times.
    
    Args:
        num_runs: Number of times to run benchmark
        seed: Seed to use for all runs
        output_dir: Directory for test results
        tolerance: Floating point comparison tolerance
        verbose: Whether to print progress
        
    Returns:
        Reproducibility report dictionary
    """
    if verbose:
        print(f"Running reproducibility test with {num_runs} runs (seed={seed})...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmark multiple times
    results_paths: List[Path] = []
    for i in range(num_runs):
        if verbose:
            print(f"  Run {i+1}/{num_runs}...")
        
        try:
            path = run_benchmark(seed, output_dir)
            results_paths.append(path)
        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            return {
                "success": False,
                "error": f"Run {i+1} failed: {str(e)}",
                "num_runs": num_runs,
                "seed": seed,
            }
    
    # Compare all pairs
    if verbose:
        print("Comparing results...")
    
    all_reproducible = True
    pair_comparisons = []
    
    for i in range(len(results_paths)):
        for j in range(i+1, len(results_paths)):
            is_match, details = compare_results(
                results_paths[i],
                results_paths[j],
                tolerance
            )
            
            pair_comparisons.append({
                "pair": (i+1, j+1),
                "files": (str(results_paths[i]), str(results_paths[j])),
                "match": is_match,
                "details": details
            })
            
            if not is_match:
                all_reproducible = False
    
    # Calculate statistics
    scores_by_run = []
    for path in results_paths:
        results = load_results(path)
        scores = extract_scores(results)
        scores_by_run.append(scores)
    
    # Build report
    report = {
        "success": True,
        "num_runs": num_runs,
        "seed": seed,
        "tolerance": tolerance,
        "all_reproducible": all_reproducible,
        "pair_comparisons": pair_comparisons,
        "results_files": [str(p) for p in results_paths],
        "score_statistics": calculate_score_statistics(scores_by_run),
    }
    
    # Save report
    report_path = output_dir / "reproducibility_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    if verbose:
        print()
        print("=" * 50)
        print(f"Reproducibility test: {'PASSED' if all_reproducible else 'FAILED'}")
        print("=" * 50)
        print(f"Report saved to: {report_path}")
        
        if not all_reproducible:
            print()
            print("Differences found:")
            for comp in pair_comparisons:
                if not comp["match"]:
                    print(f"  Run {comp['pair'][0]} vs {comp['pair'][1]}:")
                    for key, diff in comp["details"]["differences"].items():
                        print(f"    {key}: {diff.get('run1')} vs {diff.get('run2')}")
    
    return report


def calculate_score_statistics(scores_by_run: List[Dict[str, float]]) -> Dict[str, any]:
    """
    Calculate statistics across multiple runs.
    
    Args:
        scores_by_run: List of score dictionaries
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    for key in ["overall", "dls", "chg", "cbd", "dta"]:
        values = [s[key] for s in scores_by_run if s[key] is not None]
        
        if values:
            stats[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "range": float(np.max(values) - np.min(values)),
            }
    
    return stats


def print_reproducibility_report(report: Dict[str, any]):
    """
    Print a formatted reproducibility report.
    
    Args:
        report: Reproducibility report dictionary
    """
    print()
    print("Reproducibility Verification Report")
    print("=" * 50)
    print()
    
    if not report.get("success"):
        print(f"Status: FAILED")
        print(f"Error: {report.get('error', 'Unknown error')}")
        return
    
    print(f"Number of runs: {report['num_runs']}")
    print(f"Seed: {report['seed']}")
    print(f"Tolerance: {report['tolerance']}")
    print()
    
    print(f"Overall: {'REPRODUCIBLE' if report['all_reproducible'] else 'NOT REPRODUCIBLE'}")
    print()
    
    # Score statistics
    if "score_statistics" in report:
        print("Score Statistics:")
        for key, stats in report["score_statistics"].items():
            print(f"  {key.upper()}:")
            print(f"    Mean: {stats['mean']:.2f}")
            print(f"    Std:  {stats['std']:.4f}")
            print(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        print()
    
    # Pair comparisons
    print("Pair-wise Comparisons:")
    for comp in report["pair_comparisons"]:
        status = "✓" if comp["match"] else "✗"
        print(f"  {status} Run {comp['pair'][0]} vs {comp['pair'][1]}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify autoconstitution benchmark reproducibility"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output",
        default="./reproducibility_test",
        help="Output directory (default: ./reproducibility_test)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Comparison tolerance (default: 0.01)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    report = verify_reproducibility(
        num_runs=args.runs,
        seed=args.seed,
        output_dir=Path(args.output),
        tolerance=args.tolerance,
        verbose=not args.quiet
    )
    
    if not args.quiet:
        print_reproducibility_report(report)
    
    # Exit with error code if not reproducible
    if not report.get("all_reproducible", False):
        exit(1)
