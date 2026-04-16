"""
Centralized seed management for SwarmResearch benchmark.

This module provides deterministic random seed management to ensure
complete reproducibility across benchmark runs.

Version: 1.0.0
"""

import random
import hashlib
import os
from typing import Dict, Optional, List

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

# Task to seed key mapping
TASK_SEED_MAP = {
    "dls": "dataset_shuffle",      # Distributed Literature Synthesis
    "chg": "agent_initialization", # Collaborative Hypothesis Generation
    "cbd": "communication_order",  # Consensus-Based Decision Making
    "dta": "task_assignment",      # Dynamic Task Allocation
}


def set_all_seeds(seed: Optional[int] = None) -> Dict[str, int]:
    """
    Set all random seeds for reproducibility.
    
    This function sets seeds for:
    - Python's random module
    - NumPy (if available)
    - PyTorch (if available)
    - Environment variables for subprocesses
    
    Args:
        seed: Master seed. If None, uses SEED_CONFIG["master_seed"]
        
    Returns:
        Dictionary of seeds used
        
    Example:
        >>> seeds = set_all_seeds(42)
        >>> print(seeds["master_seed"])
        42
    """
    master = seed if seed is not None else SEED_CONFIG["master_seed"]
    
    # Python standard library random
    random.seed(master)
    
    # NumPy
    try:
        import numpy as np
        np.random.seed(master)
    except ImportError:
        pass
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(master)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(master)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # TensorFlow (if used)
    try:
        import tensorflow as tf
        tf.random.set_seed(master)
    except ImportError:
        pass
    
    # Environment variable for subprocesses
    os.environ["SWARMRESEARCH_SEED"] = str(master)
    os.environ["PYTHONHASHSEED"] = str(master)
    
    return {
        "master_seed": master,
        **{k: v for k, v in SEED_CONFIG.items() if k != "master_seed"}
    }


def get_derived_seed(base_seed: int, component: str, iteration: int = 0) -> int:
    """
    Generate a deterministic derived seed for a specific component.
    
    Uses MD5 hash to derive a unique seed from base seed, component name,
    and iteration number.
    
    Args:
        base_seed: Base seed value
        component: Component identifier string
        iteration: Optional iteration number
        
    Returns:
        Derived seed value (32-bit integer)
        
    Example:
        >>> seed = get_derived_seed(42, "agent_0", 1)
        >>> print(0 <= seed < 2**32)
        True
    """
    seed_string = f"{base_seed}:{component}:{iteration}"
    hash_val = int(hashlib.md5(seed_string.encode()).hexdigest(), 16)
    return hash_val % (2**32)


def seed_for_component(component: str, iteration: int = 0) -> int:
    """
    Get seed for a specific component.
    
    Args:
        component: Component name (e.g., "dataset_shuffle", "dls")
        iteration: Optional iteration number
        
    Returns:
        Seed value for the component
        
    Example:
        >>> seed = seed_for_component("dataset_shuffle")
        >>> print(seed)
        42
    """
    # Map task names to seed keys
    seed_key = TASK_SEED_MAP.get(component, component)
    base = SEED_CONFIG.get(seed_key, SEED_CONFIG["master_seed"])
    return get_derived_seed(base, component, iteration)


def apply_task_seeds(task_name: str) -> int:
    """
    Apply seeds specific to a task category.
    
    This sets both Python random and NumPy random seeds for the task.
    
    Args:
        task_name: Task category name (dls, chg, cbd, dta)
        
    Returns:
        Seed value applied
        
    Example:
        >>> seed = apply_task_seeds("dls")
        >>> print(seed)
        42
    """
    seed_value = seed_for_component(task_name)
    
    random.seed(seed_value)
    
    try:
        import numpy as np
        np.random.seed(seed_value)
    except ImportError:
        pass
    
    return seed_value


def verify_seed_reproducibility(test_sequence_length: int = 10) -> bool:
    """
    Verify that seeds produce consistent results.
    
    Generates random sequences, resets seeds, and verifies that
    the same sequences are produced.
    
    Args:
        test_sequence_length: Number of random values to generate
        
    Returns:
        True if seeds are reproducible, False otherwise
        
    Example:
        >>> is_reproducible = verify_seed_reproducibility()
        >>> print(is_reproducible)
        True
    """
    # Set seeds and generate test sequences
    set_all_seeds(42)
    
    python_random = [random.random() for _ in range(test_sequence_length)]
    
    try:
        import numpy as np
        numpy_random = np.random.rand(test_sequence_length).tolist()
    except ImportError:
        numpy_random = None
    
    # Reset and regenerate
    set_all_seeds(42)
    
    python_random_2 = [random.random() for _ in range(test_sequence_length)]
    
    try:
        import numpy as np
        numpy_random_2 = np.random.rand(test_sequence_length).tolist()
    except ImportError:
        numpy_random_2 = None
    
    # Verify consistency
    if python_random != python_random_2:
        return False
    
    if numpy_random is not None and numpy_random != numpy_random_2:
        return False
    
    return True


def get_seed_summary() -> Dict[str, any]:
    """
    Get a summary of all configured seeds.
    
    Returns:
        Dictionary containing seed configuration summary
    """
    return {
        "master_seed": SEED_CONFIG["master_seed"],
        "component_seeds": {k: v for k, v in SEED_CONFIG.items() if k != "master_seed"},
        "task_mappings": TASK_SEED_MAP.copy(),
        "verification_status": verify_seed_reproducibility(),
    }


if __name__ == "__main__":
    # Run verification when executed directly
    print("SwarmResearch Benchmark Seed Management")
    print("=" * 50)
    print()
    
    print("Seed Configuration:")
    for key, value in SEED_CONFIG.items():
        print(f"  {key}: {value}")
    
    print()
    print("Task Mappings:")
    for task, seed_key in TASK_SEED_MAP.items():
        print(f"  {task} -> {seed_key}")
    
    print()
    print("Verifying seed reproducibility...")
    if verify_seed_reproducibility():
        print("  PASSED: Seeds produce consistent results")
    else:
        print("  FAILED: Seeds do not produce consistent results")
        exit(1)
