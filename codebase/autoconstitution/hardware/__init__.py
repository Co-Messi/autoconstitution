"""
Hardware acceleration module for autoconstitution.

Provides platform-specific optimizations for:
- Apple Silicon M4 (Neural Engine, Core ML, Unified Memory)
- Hardware detection and configuration
"""

from __future__ import annotations

# Hardware detection from detector module
from .detector import (
    HardwareDetector,
    HardwareProfile,
    CPUInfo,
    GPUInfo,
    NeuralEngineInfo,
    ResourceLimits,
    ExecutionConfig,
    ComputeBackend,
    ExecutionStrategy,
    detect_hardware,
    get_optimal_config,
    get_available_backends,
    create_execution_config,
    print_hardware_info,
)

# M4-specific optimizations
from .m4 import (
    # Main classes
    M4Optimizer,
    UnifiedMemoryManager,
    NeuralEngineAccelerator,
    CoreMLInferenceEngine,
    AppleSiliconProcessPool,
    M4HardwareDetector,
    
    # Configuration classes
    UnifiedMemoryConfig,
    CoreMLConfig,
    NeuralEngineConfig,
    ProcessPoolConfig,
    
    # Data classes
    M4HardwareInfo,
    EmbeddingResult,
    InferenceResult,
    
    # Enums
    ComputeUnit,
    M4ChipVariant,
    MemoryPressureLevel,
    OptimizationLevel,
    
    # Utility functions
    create_m4_optimizer,
    optimize_numpy_for_m4,
    get_optimal_batch_size,
)

__version__ = "1.0.0"

__all__ = [
    # Hardware detection
    "HardwareDetector",
    "HardwareProfile",
    "CPUInfo",
    "GPUInfo",
    "NeuralEngineInfo",
    "ResourceLimits",
    "ExecutionConfig",
    "ComputeBackend",
    "ExecutionStrategy",
    "detect_hardware",
    "get_optimal_config",
    "get_available_backends",
    "create_execution_config",
    "print_hardware_info",
    # M4-specific classes
    "M4Optimizer",
    "UnifiedMemoryManager",
    "NeuralEngineAccelerator",
    "CoreMLInferenceEngine",
    "AppleSiliconProcessPool",
    "M4HardwareDetector",
    
    # Configuration classes
    "UnifiedMemoryConfig",
    "CoreMLConfig",
    "NeuralEngineConfig",
    "ProcessPoolConfig",
    
    # Data classes
    "M4HardwareInfo",
    "EmbeddingResult",
    "InferenceResult",
    
    # Enums
    "ComputeUnit",
    "M4ChipVariant",
    "MemoryPressureLevel",
    "OptimizationLevel",
    
    # Utility functions
    "create_m4_optimizer",
    "optimize_numpy_for_m4",
    "get_optimal_batch_size",
]
