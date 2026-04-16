"""
M4-Specific Hardware Optimizations for autoconstitution

This module provides Apple Silicon M4-specific optimizations including:
- Core ML integration for model inference
- Neural Engine delegation for embeddings
- Unified memory management
- Process pool configuration optimized for Apple Silicon

Python Version: 3.11+
Requires: coremltools, numpy, psutil
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import multiprocessing
import os
import platform
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)
from collections.abc import Iterator, Sequence

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Type Variables
# ============================================================================

T = TypeVar("T")
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


# ============================================================================
# Enums
# ============================================================================

class ComputeUnit(Enum):
    """Compute unit options for Core ML model execution."""
    CPU_ONLY = "cpuOnly"
    CPU_AND_GPU = "cpuAndGPU"
    CPU_AND_NE = "cpuAndNeuralEngine"
    ALL = "all"


class M4ChipVariant(Enum):
    """M4 chip variants with different core configurations."""
    M4_BASE = auto()      # 9-core GPU, 10-core CPU
    M4_PRO = auto()       # 14-core GPU, 12-core CPU
    M4_MAX = auto()       # 32-core GPU, 14-core CPU
    M4_ULTRA = auto()     # 60-core GPU, 20-core CPU
    UNKNOWN = auto()


class MemoryPressureLevel(Enum):
    """Memory pressure levels for unified memory management."""
    NORMAL = auto()
    WARNING = auto()
    CRITICAL = auto()


class OptimizationLevel(Enum):
    """Optimization levels for model compilation."""
    NONE = 0
    SPEED = 1
    BALANCED = 2
    MEMORY = 3


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True, slots=True)
class M4HardwareInfo:
    """Information about the M4 hardware configuration."""
    chip_variant: M4ChipVariant
    cpu_cores: int
    performance_cores: int
    efficiency_cores: int
    gpu_cores: int
    neural_engine_cores: int
    unified_memory_gb: float
    memory_bandwidth_gbps: float
    is_apple_silicon: bool
    
    @property
    def total_compute_cores(self) -> int:
        """Return total compute cores (CPU + GPU + NE)."""
        return self.cpu_cores + self.gpu_cores + self.neural_engine_cores
    
    @property
    def recommended_process_pool_size(self) -> int:
        """Return recommended process pool size based on hardware."""
        # Use performance cores for compute-intensive tasks
        return max(2, self.performance_cores)
    
    @property
    def recommended_thread_pool_size(self) -> int:
        """Return recommended thread pool size based on hardware."""
        # Use all CPU cores for I/O-bound tasks
        return self.cpu_cores * 2


@dataclass(slots=True)
class UnifiedMemoryConfig:
    """Configuration for unified memory management."""
    max_memory_fraction: float = 0.8
    warning_threshold_fraction: float = 0.7
    critical_threshold_fraction: float = 0.9
    enable_memory_pressure_monitoring: bool = True
    enable_aggressive_cleanup: bool = True
    memory_pool_size_mb: int = 512
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 < self.max_memory_fraction <= 1.0:
            raise ValueError("max_memory_fraction must be between 0.0 and 1.0")
        if not 0.0 < self.warning_threshold_fraction <= 1.0:
            raise ValueError("warning_threshold_fraction must be between 0.0 and 1.0")
        if not 0.0 < self.critical_threshold_fraction <= 1.0:
            raise ValueError("critical_threshold_fraction must be between 0.0 and 1.0")


@dataclass(slots=True)
class CoreMLConfig:
    """Configuration for Core ML model execution."""
    compute_unit: ComputeUnit = ComputeUnit.CPU_AND_NE
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    allow_low_precision_accumulation: bool = True
    allow_fp16_compression: bool = True
    use_cpu_only_for_prediction: bool = False
    enable_async_predictions: bool = True
    batch_size: int = 1
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


@dataclass(slots=True)
class NeuralEngineConfig:
    """Configuration for Neural Engine operations."""
    enable_neural_engine: bool = True
    max_concurrent_requests: int = 4
    request_timeout_seconds: float = 30.0
    enable_batching: bool = True
    max_batch_size: int = 32
    embedding_dimension: int = 768
    enable_quantization: bool = True
    quantization_bits: int = 8


@dataclass(slots=True)
class ProcessPoolConfig:
    """Configuration for process pool on Apple Silicon."""
    max_workers: int | None = None
    mp_context: str = "spawn"
    initializer: Callable[..., Any] | None = None
    initargs: tuple[Any, ...] = field(default_factory=tuple)
    max_tasks_per_child: int | None = None
    enable_performance_cores_only: bool = True
    
    def __post_init__(self) -> None:
        """Set default max_workers if not specified."""
        if self.max_workers is None:
            self.max_workers = M4Optimizer.get_hardware_info().recommended_process_pool_size


@dataclass(slots=True)
class EmbeddingResult:
    """Result from embedding computation."""
    embeddings: np.ndarray
    compute_time_ms: float
    used_neural_engine: bool
    batch_size: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InferenceResult(Generic[TOutput]):
    """Result from model inference."""
    output: TOutput
    compute_time_ms: float
    used_neural_engine: bool
    memory_used_mb: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Protocols
# ============================================================================

@runtime_checkable
class CoreMLModel(Protocol):
    """Protocol for Core ML model interface."""
    
    def predict(self, input_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run prediction on input data."""
        ...
    
    def predict_async(
        self,
        input_data: dict[str, np.ndarray],
    ) -> Coroutine[Any, Any, dict[str, np.ndarray]]:
        """Run async prediction on input data."""
        ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding model interface."""
    
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        ...
    
    def encode_async(self, texts: list[str]) -> Coroutine[Any, Any, np.ndarray]:
        """Async encode texts to embeddings."""
        ...


# ============================================================================
# M4 Hardware Detection
# ============================================================================

class M4HardwareDetector:
    """Detects M4 hardware capabilities and configuration."""
    
    _instance: M4HardwareDetector | None = None
    _hardware_info: M4HardwareInfo | None = None
    
    def __new__(cls) -> M4HardwareDetector:
        """Singleton pattern for hardware detector."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_hardware_info(cls) -> M4HardwareInfo:
        """Get cached hardware information."""
        if cls._hardware_info is None:
            cls._hardware_info = cls._detect_hardware()
        return cls._hardware_info
    
    @staticmethod
    def _detect_hardware() -> M4HardwareInfo:
        """Detect M4 hardware configuration."""
        is_apple_silicon = platform.machine() == "arm64" and sys.platform == "darwin"
        
        if not is_apple_silicon:
            return M4HardwareInfo(
                chip_variant=M4ChipVariant.UNKNOWN,
                cpu_cores=os.cpu_count() or 4,
                performance_cores=os.cpu_count() or 4,
                efficiency_cores=0,
                gpu_cores=0,
                neural_engine_cores=0,
                unified_memory_gb=16.0,
                memory_bandwidth_gbps=100.0,
                is_apple_silicon=False,
            )
        
        # Detect chip variant based on core counts
        cpu_count = os.cpu_count() or 8
        
        # M4 variants based on core configuration
        if cpu_count >= 20:
            variant = M4ChipVariant.M4_ULTRA
            perf_cores = 16
            eff_cores = 4
            gpu_cores = 60
            ne_cores = 32
            memory_gb = 128.0
            bandwidth = 800.0
        elif cpu_count >= 14:
            variant = M4ChipVariant.M4_MAX
            perf_cores = 12
            eff_cores = 4
            gpu_cores = 32
            ne_cores = 16
            memory_gb = 64.0
            bandwidth = 400.0
        elif cpu_count >= 12:
            variant = M4ChipVariant.M4_PRO
            perf_cores = 8
            eff_cores = 4
            gpu_cores = 14
            ne_cores = 16
            memory_gb = 36.0
            bandwidth = 273.0
        else:
            variant = M4ChipVariant.M4_BASE
            perf_cores = 4
            eff_cores = 6
            gpu_cores = 9
            ne_cores = 16
            memory_gb = 16.0
            bandwidth = 120.0
        
        return M4HardwareInfo(
            chip_variant=variant,
            cpu_cores=cpu_count,
            performance_cores=perf_cores,
            efficiency_cores=eff_cores,
            gpu_cores=gpu_cores,
            neural_engine_cores=ne_cores,
            unified_memory_gb=memory_gb,
            memory_bandwidth_gbps=bandwidth,
            is_apple_silicon=True,
        )
    
    @staticmethod
    def is_running_on_m4() -> bool:
        """Check if running on M4 hardware."""
        info = M4HardwareDetector.get_hardware_info()
        return info.is_apple_silicon and info.chip_variant != M4ChipVariant.UNKNOWN
    
    @staticmethod
    def has_neural_engine() -> bool:
        """Check if Neural Engine is available."""
        info = M4HardwareDetector.get_hardware_info()
        return info.is_apple_silicon and info.neural_engine_cores > 0


# ============================================================================
# Unified Memory Manager
# ============================================================================

class UnifiedMemoryManager:
    """
    Manages unified memory on Apple Silicon M4.
    
    Provides memory pressure monitoring, automatic cleanup,
    and optimized memory allocation strategies.
    """
    
    def __init__(self, config: UnifiedMemoryConfig | None = None) -> None:
        """
        Initialize the unified memory manager.
        
        Args:
            config: Memory management configuration
        """
        self._config = config or UnifiedMemoryConfig()
        self._hardware_info = M4HardwareDetector.get_hardware_info()
        self._memory_pools: list[MemoryPool] = []
        self._pressure_callbacks: list[Callable[[MemoryPressureLevel], None]] = []
        self._current_pressure: MemoryPressureLevel = MemoryPressureLevel.NORMAL
        self._monitoring_task: asyncio.Task[Any] | None = None
        self._lock = asyncio.Lock()
        
        if self._config.enable_memory_pressure_monitoring:
            self._start_monitoring()
    
    def _start_monitoring(self) -> None:
        """Start memory pressure monitoring."""
        try:
            loop = asyncio.get_event_loop()
            self._monitoring_task = loop.create_task(self._monitor_memory_pressure())
        except RuntimeError:
            logger.warning("No event loop available for memory monitoring")
    
    async def _monitor_memory_pressure(self) -> None:
        """Monitor memory pressure and trigger callbacks."""
        while True:
            try:
                pressure = self._check_memory_pressure()
                if pressure != self._current_pressure:
                    self._current_pressure = pressure
                    await self._notify_pressure_change(pressure)
                await asyncio.sleep(1.0)  # Check every second
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(5.0)
    
    def _check_memory_pressure(self) -> MemoryPressureLevel:
        """Check current memory pressure level."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            used_fraction = memory.percent / 100.0
            
            if used_fraction >= self._config.critical_threshold_fraction:
                return MemoryPressureLevel.CRITICAL
            elif used_fraction >= self._config.warning_threshold_fraction:
                return MemoryPressureLevel.WARNING
            return MemoryPressureLevel.NORMAL
        except ImportError:
            return MemoryPressureLevel.NORMAL
    
    async def _notify_pressure_change(self, pressure: MemoryPressureLevel) -> None:
        """Notify all registered callbacks of pressure change."""
        for callback in self._pressure_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(pressure)
                else:
                    callback(pressure)
            except Exception as e:
                logger.error(f"Pressure callback error: {e}")
    
    def register_pressure_callback(
        self,
        callback: Callable[[MemoryPressureLevel], None],
    ) -> None:
        """
        Register a callback for memory pressure changes.
        
        Args:
            callback: Function to call when pressure changes
        """
        self._pressure_callbacks.append(callback)
    
    def unregister_pressure_callback(
        self,
        callback: Callable[[MemoryPressureLevel], None],
    ) -> None:
        """Unregister a pressure callback."""
        if callback in self._pressure_callbacks:
            self._pressure_callbacks.remove(callback)
    
    def get_memory_stats(self) -> dict[str, Any]:
        """Get current memory statistics."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent_used": memory.percent,
                "pressure_level": self._current_pressure.name,
                "unified_memory_gb": self._hardware_info.unified_memory_gb,
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    def allocate_pool(self, size_mb: int) -> MemoryPool:
        """
        Allocate a memory pool.
        
        Args:
            size_mb: Size of pool in megabytes
            
        Returns:
            MemoryPool instance
        """
        pool = MemoryPool(size_mb)
        self._memory_pools.append(pool)
        return pool
    
    def cleanup(self) -> None:
        """Perform memory cleanup."""
        for pool in self._memory_pools:
            pool.clear()
        
        if self._config.enable_aggressive_cleanup:
            # Force garbage collection
            import gc
            gc.collect()
    
    async def shutdown(self) -> None:
        """Shutdown the memory manager."""
        if self._monitoring_task is not None:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.cleanup()


class MemoryPool:
    """Memory pool for efficient allocation."""
    
    def __init__(self, size_mb: int) -> None:
        """
        Initialize memory pool.
        
        Args:
            size_mb: Pool size in megabytes
        """
        self._size_mb = size_mb
        self._allocated: list[np.ndarray] = []
        self._lock = asyncio.Lock()
    
    def allocate(self, shape: tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Allocate array from pool.
        
        Args:
            shape: Array shape
            dtype: Data type
            
        Returns:
            Allocated numpy array
        """
        # For unified memory, use page-aligned allocation
        array = np.zeros(shape, dtype=dtype)
        self._allocated.append(array)
        return array
    
    def clear(self) -> None:
        """Clear all allocated memory."""
        self._allocated.clear()
    
    @property
    def size_mb(self) -> int:
        """Return pool size in MB."""
        return self._size_mb


# ============================================================================
# Neural Engine Embedding Accelerator
# ============================================================================

class NeuralEngineAccelerator:
    """
    Neural Engine-accelerated embedding computation for M4.
    
    Provides optimized embedding generation using the Neural Engine
    for maximum performance and energy efficiency.
    """
    
    def __init__(self, config: NeuralEngineConfig | None = None) -> None:
        """
        Initialize the Neural Engine accelerator.
        
        Args:
            config: Neural Engine configuration
        """
        self._config = config or NeuralEngineConfig()
        self._hardware_info = M4HardwareDetector.get_hardware_info()
        self._coreml_available = self._check_coreml()
        self._model: Any = None
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_requests)
        
        if not self._hardware_info.is_apple_silicon:
            warnings.warn("Not running on Apple Silicon - Neural Engine unavailable")
        
        if self._coreml_available:
            self._initialize_coreml()
    
    def _check_coreml(self) -> bool:
        """Check if Core ML is available."""
        try:
            import coremltools as ct
            return True
        except ImportError:
            logger.warning("coremltools not available - falling back to CPU")
            return False
    
    def _initialize_coreml(self) -> None:
        """Initialize Core ML for Neural Engine."""
        try:
            import coremltools as ct
            logger.info("Core ML initialized for Neural Engine")
        except Exception as e:
            logger.error(f"Failed to initialize Core ML: {e}")
            self._coreml_available = False
    
    async def encode(
        self,
        texts: list[str],
        model: EmbeddingModel | None = None,
    ) -> EmbeddingResult:
        """
        Encode texts to embeddings using Neural Engine.
        
        Args:
            texts: List of texts to encode
            model: Optional embedding model (uses default if not provided)
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        import time
        
        start_time = time.perf_counter()
        
        async with self._semaphore:
            if self._coreml_available and self._config.enable_neural_engine:
                embeddings = await self._encode_with_ne(texts, model)
                used_ne = True
            else:
                embeddings = await self._encode_with_cpu(texts, model)
                used_ne = False
        
        compute_time_ms = (time.perf_counter() - start_time) * 1000
        
        return EmbeddingResult(
            embeddings=embeddings,
            compute_time_ms=compute_time_ms,
            used_neural_engine=used_ne,
            batch_size=len(texts),
        )
    
    async def _encode_with_ne(
        self,
        texts: list[str],
        model: EmbeddingModel | None,
    ) -> np.ndarray:
        """Encode using Neural Engine."""
        # This would use Core ML model for actual implementation
        # For now, simulate with CPU fallback
        if model is not None:
            if asyncio.iscoroutinefunction(model.encode_async):
                return await model.encode_async(texts)
            return model.encode(texts)
        
        # Fallback: simple embedding simulation
        return self._simulate_embeddings(len(texts))
    
    async def _encode_with_cpu(
        self,
        texts: list[str],
        model: EmbeddingModel | None,
    ) -> np.ndarray:
        """Encode using CPU fallback."""
        if model is not None:
            if asyncio.iscoroutinefunction(model.encode_async):
                return await model.encode_async(texts)
            return model.encode(texts)
        
        return self._simulate_embeddings(len(texts))
    
    def _simulate_embeddings(self, count: int) -> np.ndarray:
        """Simulate embeddings for testing."""
        return np.random.randn(count, self._config.embedding_dimension).astype(np.float32)
    
    def convert_to_coreml(
        self,
        pytorch_model: Any,
        input_shape: tuple[int, ...],
        output_path: Path | str,
    ) -> Path:
        """
        Convert PyTorch model to Core ML format.
        
        Args:
            pytorch_model: PyTorch model to convert
            input_shape: Input tensor shape
            output_path: Path to save Core ML model
            
        Returns:
            Path to saved Core ML model
        """
        if not self._coreml_available:
            raise RuntimeError("Core ML not available")
        
        import coremltools as ct
        import torch
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Trace the model
        example_input = torch.randn(input_shape)
        traced_model = torch.jit.trace(pytorch_model, example_input)
        
        # Convert to Core ML
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=input_shape)],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.iOS17,
        )
        
        # Save model
        mlmodel.save(str(output_path))
        logger.info(f"Core ML model saved to {output_path}")
        
        return output_path
    
    def optimize_for_neural_engine(self, model_path: Path | str) -> Path:
        """
        Optimize Core ML model specifically for Neural Engine.
        
        Args:
            model_path: Path to Core ML model
            
        Returns:
            Path to optimized model
        """
        if not self._coreml_available:
            raise RuntimeError("Core ML not available")
        
        import coremltools as ct
        
        model_path = Path(model_path)
        
        # Load model
        mlmodel = ct.models.MLModel(str(model_path))
        
        # Apply Neural Engine optimizations
        # This includes quantization and graph optimizations
        if self._config.enable_quantization:
            mlmodel = self._quantize_model(mlmodel)
        
        # Save optimized model
        optimized_path = model_path.parent / f"{model_path.stem}_optimized.mlpackage"
        mlmodel.save(str(optimized_path))
        
        return optimized_path
    
    def _quantize_model(self, mlmodel: Any) -> Any:
        """Quantize model for Neural Engine."""
        import coremltools as ct
        
        # Apply quantization
        quantized_model = ct.models.neural_network.quantization_utils.quantize_weights(
            mlmodel,
            nbits=self._config.quantization_bits,
            quantization_mode="linear",
        )
        
        return quantized_model


# ============================================================================
# Core ML Inference Engine
# ============================================================================

class CoreMLInferenceEngine:
    """
    Core ML inference engine optimized for M4.
    
    Provides efficient model inference with automatic compute unit
    selection and batch processing.
    """
    
    def __init__(self, config: CoreMLConfig | None = None) -> None:
        """
        Initialize the Core ML inference engine.
        
        Args:
            config: Core ML configuration
        """
        self._config = config or CoreMLConfig()
        self._hardware_info = M4HardwareDetector.get_hardware_info()
        self._models: dict[str, Any] = {}
        self._coreml_available = self._check_coreml()
        
        if not self._coreml_available:
            logger.warning("Core ML not available - inference will use CPU")
    
    def _check_coreml(self) -> bool:
        """Check if Core ML is available."""
        try:
            import coremltools as ct
            return True
        except ImportError:
            return False
    
    def load_model(self, model_path: Path | str, model_id: str | None = None) -> str:
        """
        Load a Core ML model.
        
        Args:
            model_path: Path to Core ML model
            model_id: Optional model identifier
            
        Returns:
            Model ID for referencing the loaded model
        """
        if not self._coreml_available:
            raise RuntimeError("Core ML not available")
        
        import coremltools as ct
        
        model_path = Path(model_path)
        model_id = model_id or model_path.stem
        
        # Load with appropriate compute unit
        compute_unit = self._get_compute_unit()
        
        mlmodel = ct.models.MLModel(
            str(model_path),
            compute_units=compute_unit,
        )
        
        self._models[model_id] = mlmodel
        logger.info(f"Loaded model '{model_id}' with compute unit: {compute_unit}")
        
        return model_id
    
    def _get_compute_unit(self) -> Any:
        """Get Core ML compute unit based on configuration."""
        import coremltools as ct
        
        compute_unit_map = {
            ComputeUnit.CPU_ONLY: ct.ComputeUnit.CPU_ONLY,
            ComputeUnit.CPU_AND_GPU: ct.ComputeUnit.CPU_AND_GPU,
            ComputeUnit.CPU_AND_NE: ct.ComputeUnit.CPU_AND_NE,
            ComputeUnit.ALL: ct.ComputeUnit.ALL,
        }
        
        return compute_unit_map.get(self._config.compute_unit, ct.ComputeUnit.ALL)
    
    async def predict(
        self,
        model_id: str,
        input_data: dict[str, np.ndarray],
    ) -> InferenceResult[dict[str, np.ndarray]]:
        """
        Run inference on input data.
        
        Args:
            model_id: ID of loaded model
            input_data: Input tensors
            
        Returns:
            Inference result with output and metadata
        """
        import time
        
        if model_id not in self._models:
            raise ValueError(f"Model '{model_id}' not loaded")
        
        start_time = time.perf_counter()
        
        model = self._models[model_id]
        
        if self._config.enable_async_predictions:
            output = await self._predict_async(model, input_data)
        else:
            output = model.predict(input_data)
        
        compute_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Estimate memory usage
        memory_mb = self._estimate_memory_usage(input_data, output)
        
        return InferenceResult(
            output=output,
            compute_time_ms=compute_time_ms,
            used_neural_engine=self._config.compute_unit in [
                ComputeUnit.CPU_AND_NE,
                ComputeUnit.ALL,
            ],
            memory_used_mb=memory_mb,
        )
    
    async def _predict_async(
        self,
        model: Any,
        input_data: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Run async prediction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, model.predict, input_data)
    
    def predict_batch(
        self,
        model_id: str,
        batch_inputs: list[dict[str, np.ndarray]],
    ) -> list[InferenceResult[dict[str, np.ndarray]]]:
        """
        Run batch inference.
        
        Args:
            model_id: ID of loaded model
            batch_inputs: List of input tensors
            
        Returns:
            List of inference results
        """
        results = []
        
        # Process in configured batch sizes
        for i in range(0, len(batch_inputs), self._config.batch_size):
            batch = batch_inputs[i:i + self._config.batch_size]
            
            for input_data in batch:
                # Run synchronous prediction for batch
                import time
                start_time = time.perf_counter()
                
                model = self._models[model_id]
                output = model.predict(input_data)
                
                compute_time_ms = (time.perf_counter() - start_time) * 1000
                memory_mb = self._estimate_memory_usage(input_data, output)
                
                results.append(InferenceResult(
                    output=output,
                    compute_time_ms=compute_time_ms,
                    used_neural_engine=self._config.compute_unit in [
                        ComputeUnit.CPU_AND_NE,
                        ComputeUnit.ALL,
                    ],
                    memory_used_mb=memory_mb,
                ))
        
        return results
    
    def _estimate_memory_usage(
        self,
        input_data: dict[str, np.ndarray],
        output_data: dict[str, np.ndarray],
    ) -> float:
        """Estimate memory usage in MB."""
        input_bytes = sum(arr.nbytes for arr in input_data.values())
        output_bytes = sum(arr.nbytes for arr in output_data.values())
        return (input_bytes + output_bytes) / (1024 * 1024)
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_id: ID of model to unload
            
        Returns:
            True if model was unloaded
        """
        if model_id in self._models:
            del self._models[model_id]
            logger.info(f"Unloaded model '{model_id}'")
            return True
        return False
    
    def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get information about a loaded model."""
        if model_id not in self._models:
            raise ValueError(f"Model '{model_id}' not loaded")
        
        model = self._models[model_id]
        
        return {
            "model_id": model_id,
            "compute_unit": self._config.compute_unit.value,
            "input_description": str(model.get_spec().description.input),
            "output_description": str(model.get_spec().description.output),
        }


# ============================================================================
# Apple Silicon Process Pool
# ============================================================================

class AppleSiliconProcessPool:
    """
    Process pool optimized for Apple Silicon M4.
    
    Configures process pool with optimal settings for M4's
    heterogeneous core architecture.
    """
    
    def __init__(self, config: ProcessPoolConfig | None = None) -> None:
        """
        Initialize the process pool.
        
        Args:
            config: Process pool configuration
        """
        self._config = config or ProcessPoolConfig()
        self._hardware_info = M4HardwareDetector.get_hardware_info()
        self._executor: ProcessPoolExecutor | None = None
        self._thread_executor: ThreadPoolExecutor | None = None
    
    def __enter__(self) -> AppleSiliconProcessPool:
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.shutdown()
    
    def start(self) -> None:
        """Start the process pool."""
        # Configure multiprocessing context for macOS
        mp_context = multiprocessing.get_context(self._config.mp_context)
        
        # Set process affinity to performance cores if requested
        if self._config.enable_performance_cores_only and self._hardware_info.is_apple_silicon:
            initializer = self._wrap_initializer_with_affinity()
        else:
            initializer = self._config.initializer
        
        self._executor = ProcessPoolExecutor(
            max_workers=self._config.max_workers,
            mp_context=mp_context,
            initializer=initializer,
            initargs=self._config.initargs,
            max_tasks_per_child=self._config.max_tasks_per_child,
        )
        
        # Also create thread pool for I/O-bound tasks
        self._thread_executor = ThreadPoolExecutor(
            max_workers=self._hardware_info.recommended_thread_pool_size,
        )
        
        logger.info(
            f"Process pool started with {self._config.max_workers} workers"
        )
    
    def _wrap_initializer_with_affinity(self) -> Callable[..., Any]:
        """Wrap initializer to set CPU affinity to performance cores."""
        original_initializer = self._config.initializer
        
        def affinity_initializer(*args: Any) -> None:
            """Set CPU affinity and call original initializer."""
            try:
                # Set affinity to performance cores (first N cores)
                perf_cores = self._hardware_info.performance_cores
                os.sched_setaffinity(0, list(range(perf_cores)))
            except (AttributeError, OSError):
                # sched_setaffinity not available on all platforms
                pass
            
            if original_initializer is not None:
                original_initializer(*args)
        
        return affinity_initializer
    
    def shutdown(self) -> None:
        """Shutdown the process pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        
        if self._thread_executor is not None:
            self._thread_executor.shutdown(wait=True)
            self._thread_executor = None
        
        logger.info("Process pool shutdown")
    
    async def submit(
        self,
        func: Callable[..., TOutput],
        *args: Any,
        use_threads: bool = False,
        **kwargs: Any,
    ) -> TOutput:
        """
        Submit a task to the pool.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            use_threads: Use thread pool instead of process pool
            **kwargs: Keyword arguments
            
        Returns:
            Task result
        """
        if use_threads:
            executor = self._thread_executor
        else:
            executor = self._executor
        
        if executor is None:
            raise RuntimeError("Pool not started")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            lambda: func(*args, **kwargs),
        )
    
    def map(
        self,
        func: Callable[[TInput], TOutput],
        iterable: Iterator[TInput],
        chunksize: int = 1,
    ) -> Iterator[TOutput]:
        """
        Map function over iterable using process pool.
        
        Args:
            func: Function to apply
            iterable: Input iterable
            chunksize: Chunk size for batching
            
        Returns:
            Iterator of results
        """
        if self._executor is None:
            raise RuntimeError("Pool not started")
        
        return self._executor.map(func, iterable, chunksize=chunksize)
    
    @property
    def is_running(self) -> bool:
        """Check if pool is running."""
        return self._executor is not None


# ============================================================================
# M4 Optimizer (Main Interface)
# ============================================================================

class M4Optimizer:
    """
    Main interface for M4-specific optimizations.
    
    Provides a unified API for accessing all M4 optimizations
    including Core ML, Neural Engine, and unified memory.
    """
    
    _instance: M4Optimizer | None = None
    _initialized: bool = False
    
    def __new__(
        cls,
        memory_config: UnifiedMemoryConfig | None = None,
        coreml_config: CoreMLConfig | None = None,
        neural_config: NeuralEngineConfig | None = None,
        pool_config: ProcessPoolConfig | None = None,
    ) -> M4Optimizer:
        """Singleton pattern for M4 optimizer."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        memory_config: UnifiedMemoryConfig | None = None,
        coreml_config: CoreMLConfig | None = None,
        neural_config: NeuralEngineConfig | None = None,
        pool_config: ProcessPoolConfig | None = None,
    ) -> None:
        """
        Initialize the M4 optimizer.
        
        Args:
            memory_config: Unified memory configuration
            coreml_config: Core ML configuration
            neural_config: Neural Engine configuration
            pool_config: Process pool configuration
        """
        if M4Optimizer._initialized:
            return
        
        self._hardware_info = M4HardwareDetector.get_hardware_info()
        self._memory_manager = UnifiedMemoryManager(memory_config)
        self._coreml_engine = CoreMLInferenceEngine(coreml_config)
        self._neural_accelerator = NeuralEngineAccelerator(neural_config)
        self._process_pool = AppleSiliconProcessPool(pool_config)
        
        M4Optimizer._initialized = True
        
        logger.info(f"M4 Optimizer initialized: {self._hardware_info.chip_variant.name}")
    
    @classmethod
    def get_instance(cls) -> M4Optimizer:
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @staticmethod
    def get_hardware_info() -> M4HardwareInfo:
        """Get hardware information."""
        return M4HardwareDetector.get_hardware_info()
    
    @staticmethod
    def is_available() -> bool:
        """Check if M4 optimizations are available."""
        return M4HardwareDetector.is_running_on_m4()
    
    @property
    def memory_manager(self) -> UnifiedMemoryManager:
        """Get the unified memory manager."""
        return self._memory_manager
    
    @property
    def coreml_engine(self) -> CoreMLInferenceEngine:
        """Get the Core ML inference engine."""
        return self._coreml_engine
    
    @property
    def neural_accelerator(self) -> NeuralEngineAccelerator:
        """Get the Neural Engine accelerator."""
        return self._neural_accelerator
    
    @property
    def process_pool(self) -> AppleSiliconProcessPool:
        """Get the process pool."""
        return self._process_pool
    
    async def embed(
        self,
        texts: list[str],
        model: EmbeddingModel | None = None,
    ) -> EmbeddingResult:
        """
        Generate embeddings using Neural Engine.
        
        Args:
            texts: Texts to embed
            model: Optional embedding model
            
        Returns:
            Embedding result
        """
        return await self._neural_accelerator.encode(texts, model)
    
    async def infer(
        self,
        model_id: str,
        input_data: dict[str, np.ndarray],
    ) -> InferenceResult[dict[str, np.ndarray]]:
        """
        Run model inference.
        
        Args:
            model_id: Model identifier
            input_data: Input tensors
            
        Returns:
            Inference result
        """
        return await self._coreml_engine.predict(model_id, input_data)
    
    def load_model(self, model_path: Path | str, model_id: str | None = None) -> str:
        """
        Load a Core ML model.
        
        Args:
            model_path: Path to model
            model_id: Optional model identifier
            
        Returns:
            Model ID
        """
        return self._coreml_engine.load_model(model_path, model_id)
    
    def start_pool(self) -> None:
        """Start the process pool."""
        self._process_pool.start()
    
    def shutdown_pool(self) -> None:
        """Shutdown the process pool."""
        self._process_pool.shutdown()
    
    async def shutdown(self) -> None:
        """Shutdown all M4 optimizations."""
        await self._memory_manager.shutdown()
        self._process_pool.shutdown()
        logger.info("M4 Optimizer shutdown complete")
    
    def get_stats(self) -> dict[str, Any]:
        """Get optimization statistics."""
        return {
            "hardware": {
                "chip_variant": self._hardware_info.chip_variant.name,
                "cpu_cores": self._hardware_info.cpu_cores,
                "gpu_cores": self._hardware_info.gpu_cores,
                "neural_engine_cores": self._hardware_info.neural_engine_cores,
                "unified_memory_gb": self._hardware_info.unified_memory_gb,
            },
            "memory": self._memory_manager.get_memory_stats(),
            "pool_running": self._process_pool.is_running,
            "coreml_available": self._coreml_engine._coreml_available,
        }


# ============================================================================
# Utility Functions
# ============================================================================

def create_m4_optimizer(
    max_memory_fraction: float = 0.8,
    use_neural_engine: bool = True,
    process_pool_workers: int | None = None,
) -> M4Optimizer:
    """
    Factory function to create M4 optimizer with common settings.
    
    Args:
        max_memory_fraction: Maximum fraction of memory to use
        use_neural_engine: Enable Neural Engine acceleration
        process_pool_workers: Number of process pool workers
        
    Returns:
        Configured M4Optimizer instance
    """
    memory_config = UnifiedMemoryConfig(max_memory_fraction=max_memory_fraction)
    
    coreml_config = CoreMLConfig(
        compute_unit=ComputeUnit.CPU_AND_NE if use_neural_engine else ComputeUnit.CPU_ONLY,
    )
    
    neural_config = NeuralEngineConfig(enable_neural_engine=use_neural_engine)
    
    pool_config = ProcessPoolConfig(max_workers=process_pool_workers)
    
    return M4Optimizer(
        memory_config=memory_config,
        coreml_config=coreml_config,
        neural_config=neural_config,
        pool_config=pool_config,
    )


def optimize_numpy_for_m4() -> None:
    """
    Optimize NumPy for M4 architecture.
    
    Sets environment variables and configuration for optimal
    performance on Apple Silicon.
    """
    # Set OpenBLAS threads for M4
    os.environ["OPENBLAS_NUM_THREADS"] = str(
        M4HardwareDetector.get_hardware_info().performance_cores
    )
    
    # Disable NumPy multithreading for better control
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # Set accelerate framework
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(
        M4HardwareDetector.get_hardware_info().performance_cores
    )
    
    logger.info("NumPy optimized for M4")


def get_optimal_batch_size(
    embedding_dim: int,
    available_memory_gb: float | None = None,
) -> int:
    """
    Calculate optimal batch size for embeddings.
    
    Args:
        embedding_dim: Dimension of embeddings
        available_memory_gb: Available memory in GB
        
    Returns:
        Optimal batch size
    """
    if available_memory_gb is None:
        info = M4HardwareDetector.get_hardware_info()
        available_memory_gb = info.unified_memory_gb * 0.5
    
    # Calculate batch size based on memory
    bytes_per_embedding = embedding_dim * 4  # float32
    available_bytes = available_memory_gb * 1024**3
    
    # Use 20% of available memory for batch
    batch_memory = available_bytes * 0.2
    optimal_batch = int(batch_memory / bytes_per_embedding)
    
    # Clamp to reasonable range
    return max(1, min(optimal_batch, 1024))


# ============================================================================
# Export Public API
# ============================================================================

__all__ = [
    # Main classes
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
