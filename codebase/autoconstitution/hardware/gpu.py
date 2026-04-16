"""
autoconstitution GPU/CUDA Optimization Module

Provides comprehensive GPU acceleration support including:
- Multi-GPU support with automatic device selection
- Distributed inference across multiple GPUs
- VRAM management and memory optimization
- Gradient checkpointing for training runs
- CUDA stream management for async operations
- Mixed precision training support

Example:
    >>> from autoconstitution.hardware.gpu import GPUManager, DistributedInference
    >>> 
    >>> # Initialize GPU manager
    >>> gpu_mgr = GPUManager()
    >>> print(f"Available GPUs: {gpu_mgr.num_gpus}")
    >>> 
    >>> # Distributed inference
    >>> dist_inf = DistributedInference(gpu_mgr)
    >>> results = dist_inf.run_parallel(model, inputs)
    >>> 
    >>> # VRAM management
    >>> vram_mgr = VRAMManager(gpu_mgr)
    >>> vram_mgr.optimize_memory()
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic types
T = TypeVar("T")
ModelType = TypeVar("ModelType")
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class GPUPrecision(Enum):
    """GPU computation precision modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"


class MemoryStrategy(Enum):
    """Memory management strategies."""
    CONSERVATIVE = auto()  # Keep high memory reserve
    BALANCED = auto()      # Balanced performance/memory
    AGGRESSIVE = auto()    # Maximize performance, minimize reserve
    ADAPTIVE = auto()      # Dynamically adjust based on workload


class CheckpointStrategy(Enum):
    """Gradient checkpointing strategies."""
    NONE = auto()          # No checkpointing
    SELECTIVE = auto()     # Checkpoint selective layers
    FULL = auto()          # Checkpoint all activations
    AUTO = auto()          # Automatically determine based on memory


@dataclass
class GPUStats:
    """Real-time GPU statistics."""
    device_id: int
    name: str
    total_memory_gb: float
    allocated_memory_gb: float
    free_memory_gb: float
    utilization_percent: float
    temperature_celsius: Optional[float] = None
    power_draw_watts: Optional[float] = None
    clock_speed_mhz: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    
    @property
    def memory_utilization_percent(self) -> float:
        """Calculate memory utilization percentage."""
        if self.total_memory_gb > 0:
            return (self.allocated_memory_gb / self.total_memory_gb) * 100
        return 0.0
    
    @property
    def is_under_pressure(self) -> bool:
        """Check if GPU is under memory pressure."""
        return self.memory_utilization_percent > 90 or self.free_memory_gb < 1.0


@dataclass
class MemoryAllocation:
    """Memory allocation tracking."""
    tensor_id: str
    device_id: int
    size_bytes: int
    allocated_at: float = field(default_factory=time.time)
    
    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)
    
    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 * 1024 * 1024)
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.allocated_at


@dataclass
class CheckpointConfig:
    """Configuration for gradient checkpointing."""
    strategy: CheckpointStrategy = CheckpointStrategy.AUTO
    checkpoint_every_n_layers: int = 2
    preserve_rng_state: bool = True
    use_reentrant: bool = True
    offload_to_cpu: bool = False
    
    def should_checkpoint_layer(self, layer_idx: int, total_layers: int) -> bool:
        """Determine if a layer should be checkpointed."""
        if self.strategy == CheckpointStrategy.NONE:
            return False
        if self.strategy == CheckpointStrategy.FULL:
            return True
        if self.strategy == CheckpointStrategy.SELECTIVE:
            # Checkpoint every N layers, and always checkpoint early layers
            return layer_idx % self.checkpoint_every_n_layers == 0 or layer_idx < 2
        # AUTO: decide based on available memory
        return layer_idx % self.checkpoint_every_n_layers == 0


@dataclass
class DistributedConfig:
    """Configuration for distributed inference."""
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"  # nccl, gloo, mpi
    init_method: str = "env://"
    timeout_seconds: float = 1800.0
    
    def is_main_process(self) -> bool:
        return self.rank == 0


class CUDAStreamPool:
    """Pool of CUDA streams for asynchronous operations."""
    
    def __init__(self, num_streams: int = 4, device_id: int = 0):
        self.device_id = device_id
        self.num_streams = num_streams
        self._streams: List[Any] = []
        self._current_idx = 0
        self._lock = threading.Lock()
        self._initialize_streams()
    
    def _initialize_streams(self) -> None:
        """Initialize CUDA streams."""
        try:
            import torch
            with torch.cuda.device(self.device_id):
                for _ in range(self.num_streams):
                    stream = torch.cuda.Stream(device=self.device_id)
                    self._streams.append(stream)
        except ImportError:
            logger.warning("PyTorch not available, CUDA streams disabled")
    
    def get_stream(self) -> Any:
        """Get next available stream in round-robin fashion."""
        with self._lock:
            stream = self._streams[self._current_idx]
            self._current_idx = (self._current_idx + 1) % self.num_streams
            return stream
    
    def synchronize_all(self) -> None:
        """Synchronize all streams."""
        try:
            import torch
            for stream in self._streams:
                stream.synchronize()
        except ImportError:
            pass
    
    @contextmanager
    def stream_context(self, stream_idx: Optional[int] = None) -> Iterator[Any]:
        """Context manager for stream operations."""
        try:
            import torch
            stream = (self._streams[stream_idx] if stream_idx is not None 
                     else self.get_stream())
            with torch.cuda.stream(stream):
                yield stream
        except ImportError:
            yield None


class GPUManager:
    """
    Manages multiple GPUs for distributed computation.
    
    Provides:
    - Multi-GPU detection and selection
    - Load balancing across GPUs
    - Memory monitoring and management
    - Automatic device selection
    
    Example:
        >>> gpu_mgr = GPUManager()
        >>> device = gpu_mgr.get_optimal_device()
        >>> gpu_mgr.set_device(device)
    """
    
    def __init__(
        self,
        preferred_device_ids: Optional[List[int]] = None,
        memory_fraction: float = 0.95,
        allow_growth: bool = True
    ):
        """
        Initialize GPU manager.
        
        Args:
            preferred_device_ids: List of preferred GPU device IDs
            memory_fraction: Maximum fraction of GPU memory to use
            allow_growth: Allow memory to grow dynamically
        """
        self.preferred_device_ids = preferred_device_ids
        self.memory_fraction = memory_fraction
        self.allow_growth = allow_growth
        self._device_ids: List[int] = []
        self._gpu_info: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._stream_pools: Dict[int, CUDAStreamPool] = {}
        
        self._detect_gpus()
        self._configure_memory()
    
    def _detect_gpus(self) -> None:
        """Detect available CUDA GPUs."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                logger.warning("CUDA not available")
                return
            
            num_gpus = torch.cuda.device_count()
            logger.info(f"Detected {num_gpus} CUDA GPU(s)")
            
            for device_id in range(num_gpus):
                if (self.preferred_device_ids is not None and 
                    device_id not in self.preferred_device_ids):
                    continue
                
                props = torch.cuda.get_device_properties(device_id)
                
                self._gpu_info[device_id] = {
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "multi_processor_count": props.multi_processor_count,
                    "major": props.major,
                    "minor": props.minor,
                }
                self._device_ids.append(device_id)
                
                # Initialize stream pool for this device
                self._stream_pools[device_id] = CUDAStreamPool(
                    num_streams=4, device_id=device_id
                )
                
                logger.info(
                    f"GPU {device_id}: {props.name}, "
                    f"{props.total_memory / 1e9:.1f} GB"
                )
                
        except ImportError:
            logger.warning("PyTorch not available for GPU detection")
    
    def _configure_memory(self) -> None:
        """Configure GPU memory settings."""
        try:
            import torch
            
            for device_id in self._device_ids:
                torch.cuda.set_per_process_memory_fraction(
                    self.memory_fraction, device_id
                )
                
                if self.allow_growth:
                    # PyTorch doesn't have a direct equivalent to TF's allow_growth
                    # but we can set the allocator config
                    torch.cuda.memory.set_per_process_memory_fraction(
                        self.memory_fraction, device_id
                    )
                    
        except ImportError:
            pass
    
    @property
    def num_gpus(self) -> int:
        """Get number of available GPUs."""
        return len(self._device_ids)
    
    @property
    def device_ids(self) -> List[int]:
        """Get list of available device IDs."""
        return self._device_ids.copy()
    
    @property
    def has_gpu(self) -> bool:
        """Check if any GPU is available."""
        return self.num_gpus > 0
    
    def get_gpu_info(self, device_id: int) -> Dict[str, Any]:
        """Get information about a specific GPU."""
        if device_id not in self._gpu_info:
            raise ValueError(f"Invalid device ID: {device_id}")
        return self._gpu_info[device_id].copy()
    
    def get_stats(self, device_id: Optional[int] = None) -> Union[GPUStats, List[GPUStats]]:
        """Get GPU statistics."""
        try:
            import torch
            import pynvml
            
            pynvml.nvmlInit()
            
            def _get_single_stats(dev_id: int) -> GPUStats:
                handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
                
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_gb = mem_info.total / (1024 ** 3)
                allocated_gb = mem_info.used / (1024 ** 3)
                free_gb = mem_info.free / (1024 ** 3)
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except:
                    temp = None
                
                # Power
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except:
                    power = None
                
                # Clock speed
                try:
                    clock = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_SM
                    )
                except:
                    clock = None
                
                return GPUStats(
                    device_id=dev_id,
                    name=self._gpu_info.get(dev_id, {}).get("name", "Unknown"),
                    total_memory_gb=total_gb,
                    allocated_memory_gb=allocated_gb,
                    free_memory_gb=free_gb,
                    utilization_percent=util.gpu,
                    temperature_celsius=temp,
                    power_draw_watts=power,
                    clock_speed_mhz=clock
                )
            
            if device_id is not None:
                return _get_single_stats(device_id)
            else:
                return [_get_single_stats(did) for did in self._device_ids]
                
        except ImportError:
            logger.warning("pynvml not available for GPU stats")
            return GPUStats(
                device_id=device_id or 0,
                name="Unknown",
                total_memory_gb=0,
                allocated_memory_gb=0,
                free_memory_gb=0,
                utilization_percent=0
            )
    
    def get_optimal_device(self) -> int:
        """Get the device with most free memory."""
        with self._lock:
            if not self._device_ids:
                raise RuntimeError("No GPUs available")
            
            if len(self._device_ids) == 1:
                return self._device_ids[0]
            
            try:
                stats_list = self.get_stats()
                if isinstance(stats_list, list):
                    # Find GPU with most free memory
                    best_gpu = max(stats_list, key=lambda s: s.free_memory_gb)
                    return best_gpu.device_id
                return self._device_ids[0]
            except Exception as e:
                logger.warning(f"Could not determine optimal device: {e}")
                return self._device_ids[0]
    
    def set_device(self, device_id: int) -> None:
        """Set the current CUDA device."""
        try:
            import torch
            if device_id in self._device_ids:
                torch.cuda.set_device(device_id)
            else:
                raise ValueError(f"Invalid device ID: {device_id}")
        except ImportError:
            pass
    
    def get_stream_pool(self, device_id: int) -> CUDAStreamPool:
        """Get stream pool for a device."""
        if device_id not in self._stream_pools:
            raise ValueError(f"No stream pool for device {device_id}")
        return self._stream_pools[device_id]
    
    def synchronize(self, device_id: Optional[int] = None) -> None:
        """Synchronize CUDA operations."""
        try:
            import torch
            if device_id is not None:
                with torch.cuda.device(device_id):
                    torch.cuda.synchronize()
            else:
                torch.cuda.synchronize()
        except ImportError:
            pass
    
    def empty_cache(self) -> None:
        """Empty CUDA cache on all devices."""
        try:
            import torch
            torch.cuda.empty_cache()
            gc.collect()
        except ImportError:
            pass
    
    def memory_summary(self, device_id: Optional[int] = None) -> str:
        """Get memory summary for device(s)."""
        try:
            import torch
            
            if device_id is not None:
                return torch.cuda.memory_summary(device=device_id)
            else:
                summaries = []
                for did in self._device_ids:
                    summaries.append(f"\n=== GPU {did} ===")
                    summaries.append(torch.cuda.memory_summary(device=did))
                return "\n".join(summaries)
        except ImportError:
            return "PyTorch not available"


class VRAMManager:
    """
    Manages GPU VRAM for optimal performance.
    
    Features:
    - Memory allocation tracking
    - Automatic memory optimization
    - Memory pressure detection
    - Garbage collection triggering
    
    Example:
        >>> vram_mgr = VRAMManager(gpu_manager)
        >>> vram_mgr.optimize_memory()
        >>> with vram_mgr.memory_scope("inference"):
        ...     result = model(input)
    """
    
    def __init__(
        self,
        gpu_manager: GPUManager,
        strategy: MemoryStrategy = MemoryStrategy.ADAPTIVE,
        reserve_gb: float = 2.0
    ):
        """
        Initialize VRAM manager.
        
        Args:
            gpu_manager: GPUManager instance
            strategy: Memory management strategy
            reserve_gb: Memory to reserve (in GB)
        """
        self.gpu_manager = gpu_manager
        self.strategy = strategy
        self.reserve_gb = reserve_gb
        self._allocations: Dict[str, MemoryAllocation] = {}
        self._allocation_history: deque = deque(maxlen=1000)
        self._lock = threading.RLock()
        self._peak_memory: Dict[int, float] = {}
    
    def get_available_memory(self, device_id: int) -> float:
        """Get available memory on device in GB."""
        stats = self.gpu_manager.get_stats(device_id)
        if isinstance(stats, GPUStats):
            return max(0, stats.free_memory_gb - self.reserve_gb)
        return 0.0
    
    def can_allocate(self, size_gb: float, device_id: int) -> bool:
        """Check if allocation can be made."""
        available = self.get_available_memory(device_id)
        return available >= size_gb
    
    def track_allocation(
        self,
        tensor_id: str,
        size_bytes: int,
        device_id: int
    ) -> MemoryAllocation:
        """Track a memory allocation."""
        with self._lock:
            allocation = MemoryAllocation(
                tensor_id=tensor_id,
                device_id=device_id,
                size_bytes=size_bytes
            )
            self._allocations[tensor_id] = allocation
            self._allocation_history.append(allocation)
            return allocation
    
    def untrack_allocation(self, tensor_id: str) -> None:
        """Remove allocation tracking."""
        with self._lock:
            if tensor_id in self._allocations:
                del self._allocations[tensor_id]
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get allocation statistics."""
        with self._lock:
            total_allocated = sum(
                a.size_bytes for a in self._allocations.values()
            )
            by_device: Dict[int, int] = {}
            for alloc in self._allocations.values():
                by_device[alloc.device_id] = (
                    by_device.get(alloc.device_id, 0) + alloc.size_bytes
                )
            
            return {
                "total_allocations": len(self._allocations),
                "total_allocated_gb": total_allocated / (1024 ** 3),
                "by_device_gb": {
                    did: size / (1024 ** 3) 
                    for did, size in by_device.items()
                },
                "history_size": len(self._allocation_history)
            }
    
    def optimize_memory(self, aggressive: bool = False) -> None:
        """Optimize GPU memory usage."""
        logger.info("Optimizing GPU memory...")
        
        # Clear PyTorch cache
        self.gpu_manager.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        if aggressive:
            # More aggressive cleanup
            try:
                import torch
                for device_id in self.gpu_manager.device_ids:
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
                        # Reset peak memory stats
                        torch.cuda.reset_peak_memory_stats(device_id)
            except ImportError:
                pass
        
        logger.info("Memory optimization complete")
    
    def check_memory_pressure(self, device_id: int) -> Tuple[bool, float]:
        """
        Check if device is under memory pressure.
        
        Returns:
            Tuple of (is_under_pressure, available_gb)
        """
        stats = self.gpu_manager.get_stats(device_id)
        if isinstance(stats, GPUStats):
            available = stats.free_memory_gb - self.reserve_gb
            is_pressure = (
                stats.memory_utilization_percent > 85 or 
                available < self.reserve_gb
            )
            return is_pressure, available
        return False, 0.0
    
    def get_memory_recommendations(self) -> List[str]:
        """Get recommendations for memory optimization."""
        recommendations = []
        
        for device_id in self.gpu_manager.device_ids:
            is_pressure, available = self.check_memory_pressure(device_id)
            if is_pressure:
                stats = self.gpu_manager.get_stats(device_id)
                if isinstance(stats, GPUStats):
                    recommendations.append(
                        f"GPU {device_id}: High memory usage "
                        f"({stats.memory_utilization_percent:.1f}%). "
                        f"Consider: reducing batch size, enabling gradient "
                        f"checkpointing, or using mixed precision."
                    )
        
        if not recommendations:
            recommendations.append("Memory usage is within normal parameters")
        
        return recommendations
    
    @contextmanager
    def memory_scope(self, scope_name: str) -> Iterator[None]:
        """Context manager for memory-scoped operations."""
        try:
            logger.debug(f"Entering memory scope: {scope_name}")
            yield
        finally:
            logger.debug(f"Exiting memory scope: {scope_name}")
            # Optional: auto-cleanup based on strategy
            if self.strategy == MemoryStrategy.AGGRESSIVE:
                self.optimize_memory(aggressive=False)


class GradientCheckpointManager:
    """
    Manages gradient checkpointing for training.
    
    Reduces memory usage during training by trading computation for memory.
    
    Example:
        >>> checkpoint_mgr = GradientCheckpointManager(
        ...     config=CheckpointConfig(strategy=CheckpointStrategy.SELECTIVE)
        ... )
        >>> model = checkpoint_mgr.wrap_model(model)
        >>> output = model(input)  # Uses checkpointing
    """
    
    def __init__(
        self,
        config: Optional[CheckpointConfig] = None,
        vram_manager: Optional[VRAMManager] = None
    ):
        """
        Initialize gradient checkpoint manager.
        
        Args:
            config: Checkpoint configuration
            vram_manager: Optional VRAM manager for auto strategy
        """
        self.config = config or CheckpointConfig()
        self.vram_manager = vram_manager
        self._checkpointed_modules: Set[str] = set()
        self._memory_saved = 0.0
    
    def should_use_checkpointing(self, model_memory_gb: float) -> bool:
        """Determine if checkpointing should be used."""
        if self.config.strategy == CheckpointStrategy.NONE:
            return False
        if self.config.strategy in (CheckpointStrategy.FULL, CheckpointStrategy.SELECTIVE):
            return True
        
        # AUTO: decide based on available memory
        if self.vram_manager is not None and self.vram_manager.gpu_manager.has_gpu:
            device_id = self.vram_manager.gpu_manager.get_optimal_device()
            available = self.vram_manager.get_available_memory(device_id)
            # Use checkpointing if model would use more than 50% of available
            return model_memory_gb > available * 0.5
        
        return False
    
    def wrap_module(
        self,
        module: Any,
        layer_idx: int = 0,
        total_layers: int = 1
    ) -> Any:
        """Wrap a module with gradient checkpointing."""
        try:
            import torch
            from torch.utils.checkpoint import checkpoint
            
            if not self.config.should_checkpoint_layer(layer_idx, total_layers):
                return module
            
            class CheckpointWrapper(torch.nn.Module):
                def __init__(self, wrapped_module: torch.nn.Module, config: CheckpointConfig):
                    super().__init__()
                    self.module = wrapped_module
                    self.config = config
                    self._module_id = id(wrapped_module)
                
                def forward(self, *args, **kwargs):
                    if self.training:
                        return checkpoint(
                            self.module,
                            *args,
                            use_reentrant=self.config.use_reentrant,
                            preserve_rng_state=self.config.preserve_rng_state,
                            **kwargs
                        )
                    else:
                        return self.module(*args, **kwargs)
            
            wrapped = CheckpointWrapper(module, self.config)
            self._checkpointed_modules.add(str(id(module)))
            
            return wrapped
            
        except ImportError:
            logger.warning("PyTorch not available for checkpointing")
            return module
    
    def wrap_model(self, model: Any, layer_pattern: str = "layer") -> Any:
        """Wrap model layers with gradient checkpointing."""
        try:
            import torch
            
            if not isinstance(model, torch.nn.Module):
                return model
            
            # Find layers matching pattern
            layers = []
            for name, module in model.named_modules():
                if layer_pattern in name.lower() and isinstance(
                    module, torch.nn.Module
                ):
                    layers.append((name, module))
            
            total_layers = len(layers)
            
            # Wrap selected layers
            for idx, (name, module) in enumerate(layers):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                wrapped = self.wrap_module(module, idx, total_layers)
                setattr(parent, child_name, wrapped)
            
            logger.info(
                f"Wrapped {len(self._checkpointed_modules)}/{total_layers} "
                f"layers with gradient checkpointing"
            )
            
            return model
            
        except ImportError:
            return model
    
    def get_stats(self) -> Dict[str, Any]:
        """Get checkpointing statistics."""
        return {
            "checkpointed_modules": len(self._checkpointed_modules),
            "strategy": self.config.strategy.name,
            "checkpoint_every_n": self.config.checkpoint_every_n_layers,
            "memory_saved_gb": self._memory_saved
        }


class DistributedInference:
    """
    Distributed inference across multiple GPUs.
    
    Distributes inference workload across available GPUs for
    improved throughput and reduced latency.
    
    Example:
        >>> dist_inf = DistributedInference(gpu_manager)
        >>> results = dist_inf.run_parallel(model, inputs, batch_size=4)
    """
    
    def __init__(
        self,
        gpu_manager: GPUManager,
        max_workers: Optional[int] = None,
        load_balancing: bool = True
    ):
        """
        Initialize distributed inference.
        
        Args:
            gpu_manager: GPUManager instance
            max_workers: Maximum number of worker threads
            load_balancing: Enable dynamic load balancing
        """
        self.gpu_manager = gpu_manager
        self.max_workers = max_workers or gpu_manager.num_gpus
        self.load_balancing = load_balancing
        self._executor: Optional[ThreadPoolExecutor] = None
        self._device_queue: deque = deque()
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "total_inferences": 0,
            "total_time": 0.0,
            "by_device": {}
        }
    
    def _get_next_device(self) -> int:
        """Get next device using round-robin or load balancing."""
        with self._lock:
            if self.load_balancing:
                # Get device with most free memory
                return self.gpu_manager.get_optimal_device()
            else:
                # Round-robin
                if not self._device_queue:
                    self._device_queue = deque(self.gpu_manager.device_ids)
                device = self._device_queue.popleft()
                self._device_queue.append(device)
                return device
    
    def _run_inference_single(
        self,
        model_fn: Callable[[InputType], OutputType],
        input_data: InputType,
        device_id: int
    ) -> OutputType:
        """Run inference on a single device."""
        import torch
        
        start_time = time.time()
        
        try:
            # Set device
            self.gpu_manager.set_device(device_id)
            
            # Move input to device
            if isinstance(input_data, torch.Tensor):
                input_data = input_data.to(f"cuda:{device_id}")
            
            # Run inference
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    result = model_fn(input_data)
            
            # Update stats
            elapsed = time.time() - start_time
            with self._lock:
                self._stats["total_inferences"] += 1
                self._stats["total_time"] += elapsed
                if device_id not in self._stats["by_device"]:
                    self._stats["by_device"][device_id] = {
                        "count": 0, "time": 0.0
                    }
                self._stats["by_device"][device_id]["count"] += 1
                self._stats["by_device"][device_id]["time"] += elapsed
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed on device {device_id}: {e}")
            raise
    
    def run_parallel(
        self,
        model_fn: Callable[[InputType], OutputType],
        inputs: List[InputType],
        batch_size: int = 1
    ) -> List[OutputType]:
        """
        Run inference in parallel across GPUs.
        
        Args:
            model_fn: Function that performs inference
            inputs: List of inputs to process
            batch_size: Batch size per device
            
        Returns:
            List of results in same order as inputs
        """
        if not self.gpu_manager.has_gpu:
            # Fall back to CPU
            return [model_fn(inp) for inp in inputs]
        
        results: List[Optional[OutputType]] = [None] * len(inputs)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for idx, inp in enumerate(inputs):
                device_id = self._get_next_device()
                future = executor.submit(
                    self._run_inference_single,
                    model_fn,
                    inp,
                    device_id
                )
                futures[future] = idx
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Failed to process input {idx}: {e}")
                    raise
        
        return [r for r in results if r is not None]
    
    def run_pipeline(
        self,
        model_fns: List[Callable[[Any], Any]],
        input_data: Any,
        device_assignment: Optional[List[int]] = None
    ) -> Any:
        """
        Run a pipeline of models across different devices.
        
        Args:
            model_fns: List of model functions to run in sequence
            input_data: Initial input
            device_assignment: Optional device assignment for each stage
            
        Returns:
            Final output
        """
        import torch
        
        result = input_data
        
        for idx, model_fn in enumerate(model_fns):
            # Determine device for this stage
            if device_assignment and idx < len(device_assignment):
                device_id = device_assignment[idx]
            else:
                device_id = self._get_next_device()
            
            # Set device
            self.gpu_manager.set_device(device_id)
            
            # Move data to device
            if isinstance(result, torch.Tensor):
                result = result.to(f"cuda:{device_id}")
            
            # Run stage
            result = model_fn(result)
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get distributed inference statistics."""
        stats = self._stats.copy()
        if stats["total_inferences"] > 0:
            stats["avg_time"] = stats["total_time"] / stats["total_inferences"]
        else:
            stats["avg_time"] = 0.0
        return stats


class MixedPrecisionManager:
    """
    Manages mixed precision training and inference.
    
    Automatically selects optimal precision based on hardware capabilities.
    
    Example:
        >>> mp_mgr = MixedPrecisionManager(gpu_manager)
        >>> with mp_mgr.autocast():
        ...     output = model(input)
    """
    
    def __init__(
        self,
        gpu_manager: GPUManager,
        default_precision: GPUPrecision = GPUPrecision.FP16,
        auto_select: bool = True
    ):
        """
        Initialize mixed precision manager.
        
        Args:
            gpu_manager: GPUManager instance
            default_precision: Default precision mode
            auto_select: Automatically select best precision
        """
        self.gpu_manager = gpu_manager
        self.default_precision = default_precision
        self.auto_select = auto_select
        self._current_precision = default_precision
        
        if auto_select:
            self._current_precision = self._select_optimal_precision()
    
    def _select_optimal_precision(self) -> GPUPrecision:
        """Select optimal precision based on hardware."""
        try:
            import torch
            
            if not self.gpu_manager.has_gpu:
                return GPUPrecision.FP32
            
            # Check device capabilities
            device_id = self.gpu_manager.get_optimal_device()
            props = torch.cuda.get_device_properties(device_id)
            
            # Check for Tensor Cores (Volta and later)
            has_tensor_cores = props.major >= 7
            
            # Check for BF16 support (Ampere and later)
            supports_bf16 = props.major >= 8
            
            if supports_bf16:
                return GPUPrecision.BF16
            elif has_tensor_cores:
                return GPUPrecision.FP16
            else:
                return GPUPrecision.FP32
                
        except ImportError:
            return GPUPrecision.FP32
    
    @property
    def current_precision(self) -> GPUPrecision:
        """Get current precision mode."""
        return self._current_precision
    
    @contextmanager
    def autocast(self, enabled: bool = True) -> Iterator[None]:
        """Context manager for automatic mixed precision."""
        try:
            import torch
            
            if not enabled or not self.gpu_manager.has_gpu:
                yield
                return
            
            dtype_map = {
                GPUPrecision.FP16: torch.float16,
                GPUPrecision.BF16: torch.bfloat16,
                GPUPrecision.FP32: torch.float32,
            }
            
            dtype = dtype_map.get(self._current_precision, torch.float16)
            
            with torch.cuda.amp.autocast(dtype=dtype):
                yield
                
        except ImportError:
            yield
    
    def get_scaler(self) -> Any:
        """Get gradient scaler for training."""
        try:
            import torch
            from torch.cuda.amp import GradScaler
            
            if self._current_precision in (GPUPrecision.FP16, GPUPrecision.FP8):
                return GradScaler()
            return None
        except ImportError:
            return None


class ModelParallelManager:
    """
    Manages model parallelism across multiple GPUs.
    
    Distributes model layers across GPUs for training large models.
    
    Example:
        >>> mp_mgr = ModelParallelManager(gpu_manager)
        >>> model = mp_mgr.distribute_model(model)
    """
    
    def __init__(
        self,
        gpu_manager: GPUManager,
        split_strategy: str = "auto"
    ):
        """
        Initialize model parallel manager.
        
        Args:
            gpu_manager: GPUManager instance
            split_strategy: Strategy for splitting model ("auto", "layer", "parameter")
        """
        self.gpu_manager = gpu_manager
        self.split_strategy = split_strategy
        self._device_map: Dict[str, int] = {}
        self._layer_assignments: Dict[int, List[str]] = {}
    
    def create_device_map(
        self,
        model: Any,
        max_memory: Optional[Dict[int, str]] = None
    ) -> Dict[str, Union[int, str]]:
        """
        Create device map for model parallelism.
        
        Args:
            model: Model to distribute
            max_memory: Maximum memory per device
            
        Returns:
            Device map for model layers
        """
        try:
            from accelerate import infer_auto_device_map
            
            if max_memory is None:
                max_memory = {
                    device_id: f"{int(self.gpu_manager.get_gpu_info(device_id)['total_memory'] / 1e9 * 0.85)}GB"
                    for device_id in self.gpu_manager.device_ids
                }
            
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=["TransformerBlock"]
            )
            
            self._device_map = device_map
            return device_map
            
        except ImportError:
            logger.warning("accelerate library not available")
            return {"": 0}
    
    def distribute_model(self, model: Any) -> Any:
        """Distribute model across GPUs."""
        try:
            from accelerate import dispatch_model
            
            device_map = self.create_device_map(model)
            
            distributed_model = dispatch_model(
                model,
                device_map=device_map,
                offload_dir=None
            )
            
            logger.info(f"Model distributed across {len(set(device_map.values()))} devices")
            
            return distributed_model
            
        except ImportError:
            logger.warning("Could not distribute model, returning original")
            return model
    
    def get_layer_device(self, layer_name: str) -> int:
        """Get device assignment for a layer."""
        return self._device_map.get(layer_name, 0)


class GPUOptimizer:
    """
    High-level GPU optimization manager.
    
    Combines all GPU optimization features into a single interface.
    
    Example:
        >>> optimizer = GPUOptimizer()
        >>> optimizer.optimize_for_inference(model)
        >>> results = optimizer.run_inference(model, inputs)
    """
    
    def __init__(
        self,
        memory_fraction: float = 0.95,
        checkpoint_config: Optional[CheckpointConfig] = None,
        precision: GPUPrecision = GPUPrecision.FP16
    ):
        """
        Initialize GPU optimizer.
        
        Args:
            memory_fraction: Fraction of GPU memory to use
            checkpoint_config: Gradient checkpointing configuration
            precision: Default precision mode
        """
        self.gpu_manager = GPUManager(memory_fraction=memory_fraction)
        self.vram_manager = VRAMManager(self.gpu_manager)
        self.checkpoint_manager = GradientCheckpointManager(
            config=checkpoint_config,
            vram_manager=self.vram_manager
        )
        self.distributed_inference = DistributedInference(self.gpu_manager)
        self.mixed_precision = MixedPrecisionManager(
            self.gpu_manager,
            default_precision=precision
        )
        self.model_parallel = ModelParallelManager(self.gpu_manager)
        
        self._optimized_models: Dict[int, Any] = {}
    
    @property
    def num_gpus(self) -> int:
        """Get number of available GPUs."""
        return self.gpu_manager.num_gpus
    
    @property
    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return self.gpu_manager.has_gpu
    
    def optimize_for_inference(self, model: Any) -> Any:
        """Optimize model for inference."""
        try:
            import torch
            
            if not isinstance(model, torch.nn.Module):
                return model
            
            # Set to eval mode
            model.eval()
            
            # Compile model if available (PyTorch 2.0+)
            if hasattr(torch, "compile"):
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("Model compiled with torch.compile")
                except Exception as e:
                    logger.debug(f"Could not compile model: {e}")
            
            # Enable cudnn benchmarking
            torch.backends.cudnn.benchmark = True
            
            return model
            
        except ImportError:
            return model
    
    def optimize_for_training(
        self,
        model: Any,
        use_gradient_checkpointing: bool = True,
        use_mixed_precision: bool = True
    ) -> Any:
        """Optimize model for training."""
        try:
            import torch
            
            if not isinstance(model, torch.nn.Module):
                return model
            
            # Set to train mode
            model.train()
            
            # Apply gradient checkpointing
            if use_gradient_checkpointing:
                model = self.checkpoint_manager.wrap_model(model)
            
            # Enable cudnn benchmarking
            torch.backends.cudnn.benchmark = True
            
            return model
            
        except ImportError:
            return model
    
    def run_inference(
        self,
        model: Any,
        inputs: List[Any],
        batch_size: int = 1,
        use_distributed: bool = True
    ) -> List[Any]:
        """Run optimized inference."""
        if use_distributed and self.num_gpus > 1:
            return self.distributed_inference.run_parallel(
                model, inputs, batch_size=batch_size
            )
        else:
            # Single GPU or CPU
            with self.mixed_precision.autocast():
                return [model(inp) for inp in inputs]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            "gpu_stats": [
                self.gpu_manager.get_stats(did) 
                for did in self.gpu_manager.device_ids
            ],
            "allocation_stats": self.vram_manager.get_allocation_stats(),
            "recommendations": self.vram_manager.get_memory_recommendations()
        }
    
    def cleanup(self) -> None:
        """Clean up GPU resources."""
        self.vram_manager.optimize_memory(aggressive=True)
        self.gpu_manager.synchronize()


# Convenience functions
def get_gpu_manager(**kwargs: Any) -> GPUManager:
    """Get a GPU manager instance."""
    return GPUManager(**kwargs)


def get_vram_manager(gpu_manager: Optional[GPUManager] = None, **kwargs: Any) -> VRAMManager:
    """Get a VRAM manager instance."""
    if gpu_manager is None:
        gpu_manager = GPUManager()
    return VRAMManager(gpu_manager, **kwargs)


def optimize_for_inference(model: Any, **kwargs: Any) -> Any:
    """Convenience function to optimize model for inference."""
    optimizer = GPUOptimizer(**kwargs)
    return optimizer.optimize_for_inference(model)


def run_distributed_inference(
    model_fn: Callable[[Any], Any],
    inputs: List[Any],
    **kwargs: Any
) -> List[Any]:
    """Convenience function for distributed inference."""
    gpu_manager = GPUManager(**kwargs)
    dist_inf = DistributedInference(gpu_manager)
    return dist_inf.run_parallel(model_fn, inputs)


def print_gpu_info() -> None:
    """Print GPU information to stdout."""
    gpu_manager = GPUManager()
    
    print("=" * 60)
    print("autoconstitution GPU Information")
    print("=" * 60)
    print(f"\nNumber of GPUs: {gpu_manager.num_gpus}")
    
    for device_id in gpu_manager.device_ids:
        info = gpu_manager.get_gpu_info(device_id)
        stats = gpu_manager.get_stats(device_id)
        
        print(f"\nGPU {device_id}: {info.get('name', 'Unknown')}")
        print(f"  Total Memory: {info.get('total_memory', 0) / 1e9:.1f} GB")
        print(f"  Compute Capability: {info.get('major', 0)}.{info.get('minor', 0)}")
        print(f"  Multi-Processors: {info.get('multi_processor_count', 0)}")
        
        if isinstance(stats, GPUStats):
            print(f"  Free Memory: {stats.free_memory_gb:.1f} GB")
            print(f"  Utilization: {stats.utilization_percent:.1f}%")
            if stats.temperature_celsius:
                print(f"  Temperature: {stats.temperature_celsius}°C")
    
    print("\n" + "=" * 60)


# Export all public classes and functions
__all__ = [
    # Enums
    "GPUPrecision",
    "MemoryStrategy", 
    "CheckpointStrategy",
    
    # Data classes
    "GPUStats",
    "MemoryAllocation",
    "CheckpointConfig",
    "DistributedConfig",
    
    # Core classes
    "CUDAStreamPool",
    "GPUManager",
    "VRAMManager",
    "GradientCheckpointManager",
    "DistributedInference",
    "MixedPrecisionManager",
    "ModelParallelManager",
    "GPUOptimizer",
    
    # Convenience functions
    "get_gpu_manager",
    "get_vram_manager",
    "optimize_for_inference",
    "run_distributed_inference",
    "print_gpu_info",
]
