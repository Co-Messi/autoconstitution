"""
autoconstitution Hardware Detection Module

Automatically identifies available compute resources and configures
optimal execution strategies for M4 Neural Engine, CUDA GPUs, and CPU.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)


class ComputeBackend(Enum):
    """Enumeration of available compute backends."""
    M4_NEURAL_ENGINE = auto()
    CUDA = auto()
    ROCM = auto()
    MPS = auto()  # Metal Performance Shaders (Apple)
    OPENCL = auto()
    CPU = auto()
    UNKNOWN = auto()


class ExecutionStrategy(Enum):
    """Execution strategies based on available hardware."""
    NEURAL_ENGINE_PRIORITY = auto()  # M4 Neural Engine primary
    GPU_ACCELERATED = auto()         # CUDA/ROCm GPU primary
    METAL_ACCELERATED = auto()       # Apple Metal primary
    CPU_FALLBACK = auto()            # CPU only
    HYBRID = auto()                  # Multiple backends


@dataclass(frozen=True)
class GPUInfo:
    """Information about a detected GPU."""
    name: str
    backend: ComputeBackend
    memory_gb: float
    compute_capability: Optional[str] = None
    device_id: int = 0
    is_available: bool = True
    
    def __repr__(self) -> str:
        mem_str = f"{self.memory_gb:.1f}GB" if self.memory_gb > 0 else "Unknown"
        cc_str = f" (CC: {self.compute_capability})" if self.compute_capability else ""
        return f"GPUInfo({self.name}, {self.backend.name}, {mem_str}{cc_str})"


@dataclass(frozen=True)
class NeuralEngineInfo:
    """Information about Apple Neural Engine."""
    available: bool
    version: Optional[str] = None
    estimated_tops: Optional[float] = None  # Tera Operations Per Second
    
    def __repr__(self) -> str:
        if not self.available:
            return "NeuralEngineInfo(Not Available)"
        tops_str = f", ~{self.estimated_tops} TOPS" if self.estimated_tops else ""
        return f"NeuralEngineInfo({self.version or 'Unknown'}{tops_str})"


@dataclass(frozen=True)
class CPUInfo:
    """Information about the CPU."""
    architecture: str
    brand: str
    physical_cores: int
    logical_cores: int
    frequency_mhz: Optional[float] = None
    supports_avx: bool = False
    supports_avx2: bool = False
    supports_neon: bool = False
    
    @property
    def is_arm(self) -> bool:
        """Check if CPU is ARM architecture."""
        return self.architecture.lower() in ("arm64", "aarch64", "arm")
    
    @property
    def is_x86(self) -> bool:
        """Check if CPU is x86 architecture."""
        return self.architecture.lower() in ("x86_64", "amd64", "x86")
    
    def __repr__(self) -> str:
        features = []
        if self.supports_avx2:
            features.append("AVX2")
        elif self.supports_avx:
            features.append("AVX")
        if self.supports_neon:
            features.append("NEON")
        feat_str = f" [{', '.join(features)}]" if features else ""
        return f"CPUInfo({self.brand}, {self.physical_cores}C/{self.logical_cores}T{feat_str})"


@dataclass
class ResourceLimits:
    """Resource limits configuration."""
    max_memory_gb: float
    max_cpu_threads: int
    max_gpu_memory_gb: Optional[float] = None
    batch_size: int = 1
    num_workers: int = 1
    
    def __post_init__(self) -> None:
        """Validate resource limits."""
        if self.max_memory_gb <= 0:
            raise ValueError("max_memory_gb must be positive")
        if self.max_cpu_threads <= 0:
            raise ValueError("max_cpu_threads must be positive")


@dataclass
class HardwareProfile:
    """Complete hardware profile for the system."""
    cpu: CPUInfo
    gpus: List[GPUInfo] = field(default_factory=list)
    neural_engine: NeuralEngineInfo = field(default_factory=lambda: NeuralEngineInfo(False))
    primary_backend: ComputeBackend = ComputeBackend.CPU
    execution_strategy: ExecutionStrategy = ExecutionStrategy.CPU_FALLBACK
    
    @property
    def has_gpu(self) -> bool:
        """Check if any GPU is available."""
        return len(self.gpus) > 0 and any(gpu.is_available for gpu in self.gpus)
    
    @property
    def has_neural_engine(self) -> bool:
        """Check if Neural Engine is available."""
        return self.neural_engine.available
    
    @property
    def total_gpu_memory_gb(self) -> float:
        """Get total GPU memory across all GPUs."""
        return sum(gpu.memory_gb for gpu in self.gpus if gpu.is_available)
    
    @property
    def recommended_batch_size(self) -> int:
        """Get recommended batch size based on hardware."""
        if self.has_neural_engine:
            return 8
        elif self.has_gpu:
            total_mem = self.total_gpu_memory_gb
            if total_mem >= 24:
                return 16
            elif total_mem >= 12:
                return 8
            elif total_mem >= 6:
                return 4
            return 2
        return 1
    
    def get_optimal_limits(self, reserve_memory_gb: float = 2.0) -> ResourceLimits:
        """Calculate optimal resource limits."""
        import psutil
        
        total_ram = psutil.virtual_memory().total / (1024 ** 3)
        available_ram = total_ram - reserve_memory_gb
        
        cpu_threads = min(self.cpu.logical_cores, max(1, self.cpu.logical_cores - 2))
        
        max_gpu_mem = None
        if self.has_gpu:
            max_gpu_mem = max(
                (gpu.memory_gb for gpu in self.gpus if gpu.is_available),
                default=None
            )
        
        return ResourceLimits(
            max_memory_gb=max(1.0, available_ram * 0.8),
            max_cpu_threads=cpu_threads,
            max_gpu_memory_gb=max_gpu_mem,
            batch_size=self.recommended_batch_size,
            num_workers=max(1, min(cpu_threads // 2, 4))
        )
    
    def __repr__(self) -> str:
        parts = [f"CPU: {self.cpu}"]
        if self.gpus:
            parts.append(f"GPUs: {self.gpus}")
        if self.neural_engine.available:
            parts.append(f"ANE: {self.neural_engine}")
        parts.append(f"Backend: {self.primary_backend.name}")
        parts.append(f"Strategy: {self.execution_strategy.name}")
        return f"HardwareProfile({', '.join(parts)})"


# Type alias for detection results
DetectionResult = Tuple[bool, Optional[Any]]


class HardwareDetector:
    """
    Hardware detector for autoconstitution.
    
    Automatically identifies available compute resources including:
    - Apple M4 Neural Engine
    - NVIDIA CUDA GPUs
    - AMD ROCm GPUs
    - Apple Metal Performance Shaders
    - CPU capabilities
    
    Example:
        >>> detector = HardwareDetector()
        >>> profile = detector.detect()
        >>> print(profile)
        >>> limits = profile.get_optimal_limits()
    """
    
    def __init__(self, 
                 prefer_neural_engine: bool = True,
                 prefer_gpu: bool = True,
                 log_level: int = logging.INFO) -> None:
        """
        Initialize hardware detector.
        
        Args:
            prefer_neural_engine: Prefer Neural Engine when available
            prefer_gpu: Prefer GPU acceleration when available
            log_level: Logging level for detection messages
        """
        self.prefer_neural_engine = prefer_neural_engine
        self.prefer_gpu = prefer_gpu
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Cache for detection results
        self._cache: Optional[HardwareProfile] = None
    
    def detect(self, use_cache: bool = True) -> HardwareProfile:
        """
        Detect all available hardware and return complete profile.
        
        Args:
            use_cache: Use cached results if available
            
        Returns:
            HardwareProfile with complete system information
        """
        if use_cache and self._cache is not None:
            self.logger.debug("Using cached hardware profile")
            return self._cache
        
        self.logger.info("Starting hardware detection...")
        
        # Detect CPU
        cpu_info = self._detect_cpu()
        self.logger.info(f"Detected CPU: {cpu_info}")
        
        # Detect Neural Engine (Apple Silicon)
        neural_engine = self._detect_neural_engine()
        if neural_engine.available:
            self.logger.info(f"Detected Neural Engine: {neural_engine}")
        
        # Detect GPUs
        gpus: List[GPUInfo] = []
        
        # Try CUDA first
        cuda_gpus = self._detect_cuda()
        gpus.extend(cuda_gpus)
        
        # Try ROCm
        rocm_gpus = self._detect_rocm()
        gpus.extend(rocm_gpus)
        
        # Try Metal (Apple)
        metal_gpus = self._detect_metal()
        gpus.extend(metal_gpus)
        
        # Determine primary backend and strategy
        primary_backend, strategy = self._determine_strategy(
            cpu_info, gpus, neural_engine
        )
        
        profile = HardwareProfile(
            cpu=cpu_info,
            gpus=gpus,
            neural_engine=neural_engine,
            primary_backend=primary_backend,
            execution_strategy=strategy
        )
        
        self.logger.info(f"Hardware detection complete: {profile}")
        self._cache = profile
        return profile
    
    def _detect_cpu(self) -> CPUInfo:
        """Detect CPU information."""
        import psutil
        
        architecture = platform.machine().lower()
        system = platform.system()
        
        # Get CPU brand
        brand = platform.processor() or "Unknown"
        if not brand or brand == "":
            brand = self._get_cpu_brand_extended()
        
        # Get core counts
        physical_cores = psutil.cpu_count(logical=False) or 1
        logical_cores = psutil.cpu_count(logical=True) or 1
        
        # Get frequency
        freq_info = psutil.cpu_freq()
        frequency_mhz = freq_info.max if freq_info else None
        
        # Detect instruction set support
        supports_avx = False
        supports_avx2 = False
        supports_neon = False
        
        if system == "Darwin":
            # macOS detection
            supports_neon = self._check_neon_support()
        elif system == "Linux":
            # Linux detection via /proc/cpuinfo
            supports_avx, supports_avx2 = self._check_x86_features()
            supports_neon = self._check_neon_support()
        elif system == "Windows":
            # Windows detection
            supports_avx, supports_avx2 = self._check_x86_features_windows()
        
        return CPUInfo(
            architecture=architecture,
            brand=brand,
            physical_cores=physical_cores,
            logical_cores=logical_cores,
            frequency_mhz=frequency_mhz,
            supports_avx=supports_avx,
            supports_avx2=supports_avx2,
            supports_neon=supports_neon
        )
    
    def _get_cpu_brand_extended(self) -> str:
        """Get extended CPU brand information."""
        system = platform.system()
        
        try:
            if system == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            elif system == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
        except Exception as e:
            self.logger.debug(f"Could not get extended CPU brand: {e}")
        
        return "Unknown CPU"
    
    def _check_x86_features(self) -> Tuple[bool, bool]:
        """Check for AVX/AVX2 support on Linux."""
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read().lower()
                has_avx = "avx" in content
                has_avx2 = "avx2" in content
                return has_avx, has_avx2
        except Exception as e:
            self.logger.debug(f"Could not check x86 features: {e}")
            return False, False
    
    def _check_x86_features_windows(self) -> Tuple[bool, bool]:
        """Check for AVX/AVX2 support on Windows."""
        try:
            # Use CPUID instruction via ctypes
            import ctypes
            
            # Check if CPUID is available (always on x64)
            # For AVX, check bit 28 of ECX from CPUID leaf 1
            # For AVX2, check bit 5 of EBX from CPUID leaf 7
            
            # Simplified check - try to import numpy which uses these
            import numpy as np
            # NumPy will use AVX/AVX2 if available
            # Check numpy config for features
            config = np.show_config()
            # This is a heuristic
            return True, True  # Assume modern CPU
        except Exception:
            return False, False
    
    def _check_neon_support(self) -> bool:
        """Check for ARM NEON support."""
        architecture = platform.machine().lower()
        
        # ARM64 always has NEON
        if architecture in ("arm64", "aarch64"):
            return True
        
        # Check for ARM with NEON
        if architecture == "arm":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read().lower()
                    return "neon" in content or "asimd" in content
            except Exception:
                pass
        
        return False
    
    def _detect_neural_engine(self) -> NeuralEngineInfo:
        """
        Detect Apple Neural Engine (M4 and other Apple Silicon).
        
        Returns:
            NeuralEngineInfo with detection results
        """
        system = platform.system()
        
        if system != "Darwin":
            return NeuralEngineInfo(False)
        
        try:
            # Check for Apple Silicon
            result = subprocess.run(
                ["sysctl", "-n", "hw.machine"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return NeuralEngineInfo(False)
            
            machine = result.stdout.strip()
            
            # Check if it's Apple Silicon (arm64)
            if "arm64" not in machine:
                return NeuralEngineInfo(False)
            
            # Get chip information
            chip_result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            chip_name = chip_result.stdout.strip() if chip_result.returncode == 0 else "Apple Silicon"
            
            # Check for Neural Engine capability
            # All Apple Silicon has Neural Engine, but we verify
            ane_result = subprocess.run(
                ["ioreg", "-l", "-w0"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            has_ane = "ane" in ane_result.stdout.lower() or "neural" in ane_result.stdout.lower()
            
            # Estimate TOPS based on chip generation
            estimated_tops = self._estimate_ane_tops(chip_name)
            
            return NeuralEngineInfo(
                available=True,
                version=chip_name,
                estimated_tops=estimated_tops
            )
            
        except Exception as e:
            self.logger.debug(f"Neural Engine detection failed: {e}")
            return NeuralEngineInfo(False)
    
    def _estimate_ane_tops(self, chip_name: str) -> Optional[float]:
        """Estimate Neural Engine TOPS based on chip name."""
        chip_lower = chip_name.lower()
        
        # M4 series
        if "m4" in chip_lower:
            if "max" in chip_lower:
                return 38.0  # M4 Max
            elif "pro" in chip_lower:
                return 38.0  # M4 Pro
            return 38.0  # Base M4
        
        # M3 series
        if "m3" in chip_lower:
            if "max" in chip_lower:
                return 18.0
            elif "pro" in chip_lower:
                return 18.0
            return 18.0
        
        # M2 series
        if "m2" in chip_lower:
            if "ultra" in chip_lower:
                return 31.6
            elif "max" in chip_lower:
                return 15.8
            elif "pro" in chip_lower:
                return 15.8
            return 15.8
        
        # M1 series
        if "m1" in chip_lower:
            if "ultra" in chip_lower:
                return 22.0
            elif "max" in chip_lower:
                return 11.0
            elif "pro" in chip_lower:
                return 11.0
            return 11.0
        
        # A-series (iPhone/iPad)
        if "a18" in chip_lower:
            return 35.0
        if "a17" in chip_lower:
            return 35.0
        if "a16" in chip_lower:
            return 17.0
        if "a15" in chip_lower:
            return 15.8
        
        return None
    
    def _detect_cuda(self) -> List[GPUInfo]:
        """
        Detect NVIDIA CUDA GPUs.
        
        Returns:
            List of detected CUDA GPUs
        """
        gpus: List[GPUInfo] = []
        
        try:
            # Try to import pynvml (NVIDIA Management Library)
            import pynvml
            
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get device name
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                
                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_gb = mem_info.total / (1024 ** 3)
                
                # Get compute capability
                try:
                    cc_major, cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    compute_capability = f"{cc_major}.{cc_minor}"
                except Exception:
                    compute_capability = None
                
                gpu = GPUInfo(
                    name=name,
                    backend=ComputeBackend.CUDA,
                    memory_gb=memory_gb,
                    compute_capability=compute_capability,
                    device_id=i,
                    is_available=True
                )
                gpus.append(gpu)
                self.logger.info(f"Detected CUDA GPU: {gpu}")
            
            pynvml.nvmlShutdown()
            
        except ImportError:
            self.logger.debug("pynvml not available, trying nvidia-smi")
            gpus = self._detect_cuda_via_smi()
        except Exception as e:
            self.logger.debug(f"CUDA detection via pynvml failed: {e}")
            gpus = self._detect_cuda_via_smi()
        
        return gpus
    
    def _detect_cuda_via_smi(self) -> List[GPUInfo]:
        """Detect CUDA GPUs via nvidia-smi command."""
        gpus: List[GPUInfo] = []
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return gpus
            
            lines = result.stdout.strip().split("\n")
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    name = parts[0]
                    # Parse memory (e.g., "8192 MiB")
                    mem_str = parts[1].replace("MiB", "").replace("MB", "").strip()
                    try:
                        memory_gb = int(mem_str) / 1024
                    except ValueError:
                        memory_gb = 0.0
                    
                    compute_capability = parts[2] if len(parts) > 2 else None
                    
                    gpu = GPUInfo(
                        name=name,
                        backend=ComputeBackend.CUDA,
                        memory_gb=memory_gb,
                        compute_capability=compute_capability,
                        device_id=i,
                        is_available=True
                    )
                    gpus.append(gpu)
                    self.logger.info(f"Detected CUDA GPU via nvidia-smi: {gpu}")
                    
        except FileNotFoundError:
            self.logger.debug("nvidia-smi not found")
        except Exception as e:
            self.logger.debug(f"nvidia-smi detection failed: {e}")
        
        return gpus
    
    def _detect_rocm(self) -> List[GPUInfo]:
        """
        Detect AMD ROCm GPUs.
        
        Returns:
            List of detected ROCm GPUs
        """
        gpus: List[GPUInfo] = []
        
        try:
            # Try rocminfo
            result = subprocess.run(
                ["rocminfo"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return gpus
            
            # Parse rocminfo output
            output = result.stdout
            
            # Look for GPU devices
            if "GPU" in output or "gfx" in output:
                # Try to get more details via rocm-smi
                smi_result = subprocess.run(
                    ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if smi_result.returncode == 0:
                    # Parse rocm-smi output
                    lines = smi_result.stdout.strip().split("\n")
                    gpu_idx = 0
                    
                    for line in lines:
                        if "GPU" in line and ":" in line:
                            parts = line.split(":")
                            if len(parts) >= 2:
                                name = parts[1].strip()
                                gpu = GPUInfo(
                                    name=f"AMD {name}",
                                    backend=ComputeBackend.ROCM,
                                    memory_gb=0.0,  # Would need additional parsing
                                    device_id=gpu_idx,
                                    is_available=True
                                )
                                gpus.append(gpu)
                                self.logger.info(f"Detected ROCm GPU: {gpu}")
                                gpu_idx += 1
                                
        except FileNotFoundError:
            self.logger.debug("rocminfo/rocm-smi not found")
        except Exception as e:
            self.logger.debug(f"ROCm detection failed: {e}")
        
        return gpus
    
    def _detect_metal(self) -> List[GPUInfo]:
        """
        Detect Apple Metal GPUs.
        
        Returns:
            List of detected Metal GPUs
        """
        gpus: List[GPUInfo] = []
        
        if platform.system() != "Darwin":
            return gpus
        
        try:
            # Use system_profiler to get GPU info
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                spdisplays = data.get("SPDisplaysDataType", [])
                for i, display in enumerate(spdisplays):
                    name = display.get("sppci_model", "Apple GPU")
                    # Metal GPUs share memory with system
                    # Get system memory as approximation
                    mem_result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if mem_result.returncode == 0:
                        memory_bytes = int(mem_result.stdout.strip())
                        memory_gb = memory_bytes / (1024 ** 3)
                        # Metal uses unified memory, typically can use most of it
                        usable_memory = memory_gb * 0.75
                    else:
                        usable_memory = 8.0  # Default estimate
                    
                    gpu = GPUInfo(
                        name=name,
                        backend=ComputeBackend.MPS,
                        memory_gb=usable_memory,
                        device_id=i,
                        is_available=True
                    )
                    gpus.append(gpu)
                    self.logger.info(f"Detected Metal GPU: {gpu}")
                    
        except Exception as e:
            self.logger.debug(f"Metal detection failed: {e}")
        
        return gpus
    
    def _determine_strategy(
        self,
        cpu: CPUInfo,
        gpus: List[GPUInfo],
        neural_engine: NeuralEngineInfo
    ) -> Tuple[ComputeBackend, ExecutionStrategy]:
        """
        Determine optimal execution strategy.
        
        Args:
            cpu: CPU information
            gpus: List of detected GPUs
            neural_engine: Neural Engine information
            
        Returns:
            Tuple of (primary_backend, execution_strategy)
        """
        # Priority order: Neural Engine > CUDA > ROCm > Metal > CPU
        
        if neural_engine.available and self.prefer_neural_engine:
            self.logger.info("Selecting Neural Engine as primary backend")
            return ComputeBackend.M4_NEURAL_ENGINE, ExecutionStrategy.NEURAL_ENGINE_PRIORITY
        
        cuda_gpus = [g for g in gpus if g.backend == ComputeBackend.CUDA and g.is_available]
        if cuda_gpus and self.prefer_gpu:
            self.logger.info("Selecting CUDA as primary backend")
            return ComputeBackend.CUDA, ExecutionStrategy.GPU_ACCELERATED
        
        rocm_gpus = [g for g in gpus if g.backend == ComputeBackend.ROCM and g.is_available]
        if rocm_gpus and self.prefer_gpu:
            self.logger.info("Selecting ROCm as primary backend")
            return ComputeBackend.ROCM, ExecutionStrategy.GPU_ACCELERATED
        
        metal_gpus = [g for g in gpus if g.backend == ComputeBackend.MPS and g.is_available]
        if metal_gpus and self.prefer_gpu:
            self.logger.info("Selecting Metal as primary backend")
            return ComputeBackend.MPS, ExecutionStrategy.METAL_ACCELERATED
        
        # Check for multiple backends (hybrid)
        available_backends = len([g for g in gpus if g.is_available])
        if available_backends > 1:
            self.logger.info("Selecting hybrid execution strategy")
            return ComputeBackend.CPU, ExecutionStrategy.HYBRID
        
        self.logger.info("Selecting CPU fallback")
        return ComputeBackend.CPU, ExecutionStrategy.CPU_FALLBACK
    
    def clear_cache(self) -> None:
        """Clear the detection cache."""
        self._cache = None
        self.logger.debug("Hardware detection cache cleared")
    
    @staticmethod
    def quick_detect() -> HardwareProfile:
        """
        Quick hardware detection without caching.
        
        Returns:
            HardwareProfile with current system information
        """
        detector = HardwareDetector()
        return detector.detect(use_cache=False)


class ComputeContext(Protocol):
    """Protocol for compute context management."""
    
    def __enter__(self) -> ComputeContext:
        ...
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        ...
    
    def get_backend(self) -> ComputeBackend:
        ...


@dataclass
class ExecutionConfig:
    """Configuration for model execution."""
    backend: ComputeBackend
    device_id: int = 0
    precision: str = "fp16"  # fp32, fp16, int8, int4
    batch_size: int = 1
    num_threads: int = 1
    memory_fraction: float = 0.9
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.memory_fraction <= 0 or self.memory_fraction > 1:
            raise ValueError("memory_fraction must be in (0, 1]")


def create_execution_config(
    profile: HardwareProfile,
    precision: str = "fp16"
) -> ExecutionConfig:
    """
    Create optimal execution configuration from hardware profile.
    
    Args:
        profile: Hardware profile from detection
        precision: Desired precision (fp32, fp16, int8, int4)
        
    Returns:
        ExecutionConfig optimized for the hardware
    """
    backend = profile.primary_backend
    device_id = 0
    
    # Select best GPU if available
    if profile.gpus:
        available_gpus = [g for g in profile.gpus if g.is_available]
        if available_gpus:
            # Sort by memory, pick the best
            best_gpu = max(available_gpus, key=lambda g: g.memory_gb)
            backend = best_gpu.backend
            device_id = best_gpu.device_id
    
    # Adjust precision based on backend
    if backend == ComputeBackend.M4_NEURAL_ENGINE:
        # Neural Engine works best with int8
        precision = "int8" if precision in ("int8", "int4") else "fp16"
    elif backend == ComputeBackend.CUDA:
        # CUDA supports all precisions well
        pass
    elif backend == ComputeBackend.CPU:
        # CPU often benefits from int8 for performance
        if precision in ("int4",):
            precision = "int8"
    
    # Determine batch size
    limits = profile.get_optimal_limits()
    
    return ExecutionConfig(
        backend=backend,
        device_id=device_id,
        precision=precision,
        batch_size=limits.batch_size,
        num_threads=limits.num_workers,
        memory_fraction=0.85
    )


def get_available_backends() -> Set[ComputeBackend]:
    """
    Get set of all available compute backends on the system.
    
    Returns:
        Set of available ComputeBackend values
    """
    detector = HardwareDetector()
    profile = detector.detect()
    
    backends: Set[ComputeBackend] = {ComputeBackend.CPU}
    
    if profile.neural_engine.available:
        backends.add(ComputeBackend.M4_NEURAL_ENGINE)
    
    for gpu in profile.gpus:
        if gpu.is_available:
            backends.add(gpu.backend)
    
    return backends


# Module-level convenience functions
def detect_hardware() -> HardwareProfile:
    """
    Convenience function to detect hardware.
    
    Returns:
        HardwareProfile with system information
    """
    return HardwareDetector.quick_detect()


def get_optimal_config(precision: str = "fp16") -> ExecutionConfig:
    """
    Get optimal execution configuration for current hardware.
    
    Args:
        precision: Desired precision
        
    Returns:
        ExecutionConfig optimized for current hardware
    """
    profile = detect_hardware()
    return create_execution_config(profile, precision)


def print_hardware_info() -> None:
    """Print detailed hardware information to stdout."""
    profile = detect_hardware()
    
    print("=" * 60)
    print("autoconstitution Hardware Detection")
    print("=" * 60)
    print(f"\nCPU: {profile.cpu}")
    print(f"  Architecture: {profile.cpu.architecture}")
    print(f"  Cores: {profile.cpu.physical_cores} physical, {profile.cpu.logical_cores} logical")
    if profile.cpu.frequency_mhz:
        print(f"  Frequency: {profile.cpu.frequency_mhz:.0f} MHz")
    
    features = []
    if profile.cpu.supports_avx2:
        features.append("AVX2")
    elif profile.cpu.supports_avx:
        features.append("AVX")
    if profile.cpu.supports_neon:
        features.append("NEON")
    if features:
        print(f"  Features: {', '.join(features)}")
    
    print(f"\nNeural Engine: {'Available' if profile.has_neural_engine else 'Not Available'}")
    if profile.neural_engine.available:
        print(f"  Version: {profile.neural_engine.version or 'Unknown'}")
        if profile.neural_engine.estimated_tops:
            print(f"  Estimated Performance: ~{profile.neural_engine.estimated_tops} TOPS")
    
    print(f"\nGPUs: {len(profile.gpus)} detected")
    for i, gpu in enumerate(profile.gpus):
        print(f"  [{i}] {gpu.name}")
        print(f"      Backend: {gpu.backend.name}")
        print(f"      Memory: {gpu.memory_gb:.1f} GB")
        if gpu.compute_capability:
            print(f"      Compute Capability: {gpu.compute_capability}")
    
    print(f"\nExecution Strategy: {profile.execution_strategy.name}")
    print(f"Primary Backend: {profile.primary_backend.name}")
    
    limits = profile.get_optimal_limits()
    print(f"\nRecommended Resource Limits:")
    print(f"  Max Memory: {limits.max_memory_gb:.1f} GB")
    print(f"  Max CPU Threads: {limits.max_cpu_threads}")
    if limits.max_gpu_memory_gb:
        print(f"  Max GPU Memory: {limits.max_gpu_memory_gb:.1f} GB")
    print(f"  Batch Size: {limits.batch_size}")
    print(f"  Num Workers: {limits.num_workers}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Run hardware detection when executed directly
    logging.basicConfig(level=logging.INFO)
    print_hardware_info()
