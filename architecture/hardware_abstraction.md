# autoconstitution Hardware Abstraction Layer (HAL)

## Executive Summary

The Hardware Abstraction Layer (HAL) enables autoconstitution to run identically across hardware tiers from Apple Silicon M4 (using Core ML/Neural Engine) to NVIDIA H100 clusters (using CUDA/distributed inference) with only a single configuration flag change.

**Design Principle**: `hardware: "auto" | "m4" | "h100" | "cpu"` — one flag, zero code changes.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         autoconstitution Application                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                         ComputeEngine (Abstract)                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   MLXEngine │  │ CUDAEngine  │  │  CPUEngine  │  │ DistributedEngine   │ │
│  │  (M4/MPS)   │  │  (NVIDIA)   │  │  (Fallback) │  │   (H100 Cluster)    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│                      Hardware Capability Detector                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Apple Silicon    │  NVIDIA GPUs      │  CPU Only       │  Kubernetes       │
│  (M1/M2/M3/M4)    │  (A100/H100)      │  (Any)          │  (Ray/vLLM)       │
│  Core ML / MLX    │  CUDA / NCCL      │  PyTorch CPU    │  Distributed      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. ComputeEngine Abstraction Interface

### 2.1 Core Interface Definition

```python
# swarm_research/core/compute_engine.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, AsyncIterator, Callable
from enum import Enum, auto
import torch
import numpy as np

class ComputeBackend(Enum):
    """Supported compute backends."""
    MLX = auto()           # Apple Silicon - MLX framework
    CUDA = auto()          # NVIDIA GPUs - CUDA/cuDNN
    CPU = auto()           # CPU-only fallback
    DISTRIBUTED = auto()   # Multi-node clusters
    CORE_ML = auto()       # Apple Core ML (for inference)
    METAL = auto()         # Apple Metal Performance Shaders

@dataclass
class HardwareProfile:
    """Complete hardware capability profile."""
    backend: ComputeBackend
    device_name: str
    total_memory_gb: float
    compute_units: int
    supports_fp16: bool
    supports_bf16: bool
    supports_int8: bool
    max_batch_size: int
    optimal_batch_size: int
    memory_bandwidth_gbps: float
    compute_tflops: float
    is_distributed: bool
    node_count: int = 1
    gpus_per_node: int = 1
    
@dataclass
class InferenceConfig:
    """Configuration for inference operations."""
    model_id: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    batch_size: int = 1
    use_cache: bool = True
    quantization: Optional[str] = None  # "int8", "int4", None
    
@dataclass
class TrainingConfig:
    """Configuration for training operations."""
    learning_rate: float = 1e-4
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    num_epochs: int = 3
    mixed_precision: bool = True
    
class ComputeEngine(ABC):
    """
    Abstract base class for all compute engines.
    
    Provides unified interface for:
    - Model loading and inference
    - Training operations
    - Memory management
    - Distributed coordination
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.profile: Optional[HardwareProfile] = None
        self._initialized = False
        
    @abstractmethod
    async def initialize(self) -> HardwareProfile:
        """Initialize the compute engine and return hardware profile."""
        pass
    
    @abstractmethod
    async def load_model(self, model_path: str, **kwargs) -> Any:
        """Load a model onto the compute device."""
        pass
    
    @abstractmethod
    async def generate(
        self, 
        model: Any, 
        prompts: List[str], 
        config: InferenceConfig
    ) -> List[str]:
        """Generate text from prompts."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        model: Any,
        prompt: str,
        config: InferenceConfig
    ) -> AsyncIterator[str]:
        """Stream generated tokens."""
        pass
    
    @abstractmethod
    async def train_step(
        self,
        model: Any,
        batch: Dict[str, torch.Tensor],
        config: TrainingConfig
    ) -> Dict[str, float]:
        """Execute single training step."""
        pass
    
    @abstractmethod
    def get_memory_stats(self) -> Dict[str, float]:
        """Return current memory usage statistics."""
        pass
    
    @abstractmethod
    async def optimize_memory(self):
        """Optimize memory usage (clear cache, defragment, etc.)."""
        pass
    
    @abstractmethod
    def synchronize(self):
        """Synchronize device operations."""
        pass
    
    @property
    @abstractmethod
    def device(self) -> Any:
        """Return the underlying device object."""
        pass
    
    @property
    @abstractmethod
    def backend_type(self) -> ComputeBackend:
        """Return the compute backend type."""
        pass
    
    async def shutdown(self):
        """Gracefully shutdown the compute engine."""
        await self.optimize_memory()
        self._initialized = False
```

### 2.2 Unified Tensor Operations

```python
# swarm_research/core/tensor_ops.py

from typing import Union, TypeVar
import numpy as np

Tensor = TypeVar('Tensor', 'torch.Tensor', 'mlx.core.array', np.ndarray)

class UnifiedTensorOps:
    """
    Hardware-agnostic tensor operations.
    Automatically dispatches to appropriate backend.
    """
    
    def __init__(self, engine: ComputeEngine):
        self.engine = engine
        self._backend = engine.backend_type
        
    def create_tensor(self, data: Union[list, np.ndarray], dtype=None) -> Tensor:
        """Create tensor on appropriate device."""
        if self._backend == ComputeBackend.MLX:
            import mlx.core as mx
            return mx.array(data, dtype=dtype)
        elif self._backend in (ComputeBackend.CUDA, ComputeBackend.CPU):
            import torch
            device = self.engine.device
            return torch.tensor(data, dtype=dtype, device=device)
        else:
            return np.array(data, dtype=dtype)
    
    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiplication."""
        if self._backend == ComputeBackend.MLX:
            import mlx.core as mx
            return mx.matmul(a, b)
        elif self._backend in (ComputeBackend.CUDA, ComputeBackend.CPU):
            return torch.matmul(a, b)
        else:
            return np.matmul(a, b)
    
    def softmax(self, x: Tensor, dim: int = -1) -> Tensor:
        """Softmax activation."""
        if self._backend == ComputeBackend.MLX:
            import mlx.core as mx
            return mx.softmax(x, axis=dim)
        elif self._backend in (ComputeBackend.CUDA, ComputeBackend.CPU):
            return torch.softmax(x, dim=dim)
        else:
            exp_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
            return exp_x / np.sum(exp_x, axis=dim, keepdims=True)
    
    def to_numpy(self, tensor: Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        if self._backend == ComputeBackend.MLX:
            return np.array(tensor)
        elif self._backend in (ComputeBackend.CUDA, ComputeBackend.CPU):
            return tensor.cpu().numpy()
        else:
            return np.array(tensor)
```

---

## 3. Backend Implementations

### 3.1 MLX Engine (Apple Silicon)

```python
# swarm_research/backends/mlx_engine.py

import mlx.core as mx
import mlx.nn as nn
from typing import List, Dict, Any, AsyncIterator
import asyncio
from ..core.compute_engine import ComputeEngine, ComputeBackend, HardwareProfile, InferenceConfig, TrainingConfig

class MLXEngine(ComputeEngine):
    """
    Apple Silicon optimized compute engine using MLX.
    
    Features:
    - Unified memory architecture (CPU/GPU shared)
    - Neural Engine acceleration for Core ML models
    - Process-based parallelism for multi-core
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._device = mx.default_device()
        self._models: Dict[str, Any] = {}
        
    async def initialize(self) -> HardwareProfile:
        """Detect and profile Apple Silicon hardware."""
        import platform
        import subprocess
        
        # Get system info
        cpu_info = platform.processor()
        
        # Detect chip generation
        chip_gen = self._detect_chip_generation()
        
        # Get memory info
        mem_info = mx.metal.get_peak_memory() / (1024**3)
        
        # Detect Neural Engine
        has_ne = self._detect_neural_engine()
        
        self.profile = HardwareProfile(
            backend=ComputeBackend.MLX,
            device_name=f"Apple {chip_gen}",
            total_memory_gb=psutil.virtual_memory().total / (1024**3),
            compute_units=mx.core.cpu_count(),
            supports_fp16=True,
            supports_bf16=chip_gen in ["M2", "M3", "M4"],
            supports_int8=True,
            max_batch_size=32 if chip_gen == "M4" else 16,
            optimal_batch_size=8 if chip_gen == "M4" else 4,
            memory_bandwidth_gbps=self._get_memory_bandwidth(chip_gen),
            compute_tflops=self._get_compute_tflops(chip_gen),
            is_distributed=False,
            node_count=1,
            gpus_per_node=1
        )
        
        self._initialized = True
        return self.profile
    
    def _detect_chip_generation(self) -> str:
        """Detect Apple Silicon chip generation."""
        import platform
        machine = platform.machine()
        
        # Use system_profiler for detailed info
        try:
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True, text=True
            )
            output = result.stdout
            
            if "M4" in output:
                return "M4"
            elif "M3" in output:
                return "M3"
            elif "M2" in output:
                return "M2"
            elif "M1" in output:
                return "M1"
        except:
            pass
            
        return "M1"  # Default fallback
    
    def _detect_neural_engine(self) -> bool:
        """Check if Neural Engine is available."""
        try:
            import coremltools as ct
            return ct.utils._is_macos_version_at_least(12, 0)
        except:
            return False
    
    def _get_memory_bandwidth(self, chip: str) -> float:
        """Get memory bandwidth for chip generation."""
        bandwidths = {
            "M1": 68.25,
            "M2": 100.0,
            "M3": 100.0,
            "M4": 120.0
        }
        return bandwidths.get(chip, 68.25)
    
    def _get_compute_tflops(self, chip: str) -> float:
        """Get compute performance for chip generation."""
        tflops = {
            "M1": 11.0,
            "M2": 13.6,
            "M3": 14.0,
            "M4": 18.0  # Estimated for M4 Pro/Max
        }
        return tflops.get(chip, 11.0)
    
    async def load_model(self, model_path: str, **kwargs) -> Any:
        """Load model using MLX or convert from HuggingFace."""
        from mlx_lm import load as mlx_load
        
        # Load using mlx-lm for HuggingFace compatibility
        model, tokenizer = mlx_load(model_path)
        
        model_key = f"{model_path}_{id(model)}"
        self._models[model_key] = {"model": model, "tokenizer": tokenizer}
        
        return model_key
    
    async def generate(
        self,
        model_key: str,
        prompts: List[str],
        config: InferenceConfig
    ) -> List[str]:
        """Generate text using MLX."""
        from mlx_lm import generate as mlx_generate
        
        model_data = self._models[model_key]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        results = []
        for prompt in prompts:
            output = mlx_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=config.max_tokens,
                temp=config.temperature,
                top_p=config.top_p,
                verbose=False
            )
            results.append(output)
            
        return results
    
    async def generate_stream(
        self,
        model_key: str,
        prompt: str,
        config: InferenceConfig
    ) -> AsyncIterator[str]:
        """Stream tokens using MLX."""
        from mlx_lm import stream_generate
        
        model_data = self._models[model_key]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        for token in stream_generate(
            model, tokenizer, prompt,
            max_tokens=config.max_tokens,
            temp=config.temperature
        ):
            yield token
    
    async def train_step(
        self,
        model_key: str,
        batch: Dict[str, mx.array],
        config: TrainingConfig
    ) -> Dict[str, float]:
        """Execute training step with MLX."""
        import mlx.optimizers as optim
        
        model_data = self._models[model_key]
        model = model_data["model"]
        
        optimizer = optim.Adam(learning_rate=config.learning_rate)
        
        def loss_fn(params):
            model.update(params)
            logits = model(batch["input_ids"])
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch["labels"].reshape(-1)
            )
            return loss.mean()
        
        loss, grads = mx.value_and_grad(loss_fn)(model.parameters())
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        return {"loss": float(loss)}
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get MLX memory statistics."""
        current = mx.metal.get_active_memory() / (1024**3)
        peak = mx.metal.get_peak_memory() / (1024**3)
        
        return {
            "current_gb": current,
            "peak_gb": peak,
            "available_gb": self.profile.total_memory_gb - current
        }
    
    async def optimize_memory(self):
        """Clear MLX cache."""
        mx.metal.clear_cache()
    
    def synchronize(self):
        """Synchronize MLX operations."""
        mx.eval()
    
    @property
    def device(self) -> Any:
        return self._device
    
    @property
    def backend_type(self) -> ComputeBackend:
        return ComputeBackend.MLX
```

### 3.2 CUDA Engine (NVIDIA GPUs)

```python
# swarm_research/backends/cuda_engine.py

import torch
import torch.nn as nn
from typing import List, Dict, Any, AsyncIterator
import pynvml
from ..core.compute_engine import ComputeEngine, ComputeBackend, HardwareProfile, InferenceConfig, TrainingConfig

class CUDAEngine(ComputeEngine):
    """
    NVIDIA GPU optimized compute engine using CUDA.
    
    Features:
    - Multi-GPU support with DataParallel/DistributedDataParallel
    - Mixed precision training (FP16/BF16)
    - TensorRT optimization for inference
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device_id = config.get("device_id", 0)
        self._device = None
        self._models: Dict[str, Any] = {}
        self._use_tensorrt = config.get("use_tensorrt", False)
        
    async def initialize(self) -> HardwareProfile:
        """Initialize CUDA and detect GPU capabilities."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        # Initialize NVML for detailed info
        pynvml.nvmlInit()
        
        self._device = torch.device(f"cuda:{self.device_id}")
        torch.cuda.set_device(self.device_id)
        
        # Get GPU info
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        # Get compute capability
        major, minor = torch.cuda.get_device_capability(self.device_id)
        compute_capability = major * 10 + minor
        
        # Detect GPU generation
        gpu_gen = self._detect_gpu_generation(gpu_name.decode())
        
        self.profile = HardwareProfile(
            backend=ComputeBackend.CUDA,
            device_name=gpu_name.decode(),
            total_memory_gb=mem_info.total / (1024**3),
            compute_units=torch.cuda.get_device_properties(self.device_id).multi_processor_count,
            supports_fp16=compute_capability >= 70,
            supports_bf16=compute_capability >= 80,
            supports_int8=compute_capability >= 75,
            max_batch_size=64 if "H100" in gpu_name.decode() else 32,
            optimal_batch_size=16 if "H100" in gpu_name.decode() else 8,
            memory_bandwidth_gbps=self._get_memory_bandwidth(gpu_gen),
            compute_tflops=self._get_compute_tflops(gpu_gen),
            is_distributed=False,
            node_count=1,
            gpus_per_node=torch.cuda.device_count()
        )
        
        # Enable TF32 for Ampere+
        if compute_capability >= 80:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self._initialized = True
        return self.profile
    
    def _detect_gpu_generation(self, gpu_name: str) -> str:
        """Detect NVIDIA GPU generation."""
        if "H100" in gpu_name or "H200" in gpu_name:
            return "Hopper"
        elif "A100" in gpu_name:
            return "Ampere"
        elif "RTX 40" in gpu_name or "Ada" in gpu_name:
            return "Ada"
        elif "RTX 30" in gpu_name or "A30" in gpu_name:
            return "Ampere"
        elif "V100" in gpu_name:
            return "Volta"
        return "Unknown"
    
    def _get_memory_bandwidth(self, gpu_gen: str) -> float:
        """Get memory bandwidth for GPU generation."""
        bandwidths = {
            "Hopper": 3350,  # H100 SXM
            "Ampere": 2039,  # A100 SXM
            "Ada": 1008,     # RTX 4090
            "Volta": 900     # V100 SXM
        }
        return bandwidths.get(gpu_gen, 500)
    
    def _get_compute_tflops(self, gpu_gen: str) -> float:
        """Get compute performance for GPU generation."""
        tflops = {
            "Hopper": 989,   # H100 SXM FP16
            "Ampere": 624,   # A100 SXM FP16
            "Ada": 330,      # RTX 4090 FP16
            "Volta": 125     # V100 SXM FP16
        }
        return tflops.get(gpu_gen, 100)
    
    async def load_model(self, model_path: str, **kwargs) -> Any:
        """Load model onto CUDA device."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Determine dtype based on hardware
        dtype = torch.float16
        if self.profile.supports_bf16:
            dtype = torch.bfloat16
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if kwargs.get("multi_gpu") else None
        )
        
        if not kwargs.get("multi_gpu"):
            model = model.to(self._device)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Compile with torch.compile if available (PyTorch 2.0+)
        if hasattr(torch, "compile") and kwargs.get("compile", True):
            model = torch.compile(model, mode="reduce-overhead")
        
        model_key = f"{model_path}_{id(model)}"
        self._models[model_key] = {"model": model, "tokenizer": tokenizer}
        
        return model_key
    
    async def generate(
        self,
        model_key: str,
        prompts: List[str],
        config: InferenceConfig
    ) -> List[str]:
        """Generate text using CUDA."""
        model_data = self._models[model_key]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        results = []
        
        # Use autocast for mixed precision
        with torch.cuda.amp.autocast(enabled=self.profile.supports_fp16):
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(self._device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=True,
                    use_cache=config.use_cache,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                results.append(result)
        
        return results
    
    async def generate_stream(
        self,
        model_key: str,
        prompt: str,
        config: InferenceConfig
    ) -> AsyncIterator[str]:
        """Stream tokens using CUDA."""
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        model_data = self._models[model_key]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(self._device)
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            yield text
        
        thread.join()
    
    async def train_step(
        self,
        model_key: str,
        batch: Dict[str, torch.Tensor],
        config: TrainingConfig
    ) -> Dict[str, float]:
        """Execute training step with CUDA."""
        model_data = self._models[model_key]
        model = model_data["model"]
        
        # Move batch to device
        batch = {k: v.to(self._device) for k, v in batch.items()}
        
        # Use GradScaler for mixed precision
        scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        model.train()
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=config.mixed_precision):
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if config.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        
        return {"loss": loss.item() * config.gradient_accumulation_steps}
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get CUDA memory statistics."""
        allocated = torch.cuda.memory_allocated(self.device_id) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device_id) / (1024**3)
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "available_gb": self.profile.total_memory_gb - allocated
        }
    
    async def optimize_memory(self):
        """Clear CUDA cache."""
        torch.cuda.empty_cache()
    
    def synchronize(self):
        """Synchronize CUDA operations."""
        torch.cuda.synchronize(self.device_id)
    
    @property
    def device(self) -> Any:
        return self._device
    
    @property
    def backend_type(self) -> ComputeBackend:
        return ComputeBackend.CUDA
```

### 3.3 CPU Engine (Fallback)

```python
# swarm_research/backends/cpu_engine.py

import torch
from typing import List, Dict, Any, AsyncIterator
import psutil
from ..core.compute_engine import ComputeEngine, ComputeBackend, HardwareProfile, InferenceConfig, TrainingConfig

class CPUEngine(ComputeEngine):
    """
    CPU-only fallback compute engine.
    
    Features:
    - Multi-threading with OpenMP
    - Intel MKL/oneDNN optimizations
    - Quantization for efficiency
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._device = torch.device("cpu")
        self._models: Dict[str, Any] = {}
        self._num_threads = config.get("num_threads", psutil.cpu_count(logical=False))
        
    async def initialize(self) -> HardwareProfile:
        """Initialize CPU backend."""
        # Set thread count
        torch.set_num_threads(self._num_threads)
        
        # Detect CPU features
        cpu_info = self._get_cpu_info()
        
        self.profile = HardwareProfile(
            backend=ComputeBackend.CPU,
            device_name=cpu_info["name"],
            total_memory_gb=psutil.virtual_memory().total / (1024**3),
            compute_units=psutil.cpu_count(logical=True),
            supports_fp16=False,  # CPU FP16 is slow
            supports_bf16=True if "AVX512" in cpu_info["flags"] else False,
            supports_int8=True,
            max_batch_size=4,
            optimal_batch_size=1,
            memory_bandwidth_gbps=50,  # Estimate
            compute_tflops=0.5,  # Estimate
            is_distributed=False,
            node_count=1,
            gpus_per_node=0
        )
        
        self._initialized = True
        return self.profile
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        info = {"name": "Unknown", "flags": []}
        
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        info["name"] = line.split(":")[1].strip()
                    if "flags" in line:
                        info["flags"] = line.split(":")[1].strip().split()
                        break
        except:
            pass
        
        return info
    
    async def load_model(self, model_path: str, **kwargs) -> Any:
        """Load model for CPU inference."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Always use int8 quantization for CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            load_in_8bit=kwargs.get("quantize", True)
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model_key = f"{model_path}_{id(model)}"
        self._models[model_key] = {"model": model, "tokenizer": tokenizer}
        
        return model_key
    
    async def generate(
        self,
        model_key: str,
        prompts: List[str],
        config: InferenceConfig
    ) -> List[str]:
        """Generate text on CPU."""
        model_data = self._models[model_key]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        results = []
        
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(result)
        
        return results
    
    async def generate_stream(
        self,
        model_key: str,
        prompt: str,
        config: InferenceConfig
    ) -> AsyncIterator[str]:
        """Stream tokens on CPU."""
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        model_data = self._models[model_key]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        inputs = tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            do_sample=True
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            yield text
        
        thread.join()
    
    async def train_step(
        self,
        model_key: str,
        batch: Dict[str, torch.Tensor],
        config: TrainingConfig
    ) -> Dict[str, float]:
        """Execute training step on CPU."""
        model_data = self._models[model_key]
        model = model_data["model"]
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        model.train()
        optimizer.zero_grad()
        
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        optimizer.step()
        
        return {"loss": loss.item()}
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get system memory statistics."""
        mem = psutil.virtual_memory()
        
        return {
            "used_gb": mem.used / (1024**3),
            "available_gb": mem.available / (1024**3),
            "percent": mem.percent
        }
    
    async def optimize_memory(self):
        """No-op for CPU."""
        pass
    
    def synchronize(self):
        """No-op for CPU."""
        pass
    
    @property
    def device(self) -> Any:
        return self._device
    
    @property
    def backend_type(self) -> ComputeBackend:
        return ComputeBackend.CPU
```

### 3.4 Distributed Engine (H100 Clusters)

```python
# swarm_research/backends/distributed_engine.py

import torch
import torch.distributed as dist
from typing import List, Dict, Any, AsyncIterator
import os
import ray
from ..core.compute_engine import ComputeEngine, ComputeBackend, HardwareProfile, InferenceConfig, TrainingConfig

class DistributedEngine(ComputeEngine):
    """
    Multi-node distributed compute engine for H100 clusters.
    
    Features:
    - Ray-based cluster management
    - vLLM for distributed inference
    - DeepSpeed/FSDP for distributed training
    - Kubernetes integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._ray_address = config.get("ray_address", "auto")
        self._world_size = config.get("world_size", 1)
        self._rank = config.get("rank", 0)
        self._local_rank = config.get("local_rank", 0)
        self._models: Dict[str, Any] = {}
        self._use_vllm = config.get("use_vllm", True)
        self._use_deepspeed = config.get("use_deepspeed", True)
        
    async def initialize(self) -> HardwareProfile:
        """Initialize distributed environment."""
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(address=self._ray_address, ignore_reinit_error=True)
        
        # Initialize process group
        if "RANK" in os.environ:
            dist.init_process_group("nccl")
            self._world_size = dist.get_world_size()
            self._rank = dist.get_rank()
        
        # Get cluster resources
        cluster_resources = ray.cluster_resources()
        
        # Detect GPUs per node
        gpus_per_node = int(cluster_resources.get("GPU", 1))
        
        # Get node count from Ray
        nodes = ray.nodes()
        node_count = len([n for n in nodes if n["Alive"]])
        
        self.profile = HardwareProfile(
            backend=ComputeBackend.DISTRIBUTED,
            device_name="H100 Cluster",
            total_memory_gb=80 * gpus_per_node * node_count,  # H100 80GB
            compute_units=132 * gpus_per_node * node_count,  # H100 SMs
            supports_fp16=True,
            supports_bf16=True,
            supports_int8=True,
            max_batch_size=256 * gpus_per_node,
            optimal_batch_size=64 * gpus_per_node,
            memory_bandwidth_gbps=3350 * gpus_per_node * node_count,
            compute_tflops=989 * gpus_per_node * node_count,
            is_distributed=True,
            node_count=node_count,
            gpus_per_node=gpus_per_node
        )
        
        self._initialized = True
        return self.profile
    
    async def load_model(self, model_path: str, **kwargs) -> Any:
        """Load model across distributed workers."""
        if self._use_vllm:
            return await self._load_vllm_model(model_path, **kwargs)
        else:
            return await self._load_torch_distributed(model_path, **kwargs)
    
    async def _load_vllm_model(self, model_path: str, **kwargs) -> Any:
        """Load model using vLLM for distributed inference."""
        from vllm import LLM
        
        # vLLM handles tensor parallelism automatically
        tensor_parallel_size = kwargs.get("tensor_parallel_size", self.profile.gpus_per_node)
        
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=kwargs.get("max_model_len", 8192)
        )
        
        model_key = f"vllm_{model_path}_{id(llm)}"
        self._models[model_key] = {"llm": llm, "type": "vllm"}
        
        return model_key
    
    async def _load_torch_distributed(self, model_path: str, **kwargs) -> Any:
        """Load model using PyTorch distributed."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
        
        # Wrap with FSDP
        model = FSDP(model)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model_key = f"fsdp_{model_path}_{id(model)}"
        self._models[model_key] = {"model": model, "tokenizer": tokenizer, "type": "fsdp"}
        
        return model_key
    
    async def generate(
        self,
        model_key: str,
        prompts: List[str],
        config: InferenceConfig
    ) -> List[str]:
        """Generate text using distributed engine."""
        model_data = self._models[model_key]
        
        if model_data["type"] == "vllm":
            return await self._generate_vllm(model_data, prompts, config)
        else:
            return await self._generate_fsdp(model_data, prompts, config)
    
    async def _generate_vllm(
        self,
        model_data: Dict,
        prompts: List[str],
        config: InferenceConfig
    ) -> List[str]:
        """Generate using vLLM."""
        from vllm import SamplingParams
        
        llm = model_data["llm"]
        
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens
        )
        
        outputs = llm.generate(prompts, sampling_params)
        
        return [output.outputs[0].text for output in outputs]
    
    async def _generate_fsdp(
        self,
        model_data: Dict,
        prompts: List[str],
        config: InferenceConfig
    ) -> List[str]:
        """Generate using FSDP."""
        # Only rank 0 generates
        if self._rank != 0:
            return []
        
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        results = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").cuda()
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True
            )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(result)
        
        return results
    
    async def generate_stream(
        self,
        model_key: str,
        prompt: str,
        config: InferenceConfig
    ) -> AsyncIterator[str]:
        """Stream tokens using distributed engine."""
        # vLLM doesn't support streaming in the same way
        # Fall back to non-streaming for distributed
        model_data = self._models[model_key]
        
        if model_data["type"] == "vllm":
            from vllm import SamplingParams
            
            llm = model_data["llm"]
            sampling_params = SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens
            )
            
            outputs = llm.generate([prompt], sampling_params)
            text = outputs[0].outputs[0].text
            
            # Simulate streaming
            for token in text.split():
                yield token + " "
        else:
            # FSDP streaming
            raise NotImplementedError("Streaming not implemented for FSDP")
    
    async def train_step(
        self,
        model_key: str,
        batch: Dict[str, torch.Tensor],
        config: TrainingConfig
    ) -> Dict[str, float]:
        """Execute distributed training step."""
        model_data = self._models[model_key]
        model = model_data["model"]
        
        if self._use_deepspeed:
            return await self._train_deepspeed(model, batch, config)
        else:
            return await self._train_fsdp(model, batch, config)
    
    async def _train_deepspeed(
        self,
        model: Any,
        batch: Dict[str, torch.Tensor],
        config: TrainingConfig
    ) -> Dict[str, float]:
        """Training step with DeepSpeed."""
        import deepspeed
        
        # DeepSpeed handles optimizer and gradient accumulation
        outputs = model(**batch)
        loss = outputs.loss
        
        model.backward(loss)
        model.step()
        
        # Aggregate loss across ranks
        dist.all_reduce(loss)
        loss = loss / self._world_size
        
        return {"loss": loss.item()}
    
    async def _train_fsdp(
        self,
        model: Any,
        batch: Dict[str, torch.Tensor],
        config: TrainingConfig
    ) -> Dict[str, float]:
        """Training step with FSDP."""
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        
        # FSDP handles gradient synchronization
        
        return {"loss": loss.item()}
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get distributed memory statistics."""
        allocated = torch.cuda.memory_allocated() / (1024**3)
        
        # Gather from all ranks
        all_allocated = [torch.zeros(1).cuda() for _ in range(self._world_size)]
        dist.all_gather(all_allocated, torch.tensor([allocated]).cuda())
        
        total_allocated = sum([t.item() for t in all_allocated])
        
        return {
            "allocated_gb_per_gpu": allocated,
            "total_allocated_gb": total_allocated,
            "gpus": self._world_size
        }
    
    async def optimize_memory(self):
        """Clear distributed cache."""
        torch.cuda.empty_cache()
    
    def synchronize(self):
        """Synchronize distributed operations."""
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
    
    @property
    def device(self) -> Any:
        return torch.device(f"cuda:{self._local_rank}")
    
    @property
    def backend_type(self) -> ComputeBackend:
        return ComputeBackend.DISTRIBUTED
```

---

## 4. Auto-Detection Logic

### 4.1 Hardware Detector

```python
# swarm_research/core/hardware_detector.py

import platform
import subprocess
from typing import Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass

class HardwareTier(Enum):
    """Predefined hardware tiers."""
    M4 = "m4"                    # Apple M4 (latest)
    M3 = "m3"                    # Apple M3
    M2 = "m2"                    # Apple M2
    M1 = "m1"                    # Apple M1
    H100 = "h100"                # NVIDIA H100
    A100 = "a100"                # NVIDIA A100
    RTX4090 = "rtx4090"          # Consumer RTX
    CPU_HIGH = "cpu_high"        # High-end CPU
    CPU_LOW = "cpu_low"          # Low-end CPU

@dataclass
class DetectedHardware:
    """Result of hardware detection."""
    tier: HardwareTier
    backend: str  # "mlx", "cuda", "cpu"
    device_name: str
    memory_gb: float
    recommended_config: Dict[str, Any]

class HardwareDetector:
    """
    Automatically detect available hardware and recommend configuration.
    """
    
    @classmethod
    def detect(cls) -> DetectedHardware:
        """Detect hardware and return recommended configuration."""
        
        # Check for Apple Silicon first
        if cls._is_apple_silicon():
            return cls._detect_apple_silicon()
        
        # Check for NVIDIA GPUs
        if cls._has_nvidia_gpu():
            return cls._detect_nvidia_gpu()
        
        # Fallback to CPU
        return cls._detect_cpu()
    
    @classmethod
    def _is_apple_silicon(cls) -> bool:
        """Check if running on Apple Silicon."""
        return platform.system() == "Darwin" and platform.machine() == "arm64"
    
    @classmethod
    def _has_nvidia_gpu(cls) -> bool:
        """Check if NVIDIA GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            try:
                subprocess.run(["nvidia-smi"], capture_output=True, check=True)
                return True
            except:
                return False
    
    @classmethod
    def _detect_apple_silicon(cls) -> DetectedHardware:
        """Detect Apple Silicon generation."""
        try:
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True, text=True
            )
            output = result.stdout
            
            if "M4" in output:
                tier = HardwareTier.M4
                batch_size = 16
            elif "M3" in output:
                tier = HardwareTier.M3
                batch_size = 12
            elif "M2" in output:
                tier = HardwareTier.M2
                batch_size = 8
            else:
                tier = HardwareTier.M1
                batch_size = 4
            
            return DetectedHardware(
                tier=tier,
                backend="mlx",
                device_name=f"Apple {tier.value.upper()}",
                memory_gb=cls._get_apple_memory(),
                recommended_config={
                    "batch_size": batch_size,
                    "max_tokens": 1024,
                    "use_neural_engine": True,
                    "quantization": None
                }
            )
        except:
            return cls._detect_cpu()
    
    @classmethod
    def _detect_nvidia_gpu(cls) -> DetectedHardware:
        """Detect NVIDIA GPU generation."""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle).decode()
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            if "H100" in name:
                tier = HardwareTier.H100
                batch_size = 64
            elif "A100" in name:
                tier = HardwareTier.A100
                batch_size = 48
            elif "4090" in name:
                tier = HardwareTier.RTX4090
                batch_size = 16
            else:
                tier = HardwareTier.A100  # Default
                batch_size = 16
            
            return DetectedHardware(
                tier=tier,
                backend="cuda",
                device_name=name,
                memory_gb=mem_info.total / (1024**3),
                recommended_config={
                    "batch_size": batch_size,
                    "max_tokens": 2048,
                    "mixed_precision": True,
                    "use_tensorrt": tier in [HardwareTier.H100, HardwareTier.A100]
                }
            )
        except:
            return cls._detect_cpu()
    
    @classmethod
    def _detect_cpu(cls) -> DetectedHardware:
        """Detect CPU capabilities."""
        import psutil
        
        cpu_count = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        tier = HardwareTier.CPU_HIGH if cpu_count >= 16 else HardwareTier.CPU_LOW
        
        return DetectedHardware(
            tier=tier,
            backend="cpu",
            device_name=platform.processor() or "Unknown CPU",
            memory_gb=memory_gb,
            recommended_config={
                "batch_size": 1 if tier == HardwareTier.CPU_LOW else 2,
                "max_tokens": 512,
                "num_threads": min(cpu_count, 8),
                "quantize": True
            }
        )
    
    @classmethod
    def _get_apple_memory(cls) -> float:
        """Get Apple Silicon memory."""
        import psutil
        return psutil.virtual_memory().total / (1024**3)
```

---

## 5. Configuration Presets

### 5.1 Config Schema

```yaml
# swarm_research/config/hardware_presets.yaml

# ============================================
# Hardware Configuration Presets
# ============================================
# Usage: Set hardware: "<tier>" in config.yaml
# Options: auto, m4, m3, m2, m1, h100, a100, rtx4090, cpu

presets:
  # Apple Silicon - M4 (Latest)
  m4:
    backend: mlx
    device: mps
    compute:
      batch_size: 16
      max_tokens: 2048
      temperature: 0.7
      top_p: 0.9
    memory:
      max_memory_gb: 128
      gradient_checkpointing: true
    optimization:
      use_neural_engine: true
      quantization: null
      compile: true
    training:
      learning_rate: 1.0e-4
      warmup_steps: 100
      gradient_accumulation_steps: 1
    parallelism:
      type: process
      num_workers: 8
    
  # Apple Silicon - M3
  m3:
    backend: mlx
    device: mps
    compute:
      batch_size: 12
      max_tokens: 1536
      temperature: 0.7
      top_p: 0.9
    memory:
      max_memory_gb: 96
      gradient_checkpointing: true
    optimization:
      use_neural_engine: true
      quantization: null
      compile: true
    training:
      learning_rate: 1.0e-4
      warmup_steps: 100
      gradient_accumulation_steps: 1
    parallelism:
      type: process
      num_workers: 6

  # Apple Silicon - M2
  m2:
    backend: mlx
    device: mps
    compute:
      batch_size: 8
      max_tokens: 1024
      temperature: 0.7
      top_p: 0.9
    memory:
      max_memory_gb: 64
      gradient_checkpointing: true
    optimization:
      use_neural_engine: true
      quantization: int8
      compile: false
    training:
      learning_rate: 5.0e-5
      warmup_steps: 200
      gradient_accumulation_steps: 2
    parallelism:
      type: process
      num_workers: 4

  # Apple Silicon - M1
  m1:
    backend: mlx
    device: mps
    compute:
      batch_size: 4
      max_tokens: 512
      temperature: 0.7
      top_p: 0.9
    memory:
      max_memory_gb: 32
      gradient_checkpointing: true
    optimization:
      use_neural_engine: false
      quantization: int8
      compile: false
    training:
      learning_rate: 5.0e-5
      warmup_steps: 300
      gradient_accumulation_steps: 4
    parallelism:
      type: process
      num_workers: 2

  # NVIDIA H100 Cluster
  h100:
    backend: cuda
    device: cuda
    compute:
      batch_size: 64
      max_tokens: 8192
      temperature: 0.7
      top_p: 0.9
    memory:
      max_memory_gb: 640
      gradient_checkpointing: false
    optimization:
      mixed_precision: bfloat16
      use_tensorrt: true
      use_flash_attention: true
      compile: true
    training:
      learning_rate: 1.0e-4
      warmup_steps: 50
      gradient_accumulation_steps: 1
      use_deepspeed: true
      zero_stage: 2
    parallelism:
      type: distributed
      backend: nccl
      tensor_parallel_size: 8
      pipeline_parallel_size: 1
    kubernetes:
      enabled: true
      namespace: autoconstitution
      service_name: autoconstitution-head
      image: autoconstitution/h100:latest
      resources:
        nvidia.com/gpu: 8
        memory: 640Gi
        cpu: 64

  # NVIDIA A100
  a100:
    backend: cuda
    device: cuda
    compute:
      batch_size: 48
      max_tokens: 4096
      temperature: 0.7
      top_p: 0.9
    memory:
      max_memory_gb: 80
      gradient_checkpointing: false
    optimization:
      mixed_precision: bfloat16
      use_tensorrt: true
      use_flash_attention: true
      compile: true
    training:
      learning_rate: 1.0e-4
      warmup_steps: 100
      gradient_accumulation_steps: 1
      use_deepspeed: true
      zero_stage: 2
    parallelism:
      type: distributed
      backend: nccl
      tensor_parallel_size: 8

  # NVIDIA RTX 4090 (Consumer)
  rtx4090:
    backend: cuda
    device: cuda
    compute:
      batch_size: 16
      max_tokens: 2048
      temperature: 0.7
      top_p: 0.9
    memory:
      max_memory_gb: 24
      gradient_checkpointing: true
    optimization:
      mixed_precision: float16
      use_tensorrt: false
      use_flash_attention: true
      compile: true
    training:
      learning_rate: 5.0e-5
      warmup_steps: 200
      gradient_accumulation_steps: 4
    parallelism:
      type: data_parallel
      num_gpus: 1

  # CPU High-End
  cpu_high:
    backend: cpu
    device: cpu
    compute:
      batch_size: 2
      max_tokens: 512
      temperature: 0.7
      top_p: 0.9
    memory:
      max_memory_gb: 64
      gradient_checkpointing: true
    optimization:
      quantization: int8
      num_threads: 8
      use_mkldnn: true
    training:
      learning_rate: 1.0e-5
      warmup_steps: 500
      gradient_accumulation_steps: 8
    parallelism:
      type: thread
      num_workers: 8

  # CPU Low-End
  cpu_low:
    backend: cpu
    device: cpu
    compute:
      batch_size: 1
      max_tokens: 256
      temperature: 0.7
      top_p: 0.9
    memory:
      max_memory_gb: 16
      gradient_checkpointing: true
    optimization:
      quantization: int8
      num_threads: 4
      use_mkldnn: false
    training:
      learning_rate: 5.0e-6
      warmup_steps: 1000
      gradient_accumulation_steps: 16
    parallelism:
      type: thread
      num_workers: 2

# Auto-detection settings
auto_detection:
  enabled: true
  fallback_order:
    - cuda
    - mlx
    - cpu
  cache_results: true
  cache_path: .cache/hardware_profile.json
```

### 5.2 Configuration Loader

```python
# swarm_research/config/config_loader.py

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class HardwareConfig:
    """Hardware configuration dataclass."""
    backend: str = "auto"
    device: str = "auto"
    batch_size: int = 1
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    mixed_precision: bool = False
    quantization: Optional[str] = None
    num_workers: int = 1
    use_distributed: bool = False
    
@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 3
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
@dataclass
class SwarmConfig:
    """Main autoconstitution configuration."""
    hardware: str = "auto"  # auto, m4, m3, m2, m1, h100, a100, rtx4090, cpu
    model_id: str = "microsoft/DialoGPT-medium"
    hardware_config: HardwareConfig = field(default_factory=HardwareConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "SwarmConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        hardware = data.get('hardware', 'auto')
        
        # Load presets if hardware is specified
        if hardware != 'auto':
            preset = cls._load_preset(hardware)
            data = cls._merge_configs(preset, data)
        
        return cls._from_dict(data)
    
    @classmethod
    def _load_preset(cls, hardware: str) -> Dict[str, Any]:
        """Load hardware preset."""
        preset_path = Path(__file__).parent / "hardware_presets.yaml"
        
        with open(preset_path, 'r') as f:
            presets = yaml.safe_load(f)
        
        return presets.get('presets', {}).get(hardware, {})
    
    @classmethod
    def _merge_configs(cls, preset: Dict, user: Dict) -> Dict:
        """Merge preset with user configuration."""
        merged = preset.copy()
        
        for key, value in user.items():
            if isinstance(value, dict) and key in merged:
                merged[key].update(value)
            else:
                merged[key] = value
        
        return merged
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "SwarmConfig":
        """Create config from dictionary."""
        return cls(
            hardware=data.get('hardware', 'auto'),
            model_id=data.get('model_id', 'microsoft/DialoGPT-medium'),
            hardware_config=HardwareConfig(**data.get('hardware_config', {})),
            training_config=TrainingConfig(**data.get('training_config', {}))
        )

# Example usage configuration
example_config = """
# autoconstitution Configuration
# Set hardware: "m4" for Apple Silicon, "h100" for NVIDIA cluster

hardware: "auto"  # auto-detect, or specify: m4, m3, h100, a100, cpu

model_id: "microsoft/DialoGPT-medium"

# Optional: Override preset values
hardware_config:
  batch_size: 8  # Override default for your hardware
  max_tokens: 1024

training_config:
  learning_rate: 2.0e-4
  num_epochs: 5
"""
```

---

## 6. Engine Factory

```python
# swarm_research/core/engine_factory.py

from typing import Dict, Any, Optional
from .compute_engine import ComputeEngine, ComputeBackend
from .hardware_detector import HardwareDetector, DetectedHardware

class ComputeEngineFactory:
    """
    Factory for creating appropriate compute engine based on configuration.
    
    Usage:
        engine = ComputeEngineFactory.create("auto")  # Auto-detect
        engine = ComputeEngineFactory.create("m4")    # Force Apple Silicon
        engine = ComputeEngineFactory.create("h100")  # Force H100 cluster
    """
    
    _engines: Dict[str, Any] = {
        "mlx": None,
        "cuda": None,
        "cpu": None,
        "distributed": None
    }
    
    @classmethod
    def create(
        cls,
        hardware: str = "auto",
        config: Optional[Dict[str, Any]] = None
    ) -> ComputeEngine:
        """
        Create compute engine for specified hardware.
        
        Args:
            hardware: Hardware tier (auto, m4, m3, m2, m1, h100, a100, rtx4090, cpu)
            config: Optional configuration overrides
        
        Returns:
            Configured ComputeEngine instance
        """
        config = config or {}
        
        # Auto-detect if requested
        if hardware == "auto":
            detected = HardwareDetector.detect()
            backend = detected.backend
        else:
            # Map hardware tier to backend
            backend = cls._map_hardware_to_backend(hardware)
            config['hardware_tier'] = hardware
        
        # Create appropriate engine
        if backend == "mlx":
            return cls._create_mlx_engine(config)
        elif backend == "cuda":
            if hardware in ["h100"] or config.get("distributed", False):
                return cls._create_distributed_engine(config)
            return cls._create_cuda_engine(config)
        elif backend == "cpu":
            return cls._create_cpu_engine(config)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    @classmethod
    def _map_hardware_to_backend(cls, hardware: str) -> str:
        """Map hardware tier to backend type."""
        mapping = {
            "m4": "mlx",
            "m3": "mlx",
            "m2": "mlx",
            "m1": "mlx",
            "h100": "cuda",
            "a100": "cuda",
            "rtx4090": "cuda",
            "cpu": "cpu",
            "cpu_high": "cpu",
            "cpu_low": "cpu"
        }
        return mapping.get(hardware, "cpu")
    
    @classmethod
    def _create_mlx_engine(cls, config: Dict[str, Any]) -> ComputeEngine:
        """Create MLX engine for Apple Silicon."""
        from ..backends.mlx_engine import MLXEngine
        
        if cls._engines["mlx"] is None:
            cls._engines["mlx"] = MLXEngine(config)
        
        return cls._engines["mlx"]
    
    @classmethod
    def _create_cuda_engine(cls, config: Dict[str, Any]) -> ComputeEngine:
        """Create CUDA engine for NVIDIA GPUs."""
        from ..backends.cuda_engine import CUDAEngine
        
        if cls._engines["cuda"] is None:
            cls._engines["cuda"] = CUDAEngine(config)
        
        return cls._engines["cuda"]
    
    @classmethod
    def _create_cpu_engine(cls, config: Dict[str, Any]) -> ComputeEngine:
        """Create CPU engine."""
        from ..backends.cpu_engine import CPUEngine
        
        if cls._engines["cpu"] is None:
            cls._engines["cpu"] = CPUEngine(config)
        
        return cls._engines["cpu"]
    
    @classmethod
    def _create_distributed_engine(cls, config: Dict[str, Any]) -> ComputeEngine:
        """Create distributed engine for clusters."""
        from ..backends.distributed_engine import DistributedEngine
        
        if cls._engines["distributed"] is None:
            cls._engines["distributed"] = DistributedEngine(config)
        
        return cls._engines["distributed"]
    
    @classmethod
    def reset(cls):
        """Reset all cached engines."""
        cls._engines = {
            "mlx": None,
            "cuda": None,
            "cpu": None,
            "distributed": None
        }
```

---

## 7. Usage Examples

### 7.1 Single Config Flag Usage

```python
# Example: User only changes one line

# For M4 MacBook Pro
config = SwarmConfig.from_dict({"hardware": "m4"})
engine = ComputeEngineFactory.create("m4")

# For H100 cluster
config = SwarmConfig.from_dict({"hardware": "h100"})
engine = ComputeEngineFactory.create("h100")

# Auto-detect (works everywhere)
config = SwarmConfig.from_dict({"hardware": "auto"})
engine = ComputeEngineFactory.create("auto")
```

### 7.2 Complete Application Example

```python
# swarm_research/example_usage.py

import asyncio
from swarm_research.core.engine_factory import ComputeEngineFactory
from swarm_research.core.compute_engine import InferenceConfig
from swarm_research.config.config_loader import SwarmConfig

async def main():
    # Load configuration (hardware: "m4" or "h100" or "auto")
    config = SwarmConfig.from_yaml("config.yaml")
    
    # Create engine (single line change for different hardware)
    engine = ComputeEngineFactory.create(config.hardware)
    
    # Initialize
    profile = await engine.initialize()
    print(f"Hardware: {profile.device_name}")
    print(f"Memory: {profile.total_memory_gb} GB")
    print(f"Backend: {profile.backend}")
    
    # Load model
    model_key = await engine.load_model(config.model_id)
    
    # Configure inference
    inference_config = InferenceConfig(
        model_id=config.model_id,
        max_tokens=512,
        temperature=0.7
    )
    
    # Generate
    prompts = [
        "What is machine learning?",
        "Explain neural networks."
    ]
    
    results = await engine.generate(model_key, prompts, inference_config)
    
    for prompt, result in zip(prompts, results):
        print(f"\nPrompt: {prompt}")
        print(f"Result: {result}")
    
    # Cleanup
    await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### 7.3 YAML Configuration Examples

```yaml
# config_m4.yaml - For Apple Silicon
hardware: "m4"
model_id: "microsoft/DialoGPT-medium"

# config_h100.yaml - For H100 cluster
hardware: "h100"
model_id: "microsoft/DialoGPT-medium"

# config_auto.yaml - Auto-detect anywhere
hardware: "auto"
model_id: "microsoft/DialoGPT-medium"
```

---

## 8. File Structure

```
swarm_research/
├── core/
│   ├── __init__.py
│   ├── compute_engine.py       # Abstract base class
│   ├── tensor_ops.py            # Unified tensor operations
│   ├── hardware_detector.py     # Auto-detection logic
│   └── engine_factory.py        # Engine factory
├── backends/
│   ├── __init__.py
│   ├── mlx_engine.py            # Apple Silicon (MLX)
│   ├── cuda_engine.py           # NVIDIA GPUs (CUDA)
│   ├── cpu_engine.py            # CPU fallback
│   └── distributed_engine.py    # Multi-node (Ray/vLLM)
├── config/
│   ├── __init__.py
│   ├── config_loader.py         # Configuration loader
│   └── hardware_presets.yaml    # Hardware presets
└── example_usage.py             # Usage examples
```

---

## 9. Summary

### Key Design Decisions

1. **Single Interface**: `ComputeEngine` abstract base class provides unified API
2. **Auto-Detection**: `HardwareDetector` automatically selects optimal backend
3. **Config Presets**: YAML presets for each hardware tier
4. **Factory Pattern**: `ComputeEngineFactory` creates appropriate engine
5. **Zero Code Changes**: User only changes `hardware: "m4"` to `hardware: "h100"`

### Hardware Tier Mapping

| Hardware | Backend | Key Features |
|----------|---------|--------------|
| M4 | MLX | Neural Engine, unified memory |
| M3/M2/M1 | MLX | Core ML, process parallelism |
| H100 | CUDA + Distributed | vLLM, DeepSpeed, NCCL |
| A100 | CUDA | TensorRT, FSDP |
| RTX 4090 | CUDA | Consumer GPU support |
| CPU | CPU | MKL, quantization fallback |

### Migration Path

```python
# Before (hardware-specific code)
if torch.cuda.is_available():
    device = "cuda"
    model = model.cuda()
else:
    device = "cpu"

# After (hardware-agnostic)
engine = ComputeEngineFactory.create("auto")  # Works everywhere
```
