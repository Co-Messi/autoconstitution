# autoconstitution: Cross-Platform Scaling Architecture
## From Apple Silicon M4 to H100 Clusters

**Research Report**  
*Hardware-Aware ML Deployment for Research Swarms*

---

## Executive Summary

This report analyzes the architectural requirements for designing autoconstitution to run on Apple Silicon M4 (16-64GB unified memory) as the minimum viable target while scaling seamlessly to 8x H100 clusters for power users. The two hardware tiers represent fundamentally different computing paradigms that require careful abstraction design.

**Key Finding**: A 1000x performance gap exists between M4 Max (~54 TFLOPS FP16) and 8x H100 (~16,000 TFLOPS FP8), necessitating hardware-aware design patterns with unified abstraction layers.

---

## 1. Hardware Analysis

### 1.1 Apple Silicon M4 Architecture

#### Specifications (M4 Max)
| Component | Specification |
|-----------|---------------|
| CPU | 16 cores (12P + 4E) |
| GPU | 40 cores |
| Neural Engine | 16 cores (~38 TOPS) |
| Unified Memory | Up to 128GB (usable ~96GB for GPU) |
| Memory Bandwidth | 546 GB/s |
| FP16 GPU Performance | ~54 TFLOPS |
| FP8 Support | No (Metal limitation) |
| TDP | ~90W sustained |
| Process Node | TSMC 3nm (N3E) |

#### Key Architectural Characteristics

**Unified Memory Architecture (UMA)**
- CPU, GPU, and Neural Engine share the same physical memory pool
- Zero-copy data access between compute units
- Dynamic memory allocation by memory controller
- Eliminates PCIe transfer overhead seen in discrete GPU systems
- GPU can access ~75% of total system RAM

**Neural Engine**
- 16 dedicated cores optimized for matrix multiplication
- Supports INT8 and FP16 precision
- Operates independently as a hardware accelerator (not co-processor)
- Best for inference workloads with compatible architectures
- Limited programmability compared to GPU

**Metal Performance Shaders (MPS)**
- Apple's GPU compute framework for ML workloads
- Provides PyTorch/JAX backend via `mps` device
- ~3x slower than equivalent RTX 4090 but 80% lower power consumption
- Supports dynamic shapes and automatic differentiation

**Constraints**
- No CUDA support (fundamental incompatibility)
- No FP8 tensor operations (Metal limitation)
- Thermal throttling under sustained workloads
- Limited ecosystem compared to NVIDIA

### 1.2 NVIDIA H100 Cluster Architecture

#### Specifications (Single H100 SXM)
| Component | Specification |
|-----------|---------------|
| CUDA Cores | 16,896 |
| Tensor Cores | 4th Generation |
| GPU Memory | 80GB HBM3 |
| Memory Bandwidth | 3.35 TB/s |
| FP64 | ~51 TFLOPS |
| FP32 | ~67 TFLOPS |
| FP16 Tensor Core | ~2,000 TFLOPS |
| FP8 Tensor Core | ~4,000 TFLOPS |
| NVLink Bandwidth | 600-900 GB/s |
| TDP | Up to 700W |
| Transformer Engine | Yes (dynamic precision) |

#### 8x H100 Cluster Capabilities
| Metric | Value |
|--------|-------|
| Total Compute (FP8) | ~32,000 TFLOPS |
| Total VRAM | 640GB HBM3 |
| Combined Memory BW | 26.8 TB/s |
| NVLink Topology | Full mesh via NVSwitch |
| Inter-GPU Latency | <5 microseconds |

#### Key Architectural Characteristics

**Hopper Architecture**
- Dedicated Transformer Engine for LLM workloads
- Dynamic FP8 precision selection for optimal throughput
- Fourth-generation Tensor Cores with mixed precision
- 9x faster AI training vs A100 through optimized kernels

**HBM3 Memory Subsystem**
- 2x bandwidth of A100 (3.35 TB/s vs 1.55 TB/s)
- Stacked memory architecture for massive throughput
- Critical for large model training (reduces memory-bound stalls)

**NVLink/NVSwitch Interconnect**
- 600-900 GB/s GPU-to-GPU bandwidth
- Enables model parallelism without PCIe bottlenecks
- All-to-all communication for distributed training
- Essential for training models >80B parameters

**Multi-Instance GPU (MIG)**
- Partition single H100 into 7 isolated instances
- Useful for multi-tenant inference workloads
- Not applicable for large model training

---

## 2. Comparative Analysis

### 2.1 Performance Gap Analysis

| Metric | M4 Max | 8x H100 | Ratio |
|--------|--------|---------|-------|
| Compute (FP16/FP8) | ~54 TFLOPS | ~32,000 TFLOPS | ~600x |
| Memory Capacity | 128GB (96GB usable) | 640GB | 5x |
| Memory Bandwidth | 546 GB/s | 26.8 TB/s | ~49x |
| Power Draw | ~90W | ~5600W | - |
| Cost (Hardware) | ~$4,400 | ~$300,000+ | ~68x |

### 2.2 Architectural Philosophy Differences

| Aspect | Apple Silicon M4 | NVIDIA H100 |
|--------|------------------|-------------|
| **Design Focus** | Integration, efficiency | Raw compute, scalability |
| **Memory Model** | Unified, shared | Separate HBM, explicit transfers |
| **Compute Units** | CPU + GPU + Neural Engine | Massive GPU-only parallelism |
| **Precision Focus** | FP16, INT8 | FP8, FP16, BF16 (mixed) |
| **Ecosystem** | MLX, MPS, Core ML | CUDA, cuDNN, NCCL, TensorRT |
| **Best Use Case** | Local inference, prototyping | Large-scale training, production |

### 2.3 Software Stack Compatibility

**Apple Silicon Stack:**
- MLX (native, unified memory optimized)
- PyTorch MPS backend
- JAX with Metal plugin
- llama.cpp (Metal backend)
- Core ML (inference only)

**NVIDIA Stack:**
- PyTorch CUDA
- JAX with CUDA
- TensorFlow GPU
- FlashAttention-2/3
- DeepSpeed, Megatron-LM
- vLLM, TensorRT-LLM

---

## 3. Karpathy's 5-Minute Training Budget Concept

### 3.1 Core Principle

Andrej Karpathy's autoresearch project introduced a fixed-time training budget paradigm:

> "Each experiment runs for exactly 5 minutes, check validation bits-per-byte (val_bpb), and either keep or discard changes based on performance improvement."

This creates a **time-normalized comparison metric** where:
- Faster-converging architectures are preferred
- Hyperparameter efficiency matters
- Hardware differences are abstracted (same 5 minutes on any device)
- ~12 experiments/hour, ~100 overnight

### 3.2 Translation Across Hardware Tiers

| Hardware | 5-Minute Capability | Typical Model Size | Tokens/5min |
|----------|---------------------|-------------------|-------------|
| M4 (16GB) | Small models, prototyping | 100M-1B params | ~50M-100M |
| M4 Max (128GB) | Medium models, research | 1B-7B params | ~200M-500M |
| 1x H100 | Large models, training | 7B-13B params | ~2B-5B |
| 8x H100 | Frontier models | 13B-70B+ params | ~20B-50B |

### 3.3 Key Insight: Hardware-Specific Optimizations

Karpathy's autoresearch-mlx port revealed:

> "Hyperparameter recipes don't port cleanly across hardware. On Apple Silicon, smaller and faster-training models outperform the larger configurations that win on CUDA clusters within the same time budget."

**Implication for autoconstitution:**
- Need hardware-specific default configurations
- Abstraction layer must handle optimization selection
- Same algorithmic approach, different tuning parameters

---

## 4. Design Patterns for Cross-Platform Scaling

### 4.1 Engine Abstraction Pattern

Based on successful implementations (exo, Primus), the recommended pattern:

```python
# Abstract base class for all compute engines
class ComputeEngine(ABC):
    @abstractmethod
    def initialize(self, config: HardwareConfig) -> None:
        """Initialize hardware-specific resources"""
        pass
    
    @abstractmethod
    def allocate_tensor(self, shape: Tuple, dtype: DType) -> Tensor:
        """Allocate memory on compute device"""
        pass
    
    @abstractmethod
    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiplication (core ML operation)"""
        pass
    
    @abstractmethod
    def synchronize(self) -> None:
        """Wait for all operations to complete"""
        pass
    
    @property
    @abstractmethod
    def memory_available(self) -> int:
        """Report available memory in bytes"""
        pass
    
    @property
    @abstractmethod
    def compute_flops(self) -> float:
        """Report peak compute capability"""
        pass
```

### 4.2 Backend Registry with Auto-Detection

```python
class EngineRegistry:
    _engines: Dict[str, Type[ComputeEngine]] = {}
    
    @classmethod
    def register(cls, name: str, engine_class: Type[ComputeEngine]):
        cls._engines[name] = engine_class
    
    @classmethod
    def auto_detect(cls) -> ComputeEngine:
        """Auto-detect best available hardware"""
        # Priority: MLX (Apple) > CUDA (NVIDIA) > CPU
        if cls._is_mlx_available():
            return cls._engines['mlx']()
        elif cls._is_cuda_available():
            return cls._engines['cuda']()
        else:
            return cls._engines['cpu']()
```

### 4.3 Hardware Capability Detection

```python
@dataclass
class HardwareCapabilities:
    platform: Literal['apple_silicon', 'nvidia_cuda', 'cpu']
    unified_memory: bool
    total_memory_bytes: int
    compute_flops_fp16: float
    compute_flops_fp8: Optional[float]
    memory_bandwidth_gbps: float
    supports_nccl: bool
    supports_metal: bool
    max_tensor_cores: int
    recommended_batch_size: int
    recommended_model_size: str  # 'small', 'medium', 'large', 'xlarge'
```

### 4.4 Unified Configuration System

```yaml
# hardware_config.yaml
compute:
  backend: auto  # auto, mlx, cuda, cpu
  
  # Auto-populated based on detected hardware
  capabilities:
    max_memory_gb: auto
    compute_tier: auto  # edge, desktop, workstation, cluster
    
  # Hardware-specific optimizations
  optimizations:
    apple_silicon:
      use_neural_engine: true
      memory_fraction: 0.75
      preferred_dtype: fp16
      
    nvidia_cuda:
      use_flash_attention: true
      enable_tf32: true
      preferred_dtype: fp8  # if available
      
    distributed:
      backend: nccl  # nccl, gloo, mpi
      communication_bucket_size_mb: 25
```

---

## 5. Abstraction Layer Design

### 5.1 Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    autoconstitution Application                 │
├─────────────────────────────────────────────────────────────┤
│  Research Orchestrator  │  Experiment Manager  │  Metrics   │
├─────────────────────────────────────────────────────────────┤
│              Framework Abstraction Layer                     │
│         (PyTorch-like API, hardware-agnostic)               │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   MLX Backend │  │  CUDA Backend │  │   CPU Backend    │  │
│  │  (Apple M4)   │  │  (NVIDIA)     │  │  (Fallback)      │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│              Hardware Abstraction Layer (HAL)               │
│     (Memory management, compute scheduling, synchronization) │
├─────────────────────────────────────────────────────────────┤
│  Metal/MPS    │    CUDA/cuDNN    │    OpenMP/BLAS          │
│  (Apple)      │    (NVIDIA)      │    (Generic)            │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Key Abstraction Components

#### 5.2.1 Tensor Abstraction

```python
class UnifiedTensor:
    """Hardware-agnostic tensor that works across all backends"""
    
    def __init__(self, data: Union[np.ndarray, torch.Tensor, mx.array]):
        self._data = data
        self._backend = self._detect_backend(data)
    
    def to(self, device: str) -> 'UnifiedTensor':
        """Move tensor to specified device"""
        if device == 'mlx' and self._backend != 'mlx':
            return UnifiedTensor(mx.array(self._numpy()))
        elif device == 'cuda' and self._backend != 'cuda':
            return UnifiedTensor(torch.from_numpy(self._numpy()).cuda())
        return self
    
    def matmul(self, other: 'UnifiedTensor') -> 'UnifiedTensor':
        """Hardware-accelerated matrix multiplication"""
        if self._backend == 'mlx':
            return UnifiedTensor(mx.matmul(self._data, other._data))
        elif self._backend == 'cuda':
            return UnifiedTensor(torch.matmul(self._data, other._data))
        else:
            return UnifiedTensor(np.matmul(self._data, other._data))
```

#### 5.2.2 Training Loop Abstraction

```python
class UnifiedTrainer:
    """Hardware-agnostic training loop"""
    
    def __init__(self, engine: ComputeEngine, config: TrainingConfig):
        self.engine = engine
        self.config = self._adapt_config(config, engine.capabilities)
    
    def _adapt_config(self, config: TrainingConfig, caps: HardwareCapabilities):
        """Adapt configuration to hardware capabilities"""
        adapted = copy.deepcopy(config)
        
        # Adjust batch size based on memory
        memory_based_batch = int(caps.total_memory_bytes / 
                                (config.model_params * 4))  # 4 bytes per param
        adapted.batch_size = min(config.batch_size, memory_based_batch)
        
        # Adjust precision based on hardware support
        if not caps.compute_flops_fp8 and config.dtype == 'fp8':
            adapted.dtype = 'fp16'
        
        # Set gradient accumulation to maintain effective batch size
        adapted.gradient_accumulation_steps = (
            config.batch_size // adapted.batch_size
        )
        
        return adapted
    
    def train_step(self, batch: Batch) -> Metrics:
        """Single training step (hardware-agnostic)"""
        with self.engine.autocast():
            logits = self.model(batch.inputs)
            loss = self.criterion(logits, batch.targets)
        
        self.engine.backward(loss)
        self.engine.optimizer_step()
        
        return Metrics(loss=loss.item())
```

#### 5.2.3 Distributed Training Abstraction

```python
class DistributedStrategy(ABC):
    """Abstract distributed training strategy"""
    
    @abstractmethod
    def setup(self, world_size: int, rank: int) -> None:
        pass
    
    @abstractmethod
    def all_reduce(self, tensor: Tensor) -> None:
        pass

class SingleDeviceStrategy(DistributedStrategy):
    """No-op strategy for single-device training (M4, single H100)"""
    def setup(self, world_size: int, rank: int) -> None:
        pass
    
    def all_reduce(self, tensor: Tensor) -> None:
        pass

class NCCLStrategy(DistributedStrategy):
    """NCCL-based distributed training (H100 clusters)"""
    def setup(self, world_size: int, rank: int) -> None:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
    def all_reduce(self, tensor: Tensor) -> None:
        dist.all_reduce(tensor)

class FSDPStrategy(DistributedStrategy):
    """Fully Sharded Data Parallel for large models"""
    def setup(self, world_size: int, rank: int) -> None:
        # Initialize FSDP with appropriate sharding
        pass
```

### 5.3 Precision Abstraction

```python
class PrecisionManager:
    """Handle precision differences across hardware"""
    
    PRECISION_HIERARCHY = ['fp8', 'bf16', 'fp16', 'fp32']
    
    def __init__(self, capabilities: HardwareCapabilities):
        self.capabilities = capabilities
        self.available_precisions = self._detect_precisions()
    
    def _detect_precisions(self) -> List[str]:
        precisions = ['fp32']  # Always available
        
        if self.capabilities.compute_flops_fp16 > 0:
            precisions.append('fp16')
        
        if self.capabilities.platform == 'nvidia_cuda':
            precisions.append('bf16')
        
        if self.capabilities.compute_flops_fp8:
            precisions.append('fp8')
        
        return precisions
    
    def select_optimal(self, preferred: str) -> str:
        """Select best available precision"""
        if preferred in self.available_precisions:
            return preferred
        
        # Fall back to next best
        for p in self.PRECISION_HIERARCHY:
            if p in self.available_precisions:
                return p
        
        return 'fp32'
```

---

## 6. Scaling Considerations

### 6.1 Model Size Scaling

| Hardware Tier | Max Model Size | Strategy | Memory Optimization |
|---------------|----------------|----------|---------------------|
| M4 (16GB) | 1-3B params | Full model | 4-bit quantization |
| M4 (32GB) | 3-7B params | Full model | 4-bit quantization |
| M4 Max (64GB) | 7-13B params | Full model | 8-bit quantization |
| M4 Max (128GB) | 13-30B params | Full model | FP16 |
| 1x H100 | 30-70B params | Full model | FP16/BF16 |
| 8x H100 | 70B-400B+ params | Model parallelism | FP8 + ZeRO-3 |

### 6.2 Training Throughput Scaling

```
Single M4 (16GB):     ~10-50 tokens/sec (inference)
M4 Max (128GB):       ~50-200 tokens/sec (inference)
1x H100:              ~1000-3000 tokens/sec (training)
8x H100:              ~8000-24000 tokens/sec (training)
```

### 6.3 Communication Patterns

**Single Device (M4, 1x H100):**
- No inter-device communication
- Memory bandwidth is bottleneck
- Compute-bound for large matrices

**Multi-GPU (8x H100):**
- NVLink: 600-900 GB/s all-to-all
- Gradient all-reduce: O(n) communication
- Model parallelism: Activations exchanged between layers
- Pipeline parallelism: Bubble overhead

### 6.4 Checkpointing Strategy

```python
class CheckpointManager:
    """Hardware-aware checkpointing"""
    
    def __init__(self, engine: ComputeEngine):
        self.engine = engine
        self.strategy = self._select_strategy()
    
    def _select_strategy(self):
        if self.engine.capabilities.unified_memory:
            # Apple Silicon: Direct memory save (fast)
            return UnifiedMemoryCheckpoint()
        else:
            # NVIDIA: Async GPU->CPU->Disk pipeline
            return AsyncCheckpoint()
    
    def save(self, model, optimizer, step):
        """Save checkpoint (hardware-optimized)"""
        state = {
            'model': self.engine.get_state_dict(model),
            'optimizer': optimizer.state_dict(),
            'step': step
        }
        self.strategy.save(state, f'checkpoint_{step}.pt')
```

---

## 7. Implementation Recommendations

### 7.1 Recommended Technology Stack

| Layer | Apple Silicon | NVIDIA | Notes |
|-------|---------------|--------|-------|
| Framework | MLX | PyTorch CUDA | Native optimization |
| Attention | MLX-native | FlashAttention-2/3 | Critical for performance |
| Distributed | N/A | DeepSpeed/Megatron | For 8x H100 |
| Quantization | GGUF/MLX | bitsandbytes | Memory efficiency |
| Orchestration | Local | Ray Train | Cluster management |

### 7.2 Code Organization

```
autoconstitution/
├── core/
│   ├── __init__.py
│   ├── tensor.py          # Unified tensor abstraction
│   ├── engine.py          # Compute engine interface
│   └── trainer.py         # Hardware-agnostic training loop
├── backends/
│   ├── __init__.py
│   ├── mlx_backend.py     # Apple Silicon implementation
│   ├── cuda_backend.py    # NVIDIA implementation
│   └── cpu_backend.py     # Fallback implementation
├── distributed/
│   ├── __init__.py
│   ├── single.py          # Single-device strategy
│   ├── ddp.py             # Distributed data parallel
│   └── fsdp.py            # Fully sharded data parallel
├── config/
│   ├── hardware.py        # Hardware detection
│   ├── presets/           # Hardware-specific configs
│   │   ├── m4_16gb.yaml
│   │   ├── m4_max_128gb.yaml
│   │   ├── h100_1x.yaml
│   │   └── h100_8x.yaml
│   └── optimizer.py       # Config adaptation
└── utils/
    ├── memory.py          # Memory management
    ├── precision.py       # Precision handling
    └── checkpoint.py      # Checkpoint management
```

### 7.3 Configuration Presets

```yaml
# presets/m4_16gb.yaml
hardware:
  platform: apple_silicon
  memory_gb: 16
  
model:
  max_params: 1_000_000_000  # 1B
  dtype: q4_0  # 4-bit quantization
  
training:
  batch_size: 1
  gradient_accumulation: 32
  max_sequence_length: 512
  
optimization:
  use_neural_engine: true
  compile_model: false  # MLX doesn't support torch.compile

---
# presets/h100_8x.yaml
hardware:
  platform: nvidia_cuda
  gpu_count: 8
  memory_gb: 640
  
model:
  max_params: 70_000_000_000  # 70B
  dtype: fp8
  
training:
  batch_size: 4
  gradient_accumulation: 8
  max_sequence_length: 4096
  
optimization:
  use_flash_attention: true
  distributed_strategy: fsdp
  communication_dtype: bf16
  
distributed:
  backend: nccl
  sharding_strategy: full_shard
  backward_prefetch: backward_pre
```

---

## 8. Testing and Validation Strategy

### 8.1 Hardware Compatibility Matrix

| Feature | M4 16GB | M4 Max 128GB | 1x H100 | 8x H100 |
|---------|---------|--------------|---------|---------|
| Basic training | ✅ | ✅ | ✅ | ✅ |
| Flash Attention | ❌ | ❌ | ✅ | ✅ |
| FP8 training | ❌ | ❌ | ✅ | ✅ |
| Model parallelism | N/A | N/A | ❌ | ✅ |
| Pipeline parallelism | N/A | N/A | ❌ | ✅ |
| 70B+ models | ❌ | ❌ | ❌ | ✅ |

### 8.2 Validation Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Numerical parity | <1e-5 diff | Same loss curve across backends |
| Memory efficiency | >80% | Actual/peak memory ratio |
| Compute utilization | >70% | Achieved/theoretical FLOPS |
| Checkpoint portability | 100% | Load M4 checkpoint on H100 |

---

## 9. Conclusion

### 9.1 Key Architectural Decisions

1. **Engine Abstraction Layer**: Essential for hardware portability
2. **Auto-Detection**: Simplify user experience with automatic backend selection
3. **Hardware-Specific Presets**: Pre-tuned configurations for each tier
4. **Unified Tensor API**: Hide backend differences from research code
5. **Precision Management**: Graceful degradation when FP8 unavailable

### 9.2 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Numerical differences | Cross-backend validation tests |
| Performance gaps | Hardware-specific optimization paths |
| Ecosystem fragmentation | Abstract common operations |
| Memory limitations | Automatic quantization selection |

### 9.3 Success Criteria

- ✅ Single codebase runs on M4 (16GB) and 8x H100
- ✅ <5% performance overhead from abstraction layer
- ✅ Checkpoint portability between hardware tiers
- ✅ 5-minute training budget works on all platforms
- ✅ Automatic hardware detection and optimization

---

## References

1. Karpathy, A. (2026). autoresearch: Autonomous LLM Training Experiments
2. Apple Inc. (2024). MLX Documentation and Benchmarks
3. NVIDIA Corporation (2024). H100 Tensor Core GPU Architecture
4. Scalastic.io (2025). Apple Silicon vs NVIDIA CUDA: AI Comparison
5. exo-explore/exo (2026). Cross-platform distributed inference
6. PyTorch Documentation (2024). Distributed Training Best Practices

---

*Report generated for autoconstitution architecture planning*
*Last updated: 2025*
