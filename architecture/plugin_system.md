# autoconstitution Plugin Architecture
## Extensible Plugin System Design

---

## Executive Summary

The autoconstitution Plugin Architecture provides a clean, extensible framework for customizing all major aspects of the research system. This design transforms autoconstitution from a single-purpose demo into a general-purpose platform for AI-driven research automation.

**Key Design Goals:**
1. **Modularity**: Each plugin type has a well-defined, minimal interface
2. **Discoverability**: Plugins are auto-discovered from standard locations
3. **Composability**: Plugins can be combined and chained
4. **Testability**: Plugin interfaces enable easy mocking and testing
5. **Hot-swapping**: Plugins can be loaded/unloaded without system restart

---

## 1. Plugin System Overview

### 1.1 Plugin Categories

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SWARMRESEARCH PLUGIN ECOSYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CORE PLUGIN INTERFACES                            │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │   │
│  │  │ RatchetMetric   │  │  LLMProvider    │  │ TrainingTarget  │     │   │
│  │  │   Interface     │  │   Interface     │  │   Interface     │     │   │
│  │  │                 │  │                 │  │                 │     │   │
│  │  │ • val_bpb       │  │ • Kimi K2.5     │  │ • LM Training   │     │   │
│  │  │ • perplexity    │  │ • Claude        │  │ • RL Training   │     │   │
│  │  │ • inference_tps │  │ • GPT-4         │  │ • Fine-tuning   │     │   │
│  │  │ • memory_usage  │  │ • Ollama        │  │ • Distillation  │     │   │
│  │  │ • composite     │  │ • vLLM          │  │ • NAS           │     │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘     │   │
│  │                                                                      │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │   │
│  │  │ ExperimentRunner│  │  CriticStrategy │  │  Aggregation    │     │   │
│  │  │   Interface     │  │   Interface     │  │   Strategy      │     │   │
│  │  │                 │  │                 │  │                 │     │   │
│  │  │ • Single GPU    │  │ • Benchmark     │  │ • Simple Concat │     │   │
│  │  │ • Multi-GPU     │  │ • Generalize    │  │ • Voting        │     │   │
│  │  │ • Distributed   │  │ • Metric Sanity │  │ • Consensus     │     │   │
│  │  │ • Cloud         │  │ • Code Quality  │  │ • Refinement    │     │   │
│  │  │ • Serverless    │  │ • Scientific    │  │ • Tree-of-Thought│    │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘     │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PLUGIN DISCOVERY SYSTEM                           │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │   Entry Points      File System        Decorators       Config      │   │
│  │   ───────────       ──────────         ──────────       ──────      │   │
│  │   autoconstitution.    plugins/           @register_         YAML      │   │
│  │     metrics           ├── metrics/       metric          Files      │   │
│  │     providers         ├── providers/   @register_                     │   │
│  │     training          ├── training/      provider                    │   │
│  │     runners           ├── runners/     @register_                     │   │
│  │     critics           └── critics/       runner                     │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Plugin Architecture Principles

```python
"""
Plugin Architecture Core Principles

1. INTERFACE SEGREGATION
   - Each plugin type has exactly one interface to implement
   - Interfaces are minimal and focused
   - No plugin implements more than one interface

2. DEPENDENCY INVERSION
   - Core system depends on plugin interfaces, not implementations
   - Plugins depend only on the interface they implement
   - No circular dependencies between plugins

3. OPEN/CLOSED PRINCIPLE
   - System is open for extension (new plugins)
   - System is closed for modification (core unchanged)

4. SINGLE RESPONSIBILITY
   - Each plugin does one thing well
   - Plugins are composable, not monolithic
"""
```

---

## 2. Plugin Interface Definitions

### 2.1 Base Plugin Interface

```python
# autoconstitution/plugins/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, TypeVar, Generic
from enum import Enum, auto
from datetime import datetime
from pathlib import Path
import importlib.metadata


class PluginType(Enum):
    """All supported plugin types."""
    RATCHET_METRIC = auto()
    LLM_PROVIDER = auto()
    TRAINING_TARGET = auto()
    EXPERIMENT_RUNNER = auto()
    CRITIC_STRATEGY = auto()
    AGGREGATION_STRATEGY = auto()
    TOOL = auto()
    WORKFLOW = auto()


@dataclass
class PluginMetadata:
    """Metadata for any plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str]
    config_schema: Dict[str, Any]
    
    # Runtime info
    loaded_at: Optional[datetime] = None
    source_path: Optional[Path] = None


@dataclass
class PluginHealth:
    """Health status of a plugin."""
    healthy: bool
    last_check: datetime
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = None


class autoconstitutionPlugin(ABC):
    """
    Base interface that ALL plugins must implement.
    
    This is the root of the plugin hierarchy. Every plugin,
    regardless of type, must satisfy this minimal contract.
    """
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up plugin resources."""
        pass
    
    @abstractmethod
    async def health_check(self) -> PluginHealth:
        """Check plugin health status."""
        pass
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return JSON schema for plugin configuration."""
        return self.metadata.config_schema


T = TypeVar('T', bound=autoconstitutionPlugin)
```

### 2.2 Ratchet Metric Interface

```python
# autoconstitution/plugins/interfaces/metrics.py

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import numpy as np

from .base import autoconstitutionPlugin, PluginMetadata, PluginType


class OptimizationDirection(Enum):
    """Whether metric should be minimized or maximized."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class MetricValue:
    """A computed metric value with context."""
    value: float
    timestamp: float
    metadata: Dict[str, Any]
    raw_data: Optional[Dict[str, Any]] = None
    
    def __float__(self) -> float:
        return float(self.value)


@dataclass
class ComparisonResult:
    """Result of comparing two metric values."""
    is_better: bool
    improvement: float
    improvement_pct: float
    is_significant: bool
    epsilon_used: float


class RatchetMetric(autoconstitutionPlugin):
    """
    Interface for ratchet metrics that determine keep/discard decisions.
    
    The ratchet mechanism ensures monotonic improvement:
    - Changes are only kept if they improve the metric
    - No regressions are ever accepted
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.__class__.__name__,
            version="1.0.0",
            description=self.__doc__ or "No description",
            author="unknown",
            plugin_type=PluginType.RATCHET_METRIC,
            dependencies=[],
            config_schema=self._get_config_schema()
        )
    
    @abstractmethod
    def compute(self, evaluation_data: Dict[str, Any]) -> MetricValue:
        """Compute metric value from evaluation data."""
        pass
    
    @abstractmethod
    def compare(
        self, 
        current: MetricValue, 
        baseline: MetricValue,
        epsilon: float = 0.0
    ) -> ComparisonResult:
        """Compare two metric values."""
        pass
    
    @abstractmethod
    def get_direction(self) -> OptimizationDirection:
        """Return whether this metric should be minimized or maximized."""
        pass
    
    @abstractmethod
    def get_units(self) -> str:
        """Return human-readable units."""
        pass
    
    def format_value(self, value: float) -> str:
        """Format metric value for display."""
        return f"{value:.4f} {self.get_units()}"
    
    def validate_data(self, evaluation_data: Dict[str, Any]) -> List[str]:
        """Validate that evaluation data contains required fields."""
        return []
    
    def _get_config_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}
```

### 2.3 LLM Provider Interface

```python
# autoconstitution/plugins/interfaces/providers.py

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, AsyncIterator
from enum import Enum, auto
import time

from .base import autoconstitutionPlugin, PluginMetadata, PluginType


class ProviderCapability(Enum):
    """Capabilities that providers may support."""
    STREAMING = auto()
    FUNCTION_CALLING = auto()
    VISION = auto()
    EMBEDDINGS = auto()
    JSON_MODE = auto()
    SYSTEM_PROMPT = auto()
    TOOL_USE = auto()


@dataclass
class Message:
    """A chat message."""
    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    stop_sequences: List[str] = field(default_factory=list)
    extra_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResult:
    """Result of a completion request."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """A chunk of a streaming response."""
    content: str
    is_finished: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Result of an embedding request."""
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, int]
    latency_ms: float


@dataclass
class ProviderCapabilities:
    """Capabilities and limits of a provider."""
    supported_models: List[str]
    capabilities: List[ProviderCapability]
    max_context_length: int
    max_output_tokens: int
    rate_limits: Dict[str, Any]
    cost_per_1k_tokens: Dict[str, float]


class LLMProvider(autoconstitutionPlugin):
    """
    Interface for LLM providers.
    
    All LLM providers (Kimi, Claude, OpenAI, Ollama, vLLM, etc.)
    implement this unified interface.
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.__class__.__name__,
            version="1.0.0",
            description=self.__doc__ or "LLM Provider",
            author="unknown",
            plugin_type=PluginType.LLM_PROVIDER,
            dependencies=[],
            config_schema=self._get_config_schema()
        )
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None
    ) -> CompletionResult:
        """Generate a completion from message history."""
        pass
    
    @abstractmethod
    async def stream_complete(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion tokens as they're generated."""
        pass
    
    @abstractmethod
    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> EmbeddingResult:
        """Generate embeddings for texts."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities and limits."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check provider availability and health."""
        pass
    
    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text (approximate if not available)."""
        return len(text) // 4
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: Optional[str] = None
    ) -> float:
        """Estimate cost for a request in USD."""
        caps = self.get_capabilities()
        rates = caps.cost_per_1k_tokens
        
        input_cost = (prompt_tokens / 1000) * rates.get('input', 0)
        output_cost = (completion_tokens / 1000) * rates.get('output', 0)
        
        return input_cost + output_cost
    
    def supports(self, capability: ProviderCapability) -> bool:
        """Check if provider supports a capability."""
        return capability in self.get_capabilities().capabilities
    
    def _get_config_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "api_key": {"type": "string"},
                "base_url": {"type": "string"},
                "default_model": {"type": "string"},
                "timeout": {"type": "number", "default": 120},
                "max_retries": {"type": "integer", "default": 3}
            },
            "required": []
        }
```

### 2.4 Training Target Interface

```python
# autoconstitution/plugins/interfaces/training.py

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, AsyncIterator, Callable
from enum import Enum, auto
from pathlib import Path

from .base import autoconstitutionPlugin, PluginMetadata, PluginType
from .metrics import RatchetMetric, MetricValue


class TrainingPhase(Enum):
    """Phases of training."""
    INITIALIZING = auto()
    TRAINING = auto()
    VALIDATING = auto()
    CHECKPOINTING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class TrainingConfig:
    """Configuration for training run."""
    model_name_or_path: str
    model_config: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_steps: Optional[int] = None
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    data_config: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = "./outputs"
    save_steps: int = 500
    eval_steps: int = 100
    device: str = "auto"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    extra_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingState:
    """Current state of training."""
    phase: TrainingPhase
    current_step: int
    current_epoch: int
    global_step: int
    train_loss: float
    learning_rate: float
    start_time: float
    elapsed_seconds: float
    estimated_remaining_seconds: Optional[float] = None
    memory_used_gb: float
    memory_allocated_gb: float
    last_checkpoint_path: Optional[str] = None


@dataclass
class TrainingResult:
    """Result of a training run."""
    success: bool
    final_checkpoint_path: Optional[str]
    final_metrics: Dict[str, MetricValue]
    total_steps: int
    total_time_seconds: float
    error_message: Optional[str] = None
    primary_metric_value: Optional[MetricValue] = None


@dataclass
class TrainingProgress:
    """Progress update during training."""
    state: TrainingState
    metrics: Dict[str, float]
    logs: List[str]


class TrainingTarget(autoconstitutionPlugin):
    """
    Interface for training targets.
    
    Training targets define WHAT is being trained and HOW.
    This enables autoconstitution to optimize different types of models
    beyond just language models.
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.__class__.__name__,
            version="1.0.0",
            description=self.__doc__ or "Training Target",
            author="unknown",
            plugin_type=PluginType.TRAINING_TARGET,
            dependencies=[],
            config_schema=self._get_config_schema()
        )
    
    @abstractmethod
    async def train(
        self,
        config: TrainingConfig,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None
    ) -> TrainingResult:
        """Execute training run."""
        pass
    
    @abstractmethod
    async def stream_train(
        self,
        config: TrainingConfig
    ) -> AsyncIterator[TrainingProgress]:
        """Stream training progress updates."""
        pass
    
    @abstractmethod
    def get_primary_metric(self) -> RatchetMetric:
        """Return the primary metric for ratchet evaluation."""
        pass
    
    @abstractmethod
    async def evaluate(
        self,
        checkpoint_path: str,
        eval_config: Dict[str, Any]
    ) -> Dict[str, MetricValue]:
        """Evaluate a checkpoint."""
        pass
    
    @abstractmethod
    def get_default_config(self) -> TrainingConfig:
        """Return default training configuration."""
        pass
    
    async def resume_from_checkpoint(
        self,
        checkpoint_path: str,
        config: TrainingConfig,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None
    ) -> TrainingResult:
        """Resume training from checkpoint."""
        return await self.train(config, progress_callback)
    
    async def validate_config(self, config: TrainingConfig) -> List[str]:
        """Validate training configuration."""
        errors = []
        if config.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if config.learning_rate <= 0:
            errors.append("learning_rate must be > 0")
        return errors
    
    def estimate_resource_requirements(
        self,
        config: TrainingConfig
    ) -> Dict[str, Any]:
        """Estimate resource requirements for training."""
        return {
            "memory_gb": 16.0,
            "compute_hours": 1.0,
            "storage_gb": 1.0
        }
    
    def _get_config_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model_name_or_path": {"type": "string"},
                "batch_size": {"type": "integer", "default": 32},
                "learning_rate": {"type": "number", "default": 1e-4},
                "num_epochs": {"type": "integer", "default": 3},
                "output_dir": {"type": "string", "default": "./outputs"}
            },
            "required": ["model_name_or_path"]
        }
```

### 2.5 Experiment Runner Interface

```python
# autoconstitution/plugins/interfaces/runners.py

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, AsyncIterator
from enum import Enum, auto
from pathlib import Path

from .base import autoconstitutionPlugin, PluginMetadata, PluginType
from .training import TrainingConfig, TrainingResult


class RunnerCapability(Enum):
    """Capabilities that runners may support."""
    LOCAL_EXECUTION = auto()
    DISTRIBUTED = auto()
    GPU = auto()
    CHECKPOINTING = auto()
    PREEMPTION = auto()
    AUTO_SCALING = auto()


@dataclass
class HardwareSpec:
    """Hardware specification."""
    cpu_cores: int
    memory_gb: float
    gpu_count: int
    gpu_type: Optional[str] = None
    gpu_memory_gb: Optional[float] = None


@dataclass
class ExecutionEnvironment:
    """Execution environment configuration."""
    python_version: str = "3.11"
    cuda_version: Optional[str] = None
    pip_packages: List[str] = field(default_factory=list)
    conda_packages: List[str] = field(default_factory=list)
    system_packages: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    experiment_id: str
    experiment_name: str
    training_config: TrainingConfig
    hardware: HardwareSpec
    environment: ExecutionEnvironment = field(default_factory=ExecutionEnvironment)
    timeout_seconds: Optional[int] = None
    max_retries: int = 0
    output_bucket: Optional[str] = None
    checkpoint_bucket: Optional[str] = None
    extra_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentStatus:
    """Status of an experiment."""
    experiment_id: str
    state: str
    progress_percent: float
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    allocated_hardware: Optional[HardwareSpec] = None
    actual_hardware: Optional[HardwareSpec] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[TrainingResult] = None
    error_message: Optional[str] = None
    log_url: Optional[str] = None
    metrics_url: Optional[str] = None


@dataclass
class ResourceAllocation:
    """Allocated resources for an experiment."""
    allocation_id: str
    hardware: HardwareSpec
    estimated_cost_per_hour: float
    estimated_duration_hours: float


class ExperimentRunner(autoconstitutionPlugin):
    """
    Interface for experiment runners.
    
    Experiment runners handle WHERE and HOW experiments execute.
    They abstract away hardware and infrastructure details.
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.__class__.__name__,
            version="1.0.0",
            description=self.__doc__ or "Experiment Runner",
            author="unknown",
            plugin_type=PluginType.EXPERIMENT_RUNNER,
            dependencies=[],
            config_schema=self._get_config_schema()
        )
    
    @abstractmethod
    async def submit(self, config: ExperimentConfig) -> str:
        """Submit an experiment for execution."""
        pass
    
    @abstractmethod
    async def get_status(self, experiment_id: str) -> ExperimentStatus:
        """Get current status of an experiment."""
        pass
    
    @abstractmethod
    async def stream_logs(
        self,
        experiment_id: str,
        follow: bool = True
    ) -> AsyncIterator[str]:
        """Stream experiment logs."""
        pass
    
    @abstractmethod
    async def cancel(self, experiment_id: str) -> bool:
        """Cancel a running experiment."""
        pass
    
    @abstractmethod
    async def get_result(self, experiment_id: str) -> Optional[TrainingResult]:
        """Get final result of a completed experiment."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[RunnerCapability]:
        """Return runner capabilities."""
        pass
    
    @abstractmethod
    async def estimate_resources(
        self,
        training_config: TrainingConfig
    ) -> ResourceAllocation:
        """Estimate resources needed for training."""
        pass
    
    async def list_experiments(
        self,
        status_filter: Optional[List[str]] = None
    ) -> List[ExperimentStatus]:
        """List all experiments."""
        return []
    
    async def get_metrics(self, experiment_id: str) -> Dict[str, List[float]]:
        """Get time-series metrics for an experiment."""
        return {}
    
    async def download_artifacts(
        self,
        experiment_id: str,
        destination: str
    ) -> List[str]:
        """Download experiment artifacts."""
        return []
    
    def supports(self, capability: RunnerCapability) -> bool:
        """Check if runner supports a capability."""
        return capability in self.get_capabilities()
    
    def _get_config_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}
```

---

## 3. Plugin Discovery and Registration

### 3.1 Plugin Registry

```python
# autoconstitution/plugins/registry.py

import importlib
import importlib.metadata
import pkgutil
from pathlib import Path
from typing import Dict, List, Type, Optional, Callable
import logging
import sys

from .base import autoconstitutionPlugin, PluginMetadata, PluginType


logger = logging.getLogger(__name__)


class PluginRegistry:
    """Central registry for all plugins."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._plugins: Dict[str, autoconstitutionPlugin] = {}
        self._plugin_classes: Dict[PluginType, Dict[str, Type]] = {
            plugin_type: {} for plugin_type in PluginType
        }
        self._factories: Dict[str, Callable] = {}
        self._initialized = True
    
    def register(
        self,
        name: str,
        plugin_class: Type[autoconstitutionPlugin],
        factory: Optional[Callable] = None
    ) -> None:
        """Register a plugin class."""
        if not issubclass(plugin_class, autoconstitutionPlugin):
            raise ValueError("Plugin must inherit from autoconstitutionPlugin")
        
        temp_instance = plugin_class()
        plugin_type = temp_instance.metadata.plugin_type
        
        self._plugin_classes[plugin_type][name] = plugin_class
        if factory:
            self._factories[name] = factory
        
        logger.info(f"Registered {plugin_type.name} plugin: {name}")
    
    def register_instance(self, name: str, plugin: autoconstitutionPlugin) -> None:
        """Register a pre-configured plugin instance."""
        self._plugins[name] = plugin
        logger.info(f"Registered plugin instance: {name}")
    
    def discover_from_entry_points(self, group: Optional[str] = None) -> int:
        """Discover plugins from package entry points."""
        count = 0
        
        if group is None:
            groups = [
                "autoconstitution.metrics",
                "autoconstitution.providers",
                "autoconstitution.training",
                "autoconstitution.runners",
            ]
        else:
            groups = [group]
        
        for grp in groups:
            try:
                eps = importlib.metadata.entry_points(group=grp)
                for ep in eps:
                    try:
                        plugin_class = ep.load()
                        self.register(ep.name, plugin_class)
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to load entry point {ep.name}: {e}")
            except Exception as e:
                logger.warning(f"No entry points found for group {grp}: {e}")
        
        return count
    
    def discover_from_directory(self, directory: str) -> int:
        """Discover plugins from a directory."""
        count = 0
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Plugin directory does not exist: {directory}")
            return 0
        
        if str(dir_path) not in sys.path:
            sys.path.insert(0, str(dir_path))
        
        for finder, name, ispkg in pkgutil.iter_modules([str(dir_path)]):
            try:
                module = importlib.import_module(name)
                
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, autoconstitutionPlugin) and
                        attr is not autoconstitutionPlugin):
                        
                        self.register(attr_name, attr)
                        count += 1
                        
            except Exception as e:
                logger.error(f"Failed to import module {name}: {e}")
        
        return count
    
    def discover_all(self, plugin_dirs: Optional[List[str]] = None) -> int:
        """Discover plugins from all sources."""
        count = 0
        
        count += self.discover_from_entry_points()
        
        default_dirs = [
            "./plugins",
            "~/.autoconstitution/plugins",
            "/usr/local/share/autoconstitution/plugins",
        ]
        
        for dir_path in default_dirs:
            expanded = Path(dir_path).expanduser()
            count += self.discover_from_directory(str(expanded))
        
        if plugin_dirs:
            for dir_path in plugin_dirs:
                count += self.discover_from_directory(dir_path)
        
        return count
    
    def get(
        self,
        name: str,
        plugin_type: Optional[PluginType] = None,
        config: Optional[dict] = None
    ) -> autoconstitutionPlugin:
        """Get a plugin instance."""
        if name in self._plugins:
            return self._plugins[name]
        
        plugin_class = None
        
        if plugin_type:
            plugin_class = self._plugin_classes[plugin_type].get(name)
        else:
            for pt in PluginType:
                if name in self._plugin_classes[pt]:
                    plugin_class = self._plugin_classes[pt][name]
                    break
        
        if plugin_class is None:
            available = self.list_available()
            raise ValueError(f"Plugin not found: {name}. Available: {available}")
        
        if name in self._factories:
            instance = self._factories[name](config)
        else:
            instance = plugin_class()
        
        if config:
            import asyncio
            asyncio.create_task(instance.initialize(config))
        
        return instance
    
    def get_by_type(self, plugin_type: PluginType) -> Dict[str, Type]:
        """Get all plugins of a specific type."""
        return self._plugin_classes[plugin_type].copy()
    
    def list_available(self, plugin_type: Optional[PluginType] = None) -> List[str]:
        """List all available plugin names."""
        if plugin_type:
            return list(self._plugin_classes[plugin_type].keys())
        
        all_names = []
        for pt in PluginType:
            all_names.extend(self._plugin_classes[pt].keys())
        return all_names
    
    def get_metadata(self, name: str) -> Optional[PluginMetadata]:
        """Get metadata for a plugin."""
        try:
            plugin = self.get(name)
            return plugin.metadata
        except ValueError:
            return None
    
    async def initialize_all(self, configs: Dict[str, dict]) -> None:
        """Initialize all registered plugins."""
        for name, plugin in self._plugins.items():
            config = configs.get(name, {})
            await plugin.initialize(config)
    
    async def shutdown_all(self) -> None:
        """Shutdown all plugins."""
        for name, plugin in self._plugins.items():
            try:
                await plugin.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down plugin {name}: {e}")
    
    async def health_check_all(self) -> Dict[str, dict]:
        """Run health checks on all plugins."""
        results = {}
        for name, plugin in self._plugins.items():
            try:
                health = await plugin.health_check()
                results[name] = {
                    "healthy": health.healthy,
                    "error": health.error_message
                }
            except Exception as e:
                results[name] = {"healthy": False, "error": str(e)}
        return results


# Global registry instance
registry = PluginRegistry()


def register_plugin(name: str):
    """Decorator to register a plugin class."""
    def decorator(cls):
        registry.register(name, cls)
        return cls
    return decorator
```

### 3.2 Entry Points Configuration

```toml
# pyproject.toml example for a plugin package:

[project]
name = "autoconstitution-metrics-extra"
version = "1.0.0"

[project.entry-points."autoconstitution.metrics"]
perplexity = "autoconstitution_metrics_extra:PerplexityMetric"
bleu = "autoconstitution_metrics_extra:BleuMetric"
rouge = "autoconstitution_metrics_extra:RougeMetric"

[project.entry-points."autoconstitution.providers"]
groq = "autoconstitution_providers_extra:GroqProvider"
cohere = "autoconstitution_providers_extra:CohereProvider"

[project.entry-points."autoconstitution.training"]
rlhf = "autoconstitution_training_extra:RLHFTarget"
dpo = "autoconstitution_training_extra:DPOTarget"
```

---

## 4. Plugin Lifecycle Management

### 4.1 Lifecycle Manager

```python
# autoconstitution/plugins/lifecycle.py

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum, auto
import logging
import time

from .base import autoconstitutionPlugin, PluginHealth
from .registry import registry


logger = logging.getLogger(__name__)


class PluginState(Enum):
    """Lifecycle states for plugins."""
    UNLOADED = auto()
    LOADING = auto()
    INITIALIZING = auto()
    ACTIVE = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    SHUTTING_DOWN = auto()
    ERROR = auto()


@dataclass
class PluginInstance:
    """Managed plugin instance with lifecycle state."""
    name: str
    plugin: autoconstitutionPlugin
    state: PluginState
    config: Dict
    load_time: Optional[float] = None
    init_time: Optional[float] = None
    error_count: int = 0
    last_error: Optional[str] = None


class PluginLifecycleManager:
    """Manages the complete lifecycle of plugins."""
    
    def __init__(
        self,
        health_check_interval: float = 30.0,
        max_errors: int = 5,
        auto_restart: bool = True
    ):
        self.health_check_interval = health_check_interval
        self.max_errors = max_errors
        self.auto_restart = auto_restart
        
        self._instances: Dict[str, PluginInstance] = {}
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def load(
        self,
        name: str,
        config: Optional[Dict] = None,
        plugin_type: Optional[str] = None
    ) -> PluginInstance:
        """Load and initialize a plugin."""
        if name in self._instances:
            logger.warning(f"Plugin {name} already loaded")
            return self._instances[name]
        
        instance = PluginInstance(
            name=name,
            plugin=None,
            state=PluginState.LOADING,
            config=config or {},
            load_time=time.time()
        )
        self._instances[name] = instance
        
        try:
            plugin = registry.get(name, config=config)
            instance.plugin = plugin
            
            instance.state = PluginState.INITIALIZING
            await plugin.initialize(config or {})
            instance.init_time = time.time()
            
            health = await plugin.health_check()
            if health.healthy:
                instance.state = PluginState.ACTIVE
                logger.info(f"Plugin {name} loaded and active")
            else:
                instance.state = PluginState.DEGRADED
                instance.last_error = health.error_message
                logger.warning(f"Plugin {name} degraded: {health.error_message}")
            
            return instance
            
        except Exception as e:
            instance.state = PluginState.ERROR
            instance.last_error = str(e)
            logger.error(f"Failed to load plugin {name}: {e}")
            raise
    
    async def load_multiple(
        self,
        configs: Dict[str, Dict]
    ) -> Dict[str, PluginInstance]:
        """Load multiple plugins concurrently."""
        tasks = [
            self.load(name, config)
            for name, config in configs.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        loaded = {}
        for name, result in zip(configs.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to load {name}: {result}")
            else:
                loaded[name] = result
        
        return loaded
    
    async def start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._health_check_task is not None:
            return
        
        self._health_check_task = asyncio.create_task(
            self._health_monitoring_loop()
        )
        logger.info("Health monitoring started")
    
    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("Health monitoring stopped")
    
    async def _health_monitoring_loop(self) -> None:
        """Background loop for health checks."""
        while not self._shutdown_event.is_set():
            try:
                await self._check_all_health()
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.health_check_interval
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    async def _check_all_health(self) -> None:
        """Check health of all active plugins."""
        for name, instance in self._instances.items():
            if instance.state not in [PluginState.ACTIVE, PluginState.DEGRADED]:
                continue
            
            try:
                health = await instance.plugin.health_check()
                
                if health.healthy:
                    if instance.state == PluginState.DEGRADED:
                        instance.state = PluginState.ACTIVE
                        logger.info(f"Plugin {name} recovered")
                else:
                    instance.state = PluginState.DEGRADED
                    instance.last_error = health.error_message
                    logger.warning(f"Plugin {name} degraded: {health.error_message}")
                    
                    instance.error_count += 1
                    if instance.error_count >= self.max_errors and self.auto_restart:
                        logger.warning(f"Restarting plugin {name}")
                        await self.restart(name)
                        
            except Exception as e:
                instance.state = PluginState.UNHEALTHY
                instance.last_error = str(e)
                logger.error(f"Health check failed for {name}: {e}")
    
    async def unload(self, name: str) -> None:
        """Unload a plugin."""
        if name not in self._instances:
            return
        
        instance = self._instances[name]
        instance.state = PluginState.SHUTTING_DOWN
        
        try:
            await instance.plugin.shutdown()
            logger.info(f"Plugin {name} unloaded")
        except Exception as e:
            logger.error(f"Error unloading plugin {name}: {e}")
        finally:
            del self._instances[name]
    
    async def unload_all(self) -> None:
        """Unload all plugins."""
        self._shutdown_event.set()
        await self.stop_health_monitoring()
        
        for name in list(self._instances.keys()):
            await self.unload(name)
    
    async def restart(self, name: str) -> PluginInstance:
        """Restart a plugin."""
        if name not in self._instances:
            raise ValueError(f"Plugin {name} not loaded")
        
        config = self._instances[name].config
        await self.unload(name)
        return await self.load(name, config)
    
    def get_status(self, name: Optional[str] = None) -> Dict:
        """Get plugin status."""
        if name:
            if name not in self._instances:
                return {"error": f"Plugin {name} not found"}
            
            instance = self._instances[name]
            return {
                "name": name,
                "state": instance.state.name,
                "error_count": instance.error_count,
                "last_error": instance.last_error,
                "load_time": instance.load_time,
                "init_time": instance.init_time
            }
        
        return {
            name: self.get_status(name)
            for name in self._instances.keys()
        }
```

---

## 5. Example Plugin Implementations

### 5.1 Bits Per Byte Metric

```python
# autoconstitution/plugins/builtin/metrics.py

import numpy as np
import time
from typing import Dict, Any, List

from ..interfaces.metrics import (
    RatchetMetric, MetricValue, ComparisonResult, 
    OptimizationDirection
)
from ..registry import register_plugin


@register_plugin("bits_per_byte")
class BitsPerByteMetric(RatchetMetric):
    """
    Karpathy's val_bpb metric - bits per byte.
    
    Measures compression efficiency of a language model.
    Lower is better (fewer bits needed per byte of text).
    
    Formula: bpb = total_nats / (ln(2) * total_bytes)
    """
    
    def compute(self, evaluation_data: Dict[str, Any]) -> MetricValue:
        """Compute bits per byte from evaluation data."""
        total_nats = evaluation_data.get('total_nats')
        total_bytes = evaluation_data.get('total_bytes')
        
        if total_nats is None or total_bytes is None:
            raise ValueError("evaluation_data must contain 'total_nats' and 'total_bytes'")
        
        if total_bytes == 0:
            raise ValueError("total_bytes cannot be zero")
        
        bpb = total_nats / (np.log(2) * total_bytes)
        
        return MetricValue(
            value=bpb,
            timestamp=time.time(),
            metadata={
                'total_nats': total_nats,
                'total_bytes': total_bytes,
                'interpretation': f'{bpb:.4f} bits per byte',
                'compression_ratio': 8.0 / bpb if bpb > 0 else float('inf')
            },
            raw_data=evaluation_data
        )
    
    def compare(
        self,
        current: MetricValue,
        baseline: MetricValue,
        epsilon: float = 0.0
    ) -> ComparisonResult:
        """Compare two BPB values (lower is better)."""
        improvement = baseline.value - current.value
        improvement_pct = (improvement / baseline.value * 100) if baseline.value != 0 else 0
        
        return ComparisonResult(
            is_better=current.value < baseline.value - epsilon,
            improvement=improvement,
            improvement_pct=improvement_pct,
            is_significant=abs(improvement) > epsilon,
            epsilon_used=epsilon
        )
    
    def get_direction(self) -> OptimizationDirection:
        return OptimizationDirection.MINIMIZE
    
    def get_units(self) -> str:
        return "bits/byte"
    
    def format_value(self, value: float) -> str:
        return f"{value:.4f} bpb"
    
    def validate_data(self, evaluation_data: Dict[str, Any]) -> List[str]:
        errors = []
        
        if 'total_nats' not in evaluation_data:
            errors.append("Missing required field: total_nats")
        elif not isinstance(evaluation_data['total_nats'], (int, float)):
            errors.append("total_nats must be numeric")
        
        if 'total_bytes' not in evaluation_data:
            errors.append("Missing required field: total_bytes")
        elif not isinstance(evaluation_data['total_bytes'], (int, float)):
            errors.append("total_bytes must be numeric")
        elif evaluation_data.get('total_bytes', 0) <= 0:
            errors.append("total_bytes must be positive")
        
        return errors


@register_plugin("inference_speed")
class InferenceSpeedMetric(RatchetMetric):
    """
    Inference speed metric - tokens per second.
    Higher is better (more tokens generated per second).
    """
    
    def compute(self, evaluation_data: Dict[str, Any]) -> MetricValue:
        """Compute tokens per second."""
        tokens_generated = evaluation_data.get('tokens_generated', 0)
        generation_time_sec = evaluation_data.get('generation_time_sec', 0)
        
        if generation_time_sec <= 0:
            tps = 0.0
        else:
            tps = tokens_generated / generation_time_sec
        
        return MetricValue(
            value=tps,
            timestamp=time.time(),
            metadata={
                'tokens_generated': tokens_generated,
                'generation_time_sec': generation_time_sec,
                'interpretation': f'{tps:.2f} tokens/second'
            },
            raw_data=evaluation_data
        )
    
    def compare(
        self,
        current: MetricValue,
        baseline: MetricValue,
        epsilon: float = 0.0
    ) -> ComparisonResult:
        """Compare two TPS values (higher is better)."""
        improvement = current.value - baseline.value
        improvement_pct = (improvement / baseline.value * 100) if baseline.value != 0 else 0
        
        return ComparisonResult(
            is_better=current.value > baseline.value + epsilon,
            improvement=improvement,
            improvement_pct=improvement_pct,
            is_significant=abs(improvement) > epsilon,
            epsilon_used=epsilon
        )
    
    def get_direction(self) -> OptimizationDirection:
        return OptimizationDirection.MAXIMIZE
    
    def get_units(self) -> str:
        return "tokens/sec"
    
    def format_value(self, value: float) -> str:
        return f"{value:.1f} tok/s"
```

### 5.2 Kimi Provider

```python
# autoconstitution/plugins/builtin/providers.py

import httpx
import time
from typing import Dict, Any, Optional, List, AsyncIterator

from ..interfaces.providers import (
    LLMProvider, Message, GenerationConfig, CompletionResult,
    StreamChunk, EmbeddingResult, ProviderCapabilities, ProviderCapability
)
from ..registry import register_plugin


@register_plugin("kimi")
class KimiProvider(LLMProvider):
    """Moonshot AI Kimi K2.5 provider."""
    
    DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
    DEFAULT_MODEL = "kimi-k2.5"
    
    def __init__(self):
        self.api_key: Optional[str] = None
        self.base_url: str = self.DEFAULT_BASE_URL
        self.default_model: str = self.DEFAULT_MODEL
        self.timeout: float = 120.0
        self.client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url', self.DEFAULT_BASE_URL)
        self.default_model = config.get('default_model', self.DEFAULT_MODEL)
        self.timeout = config.get('timeout', 120.0)
        
        if not self.api_key:
            raise ValueError("Kimi provider requires 'api_key' in config")
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout
        )
    
    async def shutdown(self) -> None:
        """Clean up resources."""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        try:
            response = await self.client.get("/models")
            if response.status_code == 200:
                return {"healthy": True}
            else:
                return {
                    "healthy": False,
                    "error": f"API returned status {response.status_code}"
                }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def complete(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None
    ) -> CompletionResult:
        """Generate completion."""
        config = config or GenerationConfig()
        start_time = time.time()
        
        request = {
            "model": config.model or self.default_model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "stop": config.stop_sequences if config.stop_sequences else None
        }
        
        request.update(config.extra_options)
        
        response = await self.client.post("/chat/completions", json=request)
        response.raise_for_status()
        
        data = response.json()
        latency_ms = (time.time() - start_time) * 1000
        
        choice = data['choices'][0]
        
        return CompletionResult(
            content=choice['message']['content'],
            model=data['model'],
            usage=data.get('usage', {}),
            finish_reason=choice.get('finish_reason', 'unknown'),
            latency_ms=latency_ms,
            metadata={
                'created': data.get('created'),
                'system_fingerprint': data.get('system_fingerprint')
            }
        )
    
    async def stream_complete(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion tokens."""
        config = config or GenerationConfig()
        
        request = {
            "model": config.model or self.default_model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": True
        }
        
        async with self.client.stream(
            "POST", "/chat/completions",
            json=request
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        yield StreamChunk(content="", is_finished=True)
                        break
                    
                    try:
                        import json
                        chunk = json.loads(data)
                        delta = chunk['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        
                        yield StreamChunk(
                            content=content,
                            is_finished=chunk['choices'][0].get('finish_reason') is not None
                        )
                    except Exception:
                        continue
    
    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> EmbeddingResult:
        """Generate embeddings."""
        start_time = time.time()
        
        request = {
            "model": model or "kimi-embedding",
            "input": texts
        }
        
        response = await self.client.post("/embeddings", json=request)
        response.raise_for_status()
        
        data = response.json()
        latency_ms = (time.time() - start_time) * 1000
        
        embeddings = [item['embedding'] for item in data['data']]
        
        return EmbeddingResult(
            embeddings=embeddings,
            model=data['model'],
            usage=data.get('usage', {}),
            latency_ms=latency_ms
        )
    
    def get_capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities."""
        return ProviderCapabilities(
            supported_models=[
                "kimi-k2.5",
                "kimi-k2",
                "kimi-k1.5",
                "kimi-embedding"
            ],
            capabilities=[
                ProviderCapability.STREAMING,
                ProviderCapability.FUNCTION_CALLING,
                ProviderCapability.EMBEDDINGS,
                ProviderCapability.JSON_MODE,
                ProviderCapability.SYSTEM_PROMPT
            ],
            max_context_length=256000,
            max_output_tokens=8192,
            rate_limits={
                "requests_per_minute": 60,
                "tokens_per_minute": 100000
            },
            cost_per_1k_tokens={
                "input": 0.015,
                "output": 0.06
            }
        )


@register_plugin("ollama")
class OllamaProvider(LLMProvider):
    """Ollama local model provider."""
    
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "mistral"
    
    def __init__(self):
        self.base_url: str = self.DEFAULT_BASE_URL
        self.default_model: str = self.DEFAULT_MODEL
        self.timeout: float = 300.0
        self.client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.base_url = config.get('base_url', self.DEFAULT_BASE_URL)
        self.default_model = config.get('default_model', self.DEFAULT_MODEL)
        self.timeout = config.get('timeout', 300.0)
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    async def shutdown(self) -> None:
        """Clean up resources."""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        try:
            response = await self.client.get("/api/tags")
            if response.status_code == 200:
                return {"healthy": True}
            else:
                return {
                    "healthy": False,
                    "error": f"Ollama returned status {response.status_code}"
                }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def complete(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None
    ) -> CompletionResult:
        """Generate completion."""
        config = config or GenerationConfig()
        start_time = time.time()
        
        prompt = self._messages_to_prompt(messages)
        
        request = {
            "model": config.model or self.default_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
                "top_p": config.top_p,
                "stop": config.stop_sequences
            }
        }
        
        response = await self.client.post("/api/generate", json=request)
        response.raise_for_status()
        
        data = response.json()
        latency_ms = (time.time() - start_time) * 1000
        
        return CompletionResult(
            content=data['response'],
            model=data['model'],
            usage={
                "prompt_tokens": data.get('prompt_eval_count', 0),
                "completion_tokens": data.get('eval_count', 0),
                "total_tokens": (
                    data.get('prompt_eval_count', 0) + 
                    data.get('eval_count', 0)
                )
            },
            finish_reason="stop" if data.get('done') else "unknown",
            latency_ms=latency_ms,
            metadata={
                'total_duration_ms': data.get('total_duration', 0) / 1e6,
                'load_duration_ms': data.get('load_duration', 0) / 1e6
            }
        )
    
    async def stream_complete(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion tokens."""
        config = config or GenerationConfig()
        prompt = self._messages_to_prompt(messages)
        
        request = {
            "model": config.model or self.default_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens
            }
        }
        
        async with self.client.stream(
            "POST", "/api/generate",
            json=request
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line:
                    import json
                    try:
                        data = json.loads(line)
                        yield StreamChunk(
                            content=data.get('response', ''),
                            is_finished=data.get('done', False)
                        )
                    except json.JSONDecodeError:
                        continue
    
    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> EmbeddingResult:
        """Generate embeddings."""
        start_time = time.time()
        embeddings = []
        
        for text in texts:
            request = {
                "model": model or "nomic-embed-text",
                "prompt": text
            }
            
            response = await self.client.post("/api/embeddings", json=request)
            response.raise_for_status()
            
            data = response.json()
            embeddings.append(data['embedding'])
        
        latency_ms = (time.time() - start_time) * 1000
        
        return EmbeddingResult(
            embeddings=embeddings,
            model=model or "nomic-embed-text",
            usage={"total_tokens": sum(len(t.split()) for t in texts)},
            latency_ms=latency_ms
        )
    
    def get_capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities."""
        return ProviderCapabilities(
            supported_models=["mistral", "llama2", "codellama", "nomic-embed-text"],
            capabilities=[
                ProviderCapability.STREAMING,
                ProviderCapability.EMBEDDINGS,
                ProviderCapability.SYSTEM_PROMPT
            ],
            max_context_length=32768,
            max_output_tokens=4096,
            rate_limits={
                "requests_per_minute": float('inf'),
                "tokens_per_minute": float('inf')
            },
            cost_per_1k_tokens={
                "input": 0.0,
                "output": 0.0
            }
        )
    
    def _messages_to_prompt(self, messages: List[Message]) -> str:
        """Convert messages to Ollama prompt format."""
        prompt_parts = []
        
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        return "\n\n".join(prompt_parts)
```

### 5.3 Language Model Training Target

```python
# autoconstitution/plugins/builtin/training.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Callable, AsyncIterator
import time

from ..interfaces.training import (
    TrainingTarget, TrainingConfig, TrainingState, TrainingResult,
    TrainingProgress, TrainingPhase
)
from ..interfaces.metrics import RatchetMetric
from ..builtin.metrics import BitsPerByteMetric
from ..registry import register_plugin


@register_plugin("language_model")
class LanguageModelTarget(TrainingTarget):
    """
    Standard language model training target.
    
    Implements nanoGPT-style training loop with:
    - AdamW optimizer
    - Learning rate scheduling
    - Gradient clipping
    - Mixed precision training
    - Checkpointing
    """
    
    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.device: torch.device = None
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize training components."""
        device_str = config.get('device', 'auto')
        if device_str == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else
                'mps' if torch.backends.mps.is_available() else
                'cpu'
            )
        else:
            self.device = torch.device(device_str)
    
    async def shutdown(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check training target health."""
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return {"healthy": True}
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def train(
        self,
        config: TrainingConfig,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None
    ) -> TrainingResult:
        """Execute training run."""
        start_time = time.time()
        
        await self.initialize({
            'device': config.device,
            **config.extra_options
        })
        
        state = TrainingState(
            phase=TrainingPhase.TRAINING,
            current_step=0,
            current_epoch=0,
            global_step=0,
            train_loss=0.0,
            learning_rate=config.learning_rate,
            start_time=start_time,
            elapsed_seconds=0.0,
            memory_used_gb=0.0,
            memory_allocated_gb=0.0
        )
        
        try:
            max_steps = config.max_steps or 1000
            
            for step in range(max_steps):
                loss = await self._training_step(config)
                
                state.current_step = step
                state.global_step = step
                state.train_loss = loss
                state.elapsed_seconds = time.time() - start_time
                
                if progress_callback:
                    progress = TrainingProgress(
                        state=state,
                        metrics={'loss': loss, 'lr': config.learning_rate},
                        logs=[f"Step {step}: loss={loss:.4f}"]
                    )
                    progress_callback(progress)
                
                if step % config.save_steps == 0:
                    await self._save_checkpoint(config, step)
                
                if step % config.eval_steps == 0:
                    val_metrics = await self._validate(config)
            
            final_metrics = await self.evaluate(
                state.last_checkpoint_path or "",
                {}
            )
            
            return TrainingResult(
                success=True,
                final_checkpoint_path=state.last_checkpoint_path,
                final_metrics=final_metrics,
                total_steps=max_steps,
                total_time_seconds=time.time() - start_time,
                primary_metric_value=final_metrics.get('val_bpb')
            )
            
        except Exception as e:
            return TrainingResult(
                success=False,
                final_checkpoint_path=None,
                final_metrics={},
                total_steps=state.current_step,
                total_time_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    async def stream_train(
        self,
        config: TrainingConfig
    ) -> AsyncIterator[TrainingProgress]:
        """Stream training progress."""
        progress_queue = []
        
        def callback(progress: TrainingProgress):
            progress_queue.append(progress)
        
        import asyncio
        train_task = asyncio.create_task(self.train(config, callback))
        
        while not train_task.done():
            while progress_queue:
                yield progress_queue.pop(0)
            await asyncio.sleep(0.1)
        
        result = await train_task
        
        if result.success:
            yield TrainingProgress(
                state=TrainingState(
                    phase=TrainingPhase.COMPLETED,
                    current_step=result.total_steps,
                    current_epoch=0,
                    global_step=result.total_steps,
                    train_loss=0.0,
                    learning_rate=0.0,
                    start_time=time.time() - result.total_time_seconds,
                    elapsed_seconds=result.total_time_seconds,
                    memory_used_gb=0.0,
                    memory_allocated_gb=0.0
                ),
                metrics={k: v.value for k, v in result.final_metrics.items()},
                logs=["Training completed successfully"]
            )
    
    def get_primary_metric(self) -> RatchetMetric:
        """Return primary metric for ratchet evaluation."""
        return BitsPerByteMetric()
    
    async def evaluate(
        self,
        checkpoint_path: str,
        eval_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a checkpoint."""
        return {
            'val_bpb': type('obj', (object,), {
                'value': 2.5,
                'metadata': {}
            })(),
            'val_loss': type('obj', (object,), {
                'value': 1.7,
                'metadata': {}
            })()
        }
    
    def get_default_config(self) -> TrainingConfig:
        """Return default training configuration."""
        return TrainingConfig(
            model_name_or_path="gpt2",
            batch_size=32,
            learning_rate=1e-4,
            num_epochs=3,
            output_dir="./outputs"
        )
    
    async def _training_step(self, config: TrainingConfig) -> float:
        """Execute one training step."""
        return 2.0 + torch.randn(1).item() * 0.1
    
    async def _validate(self, config: TrainingConfig) -> Dict[str, float]:
        """Run validation."""
        return {'val_loss': 1.8}
    
    async def _save_checkpoint(self, config: TrainingConfig, step: int) -> None:
        """Save checkpoint."""
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)


@register_plugin("rlhf")
class RLHFTarget(TrainingTarget):
    """RLHF (Reinforcement Learning from Human Feedback) training target."""
    
    async def train(
        self,
        config: TrainingConfig,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None
    ) -> TrainingResult:
        """Execute RLHF training."""
        raise NotImplementedError("RLHF training not yet implemented")
    
    async def stream_train(
        self,
        config: TrainingConfig
    ) -> AsyncIterator[TrainingProgress]:
        raise NotImplementedError()
    
    def get_primary_metric(self) -> RatchetMetric:
        from ..builtin.metrics import InferenceSpeedMetric
        return InferenceSpeedMetric()
    
    async def evaluate(
        self,
        checkpoint_path: str,
        eval_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError()
    
    def get_default_config(self) -> TrainingConfig:
        return TrainingConfig(
            model_name_or_path="gpt2",
            batch_size=4,
            learning_rate=1e-5,
            num_epochs=1,
            output_dir="./outputs/rlhf"
        )
```

### 5.4 Local Experiment Runner

```python
# autoconstitution/plugins/builtin/runners.py

import asyncio
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, AsyncIterator
import time
import os

from ..interfaces.runners import (
    ExperimentRunner, ExperimentConfig, ExperimentStatus,
    HardwareSpec, ExecutionEnvironment,
    ResourceAllocation, RunnerCapability
)
from ..interfaces.training import TrainingResult
from ..registry import register_plugin


@register_plugin("local")
class LocalRunner(ExperimentRunner):
    """Local machine experiment runner."""
    
    def __init__(self):
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.experiment_status: Dict[str, ExperimentStatus] = {}
        self.output_dir: Path = Path("./experiments")
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize runner."""
        self.output_dir = Path(config.get('output_dir', './experiments'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def shutdown(self) -> None:
        """Shutdown runner."""
        for exp_id in list(self.active_processes.keys()):
            await self.cancel(exp_id)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check runner health."""
        return {"healthy": True}
    
    async def submit(self, config: ExperimentConfig) -> str:
        """Submit experiment for local execution."""
        exp_id = config.experiment_id
        
        exp_dir = self.output_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'experiment_id': config.experiment_id,
                'experiment_name': config.experiment_name,
                'training_config': config.training_config.__dict__,
            }, f, indent=2, default=str)
        
        status = ExperimentStatus(
            experiment_id=exp_id,
            state='running',
            progress_percent=0.0,
            created_at=time.time(),
            started_at=time.time()
        )
        self.experiment_status[exp_id] = status
        
        process = await self._start_training(config, exp_dir)
        self.active_processes[exp_id] = process
        
        asyncio.create_task(self._monitor_experiment(exp_id, process))
        
        return exp_id
    
    async def _start_training(
        self,
        config: ExperimentConfig,
        exp_dir: Path
    ) -> subprocess.Popen:
        """Start training subprocess."""
        cmd = [
            "python", "-m", "autoconstitution.training",
            "--config", str(exp_dir / "config.json"),
            "--output-dir", str(exp_dir)
        ]
        
        env = os.environ.copy()
        env.update(config.environment.env_vars)
        
        process = subprocess.Popen(
            cmd,
            cwd=config.environment.working_dir,
            env=env,
            stdout=open(exp_dir / "stdout.log", 'w'),
            stderr=open(exp_dir / "stderr.log", 'w'),
        )
        
        return process
    
    async def _monitor_experiment(
        self,
        exp_id: str,
        process: subprocess.Popen
    ) -> None:
        """Monitor experiment process."""
        while process.poll() is None:
            await asyncio.sleep(1)
        
        status = self.experiment_status.get(exp_id)
        if status:
            if process.returncode == 0:
                status.state = 'completed'
                status.progress_percent = 100.0
            else:
                status.state = 'failed'
                status.error_message = f"Process exited with code {process.returncode}"
            status.completed_at = time.time()
        
        if exp_id in self.active_processes:
            del self.active_processes[exp_id]
    
    async def get_status(self, experiment_id: str) -> ExperimentStatus:
        """Get experiment status."""
        if experiment_id not in self.experiment_status:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        return self.experiment_status[experiment_id]
    
    async def stream_logs(
        self,
        experiment_id: str,
        follow: bool = True
    ) -> AsyncIterator[str]:
        """Stream experiment logs."""
        exp_dir = self.output_dir / experiment_id
        log_path = exp_dir / "stdout.log"
        
        if not log_path.exists():
            return
        
        with open(log_path, 'r') as f:
            if follow:
                while True:
                    line = f.readline()
                    if line:
                        yield line.rstrip()
                    else:
                        if experiment_id not in self.active_processes:
                            break
                        await asyncio.sleep(0.1)
            else:
                for line in f:
                    yield line.rstrip()
    
    async def cancel(self, experiment_id: str) -> bool:
        """Cancel experiment."""
        if experiment_id not in self.active_processes:
            return False
        
        process = self.active_processes[experiment_id]
        
        process.terminate()
        
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        
        status = self.experiment_status.get(experiment_id)
        if status:
            status.state = 'cancelled'
            status.completed_at = time.time()
        
        del self.active_processes[experiment_id]
        
        return True
    
    async def get_result(self, experiment_id: str) -> Optional[TrainingResult]:
        """Get experiment result."""
        exp_dir = self.output_dir / experiment_id
        result_path = exp_dir / "result.json"
        
        if not result_path.exists():
            return None
        
        with open(result_path, 'r') as f:
            data = json.load(f)
        
        return TrainingResult(**data)
    
    def get_capabilities(self) -> List[RunnerCapability]:
        """Return runner capabilities."""
        return [
            RunnerCapability.LOCAL_EXECUTION,
            RunnerCapability.CHECKPOINTING
        ]
    
    async def estimate_resources(
        self,
        training_config
    ) -> ResourceAllocation:
        """Estimate resource requirements."""
        return ResourceAllocation(
            allocation_id="local",
            hardware=HardwareSpec(
                cpu_cores=8,
                memory_gb=32.0,
                gpu_count=1,
                gpu_type="RTX 4090",
                gpu_memory_gb=24.0
            ),
            estimated_cost_per_hour=0.0,
            estimated_duration_hours=1.0
        )
```

---

## 6. Plugin Configuration

### 6.1 Configuration Schema

```python
# autoconstitution/plugins/config.py

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml
import json


@dataclass
class PluginConfig:
    """Configuration for a single plugin."""
    name: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0


@dataclass
class PluginSystemConfig:
    """Configuration for the entire plugin system."""
    
    auto_discover: bool = True
    plugin_directories: List[str] = field(default_factory=list)
    entry_point_groups: List[str] = field(default_factory=lambda: [
        "autoconstitution.metrics",
        "autoconstitution.providers",
        "autoconstitution.training",
        "autoconstitution.runners",
    ])
    
    metrics: Dict[str, PluginConfig] = field(default_factory=dict)
    providers: Dict[str, PluginConfig] = field(default_factory=dict)
    training: Dict[str, PluginConfig] = field(default_factory=dict)
    runners: Dict[str, PluginConfig] = field(default_factory=dict)
    
    health_check_interval: float = 30.0
    max_plugin_errors: int = 5
    auto_restart: bool = True
    
    default_metric: str = "bits_per_byte"
    default_provider: str = "kimi"
    default_training_target: str = "language_model"
    default_runner: str = "local"
    
    @classmethod
    def from_file(cls, path: str) -> 'PluginSystemConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginSystemConfig':
        """Load configuration from dictionary."""
        metrics = {
            k: PluginConfig(name=k, **v) if isinstance(v, dict) else PluginConfig(name=k, enabled=v)
            for k, v in data.get('metrics', {}).items()
        }
        
        providers = {
            k: PluginConfig(name=k, **v) if isinstance(v, dict) else PluginConfig(name=k, enabled=v)
            for k, v in data.get('providers', {}).items()
        }
        
        training = {
            k: PluginConfig(name=k, **v) if isinstance(v, dict) else PluginConfig(name=k, enabled=v)
            for k, v in data.get('training', {}).items()
        }
        
        runners = {
            k: PluginConfig(name=k, **v) if isinstance(v, dict) else PluginConfig(name=k, enabled=v)
            for k, v in data.get('runners', {}).items()
        }
        
        return cls(
            auto_discover=data.get('auto_discover', True),
            plugin_directories=data.get('plugin_directories', []),
            entry_point_groups=data.get('entry_point_groups', [
                "autoconstitution.metrics",
                "autoconstitution.providers",
                "autoconstitution.training",
                "autoconstitution.runners",
            ]),
            metrics=metrics,
            providers=providers,
            training=training,
            runners=runners,
            health_check_interval=data.get('health_check_interval', 30.0),
            max_plugin_errors=data.get('max_plugin_errors', 5),
            auto_restart=data.get('auto_restart', True),
            default_metric=data.get('default_metric', 'bits_per_byte'),
            default_provider=data.get('default_provider', 'kimi'),
            default_training_target=data.get('default_training_target', 'language_model'),
            default_runner=data.get('default_runner', 'local'),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'auto_discover': self.auto_discover,
            'plugin_directories': self.plugin_directories,
            'entry_point_groups': self.entry_point_groups,
            'metrics': {k: {'enabled': v.enabled, 'config': v.config} 
                       for k, v in self.metrics.items()},
            'providers': {k: {'enabled': v.enabled, 'config': v.config}
                         for k, v in self.providers.items()},
            'training': {k: {'enabled': v.enabled, 'config': v.config}
                        for k, v in self.training.items()},
            'runners': {k: {'enabled': v.enabled, 'config': v.config}
                       for k, v in self.runners.items()},
            'health_check_interval': self.health_check_interval,
            'max_plugin_errors': self.max_plugin_errors,
            'auto_restart': self.auto_restart,
            'default_metric': self.default_metric,
            'default_provider': self.default_provider,
            'default_training_target': self.default_training_target,
            'default_runner': self.default_runner,
        }
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            elif path.suffix == '.json':
                json.dump(self.to_dict(), f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
```

### 6.2 Example Configuration File

```yaml
# autoconstitution_plugins.yaml

# Discovery settings
auto_discover: true
plugin_directories:
  - ./custom_plugins
  - /opt/autoconstitution/plugins

# Default selections
default_metric: bits_per_byte
default_provider: kimi
default_training_target: language_model
default_runner: local

# Metric plugins
metrics:
  bits_per_byte:
    enabled: true
    config: {}
  
  inference_speed:
    enabled: true
    config: {}
  
  perplexity:
    enabled: false
  
  composite:
    enabled: true
    config:
      metrics:
        quality:
          class: bits_per_byte
          weight: 0.7
        speed:
          class: inference_speed
          weight: 0.3

# Provider plugins
providers:
  kimi:
    enabled: true
    config:
      api_key: ${KIMI_API_KEY}
      default_model: kimi-k2.5
      timeout: 120
  
  claude:
    enabled: true
    config:
      api_key: ${ANTHROPIC_API_KEY}
      default_model: claude-3-5-sonnet-20241022
  
  ollama:
    enabled: true
    config:
      base_url: http://localhost:11434
      default_model: mistral

# Training target plugins
training:
  language_model:
    enabled: true
    config:
      device: auto
      mixed_precision: true
  
  rlhf:
    enabled: false

# Experiment runner plugins
runners:
  local:
    enabled: true
    config:
      output_dir: ./experiments
  
  docker:
    enabled: false

# Lifecycle settings
health_check_interval: 30.0
max_plugin_errors: 5
auto_restart: true
```

---

## 7. Usage Examples

### 7.1 Basic Usage

```python
# Initialize plugin system
from autoconstitution.plugins import PluginSystemConfig, registry, lifecycle

# Load configuration
config = PluginSystemConfig.from_file("autoconstitution_plugins.yaml")

# Discover plugins
registry.discover_all(config.plugin_directories)

# Initialize lifecycle manager
manager = lifecycle.PluginLifecycleManager(
    health_check_interval=config.health_check_interval,
    max_errors=config.max_plugin_errors,
    auto_restart=config.auto_restart
)

# Load plugins
for name, plugin_config in config.metrics.items():
    if plugin_config.enabled:
        await manager.load(name, plugin_config.config)

# Start health monitoring
await manager.start_health_monitoring()

# Get a plugin
metric = registry.get("bits_per_byte")
provider = registry.get("kimi")
```

### 7.2 Creating a Custom Plugin

```python
# my_custom_metric.py
from autoconstitution.plugins.interfaces.metrics import (
    RatchetMetric, MetricValue, ComparisonResult, OptimizationDirection
)
from autoconstitution.plugins.registry import register_plugin

@register_plugin("my_metric")
class MyCustomMetric(RatchetMetric):
    """My custom ratchet metric."""
    
    def compute(self, evaluation_data: Dict[str, Any]) -> MetricValue:
        value = evaluation_data['my_value']
        
        return MetricValue(
            value=value,
            timestamp=time.time(),
            metadata={'source': 'custom'}
        )
    
    def compare(self, current, baseline, epsilon=0.0) -> ComparisonResult:
        improvement = current.value - baseline.value
        
        return ComparisonResult(
            is_better=current.value > baseline.value + epsilon,
            improvement=improvement,
            improvement_pct=improvement / baseline.value * 100,
            is_significant=abs(improvement) > epsilon,
            epsilon_used=epsilon
        )
    
    def get_direction(self) -> OptimizationDirection:
        return OptimizationDirection.MAXIMIZE
    
    def get_units(self) -> str:
        return "score"
```

### 7.3 Plugin Package Structure

```
my-autoconstitution-plugin/
├── pyproject.toml
├── README.md
└── my_plugin/
    ├── __init__.py
    ├── metrics.py          # Custom metrics
    ├── providers.py        # Custom providers
    ├── training.py         # Custom training targets
    └── runners.py          # Custom experiment runners
```

```toml
# pyproject.toml
[project]
name = "my-autoconstitution-plugin"
version = "1.0.0"
description = "Custom plugins for autoconstitution"

[project.entry-points."autoconstitution.metrics"]
my_metric = "my_plugin.metrics:MyCustomMetric"

[project.entry-points."autoconstitution.providers"]
my_provider = "my_plugin.providers:MyCustomProvider"
```

---

## 8. Summary

This plugin architecture provides:

1. **Clean Interfaces**: Well-defined, minimal interfaces for each plugin type
2. **Easy Discovery**: Multiple discovery mechanisms (entry points, directories)
3. **Lifecycle Management**: Proper initialization, health monitoring, and shutdown
4. **Configuration-Driven**: YAML/JSON configuration for all plugins
5. **Extensibility**: Easy to add new plugin types and implementations

The architecture makes autoconstitution truly general-purpose, supporting:
- Any ratchet metric (not just val_bpb)
- Any LLM provider (not just Kimi)
- Any training target (not just LM training)
- Any experiment runner (not just local execution)
