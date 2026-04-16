"""
autoconstitution Experiment Module

Provides the Experiment class for wrapping timed training runs with comprehensive
metrics capture, timeout handling, and structured result production.

Python 3.11+ async-first implementation with complete type hints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)
from collections.abc import Sequence
from typing_extensions import Self

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Type Variables
# ============================================================================

T = TypeVar("T")
TConfig = TypeVar("TConfig", bound="ExperimentConfig")
TMetrics = TypeVar("TMetrics", bound="MetricsSnapshot")


# ============================================================================
# Enums
# ============================================================================

class ExperimentStatus(Enum):
    """Status of an experiment run."""
    PENDING = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    CANCELLED = auto()


class MetricType(Enum):
    """Types of metrics that can be captured."""
    SCALAR = "scalar"
    TIMESERIES = "timeseries"
    HISTOGRAM = "histogram"
    COUNTER = "counter"
    GAUGE = "gauge"


class CheckpointFormat(Enum):
    """Format for checkpoint serialization."""
    JSON = "json"
    PICKLE = "pickle"
    MESSAGEPACK = "msgpack"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True, slots=True)
class ExperimentID:
    """Unique identifier for an experiment."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value
    
    def __hash__(self) -> int:
        return hash(self.value)


@dataclass(frozen=True, slots=True)
class MetricValue:
    """A single metric value with metadata."""
    name: str
    value: float | int | str | bool
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    step: int = 0
    tags: dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "metric_type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "step": self.step,
            "tags": self.tags,
        }


@dataclass(slots=True)
class MetricsSnapshot:
    """Snapshot of metrics at a point in time."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    step: int = 0
    values: dict[str, MetricValue] = field(default_factory=dict)
    
    def add_scalar(self, name: str, value: float | int, tags: dict[str, str] | None = None) -> None:
        """Add a scalar metric."""
        self.values[name] = MetricValue(
            name=name,
            value=value,
            metric_type=MetricType.SCALAR,
            timestamp=datetime.utcnow(),
            step=self.step,
            tags=tags or {},
        )
    
    def add_counter(self, name: str, value: int, tags: dict[str, str] | None = None) -> None:
        """Add a counter metric."""
        self.values[name] = MetricValue(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            timestamp=datetime.utcnow(),
            step=self.step,
            tags=tags or {},
        )
    
    def get_scalar(self, name: str) -> float | int | None:
        """Get a scalar metric value."""
        metric = self.values.get(name)
        if metric and metric.metric_type == MetricType.SCALAR:
            return cast(float | int, metric.value)
        return None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "step": self.step,
            "values": {k: v.to_dict() for k, v in self.values.items()},
        }


@dataclass(slots=True)
class ResourceUsage:
    """Resource usage statistics."""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    io_read_mb: float = 0.0
    io_write_mb: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "memory_percent": self.memory_percent,
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_utilization": self.gpu_utilization,
            "io_read_mb": self.io_read_mb,
            "io_write_mb": self.io_write_mb,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(slots=True)
class TimingInfo:
    """Timing information for an experiment."""
    started_at: datetime | None = None
    ended_at: datetime | None = None
    elapsed_seconds: float = 0.0
    
    @property
    def is_running(self) -> bool:
        """Check if timing is active."""
        return self.started_at is not None and self.ended_at is None
    
    @property
    def duration_seconds(self) -> float:
        """Get total duration in seconds."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return self.elapsed_seconds
    
    def start(self) -> None:
        """Start timing."""
        self.started_at = datetime.utcnow()
    
    def stop(self) -> None:
        """Stop timing."""
        self.ended_at = datetime.utcnow()
        if self.started_at:
            self.elapsed_seconds = (self.ended_at - self.started_at).total_seconds()
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(slots=True)
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str = "unnamed_experiment"
    description: str = ""
    timeout_seconds: float = 3600.0  # 1 hour default
    max_retries: int = 0
    checkpoint_interval_seconds: float | None = None
    metrics_interval_seconds: float = 10.0
    save_checkpoints: bool = True
    checkpoint_format: CheckpointFormat = CheckpointFormat.JSON
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "checkpoint_interval_seconds": self.checkpoint_interval_seconds,
            "metrics_interval_seconds": self.metrics_interval_seconds,
            "save_checkpoints": self.save_checkpoints,
            "checkpoint_format": self.checkpoint_format.value,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ExperimentResult:
    """Result of an experiment execution."""
    experiment_id: ExperimentID
    status: ExperimentStatus
    config: ExperimentConfig
    timing: TimingInfo = field(default_factory=TimingInfo)
    final_metrics: MetricsSnapshot = field(default_factory=MetricsSnapshot)
    resource_usage: list[ResourceUsage] = field(default_factory=list)
    error: Exception | None = None
    error_traceback: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        """Check if experiment completed successfully."""
        return self.status == ExperimentStatus.COMPLETED
    
    @property
    def failed(self) -> bool:
        """Check if experiment failed."""
        return self.status in (ExperimentStatus.FAILED, ExperimentStatus.TIMEOUT)
    
    def get_metric(self, name: str) -> MetricValue | None:
        """Get a metric by name from final metrics."""
        return self.final_metrics.values.get(name)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": str(self.experiment_id),
            "status": self.status.name,
            "success": self.success,
            "config": self.config.to_dict(),
            "timing": self.timing.to_dict(),
            "final_metrics": self.final_metrics.to_dict(),
            "resource_usage": [r.to_dict() for r in self.resource_usage],
            "error": str(self.error) if self.error else None,
            "error_traceback": self.error_traceback,
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
            "checkpoints": self.checkpoints,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ============================================================================
# Protocols
# ============================================================================

@runtime_checkable
class MetricsCollector(Protocol):
    """Protocol for metrics collection."""
    
    async def collect(self) -> MetricsSnapshot:
        """Collect current metrics snapshot."""
        ...
    
    async def record(self, metric: MetricValue) -> None:
        """Record a single metric value."""
        ...


@runtime_checkable
class ResourceMonitor(Protocol):
    """Protocol for resource monitoring."""
    
    async def snapshot(self) -> ResourceUsage:
        """Get current resource usage snapshot."""
        ...
    
    async def start_monitoring(self, interval_seconds: float) -> None:
        """Start continuous monitoring."""
        ...
    
    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        ...


@runtime_checkable
class CheckpointStore(Protocol):
    """Protocol for checkpoint storage."""
    
    async def save(self, experiment_id: ExperimentID, data: dict[str, Any]) -> str:
        """Save checkpoint data and return checkpoint ID."""
        ...
    
    async def load(self, checkpoint_id: str) -> dict[str, Any] | None:
        """Load checkpoint data by ID."""
        ...
    
    async def list_checkpoints(self, experiment_id: ExperimentID) -> list[str]:
        """List all checkpoints for an experiment."""
        ...


# ============================================================================
# Exceptions
# ============================================================================

class ExperimentError(Exception):
    """Base exception for experiment-related errors."""
    pass


class ExperimentTimeoutError(ExperimentError):
    """Exception raised when experiment times out."""
    
    def __init__(self, message: str, elapsed_seconds: float, timeout_seconds: float) -> None:
        super().__init__(message)
        self.elapsed_seconds = elapsed_seconds
        self.timeout_seconds = timeout_seconds


class ExperimentCancelledError(ExperimentError):
    """Exception raised when experiment is cancelled."""
    pass


class CheckpointError(ExperimentError):
    """Exception raised when checkpoint operations fail."""
    pass


# ============================================================================
# Default Implementations
# ============================================================================

class DefaultResourceMonitor:
    """Default implementation of resource monitoring using psutil."""
    
    def __init__(self) -> None:
        self._monitoring = False
        self._monitor_task: asyncio.Task[Any] | None = None
        self._snapshots: list[ResourceUsage] = []
        self._lock = asyncio.Lock()
    
    async def snapshot(self) -> ResourceUsage:
        """Get current resource usage."""
        try:
            import psutil
            
            process = psutil.Process()
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = process.memory_percent()
            
            # I/O stats
            io_counters = process.io_counters()
            io_read_mb = io_counters.read_bytes / (1024 * 1024)
            io_write_mb = io_counters.write_bytes / (1024 * 1024)
            
            # Try to get GPU info if available
            gpu_memory_mb = 0.0
            gpu_utilization = 0.0
            
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_mb = mem_info.used / (1024 * 1024)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = util.gpu
                pynvml.nvmlShutdown()
            except Exception:
                pass
            
            return ResourceUsage(
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                gpu_memory_mb=gpu_memory_mb,
                gpu_utilization=gpu_utilization,
                io_read_mb=io_read_mb,
                io_write_mb=io_write_mb,
            )
        except ImportError:
            return ResourceUsage()
    
    async def start_monitoring(self, interval_seconds: float) -> None:
        """Start continuous monitoring."""
        async with self._lock:
            if self._monitoring:
                return
            self._monitoring = True
            self._monitor_task = asyncio.create_task(
                self._monitor_loop(interval_seconds)
            )
    
    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        async with self._lock:
            self._monitoring = False
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
                self._monitor_task = None
    
    async def _monitor_loop(self, interval_seconds: float) -> None:
        """Monitoring loop."""
        while self._monitoring:
            try:
                snapshot = await self.snapshot()
                async with self._lock:
                    self._snapshots.append(snapshot)
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                await asyncio.sleep(interval_seconds)
    
    def get_snapshots(self) -> list[ResourceUsage]:
        """Get all collected snapshots."""
        return self._snapshots.copy()
    
    def clear_snapshots(self) -> None:
        """Clear all snapshots."""
        self._snapshots.clear()


class FileCheckpointStore:
    """File-based checkpoint storage."""
    
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    async def save(self, experiment_id: ExperimentID, data: dict[str, Any]) -> str:
        """Save checkpoint to file."""
        checkpoint_id = f"checkpoint_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        exp_dir = self.base_dir / str(experiment_id)
        exp_dir.mkdir(exist_ok=True)
        
        filepath = exp_dir / f"{checkpoint_id}.json"
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._write_json, filepath, data
        )
        
        return checkpoint_id
    
    def _write_json(self, filepath: Path, data: dict[str, Any]) -> None:
        """Write JSON data to file (sync)."""
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    async def load(self, checkpoint_id: str) -> dict[str, Any] | None:
        """Load checkpoint from file."""
        # Search in all experiment directories
        for exp_dir in self.base_dir.iterdir():
            if exp_dir.is_dir():
                filepath = exp_dir / f"{checkpoint_id}.json"
                if filepath.exists():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, self._read_json, filepath
                    )
        return None
    
    def _read_json(self, filepath: Path) -> dict[str, Any]:
        """Read JSON data from file (sync)."""
        with open(filepath, "r") as f:
            return json.load(f)
    
    async def list_checkpoints(self, experiment_id: ExperimentID) -> list[str]:
        """List all checkpoints for an experiment."""
        exp_dir = self.base_dir / str(experiment_id)
        if not exp_dir.exists():
            return []
        
        checkpoints = []
        for filepath in exp_dir.glob("checkpoint_*.json"):
            checkpoints.append(filepath.stem)
        
        return sorted(checkpoints)


# ============================================================================
# Experiment Class
# ============================================================================

class Experiment(Generic[TConfig]):
    """
    Experiment class for wrapping timed training runs.
    
    Features:
    - Timed execution with timeout handling
    - Comprehensive metrics capture
    - Resource usage monitoring
    - Checkpoint management
    - Async-first design
    - Full type hints
    
    Type Parameters:
        TConfig: The type of experiment configuration
    
    Example:
        >>> config = ExperimentConfig(
        ...     name="training_run_1",
        ...     timeout_seconds=3600,
        ... )
        >>> experiment = Experiment(config)
        >>> 
        >>> async def training_loop(experiment: Experiment) -> dict[str, Any]:
        ...     for epoch in range(10):
        ...         loss = await train_epoch()
        ...         experiment.log_scalar("loss", loss, step=epoch)
        ...     return {"final_loss": loss}
        >>> 
        >>> result = await experiment.run(training_loop)
        >>> print(result.success)
        >>> print(result.timing.duration_seconds)
    
    Attributes:
        experiment_id: Unique identifier for this experiment
        config: Experiment configuration
        status: Current experiment status
        timing: Timing information
    """
    
    def __init__(
        self,
        config: TConfig | None = None,
        experiment_id: ExperimentID | None = None,
        resource_monitor: ResourceMonitor | None = None,
        checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        """
        Initialize the experiment.
        
        Args:
            config: Experiment configuration
            experiment_id: Optional unique identifier (auto-generated if not provided)
            resource_monitor: Optional resource monitor
            checkpoint_store: Optional checkpoint store
        """
        self._experiment_id = experiment_id or ExperimentID()
        self._config = config or cast(TConfig, ExperimentConfig())
        self._status = ExperimentStatus.PENDING
        self._timing = TimingInfo()
        self._metrics_history: list[MetricsSnapshot] = []
        self._current_metrics = MetricsSnapshot()
        self._resource_usage: list[ResourceUsage] = []
        self._checkpoints: list[str] = []
        self._artifacts: dict[str, Any] = {}
        self._error: Exception | None = None
        self._error_traceback: str | None = None
        
        # Components
        self._resource_monitor = resource_monitor or DefaultResourceMonitor()
        self._checkpoint_store = checkpoint_store
        
        # Control
        self._cancel_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._metrics_callbacks: list[Callable[[MetricValue], Coroutine[Any, Any, None]]] = []
        
        # Background tasks
        self._background_tasks: set[asyncio.Task[Any]] = set()
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def experiment_id(self) -> ExperimentID:
        """Return the unique experiment identifier."""
        return self._experiment_id
    
    @property
    def config(self) -> TConfig:
        """Return the experiment configuration."""
        return self._config
    
    @property
    def status(self) -> ExperimentStatus:
        """Return the current experiment status."""
        return self._status
    
    @property
    def timing(self) -> TimingInfo:
        """Return timing information."""
        return self._timing
    
    @property
    def is_running(self) -> bool:
        """Check if experiment is currently running."""
        return self._status == ExperimentStatus.RUNNING
    
    @property
    def is_completed(self) -> bool:
        """Check if experiment has completed."""
        return self._status in (
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.TIMEOUT,
            ExperimentStatus.CANCELLED,
        )
    
    @property
    def current_step(self) -> int:
        """Return the current step number."""
        return self._current_metrics.step
    
    # -------------------------------------------------------------------------
    # Core Methods
    # -------------------------------------------------------------------------
    
    async def run(
        self,
        train_fn: Callable[[Self], Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any,
    ) -> ExperimentResult:
        """
        Run the experiment with the given training function.
        
        Args:
            train_fn: Async function that performs the training
            *args: Additional positional arguments for train_fn
            **kwargs: Additional keyword arguments for train_fn
        
        Returns:
            ExperimentResult containing all experiment data
        
        Raises:
            ExperimentTimeoutError: If experiment exceeds timeout
            ExperimentCancelledError: If experiment is cancelled
        """
        async with self._lock:
            if self._status == ExperimentStatus.RUNNING:
                raise ExperimentError("Experiment is already running")
            
            self._status = ExperimentStatus.INITIALIZING
            self._timing.start()
            self._cancel_event.clear()
            self._pause_event.clear()
        
        logger.info(f"Starting experiment {self._experiment_id}: {self._config.name}")
        
        # Start resource monitoring
        if isinstance(self._resource_monitor, DefaultResourceMonitor):
            await self._resource_monitor.start_monitoring(self._config.metrics_interval_seconds)
        
        # Start checkpoint loop if configured
        if self._config.checkpoint_interval_seconds:
            checkpoint_task = asyncio.create_task(
                self._checkpoint_loop(self._config.checkpoint_interval_seconds)
            )
            self._background_tasks.add(checkpoint_task)
            checkpoint_task.add_done_callback(self._background_tasks.discard)
        
        try:
            async with self._lock:
                self._status = ExperimentStatus.RUNNING
            
            # Execute training function with timeout
            timeout = self._config.timeout_seconds
            
            if timeout > 0:
                result = await asyncio.wait_for(
                    self._execute_train_fn(train_fn, *args, **kwargs),
                    timeout=timeout,
                )
            else:
                result = await self._execute_train_fn(train_fn, *args, **kwargs)
            
            # Store result as artifact
            self._artifacts["result"] = result
            
            async with self._lock:
                self._status = ExperimentStatus.COMPLETED
            
            logger.info(f"Experiment {self._experiment_id} completed successfully")
            
        except asyncio.TimeoutError:
            elapsed = self._timing.elapsed_seconds
            async with self._lock:
                self._status = ExperimentStatus.TIMEOUT
            
            self._error = ExperimentTimeoutError(
                f"Experiment timed out after {elapsed:.2f}s (timeout: {timeout}s)",
                elapsed_seconds=elapsed,
                timeout_seconds=timeout,
            )
            logger.error(f"Experiment {self._experiment_id} timed out")
            
        except ExperimentCancelledError:
            async with self._lock:
                self._status = ExperimentStatus.CANCELLED
            logger.info(f"Experiment {self._experiment_id} was cancelled")
            
        except Exception as e:
            async with self._lock:
                self._status = ExperimentStatus.FAILED
            
            self._error = e
            self._error_traceback = traceback.format_exc()
            logger.error(f"Experiment {self._experiment_id} failed: {e}")
        
        finally:
            # Stop resource monitoring
            if isinstance(self._resource_monitor, DefaultResourceMonitor):
                await self._resource_monitor.stop_monitoring()
                self._resource_usage = self._resource_monitor.get_snapshots()
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
            
            # Finalize timing
            self._timing.stop()
        
        return self._build_result()
    
    async def _execute_train_fn(
        self,
        train_fn: Callable[[Self], Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute the training function with cancellation checks."""
        # Create task for training function
        train_task = asyncio.create_task(train_fn(self, *args, **kwargs))
        
        # Wait for either training completion or cancellation
        while not train_task.done():
            if self._cancel_event.is_set():
                train_task.cancel()
                try:
                    await train_task
                except asyncio.CancelledError:
                    pass
                raise ExperimentCancelledError("Experiment was cancelled")
            
            # Short sleep to allow checking for cancellation
            try:
                await asyncio.wait_for(asyncio.shield(train_task), timeout=0.1)
            except asyncio.TimeoutError:
                continue
        
        return await train_task
    
    async def _checkpoint_loop(self, interval_seconds: float) -> None:
        """Background task for periodic checkpointing."""
        while self.is_running:
            try:
                await asyncio.wait_for(
                    self._cancel_event.wait(),
                    timeout=interval_seconds,
                )
            except asyncio.TimeoutError:
                if self.is_running and self._config.save_checkpoints:
                    await self.save_checkpoint()
    
    def _build_result(self) -> ExperimentResult:
        """Build the experiment result."""
        return ExperimentResult(
            experiment_id=self._experiment_id,
            status=self._status,
            config=self._config,
            timing=self._timing,
            final_metrics=self._current_metrics,
            resource_usage=self._resource_usage,
            error=self._error,
            error_traceback=self._error_traceback,
            artifacts=self._artifacts,
            checkpoints=self._checkpoints,
        )
    
    # -------------------------------------------------------------------------
    # Metrics Logging
    # -------------------------------------------------------------------------
    
    def log_scalar(
        self,
        name: str,
        value: float | int,
        step: int | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """
        Log a scalar metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number (uses current step if not provided)
            tags: Optional tags for the metric
        """
        if step is not None:
            self._current_metrics.step = step
        
        self._current_metrics.add_scalar(name, value, tags)
        
        # Notify callbacks
        metric = self._current_metrics.values[name]
        for callback in self._metrics_callbacks:
            asyncio.create_task(callback(metric))
    
    def log_counter(
        self,
        name: str,
        value: int,
        step: int | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """
        Log a counter metric value.
        
        Args:
            name: Metric name
            value: Counter value
            step: Optional step number
            tags: Optional tags for the metric
        """
        if step is not None:
            self._current_metrics.step = step
        
        self._current_metrics.add_counter(name, value, tags)
    
    def log_metrics(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        """
        Log multiple scalar metrics at once.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number
        """
        if step is not None:
            self._current_metrics.step = step
        
        for name, value in metrics.items():
            self._current_metrics.add_scalar(name, value)
    
    def increment_step(self) -> int:
        """Increment the current step and return the new value."""
        self._current_metrics.step += 1
        return self._current_metrics.step
    
    def set_step(self, step: int) -> None:
        """Set the current step number."""
        self._current_metrics.step = step
    
    def get_metric(self, name: str) -> MetricValue | None:
        """Get a metric value by name."""
        return self._current_metrics.values.get(name)
    
    def snapshot_metrics(self) -> MetricsSnapshot:
        """Create a snapshot of current metrics."""
        snapshot = MetricsSnapshot(
            timestamp=datetime.utcnow(),
            step=self._current_metrics.step,
            values=self._current_metrics.values.copy(),
        )
        self._metrics_history.append(snapshot)
        return snapshot
    
    def register_metrics_callback(
        self,
        callback: Callable[[MetricValue], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a callback for metric updates."""
        self._metrics_callbacks.append(callback)
    
    def unregister_metrics_callback(
        self,
        callback: Callable[[MetricValue], Coroutine[Any, Any, None]],
    ) -> None:
        """Unregister a metrics callback."""
        if callback in self._metrics_callbacks:
            self._metrics_callbacks.remove(callback)
    
    # -------------------------------------------------------------------------
    # Checkpoint Management
    # -------------------------------------------------------------------------
    
    async def save_checkpoint(self, name: str | None = None) -> str | None:
        """
        Save a checkpoint of the current experiment state.
        
        Args:
            name: Optional checkpoint name
        
        Returns:
            Checkpoint ID if saved successfully, None otherwise
        """
        if not self._checkpoint_store:
            logger.warning("No checkpoint store configured")
            return None
        
        checkpoint_data = {
            "experiment_id": str(self._experiment_id),
            "timestamp": datetime.utcnow().isoformat(),
            "step": self._current_metrics.step,
            "metrics": self._current_metrics.to_dict(),
            "name": name,
        }
        
        try:
            checkpoint_id = await self._checkpoint_store.save(
                self._experiment_id,
                checkpoint_data,
            )
            self._checkpoints.append(checkpoint_id)
            logger.debug(f"Saved checkpoint: {checkpoint_id}")
            return checkpoint_id
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    async def load_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None:
        """
        Load a checkpoint by ID.
        
        Args:
            checkpoint_id: Checkpoint ID to load
        
        Returns:
            Checkpoint data if found, None otherwise
        """
        if not self._checkpoint_store:
            logger.warning("No checkpoint store configured")
            return None
        
        return await self._checkpoint_store.load(checkpoint_id)
    
    # -------------------------------------------------------------------------
    # Artifact Management
    # -------------------------------------------------------------------------
    
    def add_artifact(self, name: str, artifact: Any) -> None:
        """
        Add an artifact to the experiment.
        
        Args:
            name: Artifact name
            artifact: Artifact data (model, file path, etc.)
        """
        self._artifacts[name] = artifact
    
    def get_artifact(self, name: str) -> Any | None:
        """Get an artifact by name."""
        return self._artifacts.get(name)
    
    def list_artifacts(self) -> list[str]:
        """List all artifact names."""
        return list(self._artifacts.keys())
    
    # -------------------------------------------------------------------------
    # Control Methods
    # -------------------------------------------------------------------------
    
    def cancel(self) -> None:
        """Cancel the experiment."""
        logger.info(f"Cancelling experiment {self._experiment_id}")
        self._cancel_event.set()
    
    async def pause(self) -> None:
        """Pause the experiment (if supported by training function)."""
        async with self._lock:
            if self._status == ExperimentStatus.RUNNING:
                self._status = ExperimentStatus.PAUSED
                self._pause_event.set()
                logger.info(f"Paused experiment {self._experiment_id}")
    
    async def resume(self) -> None:
        """Resume a paused experiment."""
        async with self._lock:
            if self._status == ExperimentStatus.PAUSED:
                self._status = ExperimentStatus.RUNNING
                self._pause_event.clear()
                logger.info(f"Resumed experiment {self._experiment_id}")
    
    async def wait_until_complete(self, timeout: float | None = None) -> ExperimentResult:
        """
        Wait until the experiment completes.
        
        Args:
            timeout: Maximum wait time in seconds
        
        Returns:
            ExperimentResult
        
        Raises:
            TimeoutError: If timeout is reached
        """
        start_time = time.monotonic()
        
        while not self.is_completed:
            if timeout and (time.monotonic() - start_time) > timeout:
                raise TimeoutError("Timeout waiting for experiment completion")
            await asyncio.sleep(0.1)
        
        return self._build_result()
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> dict[str, Any]:
        """Convert experiment state to dictionary."""
        return {
            "experiment_id": str(self._experiment_id),
            "status": self._status.name,
            "config": self._config.to_dict(),
            "timing": self._timing.to_dict(),
            "current_step": self.current_step,
            "metrics_count": len(self._current_metrics.values),
            "checkpoints_count": len(self._checkpoints),
            "artifacts": list(self._artifacts.keys()),
        }
    
    def __repr__(self) -> str:
        return (
            f"Experiment(id={self._experiment_id}, "
            f"name={self._config.name}, "
            f"status={self._status.name})"
        )


# ============================================================================
# Experiment Runner
# ============================================================================

class ExperimentRunner:
    """
    Runner for executing multiple experiments with batch management.
    
    Features:
    - Sequential or parallel experiment execution
    - Result aggregation
    - Progress tracking
    """
    
    def __init__(
        self,
        max_parallel: int = 1,
        continue_on_error: bool = False,
    ) -> None:
        """
        Initialize the experiment runner.
        
        Args:
            max_parallel: Maximum number of parallel experiments
            continue_on_error: Whether to continue on experiment failure
        """
        self._max_parallel = max_parallel
        self._continue_on_error = continue_on_error
        self._experiments: list[Experiment[Any]] = []
        self._results: list[ExperimentResult] = []
        self._lock = asyncio.Lock()
    
    def add_experiment(self, experiment: Experiment[Any]) -> None:
        """Add an experiment to the runner."""
        self._experiments.append(experiment)
    
    async def run_all(
        self,
        train_fn: Callable[[Experiment[Any]], Coroutine[Any, Any, Any]],
    ) -> list[ExperimentResult]:
        """
        Run all experiments.
        
        Args:
            train_fn: Training function to use for all experiments
        
        Returns:
            List of experiment results
        """
        self._results = []
        
        if self._max_parallel == 1:
            # Sequential execution
            for experiment in self._experiments:
                try:
                    result = await experiment.run(train_fn)
                    self._results.append(result)
                    
                    if not result.success and not self._continue_on_error:
                        break
                except Exception as e:
                    logger.error(f"Experiment failed: {e}")
                    if not self._continue_on_error:
                        break
        else:
            # Parallel execution with semaphore
            semaphore = asyncio.Semaphore(self._max_parallel)
            
            async def run_with_semaphore(exp: Experiment[Any]) -> ExperimentResult | None:
                async with semaphore:
                    try:
                        return await exp.run(train_fn)
                    except Exception as e:
                        logger.error(f"Experiment failed: {e}")
                        return None
            
            tasks = [run_with_semaphore(exp) for exp in self._experiments]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, ExperimentResult):
                    self._results.append(result)
        
        return self._results
    
    def get_results(self) -> list[ExperimentResult]:
        """Get all experiment results."""
        return self._results.copy()
    
    def get_successful_results(self) -> list[ExperimentResult]:
        """Get only successful experiment results."""
        return [r for r in self._results if r.success]
    
    def get_failed_results(self) -> list[ExperimentResult]:
        """Get only failed experiment results."""
        return [r for r in self._results if r.failed]
    
    def summary(self) -> dict[str, Any]:
        """Get a summary of all results."""
        total = len(self._results)
        successful = len(self.get_successful_results())
        failed = len(self.get_failed_results())
        
        total_duration = sum(r.timing.duration_seconds for r in self._results)
        
        return {
            "total_experiments": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "total_duration_seconds": total_duration,
            "average_duration_seconds": total_duration / total if total > 0 else 0.0,
        }


# ============================================================================
# Utility Functions
# ============================================================================

def create_experiment(
    name: str,
    description: str = "",
    timeout_seconds: float = 3600.0,
    **kwargs: Any,
) -> Experiment[ExperimentConfig]:
    """
    Factory function to create an experiment with standard configuration.
    
    Args:
        name: Experiment name
        description: Experiment description
        timeout_seconds: Timeout in seconds
        **kwargs: Additional configuration options
    
    Returns:
        New Experiment instance
    """
    config = ExperimentConfig(
        name=name,
        description=description,
        timeout_seconds=timeout_seconds,
        **kwargs,
    )
    return Experiment(config)


async def run_experiment(
    name: str,
    train_fn: Callable[[Experiment[ExperimentConfig]], Coroutine[Any, Any, T]],
    timeout_seconds: float = 3600.0,
    **kwargs: Any,
) -> ExperimentResult:
    """
    Convenience function to quickly run an experiment.
    
    Args:
        name: Experiment name
        train_fn: Training function
        timeout_seconds: Timeout in seconds
        **kwargs: Additional configuration options
    
    Returns:
        ExperimentResult
    """
    experiment = create_experiment(name, timeout_seconds=timeout_seconds, **kwargs)
    return await experiment.run(train_fn)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Enums
    "ExperimentStatus",
    "MetricType",
    "CheckpointFormat",
    # Data classes
    "ExperimentID",
    "MetricValue",
    "MetricsSnapshot",
    "ResourceUsage",
    "TimingInfo",
    "ExperimentConfig",
    "ExperimentResult",
    # Protocols
    "MetricsCollector",
    "ResourceMonitor",
    "CheckpointStore",
    # Exceptions
    "ExperimentError",
    "ExperimentTimeoutError",
    "ExperimentCancelledError",
    "CheckpointError",
    # Classes
    "DefaultResourceMonitor",
    "FileCheckpointStore",
    "Experiment",
    "ExperimentRunner",
    # Functions
    "create_experiment",
    "run_experiment",
]
