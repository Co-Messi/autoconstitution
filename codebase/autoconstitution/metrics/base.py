"""
autoconstitution Pluggable Metric Interface

This module provides the abstract base class and supporting infrastructure
for all ratchet metrics in the autoconstitution framework.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)


# ============================================================================
# Type Variables
# ============================================================================

T = TypeVar("T", bound="BaseMetric")
MetricValue = TypeVar("MetricValue", int, float, bool, str, list, dict)


# ============================================================================
# Enums
# ============================================================================

class MetricType(Enum):
    """Classification of metric types for the autoconstitution framework."""

    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()
    SUMMARY = auto()
    DERIVED = auto()
    COMPOSITE = auto()


class MetricUnit(Enum):
    """Standard units for metric values."""

    NONE = "none"
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    MICROSECONDS = "microseconds"
    BYTES = "bytes"
    KILOBYTES = "kilobytes"
    MEGABYTES = "megabytes"
    GIGABYTES = "gigabytes"
    PERCENT = "percent"
    COUNT = "count"
    RATIO = "ratio"
    OPERATIONS = "operations"
    REQUESTS = "requests"
    ERRORS = "errors"


class AggregationMethod(Enum):
    """Supported aggregation methods for metric values."""

    SUM = auto()
    AVG = auto()
    MIN = auto()
    MAX = auto()
    COUNT = auto()
    LAST = auto()
    FIRST = auto()
    P50 = auto()
    P95 = auto()
    P99 = auto()


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True)
class MetricMetadata:
    """
    Immutable metadata container for metric instances.

    Attributes:
        name: Human-readable metric name
        description: Detailed description of what the metric measures
        unit: Unit of measurement
        metric_type: Classification of the metric
        labels: Optional key-value labels for categorization
        timestamp: When the metric was created/recorded
        source: Origin of the metric (e.g., component name)
        version: Schema version for serialization compatibility
    """

    name: str
    description: str = ""
    unit: MetricUnit = MetricUnit.NONE
    metric_type: MetricType = MetricType.GAUGE
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "unknown"
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "unit": self.unit.value,
            "metric_type": self.metric_type.name,
            "labels": dict(self.labels),
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MetricMetadata:
        """Create metadata instance from a dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            unit=MetricUnit(data.get("unit", "none")),
            metric_type=MetricType[data.get("metric_type", "GAUGE")],
            labels=dict(data.get("labels", {})),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.utcnow(),
            source=data.get("source", "unknown"),
            version=data.get("version", "1.0.0"),
        )

    def with_label(self, key: str, value: str) -> MetricMetadata:
        """Return a new metadata instance with an additional label."""
        new_labels = dict(self.labels)
        new_labels[key] = value
        return MetricMetadata(
            name=self.name,
            description=self.description,
            unit=self.unit,
            metric_type=self.metric_type,
            labels=new_labels,
            timestamp=self.timestamp,
            source=self.source,
            version=self.version,
        )


@dataclass(frozen=True)
class MetricThreshold:
    """
    Threshold configuration for metric alerting and validation.

    Attributes:
        warning: Value at which to trigger a warning
        critical: Value at which to trigger a critical alert
        operator: Comparison operator for threshold evaluation
    """

    warning: Optional[float] = None
    critical: Optional[float] = None
    operator: Callable[[float, float], bool] = field(
        default=lambda x, y: x >= y, compare=False
    )

    def check(self, value: float) -> Dict[str, bool]:
        """Check value against thresholds and return status."""
        return {
            "warning": self.warning is not None and self.operator(value, self.warning),
            "critical": self.critical is not None
            and self.operator(value, self.critical),
        }


# ============================================================================
# Protocols
# ============================================================================

@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that support serialization."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary."""
        ...

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create object from dictionary."""
        ...


@runtime_checkable
class Comparable(Protocol[MetricValue]):
    """Protocol for comparable metric values."""

    def __lt__(self, other: Comparable[MetricValue]) -> bool:
        ...

    def __le__(self, other: Comparable[MetricValue]) -> bool:
        ...

    def __gt__(self, other: Comparable[MetricValue]) -> bool:
        ...

    def __ge__(self, other: Comparable[MetricValue]) -> bool:
        ...


# ============================================================================
# Abstract Base Class
# ============================================================================

class BaseMetric(ABC, Generic[MetricValue]):
    """
    Abstract base class for all autoconstitution ratchet metrics.

    This class defines the interface that all metrics must implement,
    including comparison operators, serialization, and metadata support.

    Type Parameters:
        MetricValue: The type of value this metric holds

    Example:
        >>> class CounterMetric(BaseMetric[int]):
        ...     def __init__(self, name: str, value: int = 0) -> None:
        ...         self._value = value
        ...         self._metadata = MetricMetadata(name=name)
        ...
        ...     @property
        ...     def value(self) -> int:
        ...         return self._value
        ...
        ...     @property
        ...     def metadata(self) -> MetricMetadata:
        ...         return self._metadata
    """

    # Registry for metric subclasses (for deserialization)
    _registry: ClassVar[Dict[str, Type["BaseMetric"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register subclasses for deserialization support."""
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    # -------------------------------------------------------------------------
    # Abstract Properties
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def value(self) -> MetricValue:
        """Return the current metric value."""
        raise NotImplementedError

    @property
    @abstractmethod
    def metadata(self) -> MetricMetadata:
        """Return the metric metadata."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def update(self, new_value: MetricValue) -> BaseMetric[MetricValue]:
        """
        Update the metric value and return self or a new instance.

        Args:
            new_value: The new value to set

        Returns:
            The updated metric instance
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> BaseMetric[MetricValue]:
        """
        Reset the metric to its initial state.

        Returns:
            The reset metric instance
        """
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> BaseMetric[MetricValue]:
        """
        Create a deep copy of this metric.

        Returns:
            A new metric instance with the same value and metadata
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Serialization Interface
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the metric to a dictionary.

        Returns:
            Dictionary containing all metric data
        """
        return {
            "__class__": self.__class__.__name__,
            "__module__": self.__class__.__module__,
            "value": self._serialize_value(self.value),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Deserialize a metric from a dictionary.

        Args:
            data: Dictionary containing metric data

        Returns:
            Reconstructed metric instance

        Raises:
            ValueError: If the metric class is not registered
        """
        class_name = data.get("__class__")
        if class_name not in cls._registry:
            raise ValueError(f"Unknown metric class: {class_name}")

        metric_class = cls._registry[class_name]
        return metric_class._from_dict_impl(data)

    @classmethod
    @abstractmethod
    def _from_dict_impl(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Implementation-specific deserialization logic.

        Subclasses must implement this method to handle their specific
        initialization requirements.

        Args:
            data: Dictionary containing metric data

        Returns:
            Reconstructed metric instance
        """
        raise NotImplementedError

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Serialize the metric to a JSON string.

        Args:
            indent: Optional indentation for pretty printing

        Returns:
            JSON string representation of the metric
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """
        Deserialize a metric from a JSON string.

        Args:
            json_str: JSON string containing metric data

        Returns:
            Reconstructed metric instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def _serialize_value(self, value: MetricValue) -> Any:
        """
        Serialize a metric value for storage.

        Subclasses can override this for custom serialization logic.

        Args:
            value: The value to serialize

        Returns:
            Serialized value
        """
        if isinstance(value, (int, float, bool, str, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, Enum):
            return value.name
        else:
            return str(value)

    # -------------------------------------------------------------------------
    # Comparison Operators
    # -------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """
        Check equality with another metric or value.

        Args:
            other: Another metric instance or comparable value

        Returns:
            True if values are equal
        """
        if isinstance(other, BaseMetric):
            return self._compare_values(self.value, other.value) == 0
        return self._compare_values(self.value, other) == 0

    def __ne__(self, other: object) -> bool:
        """Check inequality with another metric or value."""
        return not self.__eq__(other)

    def __lt__(self, other: Union[BaseMetric[MetricValue], MetricValue]) -> bool:
        """
        Less than comparison.

        Args:
            other: Another metric instance or comparable value

        Returns:
            True if this metric's value is less than the other
        """
        if isinstance(other, BaseMetric):
            return self._compare_values(self.value, other.value) < 0
        return self._compare_values(self.value, other) < 0

    def __le__(self, other: Union[BaseMetric[MetricValue], MetricValue]) -> bool:
        """Less than or equal comparison."""
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other: Union[BaseMetric[MetricValue], MetricValue]) -> bool:
        """
        Greater than comparison.

        Args:
            other: Another metric instance or comparable value

        Returns:
            True if this metric's value is greater than the other
        """
        if isinstance(other, BaseMetric):
            return self._compare_values(self.value, other.value) > 0
        return self._compare_values(self.value, other) > 0

    def __ge__(self, other: Union[BaseMetric[MetricValue], MetricValue]) -> bool:
        """Greater than or equal comparison."""
        return self.__gt__(other) or self.__eq__(other)

    def _compare_values(
        self, a: MetricValue, b: Any
    ) -> int:
        """
        Compare two values and return comparison result.

        Args:
            a: First value
            b: Second value

        Returns:
            -1 if a < b, 0 if a == b, 1 if a > b
        """
        if a == b:
            return 0
        try:
            if a < b:
                return -1
            return 1
        except TypeError:
            # Handle incomparable types by string comparison
            return (str(a) > str(b)) - (str(a) < str(b))

    # -------------------------------------------------------------------------
    # Hash Support
    # -------------------------------------------------------------------------

    def __hash__(self) -> int:
        """
        Generate hash based on value and metadata name.

        Returns:
            Hash value for use in collections
        """
        try:
            return hash((self.value, self.metadata.name))
        except TypeError:
            # Handle unhashable values
            return hash((str(self.value), self.metadata.name))

    # -------------------------------------------------------------------------
    # String Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"value={self.value!r}, "
            f"name={self.metadata.name!r}, "
            f"unit={self.metadata.unit.value!r}"
            f")"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        unit_str = f" {self.metadata.unit.value}" if self.metadata.unit != MetricUnit.NONE else ""
        return f"{self.metadata.name}: {self.value}{unit_str}"

    # -------------------------------------------------------------------------
    # Arithmetic Operations
    # -------------------------------------------------------------------------

    def __add__(self, other: Union[BaseMetric[MetricValue], MetricValue]) -> MetricValue:
        """Add this metric's value with another metric or value."""
        if isinstance(other, BaseMetric):
            return self.value + other.value
        return self.value + other

    def __sub__(self, other: Union[BaseMetric[MetricValue], MetricValue]) -> MetricValue:
        """Subtract another metric or value from this metric's value."""
        if isinstance(other, BaseMetric):
            return self.value - other.value
        return self.value - other

    def __mul__(self, other: Union[BaseMetric[MetricValue], MetricValue]) -> MetricValue:
        """Multiply this metric's value with another metric or value."""
        if isinstance(other, BaseMetric):
            return self.value * other.value
        return self.value * other

    def __truediv__(self, other: Union[BaseMetric[MetricValue], MetricValue]) -> MetricValue:
        """Divide this metric's value by another metric or value."""
        if isinstance(other, BaseMetric):
            return self.value / other.value
        return self.value / other

    def __radd__(self, other: MetricValue) -> MetricValue:
        """Reverse addition."""
        return other + self.value

    def __rsub__(self, other: MetricValue) -> MetricValue:
        """Reverse subtraction."""
        return other - self.value

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def is_valid(self) -> bool:
        """
        Check if the metric value is valid.

        Returns:
            True if the metric value is valid
        """
        try:
            _ = self.value
            return True
        except Exception:
            return False

    def get_label(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a label value from metadata.

        Args:
            key: Label key
            default: Default value if key not found

        Returns:
            Label value or default
        """
        return self.metadata.labels.get(key, default)

    def has_label(self, key: str) -> bool:
        """
        Check if a label exists in metadata.

        Args:
            key: Label key

        Returns:
            True if label exists
        """
        return key in self.metadata.labels


# ============================================================================
# Concrete Metric Implementations
# ============================================================================

class CounterMetric(BaseMetric[int]):
    """
    A monotonically increasing counter metric.

    Counters are typically used to track cumulative values like
    total requests, errors, or operations.

    Example:
        >>> counter = CounterMetric("requests_total")
        >>> counter = counter.update(1)
        >>> print(counter.value)  # 1
    """

    def __init__(
        self,
        name: str,
        value: int = 0,
        description: str = "",
        unit: MetricUnit = MetricUnit.COUNT,
        labels: Optional[Dict[str, str]] = None,
        source: str = "unknown",
    ) -> None:
        self._value = max(0, value)
        self._metadata = MetricMetadata(
            name=name,
            description=description,
            unit=unit,
            metric_type=MetricType.COUNTER,
            labels=labels or {},
            source=source,
        )

    @property
    def value(self) -> int:
        return self._value

    @property
    def metadata(self) -> MetricMetadata:
        return self._metadata

    def update(self, new_value: int) -> CounterMetric:
        """Increment the counter by the given amount."""
        return CounterMetric(
            name=self._metadata.name,
            value=self._value + max(0, new_value),
            description=self._metadata.description,
            unit=self._metadata.unit,
            labels=dict(self._metadata.labels),
            source=self._metadata.source,
        )

    def increment(self, amount: int = 1) -> CounterMetric:
        """Convenience method to increment by a specific amount."""
        return self.update(amount)

    def reset(self) -> CounterMetric:
        """Reset counter to zero."""
        return CounterMetric(
            name=self._metadata.name,
            value=0,
            description=self._metadata.description,
            unit=self._metadata.unit,
            labels=dict(self._metadata.labels),
            source=self._metadata.source,
        )

    def clone(self) -> CounterMetric:
        """Create a copy of this counter."""
        return CounterMetric(
            name=self._metadata.name,
            value=self._value,
            description=self._metadata.description,
            unit=self._metadata.unit,
            labels=dict(self._metadata.labels),
            source=self._metadata.source,
        )

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> CounterMetric:
        metadata = MetricMetadata.from_dict(data["metadata"])
        return cls(
            name=metadata.name,
            value=data.get("value", 0),
            description=metadata.description,
            unit=metadata.unit,
            labels=dict(metadata.labels),
            source=metadata.source,
        )


class GaugeMetric(BaseMetric[float]):
    """
    A gauge metric that can go up and down.

    Gauges are used for values that can increase and decrease,
    like temperature, memory usage, or queue depth.

    Example:
        >>> gauge = GaugeMetric("memory_usage_mb", 512.0)
        >>> gauge = gauge.update(1024.0)
        >>> print(gauge.value)  # 1024.0
    """

    def __init__(
        self,
        name: str,
        value: float = 0.0,
        description: str = "",
        unit: MetricUnit = MetricUnit.NONE,
        labels: Optional[Dict[str, str]] = None,
        source: str = "unknown",
    ) -> None:
        self._value = float(value)
        self._metadata = MetricMetadata(
            name=name,
            description=description,
            unit=unit,
            metric_type=MetricType.GAUGE,
            labels=labels or {},
            source=source,
        )

    @property
    def value(self) -> float:
        return self._value

    @property
    def metadata(self) -> MetricMetadata:
        return self._metadata

    def update(self, new_value: float) -> GaugeMetric:
        """Set the gauge to a new value."""
        return GaugeMetric(
            name=self._metadata.name,
            value=float(new_value),
            description=self._metadata.description,
            unit=self._metadata.unit,
            labels=dict(self._metadata.labels),
            source=self._metadata.source,
        )

    def increment(self, amount: float = 1.0) -> GaugeMetric:
        """Increase the gauge by a specific amount."""
        return self.update(self._value + amount)

    def decrement(self, amount: float = 1.0) -> GaugeMetric:
        """Decrease the gauge by a specific amount."""
        return self.update(self._value - amount)

    def reset(self) -> GaugeMetric:
        """Reset gauge to zero."""
        return GaugeMetric(
            name=self._metadata.name,
            value=0.0,
            description=self._metadata.description,
            unit=self._metadata.unit,
            labels=dict(self._metadata.labels),
            source=self._metadata.source,
        )

    def clone(self) -> GaugeMetric:
        """Create a copy of this gauge."""
        return GaugeMetric(
            name=self._metadata.name,
            value=self._value,
            description=self._metadata.description,
            unit=self._metadata.unit,
            labels=dict(self._metadata.labels),
            source=self._metadata.source,
        )

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> GaugeMetric:
        metadata = MetricMetadata.from_dict(data["metadata"])
        return cls(
            name=metadata.name,
            value=float(data.get("value", 0.0)),
            description=metadata.description,
            unit=metadata.unit,
            labels=dict(metadata.labels),
            source=metadata.source,
        )


class HistogramMetric(BaseMetric[Dict[str, Any]]):
    """
    A histogram metric for tracking value distributions.

    Histograms track the distribution of values across configurable
    buckets, useful for latency, request sizes, etc.

    Example:
        >>> hist = HistogramMetric("request_latency_ms", buckets=[10, 50, 100, 500])
        >>> hist = hist.observe(45.0)
        >>> print(hist.value["count"])  # 1
    """

    def __init__(
        self,
        name: str,
        buckets: Optional[list] = None,
        description: str = "",
        unit: MetricUnit = MetricUnit.MILLISECONDS,
        labels: Optional[Dict[str, str]] = None,
        source: str = "unknown",
    ) -> None:
        self._buckets = sorted(buckets or [10, 50, 100, 500, 1000, 5000])
        self._counts = [0] * (len(self._buckets) + 1)
        self._sum = 0.0
        self._count = 0
        self._metadata = MetricMetadata(
            name=name,
            description=description,
            unit=unit,
            metric_type=MetricType.HISTOGRAM,
            labels=labels or {},
            source=source,
        )

    @property
    def value(self) -> Dict[str, Any]:
        return {
            "buckets": self._buckets,
            "counts": self._counts,
            "sum": self._sum,
            "count": self._count,
            "mean": self._sum / self._count if self._count > 0 else 0.0,
        }

    @property
    def metadata(self) -> MetricMetadata:
        return self._metadata

    def observe(self, value: float) -> HistogramMetric:
        """Record a new observation in the histogram."""
        new_hist = self.clone()
        new_hist._sum += value
        new_hist._count += 1

        # Find the bucket
        bucket_idx = len(new_hist._buckets)
        for i, bucket in enumerate(new_hist._buckets):
            if value <= bucket:
                bucket_idx = i
                break

        new_hist._counts[bucket_idx] += 1
        return new_hist

    def update(self, new_value: Dict[str, Any]) -> HistogramMetric:
        """Update histogram with pre-computed values (use with caution)."""
        new_hist = self.clone()
        new_hist._counts = list(new_value.get("counts", self._counts))
        new_hist._sum = new_value.get("sum", self._sum)
        new_hist._count = new_value.get("count", self._count)
        return new_hist

    def reset(self) -> HistogramMetric:
        """Reset histogram to empty state."""
        return HistogramMetric(
            name=self._metadata.name,
            buckets=list(self._buckets),
            description=self._metadata.description,
            unit=self._metadata.unit,
            labels=dict(self._metadata.labels),
            source=self._metadata.source,
        )

    def clone(self) -> HistogramMetric:
        """Create a copy of this histogram."""
        new_hist = HistogramMetric(
            name=self._metadata.name,
            buckets=list(self._buckets),
            description=self._metadata.description,
            unit=self._metadata.unit,
            labels=dict(self._metadata.labels),
            source=self._metadata.source,
        )
        new_hist._counts = list(self._counts)
        new_hist._sum = self._sum
        new_hist._count = self._count
        return new_hist

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> HistogramMetric:
        metadata = MetricMetadata.from_dict(data["metadata"])
        value_data = data.get("value", {})
        hist = cls(
            name=metadata.name,
            buckets=list(value_data.get("buckets", [10, 50, 100, 500])),
            description=metadata.description,
            unit=metadata.unit,
            labels=dict(metadata.labels),
            source=metadata.source,
        )
        hist._counts = list(value_data.get("counts", hist._counts))
        hist._sum = value_data.get("sum", 0.0)
        hist._count = value_data.get("count", 0)
        return hist


# ============================================================================
# Metric Collection
# ============================================================================

class MetricCollection:
    """
    A collection of metrics for batch operations.

    Provides utilities for managing multiple metrics together,
    including serialization and filtering.
    """

    def __init__(self, metrics: Optional[list] = None) -> None:
        self._metrics: Dict[str, BaseMetric] = {}
        if metrics:
            for metric in metrics:
                self.add(metric)

    def add(self, metric: BaseMetric) -> None:
        """Add a metric to the collection."""
        self._metrics[metric.metadata.name] = metric

    def get(self, name: str) -> Optional[BaseMetric]:
        """Get a metric by name."""
        return self._metrics.get(name)

    def remove(self, name: str) -> Optional[BaseMetric]:
        """Remove and return a metric by name."""
        return self._metrics.pop(name, None)

    def filter_by_type(self, metric_type: MetricType) -> list:
        """Filter metrics by type."""
        return [
            m for m in self._metrics.values() if m.metadata.metric_type == metric_type
        ]

    def filter_by_label(self, key: str, value: Optional[str] = None) -> list:
        """Filter metrics by label."""
        result = []
        for metric in self._metrics.values():
            if key in metric.metadata.labels:
                if value is None or metric.metadata.labels[key] == value:
                    result.append(metric)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all metrics to a dictionary."""
        return {
            "metrics": [m.to_dict() for m in self._metrics.values()],
            "count": len(self._metrics),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MetricCollection:
        """Deserialize metrics from a dictionary."""
        collection = cls()
        for metric_data in data.get("metrics", []):
            metric = BaseMetric.from_dict(metric_data)
            collection.add(metric)
        return collection

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize all metrics to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> MetricCollection:
        """Deserialize metrics from JSON."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __len__(self) -> int:
        return len(self._metrics)

    def __iter__(self):
        return iter(self._metrics.values())

    def __contains__(self, name: str) -> bool:
        return name in self._metrics


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Base classes
    "BaseMetric",
    # Data classes
    "MetricMetadata",
    "MetricThreshold",
    # Enums
    "MetricType",
    "MetricUnit",
    "AggregationMethod",
    # Protocols
    "Serializable",
    "Comparable",
    # Concrete implementations
    "CounterMetric",
    "GaugeMetric",
    "HistogramMetric",
    # Collections
    "MetricCollection",
]
