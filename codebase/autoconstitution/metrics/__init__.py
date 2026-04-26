"""
autoconstitution Metrics Module

Provides pluggable metric interfaces with abstract base classes,
comparison operators, serialization, and metadata support.
"""

from .base import (
    BaseMetric,
    MetricMetadata,
    MetricThreshold,
    MetricType,
    MetricUnit,
    AggregationMethod,
    Serializable,
    Comparable,
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    MetricCollection,
)

__all__ = [
    "BaseMetric",
    "MetricMetadata",
    "MetricThreshold",
    "MetricType",
    "MetricUnit",
    "AggregationMethod",
    "Serializable",
    "Comparable",
    "CounterMetric",
    "GaugeMetric",
    "HistogramMetric",
    "MetricCollection",
]
