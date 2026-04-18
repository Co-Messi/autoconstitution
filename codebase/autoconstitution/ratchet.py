"""
Ratchet - Core improvement tracking mechanism for autoconstitution.

Implements a ratcheting mechanism that ensures research progress only moves forward
by maintaining and validating the best-known experimental results. Provides pluggable
metric interfaces, state persistence, and async support.

Python 3.11+ async-first implementation with comprehensive type hints.

Example:
    >>> from autoconstitution.ratchet import Ratchet, MetricConfig, ComparisonMode
    >>>
    >>> # Create a ratchet for accuracy tracking
    >>> ratchet = Ratchet(
    ...     metric_name="accuracy",
    ...     comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
    ...     state_path="/path/to/state.json"
    ... )
    >>>
    >>> # Validate a new experiment
    >>> result = await ratchet.validate_experiment(
    ...     experiment_id="exp_001",
    ...     score=0.85,
    ...     metadata={"model": "v1"}
    ... )
    >>> if result.is_improvement:
    ...     print(f"New best: {result.score}")
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)

# Configure logging
logger = logging.getLogger("Ratchet")


# ============================================================================
# Type Definitions
# ============================================================================

T = TypeVar("T")
ExperimentID = str
MetricName = str


# ============================================================================
# Enums
# ============================================================================

class ComparisonMode(Enum):
    """Determines how scores are compared for improvement detection."""
    HIGHER_IS_BETTER = auto()
    LOWER_IS_BETTER = auto()
    CLOSER_TO_TARGET = auto()


class ValidationDecision(Enum):
    """Decision result from validating an experiment."""
    KEEP = "keep"           # New best, replace current
    DISCARD = "discard"     # Not an improvement
    TIE = "tie"             # Equal to current best
    FIRST = "first"         # First experiment, no comparison


# ============================================================================
# Exceptions
# ============================================================================

class RatchetError(Exception):
    """Base exception for ratchet errors."""
    pass


class MetricError(RatchetError):
    """Error in metric operations."""
    pass


class StatePersistenceError(RatchetError):
    """Error in state persistence operations."""
    pass


class ValidationError(RatchetError):
    """Error during experiment validation."""
    pass


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True, slots=True)
class MetricConfig:
    """Configuration for a metric being tracked by the ratchet."""
    name: MetricName
    comparison_mode: ComparisonMode
    target_value: float | None = None  # Required for CLOSER_TO_TARGET mode
    tolerance: float = 0.0  # Tolerance for considering values equal
    weight: float = 1.0  # Weight for multi-metric scenarios

    def __post_init__(self) -> None:
        if self.comparison_mode == ComparisonMode.CLOSER_TO_TARGET and self.target_value is None:
            raise MetricError(
                f"Metric '{self.name}': target_value required for CLOSER_TO_TARGET mode"
            )
        if self.weight < 0:
            raise MetricError(f"Metric '{self.name}': weight must be non-negative")


@dataclass(slots=True)
class ExperimentResult:
    """Result of a single experiment."""
    experiment_id: ExperimentID
    score: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "score": self.score,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentResult:
        """Create from dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            score=data["score"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", {}),
        )


@dataclass(slots=True)
class ValidationResult:
    """Result of validating an experiment against the current best."""
    experiment_id: ExperimentID
    score: float
    decision: ValidationDecision
    is_improvement: bool
    previous_best: float | None
    improvement_delta: float
    improvement_pct: float
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "score": self.score,
            "decision": self.decision.value,
            "is_improvement": self.is_improvement,
            "previous_best": self.previous_best,
            "improvement_delta": self.improvement_delta,
            "improvement_pct": self.improvement_pct,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult:
        """Create from dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            score=data["score"],
            decision=ValidationDecision(data["decision"]),
            is_improvement=data["is_improvement"],
            previous_best=data.get("previous_best"),
            improvement_delta=data.get("improvement_delta", 0.0),
            improvement_pct=data.get("improvement_pct", 0.0),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message=data.get("message", ""),
        )


@dataclass(slots=True)
class RatchetState:
    """Complete state of the ratchet for persistence."""
    metric_name: MetricName
    comparison_mode: str
    current_best_score: float | None
    current_best_experiment: ExperimentResult | None
    experiment_history: list[ExperimentResult] = field(default_factory=list)
    validation_history: list[ValidationResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "comparison_mode": self.comparison_mode,
            "current_best_score": self.current_best_score,
            "current_best_experiment": (
                self.current_best_experiment.to_dict()
                if self.current_best_experiment else None
            ),
            "experiment_history": [e.to_dict() for e in self.experiment_history],
            "validation_history": [v.to_dict() for v in self.validation_history],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RatchetState:
        """Create from dictionary."""
        return cls(
            metric_name=data["metric_name"],
            comparison_mode=data["comparison_mode"],
            current_best_score=data.get("current_best_score"),
            current_best_experiment=(
                ExperimentResult.from_dict(data["current_best_experiment"])
                if data.get("current_best_experiment") else None
            ),
            experiment_history=[
                ExperimentResult.from_dict(e)
                for e in data.get("experiment_history", [])
            ],
            validation_history=[
                ValidationResult.from_dict(v)
                for v in data.get("validation_history", [])
            ],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            version=data.get("version", 1),
        )


@dataclass(slots=True)
class RatchetStats:
    """Statistics about the ratchet's operation."""
    total_experiments: int = 0
    improvements: int = 0
    discards: int = 0
    ties: int = 0
    best_score: float | None = None
    worst_score: float | None = None
    avg_score: float = 0.0
    improvement_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_experiments": self.total_experiments,
            "improvements": self.improvements,
            "discards": self.discards,
            "ties": self.ties,
            "best_score": self.best_score,
            "worst_score": self.worst_score,
            "avg_score": self.avg_score,
            "improvement_rate": self.improvement_rate,
        }


# ============================================================================
# Pluggable Metric Interface
# ============================================================================

@runtime_checkable
class MetricCalculator(Protocol):
    """Protocol for pluggable metric calculators."""

    async def calculate(
        self,
        experiment_id: ExperimentID,
        data: dict[str, Any],
    ) -> float:
        """
        Calculate a metric score for an experiment.

        Args:
            experiment_id: Unique identifier for the experiment
            data: Experiment data to calculate metric from

        Returns:
            Calculated metric score
        """
        ...

    def get_config(self) -> MetricConfig:
        """Get the configuration for this metric."""
        ...


class AbstractMetricCalculator(ABC):
    """Abstract base class for metric calculators."""

    def __init__(self, config: MetricConfig) -> None:
        self._config = config

    @property
    def config(self) -> MetricConfig:
        """Get the metric configuration."""
        return self._config

    @abstractmethod
    async def calculate(
        self,
        experiment_id: ExperimentID,
        data: dict[str, Any],
    ) -> float:
        """Calculate the metric score."""
        pass

    def get_config(self) -> MetricConfig:
        """Get the configuration for this metric."""
        return self._config


class SimpleMetricCalculator(AbstractMetricCalculator):
    """Simple metric calculator that extracts a value from data."""

    def __init__(
        self,
        config: MetricConfig,
        value_key: str = "value",
    ) -> None:
        super().__init__(config)
        self._value_key = value_key

    async def calculate(
        self,
        experiment_id: ExperimentID,
        data: dict[str, Any],
    ) -> float:
        """Extract metric value from data."""
        if self._value_key not in data:
            raise MetricError(
                f"Metric '{self._config.name}': key '{self._value_key}' not found in data"
            )
        value = data[self._value_key]
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise MetricError(
                f"Metric '{self._config.name}': cannot convert {value} to float"
            ) from e


class CompositeMetricCalculator(AbstractMetricCalculator):
    """Calculator that combines multiple metrics with weights."""

    def __init__(
        self,
        config: MetricConfig,
        calculators: list[tuple[AbstractMetricCalculator, float]],
        aggregation: Callable[[list[float]], float] = sum,
    ) -> None:
        super().__init__(config)
        self._calculators = calculators
        self._aggregation = aggregation

    async def calculate(
        self,
        experiment_id: ExperimentID,
        data: dict[str, Any],
    ) -> float:
        """Calculate composite metric from multiple calculators."""
        scores = []
        for calculator, weight in self._calculators:
            score = await calculator.calculate(experiment_id, data)
            scores.append(score * weight)
        return self._aggregation(scores)


# ============================================================================
# State Persistence Interface
# ============================================================================

@runtime_checkable
class StatePersister(Protocol):
    """Protocol for state persistence backends."""

    async def save(self, state: RatchetState) -> None:
        """Save the ratchet state."""
        ...

    async def load(self) -> RatchetState | None:
        """Load the ratchet state."""
        ...

    async def exists(self) -> bool:
        """Check if saved state exists."""
        ...

    async def delete(self) -> bool:
        """Delete the saved state."""
        ...


class FileSystemPersister:
    """File system-based state persistence."""

    def __init__(
        self,
        path: str | Path,
        encoding: str = "utf-8",
    ) -> None:
        self._path = Path(path)
        self._encoding = encoding
        self._lock = asyncio.Lock()

    async def save(self, state: RatchetState) -> None:
        """Save state to file."""
        async with self._lock:
            try:
                # Ensure directory exists
                self._path.parent.mkdir(parents=True, exist_ok=True)

                # Write atomically
                temp_path = self._path.with_suffix(".tmp")
                data = state.to_dict()

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: temp_path.write_text(
                        json.dumps(data, indent=2),
                        encoding=self._encoding,
                    )
                )

                # Atomic rename
                await loop.run_in_executor(
                    None,
                    lambda: temp_path.replace(self._path)
                )

                logger.debug(f"State saved to {self._path}")

            except Exception as e:
                raise StatePersistenceError(f"Failed to save state: {e}") from e

    async def load(self) -> RatchetState | None:
        """Load state from file."""
        async with self._lock:
            try:
                if not await self.exists():
                    return None

                loop = asyncio.get_event_loop()
                content = await loop.run_in_executor(
                    None,
                    lambda: self._path.read_text(encoding=self._encoding)
                )

                data = json.loads(content)
                return RatchetState.from_dict(data)

            except json.JSONDecodeError as e:
                raise StatePersistenceError(f"Invalid JSON in state file: {e}") from e
            except Exception as e:
                raise StatePersistenceError(f"Failed to load state: {e}") from e

    async def exists(self) -> bool:
        """Check if state file exists."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._path.exists)

    async def delete(self) -> bool:
        """Delete state file."""
        async with self._lock:
            try:
                if not await self.exists():
                    return False

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._path.unlink)
                logger.debug(f"State deleted from {self._path}")
                return True

            except Exception as e:
                raise StatePersistenceError(f"Failed to delete state: {e}") from e


class InMemoryPersister:
    """In-memory state persistence (useful for testing)."""

    def __init__(self) -> None:
        self._state: RatchetState | None = None
        self._lock = asyncio.Lock()

    async def save(self, state: RatchetState) -> None:
        """Save state to memory."""
        async with self._lock:
            self._state = state

    async def load(self) -> RatchetState | None:
        """Load state from memory."""
        async with self._lock:
            return self._state

    async def exists(self) -> bool:
        """Check if state exists in memory."""
        async with self._lock:
            return self._state is not None

    async def delete(self) -> bool:
        """Delete state from memory."""
        async with self._lock:
            existed = self._state is not None
            self._state = None
            return existed


# ============================================================================
# Ratchet Core Class
# ============================================================================

class Ratchet:
    """
    Core improvement tracking mechanism for autoconstitution.

    The Ratchet ensures that research progress only moves forward by maintaining
    the best-known experimental result and validating new experiments against it.

    Features:
    - Tracks current best score and associated experiment
    - Validates new experiments with keep/discard decisions
    - Pluggable metric interface for custom calculations
    - Async state persistence to disk
    - Complete history tracking
    - Thread-safe async operations

    Example:
        >>> ratchet = Ratchet(
        ...     metric_name="accuracy",
        ...     comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
        ...     state_path="/data/ratchet.json"
        ... )
        >>>
        >>> # Load previous state if exists
        >>> await ratchet.load_state()
        >>>
        >>> # Validate experiments
        >>> result = await ratchet.validate_experiment("exp1", 0.85)
        >>> if result.is_improvement:
        ...     await ratchet.commit_experiment("exp1", 0.85)
    """

    def __init__(
        self,
        metric_name: MetricName,
        comparison_mode: ComparisonMode = ComparisonMode.HIGHER_IS_BETTER,
        target_value: float | None = None,
        tolerance: float = 0.0,
        state_path: str | Path | None = None,
        persister: StatePersister | None = None,
        metric_calculator: MetricCalculator | None = None,
        max_history: int = 1000,
        auto_persist: bool = True,
    ) -> None:
        """
        Initialize the Ratchet.

        Args:
            metric_name: Name of the metric being tracked
            comparison_mode: How to compare scores for improvement
            target_value: Target value for CLOSER_TO_TARGET mode
            tolerance: Tolerance for considering scores equal
            state_path: Path for file-based state persistence
            persister: Custom state persister (overrides state_path)
            metric_calculator: Custom metric calculator
            max_history: Maximum number of experiments to keep in history
            auto_persist: Automatically persist state after commits
        """
        self._metric_name = metric_name
        self._comparison_mode = comparison_mode
        self._target_value = target_value
        self._tolerance = tolerance
        self._max_history = max_history
        self._auto_persist = auto_persist

        # Initialize persister
        if persister:
            self._persister = persister
        elif state_path:
            self._persister = FileSystemPersister(state_path)
        else:
            self._persister = InMemoryPersister()

        # Initialize metric calculator
        if metric_calculator:
            self._metric_calculator = metric_calculator
        else:
            config = MetricConfig(
                name=metric_name,
                comparison_mode=comparison_mode,
                target_value=target_value,
                tolerance=tolerance,
            )
            self._metric_calculator = SimpleMetricCalculator(config)

        # State
        self._current_best_score: float | None = None
        self._current_best_experiment: ExperimentResult | None = None
        self._experiment_history: list[ExperimentResult] = []
        self._validation_history: list[ValidationResult] = []
        self._created_at = datetime.now()
        self._updated_at = datetime.now()

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Stats
        self._stats = RatchetStats()

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def metric_name(self) -> MetricName:
        """Get the metric name."""
        return self._metric_name

    @property
    def comparison_mode(self) -> ComparisonMode:
        """Get the comparison mode."""
        return self._comparison_mode

    @property
    def current_best_score(self) -> float | None:
        """Get the current best score."""
        return self._current_best_score

    @property
    def current_best_experiment(self) -> ExperimentResult | None:
        """Get the current best experiment."""
        return self._current_best_experiment

    @property
    def has_best(self) -> bool:
        """Check if a best score has been established."""
        return self._current_best_score is not None

    @property
    def experiment_count(self) -> int:
        """Get the number of experiments in history."""
        return len(self._experiment_history)

    @property
    def stats(self) -> RatchetStats:
        """Get current statistics."""
        return self._stats

    # ========================================================================
    # Core Validation Methods
    # ========================================================================

    def _compare_scores(
        self,
        new_score: float,
        current_score: float,
    ) -> tuple[ValidationDecision, float, float]:
        """
        Compare two scores and determine if new is better.

        Returns:
            Tuple of (decision, delta, percentage)
        """
        if self._comparison_mode == ComparisonMode.HIGHER_IS_BETTER:
            diff = new_score - current_score
            if abs(diff) <= self._tolerance:
                return ValidationDecision.TIE, 0.0, 0.0
            elif diff > 0:
                pct = (diff / current_score * 100) if current_score != 0 else float('inf')
                return ValidationDecision.KEEP, diff, pct
            else:
                pct = (diff / current_score * 100) if current_score != 0 else float('-inf')
                return ValidationDecision.DISCARD, diff, pct

        elif self._comparison_mode == ComparisonMode.LOWER_IS_BETTER:
            diff = current_score - new_score
            if abs(diff) <= self._tolerance:
                return ValidationDecision.TIE, 0.0, 0.0
            elif diff > 0:
                pct = (diff / current_score * 100) if current_score != 0 else float('inf')
                return ValidationDecision.KEEP, diff, pct
            else:
                pct = (diff / current_score * 100) if current_score != 0 else float('-inf')
                return ValidationDecision.DISCARD, diff, pct

        elif self._comparison_mode == ComparisonMode.CLOSER_TO_TARGET:
            assert self._target_value is not None
            current_dist = abs(current_score - self._target_value)
            new_dist = abs(new_score - self._target_value)
            diff = current_dist - new_dist

            if abs(diff) <= self._tolerance:
                return ValidationDecision.TIE, 0.0, 0.0
            elif diff > 0:
                pct = (diff / current_dist * 100) if current_dist != 0 else float('inf')
                return ValidationDecision.KEEP, diff, pct
            else:
                pct = (diff / current_dist * 100) if current_dist != 0 else float('-inf')
                return ValidationDecision.DISCARD, diff, pct

        raise ValidationError(f"Unknown comparison mode: {self._comparison_mode}")

    def _validate_locked(
        self,
        experiment_id: ExperimentID,
        score: float,
    ) -> ValidationResult:
        """Validate a score against current best. Caller must hold self._lock."""
        if self._current_best_score is None:
            return ValidationResult(
                experiment_id=experiment_id,
                score=score,
                decision=ValidationDecision.FIRST,
                is_improvement=True,
                previous_best=None,
                improvement_delta=0.0,
                improvement_pct=0.0,
                message=f"First experiment with score {score:.6f}",
            )

        decision, delta, pct = self._compare_scores(score, self._current_best_score)
        is_improvement = decision == ValidationDecision.KEEP

        if decision == ValidationDecision.KEEP:
            msg = f"Improvement: {score:.6f} vs {self._current_best_score:.6f} (+{pct:.2f}%)"
        elif decision == ValidationDecision.TIE:
            msg = f"Tie: {score:.6f} ≈ {self._current_best_score:.6f}"
        else:
            msg = f"Worse: {score:.6f} vs {self._current_best_score:.6f} ({pct:.2f}%)"

        return ValidationResult(
            experiment_id=experiment_id,
            score=score,
            decision=decision,
            is_improvement=is_improvement,
            previous_best=self._current_best_score,
            improvement_delta=delta,
            improvement_pct=pct,
            message=msg,
        )

    async def validate_experiment(
        self,
        experiment_id: ExperimentID,
        score: float,
        metadata: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """
        Validate a new experiment against the current best.

        This method only performs validation - it does not modify state.
        Use commit_experiment() to actually update the best score.

        Args:
            experiment_id: Unique identifier for the experiment
            score: The experiment's score
            metadata: Optional metadata about the experiment

        Returns:
            ValidationResult with decision and improvement metrics
        """
        async with self._lock:
            return self._validate_locked(experiment_id, score)

    async def validate_with_calculator(
        self,
        experiment_id: ExperimentID,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """
        Validate an experiment using the metric calculator.

        Args:
            experiment_id: Unique identifier for the experiment
            data: Data to calculate metric from
            metadata: Optional metadata about the experiment

        Returns:
            ValidationResult with decision and improvement metrics
        """
        score = await self._metric_calculator.calculate(experiment_id, data)
        return await self.validate_experiment(experiment_id, score, metadata)

    # ========================================================================
    # Commit Methods
    # ========================================================================

    async def commit_experiment(
        self,
        experiment_id: ExperimentID,
        score: float,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> ValidationResult:
        """
        Commit an experiment and update state if it's an improvement.

        This is a convenience method that validates and commits in one call.

        Args:
            experiment_id: Unique identifier for the experiment
            score: The experiment's score
            metadata: Optional metadata about the experiment
            metrics: Additional metrics to track
            artifacts: Paths to associated artifacts

        Returns:
            ValidationResult with decision
        """
        async with self._lock:
            # Validate first (we already hold the lock)
            result = self._validate_locked(experiment_id, score)

            # Create experiment result
            experiment = ExperimentResult(
                experiment_id=experiment_id,
                score=score,
                metadata=metadata or {},
                metrics=metrics or {},
                artifacts=artifacts or {},
            )

            # Add to history
            self._experiment_history.append(experiment)
            self._validation_history.append(result)

            # Update best if improvement
            if result.is_improvement:
                self._current_best_score = score
                self._current_best_experiment = experiment
                logger.info(
                    f"New best for '{self._metric_name}': {score:.6f} "
                    f"(previous: {result.previous_best})"
                )

            # Trim history if needed
            if len(self._experiment_history) > self._max_history:
                self._experiment_history = self._experiment_history[-self._max_history:]
            if len(self._validation_history) > self._max_history:
                self._validation_history = self._validation_history[-self._max_history:]

            # Update stats
            self._update_stats()
            self._updated_at = datetime.now()

            # Persist if enabled
            if self._auto_persist:
                await self._save_state_locked()

            return result

    async def force_commit(
        self,
        experiment_id: ExperimentID,
        score: float,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> ValidationResult:
        """
        Force commit an experiment as the new best, regardless of score.

        Args:
            experiment_id: Unique identifier for the experiment
            score: The experiment's score
            metadata: Optional metadata about the experiment
            metrics: Additional metrics to track
            artifacts: Paths to associated artifacts

        Returns:
            ValidationResult
        """
        async with self._lock:
            previous_best = self._current_best_score

            experiment = ExperimentResult(
                experiment_id=experiment_id,
                score=score,
                metadata=metadata or {},
                metrics=metrics or {},
                artifacts=artifacts or {},
            )

            self._current_best_score = score
            self._current_best_experiment = experiment
            self._experiment_history.append(experiment)

            result = ValidationResult(
                experiment_id=experiment_id,
                score=score,
                decision=ValidationDecision.KEEP,
                is_improvement=True,
                previous_best=previous_best,
                improvement_delta=score - (previous_best or 0),
                improvement_pct=0.0,
                message=f"Forced commit: {score:.6f}",
            )

            self._validation_history.append(result)
            self._update_stats()
            self._updated_at = datetime.now()

            if self._auto_persist:
                await self._save_state_locked()

            logger.info(f"Forced new best for '{self._metric_name}': {score:.6f}")
            return result

    def _update_stats(self) -> None:
        """Update internal statistics."""
        if not self._experiment_history:
            return

        scores = [e.score for e in self._experiment_history]
        higher = self._comparison_mode == ComparisonMode.HIGHER_IS_BETTER

        self._stats.total_experiments = len(self._experiment_history)
        self._stats.best_score = max(scores) if higher else min(scores)
        self._stats.worst_score = min(scores) if higher else max(scores)
        self._stats.avg_score = sum(scores) / len(scores)

        decisions = [v.decision for v in self._validation_history]
        improvements = sum(1 for d in decisions if d == ValidationDecision.KEEP)
        discards = sum(1 for d in decisions if d == ValidationDecision.DISCARD)
        ties = sum(1 for d in decisions if d == ValidationDecision.TIE)

        self._stats.improvements = improvements
        self._stats.discards = discards
        self._stats.ties = ties
        self._stats.improvement_rate = (
            improvements / len(self._validation_history) if self._validation_history else 0.0
        )

    # ========================================================================
    # State Persistence Methods
    # ========================================================================

    async def _save_state_locked(self) -> None:
        """Save current state to persister. Caller must hold self._lock."""
        state = RatchetState(
            metric_name=self._metric_name,
            comparison_mode=self._comparison_mode.name,
            current_best_score=self._current_best_score,
            current_best_experiment=self._current_best_experiment,
            experiment_history=self._experiment_history,
            validation_history=self._validation_history,
            created_at=self._created_at,
            updated_at=self._updated_at,
        )
        await self._persister.save(state)

    async def save_state(self) -> None:
        """Save current state to persister."""
        async with self._lock:
            await self._save_state_locked()

    async def load_state(self) -> bool:
        """
        Load state from persister.

        Returns:
            True if state was loaded, False if no state existed
        """
        async with self._lock:
            if not await self._persister.exists():
                return False

            state = await self._persister.load()
            if state is None:
                return False

            self._current_best_score = state.current_best_score
            self._current_best_experiment = state.current_best_experiment
            self._experiment_history = state.experiment_history
            self._validation_history = state.validation_history
            self._created_at = state.created_at
            self._updated_at = state.updated_at

            self._update_stats()

            logger.info(
                f"Loaded ratchet state for '{self._metric_name}': "
                f"best={self._current_best_score}, "
                f"experiments={len(self._experiment_history)}"
            )
            return True

    async def clear_state(self) -> bool:
        """
        Clear persisted state.

        Returns:
            True if state was cleared, False if no state existed
        """
        return await self._persister.delete()

    def export_state(self) -> RatchetState:
        """Export current state as a RatchetState object."""
        return RatchetState(
            metric_name=self._metric_name,
            comparison_mode=self._comparison_mode.name,
            current_best_score=self._current_best_score,
            current_best_experiment=self._current_best_experiment,
            experiment_history=self._experiment_history.copy(),
            validation_history=self._validation_history.copy(),
            created_at=self._created_at,
            updated_at=self._updated_at,
        )

    # ========================================================================
    # History Methods
    # ========================================================================

    async def get_experiment_history(
        self,
        limit: int | None = None,
    ) -> list[ExperimentResult]:
        """Get experiment history."""
        async with self._lock:
            history = self._experiment_history.copy()
            if limit:
                history = history[-limit:]
            return history

    async def get_validation_history(
        self,
        limit: int | None = None,
    ) -> list[ValidationResult]:
        """Get validation history."""
        async with self._lock:
            history = self._validation_history.copy()
            if limit:
                history = history[-limit:]
            return history

    async def get_experiment(self, experiment_id: ExperimentID) -> ExperimentResult | None:
        """Get a specific experiment by ID."""
        async with self._lock:
            for exp in self._experiment_history:
                if exp.experiment_id == experiment_id:
                    return exp
            return None

    async def clear_history(self) -> None:
        """Clear experiment and validation history (keeps current best)."""
        async with self._lock:
            self._experiment_history.clear()
            self._validation_history.clear()
            self._update_stats()
            self._updated_at = datetime.now()

            if self._auto_persist:
                await self._save_state_locked()

    # ========================================================================
    # Utility Methods
    # ========================================================================

    async def reset(self) -> None:
        """Reset the ratchet to initial state."""
        async with self._lock:
            self._current_best_score = None
            self._current_best_experiment = None
            self._experiment_history.clear()
            self._validation_history.clear()
            self._stats = RatchetStats()
            self._created_at = datetime.now()
            self._updated_at = datetime.now()

            if self._auto_persist:
                await self._save_state_locked()

            logger.info(f"Reset ratchet for '{self._metric_name}'")

    def to_dict(self) -> dict[str, Any]:
        """Convert ratchet to dictionary."""
        return {
            "metric_name": self._metric_name,
            "comparison_mode": self._comparison_mode.name,
            "current_best_score": self._current_best_score,
            "has_best": self.has_best,
            "experiment_count": self.experiment_count,
            "stats": self._stats.to_dict(),
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"Ratchet("
            f"metric='{self._metric_name}', "
            f"best={self._current_best_score}, "
            f"experiments={self.experiment_count}"
            f")"
        )


# ============================================================================
# Multi-Metric Ratchet
# ============================================================================

class MultiMetricRatchet:
    """
    Manages multiple ratchets for different metrics.

    Useful when tracking multiple related metrics simultaneously.

    Example:
        >>> mmr = MultiMetricRatchet()
        >>> mmr.add_ratchet(Ratchet("accuracy", ComparisonMode.HIGHER_IS_BETTER))
        >>> mmr.add_ratchet(Ratchet("loss", ComparisonMode.LOWER_IS_BETTER))
        >>>
        >>> # Validate against all metrics
        >>> results = await mmr.validate_experiment("exp1", {
        ...     "accuracy": 0.85,
        ...     "loss": 0.15
        ... })
    """

    def __init__(self) -> None:
        self._ratchets: dict[MetricName, Ratchet] = {}
        self._lock = asyncio.Lock()

    def add_ratchet(self, ratchet: Ratchet) -> None:
        """Add a ratchet for a metric."""
        self._ratchets[ratchet.metric_name] = ratchet

    def get_ratchet(self, metric_name: MetricName) -> Ratchet | None:
        """Get a ratchet by metric name."""
        return self._ratchets.get(metric_name)

    def remove_ratchet(self, metric_name: MetricName) -> Ratchet | None:
        """Remove a ratchet by metric name."""
        return self._ratchets.pop(metric_name, None)

    @property
    def metric_names(self) -> list[MetricName]:
        """Get all tracked metric names."""
        return list(self._ratchets.keys())

    async def validate_experiment(
        self,
        experiment_id: ExperimentID,
        scores: dict[MetricName, float],
        metadata: dict[str, Any] | None = None,
    ) -> dict[MetricName, ValidationResult]:
        """
        Validate an experiment against all metrics.

        Args:
            experiment_id: Unique identifier for the experiment
            scores: Dictionary mapping metric names to scores
            metadata: Optional metadata about the experiment

        Returns:
            Dictionary mapping metric names to validation results
        """
        async with self._lock:
            results: dict[MetricName, ValidationResult] = {}

            for metric_name, score in scores.items():
                ratchet = self._ratchets.get(metric_name)
                if ratchet:
                    results[metric_name] = await ratchet.validate_experiment(
                        experiment_id, score, metadata
                    )

            return results

    async def commit_experiment(
        self,
        experiment_id: ExperimentID,
        scores: dict[MetricName, float],
        metadata: dict[str, Any] | None = None,
    ) -> dict[MetricName, ValidationResult]:
        """
        Commit an experiment to all applicable ratchets.

        Args:
            experiment_id: Unique identifier for the experiment
            scores: Dictionary mapping metric names to scores
            metadata: Optional metadata about the experiment

        Returns:
            Dictionary mapping metric names to validation results
        """
        async with self._lock:
            results: dict[MetricName, ValidationResult] = {}

            for metric_name, score in scores.items():
                ratchet = self._ratchets.get(metric_name)
                if ratchet:
                    results[metric_name] = await ratchet.commit_experiment(
                        experiment_id, score, metadata
                    )

            return results

    async def save_all_states(self) -> None:
        """Save state for all ratchets."""
        async with self._lock:
            for ratchet in self._ratchets.values():
                await ratchet.save_state()

    async def load_all_states(self) -> dict[MetricName, bool]:
        """Load state for all ratchets."""
        async with self._lock:
            results = {}
            for name, ratchet in self._ratchets.items():
                results[name] = await ratchet.load_state()
            return results

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all ratchets."""
        return {
            "metrics": {
                name: {
                    "best_score": r.current_best_score,
                    "experiment_count": r.experiment_count,
                }
                for name, r in self._ratchets.items()
            }
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_accuracy_ratchet(
    state_path: str | Path | None = None,
    **kwargs: Any,
) -> Ratchet:
    """Create a ratchet configured for accuracy tracking (higher is better)."""
    return Ratchet(
        metric_name="accuracy",
        comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
        state_path=state_path,
        **kwargs,
    )


def create_loss_ratchet(
    state_path: str | Path | None = None,
    **kwargs: Any,
) -> Ratchet:
    """Create a ratchet configured for loss tracking (lower is better)."""
    return Ratchet(
        metric_name="loss",
        comparison_mode=ComparisonMode.LOWER_IS_BETTER,
        state_path=state_path,
        **kwargs,
    )


def create_target_ratchet(
    metric_name: str,
    target_value: float,
    state_path: str | Path | None = None,
    **kwargs: Any,
) -> Ratchet:
    """Create a ratchet configured for target-based tracking."""
    return Ratchet(
        metric_name=metric_name,
        comparison_mode=ComparisonMode.CLOSER_TO_TARGET,
        target_value=target_value,
        state_path=state_path,
        **kwargs,
    )


# ============================================================================
# Example Usage
# ============================================================================

async def example_usage() -> None:
    """Example of how to use the Ratchet."""

    # Create a ratchet for accuracy tracking
    ratchet = Ratchet(
        metric_name="accuracy",
        comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
    )

    # Simulate experiments
    experiments = [
        ("exp_001", 0.75),
        ("exp_002", 0.82),
        ("exp_003", 0.79),  # Worse, should be discarded
        ("exp_004", 0.85),  # New best
        ("exp_005", 0.85),  # Tie (within tolerance)
    ]

    for exp_id, score in experiments:
        result = await ratchet.commit_experiment(
            experiment_id=exp_id,
            score=score,
            metadata={"model_version": "1.0"},
        )

        print(f"{exp_id}: score={score:.2f}, decision={result.decision.value}, "
              f"improvement={result.is_improvement}")

    # Print final stats
    print(f"\nFinal best: {ratchet.current_best_score}")
    print(f"Stats: {ratchet.stats.to_dict()}")

    # Multi-metric example
    print("\n--- Multi-Metric Example ---")

    mmr = MultiMetricRatchet()
    mmr.add_ratchet(create_accuracy_ratchet())
    mmr.add_ratchet(create_loss_ratchet())

    results = await mmr.commit_experiment(
        experiment_id="exp_multi",
        scores={"accuracy": 0.88, "loss": 0.12},
        metadata={"epoch": 10},
    )

    for metric, result in results.items():
        print(f"{metric}: {result.score:.2f} -> {result.decision.value}")


# Note: example_usage() can be called manually for testing
# async def main():
#     await example_usage()
# asyncio.run(main())
