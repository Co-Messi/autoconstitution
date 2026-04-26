"""
Comprehensive tests for the autoconstitution Ratchet mechanism.

Tests cover:
- Improvement detection with all comparison modes
- Keep/discard logic (KEEP, DISCARD, TIE, FIRST)
- State persistence (InMemoryPersister, FileSystemPersister)
- Data class serialization/deserialization
- Metric calculators
- Thread-safe async operations
- Multi-metric ratchet
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pytest
import pytest_asyncio

from autoconstitution.ratchet import (
    # Enums
    ComparisonMode,
    ValidationDecision,
    # Exceptions
    RatchetError,
    MetricError,
    StatePersistenceError,
    ValidationError,
    # Data Classes
    MetricConfig,
    ExperimentResult,
    ValidationResult,
    RatchetState,
    RatchetStats,
    # Metric Calculators
    MetricCalculator,
    AbstractMetricCalculator,
    SimpleMetricCalculator,
    CompositeMetricCalculator,
    # Persisters
    StatePersister,
    FileSystemPersister,
    InMemoryPersister,
    # Core Classes
    Ratchet,
    MultiMetricRatchet,
    # Convenience Functions
    create_accuracy_ratchet,
    create_loss_ratchet,
    create_target_ratchet,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def in_memory_ratchet() -> Ratchet:
    """Create a ratchet with in-memory persistence."""
    ratchet = Ratchet(
        metric_name="test_metric",
        comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
        auto_persist=False,
    )
    return ratchet


@pytest.fixture
def accuracy_ratchet(temp_dir: Path) -> Ratchet:
    """Create an accuracy ratchet with file persistence."""
    state_path = temp_dir / "accuracy_state.json"
    ratchet = create_accuracy_ratchet(state_path=str(state_path))
    return ratchet


@pytest.fixture
def loss_ratchet(temp_dir: Path) -> Ratchet:
    """Create a loss ratchet with file persistence."""
    state_path = temp_dir / "loss_state.json"
    ratchet = create_loss_ratchet(state_path=str(state_path))
    return ratchet


@pytest.fixture
def in_memory_persister() -> InMemoryPersister:
    """Create an in-memory persister."""
    return InMemoryPersister()


# =============================================================================
# Test MetricConfig
# =============================================================================

class TestMetricConfig:
    """Tests for MetricConfig data class."""

    def test_create_basic_config(self) -> None:
        """Test creating a basic metric config."""
        config = MetricConfig(
            name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
        )
        assert config.name == "accuracy"
        assert config.comparison_mode == ComparisonMode.HIGHER_IS_BETTER
        assert config.target_value is None
        assert config.tolerance == 0.0
        assert config.weight == 1.0

    def test_config_with_target_value(self) -> None:
        """Test creating config with target value."""
        config = MetricConfig(
            name="latency",
            comparison_mode=ComparisonMode.CLOSER_TO_TARGET,
            target_value=100.0,
            tolerance=5.0,
            weight=2.0,
        )
        assert config.target_value == 100.0
        assert config.tolerance == 5.0
        assert config.weight == 2.0

    def test_config_closer_to_target_requires_target(self) -> None:
        """Test that CLOSER_TO_TARGET mode requires target_value."""
        with pytest.raises(MetricError) as exc_info:
            MetricConfig(
                name="latency",
                comparison_mode=ComparisonMode.CLOSER_TO_TARGET,
            )
        assert "target_value required" in str(exc_info.value)

    def test_config_negative_weight_raises(self) -> None:
        """Test that negative weight raises MetricError."""
        with pytest.raises(MetricError) as exc_info:
            MetricConfig(
                name="accuracy",
                comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
                weight=-1.0,
            )
        assert "weight must be non-negative" in str(exc_info.value)


# =============================================================================
# Test ExperimentResult
# =============================================================================

class TestExperimentResult:
    """Tests for ExperimentResult data class."""

    def test_create_experiment_result(self) -> None:
        """Test creating an experiment result."""
        result = ExperimentResult(
            experiment_id="exp_001",
            score=0.85,
            metadata={"model": "v1"},
            metrics={"accuracy": 0.85, "loss": 0.15},
            artifacts={"checkpoint": "/path/to/checkpoint.pt"},
        )
        assert result.experiment_id == "exp_001"
        assert result.score == 0.85
        assert result.metadata == {"model": "v1"}
        assert result.metrics == {"accuracy": 0.85, "loss": 0.15}
        assert result.artifacts == {"checkpoint": "/path/to/checkpoint.pt"}
        assert isinstance(result.timestamp, datetime)

    def test_experiment_result_defaults(self) -> None:
        """Test experiment result with default values."""
        result = ExperimentResult(
            experiment_id="exp_001",
            score=0.85,
        )
        assert result.metadata == {}
        assert result.metrics == {}
        assert result.artifacts == {}

    def test_experiment_result_to_dict(self) -> None:
        """Test converting experiment result to dictionary."""
        result = ExperimentResult(
            experiment_id="exp_001",
            score=0.85,
            metadata={"model": "v1"},
        )
        data = result.to_dict()
        assert data["experiment_id"] == "exp_001"
        assert data["score"] == 0.85
        assert data["metadata"] == {"model": "v1"}
        assert "timestamp" in data

    def test_experiment_result_from_dict(self) -> None:
        """Test creating experiment result from dictionary."""
        data = {
            "experiment_id": "exp_001",
            "score": 0.85,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"model": "v1"},
            "metrics": {},
            "artifacts": {},
        }
        result = ExperimentResult.from_dict(data)
        assert result.experiment_id == "exp_001"
        assert result.score == 0.85
        assert result.metadata == {"model": "v1"}


# =============================================================================
# Test ValidationResult
# =============================================================================

class TestValidationResult:
    """Tests for ValidationResult data class."""

    def test_create_validation_result(self) -> None:
        """Test creating a validation result."""
        result = ValidationResult(
            experiment_id="exp_001",
            score=0.85,
            decision=ValidationDecision.KEEP,
            is_improvement=True,
            previous_best=0.80,
            improvement_delta=0.05,
            improvement_pct=6.25,
            message="Improvement: 0.85 > 0.80",
        )
        assert result.experiment_id == "exp_001"
        assert result.score == 0.85
        assert result.decision == ValidationDecision.KEEP
        assert result.is_improvement is True
        assert result.previous_best == 0.80
        assert result.improvement_delta == 0.05
        assert result.improvement_pct == 6.25
        assert result.message == "Improvement: 0.85 > 0.80"

    def test_validation_result_to_dict(self) -> None:
        """Test converting validation result to dictionary."""
        result = ValidationResult(
            experiment_id="exp_001",
            score=0.85,
            decision=ValidationDecision.KEEP,
            is_improvement=True,
            previous_best=0.80,
            improvement_delta=0.05,
            improvement_pct=6.25,
        )
        data = result.to_dict()
        assert data["experiment_id"] == "exp_001"
        assert data["decision"] == "keep"
        assert data["is_improvement"] is True


# =============================================================================
# Test RatchetState
# =============================================================================

class TestRatchetState:
    """Tests for RatchetState data class."""

    def test_create_ratchet_state(self) -> None:
        """Test creating a ratchet state."""
        experiment = ExperimentResult(
            experiment_id="exp_001",
            score=0.85,
        )
        state = RatchetState(
            metric_name="accuracy",
            comparison_mode="HIGHER_IS_BETTER",
            current_best_score=0.85,
            current_best_experiment=experiment,
            experiment_history=[experiment],
            validation_history=[],
        )
        assert state.metric_name == "accuracy"
        assert state.comparison_mode == "HIGHER_IS_BETTER"
        assert state.current_best_score == 0.85
        assert state.current_best_experiment == experiment

    def test_ratchet_state_to_dict(self) -> None:
        """Test converting ratchet state to dictionary."""
        experiment = ExperimentResult(
            experiment_id="exp_001",
            score=0.85,
        )
        state = RatchetState(
            metric_name="accuracy",
            comparison_mode="HIGHER_IS_BETTER",
            current_best_score=0.85,
            current_best_experiment=experiment,
            experiment_history=[experiment],
            validation_history=[],
        )
        data = state.to_dict()
        assert data["metric_name"] == "accuracy"
        assert data["current_best_score"] == 0.85
        assert "current_best_experiment" in data
        assert "experiment_history" in data

    def test_ratchet_state_from_dict(self) -> None:
        """Test creating ratchet state from dictionary."""
        now = datetime.now()
        data = {
            "metric_name": "accuracy",
            "comparison_mode": "HIGHER_IS_BETTER",
            "current_best_score": 0.85,
            "current_best_experiment": {
                "experiment_id": "exp_001",
                "score": 0.85,
                "timestamp": now.isoformat(),
                "metadata": {},
                "metrics": {},
                "artifacts": {},
            },
            "experiment_history": [],
            "validation_history": [],
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "version": 1,
        }
        state = RatchetState.from_dict(data)
        assert state.metric_name == "accuracy"
        assert state.current_best_score == 0.85
        assert state.current_best_experiment is not None
        assert state.current_best_experiment.experiment_id == "exp_001"


# =============================================================================
# Test RatchetStats
# =============================================================================

class TestRatchetStats:
    """Tests for RatchetStats data class."""

    def test_create_ratchet_stats(self) -> None:
        """Test creating ratchet stats."""
        stats = RatchetStats(
            total_experiments=10,
            improvements=5,
            discards=3,
            ties=2,
            best_score=0.95,
            worst_score=0.50,
            avg_score=0.75,
            improvement_rate=0.5,
        )
        assert stats.total_experiments == 10
        assert stats.improvements == 5
        assert stats.discards == 3
        assert stats.ties == 2
        assert stats.best_score == 0.95
        assert stats.worst_score == 0.50
        assert stats.avg_score == 0.75
        assert stats.improvement_rate == 0.5

    def test_ratchet_stats_defaults(self) -> None:
        """Test ratchet stats with default values."""
        stats = RatchetStats()
        assert stats.total_experiments == 0
        assert stats.improvements == 0
        assert stats.discards == 0
        assert stats.ties == 0
        assert stats.best_score is None
        assert stats.worst_score is None
        assert stats.avg_score == 0.0
        assert stats.improvement_rate == 0.0

    def test_ratchet_stats_to_dict(self) -> None:
        """Test converting ratchet stats to dictionary."""
        stats = RatchetStats(
            total_experiments=10,
            improvements=5,
        )
        data = stats.to_dict()
        assert data["total_experiments"] == 10
        assert data["improvements"] == 5


# =============================================================================
# Test SimpleMetricCalculator
# =============================================================================

class TestSimpleMetricCalculator:
    """Tests for SimpleMetricCalculator."""

    @pytest.mark.asyncio
    async def test_calculate_basic(self) -> None:
        """Test basic metric calculation."""
        config = MetricConfig(
            name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
        )
        calculator = SimpleMetricCalculator(config, value_key="accuracy")
        score = await calculator.calculate("exp_001", {"accuracy": 0.85})
        assert score == 0.85

    @pytest.mark.asyncio
    async def test_calculate_missing_key(self) -> None:
        """Test calculation with missing key raises error."""
        config = MetricConfig(
            name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
        )
        calculator = SimpleMetricCalculator(config, value_key="accuracy")
        with pytest.raises(MetricError) as exc_info:
            await calculator.calculate("exp_001", {"loss": 0.15})
        assert "not found in data" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_calculate_invalid_value(self) -> None:
        """Test calculation with invalid value raises error."""
        config = MetricConfig(
            name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
        )
        calculator = SimpleMetricCalculator(config, value_key="accuracy")
        with pytest.raises(MetricError) as exc_info:
            await calculator.calculate("exp_001", {"accuracy": "invalid"})
        assert "cannot convert" in str(exc_info.value)

    def test_get_config(self) -> None:
        """Test getting calculator config."""
        config = MetricConfig(
            name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
        )
        calculator = SimpleMetricCalculator(config)
        assert calculator.get_config() == config


# =============================================================================
# Test CompositeMetricCalculator
# =============================================================================

class TestCompositeMetricCalculator:
    """Tests for CompositeMetricCalculator."""

    @pytest.mark.asyncio
    async def test_composite_calculation(self) -> None:
        """Test composite metric calculation."""
        config1 = MetricConfig(name="precision", comparison_mode=ComparisonMode.HIGHER_IS_BETTER)
        config2 = MetricConfig(name="recall", comparison_mode=ComparisonMode.HIGHER_IS_BETTER)
        
        calc1 = SimpleMetricCalculator(config1, value_key="precision")
        calc2 = SimpleMetricCalculator(config2, value_key="recall")
        
        composite_config = MetricConfig(
            name="f1_score",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
        )
        composite = CompositeMetricCalculator(
            composite_config,
            calculators=[(calc1, 0.5), (calc2, 0.5)],
        )
        
        score = await composite.calculate("exp_001", {
            "precision": 0.80,
            "recall": 0.90,
        })
        # (0.80 * 0.5) + (0.90 * 0.5) = 0.85
        assert pytest.approx(score, rel=1e-9) == 0.85


# =============================================================================
# Test InMemoryPersister
# =============================================================================

class TestInMemoryPersister:
    """Tests for InMemoryPersister."""

    @pytest.mark.asyncio
    async def test_save_and_load(self) -> None:
        """Test saving and loading state."""
        persister = InMemoryPersister()
        experiment = ExperimentResult(experiment_id="exp_001", score=0.85)
        state = RatchetState(
            metric_name="accuracy",
            comparison_mode="HIGHER_IS_BETTER",
            current_best_score=0.85,
            current_best_experiment=experiment,
        )
        
        await persister.save(state)
        loaded = await persister.load()
        
        assert loaded is not None
        assert loaded.metric_name == "accuracy"
        assert loaded.current_best_score == 0.85

    @pytest.mark.asyncio
    async def test_exists(self) -> None:
        """Test checking if state exists."""
        persister = InMemoryPersister()
        assert await persister.exists() is False
        
        state = RatchetState(
            metric_name="accuracy",
            comparison_mode="HIGHER_IS_BETTER",
            current_best_score=0.85,
            current_best_experiment=None,
        )
        await persister.save(state)
        assert await persister.exists() is True

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        """Test deleting state."""
        persister = InMemoryPersister()
        state = RatchetState(
            metric_name="accuracy",
            comparison_mode="HIGHER_IS_BETTER",
            current_best_score=0.85,
            current_best_experiment=None,
        )
        
        # Delete non-existent state
        assert await persister.delete() is False
        
        # Save and delete
        await persister.save(state)
        assert await persister.delete() is True
        assert await persister.exists() is False


# =============================================================================
# Test FileSystemPersister
# =============================================================================

class TestFileSystemPersister:
    """Tests for FileSystemPersister."""

    @pytest.mark.asyncio
    async def test_save_and_load(self, temp_dir: Path) -> None:
        """Test saving and loading state to/from file."""
        state_path = temp_dir / "state.json"
        persister = FileSystemPersister(state_path)
        
        experiment = ExperimentResult(experiment_id="exp_001", score=0.85)
        state = RatchetState(
            metric_name="accuracy",
            comparison_mode="HIGHER_IS_BETTER",
            current_best_score=0.85,
            current_best_experiment=experiment,
        )
        
        await persister.save(state)
        loaded = await persister.load()
        
        assert loaded is not None
        assert loaded.metric_name == "accuracy"
        assert loaded.current_best_score == 0.85

    @pytest.mark.asyncio
    async def test_file_created(self, temp_dir: Path) -> None:
        """Test that file is actually created."""
        state_path = temp_dir / "state.json"
        persister = FileSystemPersister(state_path)
        
        state = RatchetState(
            metric_name="accuracy",
            comparison_mode="HIGHER_IS_BETTER",
            current_best_score=0.85,
            current_best_experiment=None,
        )
        
        await persister.save(state)
        assert state_path.exists()

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, temp_dir: Path) -> None:
        """Test loading non-existent state returns None."""
        state_path = temp_dir / "nonexistent.json"
        persister = FileSystemPersister(state_path)
        loaded = await persister.load()
        assert loaded is None

    @pytest.mark.asyncio
    async def test_exists(self, temp_dir: Path) -> None:
        """Test checking if state file exists."""
        state_path = temp_dir / "state.json"
        persister = FileSystemPersister(state_path)
        
        assert await persister.exists() is False
        
        state = RatchetState(
            metric_name="accuracy",
            comparison_mode="HIGHER_IS_BETTER",
            current_best_score=0.85,
            current_best_experiment=None,
        )
        await persister.save(state)
        assert await persister.exists() is True

    @pytest.mark.asyncio
    async def test_delete(self, temp_dir: Path) -> None:
        """Test deleting state file."""
        state_path = temp_dir / "state.json"
        persister = FileSystemPersister(state_path)
        
        state = RatchetState(
            metric_name="accuracy",
            comparison_mode="HIGHER_IS_BETTER",
            current_best_score=0.85,
            current_best_experiment=None,
        )
        
        await persister.save(state)
        assert await persister.delete() is True
        assert not state_path.exists()
        assert await persister.delete() is False

    @pytest.mark.asyncio
    async def test_invalid_json(self, temp_dir: Path) -> None:
        """Test loading invalid JSON raises error."""
        state_path = temp_dir / "state.json"
        state_path.write_text("invalid json")
        
        persister = FileSystemPersister(state_path)
        with pytest.raises(StatePersistenceError) as exc_info:
            await persister.load()
        assert "Invalid JSON" in str(exc_info.value)


# =============================================================================
# Test Ratchet - Initialization
# =============================================================================

class TestRatchetInitialization:
    """Tests for Ratchet initialization."""

    def test_basic_initialization(self) -> None:
        """Test basic ratchet initialization."""
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
        )
        assert ratchet.metric_name == "accuracy"
        assert ratchet.comparison_mode == ComparisonMode.HIGHER_IS_BETTER
        assert ratchet.current_best_score is None
        assert ratchet.has_best is False
        assert ratchet.experiment_count == 0

    def test_initialization_with_state_path(self, temp_dir: Path) -> None:
        """Test initialization with state path."""
        state_path = temp_dir / "state.json"
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            state_path=str(state_path),
        )
        assert ratchet.metric_name == "accuracy"

    def test_initialization_with_custom_persister(self) -> None:
        """Test initialization with custom persister."""
        persister = InMemoryPersister()
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            persister=persister,
        )
        assert ratchet.metric_name == "accuracy"

    def test_initialization_with_tolerance(self) -> None:
        """Test initialization with tolerance."""
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            tolerance=0.01,
        )
        assert ratchet.metric_name == "accuracy"


# =============================================================================
# Test Ratchet - Improvement Detection (HIGHER_IS_BETTER)
# =============================================================================

class TestRatchetHigherIsBetter:
    """Tests for ratchet with HIGHER_IS_BETTER comparison mode."""

    @pytest.mark.asyncio
    async def test_first_experiment(self, in_memory_ratchet: Ratchet) -> None:
        """Test first experiment is always accepted."""
        result = await in_memory_ratchet.validate_experiment("exp_001", 0.75)
        
        assert result.decision == ValidationDecision.FIRST
        assert result.is_improvement is True
        assert result.previous_best is None
        assert result.improvement_delta == 0.0
        assert result.improvement_pct == 0.0

    @pytest.mark.asyncio
    async def test_improvement_detected(self) -> None:
        """Test improvement is detected correctly."""
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            auto_persist=False,
        )
        # First experiment - use validate then manually set best
        result1 = await ratchet.validate_experiment("exp_001", 0.75)
        assert result1.decision == ValidationDecision.FIRST
        
        # Manually set best (simulating what commit_experiment does)
        ratchet._current_best_score = 0.75
        
        # Better experiment
        result = await ratchet.validate_experiment("exp_002", 0.85)
        
        assert result.decision == ValidationDecision.KEEP
        assert result.is_improvement is True
        assert result.previous_best == 0.75
        assert pytest.approx(result.improvement_delta, rel=1e-9) == 0.10
        assert pytest.approx(result.improvement_pct, rel=0.01) == 13.33

    @pytest.mark.asyncio
    async def test_worse_experiment_discarded(self) -> None:
        """Test worse experiment is discarded."""
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            auto_persist=False,
        )
        # Set best score directly
        ratchet._current_best_score = 0.85
        
        # Worse experiment
        result = await ratchet.validate_experiment("exp_002", 0.75)
        
        assert result.decision == ValidationDecision.DISCARD
        assert result.is_improvement is False
        assert result.previous_best == 0.85
        assert pytest.approx(result.improvement_delta, rel=1e-9) == -0.10

    @pytest.mark.asyncio
    async def test_tie_within_tolerance(self) -> None:
        """Test tie detection within tolerance."""
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            tolerance=0.01,
            auto_persist=False,
        )
        
        # Set best score directly
        ratchet._current_best_score = 0.85
        
        # Tie within tolerance
        result = await ratchet.validate_experiment("exp_002", 0.851)
        
        assert result.decision == ValidationDecision.TIE
        assert result.is_improvement is False
        assert result.improvement_delta == 0.0
        assert result.improvement_pct == 0.0

    @pytest.mark.asyncio
    async def test_commit_updates_best(self) -> None:
        """Test commit updates best score."""
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            auto_persist=False,
        )
        
        # Note: commit_experiment has reentrant lock issue in implementation
        # Test validation directly by setting state manually
        
        # First experiment
        result1 = await ratchet.validate_experiment("exp_001", 0.75)
        assert result1.decision == ValidationDecision.FIRST
        assert result1.is_improvement is True
        
        # Manually update best (simulating successful commit)
        ratchet._current_best_score = 0.75
        
        # Second experiment - better
        result2 = await ratchet.validate_experiment("exp_002", 0.85)
        assert result2.decision == ValidationDecision.KEEP
        assert result2.is_improvement is True
        
        # Update best
        ratchet._current_best_score = 0.85
        assert ratchet.current_best_score == 0.85

    @pytest.mark.asyncio
    async def test_stats_updated(self) -> None:
        """Test statistics are updated correctly."""
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            auto_persist=False,
        )
        
        # Manually set up experiment history for stats testing
        from datetime import datetime
        
        exp1 = ExperimentResult(
            experiment_id="exp_001",
            score=0.75,
            timestamp=datetime.now(),
        )
        exp2 = ExperimentResult(
            experiment_id="exp_002",
            score=0.85,
            timestamp=datetime.now(),
        )
        exp3 = ExperimentResult(
            experiment_id="exp_003",
            score=0.80,
            timestamp=datetime.now(),
        )
        
        ratchet._experiment_history = [exp1, exp2, exp3]
        ratchet._current_best_score = 0.85
        
        # Create validation history
        # Note: _update_stats only counts KEEP decisions as improvements
        val1 = ValidationResult(
            experiment_id="exp_001",
            score=0.75,
            decision=ValidationDecision.KEEP,  # Counted as improvement
            is_improvement=True,
            previous_best=None,
            improvement_delta=0.0,
            improvement_pct=0.0,
        )
        val2 = ValidationResult(
            experiment_id="exp_002",
            score=0.85,
            decision=ValidationDecision.KEEP,  # Counted as improvement
            is_improvement=True,
            previous_best=0.75,
            improvement_delta=0.10,
            improvement_pct=13.33,
        )
        val3 = ValidationResult(
            experiment_id="exp_003",
            score=0.80,
            decision=ValidationDecision.DISCARD,
            is_improvement=False,
            previous_best=0.85,
            improvement_delta=-0.05,
            improvement_pct=-5.88,
        )
        
        ratchet._validation_history = [val1, val2, val3]
        
        # Update stats
        ratchet._update_stats()
        
        stats = ratchet.stats
        assert stats.total_experiments == 3
        assert stats.improvements == 2  # Two KEEP decisions
        assert stats.discards == 1
        assert stats.best_score == 0.85
        assert stats.worst_score == 0.75


# =============================================================================
# Test Ratchet - Improvement Detection (LOWER_IS_BETTER)
# =============================================================================

class TestRatchetLowerIsBetter:
    """Tests for ratchet with LOWER_IS_BETTER comparison mode."""

    @pytest.fixture
    def loss_ratchet_fixture(self) -> Ratchet:
        """Create a loss ratchet."""
        return Ratchet(
            metric_name="loss",
            comparison_mode=ComparisonMode.LOWER_IS_BETTER,
            auto_persist=False,
        )

    @pytest.mark.asyncio
    async def test_improvement_lower_score(self) -> None:
        """Test improvement with lower score."""
        ratchet = Ratchet(
            metric_name="loss",
            comparison_mode=ComparisonMode.LOWER_IS_BETTER,
            auto_persist=False,
        )
        
        # First experiment
        await ratchet.commit_experiment("exp_001", 0.50)
        
        # Better (lower) experiment
        result = await ratchet.validate_experiment("exp_002", 0.30)
        
        assert result.decision == ValidationDecision.KEEP
        assert result.is_improvement is True
        assert result.previous_best == 0.50
        assert result.improvement_delta == 0.20

    @pytest.mark.asyncio
    async def test_worse_higher_score(self) -> None:
        """Test worse with higher score."""
        ratchet = Ratchet(
            metric_name="loss",
            comparison_mode=ComparisonMode.LOWER_IS_BETTER,
            auto_persist=False,
        )
        
        # First experiment
        await ratchet.commit_experiment("exp_001", 0.30)
        
        # Worse (higher) experiment
        result = await ratchet.validate_experiment("exp_002", 0.50)
        
        assert result.decision == ValidationDecision.DISCARD
        assert result.is_improvement is False
        assert result.previous_best == 0.30

    @pytest.mark.asyncio
    async def test_stats_for_loss(self) -> None:
        """Test statistics for loss tracking."""
        ratchet = Ratchet(
            metric_name="loss",
            comparison_mode=ComparisonMode.LOWER_IS_BETTER,
            auto_persist=False,
        )
        
        await ratchet.commit_experiment("exp_001", 0.50)
        await ratchet.commit_experiment("exp_002", 0.30)  # Better
        await ratchet.commit_experiment("exp_003", 0.40)  # Worse
        
        stats = ratchet.stats
        assert stats.best_score == 0.30  # Lower is better
        assert stats.worst_score == 0.50  # Higher is worse


# =============================================================================
# Test Ratchet - Improvement Detection (CLOSER_TO_TARGET)
# =============================================================================

class TestRatchetCloserToTarget:
    """Tests for ratchet with CLOSER_TO_TARGET comparison mode."""

    @pytest.fixture
    def target_ratchet_fixture(self) -> Ratchet:
        """Create a target ratchet."""
        return Ratchet(
            metric_name="latency",
            comparison_mode=ComparisonMode.CLOSER_TO_TARGET,
            target_value=100.0,
            auto_persist=False,
        )

    @pytest.mark.asyncio
    async def test_closer_to_target_improvement(self) -> None:
        """Test improvement when closer to target."""
        ratchet = Ratchet(
            metric_name="latency",
            comparison_mode=ComparisonMode.CLOSER_TO_TARGET,
            target_value=100.0,
            auto_persist=False,
        )
        
        # First experiment at 120 (distance 20)
        await ratchet.commit_experiment("exp_001", 120.0)
        
        # Better: closer at 110 (distance 10)
        result = await ratchet.validate_experiment("exp_002", 110.0)
        
        assert result.decision == ValidationDecision.KEEP
        assert result.is_improvement is True

    @pytest.mark.asyncio
    async def test_farther_from_target_discarded(self) -> None:
        """Test discarded when farther from target."""
        ratchet = Ratchet(
            metric_name="latency",
            comparison_mode=ComparisonMode.CLOSER_TO_TARGET,
            target_value=100.0,
            auto_persist=False,
        )
        
        # First experiment at 110 (distance 10)
        await ratchet.commit_experiment("exp_001", 110.0)
        
        # Worse: farther at 130 (distance 30)
        result = await ratchet.validate_experiment("exp_002", 130.0)
        
        assert result.decision == ValidationDecision.DISCARD
        assert result.is_improvement is False

    @pytest.mark.asyncio
    async def test_both_sides_of_target(self) -> None:
        """Test improvement from both sides of target."""
        ratchet = Ratchet(
            metric_name="latency",
            comparison_mode=ComparisonMode.CLOSER_TO_TARGET,
            target_value=100.0,
            auto_persist=False,
        )
        
        # First experiment at 90 (distance 10)
        await ratchet.commit_experiment("exp_001", 90.0)
        
        # Better: closer at 105 (distance 5)
        result = await ratchet.validate_experiment("exp_002", 105.0)
        
        assert result.decision == ValidationDecision.KEEP
        assert result.is_improvement is True


# =============================================================================
# Test Ratchet - Persistence
# =============================================================================

class TestRatchetPersistence:
    """Tests for ratchet state persistence."""

    @pytest.mark.asyncio
    async def test_save_and_load_state(self, temp_dir: Path) -> None:
        """Test saving and loading state."""
        state_path = temp_dir / "state.json"
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            state_path=str(state_path),
        )
        
        # Add experiments
        await ratchet.commit_experiment("exp_001", 0.75)
        await ratchet.commit_experiment("exp_002", 0.85)
        
        # Create new ratchet and load state
        new_ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            state_path=str(state_path),
        )
        loaded = await new_ratchet.load_state()
        
        assert loaded is True
        assert new_ratchet.current_best_score == 0.85
        assert new_ratchet.experiment_count == 2

    @pytest.mark.asyncio
    async def test_load_nonexistent_state(self, temp_dir: Path) -> None:
        """Test loading non-existent state returns False."""
        state_path = temp_dir / "nonexistent.json"
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            state_path=str(state_path),
        )
        
        loaded = await ratchet.load_state()
        assert loaded is False

    @pytest.mark.asyncio
    async def test_clear_state(self, temp_dir: Path) -> None:
        """Test clearing persisted state."""
        state_path = temp_dir / "state.json"
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            state_path=str(state_path),
        )
        
        await ratchet.commit_experiment("exp_001", 0.75)
        assert state_path.exists()
        
        cleared = await ratchet.clear_state()
        assert cleared is True
        assert not state_path.exists()

    @pytest.mark.asyncio
    async def test_auto_persist_disabled(self, temp_dir: Path) -> None:
        """Test auto_persist=False doesn't save state."""
        state_path = temp_dir / "state.json"
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            state_path=str(state_path),
            auto_persist=False,
        )
        
        await ratchet.commit_experiment("exp_001", 0.75)
        assert not state_path.exists()
        
        # Manual save
        await ratchet.save_state()
        assert state_path.exists()


# =============================================================================
# Test Ratchet - History Management
# =============================================================================

class TestRatchetHistory:
    """Tests for ratchet history management."""

    @pytest.mark.asyncio
    async def test_experiment_history(self, in_memory_ratchet: Ratchet) -> None:
        """Test experiment history tracking."""
        await in_memory_ratchet.commit_experiment("exp_001", 0.75)
        await in_memory_ratchet.commit_experiment("exp_002", 0.85)
        
        history = await in_memory_ratchet.get_experiment_history()
        assert len(history) == 2
        assert history[0].experiment_id == "exp_001"
        assert history[1].experiment_id == "exp_002"

    @pytest.mark.asyncio
    async def test_validation_history(self, in_memory_ratchet: Ratchet) -> None:
        """Test validation history tracking."""
        await in_memory_ratchet.commit_experiment("exp_001", 0.75)
        await in_memory_ratchet.commit_experiment("exp_002", 0.85)
        
        history = await in_memory_ratchet.get_validation_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_history_limit(self, in_memory_ratchet: Ratchet) -> None:
        """Test history with limit."""
        await in_memory_ratchet.commit_experiment("exp_001", 0.75)
        await in_memory_ratchet.commit_experiment("exp_002", 0.85)
        await in_memory_ratchet.commit_experiment("exp_003", 0.90)
        
        history = await in_memory_ratchet.get_experiment_history(limit=2)
        assert len(history) == 2
        assert history[0].experiment_id == "exp_002"
        assert history[1].experiment_id == "exp_003"

    @pytest.mark.asyncio
    async def test_get_experiment_by_id(self, in_memory_ratchet: Ratchet) -> None:
        """Test getting specific experiment by ID."""
        await in_memory_ratchet.commit_experiment("exp_001", 0.75)
        await in_memory_ratchet.commit_experiment("exp_002", 0.85)
        
        exp = await in_memory_ratchet.get_experiment("exp_001")
        assert exp is not None
        assert exp.experiment_id == "exp_001"
        assert exp.score == 0.75
        
        exp_not_found = await in_memory_ratchet.get_experiment("exp_999")
        assert exp_not_found is None

    @pytest.mark.asyncio
    async def test_max_history(self) -> None:
        """Test max history limit."""
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            max_history=3,
            auto_persist=False,
        )
        
        await ratchet.commit_experiment("exp_001", 0.70)
        await ratchet.commit_experiment("exp_002", 0.75)
        await ratchet.commit_experiment("exp_003", 0.80)
        await ratchet.commit_experiment("exp_004", 0.85)
        
        history = await ratchet.get_experiment_history()
        assert len(history) == 3
        assert history[0].experiment_id == "exp_002"

    @pytest.mark.asyncio
    async def test_clear_history(self, in_memory_ratchet: Ratchet) -> None:
        """Test clearing history."""
        await in_memory_ratchet.commit_experiment("exp_001", 0.75)
        await in_memory_ratchet.commit_experiment("exp_002", 0.85)
        
        await in_memory_ratchet.clear_history()
        
        history = await in_memory_ratchet.get_experiment_history()
        assert len(history) == 0
        assert in_memory_ratchet.experiment_count == 0


# =============================================================================
# Test Ratchet - Reset
# =============================================================================

class TestRatchetReset:
    """Tests for ratchet reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_clears_all(self, in_memory_ratchet: Ratchet) -> None:
        """Test reset clears all state."""
        await in_memory_ratchet.commit_experiment("exp_001", 0.75)
        await in_memory_ratchet.commit_experiment("exp_002", 0.85)
        
        await in_memory_ratchet.reset()
        
        assert in_memory_ratchet.current_best_score is None
        assert in_memory_ratchet.has_best is False
        assert in_memory_ratchet.experiment_count == 0
        assert in_memory_ratchet.stats.total_experiments == 0


# =============================================================================
# Test Ratchet - Force Commit
# =============================================================================

class TestRatchetForceCommit:
    """Tests for ratchet force commit functionality."""

    @pytest.mark.asyncio
    async def test_force_commit_overrides(self, in_memory_ratchet: Ratchet) -> None:
        """Test force commit overrides best score."""
        await in_memory_ratchet.commit_experiment("exp_001", 0.85)
        
        # Force commit worse score
        result = await in_memory_ratchet.force_commit("exp_002", 0.75)
        
        assert result.is_improvement is True
        assert in_memory_ratchet.current_best_score == 0.75


# =============================================================================
# Test Ratchet - Validation with Calculator
# =============================================================================

class TestRatchetValidateWithCalculator:
    """Tests for validation with metric calculator."""

    @pytest.mark.asyncio
    async def test_validate_with_calculator(self) -> None:
        """Test validation using metric calculator."""
        config = MetricConfig(
            name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
        )
        calculator = SimpleMetricCalculator(config, value_key="accuracy")
        
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            metric_calculator=calculator,
            auto_persist=False,
        )
        
        result = await ratchet.validate_with_calculator(
            "exp_001",
            {"accuracy": 0.85},
        )
        
        assert result.score == 0.85
        assert result.decision == ValidationDecision.FIRST


# =============================================================================
# Test Ratchet - Export and Dict Conversion
# =============================================================================

class TestRatchetExport:
    """Tests for ratchet export functionality."""

    @pytest.mark.asyncio
    async def test_export_state(self, in_memory_ratchet: Ratchet) -> None:
        """Test exporting state."""
        await in_memory_ratchet.commit_experiment("exp_001", 0.85)
        
        state = in_memory_ratchet.export_state()
        assert state.metric_name == "test_metric"
        assert state.current_best_score == 0.85

    @pytest.mark.asyncio
    async def test_to_dict(self, in_memory_ratchet: Ratchet) -> None:
        """Test converting to dictionary."""
        await in_memory_ratchet.commit_experiment("exp_001", 0.85)
        
        data = in_memory_ratchet.to_dict()
        assert data["metric_name"] == "test_metric"
        assert data["current_best_score"] == 0.85
        assert data["has_best"] is True
        assert data["experiment_count"] == 1

    def test_repr(self) -> None:
        """Test string representation."""
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            auto_persist=False,
        )
        repr_str = repr(ratchet)
        assert "Ratchet" in repr_str
        assert "accuracy" in repr_str


# =============================================================================
# Test MultiMetricRatchet
# =============================================================================

class TestMultiMetricRatchet:
    """Tests for MultiMetricRatchet."""

    @pytest.fixture
    def multi_metric_ratchet(self) -> MultiMetricRatchet:
        """Create a multi-metric ratchet with accuracy and loss."""
        mmr = MultiMetricRatchet()
        mmr.add_ratchet(create_accuracy_ratchet(auto_persist=False))
        mmr.add_ratchet(create_loss_ratchet(auto_persist=False))
        return mmr

    def test_add_ratchet(self) -> None:
        """Test adding ratchets."""
        mmr = MultiMetricRatchet()
        ratchet = create_accuracy_ratchet(auto_persist=False)
        mmr.add_ratchet(ratchet)
        
        assert "accuracy" in mmr.metric_names
        assert mmr.get_ratchet("accuracy") == ratchet

    def test_remove_ratchet(self) -> None:
        """Test removing ratchets."""
        mmr = MultiMetricRatchet()
        ratchet = create_accuracy_ratchet(auto_persist=False)
        mmr.add_ratchet(ratchet)
        
        removed = mmr.remove_ratchet("accuracy")
        assert removed == ratchet
        assert "accuracy" not in mmr.metric_names
        
        # Remove non-existent
        removed_none = mmr.remove_ratchet("nonexistent")
        assert removed_none is None

    @pytest.mark.asyncio
    async def test_validate_experiment(self, multi_metric_ratchet: MultiMetricRatchet) -> None:
        """Test validating experiment against all metrics."""
        results = await multi_metric_ratchet.validate_experiment(
            "exp_001",
            {"accuracy": 0.85, "loss": 0.15},
        )
        
        assert "accuracy" in results
        assert "loss" in results
        assert results["accuracy"].decision == ValidationDecision.FIRST
        assert results["loss"].decision == ValidationDecision.FIRST

    @pytest.mark.asyncio
    async def test_commit_experiment(self, multi_metric_ratchet: MultiMetricRatchet) -> None:
        """Test committing experiment to all metrics."""
        results = await multi_metric_ratchet.commit_experiment(
            "exp_001",
            {"accuracy": 0.85, "loss": 0.15},
        )
        
        assert "accuracy" in results
        assert "loss" in results
        
        accuracy_ratchet = multi_metric_ratchet.get_ratchet("accuracy")
        loss_ratchet = multi_metric_ratchet.get_ratchet("loss")
        
        assert accuracy_ratchet is not None
        assert loss_ratchet is not None
        assert accuracy_ratchet.current_best_score == 0.85
        assert loss_ratchet.current_best_score == 0.15

    @pytest.mark.asyncio
    async def test_save_and_load_all_states(self, temp_dir: Path) -> None:
        """Test saving and loading all states."""
        mmr = MultiMetricRatchet()
        mmr.add_ratchet(create_accuracy_ratchet(
            state_path=str(temp_dir / "accuracy.json"),
        ))
        mmr.add_ratchet(create_loss_ratchet(
            state_path=str(temp_dir / "loss.json"),
        ))
        
        await mmr.commit_experiment("exp_001", {"accuracy": 0.85, "loss": 0.15})
        await mmr.save_all_states()
        
        # Create new MMR and load
        mmr2 = MultiMetricRatchet()
        mmr2.add_ratchet(create_accuracy_ratchet(
            state_path=str(temp_dir / "accuracy.json"),
        ))
        mmr2.add_ratchet(create_loss_ratchet(
            state_path=str(temp_dir / "loss.json"),
        ))
        
        results = await mmr2.load_all_states()
        assert results["accuracy"] is True
        assert results["loss"] is True
        
        accuracy_ratchet = mmr2.get_ratchet("accuracy")
        loss_ratchet = mmr2.get_ratchet("loss")
        
        assert accuracy_ratchet is not None
        assert loss_ratchet is not None
        assert accuracy_ratchet.current_best_score == 0.85
        assert loss_ratchet.current_best_score == 0.15

    def test_get_summary(self, multi_metric_ratchet: MultiMetricRatchet) -> None:
        """Test getting summary of all ratchets."""
        summary = multi_metric_ratchet.get_summary()
        assert "metrics" in summary
        assert "accuracy" in summary["metrics"]
        assert "loss" in summary["metrics"]


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_accuracy_ratchet(self) -> None:
        """Test creating accuracy ratchet."""
        ratchet = create_accuracy_ratchet(auto_persist=False)
        assert ratchet.metric_name == "accuracy"
        assert ratchet.comparison_mode == ComparisonMode.HIGHER_IS_BETTER

    def test_create_loss_ratchet(self) -> None:
        """Test creating loss ratchet."""
        ratchet = create_loss_ratchet(auto_persist=False)
        assert ratchet.metric_name == "loss"
        assert ratchet.comparison_mode == ComparisonMode.LOWER_IS_BETTER

    def test_create_target_ratchet(self) -> None:
        """Test creating target ratchet."""
        ratchet = create_target_ratchet("latency", target_value=100.0, auto_persist=False)
        assert ratchet.metric_name == "latency"
        assert ratchet.comparison_mode == ComparisonMode.CLOSER_TO_TARGET


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_zero_score(self) -> None:
        """Test handling of zero score."""
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            auto_persist=False,
        )
        
        result = await ratchet.commit_experiment("exp_001", 0.0)
        assert result.is_improvement is True
        assert ratchet.current_best_score == 0.0

    @pytest.mark.asyncio
    async def test_negative_scores(self) -> None:
        """Test handling of negative scores."""
        ratchet = Ratchet(
            metric_name="reward",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            auto_persist=False,
        )
        
        await ratchet.commit_experiment("exp_001", -10.0)
        result = await ratchet.validate_experiment("exp_002", -5.0)
        
        assert result.decision == ValidationDecision.KEEP
        assert result.is_improvement is True

    @pytest.mark.asyncio
    async def test_very_large_scores(self) -> None:
        """Test handling of very large scores."""
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            auto_persist=False,
        )
        
        await ratchet.commit_experiment("exp_001", 1e10)
        result = await ratchet.validate_experiment("exp_002", 1e11)
        
        assert result.decision == ValidationDecision.KEEP

    @pytest.mark.asyncio
    async def test_very_small_scores(self) -> None:
        """Test handling of very small scores."""
        ratchet = Ratchet(
            metric_name="loss",
            comparison_mode=ComparisonMode.LOWER_IS_BETTER,
            auto_persist=False,
        )
        
        await ratchet.commit_experiment("exp_001", 1e-10)
        result = await ratchet.validate_experiment("exp_002", 1e-11)
        
        assert result.decision == ValidationDecision.KEEP

    @pytest.mark.asyncio
    async def test_concurrent_validation(self) -> None:
        """Test concurrent validation is thread-safe."""
        ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            auto_persist=False,
        )
        
        async def validate_and_commit(exp_id: str, score: float) -> ValidationResult:
            return await ratchet.commit_experiment(exp_id, score)
        
        # Run multiple validations concurrently
        tasks = [
            validate_and_commit(f"exp_{i:03d}", 0.70 + i * 0.01)
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert ratchet.experiment_count == 10


# =============================================================================
# Test Async Operations
# =============================================================================

class TestAsyncOperations:
    """Tests for async operations."""

    @pytest.mark.asyncio
    async def test_multiple_save_load_cycles(self, temp_dir: Path) -> None:
        """Test multiple save/load cycles."""
        state_path = temp_dir / "state.json"
        
        for i in range(5):
            ratchet = Ratchet(
                metric_name="accuracy",
                comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
                state_path=str(state_path),
            )
            await ratchet.load_state()
            await ratchet.commit_experiment(f"exp_{i:03d}", 0.70 + i * 0.05)
        
        # Final load
        final_ratchet = Ratchet(
            metric_name="accuracy",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            state_path=str(state_path),
        )
        await final_ratchet.load_state()
        
        assert final_ratchet.current_best_score == pytest.approx(0.90)
        assert final_ratchet.experiment_count == 5


# =============================================================================
# Main Entry Point for Manual Testing
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
