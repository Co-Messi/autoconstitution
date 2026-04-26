"""
ResearcherAgent for autoconstitution.

This module implements the ResearcherAgent, which is responsible for:
- Generating hypotheses for system improvements
- Designing experiments to validate hypotheses
- Interpreting experimental results
- Proposing code changes based on cross-pollination findings

Example:
    >>> from autoconstitution.agents.researcher import ResearcherAgent
    >>> agent = ResearcherAgent(name="researcher_1")
    >>> hypothesis = await agent.generate_hypothesis(observations)
    >>> experiment = await agent.design_experiment(hypothesis)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

logger = logging.getLogger(__name__)


class IncompletePatchError(ValueError):
    """Raised when a proposed code change arrives without a real code patch.

    Returning a ``"# TODO: ..."`` string-literal as a patch is a silent bug:
    downstream code that tries to apply or diff the patch treats the comment
    as a no-op and succeeds without changing anything. This exception makes
    the missing-patch case explicit so the caller can drop-with-log or retry
    with a stricter prompt.
    """


def _extract_patch(
    content: dict[str, object],
    key: str,
    *,
    kind: str,
    source_agent: str,
) -> str:
    """Pull ``key`` out of an LLM-response dict or raise :class:`IncompletePatchError`.

    The patch must be a non-empty string that does NOT start with the telltale
    ``"# TODO:"`` marker — older code would use that marker as a placeholder
    when the real patch was missing, and we treat it the same as missing.
    """
    patch = content.get(key)
    if not isinstance(patch, str) or not patch.strip():
        raise IncompletePatchError(
            f"{kind} change proposal from {source_agent!r} had no {key!r} field."
        )
    if patch.lstrip().startswith("# TODO:"):
        raise IncompletePatchError(
            f"{kind} change proposal from {source_agent!r} shipped a TODO "
            f"placeholder instead of a real patch: {patch[:80]!r}."
        )
    return patch
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    TypeVar,
    Union,
)

# Type variables for generic types
T = TypeVar("T")
H = TypeVar("H", bound="Hypothesis")
E = TypeVar("E", bound="Experiment")
R = TypeVar("R", bound="ExperimentResult")


# ============================================================================
# Enums and Constants
# ============================================================================

class HypothesisStatus(Enum):
    """Status of a hypothesis in the research lifecycle."""
    PENDING = auto()
    VALIDATED = auto()
    REJECTED = auto()
    UNDER_INVESTIGATION = auto()
    SUPERSEDED = auto()


class ExperimentStatus(Enum):
    """Status of an experiment."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class ResultInterpretation(Enum):
    """Interpretation of experimental results."""
    SUPPORTS_HYPOTHESIS = auto()
    CONTRADICTS_HYPOTHESIS = auto()
    INCONCLUSIVE = auto()
    REQUIRES_FURTHER_TESTING = auto()


class ChangePriority(Enum):
    """Priority level for proposed code changes."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    TRIVIAL = 5


class ChangeType(Enum):
    """Type of code change proposed."""
    OPTIMIZATION = auto()
    BUG_FIX = auto()
    FEATURE_ADDITION = auto()
    REFACTORING = auto()
    CONFIGURATION = auto()
    ARCHITECTURE = auto()


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True)
class Metric:
    """A measurable metric for evaluation."""
    name: str
    value: float
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    """Represents a research hypothesis for system improvement.
    
    Attributes:
        id: Unique identifier for the hypothesis
        statement: The hypothesis statement
        rationale: Explanation of why this hypothesis was formed
        predicted_outcome: Expected result if hypothesis is true
        status: Current status in the research lifecycle
        confidence: Confidence score (0.0 to 1.0)
        parent_ids: IDs of hypotheses this builds upon
        created_at: Creation timestamp
        validated_at: Validation timestamp (if validated)
        metrics: Associated metrics
        tags: Categorization tags
    """
    id: str
    statement: str
    rationale: str
    predicted_outcome: str
    status: HypothesisStatus = HypothesisStatus.PENDING
    confidence: float = 0.5
    parent_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    validated_at: Optional[datetime] = None
    metrics: List[Metric] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(
                self, 
                "id", 
                hashlib.sha256(
                    f"{self.statement}{self.created_at}".encode()
                ).hexdigest()[:16]
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hypothesis to dictionary representation."""
        return {
            "id": self.id,
            "statement": self.statement,
            "rationale": self.rationale,
            "predicted_outcome": self.predicted_outcome,
            "status": self.status.name,
            "confidence": self.confidence,
            "parent_ids": self.parent_ids,
            "created_at": self.created_at.isoformat(),
            "validated_at": self.validated_at.isoformat() if self.validated_at else None,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in self.metrics
            ],
            "tags": list(self.tags),
        }


@dataclass
class ExperimentDesign:
    """Design parameters for an experiment.
    
    Attributes:
        control_variables: Variables held constant
        independent_variables: Variables being manipulated
        dependent_variables: Variables being measured
        sample_size: Number of samples/repetitions
        duration_seconds: Expected experiment duration
        success_criteria: Criteria for experiment success
        failure_criteria: Criteria for experiment failure
    """
    control_variables: Dict[str, Any] = field(default_factory=dict)
    independent_variables: Dict[str, Any] = field(default_factory=dict)
    dependent_variables: List[str] = field(default_factory=list)
    sample_size: int = 1
    duration_seconds: float = 60.0
    success_criteria: List[str] = field(default_factory=list)
    failure_criteria: List[str] = field(default_factory=list)


@dataclass
class Experiment:
    """Represents an experiment to validate a hypothesis.
    
    Attributes:
        id: Unique identifier for the experiment
        hypothesis_id: ID of the hypothesis being tested
        design: Experiment design parameters
        status: Current experiment status
        results: Experiment results (if completed)
        started_at: Start timestamp
        completed_at: Completion timestamp
        error_message: Error message (if failed)
    """
    id: str
    hypothesis_id: str
    design: ExperimentDesign
    status: ExperimentStatus = ExperimentStatus.PENDING
    results: Optional[ExperimentResult] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(
                self,
                "id",
                hashlib.sha256(
                    f"{self.hypothesis_id}{datetime.now()}".encode()
                ).hexdigest()[:16]
            )


@dataclass
class ExperimentResult:
    """Results from an experiment execution.
    
    Attributes:
        experiment_id: ID of the associated experiment
        metrics: Collected metrics
        observations: Qualitative observations
        raw_data: Raw experimental data
        interpretation: Interpretation of results
        confidence: Confidence in the interpretation
        analyzed_at: Analysis timestamp
    """
    experiment_id: str
    metrics: List[Metric] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    interpretation: ResultInterpretation = ResultInterpretation.INCONCLUSIVE
    confidence: float = 0.5
    analyzed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "experiment_id": self.experiment_id,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                }
                for m in self.metrics
            ],
            "observations": self.observations,
            "interpretation": self.interpretation.name,
            "confidence": self.confidence,
            "analyzed_at": self.analyzed_at.isoformat(),
        }


@dataclass
class CodeChange:
    """Represents a proposed code change.
    
    Attributes:
        id: Unique identifier for the change
        description: Human-readable description
        change_type: Type of change
        priority: Priority level
        target_files: Files to be modified
        code_patch: The actual code changes (diff format)
        rationale: Why this change is proposed
        expected_impact: Expected performance/behavior impact
        dependencies: IDs of changes this depends on
        hypothesis_ids: Hypotheses that support this change
        created_at: Creation timestamp
    """
    id: str
    description: str
    change_type: ChangeType
    priority: ChangePriority
    target_files: List[str]
    code_patch: str
    rationale: str
    expected_impact: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    hypothesis_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(
                self,
                "id",
                hashlib.sha256(
                    f"{self.description}{self.created_at}".encode()
                ).hexdigest()[:16]
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert code change to dictionary representation."""
        return {
            "id": self.id,
            "description": self.description,
            "change_type": self.change_type.name,
            "priority": self.priority.name,
            "target_files": self.target_files,
            "rationale": self.rationale,
            "expected_impact": self.expected_impact,
            "dependencies": self.dependencies,
            "hypothesis_ids": self.hypothesis_ids,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class CrossPollinationFinding:
    """A finding from the cross-pollination bus.
    
    Attributes:
        source_agent: Agent that produced the finding
        finding_type: Type of finding
        content: The actual finding content
        metadata: Additional context
        timestamp: When the finding was recorded
    """
    source_agent: str
    finding_type: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Base Agent Abstract Class
# ============================================================================

class BaseAgent(ABC):
    """Abstract base class for all swarm agents.
    
    Provides common functionality for agent lifecycle management,
    communication, and state handling.
    
    Attributes:
        name: Unique agent identifier
        is_running: Whether the agent is currently active
        message_queue: Async queue for incoming messages
    """
    
    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize the base agent.
        
        Args:
            name: Unique identifier for this agent
            **kwargs: Additional configuration parameters
        """
        self.name = name
        self._is_running = False
        self._message_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._config = kwargs
        self._lock = asyncio.Lock()
    
    @property
    def is_running(self) -> bool:
        """Check if the agent is currently running."""
        return self._is_running
    
    async def start(self) -> None:
        """Start the agent's main processing loop."""
        async with self._lock:
            if not self._is_running:
                self._is_running = True
                await self._on_start()
    
    async def stop(self) -> None:
        """Stop the agent's main processing loop."""
        async with self._lock:
            if self._is_running:
                self._is_running = False
                await self._on_stop()
    
    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send a message to this agent.
        
        Args:
            message: The message to send
        """
        await self._message_queue.put(message)
    
    @abstractmethod
    async def process(self) -> None:
        """Main processing loop - must be implemented by subclasses."""
        pass
    
    async def _on_start(self) -> None:
        """Hook called when agent starts - override in subclasses."""
        pass
    
    async def _on_stop(self) -> None:
        """Hook called when agent stops - override in subclasses."""
        pass
    
    async def get_message(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get a message from the queue.
        
        Args:
            timeout: Maximum time to wait for a message
            
        Returns:
            The message, or None if timeout expires
        """
        try:
            return await asyncio.wait_for(
                self._message_queue.get(), 
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None


# ============================================================================
# Researcher Agent
# ============================================================================

class ResearcherAgent(BaseAgent):
    """Agent responsible for research activities in the swarm.
    
    The ResearcherAgent generates hypotheses, designs experiments,
    interprets results, and proposes code changes based on findings
    from the cross-pollination bus.
    
    Attributes:
        hypotheses: Active hypotheses being investigated
        experiments: Experiments designed or running
        findings: Cross-pollination findings to process
        proposed_changes: Code changes proposed by this agent
        
    Example:
        >>> agent = ResearcherAgent(name="researcher_1")
        >>> await agent.start()
        >>> hypothesis = await agent.generate_hypothesis(
        ...     observations=["System latency increased by 20%"]
        ... )
        >>> experiment = await agent.design_experiment(hypothesis)
    """
    
    def __init__(
        self,
        name: str,
        hypothesis_generators: Optional[List[Callable[..., Coroutine[Any, Any, Hypothesis]]]] = None,
        experiment_runners: Optional[List[Callable[..., Coroutine[Any, Any, ExperimentResult]]]] = None,
        max_concurrent_experiments: int = 3,
        **kwargs: Any
    ) -> None:
        """Initialize the ResearcherAgent.
        
        Args:
            name: Unique identifier for this agent
            hypothesis_generators: Optional custom hypothesis generators
            experiment_runners: Optional custom experiment runners
            max_concurrent_experiments: Maximum parallel experiments
            **kwargs: Additional configuration parameters
        """
        super().__init__(name, **kwargs)
        
        # Storage for research artifacts
        self._hypotheses: Dict[str, Hypothesis] = {}
        self._experiments: Dict[str, Experiment] = {}
        self._findings: List[CrossPollinationFinding] = []
        self._proposed_changes: Dict[str, CodeChange] = {}
        
        # Custom generators and runners
        self._hypothesis_generators = hypothesis_generators or []
        self._experiment_runners = experiment_runners or []
        
        # Concurrency control
        self._max_concurrent_experiments = max_concurrent_experiments
        self._experiment_semaphore = asyncio.Semaphore(max_concurrent_experiments)
        
        # Processing task
        self._processing_task: Optional[asyncio.Task[None]] = None
        
        # Event callbacks
        self._on_hypothesis_generated: Optional[Callable[[Hypothesis], Coroutine[Any, Any, None]]] = None
        self._on_experiment_completed: Optional[Callable[[Experiment], Coroutine[Any, Any, None]]] = None
        self._on_change_proposed: Optional[Callable[[CodeChange], Coroutine[Any, Any, None]]] = None
    
    # ========================================================================
    # Properties
    # ========================================================================
    
    @property
    def hypotheses(self) -> Dict[str, Hypothesis]:
        """Get all hypotheses managed by this agent."""
        return self._hypotheses.copy()
    
    @property
    def active_hypotheses(self) -> List[Hypothesis]:
        """Get hypotheses currently under investigation."""
        return [
            h for h in self._hypotheses.values()
            if h.status == HypothesisStatus.UNDER_INVESTIGATION
        ]
    
    @property
    def pending_hypotheses(self) -> List[Hypothesis]:
        """Get hypotheses awaiting investigation."""
        return [
            h for h in self._hypotheses.values()
            if h.status == HypothesisStatus.PENDING
        ]
    
    @property
    def validated_hypotheses(self) -> List[Hypothesis]:
        """Get validated hypotheses."""
        return [
            h for h in self._hypotheses.values()
            if h.status == HypothesisStatus.VALIDATED
        ]
    
    @property
    def experiments(self) -> Dict[str, Experiment]:
        """Get all experiments managed by this agent."""
        return self._experiments.copy()
    
    @property
    def running_experiments(self) -> List[Experiment]:
        """Get currently running experiments."""
        return [
            e for e in self._experiments.values()
            if e.status == ExperimentStatus.RUNNING
        ]
    
    @property
    def proposed_changes(self) -> Dict[str, CodeChange]:
        """Get all proposed code changes."""
        return self._proposed_changes.copy()
    
    # ========================================================================
    # Event Callbacks
    # ========================================================================
    
    def on_hypothesis_generated(
        self, 
        callback: Callable[[Hypothesis], Coroutine[Any, Any, None]]
    ) -> None:
        """Set callback for hypothesis generation events."""
        self._on_hypothesis_generated = callback
    
    def on_experiment_completed(
        self,
        callback: Callable[[Experiment], Coroutine[Any, Any, None]]
    ) -> None:
        """Set callback for experiment completion events."""
        self._on_experiment_completed = callback
    
    def on_change_proposed(
        self,
        callback: Callable[[CodeChange], Coroutine[Any, Any, None]]
    ) -> None:
        """Set callback for code change proposal events."""
        self._on_change_proposed = callback
    
    # ========================================================================
    # Hypothesis Generation
    # ========================================================================
    
    async def generate_hypothesis(
        self,
        observations: List[str],
        context: Optional[Dict[str, Any]] = None,
        parent_hypothesis_ids: Optional[List[str]] = None,
        tags: Optional[Set[str]] = None
    ) -> Hypothesis:
        """Generate a new hypothesis based on observations.
        
        Analyzes observations and context to form a testable hypothesis
        about system improvements.
        
        Args:
            observations: List of observations to base hypothesis on
            context: Additional context for hypothesis generation
            parent_hypothesis_ids: IDs of parent hypotheses
            tags: Categorization tags
            
        Returns:
            The generated hypothesis
            
        Raises:
            ValueError: If observations list is empty
        """
        if not observations:
            raise ValueError("At least one observation is required")
        
        context = context or {}
        parent_ids = parent_hypothesis_ids or []
        tags = tags or set()
        
        # Try custom generators first
        for generator in self._hypothesis_generators:
            try:
                hypothesis = await generator(
                    observations=observations,
                    context=context,
                    parent_ids=parent_ids,
                    tags=tags
                )
                if hypothesis:
                    self._hypotheses[hypothesis.id] = hypothesis
                    if self._on_hypothesis_generated:
                        await self._on_hypothesis_generated(hypothesis)
                    return hypothesis
            except Exception as e:
                # Log error and continue to next generator
                continue
        
        # Default hypothesis generation
        hypothesis = await self._default_hypothesis_generation(
            observations, context, parent_ids, tags
        )
        
        self._hypotheses[hypothesis.id] = hypothesis
        
        if self._on_hypothesis_generated:
            await self._on_hypothesis_generated(hypothesis)
        
        return hypothesis
    
    async def _default_hypothesis_generation(
        self,
        observations: List[str],
        context: Dict[str, Any],
        parent_ids: List[str],
        tags: Set[str]
    ) -> Hypothesis:
        """Default hypothesis generation logic.
        
        Analyzes observations to identify patterns and form hypotheses.
        """
        # Combine observations into analysis context
        combined_obs = " | ".join(observations)
        
        # Identify potential improvement areas
        improvement_areas = self._identify_improvement_areas(observations)
        
        # Generate hypothesis statement based on patterns
        if improvement_areas:
            area = improvement_areas[0]
            statement = f"Improving {area} will enhance overall system performance"
            rationale = f"Based on observations: {combined_obs[:200]}"
            predicted_outcome = f"System {area} will show measurable improvement"
        else:
            statement = "System optimization through parameter tuning will improve performance"
            rationale = f"General optimization based on: {combined_obs[:200]}"
            predicted_outcome = "Performance metrics will improve by at least 10%"
        
        # Calculate initial confidence based on observation quality
        confidence = self._calculate_observation_confidence(observations, context)
        
        # Add relevant tags
        tags.update(improvement_areas)
        tags.add("auto-generated")
        
        hypothesis = Hypothesis(
            id="",
            statement=statement,
            rationale=rationale,
            predicted_outcome=predicted_outcome,
            status=HypothesisStatus.PENDING,
            confidence=confidence,
            parent_ids=parent_ids,
            tags=tags,
        )
        
        return hypothesis
    
    def _identify_improvement_areas(self, observations: List[str]) -> List[str]:
        """Identify potential improvement areas from observations."""
        areas = []
        keywords = {
            "latency": ["slow", "delay", "latency", "response time"],
            "throughput": ["bottleneck", "throughput", "rate", "capacity"],
            "memory": ["memory", "ram", "leak", "consumption"],
            "cpu": ["cpu", "processing", "compute", "utilization"],
            "error_rate": ["error", "failure", "exception", "crash"],
            "scalability": ["scale", "concurrent", "load", "traffic"],
        }
        
        combined = " ".join(observations).lower()
        
        for area, terms in keywords.items():
            if any(term in combined for term in terms):
                areas.append(area)
        
        return areas
    
    def _calculate_observation_confidence(
        self,
        observations: List[str],
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence score based on observation quality."""
        base_confidence = 0.5
        
        # More observations increase confidence
        base_confidence += min(len(observations) * 0.05, 0.2)
        
        # Context with metrics increases confidence
        if "metrics" in context:
            base_confidence += 0.1
        
        # Historical success increases confidence
        if "historical_success_rate" in context:
            base_confidence += context["historical_success_rate"] * 0.2
        
        return min(base_confidence, 1.0)
    
    async def generate_hypotheses_batch(
        self,
        observation_sets: List[List[str]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Hypothesis]:
        """Generate multiple hypotheses in parallel.
        
        Args:
            observation_sets: List of observation sets
            context: Shared context for all hypotheses
            
        Returns:
            List of generated hypotheses
        """
        tasks = [
            self.generate_hypothesis(obs, context)
            for obs in observation_sets
        ]
        return await asyncio.gather(*tasks)
    
    async def update_hypothesis_status(
        self,
        hypothesis_id: str,
        new_status: HypothesisStatus,
        confidence: Optional[float] = None
    ) -> Optional[Hypothesis]:
        """Update the status of a hypothesis.
        
        Args:
            hypothesis_id: ID of the hypothesis to update
            new_status: New status to set
            confidence: Optional new confidence score
            
        Returns:
            Updated hypothesis, or None if not found
        """
        if hypothesis_id not in self._hypotheses:
            return None
        
        hypothesis = self._hypotheses[hypothesis_id]
        
        # Create updated hypothesis
        updated = Hypothesis(
            id=hypothesis.id,
            statement=hypothesis.statement,
            rationale=hypothesis.rationale,
            predicted_outcome=hypothesis.predicted_outcome,
            status=new_status,
            confidence=confidence if confidence is not None else hypothesis.confidence,
            parent_ids=hypothesis.parent_ids,
            created_at=hypothesis.created_at,
            validated_at=datetime.now() if new_status == HypothesisStatus.VALIDATED else hypothesis.validated_at,
            metrics=hypothesis.metrics,
            tags=hypothesis.tags,
        )
        
        self._hypotheses[hypothesis_id] = updated
        return updated
    
    # ========================================================================
    # Experiment Design
    # ========================================================================
    
    async def design_experiment(
        self,
        hypothesis: Hypothesis,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Experiment:
        """Design an experiment to test a hypothesis.
        
        Creates an experiment design with appropriate variables,
        measurements, and success criteria.
        
        Args:
            hypothesis: The hypothesis to test
            constraints: Optional constraints on the experiment
            
        Returns:
            The designed experiment
        """
        constraints = constraints or {}
        
        # Update hypothesis status
        await self.update_hypothesis_status(
            hypothesis.id,
            HypothesisStatus.UNDER_INVESTIGATION
        )
        
        # Design experiment based on hypothesis type
        design = await self._create_experiment_design(hypothesis, constraints)
        
        experiment = Experiment(
            id="",
            hypothesis_id=hypothesis.id,
            design=design,
            status=ExperimentStatus.PENDING,
        )
        
        self._experiments[experiment.id] = experiment
        
        return experiment
    
    async def _create_experiment_design(
        self,
        hypothesis: Hypothesis,
        constraints: Dict[str, Any]
    ) -> ExperimentDesign:
        """Create an experiment design for a hypothesis."""
        # Extract variables from hypothesis statement
        variables = self._extract_variables(hypothesis.statement)
        
        # Determine sample size based on confidence requirements
        confidence_level = constraints.get("confidence_level", 0.95)
        sample_size = self._calculate_sample_size(confidence_level)
        
        # Set duration based on complexity
        duration = constraints.get("max_duration_seconds", 300.0)
        
        # Create success criteria from predicted outcome
        success_criteria = self._derive_success_criteria(hypothesis.predicted_outcome)
        
        # Create failure criteria
        failure_criteria = self._derive_failure_criteria(hypothesis)
        
        design = ExperimentDesign(
            control_variables=constraints.get("control_variables", {}),
            independent_variables=variables.get("independent", {}),
            dependent_variables=variables.get("dependent", ["performance", "accuracy"]),
            sample_size=sample_size,
            duration_seconds=duration,
            success_criteria=success_criteria,
            failure_criteria=failure_criteria,
        )
        
        return design
    
    def _extract_variables(self, statement: str) -> Dict[str, Any]:
        """Extract variables from a hypothesis statement."""
        variables = {
            "independent": {},
            "dependent": ["performance", "efficiency"]
        }
        
        # Simple pattern matching for common improvement areas
        patterns = {
            "latency": (["response_time", "processing_delay"], ["throughput"]),
            "throughput": (["batch_size", "parallelism"], ["latency"]),
            "memory": (["cache_size", "buffer_pool"], ["memory_usage"]),
            "cpu": (["thread_count", "worker_pool"], ["cpu_utilization"]),
        }
        
        statement_lower = statement.lower()
        for area, (ind, dep) in patterns.items():
            if area in statement_lower:
                for var in ind:
                    variables["independent"][var] = "tunable"
                variables["dependent"] = dep
                break
        
        return variables
    
    def _calculate_sample_size(self, confidence_level: float) -> int:
        """Calculate required sample size for desired confidence."""
        # Simplified calculation - higher confidence needs more samples
        base_size = 5
        if confidence_level >= 0.99:
            return base_size * 4
        elif confidence_level >= 0.95:
            return base_size * 2
        return base_size
    
    def _derive_success_criteria(self, predicted_outcome: str) -> List[str]:
        """Derive success criteria from predicted outcome."""
        criteria = []
        
        # Extract numerical improvements
        numbers = re.findall(r'(\d+(?:\.\d+)?)\s*%?', predicted_outcome)
        if numbers:
            for num in numbers:
                criteria.append(f"Improvement of at least {num}%")
        else:
            criteria.append("Measurable improvement in target metric")
        
        criteria.append("Statistical significance p < 0.05")
        
        return criteria
    
    def _derive_failure_criteria(self, hypothesis: Hypothesis) -> List[str]:
        """Derive failure criteria for an experiment."""
        return [
            "No measurable improvement",
            "Performance degradation > 5%",
            "Statistical significance not achieved",
            "Experiment error or timeout",
        ]
    
    async def run_experiment(
        self,
        experiment: Experiment,
        runner: Optional[Callable[..., Coroutine[Any, Any, ExperimentResult]]] = None
    ) -> ExperimentResult:
        """Execute an experiment.
        
        Args:
            experiment: The experiment to run
            runner: Optional custom experiment runner
            
        Returns:
            The experiment results
        """
        async with self._experiment_semaphore:
            # Update experiment status
            experiment.status = ExperimentStatus.RUNNING
            experiment.started_at = datetime.now()
            
            try:
                # Use custom runner if provided, otherwise try registered runners
                if runner:
                    result = await runner(experiment)
                elif self._experiment_runners:
                    result = await self._experiment_runners[0](experiment)
                else:
                    result = await self._default_experiment_runner(experiment)
                
                experiment.status = ExperimentStatus.COMPLETED
                experiment.results = result
                experiment.completed_at = datetime.now()
                
            except Exception as e:
                experiment.status = ExperimentStatus.FAILED
                experiment.error_message = str(e)
                experiment.completed_at = datetime.now()
                
                result = ExperimentResult(
                    experiment_id=experiment.id,
                    observations=[f"Experiment failed: {str(e)}"],
                    interpretation=ResultInterpretation.INCONCLUSIVE,
                    confidence=0.0,
                )
            
            if self._on_experiment_completed:
                await self._on_experiment_completed(experiment)
            
            return result
    
    async def _default_experiment_runner(
        self,
        experiment: Experiment
    ) -> ExperimentResult:
        """Default experiment runner - simulates experiment execution."""
        # Simulate experiment duration
        await asyncio.sleep(min(experiment.design.duration_seconds / 10, 2.0))
        
        # Generate simulated metrics
        metrics = []
        for var in experiment.design.dependent_variables:
            metrics.append(Metric(
                name=var,
                value=0.85 + (0.15 * (asyncio.get_event_loop().time() % 1)),
                unit="ratio",
            ))
        
        return ExperimentResult(
            experiment_id=experiment.id,
            metrics=metrics,
            observations=["Simulated experiment completed successfully"],
            interpretation=ResultInterpretation.SUPPORTS_HYPOTHESIS,
            confidence=0.75,
        )
    
    async def run_experiments_batch(
        self,
        experiments: List[Experiment]
    ) -> List[ExperimentResult]:
        """Run multiple experiments in parallel.
        
        Args:
            experiments: List of experiments to run
            
        Returns:
            List of experiment results
        """
        tasks = [self.run_experiment(exp) for exp in experiments]
        return await asyncio.gather(*tasks)
    
    # ========================================================================
    # Result Interpretation
    # ========================================================================
    
    async def interpret_results(
        self,
        experiment: Experiment,
        context: Optional[Dict[str, Any]] = None
    ) -> ExperimentResult:
        """Interpret the results of an experiment.
        
        Analyzes experiment results to determine whether they support
        or contradict the associated hypothesis.
        
        Args:
            experiment: The completed experiment
            context: Additional context for interpretation
            
        Returns:
            The interpreted result with updated interpretation field
            
        Raises:
            ValueError: If experiment has no results
        """
        if not experiment.results:
            raise ValueError("Experiment has no results to interpret")
        
        result = experiment.results
        hypothesis = self._hypotheses.get(experiment.hypothesis_id)
        
        if not hypothesis:
            return result
        
        # Analyze metrics against success criteria
        success_metrics = self._evaluate_success_criteria(
            result.metrics,
            experiment.design.success_criteria
        )
        
        # Determine interpretation
        if all(success_metrics.values()):
            interpretation = ResultInterpretation.SUPPORTS_HYPOTHESIS
            new_confidence = min(result.confidence * 1.2, 1.0)
            await self.update_hypothesis_status(
                hypothesis.id,
                HypothesisStatus.VALIDATED,
                new_confidence
            )
        elif any(success_metrics.values()):
            interpretation = ResultInterpretation.REQUIRES_FURTHER_TESTING
            new_confidence = result.confidence
        else:
            interpretation = ResultInterpretation.CONTRADICTS_HYPOTHESIS
            new_confidence = result.confidence * 0.5
            await self.update_hypothesis_status(
                hypothesis.id,
                HypothesisStatus.REJECTED,
                new_confidence
            )
        
        # Create interpreted result
        interpreted_result = ExperimentResult(
            experiment_id=result.experiment_id,
            metrics=result.metrics,
            observations=result.observations + [
                f"Success criteria met: {sum(success_metrics.values())}/{len(success_metrics)}"
            ],
            raw_data=result.raw_data,
            interpretation=interpretation,
            confidence=new_confidence,
            analyzed_at=datetime.now(),
        )
        
        experiment.results = interpreted_result
        
        return interpreted_result
    
    def _evaluate_success_criteria(
        self,
        metrics: List[Metric],
        criteria: List[str]
    ) -> Dict[str, bool]:
        """Evaluate metrics against success criteria."""
        results = {}
        
        for criterion in criteria:
            # Simple evaluation - check if any metric meets threshold
            if "improvement" in criterion.lower():
                # Assume improvement if metric value > 0.5
                metric_values = [m.value for m in metrics]
                results[criterion] = any(v > 0.5 for v in metric_values)
            elif "significance" in criterion.lower():
                # Assume significance for now
                results[criterion] = True
            else:
                results[criterion] = True
        
        return results
    
    async def compare_results(
        self,
        results: List[ExperimentResult]
    ) -> Dict[str, Any]:
        """Compare multiple experiment results.
        
        Args:
            results: List of results to compare
            
        Returns:
            Comparison analysis
        """
        if not results:
            return {"error": "No results to compare"}
        
        comparison = {
            "total_results": len(results),
            "supporting": sum(1 for r in results if r.interpretation == ResultInterpretation.SUPPORTS_HYPOTHESIS),
            "contradicting": sum(1 for r in results if r.interpretation == ResultInterpretation.CONTRADICTS_HYPOTHESIS),
            "inconclusive": sum(1 for r in results if r.interpretation == ResultInterpretation.INCONCLUSIVE),
            "average_confidence": sum(r.confidence for r in results) / len(results),
            "metric_comparison": {},
        }
        
        # Compare metrics across results
        metric_names = set()
        for result in results:
            for metric in result.metrics:
                metric_names.add(metric.name)
        
        for name in metric_names:
            values = [
                m.value for r in results for m in r.metrics if m.name == name
            ]
            if values:
                comparison["metric_comparison"][name] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                }
        
        return comparison
    
    # ========================================================================
    # Code Change Proposals
    # ========================================================================
    
    async def propose_code_changes(
        self,
        findings: List[CrossPollinationFinding],
        validated_hypotheses: Optional[List[Hypothesis]] = None
    ) -> List[CodeChange]:
        """Propose code changes based on cross-pollination findings.
        
        Analyzes findings from the cross-pollination bus and validated
        hypotheses to propose specific code changes.
        
        Args:
            findings: Findings from the cross-pollination bus
            validated_hypotheses: Optional validated hypotheses to consider
            
        Returns:
            List of proposed code changes
        """
        changes = []
        validated = validated_hypotheses or self.validated_hypotheses
        
        # Group findings by type
        findings_by_type: Dict[str, List[CrossPollinationFinding]] = {}
        for finding in findings:
            ftype = finding.finding_type
            if ftype not in findings_by_type:
                findings_by_type[ftype] = []
            findings_by_type[ftype].append(finding)
        
        # Generate changes based on finding types
        for finding_type, type_findings in findings_by_type.items():
            if finding_type == "performance_bottleneck":
                changes.extend(await self._propose_optimization_changes(
                    type_findings, validated
                ))
            elif finding_type == "bug_pattern":
                changes.extend(await self._propose_bug_fix_changes(
                    type_findings, validated
                ))
            elif finding_type == "architecture_issue":
                changes.extend(await self._propose_architecture_changes(
                    type_findings, validated
                ))
            elif finding_type == "configuration_optimal":
                changes.extend(await self._propose_configuration_changes(
                    type_findings, validated
                ))
            else:
                # Generic change proposal
                changes.extend(await self._propose_generic_changes(
                    type_findings, validated
                ))
        
        # Store proposed changes
        for change in changes:
            self._proposed_changes[change.id] = change
            if self._on_change_proposed:
                await self._on_change_proposed(change)
        
        return changes
    
    async def _propose_optimization_changes(
        self,
        findings: List[CrossPollinationFinding],
        hypotheses: List[Hypothesis]
    ) -> List[CodeChange]:
        """Propose optimization-related code changes."""
        changes = []
        
        for finding in findings:
            content = finding.content
            target = content.get("target_component", "unknown")
            try:
                patch = _extract_patch(
                    content, "suggested_patch",
                    kind="optimization", source_agent=finding.source_agent,
                )
            except IncompletePatchError as e:
                logger.warning("dropping optimization change proposal: %s", e)
                continue

            change = CodeChange(
                id="",
                description=f"Optimize {target} based on performance analysis",
                change_type=ChangeType.OPTIMIZATION,
                priority=ChangePriority.HIGH,
                target_files=content.get("affected_files", [f"{target}.py"]),
                code_patch=patch,
                rationale=f"Performance bottleneck identified by {finding.source_agent}",
                expected_impact={
                    "latency_reduction": content.get("latency_improvement", "unknown"),
                    "throughput_increase": content.get("throughput_improvement", "unknown"),
                },
                hypothesis_ids=[h.id for h in hypotheses if "performance" in h.tags],
            )
            changes.append(change)
        
        return changes
    
    async def _propose_bug_fix_changes(
        self,
        findings: List[CrossPollinationFinding],
        hypotheses: List[Hypothesis]
    ) -> List[CodeChange]:
        """Propose bug fix code changes."""
        changes = []
        
        for finding in findings:
            content = finding.content
            try:
                patch = _extract_patch(
                    content, "fix_patch",
                    kind="bug-fix", source_agent=finding.source_agent,
                )
            except IncompletePatchError as e:
                logger.warning("dropping bug-fix change proposal: %s", e)
                continue

            change = CodeChange(
                id="",
                description=f"Fix bug: {content.get('bug_description', 'Unknown issue')}",
                change_type=ChangeType.BUG_FIX,
                priority=ChangePriority.CRITICAL,
                target_files=content.get("affected_files", []),
                code_patch=patch,
                rationale=f"Bug pattern identified by {finding.source_agent}",
                expected_impact={
                    "error_reduction": "100% of identified pattern",
                    "stability_improvement": "high",
                },
                hypothesis_ids=[h.id for h in hypotheses if "error_rate" in h.tags],
            )
            changes.append(change)
        
        return changes
    
    async def _propose_architecture_changes(
        self,
        findings: List[CrossPollinationFinding],
        hypotheses: List[Hypothesis]
    ) -> List[CodeChange]:
        """Propose architecture-related code changes."""
        changes = []
        
        for finding in findings:
            content = finding.content
            try:
                patch = _extract_patch(
                    content, "refactoring_patch",
                    kind="architecture", source_agent=finding.source_agent,
                )
            except IncompletePatchError as e:
                logger.warning("dropping architecture change proposal: %s", e)
                continue

            change = CodeChange(
                id="",
                description=f"Architecture improvement: {content.get('recommendation', 'Refactoring')}",
                change_type=ChangeType.ARCHITECTURE,
                priority=ChangePriority.MEDIUM,
                target_files=content.get("affected_files", []),
                code_patch=patch,
                rationale=f"Architecture issue identified by {finding.source_agent}",
                expected_impact={
                    "maintainability": "improved",
                    "scalability": content.get("scalability_impact", "unknown"),
                },
                hypothesis_ids=[h.id for h in hypotheses if "scalability" in h.tags],
            )
            changes.append(change)
        
        return changes
    
    async def _propose_configuration_changes(
        self,
        findings: List[CrossPollinationFinding],
        hypotheses: List[Hypothesis]
    ) -> List[CodeChange]:
        """Propose configuration-related code changes."""
        changes = []
        
        for finding in findings:
            content = finding.content
            try:
                patch = _extract_patch(
                    content, "config_patch",
                    kind="configuration", source_agent=finding.source_agent,
                )
            except IncompletePatchError as e:
                logger.warning("dropping configuration change proposal: %s", e)
                continue

            change = CodeChange(
                id="",
                description=f"Update configuration: {content.get('config_parameter', 'Unknown')}",
                change_type=ChangeType.CONFIGURATION,
                priority=ChangePriority.LOW,
                target_files=content.get("config_files", ["config.yaml"]),
                code_patch=patch,
                rationale=f"Optimal configuration found by {finding.source_agent}",
                expected_impact={
                    "performance_gain": content.get("performance_gain", "unknown"),
                },
                hypothesis_ids=[h.id for h in hypotheses],
            )
            changes.append(change)
        
        return changes
    
    async def _propose_generic_changes(
        self,
        findings: List[CrossPollinationFinding],
        hypotheses: List[Hypothesis]
    ) -> List[CodeChange]:
        """Propose generic code changes from findings."""
        changes = []
        
        for finding in findings:
            content = finding.content
            try:
                patch = _extract_patch(
                    content, "suggested_patch",
                    kind=f"generic ({finding.finding_type})",
                    source_agent=finding.source_agent,
                )
            except IncompletePatchError as e:
                logger.warning("dropping generic change proposal: %s", e)
                continue

            change = CodeChange(
                id="",
                description=f"Improvement based on {finding.finding_type}",
                change_type=ChangeType.REFACTORING,
                priority=ChangePriority.MEDIUM,
                target_files=content.get("affected_files", []),
                code_patch=patch,
                rationale=f"Finding from {finding.source_agent}",
                expected_impact={"general_improvement": "expected"},
                hypothesis_ids=[h.id for h in hypotheses],
            )
            changes.append(change)
        
        return changes
    
    async def evaluate_change_impact(
        self,
        change: CodeChange
    ) -> Dict[str, Any]:
        """Evaluate the potential impact of a proposed change.
        
        Args:
            change: The code change to evaluate
            
        Returns:
            Impact assessment
        """
        assessment = {
            "change_id": change.id,
            "risk_level": "medium",
            "affected_components": [],
            "testing_recommendations": [],
            "rollback_complexity": "low",
        }
        
        # Assess risk based on change type
        if change.change_type == ChangeType.BUG_FIX:
            assessment["risk_level"] = "low"
            assessment["testing_recommendations"].append("Regression tests")
        elif change.change_type == ChangeType.ARCHITECTURE:
            assessment["risk_level"] = "high"
            assessment["rollback_complexity"] = "high"
            assessment["testing_recommendations"].extend([
                "Integration tests",
                "Performance tests",
                "Load tests"
            ])
        elif change.change_type == ChangeType.OPTIMIZATION:
            assessment["testing_recommendations"].extend([
                "Benchmark tests",
                "Performance regression tests"
            ])
        
        # Identify affected components from target files
        for file_path in change.target_files:
            component = file_path.split("/")[0] if "/" in file_path else "core"
            if component not in assessment["affected_components"]:
                assessment["affected_components"].append(component)
        
        return assessment
    
    # ========================================================================
    # Cross-Pollination Integration
    # ========================================================================
    
    async def receive_finding(self, finding: CrossPollinationFinding) -> None:
        """Receive a finding from the cross-pollination bus.
        
        Args:
            finding: The finding to process
        """
        self._findings.append(finding)
        
        # Auto-generate hypotheses from certain finding types
        if finding.finding_type in ["performance_bottleneck", "bug_pattern"]:
            await self.generate_hypothesis(
                observations=[
                    f"{finding.finding_type} from {finding.source_agent}",
                    str(finding.content),
                ],
                context={"finding": finding.to_dict() if hasattr(finding, 'to_dict') else finding},
                tags={finding.finding_type, "cross-pollination"}
            )
    
    async def process_findings_batch(
        self,
        auto_propose_changes: bool = True
    ) -> Dict[str, Any]:
        """Process all pending findings.
        
        Args:
            auto_propose_changes: Whether to auto-propose changes
            
        Returns:
            Processing summary
        """
        if not self._findings:
            return {"processed": 0, "hypotheses_generated": 0, "changes_proposed": 0}
        
        findings_to_process = self._findings.copy()
        self._findings.clear()
        
        hypotheses_before = len(self._hypotheses)
        
        # Generate hypotheses from findings
        for finding in findings_to_process:
            await self.receive_finding(finding)
        
        hypotheses_generated = len(self._hypotheses) - hypotheses_before
        
        changes_proposed = 0
        if auto_propose_changes:
            changes = await self.propose_code_changes(findings_to_process)
            changes_proposed = len(changes)
        
        return {
            "processed": len(findings_to_process),
            "hypotheses_generated": hypotheses_generated,
            "changes_proposed": changes_proposed,
        }
    
    # ========================================================================
    # Main Processing Loop
    # ========================================================================
    
    async def process(self) -> None:
        """Main processing loop for the ResearcherAgent.
        
        Continuously processes messages and performs research activities.
        """
        while self._is_running:
            try:
                # Check for messages
                message = await self.get_message(timeout=1.0)
                
                if message:
                    await self._handle_message(message)
                
                # Process pending hypotheses
                await self._process_pending_hypotheses()
                
                # Process pending findings
                if self._findings:
                    await self.process_findings_batch()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error and continue
                await asyncio.sleep(0.1)
    
    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle an incoming message."""
        msg_type = message.get("type", "unknown")
        
        if msg_type == "finding":
            finding_data = message.get("data", {})
            finding = CrossPollinationFinding(
                source_agent=finding_data.get("source_agent", "unknown"),
                finding_type=finding_data.get("finding_type", "unknown"),
                content=finding_data.get("content", {}),
                metadata=finding_data.get("metadata", {}),
            )
            await self.receive_finding(finding)
        
        elif msg_type == "observation":
            observations = message.get("data", {}).get("observations", [])
            if observations:
                await self.generate_hypothesis(observations)
        
        elif msg_type == "run_experiment":
            exp_id = message.get("data", {}).get("experiment_id")
            if exp_id and exp_id in self._experiments:
                await self.run_experiment(self._experiments[exp_id])
        
        elif msg_type == "interpret_results":
            exp_id = message.get("data", {}).get("experiment_id")
            if exp_id and exp_id in self._experiments:
                await self.interpret_results(self._experiments[exp_id])
    
    async def _process_pending_hypotheses(self) -> None:
        """Process hypotheses awaiting investigation."""
        pending = self.pending_hypotheses[:3]  # Process up to 3 at a time
        
        for hypothesis in pending:
            # Design experiment
            experiment = await self.design_experiment(hypothesis)
            
            # Run experiment
            await self.run_experiment(experiment)
            
            # Interpret results
            if experiment.results:
                await self.interpret_results(experiment)
    
    async def _on_start(self) -> None:
        """Hook called when agent starts."""
        self._processing_task = asyncio.create_task(self.process())
    
    async def _on_stop(self) -> None:
        """Hook called when agent stops."""
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get a summary of all research activities.
        
        Returns:
            Research activity summary
        """
        return {
            "agent_name": self.name,
            "total_hypotheses": len(self._hypotheses),
            "pending_hypotheses": len(self.pending_hypotheses),
            "active_hypotheses": len(self.active_hypotheses),
            "validated_hypotheses": len(self.validated_hypotheses),
            "total_experiments": len(self._experiments),
            "running_experiments": len(self.running_experiments),
            "pending_findings": len(self._findings),
            "proposed_changes": len(self._proposed_changes),
            "hypothesis_tags": list(set(
                tag for h in self._hypotheses.values() for tag in h.tags
            )),
        }
    
    async def export_research_data(self) -> Dict[str, Any]:
        """Export all research data for persistence.
        
        Returns:
            Complete research data export
        """
        return {
            "agent_name": self.name,
            "hypotheses": [h.to_dict() for h in self._hypotheses.values()],
            "experiments": [
                {
                    "id": e.id,
                    "hypothesis_id": e.hypothesis_id,
                    "status": e.status.name,
                    "results": e.results.to_dict() if e.results else None,
                }
                for e in self._experiments.values()
            ],
            "proposed_changes": [c.to_dict() for c in self._proposed_changes.values()],
        }
    
    async def clear_history(self) -> None:
        """Clear all research history."""
        self._hypotheses.clear()
        self._experiments.clear()
        self._findings.clear()
        self._proposed_changes.clear()


# ============================================================================
# Utility Functions
# ============================================================================

def create_researcher_agent(
    name: Optional[str] = None,
    **kwargs: Any
) -> ResearcherAgent:
    """Factory function to create a ResearcherAgent.
    
    Args:
        name: Agent name (auto-generated if None)
        **kwargs: Additional configuration
        
    Returns:
        Configured ResearcherAgent instance
    """
    if name is None:
        import random
        name = f"researcher_{random.randint(1000, 9999)}"
    
    return ResearcherAgent(name=name, **kwargs)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Main agent
    "ResearcherAgent",
    "BaseAgent",
    "create_researcher_agent",
    # Data classes
    "Hypothesis",
    "Experiment",
    "ExperimentDesign",
    "ExperimentResult",
    "CodeChange",
    "CrossPollinationFinding",
    "Metric",
    # Enums
    "HypothesisStatus",
    "ExperimentStatus",
    "ResultInterpretation",
    "ChangePriority",
    "ChangeType",
]
