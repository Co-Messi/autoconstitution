"""
SynthesiserAgent for autoconstitution Framework

This module implements the SynthesiserAgent, which is responsible for:
- Periodically reviewing all validated improvements across branches
- Identifying patterns and synergies between improvements
- Proposing composite improvements combining multiple findings
- Tracking synthesis history and composite proposal effectiveness

The SynthesiserAgent inherits from BaseAgent and provides a complete
async implementation with full type hints.

Python Version: 3.11+
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Generic,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)
from collections.abc import Sequence

from .base import (
    AgentID,
    AgentStatus,
    BaseAgent,
    BroadcastChannel,
    Checkpoint,
    CheckpointLevel,
    CheckpointStore,
    ExecutionContext,
    ExecutionResult,
    Finding,
    LLMProvider,
    Message,
    MessagePriority,
    create_finding,
)


# ============================================================================
# Type Variables
# ============================================================================

T = TypeVar("T")


# ============================================================================
# Enums
# ============================================================================

class SynthesisStatus(Enum):
    """Status of a synthesis operation."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class PatternType(Enum):
    """Types of patterns that can be identified."""
    PERFORMANCE_OPTIMIZATION = auto()
    ARCHITECTURAL_IMPROVEMENT = auto()
    BUG_FIX_PATTERN = auto()
    CONFIGURATION_TWEAK = auto()
    CODE_QUALITY = auto()
    SECURITY_ENHANCEMENT = auto()
    SCALABILITY_IMPROVEMENT = auto()
    COMPOSITE_SYNERGY = auto()


class SynergyType(Enum):
    """Types of synergies between improvements."""
    ADDITIVE = "additive"  # Improvements add up linearly
    MULTIPLICATIVE = "multiplicative"  # Improvements amplify each other
    COMPLEMENTARY = "complementary"  # Improvements fill gaps in each other
    DEPENDENT = "dependent"  # One improvement requires another
    CONFLICTING = "conflicting"  # Improvements work against each other


class CompositePriority(Enum):
    """Priority levels for composite improvements."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    DEFERRED = 5


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True, slots=True)
class ValidatedImprovement:
    """Represents a validated improvement from any branch.
    
    Attributes:
        improvement_id: Unique identifier for the improvement
        branch_id: ID of the branch where improvement was validated
        source_agent: Agent that produced the improvement
        improvement_type: Type/category of improvement
        description: Human-readable description
        changes: List of specific changes made
        metrics_before: Metrics before the improvement
        metrics_after: Metrics after the improvement
        validation_confidence: Confidence score from validation (0.0 to 1.0)
        validated_at: Timestamp of validation
        tags: Categorization tags
        metadata: Additional metadata
    """
    improvement_id: str
    branch_id: str
    source_agent: str
    improvement_type: str
    description: str
    changes: list[str] = field(default_factory=list)
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)
    validation_confidence: float = 0.5
    validated_at: datetime = field(default_factory=datetime.utcnow)
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def improvement_ratio(self) -> dict[str, float]:
        """Calculate improvement ratio for each metric."""
        ratios = {}
        for key in self.metrics_after:
            if key in self.metrics_before and self.metrics_before[key] != 0:
                ratios[key] = self.metrics_after[key] / self.metrics_before[key]
            else:
                ratios[key] = 1.0
        return ratios
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "improvement_id": self.improvement_id,
            "branch_id": self.branch_id,
            "source_agent": self.source_agent,
            "improvement_type": self.improvement_type,
            "description": self.description,
            "changes": self.changes,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "improvement_ratio": self.improvement_ratio,
            "validation_confidence": self.validation_confidence,
            "validated_at": self.validated_at.isoformat(),
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class IdentifiedPattern:
    """Represents a pattern identified across multiple improvements.
    
    Attributes:
        pattern_id: Unique identifier for the pattern
        pattern_type: Type of pattern identified
        description: Human-readable description of the pattern
        contributing_improvements: IDs of improvements that exhibit this pattern
        confidence: Confidence score (0.0 to 1.0)
        frequency: How often this pattern appears
        affected_components: Components affected by this pattern
        first_seen: Timestamp when pattern was first identified
        last_seen: Timestamp when pattern was last observed
    """
    pattern_id: str
    pattern_type: PatternType
    description: str
    contributing_improvements: list[str] = field(default_factory=list)
    confidence: float = 0.5
    frequency: int = 1
    affected_components: list[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_recurring(self) -> bool:
        """Check if this is a recurring pattern."""
        return self.frequency > 2
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.name,
            "description": self.description,
            "contributing_improvements": self.contributing_improvements,
            "confidence": self.confidence,
            "frequency": self.frequency,
            "affected_components": self.affected_components,
            "is_recurring": self.is_recurring,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class Synergy:
    """Represents a synergy between two or more improvements.
    
    Attributes:
        synergy_id: Unique identifier for the synergy
        improvement_ids: IDs of improvements that have synergy
        synergy_type: Type of synergy
        description: Human-readable description
        estimated_combined_impact: Estimated impact when combined
        confidence: Confidence score (0.0 to 1.0)
        identified_at: Timestamp when synergy was identified
    """
    synergy_id: str
    improvement_ids: list[str] = field(default_factory=list)
    synergy_type: SynergyType = SynergyType.ADDITIVE
    description: str = ""
    estimated_combined_impact: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5
    identified_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_positive(self) -> bool:
        """Check if this is a positive synergy."""
        return self.synergy_type != SynergyType.CONFLICTING
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "synergy_id": self.synergy_id,
            "improvement_ids": self.improvement_ids,
            "synergy_type": self.synergy_type.value,
            "description": self.description,
            "estimated_combined_impact": self.estimated_combined_impact,
            "confidence": self.confidence,
            "is_positive": self.is_positive,
            "identified_at": self.identified_at.isoformat(),
        }


@dataclass(slots=True)
class CompositeImprovement:
    """Represents a composite improvement combining multiple findings.
    
    Attributes:
        composite_id: Unique identifier for the composite
        name: Human-readable name
        description: Detailed description
        source_improvements: IDs of improvements being combined
        source_patterns: IDs of patterns being leveraged
        source_synergies: IDs of synergies being exploited
        proposed_changes: List of proposed code changes
        expected_benefits: Expected benefits from this composite
        priority: Priority level
        estimated_confidence: Confidence in success (0.0 to 1.0)
        created_at: Creation timestamp
        status: Current status
        applied_to_branches: Branches where this has been applied
        results: Results after application
    """
    composite_id: str
    name: str
    description: str
    source_improvements: list[str] = field(default_factory=list)
    source_patterns: list[str] = field(default_factory=list)
    source_synergies: list[str] = field(default_factory=list)
    proposed_changes: list[str] = field(default_factory=list)
    expected_benefits: dict[str, Any] = field(default_factory=dict)
    priority: CompositePriority = CompositePriority.MEDIUM
    estimated_confidence: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: SynthesisStatus = SynthesisStatus.PENDING
    applied_to_branches: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "composite_id": self.composite_id,
            "name": self.name,
            "description": self.description,
            "source_improvements": self.source_improvements,
            "source_patterns": self.source_patterns,
            "source_synergies": self.source_synergies,
            "proposed_changes": self.proposed_changes,
            "expected_benefits": self.expected_benefits,
            "priority": self.priority.name,
            "estimated_confidence": self.estimated_confidence,
            "created_at": self.created_at.isoformat(),
            "status": self.status.name,
            "applied_to_branches": self.applied_to_branches,
            "results": self.results,
        }


@dataclass(slots=True)
class SynthesisResult:
    """Result of a synthesis operation.
    
    Attributes:
        synthesis_id: Unique identifier for this synthesis
        status: Status of the synthesis
        patterns_identified: Patterns found during synthesis
        synergies_found: Synergies discovered
        composites_proposed: Composite improvements proposed
        reviewed_improvements: Number of improvements reviewed
        execution_time_ms: Time taken for synthesis
        created_at: Timestamp
        metadata: Additional metadata
    """
    synthesis_id: str
    status: SynthesisStatus
    patterns_identified: list[IdentifiedPattern] = field(default_factory=list)
    synergies_found: list[Synergy] = field(default_factory=list)
    composites_proposed: list[CompositeImprovement] = field(default_factory=list)
    reviewed_improvements: int = 0
    execution_time_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def patterns_count(self) -> int:
        """Return the number of patterns identified."""
        return len(self.patterns_identified)
    
    @property
    def synergies_count(self) -> int:
        """Return the number of synergies found."""
        return len(self.synergies_found)
    
    @property
    def composites_count(self) -> int:
        """Return the number of composites proposed."""
        return len(self.composites_proposed)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "synthesis_id": self.synthesis_id,
            "status": self.status.name,
            "patterns_identified": [p.to_dict() for p in self.patterns_identified],
            "synergies_found": [s.to_dict() for s in self.synergies_found],
            "composites_proposed": [c.to_dict() for c in self.composites_proposed],
            "reviewed_improvements": self.reviewed_improvements,
            "patterns_count": self.patterns_count,
            "synergies_count": self.synergies_count,
            "composites_count": self.composites_count,
            "execution_time_ms": self.execution_time_ms,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SynthesiserContext:
    """Context specific to synthesiser agent execution.
    
    Attributes:
        improvements_to_review: Specific improvements to review (None = all)
        focus_patterns: Pattern types to focus on
        min_confidence_threshold: Minimum confidence for inclusion
        max_composites: Maximum composites to propose
        require_synergies: Whether to require synergies for composites
        time_range: Time range for improvements to consider
        target_branches: Specific branches to focus on
    """
    improvements_to_review: list[str] | None = None
    focus_patterns: list[PatternType] = field(default_factory=list)
    min_confidence_threshold: float = 0.5
    max_composites: int = 10
    require_synergies: bool = True
    time_range: tuple[datetime, datetime] | None = None
    target_branches: list[str] = field(default_factory=list)
    
    def to_execution_context(self, task_id: str | None = None) -> ExecutionContext:
        """Convert to base ExecutionContext."""
        return ExecutionContext(
            task_id=task_id or f"synthesis_{datetime.utcnow().isoformat()}",
            parameters={
                "improvements_to_review": self.improvements_to_review,
                "focus_patterns": [p.name for p in self.focus_patterns],
                "min_confidence_threshold": self.min_confidence_threshold,
                "max_composites": self.max_composites,
                "require_synergies": self.require_synergies,
                "target_branches": self.target_branches,
            },
        )


# ============================================================================
# Protocols
# ============================================================================

@runtime_checkable
class ImprovementProvider(Protocol):
    """Protocol for providing validated improvements."""
    
    async def get_validated_improvements(
        self,
        branches: list[str] | None = None,
        since: datetime | None = None,
    ) -> list[ValidatedImprovement]:
        """Get validated improvements from branches."""
        ...
    
    async def subscribe_to_improvements(
        self,
        callback: Callable[[ValidatedImprovement], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to new improvement notifications."""
        ...


# ============================================================================
# SynthesiserAgent Implementation
# ============================================================================

class SynthesiserAgent(BaseAgent[SynthesiserContext, SynthesisResult]):
    """Agent that synthesizes validated improvements into composite proposals.
    
    The SynthesiserAgent periodically reviews all validated improvements across
    branches, identifies patterns and synergies between them, and proposes
    composite improvements that combine multiple findings for greater impact.
    
    Attributes:
        agent_id: Unique identifier for this agent instance
        review_interval_seconds: How often to perform synthesis (default: 300s)
        pattern_history: History of identified patterns
        synergy_history: History of discovered synergies
        composite_history: History of proposed composites
        synthesis_history: History of synthesis operations
        
    Example:
        >>> agent = SynthesiserAgent(review_interval_seconds=600)
        >>> await agent.initialize()
        >>> # Add improvements from branches
        >>> await agent.add_improvement(validated_improvement)
        >>> # Trigger synthesis
        >>> result = await agent.perform_synthesis()
        >>> print(f"Found {result.patterns_count} patterns")
        >>> print(f"Proposed {result.composites_count} composites")
    
    Args:
        agent_id: Optional unique identifier
        llm_provider: Optional LLM provider for synthesis generation
        checkpoint_store: Optional checkpoint storage backend
        broadcast_channel: Optional broadcast communication channel
        review_interval_seconds: Interval between automatic synthesis runs
        improvement_provider: Optional provider for fetching improvements
    """
    
    # Default configuration
    DEFAULT_REVIEW_INTERVAL: float = 300.0  # 5 minutes
    DEFAULT_MIN_CONFIDENCE: float = 0.5
    DEFAULT_MAX_COMPOSITES: int = 10
    
    def __init__(
        self,
        agent_id: AgentID | None = None,
        llm_provider: LLMProvider | None = None,
        checkpoint_store: CheckpointStore | None = None,
        broadcast_channel: BroadcastChannel | None = None,
        review_interval_seconds: float = DEFAULT_REVIEW_INTERVAL,
        improvement_provider: ImprovementProvider | None = None,
    ) -> None:
        """Initialize the SynthesiserAgent.
        
        Args:
            agent_id: Optional unique identifier
            llm_provider: Optional LLM provider for synthesis generation
            checkpoint_store: Optional checkpoint storage backend
            broadcast_channel: Optional broadcast communication channel
            review_interval_seconds: Interval between automatic synthesis runs
            improvement_provider: Optional provider for fetching improvements
        """
        super().__init__(
            agent_id=agent_id,
            llm_provider=llm_provider,
            checkpoint_store=checkpoint_store,
            broadcast_channel=broadcast_channel,
        )
        
        self._review_interval_seconds: float = review_interval_seconds
        self._improvement_provider: ImprovementProvider | None = improvement_provider
        
        # Storage for improvements and synthesis artifacts
        self._improvements: dict[str, ValidatedImprovement] = {}
        self._patterns: dict[str, IdentifiedPattern] = {}
        self._synergies: dict[str, Synergy] = {}
        self._composites: dict[str, CompositeImprovement] = {}
        self._synthesis_history: list[SynthesisResult] = []
        
        # Periodic review state
        self._review_task: asyncio.Task[None] | None = None
        self._is_reviewing: bool = False
        self._last_review_time: datetime | None = None
        
        # Findings buffer for base class integration
        self._findings_buffer: list[Finding] = []
        
        # Event callbacks
        self._on_pattern_identified: Callable[[IdentifiedPattern], Coroutine[Any, Any, None]] | None = None
        self._on_synergy_found: Callable[[Synergy], Coroutine[Any, Any, None]] | None = None
        self._on_composite_proposed: Callable[[CompositeImprovement], Coroutine[Any, Any, None]] | None = None
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def review_interval_seconds(self) -> float:
        """Return the configured review interval in seconds."""
        return self._review_interval_seconds
    
    @review_interval_seconds.setter
    def review_interval_seconds(self, value: float) -> None:
        """Set the review interval in seconds."""
        if value < 10.0:
            raise ValueError("Review interval must be at least 10 seconds")
        self._review_interval_seconds = value
    
    @property
    def improvements(self) -> dict[str, ValidatedImprovement]:
        """Return all stored improvements."""
        return self._improvements.copy()
    
    @property
    def patterns(self) -> dict[str, IdentifiedPattern]:
        """Return all identified patterns."""
        return self._patterns.copy()
    
    @property
    def synergies(self) -> dict[str, Synergy]:
        """Return all discovered synergies."""
        return self._synergies.copy()
    
    @property
    def composites(self) -> dict[str, CompositeImprovement]:
        """Return all proposed composites."""
        return self._composites.copy()
    
    @property
    def synthesis_history(self) -> list[SynthesisResult]:
        """Return the history of synthesis operations."""
        return self._synthesis_history.copy()
    
    @property
    def last_review_time(self) -> datetime | None:
        """Return the timestamp of the last review."""
        return self._last_review_time
    
    @property
    def is_reviewing(self) -> bool:
        """Check if a review is currently in progress."""
        return self._is_reviewing
    
    @property
    def improvement_count(self) -> int:
        """Return the number of stored improvements."""
        return len(self._improvements)
    
    @property
    def pattern_count(self) -> int:
        """Return the number of identified patterns."""
        return len(self._patterns)
    
    @property
    def synergy_count(self) -> int:
        """Return the number of discovered synergies."""
        return len(self._synergies)
    
    @property
    def composite_count(self) -> int:
        """Return the number of proposed composites."""
        return len(self._composites)
    
    # -------------------------------------------------------------------------
    # Event Callbacks
    # -------------------------------------------------------------------------
    
    def on_pattern_identified(
        self,
        callback: Callable[[IdentifiedPattern], Coroutine[Any, Any, None]],
    ) -> None:
        """Set callback for pattern identification events."""
        self._on_pattern_identified = callback
    
    def on_synergy_found(
        self,
        callback: Callable[[Synergy], Coroutine[Any, Any, None]],
    ) -> None:
        """Set callback for synergy discovery events."""
        self._on_synergy_found = callback
    
    def on_composite_proposed(
        self,
        callback: Callable[[CompositeImprovement], Coroutine[Any, Any, None]],
    ) -> None:
        """Set callback for composite proposal events."""
        self._on_composite_proposed = callback
    
    # -------------------------------------------------------------------------
    # Core Abstract Method Implementations
    # -------------------------------------------------------------------------
    
    async def execute(
        self,
        context: SynthesiserContext | ExecutionContext,
    ) -> ExecutionResult[SynthesisResult]:
        """Execute the synthesis process.
        
        This is the main entry point for the synthesiser agent. It performs
        a synthesis operation based on the provided context.
        
        Args:
            context: Execution context containing synthesis parameters
        
        Returns:
            ExecutionResult containing the synthesis result
        """
        start_time = time.time()
        
        self._update_status(AgentStatus.EXECUTING)
        self._findings_buffer = []
        
        try:
            # Normalize context
            synth_context = self._normalize_context(context)
            
            # Perform synthesis
            result = await self._perform_synthesis_internal(synth_context)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Create finding
            finding = create_finding(
                agent_id=self._agent_id,
                finding_type="synthesis_completed",
                content=result.to_dict(),
                confidence=0.9 if result.status == SynthesisStatus.COMPLETED else 0.5,
                synthesis_id=result.synthesis_id,
            )
            self._findings_buffer.append(finding)
            
            exec_result = ExecutionResult(
                success=result.status == SynthesisStatus.COMPLETED,
                data=result,
                execution_time_ms=execution_time,
                findings=self._findings_buffer.copy(),
                metadata={
                    "synthesis_id": result.synthesis_id,
                    "patterns_found": result.patterns_count,
                    "synergies_found": result.synergies_count,
                    "composites_proposed": result.composites_count,
                },
            )
            
            self._record_execution(exec_result)
            self._update_status(AgentStatus.IDLE)
            
            return exec_result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._update_status(AgentStatus.ERROR)
            
            return ExecutionResult(
                success=False,
                data=None,
                error=e,
                execution_time_ms=execution_time,
                findings=self._findings_buffer.copy(),
                metadata={"error_type": type(e).__name__},
            )
    
    async def report_findings(self) -> Sequence[Finding]:
        """Report findings from the agent's execution.
        
        Returns:
            Sequence of Finding objects representing synthesis results
        """
        return self._findings_buffer.copy()
    
    async def receive_broadcast(self, message: Message) -> None:
        """Receive and process a broadcast message.
        
        Handles incoming messages from other agents in the swarm.
        Specifically listens for new validated improvements to add.
        
        Args:
            message: The broadcast message to process
        """
        content = message.content
        
        if not isinstance(content, dict):
            return
        
        # Handle new improvement notifications
        if content.get("type") == "validated_improvement":
            improvement_data = content.get("improvement", {})
            improvement = self._dict_to_improvement(improvement_data)
            await self.add_improvement(improvement)
        
        # Handle synthesis request
        if content.get("type") == "synthesis_request":
            context = SynthesiserContext(
                target_branches=content.get("branches", []),
                min_confidence_threshold=content.get("min_confidence", self.DEFAULT_MIN_CONFIDENCE),
            )
            await self.execute(context)
    
    async def checkpoint(self, level: CheckpointLevel = CheckpointLevel.STANDARD) -> Checkpoint:
        """Create a checkpoint of the agent's current state.
        
        Args:
            level: Granularity level for the checkpoint
        
        Returns:
            Checkpoint object representing the saved state
        """
        state_data: dict[str, Any] = {
            "review_interval_seconds": self._review_interval_seconds,
            "improvement_count": len(self._improvements),
            "pattern_count": len(self._patterns),
            "synergy_count": len(self._synergies),
            "composite_count": len(self._composites),
            "last_review_time": self._last_review_time.isoformat() if self._last_review_time else None,
        }
        
        if level in (CheckpointLevel.STANDARD, CheckpointLevel.FULL):
            state_data["improvements"] = [imp.to_dict() for imp in list(self._improvements.values())[-50:]]
            state_data["patterns"] = [p.to_dict() for p in list(self._patterns.values())[-20:]]
            state_data["synergies"] = [s.to_dict() for s in list(self._synergies.values())[-20:]]
        
        if level == CheckpointLevel.FULL:
            state_data["all_improvements"] = [imp.to_dict() for imp in self._improvements.values()]
            state_data["all_patterns"] = [p.to_dict() for p in self._patterns.values()]
            state_data["all_synergies"] = [s.to_dict() for s in self._synergies.values()]
            state_data["all_composites"] = [c.to_dict() for c in self._composites.values()]
            state_data["synthesis_history"] = [s.to_dict() for s in self._synthesis_history[-20:]]
        
        return Checkpoint(
            agent_id=self._agent_id,
            state_data=state_data,
            level=level,
        )
    
    async def _restore_from_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """Restore agent state from a checkpoint.
        
        Args:
            checkpoint: Checkpoint to restore from
        
        Returns:
            True if restoration was successful
        """
        try:
            state_data = checkpoint.state_data
            
            if "review_interval_seconds" in state_data:
                self._review_interval_seconds = state_data["review_interval_seconds"]
            
            if "last_review_time" in state_data and state_data["last_review_time"]:
                self._last_review_time = datetime.fromisoformat(state_data["last_review_time"])
            
            # Restore improvements
            if "improvements" in state_data:
                for imp_data in state_data["improvements"]:
                    improvement = self._dict_to_improvement(imp_data)
                    self._improvements[improvement.improvement_id] = improvement
            
            return True
        except Exception:
            return False
    
    # -------------------------------------------------------------------------
    # Public API Methods
    # -------------------------------------------------------------------------
    
    async def add_improvement(
        self,
        improvement: ValidatedImprovement,
    ) -> None:
        """Add a validated improvement to the synthesiser.
        
        Args:
            improvement: The validated improvement to add
        """
        self._improvements[improvement.improvement_id] = improvement
        
        # Broadcast notification if channel available
        if self._broadcast_channel is not None:
            await self.send_message(
                content={
                    "type": "improvement_added",
                    "improvement_id": improvement.improvement_id,
                    "branch_id": improvement.branch_id,
                },
                priority=MessagePriority.NORMAL,
            )
    
    async def add_improvements_batch(
        self,
        improvements: list[ValidatedImprovement],
    ) -> int:
        """Add multiple improvements in batch.
        
        Args:
            improvements: List of improvements to add
        
        Returns:
            Number of improvements added
        """
        for improvement in improvements:
            self._improvements[improvement.improvement_id] = improvement
        return len(improvements)
    
    async def perform_synthesis(
        self,
        context: SynthesiserContext | None = None,
    ) -> SynthesisResult:
        """Perform a synthesis operation.
        
        This is a convenience method that wraps execute() and returns
        just the SynthesisResult.
        
        Args:
            context: Optional synthesis context
        
        Returns:
            The synthesis result
        """
        ctx = context or SynthesiserContext()
        result = await self.execute(ctx)
        
        if not result.success or result.data is None:
            raise SynthesisExecutionError(
                "Synthesis failed",
                agent_id=self._agent_id,
                cause=result.error,
            )
        
        return result.data
    
    async def start_periodic_review(self) -> None:
        """Start the periodic review process.
        
        This starts a background task that performs synthesis at
        regular intervals.
        """
        if self._review_task is not None and not self._review_task.done():
            return  # Already running
        
        self._review_task = asyncio.create_task(self._periodic_review_loop())
    
    async def stop_periodic_review(self) -> None:
        """Stop the periodic review process."""
        if self._review_task is not None and not self._review_task.done():
            self._review_task.cancel()
            try:
                await self._review_task
            except asyncio.CancelledError:
                pass
            self._review_task = None
    
    def get_improvements_by_branch(
        self,
        branch_id: str,
    ) -> list[ValidatedImprovement]:
        """Get all improvements from a specific branch.
        
        Args:
            branch_id: The branch ID to filter by
        
        Returns:
            List of improvements from that branch
        """
        return [
            imp for imp in self._improvements.values()
            if imp.branch_id == branch_id
        ]
    
    def get_improvements_by_type(
        self,
        improvement_type: str,
    ) -> list[ValidatedImprovement]:
        """Get all improvements of a specific type.
        
        Args:
            improvement_type: The type to filter by
        
        Returns:
            List of improvements of that type
        """
        return [
            imp for imp in self._improvements.values()
            if imp.improvement_type == improvement_type
        ]
    
    def get_patterns_by_type(
        self,
        pattern_type: PatternType,
    ) -> list[IdentifiedPattern]:
        """Get all patterns of a specific type.
        
        Args:
            pattern_type: The pattern type to filter by
        
        Returns:
            List of patterns of that type
        """
        return [
            p for p in self._patterns.values()
            if p.pattern_type == pattern_type
        ]
    
    def get_composites_by_priority(
        self,
        priority: CompositePriority,
    ) -> list[CompositeImprovement]:
        """Get all composites with a specific priority.
        
        Args:
            priority: The priority level to filter by
        
        Returns:
            List of composites with that priority
        """
        return [
            c for c in self._composites.values()
            if c.priority == priority
        ]
    
    def get_composite_by_id(
        self,
        composite_id: str,
    ) -> CompositeImprovement | None:
        """Get a composite by its ID.
        
        Args:
            composite_id: ID of the composite to retrieve
        
        Returns:
            The composite if found, None otherwise
        """
        return self._composites.get(composite_id)
    
    async def update_composite_status(
        self,
        composite_id: str,
        status: SynthesisStatus,
        results: dict[str, Any] | None = None,
    ) -> CompositeImprovement | None:
        """Update the status of a composite improvement.
        
        Args:
            composite_id: ID of the composite to update
            status: New status to set
            results: Optional results to add
        
        Returns:
            Updated composite, or None if not found
        """
        if composite_id not in self._composites:
            return None
        
        composite = self._composites[composite_id]
        
        # Create updated composite
        updated = CompositeImprovement(
            composite_id=composite.composite_id,
            name=composite.name,
            description=composite.description,
            source_improvements=composite.source_improvements,
            source_patterns=composite.source_patterns,
            source_synergies=composite.source_synergies,
            proposed_changes=composite.proposed_changes,
            expected_benefits=composite.expected_benefits,
            priority=composite.priority,
            estimated_confidence=composite.estimated_confidence,
            created_at=composite.created_at,
            status=status,
            applied_to_branches=composite.applied_to_branches,
            results={**composite.results, **(results or {})},
        )
        
        self._composites[composite_id] = updated
        return updated
    
    def get_synthesis_summary(self) -> dict[str, Any]:
        """Get a summary of all synthesis activities.
        
        Returns:
            Dictionary containing synthesis summary
        """
        return {
            "agent_id": str(self._agent_id),
            "review_interval_seconds": self._review_interval_seconds,
            "last_review_time": self._last_review_time.isoformat() if self._last_review_time else None,
            "is_reviewing": self._is_reviewing,
            "statistics": {
                "total_improvements": self.improvement_count,
                "total_patterns": self.pattern_count,
                "total_synergies": self.synergy_count,
                "total_composites": self.composite_count,
                "synthesis_operations": len(self._synthesis_history),
            },
            "patterns_by_type": {
                pt.name: len(self.get_patterns_by_type(pt))
                for pt in PatternType
            },
            "composites_by_status": {
                status.name: len([
                    c for c in self._composites.values()
                    if c.status == status
                ])
                for status in SynthesisStatus
            },
        }
    
    async def clear_history(self) -> None:
        """Clear all synthesis history."""
        self._improvements.clear()
        self._patterns.clear()
        self._synergies.clear()
        self._composites.clear()
        self._synthesis_history.clear()
        self._findings_buffer.clear()
    
    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------
    
    def _normalize_context(
        self,
        context: SynthesiserContext | ExecutionContext,
    ) -> SynthesiserContext:
        """Normalize context to SynthesiserContext type.
        
        Args:
            context: Context to normalize
        
        Returns:
            Normalized SynthesiserContext
        """
        if isinstance(context, SynthesiserContext):
            return context
        
        # Convert ExecutionContext to SynthesiserContext
        params = context.parameters
        
        focus_patterns = [
            PatternType[p] for p in params.get("focus_patterns", [])
            if p in PatternType.__members__
        ]
        
        return SynthesiserContext(
            improvements_to_review=params.get("improvements_to_review"),
            focus_patterns=focus_patterns,
            min_confidence_threshold=params.get("min_confidence_threshold", self.DEFAULT_MIN_CONFIDENCE),
            max_composites=params.get("max_composites", self.DEFAULT_MAX_COMPOSITES),
            require_synergies=params.get("require_synergies", True),
            target_branches=params.get("target_branches", []),
        )
    
    async def _periodic_review_loop(self) -> None:
        """Background loop for periodic synthesis."""
        while True:
            try:
                await asyncio.sleep(self._review_interval_seconds)
                
                # Skip if already reviewing
                if self._is_reviewing:
                    continue
                
                # Perform synthesis
                context = SynthesiserContext()
                await self.execute(context)
                
            except asyncio.CancelledError:
                break
            except Exception:
                # Log error and continue
                await asyncio.sleep(1.0)
    
    async def _perform_synthesis_internal(
        self,
        context: SynthesiserContext,
    ) -> SynthesisResult:
        """Internal method to perform synthesis.
        
        Args:
            context: Synthesis context
        
        Returns:
            Synthesis result
        """
        self._is_reviewing = True
        self._last_review_time = datetime.utcnow()
        
        synthesis_id = f"synthesis_{int(time.time())}_{len(self._synthesis_history)}"
        
        try:
            # Get improvements to review
            improvements = self._get_improvements_for_review(context)
            
            # Identify patterns
            patterns = await self._identify_patterns(improvements, context)
            
            # Find synergies
            synergies = await self._find_synergies(improvements, patterns, context)
            
            # Propose composites
            composites = await self._propose_composites(
                improvements, patterns, synergies, context
            )
            
            # Create result
            result = SynthesisResult(
                synthesis_id=synthesis_id,
                status=SynthesisStatus.COMPLETED,
                patterns_identified=list(patterns.values()),
                synergies_found=list(synergies.values()),
                composites_proposed=list(composites.values()),
                reviewed_improvements=len(improvements),
            )
            
            # Store patterns, synergies, and composites
            for pattern in patterns.values():
                self._patterns[pattern.pattern_id] = pattern
                if self._on_pattern_identified:
                    await self._on_pattern_identified(pattern)
            
            for synergy in synergies.values():
                self._synergies[synergy.synergy_id] = synergy
                if self._on_synergy_found:
                    await self._on_synergy_found(synergy)
            
            for composite in composites.values():
                self._composites[composite.composite_id] = composite
                if self._on_composite_proposed:
                    await self._on_composite_proposed(composite)
            
            self._synthesis_history.append(result)
            
            # Broadcast completion
            if self._broadcast_channel is not None:
                await self.send_message(
                    content={
                        "type": "synthesis_completed",
                        "synthesis_id": synthesis_id,
                        "patterns_found": len(patterns),
                        "synergies_found": len(synergies),
                        "composites_proposed": len(composites),
                    },
                    priority=MessagePriority.NORMAL,
                )
            
            self._is_reviewing = False
            return result
            
        except Exception as e:
            self._is_reviewing = False
            return SynthesisResult(
                synthesis_id=synthesis_id,
                status=SynthesisStatus.FAILED,
                metadata={"error": str(e)},
            )
    
    def _get_improvements_for_review(
        self,
        context: SynthesiserContext,
    ) -> list[ValidatedImprovement]:
        """Get the list of improvements to review.
        
        Args:
            context: Synthesis context
        
        Returns:
            List of improvements to review
        """
        improvements = list(self._improvements.values())
        
        # Filter by specific improvements if specified
        if context.improvements_to_review:
            improvements = [
                imp for imp in improvements
                if imp.improvement_id in context.improvements_to_review
            ]
        
        # Filter by target branches
        if context.target_branches:
            improvements = [
                imp for imp in improvements
                if imp.branch_id in context.target_branches
            ]
        
        # Filter by confidence threshold
        improvements = [
            imp for imp in improvements
            if imp.validation_confidence >= context.min_confidence_threshold
        ]
        
        # Filter by time range
        if context.time_range:
            start, end = context.time_range
            improvements = [
                imp for imp in improvements
                if start <= imp.validated_at <= end
            ]
        
        return improvements
    
    async def _identify_patterns(
        self,
        improvements: list[ValidatedImprovement],
        context: SynthesiserContext,
    ) -> dict[str, IdentifiedPattern]:
        """Identify patterns across improvements.
        
        Args:
            improvements: List of improvements to analyze
            context: Synthesis context
        
        Returns:
            Dictionary of identified patterns
        """
        patterns: dict[str, IdentifiedPattern] = {}
        
        # Group improvements by type
        by_type: dict[str, list[ValidatedImprovement]] = {}
        for imp in improvements:
            imp_type = imp.improvement_type
            if imp_type not in by_type:
                by_type[imp_type] = []
            by_type[imp_type].append(imp)
        
        # Identify type-based patterns
        for imp_type, type_improvements in by_type.items():
            if len(type_improvements) >= 2:
                pattern_id = f"pattern_type_{imp_type}_{int(time.time())}"
                
                # Determine pattern type
                pattern_type = self._map_improvement_type_to_pattern(imp_type)
                
                # Skip if not in focus patterns
                if context.focus_patterns and pattern_type not in context.focus_patterns:
                    continue
                
                affected_components = list(set(
                    comp for imp in type_improvements
                    for comp in imp.metadata.get("components", [])
                ))
                
                pattern = IdentifiedPattern(
                    pattern_id=pattern_id,
                    pattern_type=pattern_type,
                    description=f"Recurring {imp_type} improvements across {len(type_improvements)} instances",
                    contributing_improvements=[imp.improvement_id for imp in type_improvements],
                    confidence=min(0.5 + len(type_improvements) * 0.1, 0.95),
                    frequency=len(type_improvements),
                    affected_components=affected_components,
                )
                patterns[pattern_id] = pattern
        
        # Identify tag-based patterns
        by_tag: dict[str, list[ValidatedImprovement]] = {}
        for imp in improvements:
            for tag in imp.tags:
                if tag not in by_tag:
                    by_tag[tag] = []
                by_tag[tag].append(imp)
        
        for tag, tag_improvements in by_tag.items():
            if len(tag_improvements) >= 3:
                pattern_id = f"pattern_tag_{tag}_{int(time.time())}"
                
                pattern = IdentifiedPattern(
                    pattern_id=pattern_id,
                    pattern_type=PatternType.COMPOSITE_SYNERGY,
                    description=f"Common tag '{tag}' across {len(tag_improvements)} improvements",
                    contributing_improvements=[imp.improvement_id for imp in tag_improvements],
                    confidence=min(0.4 + len(tag_improvements) * 0.08, 0.9),
                    frequency=len(tag_improvements),
                )
                patterns[pattern_id] = pattern
        
        # Use LLM for advanced pattern detection if available
        if self._llm_provider is not None and len(improvements) > 5:
            llm_patterns = await self._identify_patterns_with_llm(improvements, context)
            patterns.update(llm_patterns)
        
        return patterns
    
    async def _identify_patterns_with_llm(
        self,
        improvements: list[ValidatedImprovement],
        context: SynthesiserContext,
    ) -> dict[str, IdentifiedPattern]:
        """Use LLM to identify advanced patterns.
        
        Args:
            improvements: List of improvements to analyze
            context: Synthesis context
        
        Returns:
            Dictionary of identified patterns
        """
        patterns: dict[str, IdentifiedPattern] = {}
        
        if self._llm_provider is None:
            return patterns
        
        # Build prompt
        prompt = self._build_pattern_identification_prompt(improvements)
        
        try:
            response = await self._llm_provider.generate(prompt, max_tokens=2000)
            
            # Parse response for patterns
            # This is a simplified parsing - in practice, you'd use structured output
            if "optimization" in response.lower() and "performance" in response.lower():
                pattern_id = f"pattern_llm_performance_{int(time.time())}"
                patterns[pattern_id] = IdentifiedPattern(
                    pattern_id=pattern_id,
                    pattern_type=PatternType.PERFORMANCE_OPTIMIZATION,
                    description="Performance optimization opportunities identified by LLM analysis",
                    contributing_improvements=[imp.improvement_id for imp in improvements[:5]],
                    confidence=0.7,
                    frequency=len(improvements),
                )
            
        except Exception:
            # Fail silently - LLM pattern detection is optional
            pass
        
        return patterns
    
    async def _find_synergies(
        self,
        improvements: list[ValidatedImprovement],
        patterns: dict[str, IdentifiedPattern],
        context: SynthesiserContext,
    ) -> dict[str, Synergy]:
        """Find synergies between improvements.
        
        Args:
            improvements: List of improvements to analyze
            patterns: Identified patterns
            context: Synthesis context
        
        Returns:
            Dictionary of discovered synergies
        """
        synergies: dict[str, Synergy] = {}
        
        # Find synergies between improvements in the same pattern
        for pattern in patterns.values():
            contrib_ids = pattern.contributing_improvements
            if len(contrib_ids) >= 2:
                # Check for additive effects
                synergy_id = f"synergy_{pattern.pattern_id}_{int(time.time())}"
                
                # Calculate estimated combined impact
                combined_impact: dict[str, float] = {}
                for imp_id in contrib_ids:
                    imp = self._improvements.get(imp_id)
                    if imp:
                        for metric, ratio in imp.improvement_ratio.items():
                            if metric not in combined_impact:
                                combined_impact[metric] = 1.0
                            combined_impact[metric] *= ratio
                
                synergy = Synergy(
                    synergy_id=synergy_id,
                    improvement_ids=contrib_ids[:5],  # Limit to 5
                    synergy_type=SynergyType.ADDITIVE,
                    description=f"Additive synergy from {len(contrib_ids)} improvements in pattern '{pattern.pattern_type.name}'",
                    estimated_combined_impact=combined_impact,
                    confidence=pattern.confidence * 0.9,
                )
                synergies[synergy_id] = synergy
        
        # Find synergies between improvements with common tags
        for i, imp1 in enumerate(improvements):
            for imp2 in improvements[i+1:]:
                common_tags = imp1.tags & imp2.tags
                if len(common_tags) >= 2:
                    synergy_id = f"synergy_tags_{imp1.improvement_id}_{imp2.improvement_id}"
                    
                    synergy = Synergy(
                        synergy_id=synergy_id,
                        improvement_ids=[imp1.improvement_id, imp2.improvement_id],
                        synergy_type=SynergyType.COMPLEMENTARY,
                        description=f"Complementary synergy from shared tags: {', '.join(common_tags)}",
                        estimated_combined_impact={},
                        confidence=min(imp1.validation_confidence, imp2.validation_confidence) * 0.85,
                    )
                    synergies[synergy_id] = synergy
        
        return synergies
    
    async def _propose_composites(
        self,
        improvements: list[ValidatedImprovement],
        patterns: dict[str, IdentifiedPattern],
        synergies: dict[str, Synergy],
        context: SynthesiserContext,
    ) -> dict[str, CompositeImprovement]:
        """Propose composite improvements.
        
        Args:
            improvements: List of improvements to analyze
            patterns: Identified patterns
            synergies: Discovered synergies
            context: Synthesis context
        
        Returns:
            Dictionary of proposed composites
        """
        composites: dict[str, CompositeImprovement] = {}
        
        # Generate composites from patterns with high confidence
        for pattern in patterns.values():
            if pattern.confidence < context.min_confidence_threshold:
                continue
            
            # Skip if no synergies and they're required
            if context.require_synergies and not synergies:
                continue
            
            composite_id = f"composite_{pattern.pattern_id}_{int(time.time())}"
            
            # Get contributing improvements
            contrib_improvements = [
                self._improvements.get(imp_id)
                for imp_id in pattern.contributing_improvements
                if imp_id in self._improvements
            ]
            
            if len(contrib_improvements) < 2:
                continue
            
            # Aggregate changes
            all_changes = []
            for imp in contrib_improvements:
                if imp:
                    all_changes.extend(imp.changes)
            
            # Determine priority based on pattern type
            priority = self._determine_priority(pattern.pattern_type)
            
            # Calculate estimated confidence
            avg_confidence = sum(imp.validation_confidence for imp in contrib_improvements if imp) / len(contrib_improvements)
            
            # Find related synergies
            related_synergies = [
                s.synergy_id for s in synergies.values()
                if any(imp_id in pattern.contributing_improvements for imp_id in s.improvement_ids)
            ]
            
            composite = CompositeImprovement(
                composite_id=composite_id,
                name=f"Composite: {pattern.pattern_type.name.replace('_', ' ').title()}",
                description=f"Combined improvement addressing {pattern.description}",
                source_improvements=pattern.contributing_improvements[:5],
                source_patterns=[pattern.pattern_id],
                source_synergies=related_synergies[:3],
                proposed_changes=all_changes[:10],  # Limit changes
                expected_benefits={
                    "pattern_type": pattern.pattern_type.name,
                    "contributing_improvements": len(pattern.contributing_improvements),
                    "estimated_confidence": avg_confidence,
                },
                priority=priority,
                estimated_confidence=avg_confidence * pattern.confidence,
            )
            
            composites[composite_id] = composite
            
            # Check if we've reached max composites
            if len(composites) >= context.max_composites:
                break
        
        # Generate composites from high-confidence synergies
        for synergy in synergies.values():
            if synergy.confidence < context.min_confidence_threshold:
                continue
            
            if len(composites) >= context.max_composites:
                break
            
            composite_id = f"composite_synergy_{synergy.synergy_id}_{int(time.time())}"
            
            # Skip if similar composite already exists
            if any(set(c.source_improvements) == set(synergy.improvement_ids) for c in composites.values()):
                continue
            
            composite = CompositeImprovement(
                composite_id=composite_id,
                name=f"Synergy Composite: {synergy.synergy_type.value.title()}",
                description=synergy.description,
                source_improvements=synergy.improvement_ids,
                source_patterns=[],
                source_synergies=[synergy.synergy_id],
                proposed_changes=[],  # Would be populated from actual changes
                expected_benefits=synergy.estimated_combined_impact,
                priority=CompositePriority.MEDIUM,
                estimated_confidence=synergy.confidence,
            )
            
            composites[composite_id] = composite
        
        return composites
    
    def _map_improvement_type_to_pattern(self, improvement_type: str) -> PatternType:
        """Map an improvement type to a pattern type.
        
        Args:
            improvement_type: The improvement type string
        
        Returns:
            The corresponding PatternType
        """
        type_mapping = {
            "performance": PatternType.PERFORMANCE_OPTIMIZATION,
            "optimization": PatternType.PERFORMANCE_OPTIMIZATION,
            "architecture": PatternType.ARCHITECTURAL_IMPROVEMENT,
            "refactor": PatternType.ARCHITECTURAL_IMPROVEMENT,
            "bugfix": PatternType.BUG_FIX_PATTERN,
            "bug_fix": PatternType.BUG_FIX_PATTERN,
            "config": PatternType.CONFIGURATION_TWEAK,
            "configuration": PatternType.CONFIGURATION_TWEAK,
            "quality": PatternType.CODE_QUALITY,
            "security": PatternType.SECURITY_ENHANCEMENT,
            "scale": PatternType.SCALABILITY_IMPROVEMENT,
            "scalability": PatternType.SCALABILITY_IMPROVEMENT,
        }
        
        imp_type_lower = improvement_type.lower()
        for key, pattern_type in type_mapping.items():
            if key in imp_type_lower:
                return pattern_type
        
        return PatternType.COMPOSITE_SYNERGY
    
    def _determine_priority(self, pattern_type: PatternType) -> CompositePriority:
        """Determine priority based on pattern type.
        
        Args:
            pattern_type: The pattern type
        
        Returns:
            The corresponding priority
        """
        priority_mapping = {
            PatternType.SECURITY_ENHANCEMENT: CompositePriority.CRITICAL,
            PatternType.BUG_FIX_PATTERN: CompositePriority.HIGH,
            PatternType.PERFORMANCE_OPTIMIZATION: CompositePriority.HIGH,
            PatternType.SCALABILITY_IMPROVEMENT: CompositePriority.MEDIUM,
            PatternType.ARCHITECTURAL_IMPROVEMENT: CompositePriority.MEDIUM,
            PatternType.CODE_QUALITY: CompositePriority.LOW,
            PatternType.CONFIGURATION_TWEAK: CompositePriority.LOW,
            PatternType.COMPOSITE_SYNERGY: CompositePriority.MEDIUM,
        }
        
        return priority_mapping.get(pattern_type, CompositePriority.MEDIUM)
    
    def _build_pattern_identification_prompt(
        self,
        improvements: list[ValidatedImprovement],
    ) -> str:
        """Build prompt for LLM pattern identification.
        
        Args:
            improvements: List of improvements to analyze
        
        Returns:
            Formatted prompt string
        """
        improvements_text = "\n".join([
            f"- {imp.improvement_id}: {imp.description} (type: {imp.improvement_type}, confidence: {imp.validation_confidence})"
            for imp in improvements[:10]
        ])
        
        return f"""You are a Pattern Recognition AI analyzing system improvements.

Analyze the following validated improvements and identify patterns:

IMPROVEMENTS:
{improvements_text}

Identify patterns such as:
1. Performance optimization opportunities
2. Common architectural improvements
3. Recurring bug fix patterns
4. Configuration optimizations
5. Code quality improvements

Describe any patterns you find and their potential combined impact."""
    
    def _dict_to_improvement(self, data: dict[str, Any]) -> ValidatedImprovement:
        """Convert a dictionary to a ValidatedImprovement.
        
        Args:
            data: Dictionary containing improvement data
        
        Returns:
            ValidatedImprovement object
        """
        return ValidatedImprovement(
            improvement_id=data.get("improvement_id", ""),
            branch_id=data.get("branch_id", ""),
            source_agent=data.get("source_agent", ""),
            improvement_type=data.get("improvement_type", ""),
            description=data.get("description", ""),
            changes=data.get("changes", []),
            metrics_before=data.get("metrics_before", {}),
            metrics_after=data.get("metrics_after", {}),
            validation_confidence=data.get("validation_confidence", 0.5),
            validated_at=datetime.fromisoformat(data["validated_at"]) if "validated_at" in data else datetime.utcnow(),
            tags=set(data.get("tags", [])),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Exception Classes
# ============================================================================

class SynthesisError(Exception):
    """Base exception for synthesis-related errors."""
    pass


class SynthesisExecutionError(SynthesisError):
    """Exception raised when synthesis execution fails."""
    
    def __init__(
        self,
        message: str,
        agent_id: AgentID | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.agent_id = agent_id
        self.cause = cause


class PatternIdentificationError(SynthesisError):
    """Exception raised when pattern identification fails."""
    pass


class SynergyDetectionError(SynthesisError):
    """Exception raised when synergy detection fails."""
    pass


# ============================================================================
# Factory Functions
# ============================================================================

async def create_synthesiser_agent(
    agent_id: AgentID | None = None,
    llm_provider: LLMProvider | None = None,
    checkpoint_store: CheckpointStore | None = None,
    broadcast_channel: BroadcastChannel | None = None,
    review_interval_seconds: float = 300.0,
    improvement_provider: ImprovementProvider | None = None,
    start_periodic_review: bool = False,
) -> SynthesiserAgent:
    """Factory function to create a SynthesiserAgent.
    
    Args:
        agent_id: Optional unique identifier
        llm_provider: Optional LLM provider for synthesis generation
        checkpoint_store: Optional checkpoint storage backend
        broadcast_channel: Optional broadcast communication channel
        review_interval_seconds: Interval between automatic synthesis runs
        improvement_provider: Optional provider for fetching improvements
        start_periodic_review: Whether to start periodic review immediately
    
    Returns:
        Configured SynthesiserAgent instance
    """
    agent = SynthesiserAgent(
        agent_id=agent_id,
        llm_provider=llm_provider,
        checkpoint_store=checkpoint_store,
        broadcast_channel=broadcast_channel,
        review_interval_seconds=review_interval_seconds,
        improvement_provider=improvement_provider,
    )
    
    await agent.initialize()
    
    if start_periodic_review:
        await agent.start_periodic_review()
    
    return agent


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Main agent
    "SynthesiserAgent",
    "create_synthesiser_agent",
    # Data classes
    "ValidatedImprovement",
    "IdentifiedPattern",
    "Synergy",
    "CompositeImprovement",
    "SynthesisResult",
    "SynthesiserContext",
    # Enums
    "SynthesisStatus",
    "PatternType",
    "SynergyType",
    "CompositePriority",
    # Protocols
    "ImprovementProvider",
    # Exceptions
    "SynthesisError",
    "SynthesisExecutionError",
    "PatternIdentificationError",
    "SynergyDetectionError",
]
