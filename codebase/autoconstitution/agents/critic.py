"""
ConstitutionalCriticAgent for autoconstitution Framework

This module implements a critic agent that evaluates proposed improvements using
constitutional AI principles. It generates structured critiques, argues against
acceptance, assigns confidence scores, and predicts specific failure modes.

Python Version: 3.11+
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Generic,
    TypeVar,
    cast,
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

class CriticismSeverity(Enum):
    """Severity levels for critique findings."""
    INFO = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class CritiqueStatus(Enum):
    """Status of a critique evaluation."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    REJECTED = auto()


class FailureModeCategory(Enum):
    """Categories of potential failure modes."""
    TECHNICAL = "technical"
    LOGICAL = "logical"
    ETHICAL = "ethical"
    PERFORMANCE = "performance"
    SECURITY = "security"
    USABILITY = "usability"
    MAINTAINABILITY = "maintainability"
    SCALABILITY = "scalability"


class ConstitutionalPrinciple(Enum):
    """Constitutional AI principles for evaluation."""
    HARMLESSNESS = "harmlessness"
    HONESTY = "honesty"
    HELPFULNESS = "helpfulness"
    TRANSPARENCY = "transparency"
    FAIRNESS = "fairness"
    ROBUSTNESS = "robustness"
    EFFICIENCY = "efficiency"
    CLARITY = "clarity"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True, slots=True)
class FailureMode:
    """Represents a predicted failure mode for a proposed improvement."""
    category: FailureModeCategory
    description: str
    likelihood: float  # 0.0 to 1.0
    impact: float  # 0.0 to 1.0
    mitigation: str | None = None
    
    @property
    def risk_score(self) -> float:
        """Calculate risk score as likelihood * impact."""
        return self.likelihood * self.impact
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "category": self.category.value,
            "description": self.description,
            "likelihood": self.likelihood,
            "impact": self.impact,
            "risk_score": self.risk_score,
            "mitigation": self.mitigation,
        }


@dataclass(frozen=True, slots=True)
class ConstitutionalViolation:
    """Represents a violation of a constitutional principle."""
    principle: ConstitutionalPrinciple
    severity: CriticismSeverity
    description: str
    evidence: str
    recommendation: str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "principle": self.principle.value,
            "severity": self.severity.name,
            "description": self.description,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
        }


@dataclass(frozen=True, slots=True)
class CounterArgument:
    """Represents a counter-argument against accepting a proposal."""
    argument_id: str
    title: str
    reasoning: str
    severity: CriticismSeverity
    supporting_evidence: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "argument_id": self.argument_id,
            "title": self.title,
            "reasoning": self.reasoning,
            "severity": self.severity.name,
            "supporting_evidence": self.supporting_evidence,
        }


@dataclass(slots=True)
class ProposedImprovement:
    """Represents a proposed improvement to be critiqued."""
    proposal_id: str
    title: str
    description: str
    author_id: AgentID | None = None
    proposed_changes: list[str] = field(default_factory=list)
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "proposal_id": self.proposal_id,
            "title": self.title,
            "description": self.description,
            "author_id": str(self.author_id) if self.author_id else None,
            "proposed_changes": self.proposed_changes,
            "rationale": self.rationale,
            "metadata": self.metadata,
            "submitted_at": self.submitted_at.isoformat(),
        }


@dataclass(slots=True)
class StructuredCritique:
    """Structured critique output from the ConstitutionalCriticAgent."""
    critique_id: str
    proposal_id: str
    critic_id: AgentID
    status: CritiqueStatus
    
    # Core critique components
    summary: str = ""
    counter_arguments: list[CounterArgument] = field(default_factory=list)
    constitutional_violations: list[ConstitutionalViolation] = field(default_factory=list)
    predicted_failure_modes: list[FailureMode] = field(default_factory=list)
    
    # Scoring
    confidence_score: float = 0.0  # 0.0 to 1.0 (confidence in rejection)
    overall_severity: CriticismSeverity = CriticismSeverity.INFO
    
    # Recommendation
    recommendation: str = ""
    conditions_for_acceptance: list[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def should_reject(self) -> bool:
        """Determine if the proposal should be rejected based on critique."""
        return (
            self.confidence_score > 0.7
            or self.overall_severity in (CriticismSeverity.HIGH, CriticismSeverity.CRITICAL)
            or any(v.severity == CriticismSeverity.CRITICAL for v in self.constitutional_violations)
        )
    
    @property
    def total_risk_score(self) -> float:
        """Calculate total risk score from all failure modes."""
        return sum(fm.risk_score for fm in self.predicted_failure_modes)
    
    @property
    def critical_issues_count(self) -> int:
        """Count the number of critical issues."""
        critical_violations = sum(
            1 for v in self.constitutional_violations 
            if v.severity == CriticismSeverity.CRITICAL
        )
        critical_arguments = sum(
            1 for a in self.counter_arguments 
            if a.severity == CriticismSeverity.CRITICAL
        )
        return critical_violations + critical_arguments
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "critique_id": self.critique_id,
            "proposal_id": self.proposal_id,
            "critic_id": str(self.critic_id),
            "status": self.status.name,
            "summary": self.summary,
            "counter_arguments": [a.to_dict() for a in self.counter_arguments],
            "constitutional_violations": [v.to_dict() for v in self.constitutional_violations],
            "predicted_failure_modes": [f.to_dict() for f in self.predicted_failure_modes],
            "confidence_score": self.confidence_score,
            "overall_severity": self.overall_severity.name,
            "should_reject": self.should_reject,
            "total_risk_score": self.total_risk_score,
            "critical_issues_count": self.critical_issues_count,
            "recommendation": self.recommendation,
            "conditions_for_acceptance": self.conditions_for_acceptance,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class CriticContext:
    """Context specific to critic agent execution."""
    proposal: ProposedImprovement
    evaluation_criteria: list[str] = field(default_factory=list)
    constitutional_principles: list[ConstitutionalPrinciple] = field(default_factory=list)
    focus_areas: list[str] = field(default_factory=list)
    min_confidence_threshold: float = 0.5
    require_failure_mode_prediction: bool = True
    max_counter_arguments: int = 5
    
    def to_execution_context(self, task_id: str | None = None) -> ExecutionContext:
        """Convert to base ExecutionContext."""
        return ExecutionContext(
            task_id=task_id or f"critique_{self.proposal.proposal_id}",
            parameters={
                "proposal": self.proposal.to_dict(),
                "evaluation_criteria": self.evaluation_criteria,
                "constitutional_principles": [p.value for p in self.constitutional_principles],
                "focus_areas": self.focus_areas,
                "min_confidence_threshold": self.min_confidence_threshold,
                "require_failure_mode_prediction": self.require_failure_mode_prediction,
                "max_counter_arguments": self.max_counter_arguments,
            },
        )


# ============================================================================
# ConstitutionalCriticAgent Implementation
# ============================================================================

class ConstitutionalCriticAgent(BaseAgent[CriticContext, StructuredCritique]):
    """Agent that generates structured critiques using constitutional AI principles.
    
    The ConstitutionalCriticAgent evaluates proposed improvements and generates
    detailed critiques that argue against acceptance. It follows the constitutional
    AI pattern by evaluating proposals against a set of principles and generating
    confidence scores along with specific failure mode predictions.
    
    Attributes:
        agent_id: Unique identifier for this agent instance
        constitutional_principles: List of principles to evaluate against
        critique_history: History of generated critiques
        
    Example:
        >>> agent = ConstitutionalCriticAgent()
        >>> proposal = ProposedImprovement(
        ...     proposal_id="prop_001",
        ...     title="Add caching layer",
        ...     description="Implement Redis caching for API responses"
        ... )
        >>> context = CriticContext(proposal=proposal)
        >>> result = await agent.execute(context)
        >>> critique = result.data
        >>> print(f"Confidence in rejection: {critique.confidence_score}")
    
    Args:
        agent_id: Optional unique identifier
        llm_provider: Optional LLM provider for critique generation
        checkpoint_store: Optional checkpoint storage backend
        broadcast_channel: Optional broadcast communication channel
        constitutional_principles: Optional list of principles to enforce
    """
    
    # Default constitutional principles if none provided
    DEFAULT_PRINCIPLES: list[ConstitutionalPrinciple] = [
        ConstitutionalPrinciple.HARMLESSNESS,
        ConstitutionalPrinciple.HONESTY,
        ConstitutionalPrinciple.HELPFULNESS,
        ConstitutionalPrinciple.ROBUSTNESS,
    ]
    
    # Default evaluation criteria
    DEFAULT_CRITERIA: list[str] = [
        "Technical correctness",
        "Potential side effects",
        "Edge case handling",
        "Performance implications",
        "Security considerations",
        "Maintainability impact",
    ]
    
    def __init__(
        self,
        agent_id: AgentID | None = None,
        llm_provider: LLMProvider | None = None,
        checkpoint_store: CheckpointStore | None = None,
        broadcast_channel: BroadcastChannel | None = None,
        constitutional_principles: list[ConstitutionalPrinciple] | None = None,
    ) -> None:
        """Initialize the ConstitutionalCriticAgent.
        
        Args:
            agent_id: Optional unique identifier
            llm_provider: Optional LLM provider for critique generation
            checkpoint_store: Optional checkpoint storage backend
            broadcast_channel: Optional broadcast communication channel
            constitutional_principles: Optional list of principles to enforce
        """
        super().__init__(
            agent_id=agent_id,
            llm_provider=llm_provider,
            checkpoint_store=checkpoint_store,
            broadcast_channel=broadcast_channel,
        )
        
        self._constitutional_principles: list[ConstitutionalPrinciple] = (
            constitutional_principles or self.DEFAULT_PRINCIPLES.copy()
        )
        self._critique_history: list[StructuredCritique] = []
        self._evaluation_criteria: list[str] = self.DEFAULT_CRITERIA.copy()
        self._current_critique: StructuredCritique | None = None
        self._findings_buffer: list[Finding] = []
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def constitutional_principles(self) -> list[ConstitutionalPrinciple]:
        """Return the configured constitutional principles."""
        return self._constitutional_principles.copy()
    
    @property
    def critique_history(self) -> list[StructuredCritique]:
        """Return the history of generated critiques."""
        return self._critique_history.copy()
    
    @property
    def evaluation_criteria(self) -> list[str]:
        """Return the evaluation criteria."""
        return self._evaluation_criteria.copy()
    
    @evaluation_criteria.setter
    def evaluation_criteria(self, criteria: list[str]) -> None:
        """Set the evaluation criteria."""
        self._evaluation_criteria = criteria.copy()
    
    # -------------------------------------------------------------------------
    # Core Abstract Method Implementations
    # -------------------------------------------------------------------------
    
    async def execute(
        self,
        context: CriticContext | ExecutionContext,
    ) -> ExecutionResult[StructuredCritique]:
        """Execute the critique generation process.
        
        This is the main entry point for the critic agent. It evaluates
        a proposed improvement and generates a structured critique.
        
        Args:
            context: Execution context containing the proposal to critique
        
        Returns:
            ExecutionResult containing the structured critique
        """
        import time
        start_time = time.time()
        
        self._update_status(AgentStatus.EXECUTING)
        self._findings_buffer = []
        
        try:
            # Normalize context
            critic_context = self._normalize_context(context)
            
            # Create critique structure
            critique_id = f"critique_{critic_context.proposal.proposal_id}_{int(time.time())}"
            self._current_critique = StructuredCritique(
                critique_id=critique_id,
                proposal_id=critic_context.proposal.proposal_id,
                critic_id=self._agent_id,
                status=CritiqueStatus.IN_PROGRESS,
            )
            
            # Generate critique components
            await self._generate_summary(critic_context)
            await self._generate_counter_arguments(critic_context)
            await self._evaluate_constitutional_principles(critic_context)
            await self._predict_failure_modes(critic_context)
            await self._calculate_confidence_score(critic_context)
            await self._generate_recommendation(critic_context)
            
            # Finalize critique
            self._current_critique.status = CritiqueStatus.COMPLETED
            critique = self._current_critique
            self._critique_history.append(critique)
            
            # Create finding
            finding = create_finding(
                agent_id=self._agent_id,
                finding_type="constitutional_critique",
                content=critique.to_dict(),
                confidence=critique.confidence_score,
                critique_id=critique_id,
                proposal_id=critic_context.proposal.proposal_id,
            )
            self._findings_buffer.append(finding)
            
            execution_time = (time.time() - start_time) * 1000
            result = ExecutionResult(
                success=True,
                data=critique,
                execution_time_ms=execution_time,
                findings=self._findings_buffer.copy(),
                metadata={
                    "critique_id": critique_id,
                    "proposal_id": critic_context.proposal.proposal_id,
                    "counter_arguments_count": len(critique.counter_arguments),
                    "violations_count": len(critique.constitutional_violations),
                    "failure_modes_count": len(critique.predicted_failure_modes),
                },
            )
            
            self._record_execution(result)
            self._update_status(AgentStatus.IDLE)
            
            return result
            
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
            Sequence of Finding objects representing critique results
        """
        return self._findings_buffer.copy()
    
    async def receive_broadcast(self, message: Message) -> None:
        """Receive and process a broadcast message.
        
        Handles incoming messages from other agents in the swarm.
        Specifically listens for new proposals to critique.
        
        Args:
            message: The broadcast message to process
        """
        content = message.content
        
        # Check if this is a proposal message
        if isinstance(content, dict) and content.get("type") == "proposal":
            proposal_data = content.get("proposal", {})
            # Could trigger automatic critique here
            pass
        
        # Check if this is a critique request
        if isinstance(content, dict) and content.get("type") == "critique_request":
            # Handle critique request
            pass
    
    async def checkpoint(self, level: CheckpointLevel = CheckpointLevel.STANDARD) -> Checkpoint:
        """Create a checkpoint of the agent's current state.
        
        Args:
            level: Granularity level for the checkpoint
        
        Returns:
            Checkpoint object representing the saved state
        """
        state_data: dict[str, Any] = {
            "constitutional_principles": [p.value for p in self._constitutional_principles],
            "evaluation_criteria": self._evaluation_criteria,
            "critique_history_count": len(self._critique_history),
            "current_critique": self._current_critique.to_dict() if self._current_critique else None,
        }
        
        if level in (CheckpointLevel.STANDARD, CheckpointLevel.FULL):
            state_data["critique_history"] = [c.to_dict() for c in self._critique_history[-10:]]
        
        if level == CheckpointLevel.FULL:
            state_data["all_critiques"] = [c.to_dict() for c in self._critique_history]
            state_data["findings_buffer"] = [
                {
                    "agent_id": str(f.agent_id),
                    "finding_type": f.finding_type,
                    "content": f.content,
                    "confidence": f.confidence,
                }
                for f in self._findings_buffer
            ]
        
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
            
            # Restore principles
            if "constitutional_principles" in state_data:
                self._constitutional_principles = [
                    ConstitutionalPrinciple(p) for p in state_data["constitutional_principles"]
                ]
            
            # Restore criteria
            if "evaluation_criteria" in state_data:
                self._evaluation_criteria = state_data["evaluation_criteria"]
            
            return True
        except Exception:
            return False
    
    # -------------------------------------------------------------------------
    # Public API Methods
    # -------------------------------------------------------------------------
    
    async def critique_proposal(
        self,
        proposal: ProposedImprovement,
        custom_criteria: list[str] | None = None,
    ) -> StructuredCritique:
        """Convenience method to critique a proposal.
        
        Args:
            proposal: The proposal to critique
            custom_criteria: Optional custom evaluation criteria
        
        Returns:
            Structured critique of the proposal
        """
        context = CriticContext(
            proposal=proposal,
            evaluation_criteria=custom_criteria or self._evaluation_criteria.copy(),
            constitutional_principles=self._constitutional_principles.copy(),
        )
        
        result = await self.execute(context)
        
        if not result.success or result.data is None:
            raise CriticExecutionError(
                "Failed to generate critique",
                agent_id=self._agent_id,
                cause=result.error,
            )
        
        return result.data
    
    async def stream_critique(
        self,
        proposal: ProposedImprovement,
    ) -> AsyncIterator[str]:
        """Stream critique generation in real-time.
        
        Args:
            proposal: The proposal to critique
        
        Yields:
            Chunks of the generated critique
        """
        if self._llm_provider is None:
            raise RuntimeError("LLM provider required for streaming")
        
        prompt = self._build_critique_prompt(proposal)
        
        async for chunk in self._llm_provider.generate_stream(prompt):
            yield chunk
    
    def add_constitutional_principle(
        self,
        principle: ConstitutionalPrinciple,
    ) -> None:
        """Add a constitutional principle to evaluate against.
        
        Args:
            principle: The principle to add
        """
        if principle not in self._constitutional_principles:
            self._constitutional_principles.append(principle)
    
    def remove_constitutional_principle(
        self,
        principle: ConstitutionalPrinciple,
    ) -> bool:
        """Remove a constitutional principle.
        
        Args:
            principle: The principle to remove
        
        Returns:
            True if removed, False if not found
        """
        if principle in self._constitutional_principles:
            self._constitutional_principles.remove(principle)
            return True
        return False
    
    def get_critique_by_id(self, critique_id: str) -> StructuredCritique | None:
        """Retrieve a critique by its ID.
        
        Args:
            critique_id: ID of the critique to retrieve
        
        Returns:
            The critique if found, None otherwise
        """
        for critique in self._critique_history:
            if critique.critique_id == critique_id:
                return critique
        return None
    
    def get_critiques_for_proposal(
        self,
        proposal_id: str,
    ) -> list[StructuredCritique]:
        """Get all critiques for a specific proposal.
        
        Args:
            proposal_id: ID of the proposal
        
        Returns:
            List of critiques for the proposal
        """
        return [c for c in self._critique_history if c.proposal_id == proposal_id]
    
    def clear_history(self) -> None:
        """Clear the critique history."""
        self._critique_history.clear()
    
    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------
    
    def _normalize_context(
        self,
        context: CriticContext | ExecutionContext,
    ) -> CriticContext:
        """Normalize context to CriticContext type.
        
        Args:
            context: Context to normalize
        
        Returns:
            Normalized CriticContext
        """
        if isinstance(context, CriticContext):
            return context
        
        # Convert ExecutionContext to CriticContext
        params = context.parameters
        proposal_data = params.get("proposal", {})
        
        proposal = ProposedImprovement(
            proposal_id=proposal_data.get("proposal_id", "unknown"),
            title=proposal_data.get("title", "Untitled"),
            description=proposal_data.get("description", ""),
            proposed_changes=proposal_data.get("proposed_changes", []),
            rationale=proposal_data.get("rationale", ""),
            metadata=proposal_data.get("metadata", {}),
        )
        
        principles = [
            ConstitutionalPrinciple(p) 
            for p in params.get("constitutional_principles", [])
        ] or self._constitutional_principles.copy()
        
        return CriticContext(
            proposal=proposal,
            evaluation_criteria=params.get("evaluation_criteria", self._evaluation_criteria.copy()),
            constitutional_principles=principles,
            focus_areas=params.get("focus_areas", []),
            min_confidence_threshold=params.get("min_confidence_threshold", 0.5),
            require_failure_mode_prediction=params.get("require_failure_mode_prediction", True),
            max_counter_arguments=params.get("max_counter_arguments", 5),
        )
    
    async def _generate_summary(self, context: CriticContext) -> None:
        """Generate executive summary of the critique.
        
        Args:
            context: Critic execution context
        """
        if self._current_critique is None:
            return
        
        if self._llm_provider is not None:
            prompt = self._build_summary_prompt(context)
            summary = await self._llm_provider.generate(prompt, max_tokens=500)
            self._current_critique.summary = summary.strip()
        else:
            # Fallback summary without LLM
            self._current_critique.summary = (
                f"Critique of proposal '{context.proposal.title}': "
                f"Evaluating {len(context.constitutional_principles)} constitutional principles "
                f"and {len(context.evaluation_criteria)} criteria."
            )
    
    async def _generate_counter_arguments(self, context: CriticContext) -> None:
        """Generate counter-arguments against the proposal.
        
        Args:
            context: Critic execution context
        """
        if self._current_critique is None:
            return
        
        counter_args: list[CounterArgument] = []
        
        if self._llm_provider is not None:
            prompt = self._build_counter_argument_prompt(context)
            response = await self._llm_provider.generate(prompt, max_tokens=2000)
            counter_args = self._parse_counter_arguments(response)
        else:
            # Generate rule-based counter-arguments
            counter_args = self._generate_rule_based_counter_arguments(context)
        
        # Limit to max allowed
        max_args = context.max_counter_arguments
        self._current_critique.counter_arguments = counter_args[:max_args]
    
    async def _evaluate_constitutional_principles(self, context: CriticContext) -> None:
        """Evaluate proposal against constitutional principles.
        
        Args:
            context: Critic execution context
        """
        if self._current_critique is None:
            return
        
        violations: list[ConstitutionalViolation] = []
        principles = context.constitutional_principles or self._constitutional_principles
        
        for principle in principles:
            violation = await self._check_principle_violation(principle, context)
            if violation is not None:
                violations.append(violation)
        
        self._current_critique.constitutional_violations = violations
    
    async def _predict_failure_modes(self, context: CriticContext) -> None:
        """Predict specific failure modes for the proposal.
        
        Args:
            context: Critic execution context
        """
        if self._current_critique is None:
            return
        
        if not context.require_failure_mode_prediction:
            return
        
        failure_modes: list[FailureMode] = []
        
        if self._llm_provider is not None:
            prompt = self._build_failure_mode_prompt(context)
            response = await self._llm_provider.generate(prompt, max_tokens=2000)
            failure_modes = self._parse_failure_modes(response)
        else:
            # Generate rule-based failure modes
            failure_modes = self._generate_rule_based_failure_modes(context)
        
        self._current_critique.predicted_failure_modes = failure_modes
    
    async def _calculate_confidence_score(self, context: CriticContext) -> None:
        """Calculate confidence score for the critique.
        
        Args:
            context: Critic execution context
        """
        if self._current_critique is None:
            return
        
        critique = self._current_critique
        
        # Base confidence on severity and number of issues
        severity_weights = {
            CriticismSeverity.INFO: 0.1,
            CriticismSeverity.LOW: 0.2,
            CriticismSeverity.MEDIUM: 0.4,
            CriticismSeverity.HIGH: 0.7,
            CriticismSeverity.CRITICAL: 1.0,
        }
        
        # Calculate from counter-arguments
        arg_score = 0.0
        for arg in critique.counter_arguments:
            arg_score += severity_weights.get(arg.severity, 0.2)
        arg_score = min(arg_score / max(len(critique.counter_arguments), 1), 1.0)
        
        # Calculate from constitutional violations
        violation_score = 0.0
        for violation in critique.constitutional_violations:
            violation_score += severity_weights.get(violation.severity, 0.2)
        violation_score = min(violation_score / max(len(critique.constitutional_violations), 1), 1.0)
        
        # Calculate from failure modes
        risk_score = min(critique.total_risk_score, 1.0)
        
        # Combine scores with weights
        confidence = (arg_score * 0.3) + (violation_score * 0.4) + (risk_score * 0.3)
        
        # Boost for critical issues
        if critique.critical_issues_count > 0:
            confidence = min(confidence + 0.2, 1.0)
        
        critique.confidence_score = round(confidence, 3)
        
        # Determine overall severity
        if critique.critical_issues_count > 0:
            critique.overall_severity = CriticismSeverity.CRITICAL
        elif confidence > 0.7:
            critique.overall_severity = CriticismSeverity.HIGH
        elif confidence > 0.4:
            critique.overall_severity = CriticismSeverity.MEDIUM
        elif confidence > 0.2:
            critique.overall_severity = CriticismSeverity.LOW
        else:
            critique.overall_severity = CriticismSeverity.INFO
    
    async def _generate_recommendation(self, context: CriticContext) -> None:
        """Generate final recommendation.
        
        Args:
            context: Critic execution context
        """
        if self._current_critique is None:
            return
        
        critique = self._current_critique
        
        if self._llm_provider is not None:
            prompt = self._build_recommendation_prompt(context, critique)
            recommendation = await self._llm_provider.generate(prompt, max_tokens=1000)
            critique.recommendation = recommendation.strip()
        else:
            # Generate rule-based recommendation
            if critique.should_reject:
                critique.recommendation = (
                    f"REJECT: Proposal '{context.proposal.title}' has significant issues "
                    f"(confidence: {critique.confidence_score:.1%}). "
                    f"Found {critique.critical_issues_count} critical issues."
                )
            elif critique.confidence_score > 0.4:
                critique.recommendation = (
                    f"REVISION REQUIRED: Proposal '{context.proposal.title}' needs modifications "
                    f"before acceptance (confidence: {critique.confidence_score:.1%})."
                )
            else:
                critique.recommendation = (
                    f"ACCEPT WITH CAUTION: Proposal '{context.proposal.title}' has minor concerns "
                    f"(confidence: {critique.confidence_score:.1%})."
                )
        
        # Generate conditions for acceptance
        critique.conditions_for_acceptance = self._generate_conditions(critique)
    
    async def _check_principle_violation(
        self,
        principle: ConstitutionalPrinciple,
        context: CriticContext,
    ) -> ConstitutionalViolation | None:
        """Check if a proposal violates a constitutional principle.
        
        Args:
            principle: The principle to check
            context: Critic execution context
        
        Returns:
            Violation if found, None otherwise
        """
        if self._llm_provider is not None:
            prompt = self._build_principle_check_prompt(principle, context)
            response = await self._llm_provider.generate(prompt, max_tokens=1000)
            return self._parse_principle_violation(response, principle)
        
        # Rule-based principle checking
        return self._rule_based_principle_check(principle, context)
    
    def _generate_conditions(self, critique: StructuredCritique) -> list[str]:
        """Generate conditions that would allow acceptance.
        
        Args:
            critique: The current critique
        
        Returns:
            List of conditions for acceptance
        """
        conditions: list[str] = []
        
        for violation in critique.constitutional_violations:
            if violation.severity in (CriticismSeverity.HIGH, CriticismSeverity.CRITICAL):
                conditions.append(violation.recommendation)
        
        for failure_mode in critique.predicted_failure_modes:
            if failure_mode.risk_score > 0.5 and failure_mode.mitigation:
                conditions.append(failure_mode.mitigation)
        
        return conditions[:5]  # Limit to 5 conditions
    
    # -------------------------------------------------------------------------
    # Prompt Building Methods
    # -------------------------------------------------------------------------
    
    def _build_critique_prompt(self, proposal: ProposedImprovement) -> str:
        """Build the main critique generation prompt.
        
        Args:
            proposal: The proposal to critique
        
        Returns:
            Formatted prompt string
        """
        return f"""You are a Constitutional Critic AI. Your role is to rigorously evaluate proposed improvements and argue AGAINST their acceptance.

PROPOSAL TO CRITIQUE:
Title: {proposal.title}
Description: {proposal.description}
Proposed Changes:
{chr(10).join(f"- {change}" for change in proposal.proposed_changes)}
Rationale: {proposal.rationale}

CONSTITUTIONAL PRINCIPLES TO EVALUATE:
{chr(10).join(f"- {p.value}" for p in self._constitutional_principles)}

EVALUATION CRITERIA:
{chr(10).join(f"- {c}" for c in self._evaluation_criteria)}

Your task is to generate a comprehensive critique that:
1. Identifies specific weaknesses and flaws
2. Argues against acceptance with strong reasoning
3. Predicts concrete failure modes
4. Assigns a confidence score to your critique

Be thorough, specific, and constructive in your criticism."""
    
    def _build_summary_prompt(self, context: CriticContext) -> str:
        """Build prompt for generating critique summary.
        
        Args:
            context: Critic execution context
        
        Returns:
            Formatted prompt string
        """
        return f"""Provide a brief executive summary (2-3 sentences) of the critique for this proposal:

Title: {context.proposal.title}
Description: {context.proposal.description}

Focus on the most critical concerns and overall assessment."""
    
    def _build_counter_argument_prompt(self, context: CriticContext) -> str:
        """Build prompt for generating counter-arguments.
        
        Args:
            context: Critic execution context
        
        Returns:
            Formatted prompt string
        """
        return f"""Generate up to {context.max_counter_arguments} strong counter-arguments against this proposal:

Title: {context.proposal.title}
Description: {context.proposal.description}
Rationale: {context.proposal.rationale}

For each counter-argument, provide:
1. A clear title
2. Detailed reasoning
3. Severity level (INFO, LOW, MEDIUM, HIGH, CRITICAL)
4. Supporting evidence or examples

Format as JSON array:
[
  {{
    "title": "...",
    "reasoning": "...",
    "severity": "HIGH",
    "evidence": ["...", "..."]
  }}
]"""
    
    def _build_failure_mode_prompt(self, context: CriticContext) -> str:
        """Build prompt for predicting failure modes.
        
        Args:
            context: Critic execution context
        
        Returns:
            Formatted prompt string
        """
        return f"""Predict specific failure modes that could occur if this proposal is implemented:

Title: {context.proposal.title}
Description: {context.proposal.description}
Proposed Changes:
{chr(10).join(f"- {change}" for change in context.proposal.proposed_changes)}

For each failure mode, provide:
1. Category (technical, logical, ethical, performance, security, usability, maintainability, scalability)
2. Description of the failure
3. Likelihood (0.0-1.0)
4. Impact (0.0-1.0)
5. Suggested mitigation (optional)

Format as JSON array:
[
  {{
    "category": "technical",
    "description": "...",
    "likelihood": 0.7,
    "impact": 0.8,
    "mitigation": "..."
  }}
]"""
    
    def _build_principle_check_prompt(
        self,
        principle: ConstitutionalPrinciple,
        context: CriticContext,
    ) -> str:
        """Build prompt for checking a constitutional principle.
        
        Args:
            principle: The principle to check
            context: Critic execution context
        
        Returns:
            Formatted prompt string
        """
        return f"""Evaluate whether this proposal violates the constitutional principle of {principle.value}:

Title: {context.proposal.title}
Description: {context.proposal.description}

If a violation exists, provide:
1. Severity (INFO, LOW, MEDIUM, HIGH, CRITICAL)
2. Description of the violation
3. Evidence from the proposal
4. Recommendation to address it

Format as JSON or respond with "NO_VIOLATION" if none exists."""
    
    def _build_recommendation_prompt(
        self,
        context: CriticContext,
        critique: StructuredCritique,
    ) -> str:
        """Build prompt for generating final recommendation.
        
        Args:
            context: Critic execution context
            critique: The current critique
        
        Returns:
            Formatted prompt string
        """
        return f"""Based on the following critique, provide a final recommendation:

Proposal: {context.proposal.title}
Confidence Score: {critique.confidence_score}
Critical Issues: {critique.critical_issues_count}
Total Risk Score: {critique.total_risk_score}

Counter-Arguments: {len(critique.counter_arguments)}
Constitutional Violations: {len(critique.constitutional_violations)}
Predicted Failure Modes: {len(critique.predicted_failure_modes)}

Provide a clear recommendation (ACCEPT, REVISE, or REJECT) with reasoning."""
    
    # -------------------------------------------------------------------------
    # Parsing Methods
    # -------------------------------------------------------------------------
    
    def _parse_counter_arguments(self, response: str) -> list[CounterArgument]:
        """Parse counter-arguments from LLM response.
        
        Args:
            response: LLM response text
        
        Returns:
            List of parsed counter-arguments
        """
        arguments: list[CounterArgument] = []
        
        try:
            # Try to extract JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for i, item in enumerate(data):
                    severity_str = item.get("severity", "MEDIUM").upper()
                    severity = CriticismSeverity[severity_str]
                    
                    arg = CounterArgument(
                        argument_id=f"arg_{i}",
                        title=item.get("title", f"Counter-argument {i+1}"),
                        reasoning=item.get("reasoning", ""),
                        severity=severity,
                        supporting_evidence=item.get("evidence", []),
                    )
                    arguments.append(arg)
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: create single argument from text
            arguments.append(CounterArgument(
                argument_id="arg_0",
                title="General Concern",
                reasoning=response[:500],
                severity=CriticismSeverity.MEDIUM,
            ))
        
        return arguments
    
    def _parse_failure_modes(self, response: str) -> list[FailureMode]:
        """Parse failure modes from LLM response.
        
        Args:
            response: LLM response text
        
        Returns:
            List of parsed failure modes
        """
        failure_modes: list[FailureMode] = []
        
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for item in data:
                    category_str = item.get("category", "technical").lower()
                    category = FailureModeCategory(category_str)
                    
                    fm = FailureMode(
                        category=category,
                        description=item.get("description", ""),
                        likelihood=float(item.get("likelihood", 0.5)),
                        impact=float(item.get("impact", 0.5)),
                        mitigation=item.get("mitigation"),
                    )
                    failure_modes.append(fm)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Create generic failure mode from text
            failure_modes.append(FailureMode(
                category=FailureModeCategory.TECHNICAL,
                description=response[:300],
                likelihood=0.5,
                impact=0.5,
            ))
        
        return failure_modes
    
    def _parse_principle_violation(
        self,
        response: str,
        principle: ConstitutionalPrinciple,
    ) -> ConstitutionalViolation | None:
        """Parse principle violation from LLM response.
        
        Args:
            response: LLM response text
            principle: The principle being checked
        
        Returns:
            Violation if found, None otherwise
        """
        if "NO_VIOLATION" in response.upper():
            return None
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                severity_str = data.get("severity", "MEDIUM").upper()
                severity = CriticismSeverity[severity_str]
                
                return ConstitutionalViolation(
                    principle=principle,
                    severity=severity,
                    description=data.get("description", ""),
                    evidence=data.get("evidence", ""),
                    recommendation=data.get("recommendation", ""),
                )
        except (json.JSONDecodeError, KeyError):
            pass
        
        # If we can't parse but there's content, create a generic violation
        if len(response.strip()) > 50:
            return ConstitutionalViolation(
                principle=principle,
                severity=CriticismSeverity.LOW,
                description=f"Potential concern regarding {principle.value}",
                evidence=response[:300],
                recommendation="Review and address the identified concern",
            )
        
        return None
    
    # -------------------------------------------------------------------------
    # Rule-Based Fallback Methods
    # -------------------------------------------------------------------------
    
    def _generate_rule_based_counter_arguments(
        self,
        context: CriticContext,
    ) -> list[CounterArgument]:
        """Generate counter-arguments using rules (no LLM).
        
        Args:
            context: Critic execution context
        
        Returns:
            List of counter-arguments
        """
        arguments: list[CounterArgument] = []
        proposal = context.proposal
        
        # Check for insufficient description
        if len(proposal.description) < 100:
            arguments.append(CounterArgument(
                argument_id="arg_insufficient_description",
                title="Insufficient Description",
                reasoning="The proposal lacks detailed description, making it difficult to assess potential impacts and implementation requirements.",
                severity=CriticismSeverity.MEDIUM,
                supporting_evidence=[f"Description length: {len(proposal.description)} characters"],
            ))
        
        # Check for missing rationale
        if not proposal.rationale or len(proposal.rationale) < 50:
            arguments.append(CounterArgument(
                argument_id="arg_missing_rationale",
                title="Missing or Weak Rationale",
                reasoning="The proposal does not adequately explain why the change is necessary or what problem it solves.",
                severity=CriticismSeverity.MEDIUM,
                supporting_evidence=["Rationale is missing or too brief"],
            ))
        
        # Check for no proposed changes
        if not proposal.proposed_changes:
            arguments.append(CounterArgument(
                argument_id="arg_no_changes",
                title="No Specific Changes Defined",
                reasoning="The proposal does not specify what changes will be made, making implementation and review impossible.",
                severity=CriticismSeverity.HIGH,
                supporting_evidence=["Proposed changes list is empty"],
            ))
        
        # General skepticism argument
        arguments.append(CounterArgument(
            argument_id="arg_untested_changes",
            title="Risk of Untested Changes",
            reasoning="Any change to a working system introduces risk. Without thorough testing and validation, this proposal could introduce regressions.",
            severity=CriticismSeverity.LOW,
            supporting_evidence=["All changes carry inherent risk"],
        ))
        
        return arguments
    
    def _generate_rule_based_failure_modes(
        self,
        context: CriticContext,
    ) -> list[FailureMode]:
        """Generate failure modes using rules (no LLM).
        
        Args:
            context: Critic execution context
        
        Returns:
            List of failure modes
        """
        failure_modes: list[FailureMode] = []
        
        # Generic implementation failure
        failure_modes.append(FailureMode(
            category=FailureModeCategory.TECHNICAL,
            description="Implementation may not match specification due to misunderstood requirements",
            likelihood=0.3,
            impact=0.6,
            mitigation="Require detailed implementation plan and code review",
        ))
        
        # Performance degradation
        failure_modes.append(FailureMode(
            category=FailureModeCategory.PERFORMANCE,
            description="Changes may introduce performance regressions in production",
            likelihood=0.4,
            impact=0.7,
            mitigation="Require performance testing and benchmarking",
        ))
        
        # Edge case handling
        failure_modes.append(FailureMode(
            category=FailureModeCategory.LOGICAL,
            description="Edge cases may not be properly handled, causing unexpected behavior",
            likelihood=0.5,
            impact=0.5,
            mitigation="Require comprehensive test coverage including edge cases",
        ))
        
        return failure_modes
    
    def _rule_based_principle_check(
        self,
        principle: ConstitutionalPrinciple,
        context: CriticContext,
    ) -> ConstitutionalViolation | None:
        """Check principle violation using rules (no LLM).
        
        Args:
            principle: The principle to check
            context: Critic execution context
        
        Returns:
            Violation if found, None otherwise
        """
        proposal = context.proposal
        
        if principle == ConstitutionalPrinciple.CLARITY:
            if len(proposal.description) < 50:
                return ConstitutionalViolation(
                    principle=principle,
                    severity=CriticismSeverity.MEDIUM,
                    description="Proposal lacks clarity in its description",
                    evidence=f"Description is only {len(proposal.description)} characters",
                    recommendation="Expand the description with more details about the proposed changes",
                )
        
        if principle == ConstitutionalPrinciple.ROBUSTNESS:
            if not proposal.proposed_changes:
                return ConstitutionalViolation(
                    principle=principle,
                    severity=CriticismSeverity.HIGH,
                    description="Proposal does not specify changes, making robust implementation impossible",
                    evidence="No proposed changes listed",
                    recommendation="Define specific, testable changes",
                )
        
        return None
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> dict[str, Any]:
        """Convert agent state to dictionary representation.
        
        Returns:
            Dictionary containing agent metadata and state
        """
        base_dict = super().to_dict()
        base_dict.update({
            "agent_type": "ConstitutionalCriticAgent",
            "constitutional_principles": [p.value for p in self._constitutional_principles],
            "evaluation_criteria": self._evaluation_criteria,
            "critique_history_count": len(self._critique_history),
            "current_critique_id": self._current_critique.critique_id if self._current_critique else None,
        })
        return base_dict


# ============================================================================
# Exception Classes
# ============================================================================

class CriticError(Exception):
    """Base exception for critic-related errors."""
    pass


class CriticExecutionError(CriticError):
    """Exception raised when critic execution fails."""
    
    def __init__(
        self,
        message: str,
        agent_id: AgentID | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.agent_id = agent_id
        self.cause = cause


class InvalidProposalError(CriticError):
    """Exception raised when a proposal is invalid."""
    pass


class CritiqueParseError(CriticError):
    """Exception raised when critique parsing fails."""
    pass


# ============================================================================
# Factory Functions
# ============================================================================

def create_critic_agent(
    llm_provider: LLMProvider | None = None,
    constitutional_principles: list[ConstitutionalPrinciple] | None = None,
    **kwargs: Any,
) -> ConstitutionalCriticAgent:
    """Factory function to create a ConstitutionalCriticAgent.
    
    Args:
        llm_provider: Optional LLM provider for critique generation
        constitutional_principles: Optional list of principles to enforce
        **kwargs: Additional arguments passed to the agent constructor
    
    Returns:
        Configured ConstitutionalCriticAgent instance
    """
    return ConstitutionalCriticAgent(
        llm_provider=llm_provider,
        constitutional_principles=constitutional_principles,
        **kwargs,
    )


def create_proposed_improvement(
    proposal_id: str,
    title: str,
    description: str,
    **kwargs: Any,
) -> ProposedImprovement:
    """Factory function to create a ProposedImprovement.
    
    Args:
        proposal_id: Unique identifier for the proposal
        title: Short title of the proposal
        description: Detailed description
        **kwargs: Additional proposal attributes
    
    Returns:
        New ProposedImprovement instance
    """
    return ProposedImprovement(
        proposal_id=proposal_id,
        title=title,
        description=description,
        **kwargs,
    )


# ============================================================================
# Export List
# ============================================================================

__all__ = [
    # Main class
    "ConstitutionalCriticAgent",
    # Enums
    "CriticismSeverity",
    "CritiqueStatus",
    "FailureModeCategory",
    "ConstitutionalPrinciple",
    # Data classes
    "FailureMode",
    "ConstitutionalViolation",
    "CounterArgument",
    "ProposedImprovement",
    "StructuredCritique",
    "CriticContext",
    # Exceptions
    "CriticError",
    "CriticExecutionError",
    "InvalidProposalError",
    "CritiqueParseError",
    # Factory functions
    "create_critic_agent",
    "create_proposed_improvement",
]
