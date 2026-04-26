"""
BaseAgent Abstract Class for autoconstitution Framework

This module provides the abstract base class that all autoconstitution agents inherit from.
It defines the common interface and contract for agent behavior in the swarm.

Python Version: 3.11+
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)
from collections.abc import Sequence


# ============================================================================
# Type Variables
# ============================================================================

T = TypeVar("T")
TResult = TypeVar("TResult")
TContext = TypeVar("TContext")


# ============================================================================
# Enums
# ============================================================================

class AgentStatus(Enum):
    """Enumeration of possible agent states."""
    IDLE = auto()
    EXECUTING = auto()
    WAITING = auto()
    ERROR = auto()
    TERMINATED = auto()


class MessagePriority(Enum):
    """Priority levels for inter-agent communication."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class CheckpointLevel(Enum):
    """Levels of checkpoint granularity."""
    NONE = auto()
    MINIMAL = auto()
    STANDARD = auto()
    FULL = auto()


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True, slots=True)
class AgentID:
    """Unique identifier for an agent in the swarm."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Message:
    """Message for inter-agent communication."""
    sender_id: AgentID
    content: Any
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Finding:
    """Represents a finding or result from an agent's execution."""
    agent_id: AgentID
    finding_type: str
    content: Any
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Checkpoint:
    """Represents a checkpoint of agent state."""
    agent_id: AgentID
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    state_data: dict[str, Any] = field(default_factory=dict)
    level: CheckpointLevel = CheckpointLevel.STANDARD
    
    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to dictionary representation."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "agent_id": str(self.agent_id),
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.name,
            "state_data": self.state_data,
        }


@dataclass(slots=True)
class ExecutionContext:
    """Context passed to agent during execution."""
    task_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    parent_context: ExecutionContext | None = None
    depth: int = 0
    timeout_seconds: float | None = None
    
    def create_child(self, **override_params: Any) -> ExecutionContext:
        """Create a child execution context."""
        return ExecutionContext(
            task_id=self.task_id,
            parameters={**self.parameters, **override_params},
            parent_context=self,
            depth=self.depth + 1,
            timeout_seconds=self.timeout_seconds,
        )


@dataclass(slots=True)
class ExecutionResult(Generic[TResult]):
    """Result of agent execution."""
    success: bool
    data: TResult | None = None
    error: Exception | None = None
    execution_time_ms: float = 0.0
    findings: list[Finding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Protocols (Structural Subtyping)
# ============================================================================

@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for provider-agnostic LLM interface.
    
    Any LLM provider implementation must conform to this protocol
to be used with autoconstitution agents.
    """
    
    async def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt."""
        ...
    
    async def generate_stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream generated text tokens from a prompt."""
        ...
    
    async def embed(
        self,
        text: str,
        **kwargs: Any,
    ) -> list[float]:
        """Generate embedding vector for text."""
        ...
    
    def get_model_info(self) -> dict[str, Any]:
        """Return information about the underlying model."""
        ...


@runtime_checkable
class CheckpointStore(Protocol):
    """Protocol for checkpoint storage backend."""
    
    async def save(self, checkpoint: Checkpoint) -> str:
        """Save a checkpoint and return its storage key."""
        ...
    
    async def load(self, checkpoint_id: str) -> Checkpoint | None:
        """Load a checkpoint by ID."""
        ...
    
    async def list_checkpoints(
        self,
        agent_id: AgentID | None = None,
    ) -> list[str]:
        """List available checkpoint IDs."""
        ...
    
    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint by ID."""
        ...


@runtime_checkable
class BroadcastChannel(Protocol):
    """Protocol for broadcast communication channel."""
    
    async def broadcast(self, message: Message) -> None:
        """Broadcast a message to all subscribed agents."""
        ...
    
    async def subscribe(
        self,
        agent_id: AgentID,
        handler: Callable[[Message], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe an agent to receive broadcasts."""
        ...
    
    async def unsubscribe(self, agent_id: AgentID) -> None:
        """Unsubscribe an agent from broadcasts."""
        ...


# ============================================================================
# BaseAgent Abstract Class
# ============================================================================

class BaseAgent(ABC, Generic[TContext, TResult]):
    """Abstract base class for all autoconstitution agents.
    
    This class defines the common interface and contract for agent behavior
    in the swarm. All concrete agent implementations must inherit from this
    class and implement the abstract methods.
    
    Type Parameters:
        TContext: The type of context this agent accepts during execution
        TResult: The type of result this agent produces
    
    Example:
        class MyAgent(BaseAgent[MyContext, MyResult]):
            async def execute(self, context: MyContext) -> ExecutionResult[MyResult]:
                # Implementation here
                pass
    
    Attributes:
        agent_id: Unique identifier for this agent instance
        status: Current operational status of the agent
        llm_provider: Optional LLM provider for language model operations
        checkpoint_store: Optional storage backend for checkpoints
    """
    
    def __init__(
        self,
        agent_id: AgentID | None = None,
        llm_provider: LLMProvider | None = None,
        checkpoint_store: CheckpointStore | None = None,
        broadcast_channel: BroadcastChannel | None = None,
    ) -> None:
        """Initialize the base agent.
        
        Args:
            agent_id: Optional unique identifier (auto-generated if not provided)
            llm_provider: Optional LLM provider for language operations
            checkpoint_store: Optional checkpoint storage backend
            broadcast_channel: Optional broadcast communication channel
        """
        self._agent_id: AgentID = agent_id or AgentID()
        self._status: AgentStatus = AgentStatus.IDLE
        self._llm_provider: LLMProvider | None = llm_provider
        self._checkpoint_store: CheckpointStore | None = checkpoint_store
        self._broadcast_channel: BroadcastChannel | None = broadcast_channel
        self._message_handlers: list[Callable[[Message], Coroutine[Any, Any, None]]] = []
        self._execution_history: list[ExecutionResult[TResult]] = []
        self._created_at: datetime = datetime.utcnow()
        self._last_activity: datetime = datetime.utcnow()
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def agent_id(self) -> AgentID:
        """Return the unique identifier for this agent."""
        return self._agent_id
    
    @property
    def status(self) -> AgentStatus:
        """Return the current operational status."""
        return self._status
    
    @property
    def llm_provider(self) -> LLMProvider | None:
        """Return the configured LLM provider."""
        return self._llm_provider
    
    @llm_provider.setter
    def llm_provider(self, provider: LLMProvider | None) -> None:
        """Set or update the LLM provider."""
        self._llm_provider = provider
    
    @property
    def checkpoint_store(self) -> CheckpointStore | None:
        """Return the configured checkpoint store."""
        return self._checkpoint_store
    
    @property
    def created_at(self) -> datetime:
        """Return the agent creation timestamp."""
        return self._created_at
    
    @property
    def last_activity(self) -> datetime:
        """Return the timestamp of last agent activity."""
        return self._last_activity
    
    @property
    def execution_count(self) -> int:
        """Return the number of executions performed."""
        return len(self._execution_history)
    
    # -------------------------------------------------------------------------
    # Abstract Methods (Must be implemented by subclasses)
    # -------------------------------------------------------------------------
    
    @abstractmethod
    async def execute(
        self,
        context: TContext | ExecutionContext,
    ) -> ExecutionResult[TResult]:
        """Execute the agent's primary task.
        
        This is the main entry point for agent operation. Subclasses must
        implement this method to define their specific behavior.
        
        Args:
            context: Execution context containing task parameters and metadata
        
        Returns:
            ExecutionResult containing the outcome of execution
        
        Raises:
            AgentExecutionError: If execution fails in an unrecoverable way
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    @abstractmethod
    async def report_findings(self) -> Sequence[Finding]:
        """Report findings from the agent's execution.
        
        Returns a sequence of findings that represent the agent's
        discoveries, results, or outputs from its operations.
        
        Returns:
            Sequence of Finding objects representing agent discoveries
        """
        raise NotImplementedError("Subclasses must implement report_findings()")
    
    @abstractmethod
    async def receive_broadcast(self, message: Message) -> None:
        """Receive and process a broadcast message.
        
        This method is called when the agent receives a message from
        the broadcast channel. Subclasses should implement appropriate
        handling logic for their specific use case.
        
        Args:
            message: The broadcast message to process
        """
        raise NotImplementedError("Subclasses must implement receive_broadcast()")
    
    @abstractmethod
    async def checkpoint(self, level: CheckpointLevel = CheckpointLevel.STANDARD) -> Checkpoint:
        """Create a checkpoint of the agent's current state.
        
        Checkpoints allow for state persistence and recovery. The level
        parameter controls the granularity of the checkpoint.
        
        Args:
            level: Granularity level for the checkpoint
        
        Returns:
            Checkpoint object representing the saved state
        """
        raise NotImplementedError("Subclasses must implement checkpoint()")
    
    # -------------------------------------------------------------------------
    # Concrete Methods (May be overridden by subclasses)
    # -------------------------------------------------------------------------
    
    async def initialize(self) -> None:
        """Initialize the agent before first use.
        
        Override this method to perform any setup required before
        the agent can execute tasks. Default implementation subscribes
        to the broadcast channel if configured.
        """
        if self._broadcast_channel is not None:
            await self._broadcast_channel.subscribe(
                self._agent_id,
                self._on_broadcast_received,
            )
    
    async def shutdown(self) -> None:
        """Clean up resources when agent is being terminated.
        
        Override this method to perform cleanup. Default implementation
        unsubscribes from the broadcast channel.
        """
        self._status = AgentStatus.TERMINATED
        if self._broadcast_channel is not None:
            await self._broadcast_channel.unsubscribe(self._agent_id)
    
    async def send_message(
        self,
        content: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Send a broadcast message to other agents.
        
        Args:
            content: The message content
            priority: Priority level for the message
            metadata: Optional metadata to attach to the message
        """
        if self._broadcast_channel is None:
            raise RuntimeError("No broadcast channel configured")
        
        message = Message(
            sender_id=self._agent_id,
            content=content,
            priority=priority,
            metadata=metadata or {},
        )
        await self._broadcast_channel.broadcast(message)
    
    async def save_checkpoint(
        self,
        level: CheckpointLevel = CheckpointLevel.STANDARD,
    ) -> str | None:
        """Create and save a checkpoint to the configured store.
        
        Args:
            level: Granularity level for the checkpoint
        
        Returns:
            Checkpoint ID if saved successfully, None otherwise
        """
        checkpoint = await self.checkpoint(level)
        
        if self._checkpoint_store is not None:
            return await self._checkpoint_store.save(checkpoint)
        
        return None
    
    async def load_checkpoint(self, checkpoint_id: str) -> bool:
        """Load and restore state from a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to load
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if self._checkpoint_store is None:
            return False
        
        checkpoint = await self._checkpoint_store.load(checkpoint_id)
        if checkpoint is None:
            return False
        
        return await self._restore_from_checkpoint(checkpoint)
    
    async def generate_with_llm(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate text using the configured LLM provider.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional parameters for the LLM
        
        Returns:
            Generated text response
        
        Raises:
            RuntimeError: If no LLM provider is configured
        """
        if self._llm_provider is None:
            raise RuntimeError("No LLM provider configured")
        
        return await self._llm_provider.generate(prompt, **kwargs)
    
    async def stream_with_llm(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream text generation using the configured LLM provider.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional parameters for the LLM
        
        Yields:
            Generated text tokens
        
        Raises:
            RuntimeError: If no LLM provider is configured
        """
        if self._llm_provider is None:
            raise RuntimeError("No LLM provider configured")
        
        async for token in self._llm_provider.generate_stream(prompt, **kwargs):
            yield token
    
    async def embed_with_llm(
        self,
        text: str,
        **kwargs: Any,
    ) -> list[float]:
        """Generate embeddings using the configured LLM provider.
        
        Args:
            text: Text to embed
            **kwargs: Additional parameters for the LLM
        
        Returns:
            Embedding vector
        
        Raises:
            RuntimeError: If no LLM provider is configured
        """
        if self._llm_provider is None:
            raise RuntimeError("No LLM provider configured")
        
        return await self._llm_provider.embed(text, **kwargs)
    
    def get_execution_history(self) -> list[ExecutionResult[TResult]]:
        """Return the history of execution results."""
        return self._execution_history.copy()
    
    def clear_execution_history(self) -> None:
        """Clear the execution history."""
        self._execution_history.clear()
    
    def register_message_handler(
        self,
        handler: Callable[[Message], Coroutine[Any, Any, None]],
    ) -> None:
        """Register an additional message handler.
        
        Args:
            handler: Async callable to handle incoming messages
        """
        self._message_handlers.append(handler)
    
    def unregister_message_handler(
        self,
        handler: Callable[[Message], Coroutine[Any, Any, None]],
    ) -> None:
        """Unregister a message handler.
        
        Args:
            handler: Handler to remove
        """
        if handler in self._message_handlers:
            self._message_handlers.remove(handler)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert agent state to dictionary representation.
        
        Returns:
            Dictionary containing agent metadata and state
        """
        return {
            "agent_id": str(self._agent_id),
            "status": self._status.name,
            "created_at": self._created_at.isoformat(),
            "last_activity": self._last_activity.isoformat(),
            "execution_count": self.execution_count,
            "has_llm_provider": self._llm_provider is not None,
            "has_checkpoint_store": self._checkpoint_store is not None,
            "has_broadcast_channel": self._broadcast_channel is not None,
        }
    
    # -------------------------------------------------------------------------
    # Protected Methods (For subclass use)
    # -------------------------------------------------------------------------
    
    def _update_status(self, status: AgentStatus) -> None:
        """Update the agent's operational status.
        
        Args:
            status: New status to set
        """
        self._status = status
        self._last_activity = datetime.utcnow()
    
    def _record_execution(self, result: ExecutionResult[TResult]) -> None:
        """Record an execution result in history.
        
        Args:
            result: Execution result to record
        """
        self._execution_history.append(result)
        self._last_activity = datetime.utcnow()
    
    async def _on_broadcast_received(self, message: Message) -> None:
        """Internal handler for broadcast messages.
        
        This method routes messages to the abstract receive_broadcast
        method and any registered handlers.
        
        Args:
            message: Received broadcast message
        """
        await self.receive_broadcast(message)
        
        for handler in self._message_handlers:
            await handler(message)
    
    @abstractmethod
    async def _restore_from_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """Restore agent state from a checkpoint.
        
        Subclasses must implement this to define how their state
        is restored from a checkpoint.
        
        Args:
            checkpoint: Checkpoint to restore from
        
        Returns:
            True if restoration was successful
        """
        raise NotImplementedError("Subclasses must implement _restore_from_checkpoint()")


# ============================================================================
# Exception Classes
# ============================================================================

class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass


class AgentExecutionError(AgentError):
    """Exception raised when agent execution fails."""
    
    def __init__(
        self,
        message: str,
        agent_id: AgentID | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.agent_id = agent_id
        self.cause = cause


class AgentConfigurationError(AgentError):
    """Exception raised when agent configuration is invalid."""
    pass


class CheckpointError(AgentError):
    """Exception raised when checkpoint operations fail."""
    pass


# ============================================================================
# Utility Functions
# ============================================================================

def create_execution_context(
    task_id: str | None = None,
    **parameters: Any,
) -> ExecutionContext:
    """Factory function to create an execution context.
    
    Args:
        task_id: Optional task identifier (auto-generated if not provided)
        **parameters: Additional context parameters
    
    Returns:
        New ExecutionContext instance
    """
    return ExecutionContext(
        task_id=task_id or str(uuid.uuid4()),
        parameters=parameters,
    )


def create_finding(
    agent_id: AgentID,
    finding_type: str,
    content: Any,
    confidence: float = 1.0,
    **metadata: Any,
) -> Finding:
    """Factory function to create a Finding.
    
    Args:
        agent_id: ID of the agent creating the finding
        finding_type: Type/category of the finding
        content: The finding content
        confidence: Confidence score (0.0 to 1.0)
        **metadata: Additional metadata
    
    Returns:
        New Finding instance
    """
    return Finding(
        agent_id=agent_id,
        finding_type=finding_type,
        content=content,
        confidence=confidence,
        metadata=metadata,
    )
