"""
CrossPollinationBus - Shared findings broadcast system for autoconstitution.

This module implements a pub/sub broadcast system that allows agents to share
validated improvements with all active agents in the swarm. It includes frequency
controls to prevent information flooding.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
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
    runtime_checkable,
)
from contextlib import asynccontextmanager
import logging

# Configure logging
logger = logging.getLogger(__name__)


class FindingPriority(Enum):
    """Priority levels for findings broadcast."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class FindingType(Enum):
    """Types of findings that can be broadcast."""
    HYPERPARAMETER_IMPROVEMENT = auto()
    ARCHITECTURE_CHANGE = auto()
    DATA_PREPROCESSING = auto()
    FEATURE_ENGINEERING = auto()
    OPTIMIZATION_TECHNIQUE = auto()
    VALIDATION_STRATEGY = auto()
    CUSTOM = auto()


@dataclass(frozen=True, slots=True)
class AgentId:
    """Unique identifier for an agent in the swarm."""
    id: str
    
    def __str__(self) -> str:
        return self.id


@dataclass(frozen=True, slots=True)
class Finding:
    """A validated finding/improvement to be broadcast."""
    agent_id: AgentId
    finding_type: FindingType
    priority: FindingPriority
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    finding_id: str = field(default_factory=lambda: f"finding_{time.time()}_{id(object())}")
    
    def __hash__(self) -> int:
        return hash(self.finding_id)


@dataclass(frozen=True, slots=True)
class BroadcastMessage:
    """Wrapper for a finding with broadcast metadata."""
    finding: Finding
    broadcast_timestamp: float
    sequence_number: int
    
    @property
    def latency(self) -> float:
        """Calculate latency from finding creation to broadcast."""
        return self.broadcast_timestamp - self.finding.timestamp


# Type variable for subscriber callbacks
T = TypeVar("T")

# Subscriber callback type
SubscriberCallback = Callable[[Finding], Coroutine[Any, Any, None]]
SyncSubscriberCallback = Callable[[Finding], None]


@runtime_checkable
class FrequencyController(Protocol):
    """Protocol for frequency control implementations."""
    
    async def should_allow_broadcast(self, finding: Finding) -> bool:
        """Determine if a finding should be allowed to broadcast."""
        ...
    
    async def record_broadcast(self, finding: Finding) -> None:
        """Record that a broadcast occurred."""
        ...
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get frequency control statistics."""
        ...


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for frequency control.
    
    Implements a token bucket algorithm where tokens are consumed
    for each broadcast and refilled at a constant rate.
    """
    
    def __init__(
        self,
        max_tokens: float = 10.0,
        refill_rate: float = 1.0,
        refill_period: float = 1.0,
    ) -> None:
        """
        Initialize the token bucket rate limiter.
        
        Args:
            max_tokens: Maximum number of tokens in the bucket
            refill_rate: Number of tokens to add per refill period
            refill_period: Time in seconds between refills
        """
        self._max_tokens = max_tokens
        self._refill_rate = refill_rate
        self._refill_period = refill_period
        self._tokens: Dict[AgentId, float] = defaultdict(lambda: max_tokens)
        self._last_refill: Dict[AgentId, float] = defaultdict(time.time)
        self._total_requests: Dict[AgentId, int] = defaultdict(int)
        self._allowed_requests: Dict[AgentId, int] = defaultdict(int)
        self._lock = asyncio.Lock()
    
    def _refill_tokens(self, agent_id: AgentId) -> None:
        """Refill tokens for an agent based on elapsed time."""
        # Avoid the defaultdict(time.time) race: its lazy-init calls time.time()
        # *after* `now = time.time()` above runs, producing negative elapsed that
        # drains an otherwise-full bucket. Touch _last_refill first so any
        # auto-init completes before we measure.
        if agent_id not in self._last_refill:
            self._last_refill[agent_id] = time.time()
        now = time.time()
        elapsed = max(0.0, now - self._last_refill[agent_id])
        tokens_to_add = (elapsed / self._refill_period) * self._refill_rate
        self._tokens[agent_id] = min(self._max_tokens, self._tokens[agent_id] + tokens_to_add)
        self._last_refill[agent_id] = now
    
    async def should_allow_broadcast(self, finding: Finding) -> bool:
        """Check if broadcast should be allowed based on token availability."""
        async with self._lock:
            self._refill_tokens(finding.agent_id)
            self._total_requests[finding.agent_id] += 1
            
            # Critical priority bypasses rate limiting
            if finding.priority == FindingPriority.CRITICAL:
                self._allowed_requests[finding.agent_id] += 1
                return True
            
            # High priority uses half the tokens
            token_cost = 0.5 if finding.priority == FindingPriority.HIGH else 1.0
            
            if self._tokens[finding.agent_id] >= token_cost:
                self._tokens[finding.agent_id] -= token_cost
                self._allowed_requests[finding.agent_id] += 1
                return True
            
            return False
    
    async def record_broadcast(self, finding: Finding) -> None:
        """Record broadcast (tokens already consumed in should_allow)."""
        pass  # Tokens are consumed in should_allow_broadcast
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        async with self._lock:
            total_req = sum(self._total_requests.values())
            allowed_req = sum(self._allowed_requests.values())
            return {
                "total_requests": total_req,
                "allowed_requests": allowed_req,
                "rejected_requests": total_req - allowed_req,
                "allow_rate": allowed_req / total_req if total_req > 0 else 0.0,
                "agent_tokens": {str(k): v for k, v in self._tokens.items()},
            }


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts limits based on system load.
    
    Reduces broadcast rates when system is under high load and
    increases them when the system is idle.
    """
    
    def __init__(
        self,
        base_rate: float = 10.0,
        min_rate: float = 1.0,
        max_rate: float = 50.0,
        load_check_interval: float = 5.0,
    ) -> None:
        """
        Initialize the adaptive rate limiter.
        
        Args:
            base_rate: Base broadcasts per second allowed
            min_rate: Minimum broadcasts per second
            max_rate: Maximum broadcasts per second
            load_check_interval: Seconds between load checks
        """
        self._base_rate = base_rate
        self._min_rate = min_rate
        self._max_rate = max_rate
        self._load_check_interval = load_check_interval
        self._current_rate = base_rate
        self._broadcast_times: List[float] = []
        self._last_load_check = time.time()
        self._lock = asyncio.Lock()
    
    async def _update_rate(self) -> None:
        """Update the current rate based on system load."""
        now = time.time()
        
        # Clean old broadcast times
        cutoff = now - self._load_check_interval
        self._broadcast_times = [t for t in self._broadcast_times if t > cutoff]
        
        # Calculate current load
        current_load = len(self._broadcast_times) / self._load_check_interval
        
        # Adjust rate based on load
        if current_load > self._current_rate * 0.8:
            # High load - reduce rate
            self._current_rate = max(self._min_rate, self._current_rate * 0.9)
        elif current_load < self._current_rate * 0.3:
            # Low load - increase rate
            self._current_rate = min(self._max_rate, self._current_rate * 1.1)
        
        self._last_load_check = now
    
    async def should_allow_broadcast(self, finding: Finding) -> bool:
        """Check if broadcast should be allowed based on adaptive rate."""
        async with self._lock:
            now = time.time()
            
            # Update rate periodically
            if now - self._last_load_check > self._load_check_interval:
                await self._update_rate()
            
            # Critical priority bypasses rate limiting
            if finding.priority == FindingPriority.CRITICAL:
                return True
            
            # Clean old broadcast times
            cutoff = now - 1.0  # 1 second window
            self._broadcast_times = [t for t in self._broadcast_times if t > cutoff]
            
            # Check if under rate limit
            priority_multiplier = {
                FindingPriority.LOW: 0.5,
                FindingPriority.MEDIUM: 1.0,
                FindingPriority.HIGH: 2.0,
                FindingPriority.CRITICAL: float("inf"),
            }
            
            effective_rate = self._current_rate * priority_multiplier.get(finding.priority, 1.0)
            
            if len(self._broadcast_times) < effective_rate:
                return True
            
            return False
    
    async def record_broadcast(self, finding: Finding) -> None:
        """Record that a broadcast occurred."""
        async with self._lock:
            self._broadcast_times.append(time.time())
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get adaptive rate limiter statistics."""
        async with self._lock:
            return {
                "current_rate": self._current_rate,
                "base_rate": self._base_rate,
                "min_rate": self._min_rate,
                "max_rate": self._max_rate,
                "recent_broadcasts": len(self._broadcast_times),
            }


class CompositeFrequencyController:
    """
    Combines multiple frequency controllers.
    
    All controllers must approve for a broadcast to be allowed.
    """
    
    def __init__(self, controllers: List[FrequencyController]) -> None:
        """
        Initialize with a list of frequency controllers.
        
        Args:
            controllers: List of frequency controllers to combine
        """
        self._controllers = controllers
    
    async def should_allow_broadcast(self, finding: Finding) -> bool:
        """Check if all controllers allow the broadcast."""
        results = await asyncio.gather(
            *[c.should_allow_broadcast(finding) for c in self._controllers]
        )
        return all(results)
    
    async def record_broadcast(self, finding: Finding) -> None:
        """Record broadcast in all controllers."""
        await asyncio.gather(
            *[c.record_broadcast(finding) for c in self._controllers]
        )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all controllers."""
        stats = await asyncio.gather(*[c.get_stats() for c in self._controllers])
        return {
            f"controller_{i}": stat
            for i, stat in enumerate(stats)
        }


class CrossPollinationBus:
    """
    Shared findings broadcast system for autoconstitution.
    
    Implements a pub/sub pattern where agents can publish validated
    improvements and subscribe to receive broadcasts from other agents.
    Includes frequency controls to prevent information flooding.
    
    Example:
        ```python
        # Create bus with token bucket rate limiting
        limiter = TokenBucketRateLimiter(max_tokens=10, refill_rate=2.0)
        bus = CrossPollinationBus(frequency_controller=limiter)
        
        # Subscribe to findings
        async def on_finding(finding: Finding) -> None:
            print(f"Received: {finding.finding_type}")
        
        await bus.subscribe(agent_id, on_finding)
        
        # Publish a finding
        finding = Finding(
            agent_id=agent_id,
            finding_type=FindingType.HYPERPARAMETER_IMPROVEMENT,
            priority=FindingPriority.HIGH,
            payload={"lr": 0.001, "accuracy": 0.95},
        )
        await bus.publish(finding)
        ```
    """
    
    def __init__(
        self,
        frequency_controller: Optional[FrequencyController] = None,
        max_queue_size: int = 1000,
        enable_metrics: bool = True,
    ) -> None:
        """
        Initialize the CrossPollinationBus.
        
        Args:
            frequency_controller: Controller for broadcast frequency limiting
            max_queue_size: Maximum size of internal message queue
            enable_metrics: Whether to collect metrics
        """
        self._frequency_controller = frequency_controller or TokenBucketRateLimiter()
        self._max_queue_size = max_queue_size
        self._enable_metrics = enable_metrics
        
        # Subscriber management
        self._subscribers: Dict[AgentId, List[SubscriberCallback]] = defaultdict(list)
        self._sync_subscribers: Dict[AgentId, List[SyncSubscriberCallback]] = defaultdict(list)
        self._global_subscribers: List[SubscriberCallback] = []
        self._global_sync_subscribers: List[SyncSubscriberCallback] = []
        
        # Message queue for async processing
        self._message_queue: asyncio.Queue[Finding] = asyncio.Queue(maxsize=max_queue_size)
        
        # State management
        self._sequence_number = 0
        self._running = False
        self._processor_task: Optional[asyncio.Task[None]] = None
        self._lock = asyncio.Lock()
        
        # Metrics
        self._metrics: Dict[str, Any] = {
            "total_published": 0,
            "total_broadcast": 0,
            "total_dropped": 0,
            "total_rejected": 0,
            "findings_by_type": defaultdict(int),
            "findings_by_priority": defaultdict(int),
            "broadcast_latency": [],
        }
    
    async def start(self) -> None:
        """Start the broadcast processor."""
        async with self._lock:
            if self._running:
                return
            
            self._running = True
            self._processor_task = asyncio.create_task(self._process_messages())
            logger.info("CrossPollinationBus started")
    
    async def stop(self) -> None:
        """Stop the broadcast processor."""
        async with self._lock:
            if not self._running:
                return
            
            self._running = False
            
            if self._processor_task:
                self._processor_task.cancel()
                try:
                    await self._processor_task
                except asyncio.CancelledError:
                    pass
                self._processor_task = None
            
            logger.info("CrossPollinationBus stopped")
    
    @asynccontextmanager
    async def run(self) -> AsyncIterator[CrossPollinationBus]:
        """Context manager for running the bus."""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()
    
    async def publish(self, finding: Finding) -> bool:
        """
        Publish a finding to the broadcast bus.
        
        Args:
            finding: The finding to publish
            
        Returns:
            True if the finding was accepted for broadcast, False otherwise
        """
        if not self._running:
            raise RuntimeError("Bus is not running. Call start() first.")
        
        # Update metrics
        if self._enable_metrics:
            self._metrics["total_published"] += 1
            self._metrics["findings_by_type"][finding.finding_type.name] += 1
            self._metrics["findings_by_priority"][finding.priority.name] += 1
        
        # Check frequency control
        allowed = await self._frequency_controller.should_allow_broadcast(finding)
        
        if not allowed:
            if self._enable_metrics:
                self._metrics["total_rejected"] += 1
            logger.debug(f"Finding rejected by frequency controller: {finding.finding_id}")
            return False
        
        # Record the broadcast
        await self._frequency_controller.record_broadcast(finding)
        
        # Add to message queue
        try:
            self._message_queue.put_nowait(finding)
            return True
        except asyncio.QueueFull:
            if self._enable_metrics:
                self._metrics["total_dropped"] += 1
            logger.warning(f"Message queue full, dropping finding: {finding.finding_id}")
            return False
    
    async def _process_messages(self) -> None:
        """Process messages from the queue and broadcast to subscribers."""
        while self._running:
            try:
                # Wait for a message with timeout
                finding = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
                
                await self._broadcast(finding)
                self._message_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _broadcast(self, finding: Finding) -> None:
        """Broadcast a finding to all subscribers."""
        broadcast_time = time.time()
        
        async with self._lock:
            self._sequence_number += 1
            message = BroadcastMessage(
                finding=finding,
                broadcast_timestamp=broadcast_time,
                sequence_number=self._sequence_number,
            )
            
            if self._enable_metrics:
                self._metrics["total_broadcast"] += 1
                self._metrics["broadcast_latency"].append(message.latency)
                # Keep only last 1000 latency measurements
                if len(self._metrics["broadcast_latency"]) > 1000:
                    self._metrics["broadcast_latency"] = self._metrics["broadcast_latency"][-1000:]
        
        # Collect all async callbacks
        callbacks: List[Coroutine[Any, Any, None]] = []
        
        # Global async subscribers
        for callback in self._global_subscribers:
            callbacks.append(self._safe_callback(callback, finding))
        
        # Agent-specific async subscribers
        for agent_id, agent_callbacks in self._subscribers.items():
            # Don't send back to the originating agent unless it's a self-notify
            if agent_id == finding.agent_id:
                continue
            for callback in agent_callbacks:
                callbacks.append(self._safe_callback(callback, finding))
        
        # Execute all async callbacks concurrently
        if callbacks:
            await asyncio.gather(*callbacks, return_exceptions=True)
        
        # Execute sync callbacks
        for callback in self._global_sync_subscribers:
            self._safe_sync_callback(callback, finding)
        
        for agent_id, agent_callbacks in self._sync_subscribers.items():
            if agent_id == finding.agent_id:
                continue
            for callback in agent_callbacks:
                self._safe_sync_callback(callback, finding)
    
    async def _safe_callback(
        self,
        callback: SubscriberCallback,
        finding: Finding,
    ) -> None:
        """Execute a callback safely, catching exceptions."""
        try:
            await callback(finding)
        except Exception as e:
            logger.error(f"Error in subscriber callback: {e}")
    
    def _safe_sync_callback(
        self,
        callback: SyncSubscriberCallback,
        finding: Finding,
    ) -> None:
        """Execute a sync callback safely, catching exceptions."""
        try:
            callback(finding)
        except Exception as e:
            logger.error(f"Error in sync subscriber callback: {e}")
    
    async def subscribe(
        self,
        agent_id: AgentId,
        callback: SubscriberCallback,
    ) -> None:
        """
        Subscribe an agent to receive broadcasts.
        
        Args:
            agent_id: The agent's unique identifier
            callback: Async callback function to receive findings
        """
        async with self._lock:
            self._subscribers[agent_id].append(callback)
            logger.debug(f"Agent {agent_id} subscribed (async)")
    
    async def subscribe_sync(
        self,
        agent_id: AgentId,
        callback: SyncSubscriberCallback,
    ) -> None:
        """
        Subscribe an agent with a synchronous callback.
        
        Args:
            agent_id: The agent's unique identifier
            callback: Sync callback function to receive findings
        """
        async with self._lock:
            self._sync_subscribers[agent_id].append(callback)
            logger.debug(f"Agent {agent_id} subscribed (sync)")
    
    async def unsubscribe(
        self,
        agent_id: AgentId,
        callback: Optional[SubscriberCallback] = None,
    ) -> None:
        """
        Unsubscribe an agent from broadcasts.
        
        Args:
            agent_id: The agent's unique identifier
            callback: Specific callback to remove, or None to remove all
        """
        async with self._lock:
            if callback is None:
                self._subscribers.pop(agent_id, None)
                self._sync_subscribers.pop(agent_id, None)
            else:
                if agent_id in self._subscribers:
                    self._subscribers[agent_id] = [
                        cb for cb in self._subscribers[agent_id] if cb is not callback
                    ]
            logger.debug(f"Agent {agent_id} unsubscribed")
    
    async def subscribe_global(self, callback: SubscriberCallback) -> None:
        """
        Subscribe to all broadcasts globally.
        
        Args:
            callback: Async callback function to receive all findings
        """
        async with self._lock:
            self._global_subscribers.append(callback)
    
    async def subscribe_global_sync(self, callback: SyncSubscriberCallback) -> None:
        """
        Subscribe to all broadcasts globally with a sync callback.
        
        Args:
            callback: Sync callback function to receive all findings
        """
        async with self._lock:
            self._global_sync_subscribers.append(callback)
    
    async def unsubscribe_global(self, callback: SubscriberCallback) -> None:
        """
        Unsubscribe a global subscriber.
        
        Args:
            callback: The callback to remove
        """
        async with self._lock:
            self._global_subscribers = [
                cb for cb in self._global_subscribers if cb is not callback
            ]
    
    async def get_subscriber_count(self) -> int:
        """Get the total number of subscribers."""
        async with self._lock:
            return (
                sum(len(cbs) for cbs in self._subscribers.values()) +
                sum(len(cbs) for cbs in self._sync_subscribers.values()) +
                len(self._global_subscribers) +
                len(self._global_sync_subscribers)
            )
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        async with self._lock:
            metrics = dict(self._metrics)
            
            # Calculate average latency
            latencies = metrics.get("broadcast_latency", [])
            if latencies:
                metrics["avg_latency"] = sum(latencies) / len(latencies)
                metrics["max_latency"] = max(latencies)
                metrics["min_latency"] = min(latencies)
            else:
                metrics["avg_latency"] = 0.0
                metrics["max_latency"] = 0.0
                metrics["min_latency"] = 0.0
            
            # Get frequency controller stats
            freq_stats = await self._frequency_controller.get_stats()
            metrics["frequency_controller"] = freq_stats
            
            return metrics
    
    async def clear_metrics(self) -> None:
        """Clear all metrics."""
        async with self._lock:
            self._metrics = {
                "total_published": 0,
                "total_broadcast": 0,
                "total_dropped": 0,
                "total_rejected": 0,
                "findings_by_type": defaultdict(int),
                "findings_by_priority": defaultdict(int),
                "broadcast_latency": [],
            }


class AgentPollinationClient:
    """
    Client for agents to interact with the CrossPollinationBus.
    
    Provides a simplified interface for agents to publish findings
    and receive broadcasts from other agents.
    """
    
    def __init__(
        self,
        agent_id: AgentId,
        bus: CrossPollinationBus,
    ) -> None:
        """
        Initialize the client.
        
        Args:
            agent_id: The agent's unique identifier
            bus: The CrossPollinationBus instance
        """
        self._agent_id = agent_id
        self._bus = bus
        self._received_findings: asyncio.Queue[Finding] = asyncio.Queue()
        self._subscribed = False
    
    async def subscribe(self) -> None:
        """Subscribe this agent to receive broadcasts."""
        if self._subscribed:
            return
        
        async def on_finding(finding: Finding) -> None:
            await self._received_findings.put(finding)
        
        await self._bus.subscribe(self._agent_id, on_finding)
        self._subscribed = True
    
    async def unsubscribe(self) -> None:
        """Unsubscribe this agent from broadcasts."""
        await self._bus.unsubscribe(self._agent_id)
        self._subscribed = False
    
    async def publish_improvement(
        self,
        finding_type: FindingType,
        priority: FindingPriority,
        payload: Dict[str, Any],
    ) -> bool:
        """
        Publish an improvement finding.
        
        Args:
            finding_type: Type of the finding
            priority: Priority level
            payload: Finding data
            
        Returns:
            True if published successfully
        """
        finding = Finding(
            agent_id=self._agent_id,
            finding_type=finding_type,
            priority=priority,
            payload=payload,
        )
        return await self._bus.publish(finding)
    
    async def get_next_finding(self, timeout: Optional[float] = None) -> Optional[Finding]:
        """
        Get the next received finding.
        
        Args:
            timeout: Maximum time to wait, or None for indefinite
            
        Returns:
            The next finding, or None if timeout
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(
                    self._received_findings.get(),
                    timeout=timeout
                )
            else:
                return await self._received_findings.get()
        except asyncio.TimeoutError:
            return None
    
    async def get_findings_iter(self) -> AsyncIterator[Finding]:
        """Get an async iterator over received findings."""
        while True:
            finding = await self._received_findings.get()
            yield finding
    
    @property
    def agent_id(self) -> AgentId:
        """Get the agent ID."""
        return self._agent_id
    
    @property
    def pending_findings(self) -> int:
        """Get the number of pending findings."""
        return self._received_findings.qsize()


# Convenience factory functions

def create_default_bus(
    max_tokens: float = 10.0,
    refill_rate: float = 2.0,
    max_queue_size: int = 1000,
) -> CrossPollinationBus:
    """
    Create a CrossPollinationBus with default token bucket rate limiting.
    
    Args:
        max_tokens: Maximum tokens in the bucket
        refill_rate: Tokens added per second
        max_queue_size: Maximum message queue size
        
    Returns:
        Configured CrossPollinationBus instance
    """
    limiter = TokenBucketRateLimiter(
        max_tokens=max_tokens,
        refill_rate=refill_rate,
    )
    return CrossPollinationBus(
        frequency_controller=limiter,
        max_queue_size=max_queue_size,
    )


def create_adaptive_bus(
    base_rate: float = 10.0,
    min_rate: float = 1.0,
    max_rate: float = 50.0,
    max_queue_size: int = 1000,
) -> CrossPollinationBus:
    """
    Create a CrossPollinationBus with adaptive rate limiting.
    
    Args:
        base_rate: Base broadcasts per second
        min_rate: Minimum broadcasts per second
        max_rate: Maximum broadcasts per second
        max_queue_size: Maximum message queue size
        
    Returns:
        Configured CrossPollinationBus instance
    """
    limiter = AdaptiveRateLimiter(
        base_rate=base_rate,
        min_rate=min_rate,
        max_rate=max_rate,
    )
    return CrossPollinationBus(
        frequency_controller=limiter,
        max_queue_size=max_queue_size,
    )


def create_composite_bus(
    max_queue_size: int = 1000,
) -> CrossPollinationBus:
    """
    Create a CrossPollinationBus with composite frequency control.
    
    Combines token bucket and adaptive rate limiting.
    
    Args:
        max_queue_size: Maximum message queue size
        
    Returns:
        Configured CrossPollinationBus instance
    """
    token_bucket = TokenBucketRateLimiter(max_tokens=10.0, refill_rate=2.0)
    adaptive = AdaptiveRateLimiter(base_rate=10.0, min_rate=1.0, max_rate=50.0)
    composite = CompositeFrequencyController([token_bucket, adaptive])
    
    return CrossPollinationBus(
        frequency_controller=composite,
        max_queue_size=max_queue_size,
    )
