"""
SwarmOrchestrator - Core orchestrator for autoconstitution.

Implements PARL (Parallel Autonomous Research Layer) principles for autoresearch.
Master controller that manages sub-agents, task DAGs, and dynamic reallocation.

Python 3.11+ async-first implementation with comprehensive type hints.
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from functools import wraps
from heapq import heappush, heappop
from typing import (
    Any,
    Callable,
    Coroutine,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)
from typing_extensions import Self, Unpack

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SwarmOrchestrator")


# ============================================================================
# Type Definitions
# ============================================================================

T = TypeVar("T")
TaskResult = TypeVar("TaskResult")
AgentID = str
TaskID = str
BranchID = str


class TaskStatus(enum.Enum):
    """Status of a task in the DAG."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class AgentStatus(enum.Enum):
    """Status of a sub-agent."""
    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    TERMINATED = "terminated"
    ERROR = "error"


class BranchPriority(enum.IntEnum):
    """Priority levels for research branches."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True, slots=True)
class TaskDependency:
    """Represents a dependency between tasks."""
    task_id: TaskID
    optional: bool = False
    timeout: Optional[float] = None


@dataclass(slots=True)
class TaskMetrics:
    """Performance metrics for a single task execution."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    cpu_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    retry_count: int = 0
    error_count: int = 0
    
    @property
    def duration_ms(self) -> float:
        """Calculate task duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_ms": self.duration_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "memory_peak_mb": self.memory_peak_mb,
            "io_read_bytes": self.io_read_bytes,
            "io_write_bytes": self.io_write_bytes,
            "retry_count": self.retry_count,
            "error_count": self.error_count,
        }


@dataclass(slots=True)
class BranchMetrics:
    """Aggregated performance metrics for a research branch."""
    branch_id: BranchID
    created_at: datetime = field(default_factory=datetime.now)
    task_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    total_duration_ms: float = 0.0
    avg_task_duration_ms: float = 0.0
    success_rate: float = 0.0
    throughput_tasks_per_sec: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update(self, task_metrics: TaskMetrics, success: bool) -> None:
        """Update branch metrics with a new task result."""
        self.task_count += 1
        self.total_duration_ms += task_metrics.duration_ms
        self.avg_task_duration_ms = self.total_duration_ms / self.task_count
        
        if success:
            self.completed_count += 1
        else:
            self.failed_count += 1
        
        self.success_rate = self.completed_count / self.task_count
        elapsed_sec = (datetime.now() - self.created_at).total_seconds()
        if elapsed_sec > 0:
            self.throughput_tasks_per_sec = self.task_count / elapsed_sec
        self.last_updated = datetime.now()
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "task_count": self.task_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "success_rate": self.success_rate,
            "avg_task_duration_ms": self.avg_task_duration_ms,
            "throughput_tasks_per_sec": self.throughput_tasks_per_sec,
        }


@dataclass(slots=True)
class AgentMetrics:
    """Performance metrics for a sub-agent."""
    agent_id: AgentID
    branch_id: BranchID
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    current_load: float = 0.0  # 0.0 to 1.0
    efficiency_score: float = 1.0  # Derived metric
    last_heartbeat: datetime = field(default_factory=datetime.now)
    
    def update_execution(self, duration_ms: float, success: bool) -> None:
        """Update agent metrics after task execution."""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        
        self.total_execution_time_ms += duration_ms
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks > 0:
            self.avg_execution_time_ms = self.total_execution_time_ms / total_tasks
            success_rate = self.tasks_completed / total_tasks
            # Efficiency = success_rate / avg_time (normalized)
            self.efficiency_score = success_rate / (1 + self.avg_execution_time_ms / 1000)
    
    def heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        self.last_heartbeat = datetime.now()
    
    def is_alive(self, timeout_sec: float = 30.0) -> bool:
        """Check if agent is still alive based on heartbeat."""
        elapsed = (datetime.now() - self.last_heartbeat).total_seconds()
        return elapsed < timeout_sec


@dataclass(slots=True)
class TaskNode:
    """A node in the task DAG representing a unit of work."""
    task_id: TaskID
    branch_id: BranchID
    name: str
    coro: Callable[..., Coroutine[Any, Any, Any]]
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    dependencies: set[TaskDependency] = field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    retry_limit: int = 3
    timeout_sec: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[Exception] = None
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    
    def __hash__(self) -> int:
        return hash(self.task_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaskNode):
            return NotImplemented
        return self.task_id == other.task_id
    
    @property
    def is_ready(self) -> bool:
        """Check if task is ready to execute (all dependencies met)."""
        return self.status == TaskStatus.PENDING
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "branch_id": self.branch_id,
            "name": self.name,
            "status": self.status.value,
            "priority": self.priority,
            "retry_limit": self.retry_limit,
            "created_at": self.created_at.isoformat(),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(slots=True)
class ResearchBranch:
    """A research branch containing related tasks and agents."""
    branch_id: BranchID
    name: str
    description: str = ""
    priority: BranchPriority = BranchPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    parent_branch: Optional[BranchID] = None
    child_branches: set[BranchID] = field(default_factory=set)
    task_ids: set[TaskID] = field(default_factory=set)
    agent_ids: set[AgentID] = field(default_factory=set)
    metrics: BranchMetrics = field(init=False)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        self.metrics = BranchMetrics(branch_id=self.branch_id)
    
    def add_task(self, task_id: TaskID) -> None:
        """Add a task to this branch."""
        self.task_ids.add(task_id)
    
    def add_agent(self, agent_id: AgentID) -> None:
        """Add an agent to this branch."""
        self.agent_ids.add(agent_id)
    
    def remove_agent(self, agent_id: AgentID) -> None:
        """Remove an agent from this branch."""
        self.agent_ids.discard(agent_id)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.name,
            "task_count": len(self.task_ids),
            "agent_count": len(self.agent_ids),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(slots=True)
class SubAgent:
    """A sub-agent managed by the orchestrator."""
    agent_id: AgentID
    branch_id: BranchID
    name: str
    capabilities: set[str] = field(default_factory=set)
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[TaskID] = None
    created_at: datetime = field(default_factory=datetime.now)
    metrics: AgentMetrics = field(init=False)
    _task_queue: asyncio.Queue[TaskNode] = field(default_factory=asyncio.Queue)
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    
    def __post_init__(self) -> None:
        self.metrics = AgentMetrics(
            agent_id=self.agent_id,
            branch_id=self.branch_id,
        )
    
    async def assign_task(self, task: TaskNode) -> None:
        """Assign a task to this agent."""
        await self._task_queue.put(task)
        self.status = AgentStatus.BUSY
    
    async def get_next_task(self) -> Optional[TaskNode]:
        """Get the next task from the queue (non-blocking)."""
        try:
            return self._task_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
    
    def has_capacity(self, max_queue_size: int = 1) -> bool:
        """Check if agent can accept more tasks."""
        return self._task_queue.qsize() < max_queue_size
    
    def cancel(self) -> None:
        """Signal agent to cancel current operations."""
        self._cancel_event.set()
    
    def is_cancelled(self) -> bool:
        """Check if agent has been cancelled."""
        return self._cancel_event.is_set()
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "branch_id": self.branch_id,
            "name": self.name,
            "status": self.status.value,
            "capabilities": list(self.capabilities),
            "current_task": self.current_task,
            "metrics": {
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "efficiency_score": self.metrics.efficiency_score,
            },
        }


# ============================================================================
# Exceptions
# ============================================================================

class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    pass


class TaskDAGError(OrchestratorError):
    """Error in task DAG operations."""
    pass


class CircularDependencyError(TaskDAGError):
    """Circular dependency detected in task graph."""
    pass


class AgentError(OrchestratorError):
    """Error in agent operations."""
    pass


class BranchError(OrchestratorError):
    """Error in branch operations."""
    pass


class ReallocationError(OrchestratorError):
    """Error during agent reallocation."""
    pass


# ============================================================================
# Task DAG Implementation
# ============================================================================

class TaskDAG:
    """
    Directed Acyclic Graph for managing task dependencies.
    
    Supports:
    - Topological ordering for execution
    - Cycle detection
    - Dynamic task addition/removal
    - Priority-based scheduling
    """
    
    def __init__(self) -> None:
        self._tasks: dict[TaskID, TaskNode] = {}
        self._dependencies: dict[TaskID, set[TaskID]] = defaultdict(set)
        self._dependents: dict[TaskID, set[TaskID]] = defaultdict(set)
        self._lock = asyncio.Lock()
    
    async def add_task(self, task: TaskNode) -> None:
        """Add a task to the DAG."""
        async with self._lock:
            if task.task_id in self._tasks:
                raise TaskDAGError(f"Task {task.task_id} already exists")
            
            self._tasks[task.task_id] = task
            
            # Register dependencies
            for dep in task.dependencies:
                self._dependencies[task.task_id].add(dep.task_id)
                self._dependents[dep.task_id].add(task.task_id)
            
            # Check for cycles
            if self._has_cycle():
                # Rollback
                del self._tasks[task.task_id]
                for dep in task.dependencies:
                    self._dependencies[task.task_id].discard(dep.task_id)
                    self._dependents[dep.task_id].discard(task.task_id)
                raise CircularDependencyError(
                    f"Adding task {task.task_id} would create a cycle"
                )
    
    async def remove_task(self, task_id: TaskID) -> Optional[TaskNode]:
        """Remove a task from the DAG."""
        async with self._lock:
            task = self._tasks.pop(task_id, None)
            if task:
                # Clean up dependency mappings
                for dep_task_id in self._dependencies[task_id]:
                    self._dependents[dep_task_id].discard(task_id)
                for dep_task_id in self._dependents[task_id]:
                    self._dependencies[dep_task_id].discard(task_id)
                del self._dependencies[task_id]
                del self._dependents[task_id]
            return task
    
    async def get_task(self, task_id: TaskID) -> Optional[TaskNode]:
        """Get a task by ID."""
        async with self._lock:
            return self._tasks.get(task_id)
    
    async def get_ready_tasks(self, branch_id: Optional[BranchID] = None) -> list[TaskNode]:
        """Get all tasks that are ready to execute (dependencies satisfied)."""
        async with self._lock:
            ready = []
            for task_id, task in self._tasks.items():
                if task.status != TaskStatus.PENDING:
                    continue
                if branch_id and task.branch_id != branch_id:
                    continue
                
                # Check if all dependencies are completed
                deps_satisfied = all(
                    self._tasks.get(dep.task_id, TaskNode(
                        task_id="", branch_id="", name="", coro=lambda: None
                    )).status == TaskStatus.COMPLETED
                    for dep in task.dependencies
                )
                
                if deps_satisfied:
                    ready.append(task)
            
            # Sort by priority (lower = higher priority)
            ready.sort(key=lambda t: t.priority)
            return ready
    
    async def update_task_status(
        self,
        task_id: TaskID,
        status: TaskStatus,
        result: Any = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Update the status of a task."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = status
                if status == TaskStatus.RUNNING:
                    task.metrics.start_time = datetime.now()
                elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    task.metrics.end_time = datetime.now()
                    task.completed_at = datetime.now()
                task.result = result
                task.error = error
    
    async def get_dependents(self, task_id: TaskID) -> set[TaskID]:
        """Get all tasks that depend on the given task."""
        async with self._lock:
            return self._dependents.get(task_id, set()).copy()
    
    async def get_dependencies(self, task_id: TaskID) -> set[TaskID]:
        """Get all dependencies of the given task."""
        async with self._lock:
            return self._dependencies.get(task_id, set()).copy()
    
    def _has_cycle(self) -> bool:
        """Detect cycles using DFS (internal, no lock)."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {task_id: WHITE for task_id in self._tasks}
        
        def dfs(task_id: TaskID) -> bool:
            color[task_id] = GRAY
            for neighbor in self._dependencies.get(task_id, set()):
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    return True  # Back edge found
                if color[neighbor] == WHITE and dfs(neighbor):
                    return True
            color[task_id] = BLACK
            return False
        
        for task_id in self._tasks:
            if color[task_id] == WHITE:
                if dfs(task_id):
                    return True
        return False
    
    async def topological_sort(self) -> list[TaskID]:
        """Return tasks in topological order."""
        async with self._lock:
            in_degree = {task_id: len(deps) for task_id, deps in self._dependencies.items()}
            queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
            result = []
            
            while queue:
                # Sort by priority
                queue.sort(key=lambda tid: self._tasks[tid].priority)
                task_id = queue.pop(0)
                result.append(task_id)
                
                for dependent in self._dependents.get(task_id, set()):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
            
            if len(result) != len(self._tasks):
                raise CircularDependencyError("Cycle detected in task graph")
            
            return result
    
    async def get_branch_tasks(self, branch_id: BranchID) -> list[TaskNode]:
        """Get all tasks belonging to a branch."""
        async with self._lock:
            return [
                task for task in self._tasks.values()
                if task.branch_id == branch_id
            ]
    
    async def get_stats(self) -> dict[str, Any]:
        """Get DAG statistics."""
        async with self._lock:
            status_counts = defaultdict(int)
            for task in self._tasks.values():
                status_counts[task.status.value] += 1
            
            return {
                "total_tasks": len(self._tasks),
                "status_breakdown": dict(status_counts),
                "total_dependencies": sum(len(deps) for deps in self._dependencies.values()),
            }


# ============================================================================
# Performance Monitor
# ============================================================================

class PerformanceMonitor:
    """
    Monitors and analyzes branch and agent performance metrics.
    
    Provides:
    - Real-time metric collection
    - Trend analysis
    - Bottleneck detection
    - Performance alerts
    """
    
    def __init__(self, history_window: int = 1000) -> None:
        self._branch_metrics: dict[BranchID, BranchMetrics] = {}
        self._agent_metrics: dict[AgentID, AgentMetrics] = {}
        self._task_history: list[tuple[datetime, TaskID, TaskMetrics, bool]] = []
        self._history_window = history_window
        self._lock = asyncio.Lock()
        self._alert_handlers: list[Callable[[str, dict[str, Any]], None]] = []
    
    def add_alert_handler(
        self,
        handler: Callable[[str, dict[str, Any]], None]
    ) -> None:
        """Add an alert handler callback."""
        self._alert_handlers.append(handler)
    
    async def record_task_completion(
        self,
        task: TaskNode,
        success: bool,
    ) -> None:
        """Record task completion metrics."""
        async with self._lock:
            # Update branch metrics
            branch_metrics = self._branch_metrics.get(task.branch_id)
            if branch_metrics:
                branch_metrics.update(task.metrics, success)
            
            # Update agent metrics
            # (Agent ID would be passed in or stored in task)
            
            # Add to history
            self._task_history.append((
                datetime.now(),
                task.task_id,
                task.metrics,
                success,
            ))
            
            # Trim history
            if len(self._task_history) > self._history_window:
                self._task_history = self._task_history[-self._history_window:]
            
            # Check for alerts
            await self._check_alerts(task, success)
    
    async def register_branch(self, branch: ResearchBranch) -> None:
        """Register a branch for monitoring."""
        async with self._lock:
            self._branch_metrics[branch.branch_id] = branch.metrics
    
    async def register_agent(self, agent: SubAgent) -> None:
        """Register an agent for monitoring."""
        async with self._lock:
            self._agent_metrics[agent.agent_id] = agent.metrics
    
    async def update_agent_metrics(self, agent_id: AgentID, duration_ms: float, success: bool) -> None:
        """Update agent metrics after task execution."""
        async with self._lock:
            metrics = self._agent_metrics.get(agent_id)
            if metrics:
                metrics.update_execution(duration_ms, success)
                metrics.heartbeat()
    
    async def get_branch_performance(self, branch_id: BranchID) -> Optional[BranchMetrics]:
        """Get performance metrics for a branch."""
        async with self._lock:
            return self._branch_metrics.get(branch_id)
    
    async def get_agent_performance(self, agent_id: AgentID) -> Optional[AgentMetrics]:
        """Get performance metrics for an agent."""
        async with self._lock:
            return self._agent_metrics.get(agent_id)
    
    async def get_all_branch_metrics(self) -> dict[BranchID, BranchMetrics]:
        """Get all branch metrics."""
        async with self._lock:
            return self._branch_metrics.copy()
    
    async def get_all_agent_metrics(self) -> dict[AgentID, AgentMetrics]:
        """Get all agent metrics."""
        async with self._lock:
            return self._agent_metrics.copy()
    
    async def identify_bottlenecks(self) -> list[dict[str, Any]]:
        """Identify performance bottlenecks."""
        async with self._lock:
            bottlenecks = []
            
            # Find branches with low success rates
            for branch_id, metrics in self._branch_metrics.items():
                if metrics.success_rate < 0.5 and metrics.task_count > 5:
                    bottlenecks.append({
                        "type": "low_success_rate",
                        "branch_id": branch_id,
                        "success_rate": metrics.success_rate,
                        "severity": "high" if metrics.success_rate < 0.3 else "medium",
                    })
            
            # Find agents with low efficiency
            for agent_id, metrics in self._agent_metrics.items():
                if metrics.efficiency_score < 0.3 and metrics.tasks_completed > 5:
                    bottlenecks.append({
                        "type": "low_agent_efficiency",
                        "agent_id": agent_id,
                        "efficiency": metrics.efficiency_score,
                        "severity": "medium",
                    })
            
            return bottlenecks
    
    async def _check_alerts(self, task: TaskNode, success: bool) -> None:
        """Check for alert conditions."""
        alerts = []
        
        # Check for repeated failures
        if not success:
            recent_failures = sum(
                1 for _, tid, _, s in self._task_history[-10:]
                if not s
            )
            if recent_failures >= 5:
                alerts.append({
                    "type": "repeated_failures",
                    "message": f"{recent_failures} recent failures detected",
                    "count": recent_failures,
                })
        
        # Check for slow tasks
        if task.metrics.duration_ms > 60000:  # 1 minute
            alerts.append({
                "type": "slow_task",
                "message": f"Task {task.task_id} took {task.metrics.duration_ms}ms",
                "duration_ms": task.metrics.duration_ms,
            })
        
        # Trigger alert handlers
        for alert in alerts:
            for handler in self._alert_handlers:
                try:
                    handler(alert["type"], alert)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")
    
    async def get_health_report(self) -> dict[str, Any]:
        """Generate overall health report."""
        async with self._lock:
            total_tasks = len(self._task_history)
            successful_tasks = sum(1 for _, _, _, s in self._task_history if s)
            
            return {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "overall_success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
                "active_branches": len(self._branch_metrics),
                "active_agents": len(self._agent_metrics),
                "bottlenecks": await self.identify_bottlenecks(),
            }


# ============================================================================
# Agent Pool Manager
# ============================================================================

class AgentPoolManager:
    """
    Manages the pool of sub-agents with dynamic reallocation capabilities.
    
    Features:
    - Dynamic agent spawning/termination
    - Load balancing across branches
    - Capability-based task assignment
    - Resource optimization
    """
    
    def __init__(
        self,
        min_agents: int = 2,
        max_agents: int = 100,
        spawn_threshold: float = 0.8,
        idle_timeout_sec: float = 300.0,
    ) -> None:
        self._agents: dict[AgentID, SubAgent] = {}
        self._branch_agents: dict[BranchID, set[AgentID]] = defaultdict(set)
        self._capability_index: dict[str, set[AgentID]] = defaultdict(set)
        self._min_agents = min_agents
        self._max_agents = max_agents
        self._spawn_threshold = spawn_threshold
        self._idle_timeout_sec = idle_timeout_sec
        self._lock = asyncio.Lock()
        self._reallocation_history: list[dict[str, Any]] = []
    
    async def spawn_agent(
        self,
        branch_id: BranchID,
        name: Optional[str] = None,
        capabilities: Optional[set[str]] = None,
    ) -> SubAgent:
        """Spawn a new sub-agent."""
        async with self._lock:
            if len(self._agents) >= self._max_agents:
                raise AgentError(f"Maximum agent limit ({self._max_agents}) reached")
            
            agent_id = f"agent_{uuid.uuid4().hex[:8]}"
            agent = SubAgent(
                agent_id=agent_id,
                branch_id=branch_id,
                name=name or f"Agent-{agent_id[-4:]}",
                capabilities=capabilities or set(),
            )
            
            self._agents[agent_id] = agent
            self._branch_agents[branch_id].add(agent_id)
            
            for cap in agent.capabilities:
                self._capability_index[cap].add(agent_id)
            
            logger.info(f"Spawned agent {agent_id} for branch {branch_id}")
            return agent
    
    async def terminate_agent(self, agent_id: AgentID, force: bool = False) -> bool:
        """Terminate a sub-agent."""
        async with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return False
            
            if agent.status == AgentStatus.BUSY and not force:
                logger.warning(f"Agent {agent_id} is busy, cannot terminate without force")
                return False
            
            # Cancel agent operations
            agent.cancel()
            agent.status = AgentStatus.TERMINATED
            
            # Remove from indices
            del self._agents[agent_id]
            self._branch_agents[agent.branch_id].discard(agent_id)
            
            for cap in agent.capabilities:
                self._capability_index[cap].discard(agent_id)
            
            logger.info(f"Terminated agent {agent_id}")
            return True
    
    async def get_agent(self, agent_id: AgentID) -> Optional[SubAgent]:
        """Get an agent by ID."""
        async with self._lock:
            return self._agents.get(agent_id)
    
    async def get_branch_agents(self, branch_id: BranchID) -> list[SubAgent]:
        """Get all agents assigned to a branch."""
        async with self._lock:
            return [
                self._agents[aid]
                for aid in self._branch_agents.get(branch_id, set())
                if aid in self._agents
            ]
    
    async def find_best_agent(
        self,
        branch_id: BranchID,
        required_capabilities: Optional[set[str]] = None,
    ) -> Optional[SubAgent]:
        """Find the best available agent for a task."""
        async with self._lock:
            candidates = []
            
            for agent_id in self._branch_agents.get(branch_id, set()):
                agent = self._agents.get(agent_id)
                if not agent or agent.status != AgentStatus.IDLE:
                    continue
                
                # Check capabilities
                if required_capabilities:
                    if not required_capabilities.issubset(agent.capabilities):
                        continue
                
                # Score based on efficiency and load
                score = agent.metrics.efficiency_score * (1 - agent.metrics.current_load)
                candidates.append((score, agent))
            
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                return candidates[0][1]
            
            return None
    
    async def reallocate_agent(
        self,
        agent_id: AgentID,
        target_branch_id: BranchID,
    ) -> bool:
        """Reallocate an agent to a different branch."""
        async with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                raise ReallocationError(f"Agent {agent_id} not found")
            
            if agent.status == AgentStatus.BUSY:
                raise ReallocationError(f"Cannot reallocate busy agent {agent_id}")
            
            old_branch = agent.branch_id
            
            # Update indices
            self._branch_agents[old_branch].discard(agent_id)
            self._branch_agents[target_branch_id].add(agent_id)
            
            # Update agent
            agent.branch_id = target_branch_id
            
            # Record reallocation
            self._reallocation_history.append({
                "timestamp": datetime.now().isoformat(),
                "agent_id": agent_id,
                "from_branch": old_branch,
                "to_branch": target_branch_id,
            })
            
            logger.info(f"Reallocated agent {agent_id} from {old_branch} to {target_branch_id}")
            return True
    
    async def auto_scale(self, branch_loads: dict[BranchID, float]) -> dict[str, Any]:
        """Automatically scale agents based on branch loads."""
        async with self._lock:
            actions = {
                "spawned": [],
                "terminated": [],
                "reallocated": [],
            }
            
            # Identify overloaded branches
            for branch_id, load in branch_loads.items():
                if load > self._spawn_threshold:
                    # Spawn new agent
                    if len(self._agents) < self._max_agents:
                        agent = await self.spawn_agent(branch_id)
                        actions["spawned"].append(agent.agent_id)
            
            # Identify underutilized agents
            idle_agents = [
                agent for agent in self._agents.values()
                if agent.status == AgentStatus.IDLE
            ]
            
            # Terminate excess idle agents
            excess = len(self._agents) - self._min_agents
            for agent in sorted(
                idle_agents,
                key=lambda a: a.metrics.last_heartbeat
            ):
                if excess <= 0:
                    break
                
                idle_time = (datetime.now() - agent.metrics.last_heartbeat).total_seconds()
                if idle_time > self._idle_timeout_sec:
                    if await self.terminate_agent(agent.agent_id):
                        actions["terminated"].append(agent.agent_id)
                        excess -= 1
            
            # Reallocate from low-priority to high-priority branches
            # (Would need branch priorities passed in)
            
            return actions
    
    async def get_stats(self) -> dict[str, Any]:
        """Get agent pool statistics."""
        async with self._lock:
            status_counts = defaultdict(int)
            for agent in self._agents.values():
                status_counts[agent.status.value] += 1
            
            return {
                "total_agents": len(self._agents),
                "status_breakdown": dict(status_counts),
                "branch_distribution": {
                    branch_id: len(agent_ids)
                    for branch_id, agent_ids in self._branch_agents.items()
                },
                "capability_index": {
                    cap: len(agent_ids)
                    for cap, agent_ids in self._capability_index.items()
                },
                "reallocation_history_count": len(self._reallocation_history),
            }


# ============================================================================
# SwarmOrchestrator - Main Class
# ============================================================================

class SwarmOrchestrator:
    """
    Master orchestrator for autoconstitution.
    
    Implements PARL (Parallel Autonomous Research Layer) principles:
    - Parallel execution of research tasks
    - Autonomous agent management
    - Dynamic resource allocation
    - Layered architecture with clear separation of concerns
    
    Features:
    - Task DAG management with dependency resolution
    - Sub-agent spawning and lifecycle management
    - Performance monitoring and metrics collection
    - Dynamic agent reallocation based on load
    - Branch-based research organization
    - Fault tolerance and recovery
    
    Example:
        >>> orchestrator = SwarmOrchestrator()
        >>> await orchestrator.initialize()
        >>> 
        >>> # Create a research branch
        >>> branch = await orchestrator.create_branch(
        ...     name="Literature Review",
        ...     priority=BranchPriority.HIGH
        ... )
        >>> 
        >>> # Add tasks
        >>> task1 = await orchestrator.add_task(
        ...     branch_id=branch.branch_id,
        ...     name="Search Papers",
        ...     coro=search_papers,
        ...     args=("machine learning",)
        ... )
        >>> 
        >>> task2 = await orchestrator.add_task(
        ...     branch_id=branch.branch_id,
        ...     name="Analyze Papers",
        ...     coro=analyze_papers,
        ...     dependencies={TaskDependency(task1.task_id)}
        ... )
        >>> 
        >>> # Execute
        >>> await orchestrator.execute_branch(branch.branch_id)
        >>> 
        >>> # Cleanup
        >>> await orchestrator.shutdown()
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 50,
        task_timeout_sec: float = 300.0,
        enable_auto_scaling: bool = True,
        enable_monitoring: bool = True,
    ) -> None:
        """
        Initialize the SwarmOrchestrator.
        
        Args:
            max_concurrent_tasks: Maximum number of tasks to run concurrently
            task_timeout_sec: Default timeout for task execution
            enable_auto_scaling: Enable automatic agent scaling
            enable_monitoring: Enable performance monitoring
        """
        self._max_concurrent_tasks = max_concurrent_tasks
        self._task_timeout_sec = task_timeout_sec
        self._enable_auto_scaling = enable_auto_scaling
        self._enable_monitoring = enable_monitoring
        
        # Core components
        self._dag = TaskDAG()
        self._agent_pool = AgentPoolManager()
        self._monitor = PerformanceMonitor() if enable_monitoring else None
        
        # State
        self._branches: dict[BranchID, ResearchBranch] = {}
        self._running_tasks: dict[TaskID, asyncio.Task[Any]] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._shutdown_event = asyncio.Event()
        self._initialized = False
        
        # Background tasks
        self._background_tasks: set[asyncio.Task[Any]] = set()
        
        # Locks
        self._branch_lock = asyncio.Lock()
        self._task_lock = asyncio.Lock()
    
    # ========================================================================
    # Lifecycle Methods
    # ========================================================================
    
    async def initialize(self) -> None:
        """Initialize the orchestrator and start background tasks."""
        if self._initialized:
            return
        
        logger.info("Initializing SwarmOrchestrator...")
        
        # Start monitoring task
        if self._enable_monitoring:
            monitor_task = asyncio.create_task(self._monitoring_loop())
            self._background_tasks.add(monitor_task)
            monitor_task.add_done_callback(self._background_tasks.discard)
        
        # Start auto-scaling task
        if self._enable_auto_scaling:
            scaling_task = asyncio.create_task(self._auto_scaling_loop())
            self._background_tasks.add(scaling_task)
            scaling_task.add_done_callback(self._background_tasks.discard)
        
        self._initialized = True
        logger.info("SwarmOrchestrator initialized successfully")
    
    async def shutdown(self, timeout_sec: float = 30.0) -> None:
        """
        Gracefully shutdown the orchestrator.
        
        Args:
            timeout_sec: Timeout for graceful shutdown
        """
        logger.info("Shutting down SwarmOrchestrator...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel running tasks
        async with self._task_lock:
            for task_id, task in list(self._running_tasks.items()):
                task.cancel()
                logger.debug(f"Cancelled task {task_id}")
        
        # Wait for background tasks
        if self._background_tasks:
            await asyncio.gather(
                *self._background_tasks,
                return_exceptions=True
            )
        
        # Terminate all agents
        agent_ids = list(self._agent_pool._agents.keys())
        for agent_id in agent_ids:
            await self._agent_pool.terminate_agent(agent_id, force=True)
        
        self._initialized = False
        logger.info("SwarmOrchestrator shutdown complete")
    
    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()
    
    # ========================================================================
    # Branch Management
    # ========================================================================
    
    async def create_branch(
        self,
        name: str,
        description: str = "",
        priority: BranchPriority = BranchPriority.NORMAL,
        parent_branch: Optional[BranchID] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ResearchBranch:
        """
        Create a new research branch.
        
        Args:
            name: Branch name
            description: Branch description
            priority: Branch priority level
            parent_branch: Optional parent branch ID
            metadata: Additional metadata
        
        Returns:
            The created ResearchBranch
        """
        branch_id = f"branch_{uuid.uuid4().hex[:8]}"
        
        branch = ResearchBranch(
            branch_id=branch_id,
            name=name,
            description=description,
            priority=priority,
            parent_branch=parent_branch,
            metadata=metadata or {},
        )
        
        async with self._branch_lock:
            self._branches[branch_id] = branch
            
            # Register with parent if specified
            if parent_branch and parent_branch in self._branches:
                self._branches[parent_branch].child_branches.add(branch_id)
        
        # Register with monitor
        if self._monitor:
            await self._monitor.register_branch(branch)
        
        logger.info(f"Created branch {branch_id}: {name}")
        return branch
    
    async def get_branch(self, branch_id: BranchID) -> Optional[ResearchBranch]:
        """Get a branch by ID."""
        async with self._branch_lock:
            return self._branches.get(branch_id)
    
    async def list_branches(self) -> list[ResearchBranch]:
        """List all branches."""
        async with self._branch_lock:
            return list(self._branches.values())
    
    async def delete_branch(
        self,
        branch_id: BranchID,
        cascade: bool = False,
    ) -> bool:
        """
        Delete a research branch.
        
        Args:
            branch_id: Branch to delete
            cascade: Also delete child branches
        
        Returns:
            True if deleted, False if not found
        """
        async with self._branch_lock:
            branch = self._branches.get(branch_id)
            if not branch:
                return False
            
            # Check for child branches
            if branch.child_branches and not cascade:
                raise BranchError(
                    f"Branch {branch_id} has child branches, use cascade=True"
                )
            
            # Delete child branches if cascading
            if cascade:
                for child_id in list(branch.child_branches):
                    await self.delete_branch(child_id, cascade=True)
            
            # Remove from parent
            if branch.parent_branch and branch.parent_branch in self._branches:
                self._branches[branch.parent_branch].child_branches.discard(branch_id)
            
            # Terminate branch agents
            for agent_id in list(branch.agent_ids):
                await self._agent_pool.terminate_agent(agent_id, force=True)
            
            # Remove branch tasks from DAG
            for task_id in list(branch.task_ids):
                await self._dag.remove_task(task_id)
            
            del self._branches[branch_id]
            logger.info(f"Deleted branch {branch_id}")
            return True
    
    # ========================================================================
    # Task Management
    # ========================================================================
    
    async def add_task(
        self,
        branch_id: BranchID,
        name: str,
        coro: Callable[..., Coroutine[Any, Any, TaskResult]],
        args: Optional[tuple[Any, ...]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        dependencies: Optional[set[TaskDependency]] = None,
        priority: int = 0,
        retry_limit: int = 3,
        timeout_sec: Optional[float] = None,
    ) -> TaskNode:
        """
        Add a task to a branch.
        
        Args:
            branch_id: Target branch ID
            name: Task name
            coro: Coroutine function to execute
            args: Positional arguments for coro
            kwargs: Keyword arguments for coro
            dependencies: Set of task dependencies
            priority: Task priority (lower = higher priority)
            retry_limit: Maximum retry attempts
            timeout_sec: Execution timeout
        
        Returns:
            The created TaskNode
        """
        # Validate branch
        branch = await self.get_branch(branch_id)
        if not branch:
            raise BranchError(f"Branch {branch_id} not found")
        
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        task = TaskNode(
            task_id=task_id,
            branch_id=branch_id,
            name=name,
            coro=coro,
            args=args or (),
            kwargs=kwargs or {},
            dependencies=dependencies or set(),
            priority=priority,
            retry_limit=retry_limit,
            timeout_sec=timeout_sec or self._task_timeout_sec,
        )
        
        # Add to DAG
        await self._dag.add_task(task)
        
        # Add to branch
        branch.add_task(task_id)
        
        logger.debug(f"Added task {task_id} to branch {branch_id}")
        return task
    
    async def get_task(self, task_id: TaskID) -> Optional[TaskNode]:
        """Get a task by ID."""
        return await self._dag.get_task(task_id)
    
    async def cancel_task(self, task_id: TaskID) -> bool:
        """Cancel a running or pending task."""
        async with self._task_lock:
            # Cancel if running
            running_task = self._running_tasks.get(task_id)
            if running_task:
                running_task.cancel()
                del self._running_tasks[task_id]
            
            # Update status
            await self._dag.update_task_status(task_id, TaskStatus.CANCELLED)
            
            logger.debug(f"Cancelled task {task_id}")
            return True
    
    async def get_branch_tasks(self, branch_id: BranchID) -> list[TaskNode]:
        """Get all tasks in a branch."""
        return await self._dag.get_branch_tasks(branch_id)
    
    async def get_task_status(self, task_id: TaskID) -> Optional[TaskStatus]:
        """Get the status of a task."""
        task = await self.get_task(task_id)
        return task.status if task else None
    
    async def wait_for_task(
        self,
        task_id: TaskID,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Wait for a task to complete and return its result.
        
        Args:
            task_id: Task to wait for
            timeout: Maximum wait time
        
        Returns:
            Task result
        
        Raises:
            TimeoutError: If timeout is reached
            Exception: If task failed
        """
        start_time = time.monotonic()
        
        while True:
            task = await self.get_task(task_id)
            if not task:
                raise TaskDAGError(f"Task {task_id} not found")
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            
            if task.status == TaskStatus.FAILED:
                raise task.error or Exception(f"Task {task_id} failed")
            
            if task.status == TaskStatus.CANCELLED:
                raise asyncio.CancelledError(f"Task {task_id} was cancelled")
            
            if timeout and (time.monotonic() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")
            
            await asyncio.sleep(0.1)
    
    # ========================================================================
    # Execution
    # ========================================================================
    
    async def execute_task(self, task_id: TaskID) -> Any:
        """
        Execute a single task.
        
        Args:
            task_id: Task to execute
        
        Returns:
            Task result
        """
        task = await self.get_task(task_id)
        if not task:
            raise TaskDAGError(f"Task {task_id} not found")
        
        async with self._semaphore:
            # Update status
            await self._dag.update_task_status(task_id, TaskStatus.RUNNING)
            task.scheduled_at = datetime.now()
            
            # Find or spawn agent
            agent = await self._agent_pool.find_best_agent(task.branch_id)
            if not agent:
                agent = await self._agent_pool.spawn_agent(task.branch_id)
                
                # Register with monitor
                if self._monitor:
                    await self._monitor.register_agent(agent)
            
            agent.status = AgentStatus.BUSY
            agent.current_task = task_id
            
            logger.debug(f"Executing task {task_id} on agent {agent.agent_id}")
            
            try:
                # Execute with timeout
                start_time = time.monotonic()
                
                coro = task.coro(*task.args, **task.kwargs)
                
                if task.timeout_sec:
                    result = await asyncio.wait_for(
                        coro,
                        timeout=task.timeout_sec,
                    )
                else:
                    result = await coro
                
                duration_ms = (time.monotonic() - start_time) * 1000
                
                # Update metrics
                task.metrics.end_time = datetime.now()
                task.metrics.cpu_time_ms = duration_ms
                
                # Update status
                await self._dag.update_task_status(
                    task_id,
                    TaskStatus.COMPLETED,
                    result=result,
                )
                
                # Update agent metrics
                agent.metrics.update_execution(duration_ms, success=True)
                agent.status = AgentStatus.IDLE
                agent.current_task = None
                
                # Record with monitor
                if self._monitor:
                    await self._monitor.record_task_completion(task, success=True)
                    await self._monitor.update_agent_metrics(
                        agent.agent_id,
                        duration_ms,
                        success=True,
                    )
                
                logger.debug(f"Task {task_id} completed successfully")
                return result
                
            except asyncio.TimeoutError as e:
                await self._handle_task_failure(task, agent, e, TaskStatus.FAILED)
                raise
                
            except Exception as e:
                await self._handle_task_failure(task, agent, e, TaskStatus.FAILED)
                raise
    
    async def _handle_task_failure(
        self,
        task: TaskNode,
        agent: SubAgent,
        error: Exception,
        status: TaskStatus,
    ) -> None:
        """Handle task failure and retry logic."""
        task.metrics.error_count += 1
        
        # Check retry limit
        if task.metrics.retry_count < task.retry_limit:
            task.metrics.retry_count += 1
            await self._dag.update_task_status(
                task.task_id,
                TaskStatus.RETRYING,
                error=error,
            )
            logger.warning(
                f"Task {task.task_id} failed, retrying ({task.metrics.retry_count}/{task.retry_limit})"
            )
        else:
            await self._dag.update_task_status(
                task.task_id,
                status,
                error=error,
            )
            
            # Update agent metrics
            duration_ms = task.metrics.duration_ms
            agent.metrics.update_execution(duration_ms, success=False)
            
            # Record with monitor
            if self._monitor:
                await self._monitor.record_task_completion(task, success=False)
                await self._monitor.update_agent_metrics(
                    agent.agent_id,
                    duration_ms,
                    success=False,
                )
            
            agent.status = AgentStatus.IDLE
            agent.current_task = None
            
            logger.error(f"Task {task.task_id} failed permanently: {error}")
    
    async def execute_branch(
        self,
        branch_id: BranchID,
        max_parallel: Optional[int] = None,
    ) -> dict[TaskID, Any]:
        """
        Execute all tasks in a branch respecting dependencies.
        
        Args:
            branch_id: Branch to execute
            max_parallel: Maximum parallel tasks (None = use default)
        
        Returns:
            Dictionary mapping task IDs to results
        """
        branch = await self.get_branch(branch_id)
        if not branch:
            raise BranchError(f"Branch {branch_id} not found")
        
        logger.info(f"Executing branch {branch_id}: {branch.name}")
        
        results: dict[TaskID, Any] = {}
        completed_tasks: set[TaskID] = set()
        failed_tasks: set[TaskID] = set()
        
        semaphore = asyncio.Semaphore(max_parallel or self._max_concurrent_tasks)
        
        async def execute_with_semaphore(task_id: TaskID) -> None:
            async with semaphore:
                try:
                    result = await self.execute_task(task_id)
                    results[task_id] = result
                    completed_tasks.add(task_id)
                except Exception as e:
                    failed_tasks.add(task_id)
                    logger.error(f"Task {task_id} failed: {e}")
        
        # Execute tasks in waves based on dependencies
        while not self._shutdown_event.is_set():
            # Get ready tasks
            ready_tasks = await self._dag.get_ready_tasks(branch_id)
            
            # Filter out already processed tasks
            ready_tasks = [
                t for t in ready_tasks
                if t.task_id not in completed_tasks
                and t.task_id not in failed_tasks
            ]
            
            if not ready_tasks:
                # Check if all tasks are done
                all_tasks = await self._dag.get_branch_tasks(branch_id)
                all_done = all(
                    t.task_id in completed_tasks or t.task_id in failed_tasks
                    or t.status == TaskStatus.CANCELLED
                    for t in all_tasks
                )
                if all_done:
                    break
                
                # Wait a bit and check again
                await asyncio.sleep(0.1)
                continue
            
            # Execute ready tasks concurrently
            task_coros = [
                execute_with_semaphore(task.task_id)
                for task in ready_tasks
            ]
            
            await asyncio.gather(*task_coros, return_exceptions=True)
        
        # Update branch metrics
        branch.metrics.completed_count = len(completed_tasks)
        
        logger.info(
            f"Branch {branch_id} execution complete: "
            f"{len(completed_tasks)} completed, {len(failed_tasks)} failed"
        )
        
        return results
    
    async def execute_all(self) -> dict[BranchID, dict[TaskID, Any]]:
        """Execute all branches and return all results."""
        branches = await self.list_branches()
        
        # Sort branches by priority
        branches.sort(key=lambda b: b.priority.value)
        
        all_results: dict[BranchID, dict[TaskID, Any]] = {}
        
        for branch in branches:
            if self._shutdown_event.is_set():
                break
            
            results = await self.execute_branch(branch.branch_id)
            all_results[branch.branch_id] = results
        
        return all_results
    
    # ========================================================================
    # Agent Management
    # ========================================================================
    
    async def spawn_agent(
        self,
        branch_id: BranchID,
        name: Optional[str] = None,
        capabilities: Optional[set[str]] = None,
    ) -> SubAgent:
        """Spawn a new sub-agent for a branch."""
        agent = await self._agent_pool.spawn_agent(branch_id, name, capabilities)
        
        # Add to branch
        branch = await self.get_branch(branch_id)
        if branch:
            branch.add_agent(agent.agent_id)
        
        # Register with monitor
        if self._monitor:
            await self._monitor.register_agent(agent)
        
        return agent
    
    async def terminate_agent(self, agent_id: AgentID, force: bool = False) -> bool:
        """Terminate a sub-agent."""
        agent = await self._agent_pool.get_agent(agent_id)
        if not agent:
            return False
        
        # Remove from branch
        branch = await self.get_branch(agent.branch_id)
        if branch:
            branch.remove_agent(agent_id)
        
        return await self._agent_pool.terminate_agent(agent_id, force)
    
    async def reallocate_agent(
        self,
        agent_id: AgentID,
        target_branch_id: BranchID,
    ) -> bool:
        """
        Reallocate an agent to a different branch.
        
        Args:
            agent_id: Agent to reallocate
            target_branch_id: Target branch
        
        Returns:
            True if successful
        """
        agent = await self._agent_pool.get_agent(agent_id)
        if not agent:
            raise AgentError(f"Agent {agent_id} not found")
        
        # Remove from old branch
        old_branch = await self.get_branch(agent.branch_id)
        if old_branch:
            old_branch.remove_agent(agent_id)
        
        # Reallocate
        success = await self._agent_pool.reallocate_agent(agent_id, target_branch_id)
        
        # Add to new branch
        new_branch = await self.get_branch(target_branch_id)
        if new_branch:
            new_branch.add_agent(agent_id)
        
        return success
    
    async def get_branch_agents(self, branch_id: BranchID) -> list[SubAgent]:
        """Get all agents in a branch."""
        return await self._agent_pool.get_branch_agents(branch_id)
    
    # ========================================================================
    # Dynamic Reallocation
    # ========================================================================
    
    async def analyze_and_reallocate(self) -> dict[str, Any]:
        """
        Analyze branch performance and reallocate agents dynamically.
        
        Returns:
            Report of reallocation actions taken
        """
        if not self._monitor:
            return {"error": "Monitoring not enabled"}
        
        actions = {
            "analyzed_at": datetime.now().isoformat(),
            "bottlenecks": [],
            "reallocations": [],
            "spawns": [],
            "terminations": [],
        }
        
        # Identify bottlenecks
        bottlenecks = await self._monitor.identify_bottlenecks()
        actions["bottlenecks"] = bottlenecks
        
        # Calculate branch loads
        branch_loads: dict[BranchID, float] = {}
        for branch_id, branch in self._branches.items():
            pending_tasks = sum(
                1 for t in await self._dag.get_branch_tasks(branch_id)
                if t.status == TaskStatus.PENDING
            )
            agent_count = len(branch.agent_ids)
            load = pending_tasks / max(agent_count, 1)
            branch_loads[branch_id] = load
        
        # Auto-scale based on loads
        if self._enable_auto_scaling:
            scale_actions = await self._agent_pool.auto_scale(branch_loads)
            actions["spawns"] = scale_actions.get("spawned", [])
            actions["terminations"] = scale_actions.get("terminated", [])
        
        # Reallocate from low-load to high-load branches
        if branch_loads:
            sorted_branches = sorted(
                branch_loads.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            high_load_branches = [
                (bid, load) for bid, load in sorted_branches if load > 2.0
            ]
            low_load_branches = [
                (bid, load) for bid, load in sorted_branches if load < 0.5
            ]
            
            for high_bid, _ in high_load_branches:
                if low_load_branches:
                    low_bid, _ = low_load_branches.pop(0)
                    
                    # Find idle agent in low-load branch
                    agents = await self._agent_pool.get_branch_agents(low_bid)
                    idle_agents = [a for a in agents if a.status == AgentStatus.IDLE]
                    
                    if idle_agents:
                        agent = idle_agents[0]
                        try:
                            await self.reallocate_agent(agent.agent_id, high_bid)
                            actions["reallocations"].append({
                                "agent_id": agent.agent_id,
                                "from_branch": low_bid,
                                "to_branch": high_bid,
                            })
                        except ReallocationError as e:
                            logger.warning(f"Reallocation failed: {e}")
        
        return actions
    
    # ========================================================================
    # Monitoring & Metrics
    # ========================================================================
    
    async def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive metrics."""
        dag_stats = await self._dag.get_stats()
        agent_stats = await self._agent_pool.get_stats()
        
        metrics = {
            "dag": dag_stats,
            "agents": agent_stats,
            "branches": {
                branch_id: branch.to_dict()
                for branch_id, branch in self._branches.items()
            },
        }
        
        if self._monitor:
            metrics["health"] = await self._monitor.get_health_report()
        
        return metrics
    
    async def get_branch_metrics(self, branch_id: BranchID) -> Optional[BranchMetrics]:
        """Get metrics for a specific branch."""
        if self._monitor:
            return await self._monitor.get_branch_performance(branch_id)
        return None
    
    async def get_agent_metrics(self, agent_id: AgentID) -> Optional[AgentMetrics]:
        """Get metrics for a specific agent."""
        if self._monitor:
            return await self._monitor.get_agent_performance(agent_id)
        return None
    
    # ========================================================================
    # Background Tasks
    # ========================================================================
    
    async def _monitoring_loop(self) -> None:
        """Background task for continuous monitoring."""
        logger.info("Starting monitoring loop")
        
        while not self._shutdown_event.is_set():
            try:
                # Check agent health
                for agent in list(self._agent_pool._agents.values()):
                    if not agent.metrics.is_alive():
                        logger.warning(f"Agent {agent.agent_id} appears dead, terminating")
                        await self.terminate_agent(agent.agent_id, force=True)
                
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    async def _auto_scaling_loop(self) -> None:
        """Background task for automatic scaling."""
        logger.info("Starting auto-scaling loop")
        
        while not self._shutdown_event.is_set():
            try:
                await self.analyze_and_reallocate()
                
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def generate_report(self) -> dict[str, Any]:
        """Generate a comprehensive status report."""
        return {
            "orchestrator": {
                "initialized": self._initialized,
                "max_concurrent_tasks": self._max_concurrent_tasks,
                "auto_scaling_enabled": self._enable_auto_scaling,
                "monitoring_enabled": self._enable_monitoring,
            },
            "branches": [
                branch.to_dict() for branch in self._branches.values()
            ],
            "running_tasks": len(self._running_tasks),
        }
    
    async def export_state(self) -> dict[str, Any]:
        """Export orchestrator state for persistence."""
        async with self._branch_lock:
            return {
                "branches": {
                    bid: {
                        "branch_id": b.branch_id,
                        "name": b.name,
                        "description": b.description,
                        "priority": b.priority.name,
                        "task_ids": list(b.task_ids),
                        "agent_ids": list(b.agent_ids),
                        "metadata": b.metadata,
                    }
                    for bid, b in self._branches.items()
                },
                "dag_stats": await self._dag.get_stats(),
                "agent_stats": await self._agent_pool.get_stats(),
            }


# ============================================================================
# Decorators and Utilities
# ============================================================================

def task(
    priority: int = 0,
    retry_limit: int = 3,
    timeout_sec: Optional[float] = None,
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    """
    Decorator to mark a function as a task with metadata.
    
    Args:
        priority: Task priority (lower = higher priority)
        retry_limit: Maximum retry attempts
        timeout_sec: Execution timeout
    """
    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]]
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        func._task_priority = priority  # type: ignore
        func._task_retry_limit = retry_limit  # type: ignore
        func._task_timeout_sec = timeout_sec  # type: ignore
        return func
    return decorator


def retryable(
    max_retries: int = 3,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    backoff_sec: float = 1.0,
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    """
    Decorator to add retry logic to async functions.
    
    Args:
        max_retries: Maximum retry attempts
        exceptions: Exceptions to catch and retry
        backoff_sec: Backoff time between retries
    """
    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]]
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Optional[Exception] = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_retries:
                        wait_time = backoff_sec * (2 ** attempt)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}), "
                            f"retrying in {wait_time}s: {e}"
                        )
                        await asyncio.sleep(wait_time)
            
            raise last_error or Exception(f"{func.__name__} failed after {max_retries} retries")
        
        return wrapper
    return decorator


# ============================================================================
# Example Usage
# ============================================================================

async def example_usage() -> None:
    """Example of how to use the SwarmOrchestrator."""
    
    # Sample task implementations
    async def search_papers(query: str) -> list[str]:
        await asyncio.sleep(0.1)
        return [f"Paper about {query} #{i}" for i in range(3)]
    
    async def analyze_papers(papers: list[str]) -> dict[str, Any]:
        await asyncio.sleep(0.1)
        return {"analyzed": len(papers), "insights": ["insight1", "insight2"]}
    
    async def write_summary(analysis: dict[str, Any]) -> str:
        await asyncio.sleep(0.1)
        return f"Summary of {analysis['analyzed']} papers"
    
    # Create orchestrator
    async with SwarmOrchestrator(
        max_concurrent_tasks=10,
        enable_auto_scaling=True,
        enable_monitoring=True,
    ) as orchestrator:
        
        # Create branch
        branch = await orchestrator.create_branch(
            name="Literature Review",
            description="Automated literature review on ML",
            priority=BranchPriority.HIGH,
        )
        
        # Add tasks with dependencies
        task1 = await orchestrator.add_task(
            branch_id=branch.branch_id,
            name="Search Papers",
            coro=search_papers,
            args=("machine learning",),
            priority=0,
        )
        
        task2 = await orchestrator.add_task(
            branch_id=branch.branch_id,
            name="Analyze Papers",
            coro=analyze_papers,
            args=(["paper1", "paper2"],),  # Would use task1.result in real usage
            dependencies={TaskDependency(task1.task_id)},
            priority=1,
        )
        
        task3 = await orchestrator.add_task(
            branch_id=branch.branch_id,
            name="Write Summary",
            coro=write_summary,
            args=({"analyzed": 2, "insights": []},),  # Would use task2.result
            dependencies={TaskDependency(task2.task_id)},
            priority=2,
        )
        
        # Execute branch
        results = await orchestrator.execute_branch(branch.branch_id)
        
        # Print results
        print("Execution Results:")
        for task_id, result in results.items():
            print(f"  {task_id}: {result}")
        
        # Get metrics
        metrics = await orchestrator.get_metrics()
        print(f"\nMetrics: {json.dumps(metrics, indent=2, default=str)}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
