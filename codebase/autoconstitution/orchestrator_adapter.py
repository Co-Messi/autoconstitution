"""
Orchestrator Adapter - Bridges SwarmOrchestrator with BaseAgent framework.

This module provides adapters to integrate the SwarmOrchestrator with the
existing BaseAgent class hierarchy, enabling seamless use of both systems.

Python 3.11+ async-first implementation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar, cast

from .orchestrator import (
    SwarmOrchestrator,
    TaskNode,
    TaskDependency,
    SubAgent as OrchestratorAgent,
    ResearchBranch,
    BranchPriority,
    TaskStatus,
    AgentStatus as OrchestratorAgentStatus,
)
from .agents.base import (
    BaseAgent,
    AgentID,
    AgentStatus,
    ExecutionContext,
    ExecutionResult,
    Finding,
    Message,
    MessagePriority,
    LLMProvider,
    CheckpointStore,
    BroadcastChannel,
)

# Type variables for generic types
TContext = TypeVar("TContext")
TResult = TypeVar("TResult")


@dataclass(slots=True)
class AgentAdapterConfig:
    """Configuration for agent adapter."""
    enable_checkpoints: bool = True
    enable_broadcast: bool = True
    checkpoint_interval_sec: float = 60.0
    max_execution_history: int = 100


class AgentTaskWrapper(Generic[TContext, TResult]):
    """
    Wraps a BaseAgent execution as an orchestrator task.
    
    This adapter allows BaseAgent instances to be executed within the
    SwarmOrchestrator's task DAG system.
    
    Example:
        >>> agent = MyResearchAgent()
        >>> wrapper = AgentTaskWrapper(agent, execution_context)
        >>> task = await orchestrator.add_task(
        ...     branch_id=branch.branch_id,
        ...     name="Research Task",
        ...     coro=wrapper.execute,
        ... )
    """
    
    def __init__(
        self,
        agent: BaseAgent[TContext, TResult],
        context: TContext,
        config: AgentAdapterConfig | None = None,
    ) -> None:
        """
        Initialize the agent task wrapper.
        
        Args:
            agent: The BaseAgent instance to wrap
            context: Execution context for the agent
            config: Optional adapter configuration
        """
        self._agent = agent
        self._context = context
        self._config = config or AgentAdapterConfig()
        self._execution_count: int = 0
        self._last_result: ExecutionResult[TResult] | None = None
    
    async def execute(self) -> ExecutionResult[TResult]:
        """
        Execute the wrapped agent.
        
        Returns:
            ExecutionResult containing the agent's output
        """
        self._execution_count += 1
        
        # Update agent status
        self._agent._status = AgentStatus.EXECUTING
        
        try:
            # Execute the agent
            result = await self._agent.execute(self._context)
            self._last_result = result
            
            # Update status based on result
            self._agent._status = (
                AgentStatus.IDLE if result.success else AgentStatus.ERROR
            )
            
            return result
            
        except Exception as e:
            self._agent._status = AgentStatus.ERROR
            return ExecutionResult(
                success=False,
                error=e,
                execution_time_ms=0.0,
            )
    
    @property
    def agent(self) -> BaseAgent[TContext, TResult]:
        """Get the wrapped agent."""
        return self._agent
    
    @property
    def execution_count(self) -> int:
        """Get number of executions."""
        return self._execution_count
    
    @property
    def last_result(self) -> ExecutionResult[TResult] | None:
        """Get the last execution result."""
        return self._last_result


class OrchestratorAgentAdapter:
    """
    Adapts a BaseAgent to work as an Orchestrator SubAgent.
    
    This adapter allows BaseAgent instances to be managed by the
    AgentPoolManager and participate in the orchestrator's
    lifecycle management.
    
    Example:
        >>> base_agent = MyResearchAgent()
        >>> adapted = OrchestratorAgentAdapter(
        ...     base_agent,
        ...     branch_id="research_branch",
        ... )
        >>> await orchestrator._agent_pool.spawn_adapted_agent(adapted)
    """
    
    def __init__(
        self,
        agent: BaseAgent[Any, Any],
        branch_id: str,
        capabilities: set[str] | None = None,
    ) -> None:
        """
        Initialize the adapter.
        
        Args:
            agent: The BaseAgent to adapt
            branch_id: Branch ID for the orchestrator
            capabilities: Optional capability tags
        """
        self._base_agent = agent
        self._branch_id = branch_id
        self._capabilities = capabilities or set()
        self._orchestrator_agent: OrchestratorAgent | None = None
        self._task_queue: asyncio.Queue[AgentTaskWrapper[Any, Any]] = asyncio.Queue()
        self._is_running: bool = False
        self._task: asyncio.Task[None] | None = None
    
    async def start(self) -> None:
        """Start the agent's processing loop."""
        if self._is_running:
            return
        
        self._is_running = True
        self._task = asyncio.create_task(self._processing_loop())
    
    async def stop(self) -> None:
        """Stop the agent's processing loop."""
        self._is_running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
    
    async def _processing_loop(self) -> None:
        """Main processing loop for the adapted agent."""
        while self._is_running:
            try:
                # Wait for tasks with timeout
                wrapper = await asyncio.wait_for(
                    self._task_queue.get(),
                    timeout=1.0,
                )
                
                # Execute the task
                await wrapper.execute()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue processing
                import logging
                logging.getLogger("AgentAdapter").error(
                    f"Agent {self._base_agent.agent_id} error: {e}"
                )
    
    async def submit_task(
        self,
        wrapper: AgentTaskWrapper[Any, Any],
    ) -> None:
        """Submit a task for execution."""
        await self._task_queue.put(wrapper)
    
    @property
    def base_agent(self) -> BaseAgent[Any, Any]:
        """Get the underlying BaseAgent."""
        return self._base_agent
    
    @property
    def branch_id(self) -> str:
        """Get the branch ID."""
        return self._branch_id
    
    @property
    def capabilities(self) -> set[str]:
        """Get agent capabilities."""
        return self._capabilities.copy()
    
    @property
    def is_running(self) -> bool:
        """Check if the agent is running."""
        return self._is_running
    
    @property
    def queue_size(self) -> int:
        """Get current task queue size."""
        return self._task_queue.qsize()


class ExtendedSwarmOrchestrator(SwarmOrchestrator):
    """
    Extended SwarmOrchestrator with BaseAgent integration.
    
    This extended orchestrator provides seamless integration with the
    BaseAgent framework, allowing both coroutine-based tasks and
    BaseAgent instances to be managed within the same system.
    
    Example:
        >>> async with ExtendedSwarmOrchestrator() as orchestrator:
        ...     branch = await orchestrator.create_branch("Research")
        ...     
        ...     # Add coroutine-based task
        ...     task1 = await orchestrator.add_task(
        ...         branch_id=branch.branch_id,
        ...         name="Search",
        ...         coro=search_function,
        ...     )
        ...     
        ...     # Add BaseAgent task
        ...     agent = MyResearchAgent()
        ...     task2 = await orchestrator.add_agent_task(
        ...         branch_id=branch.branch_id,
        ...         name="Analyze",
        ...         agent=agent,
        ...         context=execution_context,
        ...         dependencies={TaskDependency(task1.task_id)},
        ...     )
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 50,
        task_timeout_sec: float = 300.0,
        enable_auto_scaling: bool = True,
        enable_monitoring: bool = True,
    ) -> None:
        """Initialize the extended orchestrator."""
        super().__init__(
            max_concurrent_tasks=max_concurrent_tasks,
            task_timeout_sec=task_timeout_sec,
            enable_auto_scaling=enable_auto_scaling,
            enable_monitoring=enable_monitoring,
        )
        
        # Track adapted agents
        self._adapted_agents: dict[str, OrchestratorAgentAdapter] = {}
        self._agent_adapter_lock = asyncio.Lock()
    
    async def add_agent_task(
        self,
        branch_id: str,
        name: str,
        agent: BaseAgent[Any, Any],
        context: Any,
        dependencies: set[TaskDependency] | None = None,
        priority: int = 0,
        retry_limit: int = 3,
        timeout_sec: float | None = None,
        capabilities: set[str] | None = None,
    ) -> TaskNode:
        """
        Add a BaseAgent as a task in the orchestrator.
        
        Args:
            branch_id: Target branch ID
            name: Task name
            agent: BaseAgent instance to execute
            context: Execution context for the agent
            dependencies: Task dependencies
            priority: Task priority
            retry_limit: Maximum retry attempts
            timeout_sec: Execution timeout
            capabilities: Agent capabilities
        
        Returns:
            Created TaskNode
        """
        # Create or get adapter for this agent
        agent_key = str(agent.agent_id)
        
        async with self._agent_adapter_lock:
            if agent_key not in self._adapted_agents:
                adapter = OrchestratorAgentAdapter(
                    agent=agent,
                    branch_id=branch_id,
                    capabilities=capabilities or set(),
                )
                self._adapted_agents[agent_key] = adapter
                await adapter.start()
            else:
                adapter = self._adapted_agents[agent_key]
        
        # Create task wrapper
        wrapper = AgentTaskWrapper(agent, context)
        
        # Add as task
        return await self.add_task(
            branch_id=branch_id,
            name=name,
            coro=wrapper.execute,
            dependencies=dependencies,
            priority=priority,
            retry_limit=retry_limit,
            timeout_sec=timeout_sec,
        )
    
    async def register_agent(
        self,
        branch_id: str,
        agent: BaseAgent[Any, Any],
        capabilities: set[str] | None = None,
    ) -> OrchestratorAgentAdapter:
        """
        Register a BaseAgent with the orchestrator.
        
        Args:
            branch_id: Branch to register agent to
            agent: BaseAgent instance
            capabilities: Optional capabilities
        
        Returns:
            Agent adapter instance
        """
        adapter = OrchestratorAgentAdapter(
            agent=agent,
            branch_id=branch_id,
            capabilities=capabilities or set(),
        )
        
        async with self._agent_adapter_lock:
            self._adapted_agents[str(agent.agent_id)] = adapter
        
        await adapter.start()
        
        # Add to branch
        branch = await self.get_branch(branch_id)
        if branch:
            # Create a placeholder agent ID for tracking
            pass
        
        return adapter
    
    async def unregister_agent(self, agent_id: AgentID) -> bool:
        """
        Unregister a BaseAgent from the orchestrator.
        
        Args:
            agent_id: Agent ID to unregister
        
        Returns:
            True if unregistered, False if not found
        """
        agent_key = str(agent_id)
        
        async with self._agent_adapter_lock:
            adapter = self._adapted_agents.pop(agent_key, None)
        
        if adapter:
            await adapter.stop()
            return True
        
        return False
    
    async def shutdown(self, timeout_sec: float = 30.0) -> None:
        """Gracefully shutdown the orchestrator and all adapted agents."""
        # Stop all adapted agents
        async with self._agent_adapter_lock:
            adapters = list(self._adapted_agents.values())
            self._adapted_agents.clear()
        
        for adapter in adapters:
            await adapter.stop()
        
        # Call parent shutdown
        await super().shutdown(timeout_sec)
    
    async def get_agent_status(self, agent_id: AgentID) -> dict[str, Any] | None:
        """
        Get status of a registered BaseAgent.
        
        Args:
            agent_id: Agent ID to query
        
        Returns:
            Status dictionary or None if not found
        """
        agent_key = str(agent_id)
        
        async with self._agent_adapter_lock:
            adapter = self._adapted_agents.get(agent_key)
        
        if not adapter:
            return None
        
        return {
            "agent_id": str(agent_id),
            "branch_id": adapter.branch_id,
            "is_running": adapter.is_running,
            "queue_size": adapter.queue_size,
            "capabilities": list(adapter.capabilities),
            "base_agent_status": adapter.base_agent._status.name,
        }
    
    async def list_adapted_agents(self) -> list[dict[str, Any]]:
        """List all registered BaseAgents."""
        async with self._agent_adapter_lock:
            return [
                {
                    "agent_id": str(adapter.base_agent.agent_id),
                    "branch_id": adapter.branch_id,
                    "is_running": adapter.is_running,
                    "capabilities": list(adapter.capabilities),
                }
                for adapter in self._adapted_agents.values()
            ]


# Convenience function for creating agent tasks
def create_agent_task_wrapper(
    agent: BaseAgent[TContext, TResult],
    context: TContext,
    config: AgentAdapterConfig | None = None,
) -> AgentTaskWrapper[TContext, TResult]:
    """
    Create a task wrapper for a BaseAgent.
    
    Args:
        agent: BaseAgent instance
        context: Execution context
        config: Optional adapter configuration
    
    Returns:
        Configured AgentTaskWrapper
    """
    return AgentTaskWrapper(agent, context, config)


__all__ = [
    "AgentAdapterConfig",
    "AgentTaskWrapper",
    "OrchestratorAgentAdapter",
    "ExtendedSwarmOrchestrator",
    "create_agent_task_wrapper",
]
