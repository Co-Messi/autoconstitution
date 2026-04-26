# autoconstitution Comprehensive Testing Strategy

## Executive Summary

This document defines the comprehensive testing strategy for autoconstitution, a massively parallel multi-agent AI research system. The testing approach follows a **multi-layered pyramid** that ensures correctness at every level—from individual components to full-system benchmarks.

### Testing Philosophy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SWARMRESEARCH TESTING PYRAMID                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                              ▲                                               │
│                             ╱ ╲                                              │
│                            ╱   ╲     Benchmark Tests                         │
│                           ╱  5% ╲    (End-to-end performance)                │
│                          ╱───────╲                                           │
│                         ╱         ╲                                          │
│                        ╱    10%    ╲   Simulation Tests                      │
│                       ╱  (Scenario- ╲  (Realistic workload simulation)       │
│                      ╱   based E2E)  ╲                                       │
│                     ╱─────────────────╲                                      │
│                    ╱                   ╲                                     │
│                   ╱        15%          ╲  Property-Based Tests              │
│                  ╱   (Invariant checks)  ╲ (Generative, edge case discovery) │
│                 ╱─────────────────────────╲                                  │
│                ╱                           ╲                                 │
│               ╱             30%              ╲  Integration Tests            │
│              ╱    (Component interactions)    ╲ (Cross-module verification)  │
│             ╱──────────────────────────────────╲                             │
│            ╱                                     ╲                           │
│           ╱                   40%                  ╲  Unit Tests             │
│          ╱         (Individual components)          ╲ (Function-level)       │
│         ╱────────────────────────────────────────────╲                       │
│                                                                              │
│  Total Target Coverage: >90% line coverage, 100% critical path coverage     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Unit Testing Approach

### 1.1 Testing Framework Stack

```python
# pytest configuration: pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--verbose",
    "--cov=swarm_research",
    "--cov-report=term-missing",
    "--cov-report=html:coverage_html",
    "--cov-fail-under=90"
]
markers = [
    "unit: Unit tests (fast, isolated)",
    "integration: Integration tests (slower, cross-module)",
    "property: Property-based tests (Hypothesis)",
    "simulation: Simulation tests (realistic scenarios)",
    "benchmark: Performance benchmarks",
    "slow: Tests that take >10s",
    "requires_redis: Tests requiring Redis",
    "requires_gpu: Tests requiring GPU"
]
```

### 1.2 Unit Test Organization

```
tests/
├── unit/
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures
│   ├── test_providers/                # Provider adapter tests
│   │   ├── __init__.py
│   │   ├── test_kimi_adapter.py
│   │   ├── test_claude_adapter.py
│   │   ├── test_openai_adapter.py
│   │   └── test_ollama_adapter.py
│   ├── test_orchestrator/             # Orchestrator component tests
│   │   ├── __init__.py
│   │   ├── test_swarm_manager.py
│   │   ├── test_task_scheduler.py
│   │   ├── test_result_aggregator.py
│   │   └── test_health_monitor.py
│   ├── test_communication/            # Communication layer tests
│   │   ├── __init__.py
│   │   ├── test_message_bus.py
│   │   ├── test_message_serialization.py
│   │   └── test_routing.py
│   ├── test_state/                    # State management tests
│   │   ├── __init__.py
│   │   ├── test_checkpointing.py
│   │   ├── test_state_persistence.py
│   │   └── test_recovery.py
│   └── test_agent_worker/             # Agent worker tests
│       ├── __init__.py
│       ├── test_agent_lifecycle.py
│       ├── test_tool_execution.py
│       └── test_context_management.py
```

### 1.3 Core Unit Testing Patterns

#### Pattern 1: Provider Adapter Tests

```python
# tests/unit/test_providers/test_base_adapter.py
import pytest
from unittest.mock import Mock, AsyncMock, patch
from swarm_research.providers.base import LLMProvider, GenerationConfig
from swarm_research.providers.exceptions import ProviderError, RateLimitError


class TestLLMProviderInterface:
    """Test the unified provider interface contract."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider implementing the interface."""
        provider = Mock(spec=LLMProvider)
        provider.complete = AsyncMock()
        provider.stream_complete = AsyncMock()
        provider.get_capabilities = Mock(return_value=["chat", "streaming"])
        provider.health_check = AsyncMock(return_value=True)
        return provider
    
    @pytest.mark.unit
    async def test_complete_returns_valid_structure(self, mock_provider):
        """Verify complete() returns properly structured result."""
        # Arrange
        messages = [{"role": "user", "content": "Hello"}]
        config = GenerationConfig(temperature=0.7, max_tokens=100)
        
        mock_provider.complete.return_value = {
            "content": "Hello! How can I help?",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "finish_reason": "stop"
        }
        
        # Act
        result = await mock_provider.complete(messages, config)
        
        # Assert
        assert "content" in result
        assert "usage" in result
        assert result["usage"]["prompt_tokens"] >= 0
        assert result["usage"]["completion_tokens"] >= 0
    
    @pytest.mark.unit
    async def test_rate_limit_handling(self, mock_provider):
        """Verify rate limits are properly handled."""
        # Arrange
        mock_provider.complete.side_effect = RateLimitError(
            "Rate limit exceeded", retry_after=60
        )
        
        # Act & Assert
        with pytest.raises(RateLimitError) as exc_info:
            await mock_provider.complete([], GenerationConfig())
        
        assert exc_info.value.retry_after == 60
    
    @pytest.mark.unit
    async def test_streaming_yields_chunks(self, mock_provider):
        """Verify streaming produces valid chunks."""
        # Arrange
        chunks = [
            {"content": "Hello", "finish_reason": None},
            {"content": " world", "finish_reason": "stop"}
        ]
        mock_provider.stream_complete.return_value = self._async_generator(chunks)
        
        # Act
        collected = []
        async for chunk in mock_provider.stream_complete([], GenerationConfig()):
            collected.append(chunk)
        
        # Assert
        assert len(collected) == 2
        assert "".join(c["content"] for c in collected) == "Hello world"
    
    @staticmethod
    async def _async_generator(items):
        for item in items:
            yield item
```

#### Pattern 2: State Machine Tests

```python
# tests/unit/test_agent_worker/test_agent_lifecycle.py
import pytest
from unittest.mock import Mock
from swarm_research.agent_worker import AgentWorker, AgentState


class TestAgentStateMachine:
    """Test agent lifecycle state transitions."""
    
    @pytest.fixture
    def agent(self):
        return AgentWorker(
            agent_id="test-agent-1",
            provider=Mock(),
            system_prompt="You are a test agent."
        )
    
    @pytest.mark.unit
    def test_initial_state_is_idle(self, agent):
        """Agent starts in IDLE state."""
        assert agent.state == AgentState.IDLE
    
    @pytest.mark.unit
    async def test_execute_transitions_to_executing(self, agent):
        """Execute transitions through proper states."""
        # Arrange
        task = Mock()
        
        # Act
        execution_task = asyncio.create_task(agent.execute(task))
        
        # Assert - during execution
        assert agent.state == AgentState.EXECUTING
        
        # Complete
        await execution_task
        assert agent.state == AgentState.IDLE
    
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_transition", [
        (AgentState.EXECUTING, AgentState.IDLE),  # Can't go directly to idle
        (AgentState.COMPLETED, AgentState.FAILED),  # Terminal states
        (AgentState.FAILED, AgentState.EXECUTING),  # Must reset first
    ])
    async def test_invalid_state_transitions_rejected(self, agent, invalid_transition):
        """Invalid state transitions are rejected."""
        from_state, to_state = invalid_transition
        agent._state = from_state
        
        with pytest.raises(InvalidStateTransitionError):
            await agent._transition_to(to_state)
    
    @pytest.mark.unit
    def test_all_states_reachable(self, agent):
        """Verify state machine coverage."""
        valid_transitions = {
            AgentState.IDLE: [AgentState.EXECUTING],
            AgentState.EXECUTING: [AgentState.COMPLETED, AgentState.FAILED, AgentState.YIELDED],
            AgentState.YIELDED: [AgentState.EXECUTING, AgentState.COMPLETED],
            AgentState.COMPLETED: [AgentState.IDLE],  # Reset
            AgentState.FAILED: [AgentState.IDLE],  # Reset
        }
        
        for state, allowed in valid_transitions.items():
            for next_state in allowed:
                assert agent._is_valid_transition(state, next_state)
```

#### Pattern 3: Task Scheduler Tests

```python
# tests/unit/test_orchestrator/test_task_scheduler.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from swarm_research.orchestrator import TaskScheduler, Task, TaskPriority


class TestTaskScheduler:
    """Test task scheduling and execution."""
    
    @pytest.fixture
    def scheduler(self):
        return TaskScheduler(max_concurrent=3)
    
    @pytest.mark.unit
    async def test_task_priority_ordering(self, scheduler):
        """Higher priority tasks execute first."""
        # Arrange
        execution_order = []
        
        async def tracked_task(name):
            execution_order.append(name)
            await asyncio.sleep(0.01)
        
        # Act - submit in reverse priority order
        tasks = [
            scheduler.submit(Task(id=f"low-{i}", priority=TaskPriority.LOW, 
                                 coro=tracked_task(f"low-{i}")))
            for i in range(3)
        ] + [
            scheduler.submit(Task(id=f"high-{i}", priority=TaskPriority.HIGH,
                                 coro=tracked_task(f"high-{i}")))
            for i in range(3)
        ]
        
        await asyncio.gather(*tasks)
        
        # Assert - high priority tasks executed first
        high_count = sum(1 for name in execution_order[:3] if name.startswith("high"))
        assert high_count == 3
    
    @pytest.mark.unit
    async def test_dependency_resolution(self, scheduler):
        """Tasks with dependencies execute in correct order."""
        # Arrange
        execution_order = []
        
        async def make_task(name, dependencies=None):
            async def task_fn():
                execution_order.append(name)
            return Task(id=name, coro=task_fn(), dependencies=dependencies or [])
        
        # Create dependency chain: C depends on B depends on A
        task_a = await make_task("A")
        task_b = await make_task("B", dependencies=["A"])
        task_c = await make_task("C", dependencies=["B"])
        
        # Act
        await asyncio.gather(
            scheduler.submit(task_c),
            scheduler.submit(task_a),
            scheduler.submit(task_b)
        )
        
        # Assert
        assert execution_order == ["A", "B", "C"]
    
    @pytest.mark.unit
    async def test_circular_dependency_detection(self, scheduler):
        """Circular dependencies are detected and rejected."""
        # Arrange - A -> B -> C -> A
        task_a = Task(id="A", coro=asyncio.sleep(0), dependencies=["C"])
        task_b = Task(id="B", coro=asyncio.sleep(0), dependencies=["A"])
        task_c = Task(id="C", coro=asyncio.sleep(0), dependencies=["B"])
        
        # Act & Assert
        with pytest.raises(CircularDependencyError):
            await scheduler.submit(task_a)
            await scheduler.submit(task_b)
            await scheduler.submit(task_c)
    
    @pytest.mark.unit
    async def test_concurrency_limit_enforcement(self, scheduler):
        """Concurrency limit is respected."""
        # Arrange
        active_count = 0
        max_active = 0
        
        async def tracking_task():
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.1)
            active_count -= 1
        
        # Act - submit more tasks than limit
        tasks = [
            scheduler.submit(Task(id=f"task-{i}", coro=tracking_task()))
            for i in range(10)
        ]
        
        await asyncio.gather(*tasks)
        
        # Assert
        assert max_active <= 3  # scheduler max_concurrent
```

### 1.4 Mocking Strategy

```python
# tests/unit/conftest.py - Shared fixtures
import pytest
from unittest.mock import Mock, AsyncMock
from contextlib import asynccontextmanager


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider."""
    provider = Mock()
    provider.complete = AsyncMock(return_value={
        "content": "Mock response",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20}
    })
    provider.stream_complete = AsyncMock()
    provider.health_check = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def mock_message_bus():
    """Create a mock message bus."""
    bus = Mock()
    bus.publish = AsyncMock()
    bus.subscribe = Mock(return_value=asynccontextmanager(lambda: (yield Mock())))
    bus.request = AsyncMock()
    return bus


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = Mock()
    store.add = AsyncMock()
    store.search = AsyncMock(return_value=[])
    store.delete = AsyncMock()
    return store


@pytest.fixture
def temp_state_dir(tmp_path):
    """Provide temporary directory for state tests."""
    return tmp_path / "state"


@pytest.fixture
async def clean_redis():
    """Provide isolated Redis instance for tests."""
    import redis.asyncio as redis
    client = redis.Redis(host="localhost", port=6379, db=15)  # Use DB 15 for tests
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.close()
```

---

## 2. Integration Testing

### 2.1 Integration Test Architecture

```
tests/
├── integration/
│   ├── __init__.py
│   ├── conftest.py                    # Integration test fixtures
│   ├── test_orchestrator_flow/        # End-to-end orchestration
│   │   ├── __init__.py
│   │   ├── test_research_workflow.py
│   │   ├── test_agent_spawning.py
│   │   └── test_branch_management.py
│   ├── test_communication_stack/      # Full communication tests
│   │   ├── __init__.py
│   │   ├── test_message_flow.py
│   │   ├── test_pub_sub.py
│   │   └── test_rpc_patterns.py
│   ├── test_provider_integration/     # Real provider tests
│   │   ├── __init__.py
│   │   ├── test_kimi_integration.py
│   │   ├── test_claude_integration.py
│   │   └── test_ollama_integration.py
│   └── test_persistence/              # State persistence tests
│       ├── __init__.py
│       ├── test_checkpoint_recovery.py
│       └── test_migration.py
```

### 2.2 Integration Test Patterns

#### Pattern 1: Orchestrator Flow Test

```python
# tests/integration/test_orchestrator_flow/test_research_workflow.py
import pytest
import asyncio
from swarm_research.orchestrator import autoconstitutionOrchestrator, ResearchContext


@pytest.mark.integration
@pytest.mark.slow
class TestResearchWorkflow:
    """Test complete research workflows."""
    
    @pytest.fixture
    async def orchestrator(self, mock_provider):
        """Create configured orchestrator."""
        context = ResearchContext(
            problem_statement="Find the optimal sorting algorithm for small arrays",
            domain="algorithms",
            max_depth=2,
            exploration_budget=10
        )
        orchestrator = autoconstitutionOrchestrator(
            context=context,
            max_concurrent_agents=3,
            provider=mock_provider
        )
        yield orchestrator
        await orchestrator.stop_research()
    
    async def test_simple_research_completes(self, orchestrator):
        """Basic research workflow completes successfully."""
        # Act
        final_state = await orchestrator.start_research()
        
        # Assert
        assert final_state.version > 0
        assert final_state.best_hypothesis is not None
        assert len(final_state.consolidated_findings) > 0
    
    async def test_agent_spawning_and_cleanup(self, orchestrator):
        """Agents are properly spawned and cleaned up."""
        # Act
        await orchestrator.start_research()
        
        # Assert
        assert len(orchestrator.agents) <= orchestrator.max_concurrent_agents
        
        # Cleanup
        await orchestrator.stop_research()
        assert all(a.state == AgentState.TERMINATED 
                  for a in orchestrator.agents.values())
    
    async def test_branch_convergence_detection(self, orchestrator):
        """Converging branches are detected and handled."""
        # Arrange - mock to simulate convergence
        for branch in orchestrator.branches.values():
            branch.gradient_score = 0.01  # Very low progress
        
        # Act
        await orchestrator.start_research()
        
        # Assert - stagnant branches should be reallocated
        stagnant_count = sum(
            1 for b in orchestrator.branches.values()
            if b.status == BranchStatus.STAGNANT
        )
        assert stagnant_count > 0
```

#### Pattern 2: Communication Stack Test

```python
# tests/integration/test_communication_stack/test_message_flow.py
import pytest
import asyncio
from swarm_research.communication import MessageBus, SwarmMessage, MessageType


@pytest.mark.integration
class TestMessageFlow:
    """Test message passing through the full stack."""
    
    @pytest.fixture
    async def message_bus(self):
        """Create real message bus instance."""
        bus = MessageBus(transport="redis", host="localhost", port=6379)
        await bus.connect()
        yield bus
        await bus.disconnect()
    
    async def test_pub_sub_message_delivery(self, message_bus):
        """Messages are delivered via pub/sub."""
        # Arrange
        received_messages = []
        
        async def subscriber():
            async with message_bus.subscribe("test.topic") as subscription:
                async for msg in subscription:
                    received_messages.append(msg)
                    if len(received_messages) >= 3:
                        break
        
        # Act
        sub_task = asyncio.create_task(subscriber())
        
        # Publish messages
        for i in range(3):
            await message_bus.publish(
                "test.topic",
                SwarmMessage(
                    message_type=MessageType.DIRECT_MESSAGE,
                    payload={"index": i}
                )
            )
        
        await asyncio.wait_for(sub_task, timeout=5.0)
        
        # Assert
        assert len(received_messages) == 3
        assert [m.payload["index"] for m in received_messages] == [0, 1, 2]
    
    async def test_rpc_request_response(self, message_bus):
        """RPC pattern works end-to-end."""
        # Arrange - set up RPC handler
        async def echo_handler(request):
            return {"echo": request.payload}
        
        await message_bus.register_rpc_handler("echo", echo_handler)
        
        # Act
        response = await message_bus.request(
            "echo",
            SwarmMessage(payload={"test": "data"}),
            timeout=5.0
        )
        
        # Assert
        assert response.payload["echo"]["test"] == "data"
    
    async def test_message_ordering_guarantees(self, message_bus):
        """Message ordering is preserved within a channel."""
        # Arrange
        messages = []
        
        async def ordered_consumer():
            async with message_bus.subscribe("ordered.channel") as sub:
                async for msg in sub:
                    messages.append(msg.payload["sequence"])
                    if len(messages) >= 100:
                        break
        
        # Act
        consumer_task = asyncio.create_task(ordered_consumer())
        
        # Publish in sequence
        for i in range(100):
            await message_bus.publish(
                "ordered.channel",
                SwarmMessage(payload={"sequence": i})
            )
        
        await asyncio.wait_for(consumer_task, timeout=10.0)
        
        # Assert - strict ordering
        assert messages == list(range(100))
```

#### Pattern 3: State Persistence Test

```python
# tests/integration/test_persistence/test_checkpoint_recovery.py
import pytest
import asyncio
from swarm_research.state import StateManager, SessionState


@pytest.mark.integration
class TestCheckpointRecovery:
    """Test state persistence and recovery."""
    
    @pytest.fixture
    async def state_manager(self, tmp_path):
        """Create state manager with temp storage."""
        manager = StateManager(storage_path=tmp_path / "checkpoints")
        await manager.initialize()
        yield manager
        await manager.shutdown()
    
    async def test_checkpoint_create_and_restore(self, state_manager):
        """State can be checkpointed and restored."""
        # Arrange
        original_state = SessionState()
        original_state.global_state.iteration = 42
        original_state.branches = {"branch-1": Mock()}
        
        # Act - checkpoint
        checkpoint_id = await state_manager.checkpoint(original_state)
        
        # Modify state
        original_state.global_state.iteration = 100
        
        # Restore
        restored_state = await state_manager.restore(checkpoint_id)
        
        # Assert
        assert restored_state.global_state.iteration == 42
        assert "branch-1" in restored_state.branches
    
    async def test_incremental_checkpointing(self, state_manager):
        """Incremental checkpoints save only changes."""
        # Arrange
        state = SessionState()
        
        # Act - create base checkpoint
        base_id = await state_manager.checkpoint(state, full=True)
        base_size = await state_manager.get_checkpoint_size(base_id)
        
        # Modify slightly and create incremental
        state.global_state.iteration += 1
        incremental_id = await state_manager.checkpoint(state, full=False)
        incremental_size = await state_manager.get_checkpoint_size(incremental_id)
        
        # Assert
        assert incremental_size < base_size
    
    async def test_recovery_after_crash_simulation(self, state_manager):
        """System recovers to consistent state after crash."""
        # Arrange - create state and simulate work
        state = SessionState()
        state.global_state.iteration = 50
        state.ratchet_state.best_score = 0.85
        
        # Create checkpoint
        await state_manager.checkpoint(state)
        
        # Simulate crash - create new manager instance
        new_manager = StateManager(storage_path=state_manager.storage_path)
        await new_manager.initialize()
        
        # Act - recover
        recovered_state = await new_manager.recover_latest()
        
        # Assert
        assert recovered_state.global_state.iteration == 50
        assert recovered_state.ratchet_state.best_score == 0.85
```

### 2.3 Test Containers for Integration Tests

```yaml
# tests/integration/docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
      POSTGRES_DB: swarm_test
    ports:
      - "5432:5432"
    volumes:
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
      
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    environment:
      - ALLOW_RESET=true
      
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
```

---

## 3. Property-Based Testing

### 3.1 Hypothesis Configuration

```python
# tests/property/conftest.py
import pytest
from hypothesis import settings, Verbosity

# Configure Hypothesis profiles
settings.register_profile("ci", max_examples=100, deadline=None)
settings.register_profile("dev", max_examples=50, deadline=None)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)

# Load profile from environment
settings.load_profile("ci")
```

### 3.2 Property Test Patterns

#### Pattern 1: Message Serialization Properties

```python
# tests/property/test_message_properties.py
import pytest
from hypothesis import given, strategies as st
from swarm_research.communication import SwarmMessage, MessageType
from datetime import datetime


class TestMessageProperties:
    """Property-based tests for message handling."""
    
    @given(
        message_type=st.sampled_from(MessageType),
        payload=st.dictionaries(st.text(), st.one_of(st.text(), st.integers())),
        priority=st.integers(min_value=1, max_value=10)
    )
    def test_message_roundtrip_serialization(self, message_type, payload, priority):
        """Any valid message serializes and deserializes correctly."""
        # Arrange
        original = SwarmMessage(
            message_type=message_type,
            payload=payload,
            priority=priority,
            timestamp=datetime.utcnow()
        )
        
        # Act
        serialized = original.to_json()
        deserialized = SwarmMessage.from_json(serialized)
        
        # Assert
        assert deserialized.message_type == original.message_type
        assert deserialized.payload == original.payload
        assert deserialized.priority == original.priority
    
    @given(
        messages=st.lists(
            st.builds(SwarmMessage,
                     message_type=st.sampled_from(MessageType),
                     payload=st.dictionaries(st.text(), st.text())),
            min_size=1, max_size=100
        )
    )
    def test_message_batch_processing(self, messages):
        """Batched messages maintain individual integrity."""
        # Act
        batch = SwarmMessage.create_batch(messages)
        recovered = SwarmMessage.from_batch(batch)
        
        # Assert
        assert len(recovered) == len(messages)
        for orig, recov in zip(messages, recovered):
            assert orig.message_type == recov.message_type
            assert orig.payload == recov.payload
    
    @given(
        content=st.text(min_size=1, max_size=10000),
        encoding=st.sampled_from(["utf-8", "ascii", "latin-1"])
    )
    def test_message_content_encoding(self, content, encoding):
        """Message content handles various encodings."""
        # Arrange
        message = SwarmMessage(
            message_type=MessageType.DIRECT_MESSAGE,
            payload={"content": content}
        )
        
        # Act & Assert - should not raise
        try:
            serialized = message.to_json()
            deserialized = SwarmMessage.from_json(serialized)
            assert deserialized.payload["content"] == content
        except UnicodeError:
            # Some encodings may not support all characters - that's OK
            pass
```

#### Pattern 2: State Transition Properties

```python
# tests/property/test_state_properties.py
import pytest
from hypothesis import given, strategies as st, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition
from swarm_research.state import SessionState, AgentState, BranchState


class TestStateTransitions(RuleBasedStateMachine):
    """Stateful property testing for state transitions."""
    
    def __init__(self):
        super().__init__()
        self.state = SessionState()
        self.active_agents = set()
        self.active_branches = set()
    
    @rule(agent_id=st.uuids())
    def create_agent(self, agent_id):
        """Creating an agent adds it to the state."""
        if agent_id not in self.state.agents:
            self.state.agents[agent_id] = AgentState(agent_id=agent_id)
            self.active_agents.add(agent_id)
    
    @rule(agent_id=st.uuids())
    @precondition(lambda self: len(self.active_agents) > 0)
    def activate_agent(self, agent_id):
        """Activating an agent changes its state."""
        assume(agent_id in self.active_agents)
        agent = self.state.agents[agent_id]
        old_state = agent.status
        agent.status = AgentState.ACTIVE
        assert agent.status != old_state or old_state == AgentState.ACTIVE
    
    @rule(agent_id=st.uuids())
    @precondition(lambda self: len(self.active_agents) > 0)
    def terminate_agent(self, agent_id):
        """Terminating an agent removes it from active set."""
        assume(agent_id in self.active_agents)
        self.state.agents[agent_id].status = AgentState.TERMINATED
        self.active_agents.discard(agent_id)
    
    @rule()
    def invariant_agent_count_matches(self):
        """Active agent count is consistent."""
        active_in_state = sum(
            1 for a in self.state.agents.values()
            if a.status != AgentState.TERMINATED
        )
        assert active_in_state == len(self.active_agents)
    
    @rule()
    def invariant_ratchet_never_decreases(self):
        """Ratchet state never decreases."""
        if not hasattr(self, '_last_ratchet'):
            self._last_ratchet = self.state.ratchet_state.best_score
        else:
            assert self.state.ratchet_state.best_score >= self._last_ratchet
            self._last_ratchet = self.state.ratchet_state.best_score


TestStateTransitionsTest = TestStateTransitions.TestCase
```

#### Pattern 3: Task Scheduling Properties

```python
# tests/property/test_scheduler_properties.py
import pytest
from hypothesis import given, strategies as st
from swarm_research.orchestrator import TaskScheduler, Task, TaskPriority
import asyncio


class TestSchedulerProperties:
    """Property-based tests for task scheduling."""
    
    @given(
        priorities=st.lists(
            st.sampled_from(TaskPriority),
            min_size=1, max_size=50
        )
    )
    async def test_priority_ordering_maintained(self, priorities):
        """Higher priority tasks are always scheduled before lower priority."""
        # Arrange
        scheduler = TaskScheduler(max_concurrent=1)
        execution_order = []
        
        async def tracking_task(name):
            execution_order.append(name)
        
        # Create tasks with varying priorities
        tasks = []
        for i, priority in enumerate(priorities):
            task = Task(
                id=f"task-{i}",
                priority=priority,
                coro=tracking_task(f"task-{i}-{priority.name}")
            )
            tasks.append(scheduler.submit(task))
        
        # Act
        await asyncio.gather(*tasks)
        
        # Assert - verify ordering
        priority_values = {p: i for i, p in enumerate(TaskPriority)}
        for i in range(len(execution_order) - 1):
            current_priority = priority_values[
                TaskPriority[execution_order[i].split("-")[-1]]
            ]
            next_priority = priority_values[
                TaskPriority[execution_order[i + 1].split("-")[-1]]
            ]
            assert current_priority <= next_priority
    
    @given(
        task_count=st.integers(min_value=1, max_value=20),
        max_concurrent=st.integers(min_value=1, max_value=10)
    )
    async def test_concurrency_limit_never_exceeded(self, task_count, max_concurrent):
        """Concurrency limit is never exceeded regardless of task count."""
        assume(task_count > max_concurrent)
        
        # Arrange
        scheduler = TaskScheduler(max_concurrent=max_concurrent)
        active_count = 0
        max_observed = 0
        
        async def tracking_task():
            nonlocal active_count, max_observed
            active_count += 1
            max_observed = max(max_observed, active_count)
            await asyncio.sleep(0.01)
            active_count -= 1
        
        # Act
        tasks = [
            scheduler.submit(Task(id=f"task-{i}", coro=tracking_task()))
            for i in range(task_count)
        ]
        await asyncio.gather(*tasks)
        
        # Assert
        assert max_observed <= max_concurrent
```

### 3.3 Custom Hypothesis Strategies

```python
# tests/property/strategies.py
from hypothesis import strategies as st
from swarm_research.communication import SwarmMessage, MessageType, AgentAddress
from swarm_research.state import SessionState, AgentState, BranchState
import uuid
from datetime import datetime


# Custom strategies for autoconstitution

def agent_addresses():
    """Strategy for generating valid agent addresses."""
    return st.builds(
        AgentAddress,
        agent_id=st.uuids(),
        agent_type=st.sampled_from(["researcher", "critic", "synthesizer", "verifier"]),
        node_id=st.sampled_from(["node-1", "node-2", "node-3"]),
        capabilities=st.lists(st.text(), max_size=5)
    )


def swarm_messages():
    """Strategy for generating valid swarm messages."""
    return st.builds(
        SwarmMessage,
        message_id=st.uuids(),
        correlation_id=st.uuids(),
        parent_id=st.one_of(st.none(), st.uuids()),
        timestamp=st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2030, 12, 31)
        ),
        source=agent_addresses(),
        message_type=st.sampled_from(MessageType),
        payload=st.dictionaries(st.text(), st.one_of(st.text(), st.integers(), st.floats())),
        priority=st.integers(min_value=1, max_value=10)
    )


def session_states():
    """Strategy for generating valid session states."""
    return st.builds(
        SessionState,
        session_id=st.uuids(),
        global_state=st.fixed_dictionaries({
            "iteration": st.integers(min_value=0),
            "status": st.sampled_from(["running", "paused", "completed"])
        }),
        branches=st.dictionaries(
            st.uuids(),
            st.fixed_dictionaries({
                "status": st.sampled_from(["exploring", "converging", "stagnant"]),
                "depth": st.integers(min_value=0, max_value=10)
            })
        )
    )


# Register strategies for common types
st.register_type_strategy(AgentAddress, agent_addresses())
st.register_type_strategy(SwarmMessage, swarm_messages())
```

---

## 4. Simulation Testing

### 4.1 Simulation Framework Architecture

```
tests/
├── simulation/
│   ├── __init__.py
│   ├── scenarios/                     # Predefined test scenarios
│   │   ├── __init__.py
│   │   ├── base_scenario.py
│   │   ├── research_scenarios.py
│   │   ├── failure_scenarios.py
│   │   └── load_scenarios.py
│   ├── fixtures/                      # Simulation fixtures
│   │   ├── __init__.py
│   │   ├── mock_providers.py
│   │   └── workload_generators.py
│   └── test_simulations.py            # Main simulation tests
```

### 4.2 Simulation Scenarios

#### Scenario 1: Research Workflow Simulation

```python
# tests/simulation/scenarios/research_scenarios.py
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime
import random


@dataclass
class ResearchScenario:
    """Defines a research simulation scenario."""
    name: str
    problem_statement: str
    expected_duration: float  # seconds
    agent_count: int
    failure_rate: float  # 0-1
    network_latency_ms: float
    provider_response_time_ms: float


class ResearchScenarios:
    """Collection of realistic research scenarios."""
    
    SIMPLE_MATH = ResearchScenario(
        name="simple_math",
        problem_statement="Find the sum of all primes below 1000",
        expected_duration=30.0,
        agent_count=3,
        failure_rate=0.0,
        network_latency_ms=1.0,
        provider_response_time_ms=500.0
    )
    
    ALGORITHM_DESIGN = ResearchScenario(
        name="algorithm_design",
        problem_statement="Design an O(n log n) sorting algorithm for linked lists",
        expected_duration=120.0,
        agent_count=5,
        failure_rate=0.05,
        network_latency_ms=5.0,
        provider_response_time_ms=1000.0
    )
    
    COMPLEX_RESEARCH = ResearchScenario(
        name="complex_research",
        problem_statement=(
            "Investigate the relationship between transformer attention patterns "
            "and circuit complexity in language models"
        ),
        expected_duration=600.0,
        agent_count=10,
        failure_rate=0.1,
        network_latency_ms=10.0,
        provider_response_time_ms=2000.0
    )
    
    STRESS_TEST = ResearchScenario(
        name="stress_test",
        problem_statement="Optimize hyperparameters for a neural network",
        expected_duration=300.0,
        agent_count=50,
        failure_rate=0.2,
        network_latency_ms=50.0,
        provider_response_time_ms=3000.0
    )
```

#### Scenario 2: Failure Injection

```python
# tests/simulation/scenarios/failure_scenarios.py
import asyncio
from dataclasses import dataclass
from typing import Callable, Optional
import random


@dataclass
class FailureInjection:
    """Defines when and how to inject failures."""
    failure_type: str  # "agent_crash", "network_partition", "provider_timeout"
    trigger_condition: Callable[[], bool]
    recovery_time_seconds: float
    affected_agents: Optional[list] = None


class FailureScenarios:
    """Failure injection scenarios for resilience testing."""
    
    @staticmethod
    def random_agent_crash(crash_probability: float = 0.1):
        """Random agent crashes during execution."""
        return FailureInjection(
            failure_type="agent_crash",
            trigger_condition=lambda: random.random() < crash_probability,
            recovery_time_seconds=5.0
        )
    
    @staticmethod
    def network_partition(partition_duration_seconds: float = 10.0):
        """Simulate network partition between agent groups."""
        return FailureInjection(
            failure_type="network_partition",
            trigger_condition=lambda: random.random() < 0.05,
            recovery_time_seconds=partition_duration_seconds
        )
    
    @staticmethod
    def provider_degradation(degradation_pattern: str = "gradual"):
        """Simulate LLM provider performance degradation."""
        patterns = {
            "gradual": lambda: random.random() < 0.1,
            "sudden": lambda: random.random() < 0.01,
            "intermittent": lambda: random.random() < 0.3
        }
        return FailureInjection(
            failure_type="provider_timeout",
            trigger_condition=patterns.get(degradation_pattern, patterns["gradual"]),
            recovery_time_seconds=30.0
        )
    
    @staticmethod
    def cascading_failure(seed_agents: int = 2):
        """Simulate cascading failures starting from seed agents."""
        return FailureInjection(
            failure_type="cascading_failure",
            trigger_condition=lambda: random.random() < 0.05,
            recovery_time_seconds=60.0,
            affected_agents=list(range(seed_agents))
        )
```

### 4.3 Simulation Test Implementation

```python
# tests/simulation/test_simulations.py
import pytest
import asyncio
from datetime import datetime
from typing import List, Dict
import statistics

from swarm_research.simulation import SimulationRunner
from swarm_research.simulation.scenarios import (
    ResearchScenarios, FailureScenarios
)


@pytest.mark.simulation
@pytest.mark.slow
class TestResearchSimulations:
    """Simulation tests for realistic research scenarios."""
    
    @pytest.fixture
    async def simulation_runner(self):
        """Create simulation runner."""
        runner = SimulationRunner()
        await runner.initialize()
        yield runner
        await runner.shutdown()
    
    async def test_simple_math_scenario(self, simulation_runner):
        """Simple math problem completes within expected time."""
        # Arrange
        scenario = ResearchScenarios.SIMPLE_MATH
        
        # Act
        result = await simulation_runner.run_scenario(scenario)
        
        # Assert
        assert result.completed
        assert result.duration_seconds < scenario.expected_duration * 1.5
        assert result.solution_correct
        assert result.agents_spawned == scenario.agent_count
    
    async def test_algorithm_design_with_failures(self, simulation_runner):
        """Algorithm design handles agent failures gracefully."""
        # Arrange
        scenario = ResearchScenarios.ALGORITHM_DESIGN
        failure = FailureScenarios.random_agent_crash(crash_probability=0.1)
        
        # Act
        result = await simulation_runner.run_scenario(
            scenario,
            failure_injections=[failure]
        )
        
        # Assert
        assert result.completed
        assert result.agents_restarted > 0  # Failures occurred
        assert result.duration_seconds < scenario.expected_duration * 2.0
    
    async def test_stress_test_scalability(self, simulation_runner):
        """System scales under high agent count."""
        # Arrange
        scenario = ResearchScenarios.STRESS_TEST
        
        # Act
        metrics = []
        for agent_count in [10, 25, 50]:
            scenario.agent_count = agent_count
            result = await simulation_runner.run_scenario(scenario)
            metrics.append({
                "agent_count": agent_count,
                "duration": result.duration_seconds,
                "throughput": result.tasks_completed / result.duration_seconds
            })
        
        # Assert - throughput should scale sub-linearly but positively
        throughputs = [m["throughput"] for m in metrics]
        assert throughputs[-1] > throughputs[0]  # More agents = more throughput
        
        # Efficiency should decrease but not collapse
        efficiency_ratios = [
            (m["throughput"] / m["agent_count"]) for m in metrics
        ]
        assert efficiency_ratios[-1] > efficiency_ratios[0] * 0.3  # At least 30% efficiency
    
    async def test_network_partition_recovery(self, simulation_runner):
        """System recovers from network partitions."""
        # Arrange
        scenario = ResearchScenarios.ALGORITHM_DESIGN
        partition = FailureScenarios.network_partition(partition_duration_seconds=5.0)
        
        # Act
        result = await simulation_runner.run_scenario(
            scenario,
            failure_injections=[partition]
        )
        
        # Assert
        assert result.completed
        assert result.partition_events > 0
        assert result.recovery_time_seconds < 30.0
    
    async def test_provider_failover(self, simulation_runner):
        """System fails over when providers fail."""
        # Arrange
        scenario = ResearchScenarios.COMPLEX_RESEARCH
        provider_failure = FailureScenarios.provider_degradation("sudden")
        
        # Configure multiple providers
        simulation_runner.configure_providers([
            {"name": "primary", "failure_rate": 0.3},
            {"name": "backup", "failure_rate": 0.0}
        ])
        
        # Act
        result = await simulation_runner.run_scenario(
            scenario,
            failure_injections=[provider_failure]
        )
        
        # Assert
        assert result.completed
        assert result.provider_failovers > 0
        assert result.backup_provider_usage > 0


class SimulationMetrics:
    """Collect and analyze simulation metrics."""
    
    def __init__(self):
        self.runs: List[Dict] = []
    
    def add_run(self, result):
        """Add a simulation run result."""
        self.runs.append({
            "timestamp": datetime.utcnow(),
            "duration": result.duration_seconds,
            "agents": result.agents_spawned,
            "tasks": result.tasks_completed,
            "failures": result.failure_count,
            "success": result.completed
        })
    
    def get_statistics(self) -> Dict:
        """Compute aggregate statistics."""
        if not self.runs:
            return {}
        
        durations = [r["duration"] for r in self.runs]
        success_rate = sum(r["success"] for r in self.runs) / len(self.runs)
        
        return {
            "total_runs": len(self.runs),
            "success_rate": success_rate,
            "mean_duration": statistics.mean(durations),
            "std_duration": statistics.stdev(durations) if len(durations) > 1 else 0,
            "min_duration": min(durations),
            "max_duration": max(durations)
        }
```

### 4.4 Workload Generation

```python
# tests/simulation/fixtures/workload_generators.py
import random
from typing import Iterator, Dict, Any
from dataclasses import dataclass


@dataclass
class WorkloadPattern:
    """Defines a workload pattern for simulation."""
    name: str
    arrival_rate: float  # tasks per second
    burstiness: float  # 0=steady, 1=very bursty
    task_complexity: str  # "simple", "medium", "complex"


class WorkloadGenerators:
    """Generate realistic workloads for testing."""
    
    @staticmethod
    def steady_load(tasks_per_second: float = 1.0) -> WorkloadPattern:
        """Steady, predictable workload."""
        return WorkloadPattern(
            name="steady",
            arrival_rate=tasks_per_second,
            burstiness=0.1,
            task_complexity="medium"
        )
    
    @staticmethod
    def burst_load(base_rate: float = 0.5, burst_multiplier: float = 10.0) -> WorkloadPattern:
        """Workload with periodic bursts."""
        return WorkloadPattern(
            name="burst",
            arrival_rate=base_rate * burst_multiplier,
            burstiness=0.8,
            task_complexity="simple"
        )
    
    @staticmethod
    def diurnal_pattern(peak_hour: int = 14) -> WorkloadPattern:
        """Workload following daily patterns."""
        return WorkloadPattern(
            name="diurnal",
            arrival_rate=2.0,  # Base rate
            burstiness=0.5,
            task_complexity="mixed"
        )
    
    @classmethod
    def generate_tasks(cls, pattern: WorkloadPattern, duration_seconds: float) -> Iterator[Dict]:
        """Generate tasks according to pattern."""
        import time
        start_time = time.time()
        task_id = 0
        
        while time.time() - start_time < duration_seconds:
            # Calculate inter-arrival time
            base_interval = 1.0 / pattern.arrival_rate
            
            # Add burstiness
            if pattern.burstiness > 0:
                if random.random() < pattern.burstiness:
                    base_interval *= random.uniform(0.1, 0.5)  # Burst
                else:
                    base_interval *= random.uniform(0.8, 1.2)  # Normal
            
            yield {
                "id": f"task-{task_id}",
                "complexity": pattern.task_complexity,
                "timestamp": time.time()
            }
            task_id += 1
            
            time.sleep(max(0, base_interval))
```

---

## 5. Benchmark Testing

### 5.1 Benchmark Framework

```
tests/
├── benchmark/
│   ├── __init__.py
│   ├── conftest.py                    # Benchmark fixtures
│   ├── benchmarks/                    # Benchmark definitions
│   │   ├── __init__.py
│   │   ├── test_provider_latency.py
│   │   ├── test_throughput.py
│   │   ├── test_scalability.py
│   │   └── test_efficiency.py
│   ├── baselines/                     # Baseline measurements
│   │   ├── __init__.py
│   │   └── karpathy_baseline.py
│   └── results/                       # Benchmark results storage
│       └── .gitkeep
```

### 5.2 Benchmark Test Implementation

#### Benchmark 1: Provider Latency

```python
# tests/benchmark/benchmarks/test_provider_latency.py
import pytest
import asyncio
import time
from typing import List, Dict
import statistics

from swarm_research.providers import ProviderRegistry
from swarm_research.benchmark import BenchmarkReporter


@pytest.mark.benchmark
class TestProviderLatency:
    """Benchmark LLM provider latency characteristics."""
    
    @pytest.fixture
    def reporter(self):
        return BenchmarkReporter("provider_latency")
    
    @pytest.mark.parametrize("provider_name", ["kimi", "claude", "openai"])
    async def test_provider_p50_latency(self, reporter, provider_name):
        """Measure median (p50) latency for each provider."""
        # Arrange
        provider = ProviderRegistry.get(provider_name)
        messages = [{"role": "user", "content": "What is 2+2?"}]
        latencies = []
        
        # Warmup
        for _ in range(3):
            await provider.complete(messages)
        
        # Measure
        for _ in range(50):
            start = time.perf_counter()
            await provider.complete(messages)
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
        
        # Report
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        
        reporter.record({
            "provider": provider_name,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "std_ms": statistics.stdev(latencies)
        })
        
        # Assert reasonable performance
        assert p50 < 5000, f"{provider_name} p50 latency too high: {p50}ms"
        assert p95 < 10000, f"{provider_name} p95 latency too high: {p95}ms"
    
    async def test_concurrent_request_scaling(self, reporter):
        """Measure latency under concurrent load."""
        provider = ProviderRegistry.get("kimi")
        messages = [{"role": "user", "content": "Hello"}]
        
        results = []
        for concurrency in [1, 5, 10, 20]:
            latencies = []
            
            async def make_request():
                start = time.perf_counter()
                await provider.complete(messages)
                return (time.perf_counter() - start) * 1000
            
            # Run concurrent requests
            tasks = [make_request() for _ in range(concurrency)]
            latencies = await asyncio.gather(*tasks)
            
            results.append({
                "concurrency": concurrency,
                "p50_ms": statistics.median(latencies),
                "throughput_rps": concurrency / (sum(latencies) / len(latencies) / 1000)
            })
        
        reporter.record({"concurrent_scaling": results})
        
        # Assert throughput scales reasonably
        base_throughput = results[0]["throughput_rps"]
        max_throughput = results[-1]["throughput_rps"]
        assert max_throughput > base_throughput * 2  # At least 2x scaling
```

#### Benchmark 2: System Throughput

```python
# tests/benchmark/benchmarks/test_throughput.py
import pytest
import asyncio
import time
from swarm_research.orchestrator import autoconstitutionOrchestrator
from swarm_research.benchmark import BenchmarkReporter


@pytest.mark.benchmark
class TestSystemThroughput:
    """Benchmark overall system throughput."""
    
    @pytest.fixture
    def reporter(self):
        return BenchmarkReporter("system_throughput")
    
    async def test_tasks_per_second(self, reporter):
        """Measure task completion rate."""
        # Arrange
        orchestrator = autoconstitutionOrchestrator(max_concurrent_agents=10)
        await orchestrator.initialize()
        
        task_count = 100
        start_time = time.perf_counter()
        
        # Act - submit and complete tasks
        tasks = [
            orchestrator.submit_simple_task(f"Task {i}")
            for i in range(task_count)
        ]
        await asyncio.gather(*tasks)
        
        duration = time.perf_counter() - start_time
        throughput = task_count / duration
        
        reporter.record({
            "tasks_completed": task_count,
            "duration_seconds": duration,
            "throughput_tps": throughput,
            "agents_used": 10
        })
        
        # Assert
        assert throughput > 1.0, f"Throughput too low: {throughput} tasks/sec"
    
    async def test_message_throughput(self, reporter):
        """Measure message processing rate."""
        from swarm_research.communication import MessageBus
        
        bus = MessageBus()
        await bus.connect()
        
        message_count = 10000
        received = 0
        
        async def subscriber():
            nonlocal received
            async with bus.subscribe("benchmark") as sub:
                async for _ in sub:
                    received += 1
                    if received >= message_count:
                        break
        
        # Start subscriber
        sub_task = asyncio.create_task(subscriber())
        
        # Publish messages
        start_time = time.perf_counter()
        for i in range(message_count):
            await bus.publish("benchmark", {"index": i})
        
        await asyncio.wait_for(sub_task, timeout=30.0)
        duration = time.perf_counter() - start_time
        
        throughput = message_count / duration
        reporter.record({
            "messages_processed": message_count,
            "duration_seconds": duration,
            "throughput_mps": throughput
        })
        
        assert throughput > 1000, f"Message throughput too low: {throughput} msg/sec"
```

#### Benchmark 3: Scalability

```python
# tests/benchmark/benchmarks/test_scalability.py
import pytest
import asyncio
import time
import statistics
from swarm_research.orchestrator import autoconstitutionOrchestrator
from swarm_research.benchmark import BenchmarkReporter


@pytest.mark.benchmark
class TestScalability:
    """Benchmark system scalability characteristics."""
    
    @pytest.fixture
    def reporter(self):
        return BenchmarkReporter("scalability")
    
    async def test_agent_count_scaling(self, reporter):
        """Measure performance with varying agent counts."""
        results = []
        
        for agent_count in [1, 2, 5, 10, 20, 50]:
            orchestrator = autoconstitutionOrchestrator(
                max_concurrent_agents=agent_count
            )
            await orchestrator.initialize()
            
            # Run workload
            task_count = min(100, agent_count * 10)
            start = time.perf_counter()
            
            tasks = [
                orchestrator.submit_simple_task(f"Task {i}")
                for i in range(task_count)
            ]
            await asyncio.gather(*tasks)
            
            duration = time.perf_counter() - start
            throughput = task_count / duration
            
            results.append({
                "agent_count": agent_count,
                "task_count": task_count,
                "duration_seconds": duration,
                "throughput_tps": throughput,
                "efficiency": throughput / agent_count
            })
            
            await orchestrator.shutdown()
        
        reporter.record({"scaling_results": results})
        
        # Analyze scaling efficiency
        base_throughput = results[0]["throughput_tps"]
        max_throughput = results[-1]["throughput_tps"]
        scaling_factor = max_throughput / base_throughput
        
        # Should scale at least 10x with 50x agents (20% efficiency)
        assert scaling_factor > 10, f"Poor scaling: {scaling_factor}x with 50x agents"
    
    async def test_memory_scaling(self, reporter):
        """Measure memory usage with increasing agents."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        results = []
        
        for agent_count in [1, 10, 50, 100]:
            # Measure baseline
            baseline_mem = process.memory_info().rss / 1024 / 1024  # MB
            
            orchestrator = autoconstitutionOrchestrator(
                max_concurrent_agents=agent_count
            )
            await orchestrator.initialize()
            
            # Measure with agents
            peak_mem = process.memory_info().rss / 1024 / 1024  # MB
            memory_per_agent = (peak_mem - baseline_mem) / agent_count
            
            results.append({
                "agent_count": agent_count,
                "baseline_mb": baseline_mem,
                "peak_mb": peak_mem,
                "memory_per_agent_mb": memory_per_agent
            })
            
            await orchestrator.shutdown()
        
        reporter.record({"memory_scaling": results})
        
        # Memory per agent should be reasonable (< 50MB)
        avg_memory_per_agent = statistics.mean(r["memory_per_agent_mb"] for r in results[1:])
        assert avg_memory_per_agent < 50, f"Memory per agent too high: {avg_memory_per_agent}MB"
```

### 5.3 Baseline Comparison

```python
# tests/benchmark/baselines/karpathy_baseline.py
import pytest
import time
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class BaselineResult:
    """Result from baseline system."""
    problem: str
    duration_seconds: float
    solution_correct: bool
    solution_quality: float  # 0-1
    tokens_used: int


class KarpathyBaseline:
    """
    Simulated baseline matching Karpathy's single-agent approach.
    Used for comparison benchmarking.
    """
    
    def __init__(self):
        self.temperature = 0.7
        self.max_tokens = 4096
    
    async def solve(self, problem: str) -> BaselineResult:
        """Solve a problem using single-agent approach."""
        start = time.perf_counter()
        
        # Simulate single-agent reasoning
        await self._chain_of_thought(problem)
        solution = await self._generate_solution(problem)
        
        duration = time.perf_counter() - start
        
        return BaselineResult(
            problem=problem,
            duration_seconds=duration,
            solution_correct=self._evaluate_correctness(solution),
            solution_quality=self._evaluate_quality(solution),
            tokens_used=self._estimate_tokens(problem, solution)
        )
    
    async def _chain_of_thought(self, problem: str):
        """Simulate chain-of-thought reasoning."""
        # Baseline: single pass, no parallel exploration
        await asyncio.sleep(0.5)  # Simulated thinking time
    
    async def _generate_solution(self, problem: str) -> str:
        """Generate solution."""
        await asyncio.sleep(1.0)  # Simulated generation time
        return f"Solution for: {problem}"
    
    def _evaluate_correctness(self, solution: str) -> bool:
        """Evaluate solution correctness."""
        return True  # Simulated
    
    def _evaluate_quality(self, solution: str) -> float:
        """Evaluate solution quality (0-1)."""
        return 0.75  # Simulated baseline quality
    
    def _estimate_tokens(self, problem: str, solution: str) -> int:
        """Estimate token usage."""
        return len(problem.split()) + len(solution.split())


@pytest.mark.benchmark
class TestBaselineComparison:
    """Compare autoconstitution against Karpathy baseline."""
    
    @pytest.fixture
    def baseline(self):
        return KarpathyBaseline()
    
    @pytest.fixture
    def reporter(self):
        from swarm_research.benchmark import BenchmarkReporter
        return BenchmarkReporter("baseline_comparison")
    
    async def test_time_to_solution_comparison(self, baseline, reporter):
        """Compare time-to-solution against baseline."""
        problems = [
            "Find the maximum subarray sum",
            "Implement a LRU cache",
            "Design a rate limiter"
        ]
        
        results = []
        for problem in problems:
            # Baseline
            baseline_result = await baseline.solve(problem)
            
            # autoconstitution
            swarm_result = await self._run_swarm(problem)
            
            improvement = (
                (baseline_result.duration_seconds - swarm_result.duration_seconds)
                / baseline_result.duration_seconds * 100
            )
            
            results.append({
                "problem": problem,
                "baseline_seconds": baseline_result.duration_seconds,
                "swarm_seconds": swarm_result.duration_seconds,
                "improvement_percent": improvement
            })
        
        reporter.record({"time_comparison": results})
        
        # Assert autoconstitution is faster
        avg_improvement = statistics.mean(r["improvement_percent"] for r in results)
        assert avg_improvement > 20, f"Not enough improvement: {avg_improvement}%"
    
    async def _run_swarm(self, problem: str):
        """Run problem through autoconstitution."""
        orchestrator = autoconstitutionOrchestrator(max_concurrent_agents=5)
        await orchestrator.initialize()
        
        start = time.perf_counter()
        result = await orchestrator.research(problem)
        duration = time.perf_counter() - start
        
        await orchestrator.shutdown()
        
        return type('obj', (object,), {
            'duration_seconds': duration,
            'solution_correct': True,
            'solution_quality': 0.85
        })()
```

### 5.4 Benchmark Reporting

```python
# tests/benchmark/conftest.py
import pytest
import json
from datetime import datetime
from pathlib import Path


class BenchmarkReporter:
    """Report and store benchmark results."""
    
    RESULTS_DIR = Path(__file__).parent / "results"
    
    def __init__(self, benchmark_name: str):
        self.benchmark_name = benchmark_name
        self.results = {
            "benchmark": benchmark_name,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {}
        }
    
    def record(self, data: dict):
        """Record benchmark data."""
        self.results["data"].update(data)
    
    def save(self):
        """Save results to file."""
        self.RESULTS_DIR.mkdir(exist_ok=True)
        
        filename = f"{self.benchmark_name}_{datetime.utcnow():%Y%m%d_%H%M%S}.json"
        filepath = self.RESULTS_DIR / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return filepath


@pytest.fixture(scope="session", autouse=True)
def benchmark_reporting():
    """Setup benchmark reporting."""
    yield
    
    # Generate summary report after all benchmarks
    reporter = BenchmarkReporter("summary")
    
    # Collect all results
    all_results = []
    for result_file in BenchmarkReporter.RESULTS_DIR.glob("*.json"):
        with open(result_file) as f:
            all_results.append(json.load(f))
    
    # Generate comparison with previous runs
    summary = {
        "total_benchmarks": len(all_results),
        "timestamp": datetime.utcnow().isoformat(),
        "results": all_results
    }
    
    summary_path = BenchmarkReporter.RESULTS_DIR / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
```

---

## 6. Test Execution Strategy

### 6.1 CI/CD Pipeline Integration

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e ".[test]"
      - run: pytest -m unit --cov --cov-report=xml
      - uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7
        ports: ['6379:6379']
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
        ports: ['5432:5432']
    steps:
      - uses: actions/checkout@v3
      - run: pip install -e ".[test]"
      - run: pytest -m integration --tb=short

  property-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -e ".[test]"
      - run: pytest -m property --hypothesis-profile=ci

  simulation-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - run: pip install -e ".[test]"
      - run: pytest -m simulation --tb=short -x

  benchmark-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - run: pip install -e ".[test]"
      - run: pytest -m benchmark --tb=short
      - uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: tests/benchmark/results/
```

### 6.2 Local Development Commands

```bash
# Run all unit tests
pytest -m unit -v

# Run with coverage
pytest -m unit --cov=swarm_research --cov-report=html

# Run specific test category
pytest -m integration -v
pytest -m property --hypothesis-profile=dev
pytest -m simulation -x
pytest -m benchmark

# Run tests matching pattern
pytest -k "test_provider" -v

# Run with debugging
pytest -m unit --pdb -x

# Parallel execution
pytest -m unit -n auto
```

### 6.3 Test Quality Metrics

```python
# tests/quality_metrics.py
"""Track and report test quality metrics."""

from pathlib import Path
import ast
import subprocess


def calculate_test_quality():
    """Calculate comprehensive test quality metrics."""
    
    metrics = {
        "coverage": {
            "line_coverage": get_line_coverage(),
            "branch_coverage": get_branch_coverage(),
            "critical_path_coverage": get_critical_path_coverage()
        },
        "test_characteristics": {
            "total_tests": count_tests(),
            "tests_per_module": get_tests_per_module(),
            "assertion_density": get_assertion_density(),
            "mock_usage_ratio": get_mock_usage_ratio()
        },
        "execution_metrics": {
            "avg_test_duration": get_avg_test_duration(),
            "slowest_tests": get_slowest_tests(10),
            "flaky_tests": get_flaky_tests()
        },
        "maintainability": {
            "test_code_ratio": get_test_code_ratio(),
            "duplicate_test_patterns": find_duplicate_patterns(),
            "orphaned_tests": find_orphaned_tests()
        }
    }
    
    return metrics


def get_line_coverage():
    """Get line coverage percentage."""
    result = subprocess.run(
        ["pytest", "--cov=swarm_research", "--cov-report=term"],
        capture_output=True, text=True
    )
    # Parse coverage from output
    for line in result.stdout.split('\n'):
        if 'TOTAL' in line:
            return float(line.split()[-1].rstrip('%'))
    return 0.0


def count_tests():
    """Count total number of tests."""
    result = subprocess.run(
        ["pytest", "--collect-only", "-q"],
        capture_output=True, text=True
    )
    return len([l for l in result.stdout.split('\n') if '::' in l])


def get_assertion_density():
    """Calculate average assertions per test."""
    test_files = Path("tests").rglob("test_*.py")
    
    total_assertions = 0
    total_tests = 0
    
    for file in test_files:
        with open(file) as f:
            tree = ast.parse(f.read())
            
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                total_tests += 1
                total_assertions += sum(
                    1 for n in ast.walk(node)
                    if isinstance(n, ast.Assert) or 
                    (isinstance(n, ast.Call) and 
                     isinstance(n.func, ast.Attribute) and
                     n.func.attr in ['assertEqual', 'assertTrue', 'assertRaises'])
                )
    
    return total_assertions / total_tests if total_tests > 0 else 0


if __name__ == "__main__":
    import json
    metrics = calculate_test_quality()
    print(json.dumps(metrics, indent=2))
```

---

## 7. Summary

### Testing Strategy Overview

| Layer | Coverage Target | Tools | Execution Frequency |
|-------|----------------|-------|---------------------|
| Unit Tests | 90% line coverage | pytest, unittest.mock | Every commit |
| Integration Tests | 100% critical paths | pytest, Docker | Every PR |
| Property Tests | Edge case discovery | Hypothesis | Every PR |
| Simulation Tests | Realistic scenarios | Custom framework | Daily |
| Benchmark Tests | Performance baselines | pytest-benchmark | Weekly |

### Key Testing Principles

1. **Test at the Right Level**: Unit tests for logic, integration for interactions, simulation for behavior
2. **Fail Fast**: Quick feedback with fast unit tests, comprehensive coverage with slower tests
3. **Deterministic Tests**: No flaky tests - use proper isolation and mocking
4. **Realistic Data**: Property-based tests with realistic generators
5. **Continuous Monitoring**: Benchmark trends tracked over time
6. **Test as Documentation**: Clear test names and docstrings explain behavior

### Success Criteria

- **Coverage**: >90% line coverage, 100% critical path coverage
- **Performance**: No test suite regression >10%
- **Reliability**: <0.1% flaky test rate
- **Speed**: Unit tests complete in <2 minutes
- **Quality**: All tests have clear assertions and meaningful names

---

*This testing strategy ensures autoconstitution maintains high quality while enabling rapid iteration and confident deployments.*
