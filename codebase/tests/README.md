# autoconstitution Orchestrator Tests

Comprehensive pytest-based test suite for the autoconstitution orchestrator.

## Test Coverage

### Data Classes Tests
- **TestTaskMetrics**: Task metrics calculation and serialization
- **TestBranchMetrics**: Branch-level metrics aggregation
- **TestAgentMetrics**: Agent performance tracking and efficiency scoring
- **TestTaskNode**: Task node creation, status management, and serialization
- **TestResearchBranch**: Branch creation, task/agent management
- **TestSubAgent**: Agent lifecycle, task queue, and cancellation

### Core Components Tests
- **TestTaskDAG**: 
  - Task addition/removal
  - Dependency management
  - Circular dependency detection
  - Ready task identification
  - Status updates
  - Topological sorting

- **TestPerformanceMonitor**:
  - Branch/agent registration
  - Task completion recording
  - Bottleneck identification
  - Alert handling
  - Health report generation

### Agent Pool Management Tests
- **TestAgentPoolManagerSpawnKill**:
  - Agent spawning with capabilities
  - Agent termination (normal/forced)
  - Max agent limits
  - Best agent selection

- **TestAgentPoolManagerReallocation**:
  - Agent reallocation between branches
  - History tracking
  - Auto-scaling based on load
  - Idle agent termination

### Orchestrator Integration Tests
- **TestSwarmOrchestratorLifecycle**:
  - Initialization and shutdown
  - Context manager support

- **TestSwarmOrchestratorBranchManagement**:
  - Branch CRUD operations
  - Parent-child relationships
  - Cascade deletion

- **TestSwarmOrchestratorTaskManagement**:
  - Task creation with dependencies
  - Task cancellation
  - Status tracking

- **TestSwarmOrchestratorAgentManagement**:
  - Agent spawn/terminate through orchestrator
  - Cross-branch reallocation

- **TestSwarmOrchestratorTaskExecution**:
  - Single task execution
  - Branch execution with dependencies
  - Timeout handling
  - Failure and retry logic

- **TestSwarmOrchestratorDynamicReallocation**:
  - Load analysis
  - Dynamic reallocation
  - Metrics collection

- **TestSwarmOrchestratorUtilities**:
  - Report generation
  - State export

### Exception Tests
- **TestExceptions**: Exception hierarchy and message handling

### Integration Tests
- **TestIntegration**:
  - Full workflow execution
  - Concurrent branch execution

## Running Tests

```bash
# Run all tests
pytest tests/test_orchestrator.py -v

# Run specific test class
pytest tests/test_orchestrator.py::TestTaskDAG -v

# Run specific test
pytest tests/test_orchestrator.py::TestTaskDAG::test_dag_add_task -v

# Run with coverage
pytest tests/test_orchestrator.py --cov=autoconstitution --cov-report=term-missing
```

## Test Fixtures

The test suite provides several fixtures:
- `task_dag`: Fresh TaskDAG instance
- `agent_pool`: Fresh AgentPoolManager instance
- `performance_monitor`: Fresh PerformanceMonitor instance
- `orchestrator`: Initialized SwarmOrchestrator instance
- `sample_coro`: Sample async function for testing
- `failing_coro`: Async function that raises an exception
- `timeout_coro`: Async function that takes too long

## Notes

- Tests use `pytest-asyncio` for async test support
- The `AsyncRLock` class patches `asyncio.RLock` which doesn't exist in Python's asyncio module
- Some tests are skipped due to known bugs in the orchestrator implementation (see skip markers)
