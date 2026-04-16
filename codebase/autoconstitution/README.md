# SwarmOrchestrator

A Python 3.11+ async-first orchestrator implementing PARL (Parallel Autonomous Research Layer) principles for autonomous research systems.

## Features

- **Master Controller**: Spawns and manages sub-agents dynamically
- **Task DAG Management**: Full dependency graph with cycle detection and topological ordering
- **Performance Monitoring**: Real-time metrics collection and bottleneck detection
- **Dynamic Reallocation**: Automatic agent scaling and load balancing
- **Branch-Based Organization**: Hierarchical research organization with priorities
- **Fault Tolerance**: Retry logic, health checks, and graceful degradation

## Installation

```bash
pip install autoconstitution
```

## Quick Start

```python
import asyncio
from autoconstitution import SwarmOrchestrator, TaskDependency, BranchPriority

async def main():
    # Create orchestrator
    async with SwarmOrchestrator() as orchestrator:
        
        # Create a research branch
        branch = await orchestrator.create_branch(
            name="Literature Review",
            priority=BranchPriority.HIGH
        )
        
        # Define tasks
        async def search_papers(query: str):
            return [f"Paper on {query}"]
        
        async def analyze_papers(papers: list):
            return {"count": len(papers)}
        
        # Add tasks with dependencies
        task1 = await orchestrator.add_task(
            branch_id=branch.branch_id,
            name="Search",
            coro=search_papers,
            args=("ML",)
        )
        
        task2 = await orchestrator.add_task(
            branch_id=branch.branch_id,
            name="Analyze",
            coro=analyze_papers,
            dependencies={TaskDependency(task1.task_id)}
        )
        
        # Execute
        results = await orchestrator.execute_branch(branch.branch_id)
        print(results)

asyncio.run(main())
```

## Architecture

### Core Components

```
SwarmOrchestrator
├── TaskDAG              # Manages task dependencies
├── AgentPoolManager     # Manages sub-agent lifecycle
├── PerformanceMonitor   # Collects and analyzes metrics
└── ResearchBranch       # Organizes related tasks/agents
```

### Task DAG

The TaskDAG provides:
- Cycle detection using DFS
- Topological sorting for execution order
- Dynamic task addition/removal
- Priority-based scheduling

### Agent Pool Manager

Features:
- Dynamic agent spawning/termination
- Capability-based task assignment
- Load balancing across branches
- Auto-scaling based on demand

### Performance Monitor

Tracks:
- Branch-level metrics (success rate, throughput)
- Agent-level metrics (efficiency, load)
- Task-level metrics (duration, retries)
- Bottleneck identification

## API Reference

### SwarmOrchestrator

```python
class SwarmOrchestrator:
    def __init__(
        self,
        max_concurrent_tasks: int = 50,
        task_timeout_sec: float = 300.0,
        enable_auto_scaling: bool = True,
        enable_monitoring: bool = True,
    )
```

#### Methods

- `async initialize()` - Start background tasks
- `async shutdown()` - Graceful shutdown
- `async create_branch(...)` - Create research branch
- `async add_task(...)` - Add task to DAG
- `async execute_branch(branch_id)` - Execute all branch tasks
- `async spawn_agent(...)` - Create sub-agent
- `async reallocate_agent(agent_id, target_branch)` - Move agent
- `async get_metrics()` - Get comprehensive metrics

### Decorators

```python
from autoconstitution import task, retryable

@task(priority=0, retry_limit=3, timeout_sec=60.0)
async def my_task():
    pass

@retryable(max_retries=3, backoff_sec=1.0)
async def fragile_operation():
    pass
```

## Advanced Usage

### Custom Branch Priorities

```python
from autoconstitution import BranchPriority

branch = await orchestrator.create_branch(
    name="Critical Analysis",
    priority=BranchPriority.CRITICAL  # Highest priority
)
```

### Agent Capabilities

```python
agent = await orchestrator.spawn_agent(
    branch_id=branch.branch_id,
    name="GPU Worker",
    capabilities={"gpu", "pytorch", "cuda"}
)
```

### Dynamic Reallocation

```python
# Manually trigger analysis and reallocation
report = await orchestrator.analyze_and_reallocate()
print(report["bottlenecks"])
print(report["reallocations"])
```

### Monitoring Integration

```python
# Add custom alert handler
def on_alert(alert_type: str, data: dict):
    print(f"ALERT: {alert_type} - {data}")

orchestrator._monitor.add_alert_handler(on_alert)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_concurrent_tasks` | 50 | Max parallel task executions |
| `task_timeout_sec` | 300 | Default task timeout |
| `enable_auto_scaling` | True | Auto-scale agents |
| `enable_monitoring` | True | Enable metrics collection |

## Error Handling

```python
from autoconstitution import (
    OrchestratorError,
    TaskDAGError,
    CircularDependencyError,
    AgentError,
    BranchError,
)

try:
    await orchestrator.add_task(...)
except CircularDependencyError as e:
    print(f"Invalid dependency: {e}")
```

## Performance Tuning

1. **Adjust concurrency**: Increase `max_concurrent_tasks` for I/O-bound workloads
2. **Tune timeouts**: Set appropriate `task_timeout_sec` for your tasks
3. **Auto-scaling**: Configure `spawn_threshold` and `idle_timeout_sec`
4. **Priorities**: Use task priorities to optimize execution order

## License

MIT License
