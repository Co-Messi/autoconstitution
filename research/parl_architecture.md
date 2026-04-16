# PARL (Parallel-Agent Reinforcement Learning) Architecture Analysis
## Comprehensive Research Report for SwarmResearch Orchestrator Design

**Source:** Kimi K2.5 Technical Report + Open-Source Community Implementation  
**Date:** Research compiled from arXiv:2602.02276v1 and community implementations  
**Classification:** Technical Architecture Deep Dive

---

## Executive Summary

PARL (Parallel-Agent Reinforcement Learning) is Moonshot AI's novel training paradigm that enables an orchestrator LLM to learn task decomposition and parallel sub-agent coordination. The key innovation is that **parallelism itself is a learned skill**, not a hand-coded heuristic. PARL addresses the fundamental "serial collapse" problem where multi-agent systems default to sequential execution despite having parallel capacity.

**Key Performance Metrics:**
- Up to **100 concurrent sub-agents**
- Up to **1,500 coordinated tool calls** per task
- **4.5x wall-clock speedup** over single-agent baselines
- **80% reduction** in end-to-end runtime on complex tasks
- BrowseComp improvement: 60.6% → 78.4% (+17.8 points)
- WideSearch improvement: 72.7% → 79.0% (+6.3 points)

---

## 1. PARL Architecture Overview

### 1.1 Core Design Philosophy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PARL ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐                                                         │
│  │   TASK INPUT    │                                                         │
│  └────────┬────────┘                                                         │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              TRAINABLE ORCHESTRATOR (Kimi K2.5)                      │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │   Task       │  │   Agent      │  │   Result     │               │    │
│  │  │ Decomposition│──│  Allocation  │──│ Aggregation  │               │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│           │                                                                  │
│           │ create_subagent() / assign_task()                               │
│           ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    FROZEN SUB-AGENTS (x100 max)                      │    │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      ┌────────┐        │    │
│  │  │Agent 1 │ │Agent 2 │ │Agent 3 │ │Agent 4 │ ...  │Agent N │        │    │
│  │  │Research│ │  Code  │ │  Fact  │ │  Data  │      │  Misc  │        │    │
│  │  └────────┘ └────────┘ └────────┘ └────────┘      └────────┘        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Architectural Decoupling

PARL uses a **deliberately decoupled architecture**:

| Component | Status | Role |
|-----------|--------|------|
| **Orchestrator** | Trainable | Decides WHEN to create sub-agents, WHAT tasks to assign, HOW to aggregate results |
| **Sub-agents** | Frozen | Execute assigned subtasks independently; trajectories excluded from optimization |

**Rationale for Decoupling:**
1. **Credit Assignment Ambiguity:** In end-to-end co-optimization, a correct final answer doesn't guarantee flawless sub-agent execution
2. **Training Instability:** Outcome-based rewards are sparse and noisy in multi-agent settings
3. **Disentanglement:** Separates high-level coordination logic from low-level execution proficiency

By freezing sub-agents and treating their outputs as environmental observations, only the orchestrator's coordination logic gets optimized.

---

## 2. Orchestrator Task Decomposition

### 2.1 Dynamic Task Decomposition Process

The orchestrator learns to decompose tasks through RL-driven exploration:

```python
# Conceptual orchestrator decision flow
def orchestrator_decompose(task_input):
    # 1. Analyze task structure
    task_analysis = analyze_parallelization_opportunity(task_input)
    
    # 2. Identify independent subtasks
    subtasks = identify_independent_components(task_analysis)
    
    # 3. Determine optimal parallelization strategy
    parallel_groups = schedule_parallel_execution(subtasks)
    
    # 4. Create specialized sub-agents
    for group in parallel_groups:
        for subtask in group:
            agent = create_subagent(
                role=infer_specialization(subtask),
                tools=assign_tool_access(subtask),
                context=derive_context(subtask)
            )
            assign_task(agent, subtask)
    
    # 5. Aggregate results
    return aggregate_results(wait_for_completion())
```

### 2.2 Task Types That Benefit from Parallelization

PARL training uses synthetic prompts designed to stress sequential execution:

| Task Category | Description | Example |
|--------------|-------------|---------|
| **Wide Search** | Simultaneous exploration of many independent sources | "Find top YouTube creators across 100 niche domains" |
| **Deep Search** | Multiple reasoning branches with delayed aggregation | "Compare 50 companies across 5 dimensions" |
| **Batch Processing** | Large-scale file/document operations | "Analyze 100+ documents, download 1000+ files" |
| **Multi-domain Analysis** | Heterogeneous expertise required | Market research requiring financial + technical + legal analysis |

### 2.3 Decomposition Strategy Learning

**Key Insight:** The prompts do NOT explicitly instruct parallelization. Instead, they shape the task distribution such that parallel strategies are naturally favored.

```
Training Progression:
┌────────────────────────────────────────────────────────────────┐
│  Early Training        →  Mid Training        →  Late Training │
│  ─────────────           ────────────          ─────────────   │
│  • High λ1, λ2           • Decreasing λ        • λ1, λ2 ≈ 0    │
│  • Reward parallelism    • Balance parallel    • Reward only   │
│  • Explore concurrent    •   with success      • task success  │
│    scheduling                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Agent Allocation Strategies

### 3.1 Dynamic Agent Specialization

Sub-agents are **dynamically instantiated** with specialized configurations:

```python
# Agent specialization inference
specialization_map = {
    "web_research": {
        "system_prompt": "You are a research specialist...",
        "tools": ["web_search", "browser", "pdf_parser"],
        "temperature": 0.7
    },
    "code_generation": {
        "system_prompt": "You are a coding expert...",
        "tools": ["code_interpreter", "file_editor", "shell"],
        "temperature": 0.2
    },
    "fact_checker": {
        "system_prompt": "You verify claims...",
        "tools": ["web_search", "knowledge_base"],
        "temperature": 0.1
    },
    "data_analyst": {
        "system_prompt": "You analyze datasets...",
        "tools": ["code_interpreter", "visualization"],
        "temperature": 0.3
    }
}
```

### 3.2 Resource Allocation

The orchestrator manages:
- **Tool access:** Which agents get search, browser, code execution, etc.
- **Context windows:** Independent context per sub-agent
- **Execution environment:** Separate computational entities

### 3.3 Scaling Strategy

```
Training Efficiency Optimization:
┌────────────────────────────────────────────────────────────────┐
│  Phase 1: Small Sub-agents                                     │
│  ─────────────────────────                                     │
│  • Train orchestrator with smaller model instances             │
│  • Faster iteration, cheaper training                          │
│                                                                │
│  Phase 2: Scale Up                                             │
│  ────────────────                                              │
│  • Transition to larger sub-agent models                       │
│  • Dynamic inference instance ratio adjustment                 │
│  • Maximize cluster resource usage                             │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. Reward Structure and Anti-Collapse Mechanisms

### 4.1 PARL Reward Function

The complete reward formula from the Kimi K2.5 technical report:

```
r_PARL(x,y) = λ₁ · r_parallel + λ₂ · r_finish + r_perf(x,y)

where:
- r_parallel: instantiation reward (mitigates serial collapse)
- r_finish: sub-agent finish rate (prevents spurious parallelism)  
- r_perf(x,y): task-level outcome (primary objective)
- λ₁, λ₂: annealed to zero over training
```

### 4.2 Reward Components Deep Dive

#### 4.2.1 Instantiation Reward (r_parallel)

```python
def compute_instantiation_reward(num_subagents, max_subagents=100):
    """
    Mitigates SERIAL COLLAPSE - the tendency to default to single-agent execution
    """
    normalized_count = num_subagents.float() / max_subagents
    return normalized_count.clamp(0.0, 1.0)
```

**Purpose:** Incentivizes sub-agent creation and concurrent execution exploration

**Annealing:** λ₁ starts at 0.1, anneals to 0.0 over training

#### 4.2.2 Finish Reward (r_finish)

```python
def compute_finish_reward(completed_subtasks, assigned_subtasks, eps=1e-8):
    """
    Prevents SPURIOUS PARALLELISM - spawning agents without meaningful work
    """
    rate = completed_subtasks.float() / (assigned_subtasks.float() + eps)
    return rate.clamp(0.0, 1.0)
```

**Purpose:** Ensures sub-agents actually complete assigned work, not just spawn

**Annealing:** λ₂ starts at 0.1, anneals to 0.0 over training

#### 4.2.3 Performance Reward (r_perf)

```python
def compute_task_quality(trajectory_features, success_indicators):
    """
    Task-level outcome - the PRIMARY objective
    """
    quality = trajectory_features.mean(dim=-1)
    quality = quality * success_indicators.float()
    return quality.clamp(0.0, 1.0)
```

**Purpose:** Evaluates overall success and quality of the solution

**Note:** This is the ONLY reward component at the end of training (λ₁=λ₂=0)

### 4.3 Staged Reward Annealing

```python
class PARLReward(nn.Module):
    def anneal_lambda1(self, training_step: int) -> torch.Tensor:
        """Anneal λ₁ from init → final over training"""
        progress = min(1.0, training_step / self.total_training_steps)
        lam = self.lambda1_init + (self.lambda1_final - self.lambda1_init) * progress
        return torch.tensor(lam, dtype=torch.float32, device=self.device)
```

**Training Schedule:**

| Phase | λ₁ (Parallel) | λ₂ (Finish) | r_perf | Focus |
|-------|---------------|-------------|--------|-------|
| Early | 0.1 | 0.1 | 1.0 | Explore parallelism |
| Mid | 0.05 | 0.05 | 1.0 | Balance parallel + success |
| Late | 0.0 | 0.0 | 1.0 | Optimize task success only |

### 4.4 Serial Collapse: The Core Problem

**Definition:** Even with parallel capacity, orchestrators tend to degenerate to single-agent sequential execution.

**Why It Happens:**
1. **Coordination Complexity:** Parallel execution requires handling state sync, dependency management
2. **Risk Aversion:** Single-threaded execution is "safer" with no conflicts
3. **Sparse Training Signals:** Parallel execution feedback is delayed and sparse

**PARL Solution:**
- Early training: HEAVILY reward parallelism (high λ₁)
- Force exploration of concurrent scheduling spaces
- Gradually shift to task success optimization

### 4.5 Spurious Parallelism: The Counter-Problem

**Definition:** Orchestrator spawns many sub-agents without meaningful task decomposition.

**Why It Happens:** Reward hacking - maximize r_parallel by spawning agents that do nothing.

**PARL Solution:**
- r_finish rewards COMPLETED subtasks, not just spawned agents
- Ratio of completed/assigned must be high for reward

---

## 5. Critical Steps Metric

### 5.1 Latency-Oriented Evaluation

Traditional metric (total steps) doesn't capture parallel efficiency. PARL introduces **Critical Steps** - analogous to critical path in parallel computation.

### 5.2 Mathematical Definition

```
CriticalSteps = Σₜ (S_main^(t) + maxᵢ S_sub,i^(t))

where:
- S_main^(t): steps by main agent in stage t (typically 1)
- S_sub,i^(t): steps by i-th subagent in stage t
- maxᵢ: the longest-running subagent governs stage duration
```

### 5.3 Implementation

```python
class CriticalStepsMetric(nn.Module):
    def forward(self, main_steps: torch.Tensor, sub_steps: torch.Tensor) -> torch.Tensor:
        """
        Compute critical steps: Σ_t (S_main^(t) + max_i S_sub,i^(t))
        
        Args:
            main_steps: (batch_size, num_stages) - typically 1 per stage
            sub_steps: (batch_size, num_stages, num_subagents)
        
        Returns:
            Total critical steps (batch_size,)
        """
        max_sub_steps = sub_steps.max(dim=-1).values
        max_sub_steps = max_sub_steps.clamp(min=0.0)  # Handle empty stages
        critical_steps_per_stage = main_steps + max_sub_steps
        return critical_steps_per_stage.sum(dim=-1)
```

### 5.4 Why Critical Steps Matter

| Scenario | Total Steps | Critical Steps | Efficiency |
|----------|-------------|----------------|------------|
| 10 agents, each 5 steps | 50 | 1 + 5 = 6 | High |
| 10 agents, unbalanced (1-10 steps) | 55 | 1 + 10 = 11 | Lower |
| Sequential execution | 50 | 50 | Lowest |

**Key Insight:** Well-balanced task decomposition that shortens the longest parallel branch directly reduces critical steps.

---

## 6. Communication Patterns Between Agents

### 6.1 Communication Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMMUNICATION PATTERNS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐                                                         │
│  │  ORCHESTRATOR   │◄──────────────────────────────────────┐                │
│  │   (Central Hub) │                                     │                │
│  └────────┬────────┘                                     │                │
│           │ assign_task()                                │ results        │
│           │                                              │                │
│           ▼                                              │                │
│  ┌──────────────────────────────────────────────────┐   │                │
│  │         MESSAGE PASSING BUS (Shared Memory)       │   │                │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐          │   │                │
│  │  │ Agent 1 │  │ Agent 2 │  │ Agent 3 │  ...     │   │                │
│  │  └────┬────┘  └────┬────┘  └────┬────┘          │   │                │
│  │       │            │            │               │   │                │
│  │       └────────────┴────────────┘               │   │                │
│  │                    │                             │   │                │
│  │                    ▼                             │   │                │
│  │           ┌───────────────┐                      │   │                │
│  │           │ Shared State  │◄─────────────────────┘   │                │
│  │           │   Store       │                          │                │
│  │           └───────────────┘                          │                │
│  └──────────────────────────────────────────────────┘   │                │
│                                                          │                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Communication Patterns

| Pattern | Use Case | Implementation |
|---------|----------|----------------|
| **Hub-and-Spoke** | Central orchestration | Orchestrator ↔ Sub-agents only |
| **Broadcast** | Shared context | Orchestrator sends context to all agents |
| **Gather** | Result aggregation | Sub-agents return to orchestrator |
| **Direct (limited)** | Agent-to-agent | Through shared state store |

### 6.3 Conflict Resolution

When sub-agents return conflicting information:

1. **Source Reliability Weighting:** Weight sources by historical reliability
2. **Corroboration Seeking:** Look for supporting evidence across agents
3. **Confidence Assessment:** Present conflicting perspectives with confidence scores
4. **Orchestrator Arbitration:** Final synthesis by orchestrator

---

## 7. Task DAG Management

### 7.1 Implicit DAG Construction

PARL doesn't use explicit DAG specification. Instead:

```
Task Decomposition → Implicit Dependency Graph → Parallel Scheduling
```

### 7.2 Execution Stage Model

```
Stage 1: Orchestrator analyzes task
    │
    ▼
Stage 2: Spawn parallel group A (agents 1-5)
    │
    ▼
Stage 3: Spawn parallel group B (agents 6-10) [depends on A?]
    │
    ▼
Stage 4: Orchestrator aggregates and synthesizes
```

### 7.3 Dependency Handling

The orchestrator learns to identify:
- **Independent subtasks:** Can execute in parallel (same stage)
- **Dependent subtasks:** Must execute sequentially (different stages)

**Example:**
```
Task C requires output from Task A but not Task B

Sequential: A → B → C (3 stages)
Parallel:   A ─┬─→ C (2 stages)
             B ─┘
```

### 7.4 DAG-Aware Scheduling

```python
# Conceptual scheduling logic
def schedule_parallel_groups(subtasks):
    """
    Group subtasks into parallel execution stages
    """
    stages = []
    remaining = set(subtasks)
    
    while remaining:
        # Find subtasks with no unresolved dependencies
        ready = {t for t in remaining if t.dependencies <= completed}
        
        if not ready:
            raise DependencyCycleError()
        
        stages.append(ParallelGroup(ready))
        completed.update(ready)
        remaining -= ready
    
    return stages
```

---

## 8. Agent Lifecycle Management

### 8.1 Lifecycle States

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENT LIFECYCLE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐    spawn()     ┌─────────┐    assign()    ┌─────────┐         │
│  │  IDLE   │───────────────►│ CREATED │───────────────►│ RUNNING │         │
│  └─────────┘                └─────────┘                └────┬────┘         │
│                                                             │              │
│                                                             │ execute()    │
│                                                             ▼              │
│  ┌─────────┐    terminate() ┌─────────┐               ┌─────────┐          │
│  │  DONE   │◄───────────────│COMPLETE │◄──────────────│EXECUTING│          │
│  └─────────┘                └─────────┘               └─────────┘          │
│       ▲                                                      │              │
│       │                    ┌─────────┐                       │ error        │
│       └────────────────────│  ERROR  │◄──────────────────────┘              │
│                            └─────────┘                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Lifecycle Operations

| Operation | Description | When Used |
|-----------|-------------|-----------|
| **spawn(fork)** | Create child agent with new mission | Delegate subtask |
| **continue(exec)** | Replace agent image, preserve identity | Context limit escape |
| **terminate(exit)** | Signal outcome to parent, release resources | Task completion |

### 8.3 Context Management

Each sub-agent operates with:
- **Independent context window:** No shared context between agents
- **Inherited environment:** Parent's env vars and context history
- **Fresh reasoning space:** Clean slate for focused subtask execution

### 8.4 Resource Cleanup

```python
# Conceptual cleanup
class AgentLifecycle:
    def terminate(self, exit_status):
        """
        Clean agent termination
        """
        # Release memory
        self.context.clear()
        
        # Close file descriptors
        for fd in self.open_fds:
            fd.close()
        
        # Notify parent
        self.parent.signal(self.pid, exit_status)
        
        # Make exit status available for inspection
        self.status = exit_status
```

---

## 9. Implementation Recommendations for SwarmResearch

### 9.1 Core Components to Implement

```python
# Recommended SwarmResearch architecture

class SwarmOrchestrator:
    """
    PARL-inspired orchestrator for SwarmResearch
    """
    def __init__(self, config: OrchestratorConfig):
        self.reward_fn = PARLReward(
            lambda1_init=0.1,
            lambda1_final=0.0,
            lambda2_init=0.1,
            lambda2_final=0.0,
            total_training_steps=10000
        )
        self.critical_steps = CriticalStepsMetric()
        self.subagent_pool = SubAgentPool(max_agents=100)
        self.shared_memory = SharedMemoryStore()
        
    async def execute_task(self, task: Task) -> Result:
        # 1. Decompose task
        subtasks = await self.decompose(task)
        
        # 2. Schedule parallel groups
        stages = self.schedule_parallel_groups(subtasks)
        
        # 3. Execute with DAG-aware scheduling
        results = await self.execute_stages(stages)
        
        # 4. Aggregate and return
        return self.aggregate_results(results)
```

### 9.2 Recommended Directory Structure

```
swarmresearch/
├── orchestrator/
│   ├── __init__.py
│   ├── core.py              # Main orchestrator class
│   ├── decomposition.py     # Task decomposition logic
│   ├── scheduling.py        # Parallel group scheduling
│   └── aggregation.py       # Result aggregation
├── rewards/
│   ├── __init__.py
│   ├── parl_reward.py       # PARL reward implementation
│   └── annealing.py         # Lambda annealing schedule
├── metrics/
│   ├── __init__.py
│   └── critical_steps.py    # Critical steps metric
├── agents/
│   ├── __init__.py
│   ├── lifecycle.py         # Agent spawn/terminate
│   ├── pool.py              # Agent pool management
│   └── specialization.py    # Role inference
├── communication/
│   ├── __init__.py
│   ├── bus.py               # Message passing bus
│   └── shared_memory.py     # Shared state store
└── dag/
    ├── __init__.py
    ├── builder.py           # Implicit DAG construction
    └── executor.py          # DAG-aware execution
```

### 9.3 Key Implementation Details

#### 9.3.1 Reward Function Integration

```python
# Training loop integration
for step in range(total_training_steps):
    # Rollout episode
    trajectory = await orchestrator.execute_task(task)
    
    # Compute PARL reward
    reward_components = reward_fn.compute_full_reward(
        num_subagents=trajectory.num_agents,
        trajectory_features=extract_features(trajectory),
        success=trajectory.success,
        training_step=step,
        completed_subtasks=trajectory.completed_count,
        assigned_subtasks=trajectory.assigned_count
    )
    
    # Update policy
    loss = policy.update(trajectory, reward_components['total_reward'])
```

#### 9.3.2 Critical Steps Tracking

```python
# During execution
stage_metrics = []
for stage in execution_stages:
    main_steps = 1  # Orchestrator step
    sub_steps = [agent.steps for agent in stage.agents]
    
    critical_per_stage = main_steps + max(sub_steps)
    stage_metrics.append(critical_per_stage)

total_critical_steps = sum(stage_metrics)
```

#### 9.3.3 Anti-Collapse Monitoring

```python
# Monitor for serial collapse during training
def detect_serial_collapse(episode_stats):
    """
    Alert if orchestrator is defaulting to sequential execution
    """
    avg_parallelism = episode_stats.num_agents / episode_stats.num_stages
    
    if avg_parallelism < 1.5 and episode_stats.training_step > 1000:
        logger.warning("Potential serial collapse detected!")
        logger.warning(f"Avg parallelism: {avg_parallelism}")
        # Consider increasing λ₁ temporarily
```

### 9.4 Training Configuration Recommendations

```yaml
# parl_config.yaml
reward:
  lambda1:
    init: 0.1
    final: 0.0
  lambda2:
    init: 0.1
    final: 0.0
  annealing_schedule: "linear"
  total_training_steps: 10000

orchestrator:
  max_subagents: 100
  max_tool_calls: 1500
  training:
    phase1_subagent_size: "small"   # Start small
    phase2_subagent_size: "large"   # Scale up

metrics:
  use_critical_steps: true
  orchestration_overhead: 0.1

training:
  optimizer: "AdamW"
  learning_rate: 1e-5
  batch_size: 32
  episodes_per_update: 8
```

---

## 10. Code Structure Suggestions

### 10.1 Complete PARL Reward Implementation

```python
# swarmresearch/rewards/parl_reward.py
import torch
import torch.nn as nn


class PARLReward(nn.Module):
    """
    Parallel-Agent Reinforcement Learning Reward Function
    
    Implements: r_PARL(x,y) = λ₁·r_parallel + λ₂·r_finish + r_perf(x,y)
    """
    
    def __init__(
        self,
        lambda1_init: float = 0.1,
        lambda1_final: float = 0.0,
        lambda2_init: float = 0.1,
        lambda2_final: float = 0.0,
        total_training_steps: int = 10000,
        device: str = "cpu",
    ):
        super().__init__()
        self.lambda1_init = lambda1_init
        self.lambda1_final = lambda1_final
        self.lambda2_init = lambda2_init
        self.lambda2_final = lambda2_final
        self.total_training_steps = total_training_steps
        self.device = device
        
    def anneal_lambda(self, init_val: float, final_val: float, step: int) -> torch.Tensor:
        """Linear annealing from init to final."""
        progress = min(1.0, step / self.total_training_steps)
        val = init_val + (final_val - init_val) * progress
        return torch.tensor(val, device=self.device)
    
    def compute_instantiation_reward(
        self, 
        num_subagents: torch.Tensor,
        max_subagents: int = 100
    ) -> torch.Tensor:
        """r_parallel: encourages sub-agent creation."""
        return (num_subagents.float() / max_subagents).clamp(0, 1)
    
    def compute_finish_reward(
        self,
        completed: torch.Tensor,
        assigned: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """r_finish: rewards completed subtasks."""
        return (completed.float() / (assigned.float() + eps)).clamp(0, 1)
    
    def compute_performance_reward(
        self,
        success: torch.Tensor,
        quality: torch.Tensor
    ) -> torch.Tensor:
        """r_perf: task-level outcome."""
        return (success.float() * quality).clamp(0, 1)
    
    def forward(
        self,
        num_subagents: torch.Tensor,
        completed_subtasks: torch.Tensor,
        assigned_subtasks: torch.Tensor,
        success: torch.Tensor,
        quality: torch.Tensor,
        training_step: int
    ) -> dict:
        """Compute full PARL reward with all components."""
        
        # Anneal lambdas
        lambda1 = self.anneal_lambda(self.lambda1_init, self.lambda1_final, training_step)
        lambda2 = self.anneal_lambda(self.lambda2_init, self.lambda2_final, training_step)
        
        # Compute reward components
        r_parallel = self.compute_instantiation_reward(num_subagents)
        r_finish = self.compute_finish_reward(completed_subtasks, assigned_subtasks)
        r_perf = self.compute_performance_reward(success, quality)
        
        # Total reward
        total = lambda1 * r_parallel + lambda2 * r_finish + r_perf
        
        return {
            "total_reward": total,
            "r_parallel": r_parallel,
            "r_finish": r_finish,
            "r_perf": r_perf,
            "lambda1": lambda1,
            "lambda2": lambda2,
        }
```

### 10.2 Critical Steps Metric Implementation

```python
# swarmresearch/metrics/critical_steps.py
import torch
import torch.nn as nn


class CriticalStepsMetric(nn.Module):
    """
    Latency-oriented metric for parallel agent execution.
    
    CriticalSteps = Σ_t (S_main^(t) + max_i S_sub,i^(t))
    """
    
    def __init__(self, orchestration_overhead: float = 0.1):
        super().__init__()
        self.overhead = orchestration_overhead
    
    def forward(
        self,
        main_steps: torch.Tensor,  # (batch, num_stages)
        sub_steps: torch.Tensor    # (batch, num_stages, max_subagents)
    ) -> torch.Tensor:
        """
        Compute critical steps across all stages.
        
        Args:
            main_steps: Orchestrator steps per stage (typically 1)
            sub_steps: Sub-agent steps per stage per agent
        
        Returns:
            Total critical steps per batch element
        """
        # Max steps among parallel sub-agents in each stage
        max_sub = sub_steps.max(dim=-1).values  # (batch, num_stages)
        max_sub = max_sub.clamp(min=0)  # Handle empty stages
        
        # Critical steps per stage
        critical_per_stage = main_steps + max_sub + self.overhead
        
        # Sum across stages
        return critical_per_stage.sum(dim=-1)
```

### 10.3 Orchestrator Core Implementation

```python
# swarmresearch/orchestrator/core.py
from typing import List, Dict, Any
import asyncio


class SwarmOrchestrator:
    """
    PARL-inspired orchestrator for parallel agent execution.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reward_fn = PARLReward(**config['reward'])
        self.critical_metric = CriticalStepsMetric()
        self.agent_pool = AgentPool(config['max_subagents'])
        self.shared_memory = SharedMemory()
        
    async def decompose_task(self, task: Task) -> List[SubTask]:
        """
        Decompose task into parallelizable subtasks.
        Learned through RL - not hardcoded.
        """
        # This would be the policy network output
        # For now, placeholder for learned decomposition
        raise NotImplementedError("Use trained policy model")
    
    def schedule_parallel_groups(
        self, 
        subtasks: List[SubTask]
    ) -> List[ParallelGroup]:
        """
        Group subtasks into parallel execution stages.
        Respects dependencies.
        """
        # Build implicit DAG from dependencies
        dag = self._build_dag(subtasks)
        
        # Topological sort into stages
        stages = []
        remaining = set(subtasks)
        completed = set()
        
        while remaining:
            # Find ready tasks (all deps satisfied)
            ready = {
                t for t in remaining 
                if t.dependencies <= completed
            }
            
            if not ready:
                raise DependencyCycleError("Circular dependency detected")
            
            stages.append(ParallelGroup(ready))
            completed.update(ready)
            remaining -= ready
        
        return stages
    
    async def execute_stages(
        self, 
        stages: List[ParallelGroup]
    ) -> List[AgentResult]:
        """
        Execute parallel groups sequentially.
        Agents within each group run in parallel.
        """
        all_results = []
        
        for stage in stages:
            # Spawn agents for this stage
            agents = [
                await self.agent_pool.spawn(
                    role=infer_role(task),
                    task=task,
                    tools=assign_tools(task)
                )
                for task in stage.tasks
            ]
            
            # Execute in parallel
            results = await asyncio.gather(*[
                agent.execute() for agent in agents
            ])
            
            all_results.extend(results)
            
            # Store in shared memory for downstream stages
            for task, result in zip(stage.tasks, results):
                self.shared_memory.store(task.id, result)
        
        return all_results
    
    def aggregate_results(self, results: List[AgentResult]) -> FinalResult:
        """
        Synthesize sub-agent outputs into final response.
        """
        # Conflict resolution, synthesis, formatting
        return ResultSynthesizer.synthesize(results)
    
    async def execute(self, task: Task) -> FinalResult:
        """
        Main entry point: execute task with parallel agent swarm.
        """
        # Decompose
        subtasks = await self.decompose_task(task)
        
        # Schedule
        stages = self.schedule_parallel_groups(subtasks)
        
        # Execute
        results = await self.execute_stages(stages)
        
        # Aggregate
        return self.aggregate_results(results)
```

---

## 11. References and Further Reading

### Primary Sources
1. **Kimi K2.5 Technical Report:** arXiv:2602.02276v1 [cs.CL]
2. **Open-Source PARL Implementation:** https://github.com/The-Swarm-Corporation/PARL
3. **Moonshot AI Blog:** https://www.kimi.com/blog/kimi-k2-5

### Related Work
- **M-GRPO:** Multi-Agent Group Relative Policy Optimization
- **C3:** Contextual Counterfactual Credit Assignment for Multi-Agent RL
- **DAG-Plan:** LLM-based task decomposition with DAG structure

### Benchmarks
- **BrowseComp:** Browsing hard-to-find information & deep reasoning
- **WideSearch:** Large-scale retrieval
- **Swarm Bench (in-house):** Real-world complexity across 4 domains

---

## 12. Summary for SwarmResearch Implementation

### Critical Success Factors

1. **Decoupled Architecture:** Train orchestrator, freeze sub-agents
2. **Staged Reward Annealing:** Early parallelism → Late task success
3. **Critical Steps Metric:** Measure latency, not total work
4. **Anti-Collapse Monitoring:** Track and prevent serial collapse
5. **Dependency-Aware Scheduling:** Implicit DAG construction

### Implementation Priority

| Priority | Component | Effort | Impact |
|----------|-----------|--------|--------|
| P0 | PARL Reward Function | Low | Critical |
| P0 | Critical Steps Metric | Low | Critical |
| P1 | Agent Lifecycle Management | Medium | High |
| P1 | Task Decomposition Policy | High | High |
| P2 | DAG-Aware Scheduler | Medium | Medium |
| P2 | Shared Memory / Communication | Medium | Medium |
| P3 | Conflict Resolution | Low | Low |

---

*Report compiled for SwarmResearch orchestrator design. All technical details extracted from publicly available sources including the Kimi K2.5 technical report and open-source community implementations.*
