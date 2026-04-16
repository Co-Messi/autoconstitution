# autoconstitution Architecture

## Comprehensive Technical Documentation

**Version:** 1.0  
**Last Updated:** April 2026  
**Status:** Design Complete, Implementation In Progress

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Component Architecture](#3-component-architecture)
4. [Data Flow](#4-data-flow)
5. [Design Decisions & Rationale](#5-design-decisions--rationale)
6. [Comparison with Karpathy's autoresearch](#6-comparison-with-karpathys-autoresearch)
7. [Implementation Details](#7-implementation-details)
8. [Scaling Strategy](#8-scaling-strategy)

---

## 1. Executive Summary

autoconstitution is a **massively parallel collaborative AI research system** that extends Karpathy's autoresearch paradigm to multiple coordinated agents. It implements the **PARL (Parallel Autonomous Research Layer)** architecture, enabling hundreds of AI agents to explore research problems simultaneously while sharing findings and avoiding redundant work.

### Key Innovations

| Feature | Description | Advantage |
|---------|-------------|-----------|
| **Parallel Agent Pools** | Git worktree isolation per agent | Safe parallel exploration |
| **Cross-Pollination Bus** | Pub/sub findings sharing with rate limiting | Prevents premature convergence |
| **Dynamic Reallocation** | Agents migrate based on progress gradients | Resources flow to promising areas |
| **Global Ratchet** | Best results only improve, never regress | Guaranteed progress |
| **Provider Agnostic** | Unified interface for all LLM providers | Flexibility and failover |

### Target Use Cases

- Automated machine learning research
- Hyperparameter optimization at scale
- Neural architecture search
- Scientific hypothesis generation and validation
- Code optimization and refactoring

---

## 2. System Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SWARM RESEARCH SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         ORCHESTRATION LAYER                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
│  │  │   Swarm     │  │   Task      │  │   Result    │  │   Performance   │  │   │
│  │  │  Manager    │  │  Scheduler  │  │  Aggregator │  │    Monitor      │  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │   │
│  │         └─────────────────┴─────────────────┴─────────────────┘          │   │
│  │                              │                                           │   │
│  │                    ┌─────────┴─────────┐                                 │   │
│  │                    │  Cross-Pollination │ (Shared Findings Bus)          │   │
│  │                    │      Bus          │                                │   │
│  │                    └─────────┬─────────┘                                 │   │
│  └──────────────────────────────┼───────────────────────────────────────────┘   │
│                                 │                                                │
│  ┌──────────────────────────────┼───────────────────────────────────────────┐   │
│  │                    AGENT EXECUTION LAYER                                  │   │
│  │  ┌───────────────────────────┼───────────────────────────────────────┐   │   │
│  │  │                    Agent Worker Pool                               │   │   │
│  │  │  ┌─────────┐ ┌─────────┐ │ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │   │
│  │  │  │ Worker 1│ │ Worker 2│ │ │ Worker N│ │ Worker..│ │ Worker..│   │   │   │
│  │  │  │(BranchA)│ │(BranchA)│ │ │(BranchB)│ │(BranchC)│ │(BranchD)│   │   │   │
│  │  │  └────┬────┘ └────┬────┘ │ └────┬────┘ └────┬────┘ └────┬────┘   │   │   │
│  │  │       └───────────┴──────┼──────┴───────────┴───────────┘         │   │   │
│  │  │                          │                                        │   │   │
│  │  │  ┌───────────────────────┴───────────────────────┐                │   │   │
│  │  │  │         Provider Abstraction Layer            │                │   │   │
│  │  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────┐  │                │   │   │
│  │  │  │  │  Kimi   │ │ Claude  │ │ OpenAI  │ │Ollam│  │                │   │   │
│  │  │  │  │ Adapter │ │ Adapter │ │ Adapter │ │  a  │  │                │   │   │
│  │  │  │  └─────────┘ └─────────┘ └─────────┘ └─────┘  │                │   │   │
│  │  │  └───────────────────────────────────────────────┘                │   │   │
│  │  └───────────────────────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                 │                                                │
│  ┌──────────────────────────────┼───────────────────────────────────────────┐   │
│  │                      STORAGE LAYER                                        │   │
│  │  ┌─────────────┐  ┌─────────┴─────────┐  ┌─────────────┐  ┌─────────┐    │   │
│  │  │  Vector DB  │  │   Object Store    │  │    Cache    │  │  State  │    │   │
│  │  │(Embeddings) │  │  (Artifacts/Logs) │  │   (Redis)   │  │  Store  │    │   │
│  │  │  ChromaDB   │  │    MinIO/S3       │  │             │  │ (SQLite │    │   │
│  │  │   pgvector  │  │                   │  │             │  │/Postgres│    │   │
│  │  └─────────────┘  └─────────────────┘  └─────────────┘  └─────────┘    │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Design Principles

1. **Provider Agnosticism**: All LLM providers implement a common interface
2. **Horizontal Scalability**: Stateless components enable linear scaling
3. **Message-Driven**: Async message passing decouples components
4. **Minimal Core**: Karpathy's 3-file philosophy extended, not replaced
5. **Progressive Enhancement**: Start simple, add complexity only when needed

### 2.3 PARL Architecture

autoconstitution implements **PARL (Parallel Autonomous Research Layer)**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PARL ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  LAYER 3: Global Orchestrator (Meta-Agent)                      │    │
│  │  • Task complexity assessment                                   │    │
│  │  • Dynamic routing (subagent vs team mode)                      │    │
│  │  • Global best tracking (ratchet)                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  LAYER 2: Agent Teams (by specialization)                       │    │
│  │  • Architecture agents                                          │    │
│  │  • Optimization agents                                          │    │
│  │  • Data agents                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  LAYER 1: Individual Agents (autoresearch-style)                │    │
│  │  • Git worktree isolation                                       │    │
│  │  • 5-minute training loop                                       │    │
│  │  • Local results tracking                                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  LAYER 0: Shared Infrastructure                                 │    │
│  │  • Code merge service                                           │    │
│  │  • Results database                                             │    │
│  │  • Failure knowledge base                                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Architecture

### 3.1 Component Detail View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPONENT DETAIL VIEW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  SWARM ORCHESTRATOR (orchestrator.py)                                  │  │
│  │  ─────────────────────────────────────                                 │  │
│  │  Responsibilities:                                                     │  │
│  │    • Master controller for entire swarm                                │  │
│  │    • Task DAG management with dependency resolution                    │  │
│  │    • Sub-agent spawning and lifecycle management                       │  │
│  │    • Dynamic agent reallocation based on load                          │  │
│  │    • Performance monitoring and metrics collection                     │  │
│  │    • Branch-based research organization                                │  │
│  │                                                                        │  │
│  │  Components:                                                           │  │
│  │    ├── TaskDAG: Directed acyclic graph for task dependencies           │  │
│  │    ├── AgentPoolManager: Dynamic agent lifecycle management            │  │
│  │    ├── PerformanceMonitor: Real-time metrics and bottleneck detection  │  │
│  │    └── ResearchBranch: Hierarchical research organization              │  │
│  │                                                                        │  │
│  │  Interface:                                                            │  │
│  │    create_branch(name, priority) -> ResearchBranch                     │  │
│  │    add_task(branch_id, coro, dependencies) -> TaskNode                 │  │
│  │    execute_branch(branch_id) -> Dict[TaskID, Result]                   │  │
│  │    spawn_agent(branch_id, capabilities) -> SubAgent                    │  │
│  │    reallocate_agent(agent_id, target_branch) -> bool                   │  │
│  │    analyze_and_reallocate() -> ReallocationReport                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  TASK DAG (TaskDAG class)                                              │  │
│  │  ────────────────────────                                              │  │
│  │  Responsibilities:                                                     │  │
│  │    • Manage task dependencies                                          │  │
│  │    • Detect cycles using DFS                                           │  │
│  │    • Topological sorting for execution order                           │  │
│  │    • Dynamic task addition/removal                                     │  │
│  │    • Priority-based scheduling                                         │  │
│  │                                                                        │  │
│  │  Key Methods:                                                          │  │
│  │    add_task(task: TaskNode) -> None                                    │  │
│  │    remove_task(task_id: TaskID) -> Optional[TaskNode]                  │  │
│  │    get_ready_tasks(branch_id) -> List[TaskNode]                        │  │
│  │    topological_sort() -> List[TaskID]                                  │  │
│  │    _has_cycle() -> bool  # DFS-based detection                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  AGENT POOL MANAGER (AgentPoolManager class)                           │  │
│  │  ───────────────────────────────────────────                           │  │
│  │  Responsibilities:                                                     │  │
│  │    • Dynamic agent spawning/termination                                │  │
│  │    • Load balancing across branches                                    │  │
│  │    • Capability-based task assignment                                  │  │
│  │    • Resource optimization                                             │  │
│  │    • Auto-scaling based on demand                                      │  │
│  │                                                                        │  │
│  │  Key Methods:                                                          │  │
│  │    spawn_agent(branch_id, name, capabilities) -> SubAgent              │  │
│  │    terminate_agent(agent_id, force) -> bool                            │  │
│  │    find_best_agent(branch_id, required_capabilities) -> SubAgent       │  │
│  │    reallocate_agent(agent_id, target_branch) -> bool                   │  │
│  │    auto_scale(branch_loads) -> ScalingActions                          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  PERFORMANCE MONITOR (PerformanceMonitor class)                        │  │
│  │  ──────────────────────────────────────────────                        │  │
│  │  Responsibilities:                                                     │  │
│  │    • Real-time metric collection                                       │  │
│  │    • Trend analysis                                                    │  │
│  │    • Bottleneck detection                                              │  │
│  │    • Performance alerts                                                │  │
│  │                                                                        │  │
│  │  Metrics Tracked:                                                      │  │
│  │    • Branch-level: success_rate, throughput, avg_task_duration         │  │
│  │    • Agent-level: efficiency_score, tasks_completed, current_load      │  │
│  │    • Task-level: duration_ms, retry_count, error_count                 │  │
│  │                                                                        │  │
│  │  Key Methods:                                                          │  │
│  │    record_task_completion(task, success) -> None                       │  │
│  │    identify_bottlenecks() -> List[Bottleneck]                          │  │
│  │    get_health_report() -> HealthReport                                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  RATCHET MECHANISM (ratchet2.py)                                       │  │
│  │  ───────────────────────────────                                       │  │
│  │  Responsibilities:                                                     │  │
│  │    • Ensure progress only moves forward                                │  │
│  │    • Track best experimental results                                   │  │
│  │    • Validate new experiments against current best                     │  │
│  │    • Pluggable metric interface                                        │  │
│  │    • Async state persistence                                           │  │
│  │                                                                        │  │
│  │  Comparison Modes:                                                     │  │
│  │    • HIGHER_IS_BETTER: For metrics like accuracy                       │  │
│  │    • LOWER_IS_BETTER: For metrics like loss                            │  │
│  │    • CLOSER_TO_TARGET: For metrics with optimal value                  │  │
│  │                                                                        │  │
│  │  Key Methods:                                                          │  │
│  │    validate_experiment(exp_id, score) -> ValidationResult              │  │
│  │    commit_experiment(exp_id, score, metadata) -> ValidationResult      │  │
│  │    save_state() / load_state() -> None                                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  CROSS-POLLINATION BUS (pollination.py)                                │  │
│  │  ──────────────────────────────────────                                │  │
│  │  Responsibilities:                                                     │  │
│  │    • Pub/sub broadcast system for agent findings                       │  │
│  │    • Frequency controls to prevent information flooding                │  │
│  │    • Subscription management with filtering                            │  │
│  │    • Information decay mechanisms                                      │  │
│  │    • Anti-convergence architecture                                     │  │
│  │                                                                        │  │
│  │  Key Components:                                                       │  │
│  │    • TokenBucketRateLimiter: Per-agent rate limiting                   │  │
│  │    • AdaptiveRateLimiter: Load-based adjustment                        │  │
│  │    • SubscriptionMatcher: Finding-to-agent matching                    │  │
│  │    • InformationDecay: Time-based relevance reduction                  │  │
│  │    • ConvergenceMonitor: Detect and mitigate premature convergence     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Provider Abstraction Layer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PROVIDER ADAPTER HIERARCHY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                         ┌─────────────────┐                                  │
│                         │  LLMProvider    │  <<abstract>>                   │
│                         │   (Interface)   │                                  │
│                         └────────┬────────┘                                  │
│                                  │                                           │
│            ┌─────────────────────┼─────────────────────┐                     │
│            │                     │                     │                     │
│    ┌───────┴───────┐    ┌───────┴───────┐    ┌───────┴───────┐             │
│    │ CloudProvider │    │ LocalProvider │    │HybridProvider │             │
│    │  (abstract)   │    │  (abstract)   │    │  (abstract)   │             │
│    └───────┬───────┘    └───────┬───────┘    └───────┬───────┘             │
│            │                     │                     │                     │
│    ┌───────┴───────┐    ┌───────┴───────┐            │                     │
│    │               │    │               │            │                     │
│ ┌──┴───┐ ┌──┴───┐ ┌┴────┐ ┌────┐    ┌──┴───┐        │                     │
│ │ Kimi │ │Claude│ │OpenAI│ │Ollama│ │vLLM  │    ┌───┴───┐                 │
│ │Adapter│ │Adapter│ │Adapter│ │Adapter│ │Adapter│    │Router │                 │
│ └──────┘ └──────┘ └─────┘ └────┘    └──────┘    └───────┘                 │
│                                                                              │
│  Provider Capabilities Matrix:                                               │
│  ┌──────────┬─────────┬──────────┬────────┬─────────┬────────┐              │
│  │ Feature  │  Kimi   │  Claude  │ OpenAI │ Ollama  │  vLLM  │              │
│  ├──────────┼─────────┼──────────┼────────┼─────────┼────────┤              │
│  │ Streaming│    ✓    │    ✓     │   ✓    │    ✓    │   ✓    │              │
│  │ Functions│    ✓    │    ✓     │   ✓    │    ✓    │   ✓    │              │
│  │ Vision   │    ✓    │    ✓     │   ✓    │    ✗    │   ✓    │              │
│  │ Embeddings│   ✓    │    ✓     │   ✓    │    ✓    │   ✓    │              │
│  │ Local    │    ✗    │    ✗     │   ✗    │    ✓    │   ✓    │              │
│  │ Cost Track│   ✓    │    ✓     │   ✓    │    ✓    │   ✓    │              │
│  └──────────┴─────────┴──────────┴────────┴─────────┴────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow

### 4.1 Request Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        REQUEST LIFECYCLE FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: Submission                                                         │
│  ═════════════════                                                           │
│                                                                              │
│  User ──► [API Gateway] ──► [Task Scheduler] ──► [Priority Queue]          │
│         POST /research      validate & parse      assign priority            │
│                                                                              │
│  Phase 2: Decomposition                                                      │
│  ═══════════════════════                                                     │
│                                                                              │
│  [Task Scheduler] ──► [Decomposer Agent] ──► Subtasks[]                      │
│                     analyze & break down      parallelizable units           │
│                                                                              │
│  Phase 3: Distribution                                                       │
│  ═══════════════════════                                                     │
│                                                                              │
│  Subtasks[] ──► [Message Bus] ──► [Worker Pool]                              │
│               publish tasks         pull & execute                           │
│                                                                              │
│  Phase 4: Execution                                                          │
│  ══════════════════                                                          │
│                                                                              │
│  [Worker] ──► [Provider Router] ──► [LLM Provider] ──► Response            │
│           select provider         generate completion                        │
│                                                                              │
│  Phase 5: Aggregation                                                        │
│  ═══════════════════════                                                     │
│                                                                              │
│  Responses[] ──► [Result Aggregator] ──► [Consensus Check] ──► Final Result │
│              merge & validate      voting/refinement                         │
│                                                                              │
│  Phase 6: Delivery                                                           │
│  ═══════════════════                                                         │
│                                                                              │
│  Final Result ──► [Response Formatter] ──► [User]                            │
│               format & cite              streaming or batch                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Task Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TASK EXECUTION FLOW DIAGRAM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐           │
│  │  PENDING │────▶│ SCHEDULED│────▶│ RUNNING  │────▶│COMPLETED │           │
│  └──────────┘     └──────────┘     └──────────┘     └──────────┘           │
│       │                 │               │                │                  │
│       │                 │               │                │                  │
│       ▼                 │               │                │                  │
│  ┌──────────┐           │               │                │                  │
│  │CANCELLED │◄──────────┴───────────────┴────────────────┘                  │
│  └──────────┘                                                              │
│       ▲                                                                    │
│       │                 │               │                │                  │
│       │                 │               ▼                │                  │
│       │                 │          ┌──────────┐          │                  │
│       │                 │          │  FAILED  │──────────┘                  │
│       │                 │          └──────────┘                             │
│       │                 │               │                                   │
│       │                 │               ▼                                   │
│       │                 │          ┌──────────┐                             │
│       │                 └─────────▶│ RETRYING │─────────────────────────────┘
│       │                            └──────────┘                             │
│       │                                                                     │
│       └────────────────────────────────────────────────────────────────▶    │
│                                                                              │
│  State Transitions:                                                          │
│  • PENDING → SCHEDULED: Dependencies satisfied                               │
│  • SCHEDULED → RUNNING: Agent assigned and executing                         │
│  • RUNNING → COMPLETED: Task finished successfully                           │
│  • RUNNING → FAILED: Task failed, retry if attempts remain                   │
│  • FAILED → RETRYING: Retry attempt initiated                                │
│  • RETRYING → RUNNING: Retry executing                                       │
│  • Any → CANCELLED: Explicit cancellation requested                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Cross-Pollination Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CROSS-POLLINATION MESSAGE FLOW                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Agent Discovery                                                             │
│        │                                                                     │
│        ▼                                                                     │
│  ┌─────────────┐                                                            │
│  │  Validate   │──▶ Local reproduction (3x)                                 │
│  │  (internal) │──▶ Significance test                                       │
│  └──────┬──────┘──▶ Novelty check                                           │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────┐                                                            │
│  │   Publish   │──▶ Rate limit check                                        │
│  │   Request   │──▶ Priority assignment                                     │
│  └──────┬──────┘──▶ Envelope creation                                       │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │   Route &   │────▶│   Filter    │────▶│   Queue &   │                   │
│  │   Filter    │     │             │     │   Deliver   │                   │
│  └─────────────┘     └─────────────┘     └─────────────┘                   │
│       │                                        │                             │
│       │ Subscription matching                  │ Priority queuing            │
│       │ Diversity filtering                    │ Delivery with decay         │
│       │ Recipient selection                    │ Consumption tracking        │
│       │                                        │                             │
│       ▼                                        ▼                             │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    SUBSCRIBER AGENTS                         │            │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │            │
│  │  │ Agent 1 │  │ Agent 2 │  │ Agent 3 │  │ Agent N │        │            │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Design Decisions & Rationale

### 5.1 Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Message Bus** | Redis → RabbitMQ/NATS | Redis for simple deployments, RabbitMQ for complex routing, NATS for cloud-native |
| **State Storage** | SQLite → PostgreSQL | SQLite for single-node, PostgreSQL for HA and complex queries |
| **Vector DB** | ChromaDB → pgvector | ChromaDB for embedded, pgvector for unified SQL+vector |
| **API Framework** | FastAPI | Native async, automatic OpenAPI, Python-native |
| **Worker Framework** | Asyncio + uvloop | Lightweight, no external dependencies for core |
| **Container Orchestration** | Docker Compose → K8s | Compose for dev/small, K8s for production scale |
| **Provider Abstraction** | Adapter Pattern | Clean separation, easy to add new providers |
| **Agent Communication** | Message Bus over RPC | Better decoupling, natural async, easier scaling |
| **Code Isolation** | Git worktrees | Native git support, easy merge, proven technology |
| **Progress Tracking** | Ratchet mechanism | Guarantees monotonic improvement, simple mental model |

### 5.2 Key Design Rationale

#### Why Message-Driven Architecture?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MESSAGE-DRIVEN vs RPC COMPARISON                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Message-Driven (Selected)                    RPC (Rejected)                 │
│  ═════════════════════════                    ══════════════                 │
│                                                                              │
│  ✓ Natural async/await patterns               ✗ Tight coupling               │
│  ✓ Easy horizontal scaling                    ✗ Harder to scale              │
│  ✓ Built-in buffering under load              ✗ Backpressure complex         │
│  ✓ Service discovery simplified               ✗ Service registry needed      │
│  ✓ Natural event sourcing                     ✗ Additional logging needed    │
│  ✓ Better fault isolation                     ✗ Cascading failures           │
│                                                                              │
│  Trade-offs:                                                                 │
│  - Eventual consistency vs strong consistency                                │
│  - Message delivery guarantees (at-least-once vs exactly-once)               │
│  - Debugging complexity (distributed traces needed)                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Why Git Worktree Isolation?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GIT WORKTREE ISOLATION BENEFITS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Traditional Approach (Shared Directory):                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  /workspace/                                                        │    │
│  │    ├── agent_1/  ←── Conflicts when multiple agents edit same files │    │
│  │    ├── agent_2/  ←── Complex locking required                       │    │
│  │    └── shared/   ←── Race conditions on shared resources            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Git Worktree Approach (Selected):                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  /workspace/                                                        │    │
│  │    ├── .git/           ←── Single repository, multiple worktrees    │    │
│  │    ├── main/           ←── Base branch (best known)                 │    │
│  │    ├── worktrees/                                                   │    │
│  │    │    ├── agent_001/ ←── Independent working directory            │    │
│  │    │    ├── agent_002/ ←── Each agent has full git capabilities     │    │
│  │    │    └── agent_003/ ←── Easy diff, merge, and conflict resolution│    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Benefits:                                                                   │
│  1. True isolation - no file-level conflicts between agents                  │
│  2. Native git operations - diff, merge, rebase just work                    │
│  3. Easy promotion - git merge to propagate improvements                     │
│  4. Conflict resolution - git's proven merge algorithms                      │
│  5. Audit trail - complete history of all changes                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Why the Ratchet Mechanism?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RATCHET MECHANISM DESIGN                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Problem: In parallel exploration, how do we ensure progress?                │
│                                                                              │
│  Without Ratchet:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Agent 1: accuracy=0.85 ──┐                                         │    │
│  │  Agent 2: accuracy=0.82 ──┼──► Which is "best"? Can we regress?    │    │
│  │  Agent 3: accuracy=0.87 ──┘                                         │    │
│  │                                                                     │    │
│  │  Risk: Next iteration might produce 0.83 (regression)               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  With Ratchet:                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Global Best: accuracy=0.87 (locked)                                │    │
│  │                                                                     │    │
│  │  Agent 1: accuracy=0.85 ──┐                                         │    │
│  │  Agent 2: accuracy=0.82 ──┼──► All compared against 0.87            │    │
│  │  Agent 3: accuracy=0.88 ──┘    Only 0.88+ replaces best             │    │
│  │                                                                     │    │
│  │  Guarantee: Best score never decreases                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Ratchet Properties:                                                         │
│  • Monotonic: best_score only increases (or decreases for loss)              │
│  • Persistent: state saved to disk, survives restarts                        │
│  • Validated: new results must beat current best by tolerance                │
│  • Auditable: complete history of all experiments                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Comparison with Karpathy's autoresearch

### 6.1 Feature Comparison Matrix

| Feature | Karpathy's autoresearch | autoconstitution |
|---------|------------------------|---------------|
| **Agent Count** | 1 | N (scalable to 100+) |
| **Parallel Execution** | ❌ | ✅ |
| **Cross-Agent Sharing** | ❌ | ✅ (Cross-Pollination Bus) |
| **Dynamic Reallocation** | ❌ | ✅ |
| **Provider Agnostic** | ❌ (CUDA only) | ✅ (Kimi, Claude, OpenAI, Ollama) |
| **Task Dependencies** | ❌ | ✅ (Task DAG) |
| **Branch Organization** | ❌ | ✅ |
| **Auto-Scaling** | ❌ | ✅ |
| **Performance Monitoring** | Basic | Comprehensive |
| **Code Isolation** | Single workspace | Git worktrees |
| **Global Progress Tracking** | ❌ | ✅ (Ratchet) |
| **Failure Recovery** | Manual | Automatic |

### 6.2 Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              KARPATHY'S AUTORESEARCH vs AUTOCONSTITUTION                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Karpathy's autoresearch:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │   ┌─────────────┐                                                   │    │
│  │   │   Agent     │                                                   │    │
│  │   │  (Single)   │◄────► LLM API                                     │    │
│  │   └──────┬──────┘                                                   │    │
│  │          │                                                          │    │
│  │          ▼                                                          │    │
│  │   ┌─────────────┐     ┌─────────────┐                               │    │
│  │   │  Code Editor│────▶│  Training   │                               │    │
│  │   │             │◄────│   Loop      │                               │    │
│  │   └─────────────┘     └─────────────┘                               │    │
│  │          │                                                          │    │
│  │          ▼                                                          │    │
│  │   ┌─────────────┐                                                   │    │
│  │   │ results.tsv │                                                   │    │
│  │   └─────────────┘                                                   │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  autoconstitution:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │   ┌─────────────┐                                                   │    │
│  │   │ Orchestrator│◄────► LLM APIs (Multiple Providers)               │    │
│  │   │  (Master)   │                                                   │    │
│  │   └──────┬──────┘                                                   │    │
│  │          │                                                          │    │
│  │    ┌─────┼─────┬─────────┬─────────┐                                │    │
│  │    │     │     │         │         │                                │    │
│  │    ▼     ▼     ▼         ▼         ▼                                │    │
│  │  ┌────┐┌────┐┌────┐  ┌────┐   ┌────┐                               │    │
│  │  │A-1 ││A-2 ││A-3 │  │A-N │   │A.. │  ←── Parallel agents           │    │
│  │  └──┬─┘└─┬──┘└─┬──┘  └──┬─┘   └──┬─┘                               │    │
│  │     │    │     │        │        │                                  │    │
│  │     └────┴─────┴────────┴────────┘                                  │    │
│  │              │                                                      │    │
│  │              ▼                                                      │    │
│  │   ┌─────────────────────┐                                           │    │
│  │   │ Cross-Pollination   │  ←── Shared findings bus                  │    │
│  │   │       Bus           │                                           │    │
│  │   └─────────────────────┘                                           │    │
│  │              │                                                      │    │
│  │              ▼                                                      │    │
│  │   ┌─────────────────────┐                                           │    │
│  │   │   Global Ratchet    │  ←── Best result tracking                 │    │
│  │   └─────────────────────┘                                           │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Philosophy Comparison

| Aspect | Karpathy's autoresearch | autoconstitution |
|--------|------------------------|---------------|
| **Core Metaphor** | Single PhD student | Entire research community |
| **Scaling Model** | Vertical (bigger GPU) | Horizontal (more agents) |
| **Collaboration** | None | SETI@home-style distributed |
| **Knowledge Sharing** | N/A | Cross-pollination bus |
| **Code Philosophy** | Minimal (3 files) | Extended minimal (modular) |
| **Complexity** | Low, single-purpose | Medium, multi-purpose |
| **Deployment** | Single machine | Single machine → Cluster |

### 6.4 Code Structure Comparison

```
Karpathy's autoresearch (3-file philosophy):
┌─────────────────────────────────────────────────────────────────┐
│  train.py     - Training loop, optimization, checkpointing      │
│  model.py     - Model architecture, forward/backward pass       │
│  data.py      - Data loading, preprocessing, batching           │
└─────────────────────────────────────────────────────────────────┘

autoconstitution Extension:
┌─────────────────────────────────────────────────────────────────┐
│  agent_worker.py      ←→  train.py  (core execution loop)       │
│  orchestrator.py      ←→  model.py  (orchestration topology)    │
│  task_scheduler.py    ←→  data.py   (task flow and distribution)│
├─────────────────────────────────────────────────────────────────┤
│  ratchet2.py          - Progress tracking (ratchet mechanism)   │
│  pollination.py       - Cross-agent findings sharing            │
│  providers/           - LLM provider adapters                   │
│  checkpoint.py        - State persistence and recovery          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Implementation Details

### 7.1 Core Classes

```python
# SwarmOrchestrator - Main entry point
class SwarmOrchestrator:
    """
    Master orchestrator for autoconstitution.
    
    Implements PARL (Parallel Autonomous Research Layer) principles:
    - Parallel execution of research tasks
    - Autonomous agent management
    - Dynamic resource allocation
    - Layered architecture with clear separation of concerns
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 50,
        task_timeout_sec: float = 300.0,
        enable_auto_scaling: bool = True,
        enable_monitoring: bool = True,
    ) -> None:
        self._dag = TaskDAG()
        self._agent_pool = AgentPoolManager()
        self._monitor = PerformanceMonitor() if enable_monitoring else None
        # ...

# TaskDAG - Dependency management
class TaskDAG:
    """
    Directed Acyclic Graph for managing task dependencies.
    
    Supports:
    - Topological ordering for execution
    - Cycle detection
    - Dynamic task addition/removal
    - Priority-based scheduling
    """
    
    async def add_task(self, task: TaskNode) -> None:
        """Add a task to the DAG with cycle detection."""
        # ...
    
    async def get_ready_tasks(self, branch_id: Optional[BranchID] = None) -> list[TaskNode]:
        """Get all tasks ready to execute (dependencies satisfied)."""
        # ...

# Ratchet - Progress tracking
class Ratchet:
    """
    Core improvement tracking mechanism.
    
    Ensures research progress only moves forward by maintaining
    the best-known experimental result.
    """
    
    async def validate_experiment(
        self,
        experiment_id: ExperimentID,
        score: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate a new experiment against the current best."""
        # ...
```

### 7.2 Type Hierarchy

```
Any
├── Exception
│   ├── OrchestratorError
│   │   ├── TaskDAGError
│   │   │   └── CircularDependencyError
│   │   ├── AgentError
│   │   ├── BranchError
│   │   └── ReallocationError
│   └── RatchetError
│       ├── MetricError
│       ├── StatePersistenceError
│       └── ValidationError
│
├── Enum
│   ├── TaskStatus (PENDING, SCHEDULED, RUNNING, COMPLETED, FAILED, CANCELLED, RETRYING)
│   ├── AgentStatus (IDLE, BUSY, PAUSED, TERMINATED, ERROR)
│   ├── BranchPriority (CRITICAL, HIGH, NORMAL, LOW, BACKGROUND)
│   ├── ComparisonMode (HIGHER_IS_BETTER, LOWER_IS_BETTER, CLOSER_TO_TARGET)
│   ├── ValidationDecision (KEEP, DISCARD, TIE, FIRST)
│   ├── FindingType (HYPERPARAMETER_IMPROVEMENT, ARCHITECTURE_CHANGE, ...)
│   └── FindingPriority (LOW, MEDIUM, HIGH, CRITICAL)
│
├── dataclass
│   ├── TaskNode (task_id, branch_id, name, coro, dependencies, status, ...)
│   ├── ResearchBranch (branch_id, name, priority, task_ids, agent_ids, ...)
│   ├── SubAgent (agent_id, branch_id, capabilities, status, metrics, ...)
│   ├── TaskMetrics (duration_ms, cpu_time_ms, retry_count, error_count, ...)
│   ├── BranchMetrics (task_count, success_rate, throughput, ...)
│   ├── AgentMetrics (tasks_completed, efficiency_score, ...)
│   ├── ExperimentResult (experiment_id, score, metadata, ...)
│   ├── ValidationResult (decision, is_improvement, improvement_delta, ...)
│   └── RatchetState (current_best_score, experiment_history, ...)
│
└── ABC/Protocol
    ├── LLMProvider (complete, stream_complete, embed, health_check)
    ├── MetricCalculator (calculate, get_config)
    ├── StatePersister (save, load, exists, delete)
    └── FrequencyController (should_allow_broadcast, record_broadcast)
```

### 7.3 Async Patterns

```python
# Context manager for lifecycle management
async with SwarmOrchestrator() as orchestrator:
    # Orchestrator automatically initialized
    branch = await orchestrator.create_branch(name="Research")
    # ...
# Automatic shutdown on exit

# Semaphore-based concurrency control
async with self._semaphore:
    # Limited concurrent execution
    result = await self.execute_task(task_id)

# Background task management
monitor_task = asyncio.create_task(self._monitoring_loop())
self._background_tasks.add(monitor_task)
monitor_task.add_done_callback(self._background_tasks.discard)

# Retry with exponential backoff
@retryable(max_retries=3, backoff_sec=1.0)
async def fragile_operation():
    # Retries with 1s, 2s, 4s delays
    pass
```

---

## 8. Scaling Strategy

### 8.1 Deployment Tiers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SCALING DEPLOYMENT TIERS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TIER 1: Development (Mac Mini M4)                                           │
│  ═══════════════════════════════════                                         │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  [FastAPI Server] ──► [In-Memory Queue] ──► [Ollama (local)]        │    │
│  │       │                      │                    │                 │    │
│  │       ▼                      ▼                    ▼                 │    │
│  │  [SQLite]              [Asyncio Tasks]      [Mistral 7B]            │    │
│  │                                                                     │    │
│  │  Resources: 16GB RAM, 8-core CPU, 18GB unified memory               │    │
│  │  Agents: 5-10 concurrent                                            │    │
│  │  Latency: ~2-5s per request                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  TIER 2: Small Production (Single Server)                                    │
│  ═════════════════════════════════════════                                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  [Load Balancer]                                                    │    │
│  │       │                                                             │    │
│  │       ▼                                                             │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                    Application Server                        │    │    │
│  │  │  [FastAPI] ──► [Redis Queue] ──► [Worker Pool (4-8)]       │    │    │
│  │  │       │              │                 │                    │    │    │
│  │  │       ▼              ▼                 ▼                    │    │    │
│  │  │  [PostgreSQL]   [Redis Cache]    [Provider Router]         │    │    │
│  │  │                                         │                   │    │    │
│  │  │                    ┌────────────────────┼────────────┐      │    │    │
│  │  │                    ▼                    ▼            ▼      │    │    │
│  │  │              [Kimi/Claude]        [OpenAI]      [Ollama]    │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                     │    │
│  │  Resources: 32GB RAM, 16-core CPU, 1Gbps network                    │    │
│  │  Agents: 50-100 concurrent                                          │    │
│  │  Throughput: ~100 req/min                                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  TIER 3: Medium Scale (Kubernetes Cluster)                                   │
│  ═══════════════════════════════════════════                                 │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  [Ingress Controller] ──► [API Gateway (3 replicas)]                │    │
│  │                                    │                                │    │
│  │                                    ▼                                │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │              Message Queue Cluster (RabbitMQ/NATS)           │    │    │
│  │  │                     (3-node cluster)                         │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                    │                                │    │
│  │                    ┌───────────────┼───────────────┐                │    │
│  │                    ▼               ▼               ▼                │    │
│  │  ┌─────────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │    │
│  │  │   Worker Pool A     │ │  Worker Pool B  │ │  Worker Pool C  │   │    │
│  │  │   (General Tasks)   │ │ (Analysis Tasks)│ │ (Code Tasks)    │   │    │
│  │  │   (10 replicas)     │ │  (5 replicas)   │ │  (5 replicas)   │   │    │
│  │  └─────────────────────┘ └─────────────────┘ └─────────────────┘   │    │
│  │                                                                     │    │
│  │  [PostgreSQL HA] ──► [Redis Cluster] ──► [MinIO Object Store]     │    │
│  │                                                                     │    │
│  │  Resources: 3-10 nodes, 64GB+ RAM per node                          │    │
│  │  Agents: 500-2000 concurrent                                        │    │
│  │  Throughput: ~1000 req/min                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  TIER 4: Large Scale (H100 GPU Cluster)                                      │
│  ═══════════════════════════════════════                                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  [Global Load Balancer] ──► [Regional Clusters]                     │    │
│  │                                    │                                │    │
│  │  ┌───────────────────────────────┐ │ ┌─────────────────────────────┐│    │
│  │  │     GPU Compute Cluster       │ │ │    CPU Compute Cluster      ││    │
│  │  │  ┌─────────────────────────┐  │ │ │  ┌─────────────────────┐    ││    │
│  │  │  │  vLLM Inference Nodes   │  │ │ │  │  Worker Nodes       │    ││    │
│  │  │  │  (8x H100 per node)     │  │ │ │  │  (32 cores, 128GB)  │    ││    │
│  │  │  │  - Llama 3 70B          │  │ │ │  │                     │    ││    │
│  │  │  │  - Mixtral 8x22B        │  │ │ │  │                     │    ││    │
│  │  │  │  - Custom fine-tunes    │  │ │ │  │                     │    ││    │
│  │  │  └─────────────────────────┘  │ │ │  └─────────────────────┘    ││    │
│  │  │         (10-50 nodes)         │ │ │       (20-100 nodes)        ││    │
│  │  └───────────────────────────────┘ │ └─────────────────────────────┘│    │
│  │                                                                     │    │
│  │  [Ceph/S3 Storage] ──► [Kafka Cluster] ──► [ClickHouse Analytics] │    │
│  │                                                                     │    │
│  │  Resources: 100+ nodes, 3.2TB+ GPU memory                           │    │
│  │  Agents: 10,000+ concurrent                                         │    │
│  │  Throughput: ~10,000+ req/min                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Performance Targets

| Metric | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|--------|--------|--------|--------|--------|
| Concurrent Agents | 5-10 | 50-100 | 500-2000 | 10,000+ |
| Tasks/min | 10 | 100 | 1,000 | 10,000+ |
| Avg Latency | 2-5s | 1-3s | <1s | <500ms |
| Storage | SQLite | PostgreSQL | PostgreSQL HA | Distributed |
| Message Queue | In-memory | Redis | RabbitMQ/NATS | Kafka |

---

## 9. Conclusion

autoconstitution represents a significant evolution of Karpathy's autoresearch paradigm, extending it from a single-agent system to a massively parallel collaborative platform. The architecture emphasizes:

1. **Scalability**: From a single Mac Mini to H100 clusters
2. **Reliability**: Ratchet mechanism guarantees progress
3. **Flexibility**: Provider-agnostic design
4. **Efficiency**: Dynamic reallocation and cross-pollination
5. **Simplicity**: Extended minimal philosophy

The system fills a critical gap identified in the research: no existing implementation combines parallel multi-agent exploration with cross-agent code sharing in the autoresearch paradigm. autoconstitution aims to be the first.

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **PARL** | Parallel Autonomous Research Layer - the core architecture pattern |
| **Ratchet** | Mechanism ensuring progress only moves forward |
| **Cross-Pollination** | Sharing of findings between agents |
| **Task DAG** | Directed Acyclic Graph for task dependencies |
| **Git Worktree** | Git feature for multiple working directories |
| **Branch** | Logical grouping of related tasks and agents |
| **Agent Pool** | Managed collection of sub-agents |

## Appendix B: References

1. Karpathy, A. (2026). autoresearch. GitHub repository.
2. Shen, Y. et al. (2026). An Empirical Study of Multi-Agent Collaboration for Automated Research. arXiv:2603.29632
3. Schmidgall, S. et al. (2025). AgentRxiv: Towards Collaborative Autonomous Research. arXiv:2503.18102
4. Sun, Q. et al. (2026). KernelSkill: A Multi-Agent Framework for GPU Kernel Optimization. arXiv:2603.10085

---

*Document Version: 1.0*  
*Generated: April 2026*
