# SwarmResearch System Architecture

## Executive Summary

SwarmResearch is a massively parallel collaborative AI research system designed to orchestrate hundreds of AI agents across multiple LLM providers. The architecture follows a **layered, message-driven design** that scales from a single Mac Mini M4 to an H100 GPU cluster while maintaining clean separation of concerns and Karpathy's minimalist philosophy.

---

## 1. High-Level Architecture Overview

### 1.1 Core Design Principles

1. **Provider Agnosticism**: All LLM providers implement a common interface
2. **Horizontal Scalability**: Stateless components enable linear scaling
3. **Message-Driven**: Async message passing decouples components
4. **Minimal Core**: Karpathy's 3-file philosophy extended, not replaced
5. **Progressive Enhancement**: Start simple, add complexity only when needed

### 1.2 System Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SWARM RESEARCH SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         ORCHESTRATION LAYER                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │   Swarm     │  │   Task      │  │   Result    │  │   Health    │  │   │
│  │  │  Manager    │  │  Scheduler  │  │  Aggregator │  │   Monitor   │  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │   │
│  │         └─────────────────┴─────────────────┴─────────────────┘      │   │
│  │                              │                                        │   │
│  │                    ┌─────────┴─────────┐                              │   │
│  │                    │  Message Bus      │ (Redis/RabbitMQ/NATS)       │   │
│  │                    │  (Pub/Sub Queue)  │                              │   │
│  │                    └─────────┬─────────┘                              │   │
│  └──────────────────────────────┼────────────────────────────────────────┘   │
│                                 │                                            │
│  ┌──────────────────────────────┼────────────────────────────────────────┐   │
│  │                    AGENT EXECUTION LAYER                              │   │
│  │  ┌───────────────────────────┼─────────────────────────────────────┐  │   │
│  │  │                    Agent Worker Pool                             │  │   │
│  │  │  ┌─────────┐ ┌─────────┐ │ ┌─────────┐ ┌─────────┐ ┌─────────┐ │  │   │
│  │  │  │ Worker 1│ │ Worker 2│ │ │ Worker N│ │ Worker..│ │ Worker..│ │  │   │
│  │  │  │ (Local) │ │ (Local) │ │ │ (Local) │ │ (Cloud) │ │ (Cloud) │ │  │   │
│  │  │  └────┬────┘ └────┬────┘ │ └────┬────┘ └────┬────┘ └────┬────┘ │  │   │
│  │  │       └───────────┴──────┼──────┴───────────┴───────────┘      │  │   │
│  │  │                          │                                      │  │   │
│  │  │  ┌───────────────────────┴───────────────────────┐              │  │   │
│  │  │  │         Provider Abstraction Layer            │              │  │   │
│  │  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────┐  │              │  │   │
│  │  │  │  │  Kimi   │ │ Claude  │ │ OpenAI  │ │Ollam│  │              │  │   │
│  │  │  │  │ Adapter │ │ Adapter │ │ Adapter │ │  a  │  │              │  │   │
│  │  │  │  └─────────┘ └─────────┘ └─────────┘ └─────┘  │              │  │   │
│  │  │  └───────────────────────────────────────────────┘              │  │   │
│  │  └─────────────────────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                 │                                            │
│  ┌──────────────────────────────┼────────────────────────────────────────┐   │
│  │                      STORAGE LAYER                                      │   │
│  │  ┌─────────────┐  ┌─────────┴─────────┐  ┌─────────────┐  ┌─────────┐  │   │
│  │  │  Vector DB  │  │   Object Store    │  │    Cache    │  │  State  │  │   │
│  │  │(Embeddings) │  │  (Artifacts/Logs) │  │   (Redis)   │  │  Store  │  │   │
│  │  │  ChromaDB   │  │    MinIO/S3       │  │             │  │ (SQLite │  │   │
│  │  │   pgvector  │  │                   │  │             │  │/Postgres│  │   │
│  │  └─────────────┘  └─────────────────┘  └─────────────┘  └─────────┘  │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Architecture

### 2.1 Core Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPONENT DETAIL VIEW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  SWARM MANAGER (swarm_manager.py)                                      │  │
│  │  ─────────────────────────────────                                     │  │
│  │  Responsibilities:                                                     │  │
│  │    • Agent lifecycle management (spawn, monitor, terminate)            │  │
│  │    • Swarm topology configuration                                      │  │
│  │    • Resource allocation across providers                              │  │
│  │    • Fault tolerance and recovery                                      │  │
│  │                                                                        │  │
│  │  Interface:                                                            │  │
│  │    create_swarm(config: SwarmConfig) -> SwarmHandle                    │  │
│  │    spawn_agent(role: Role, provider: Provider) -> AgentHandle          │  │
│  │    broadcast(message: Message, filter: Filter) -> None                 │  │
│  │    get_swarm_state() -> SwarmState                                     │  │
│  │                                                                        │  │
│  │  Events:                                                               │  │
│  │    AGENT_SPAWNED, AGENT_COMPLETED, AGENT_FAILED, SWARM_REBALANCED      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  TASK SCHEDULER (task_scheduler.py)                                    │  │
│  │  ──────────────────────────────────                                    │  │
│  │  Responsibilities:                                                     │  │
│  │    • Task decomposition and dependency management                      │  │
│  │    • Priority queue management                                         │  │
│  │    • Load balancing across workers                                     │  │
│  │    • Deadlock detection and resolution                                 │  │
│  │                                                                        │  │
│  │  Interface:                                                            │  │
│  │    submit_task(task: Task) -> TaskHandle                               │  │
│  │    create_workflow(definition: Workflow) -> WorkflowHandle             │  │
│  │    get_task_status(handle: TaskHandle) -> TaskStatus                   │  │
│  │    cancel_task(handle: TaskHandle) -> bool                             │  │
│  │                                                                        │  │
│  │  Algorithms:                                                           │  │
│  │    • Work-stealing for load balancing                                  │  │
│  │    • Topological sort for dependency resolution                        │  │
│  │    • Backpressure for flow control                                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  AGENT WORKER (agent_worker.py) - Karpathy's Core Extended             │  │
│  │  ────────────────────────────────────────────────                      │  │
│  │  Responsibilities:                                                     │  │
│  │    • Execute individual agent tasks                                    │  │
│  │    • Manage conversation context                                       │  │
│  │    • Handle tool use and function calling                              │  │
│  │    • Report progress and results                                       │  │
│  │                                                                        │  │
│  │  Interface:                                                            │  │
│  │    execute(task: Task, context: Context) -> Result                     │  │
│  │    stream_execute(task: Task, context: Context) -> AsyncIterator       │  │
│  │    get_capabilities() -> List[Capability]                              │  │
│  │    update_system_prompt(prompt: str) -> None                           │  │
│  │                                                                        │  │
│  │  State Machine:                                                        │  │
│  │    IDLE -> EXECUTING -> [COMPLETED | FAILED | YIELDED] -> IDLE         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  RESULT AGGREGATOR (result_aggregator.py)                              │  │
│  │  ────────────────────────────────────────                              │  │
│  │  Responsibilities:                                                     │  │
│  │    • Collect and merge partial results                                 │  │
│  │    • Consensus building for voting patterns                            │  │
│  │    • Conflict resolution for divergent outputs                         │  │
│  │    • Output formatting and validation                                  │  │
│  │                                                                        │  │
│  │  Strategies:                                                           │  │
│  │    • Simple aggregation (concatenation, averaging)                     │  │
│  │    • Voting-based consensus                                            │  │
│  │    • Iterative refinement (meta-agent review)                          │  │
│  │    • Tree-of-thought synthesis                                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  HEALTH MONITOR (health_monitor.py)                                    │  │
│  │  ──────────────────────────────────                                    │  │
│  │  Responsibilities:                                                     │  │
│  │    • Track provider health and latency                                 │  │
│  │    • Detect and handle failures                                        │  │
│  │    • Automatic failover and recovery                                   │  │
│  │    • Performance metrics collection                                    │  │
│  │                                                                        │  │
│  │  Metrics:                                                              │  │
│  │    • Response latency (p50, p95, p99)                                  │  │
│  │    • Error rates by provider                                           │  │
│  │    • Token throughput                                                  │  │
│  │    • Cost per request                                                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Provider Abstraction Layer

### 3.1 Unified Provider Interface

```python
# Core abstraction - all providers implement this
class LLMProvider(ABC):
    """Unified interface for all LLM providers"""
    
    @abstractmethod
    async def complete(
        self, 
        messages: List[Message],
        config: GenerationConfig
    ) -> CompletionResult:
        """Generate completion from message history"""
        pass
    
    @abstractmethod
    async def stream_complete(
        self,
        messages: List[Message],
        config: GenerationConfig
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion tokens"""
        pass
    
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[Embedding]:
        """Generate embeddings for texts"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities and limits"""
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check provider availability"""
        pass
```

### 3.2 Provider Adapter Architecture

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
│  │ Streaming│   ✓     │    ✓     │   ✓    │    ✓    │   ✓    │              │
│  │ Functions│   ✓     │    ✓     │   ✓    │    ✓    │   ✓    │              │
│  │ Vision   │   ✓     │    ✓     │   ✓    │    ✗    │   ✓    │              │
│  │ Embeddings│   ✓    │    ✓     │   ✓    │    ✓    │   ✓    │              │
│  │ Local    │   ✗     │    ✗     │   ✗    │    ✓    │   ✓    │              │
│  │ Cost Track│   ✓    │    ✓     │   ✓    │    ✓    │   ✓    │              │
│  └──────────┴─────────┴──────────┴────────┴─────────┴────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Provider Router

```python
class ProviderRouter:
    """Intelligently routes requests to optimal provider"""
    
    def __init__(self, providers: List[LLMProvider], strategy: RoutingStrategy):
        self.providers = providers
        self.strategy = strategy
        self.health_monitor = HealthMonitor()
    
    async def route(
        self, 
        request: CompletionRequest,
        preferences: ProviderPreferences
    ) -> LLMProvider:
        """Select best provider based on strategy and health"""
        
        candidates = self._filter_by_capabilities(request)
        candidates = self._filter_by_health(candidates)
        candidates = self._filter_by_preferences(candidates, preferences)
        
        return self.strategy.select(candidates, request)
    
    # Routing strategies:
    # - LEAST_COST: Minimize API costs
    # - LOWEST_LATENCY: Fastest response
    # - HIGHEST_QUALITY: Best model for task
    # - ROUND_ROBIN: Distribute load evenly
    # - FALLBACK: Primary with automatic failover
```

---

## 4. Data Flow Architecture

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

### 4.2 Message Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MESSAGE FLOW (Async/Pub-Sub)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Topics:                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ tasks.new   │  │ tasks.assigned│  │ tasks.completed│  │ tasks.failed │    │
│  │ (pub: API)  │  │ (pub: Sched)  │  │ (pub: Worker)  │  │ (pub: Worker)│    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ agents.heartbeat│  │ agents.spawn  │  │ agents.terminate│  │ swarm.rebalance│
│  │ (pub: Agent)    │  │ (pub: Manager)│  │ (pub: Manager)  │  │ (pub: Monitor) │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                                              │
│  Message Structure:                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ {                                                                   │    │
│  │   "message_id": "uuid",                                             │    │
│  │   "timestamp": "ISO8601",                                           │    │
│  │   "topic": "tasks.new",                                             │    │
│  │   "payload": { ... },                                               │    │
│  │   "metadata": {                                                     │    │
│  │     "priority": 5,                                                  │    │
│  │     "correlation_id": "uuid",                                       │    │
│  │     "source": "api-gateway",                                        │    │
│  │     "ttl": 3600                                                     │    │
│  │   }                                                                 │    │
│  │ }                                                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Flow Example:                                                               │
│                                                                              │
│  Client ──► [tasks.new] ──► Scheduler ──► [tasks.assigned] ──► Worker      │
│                                              │                               │
│                                              ▼                               │
│  Client ◄── [tasks.completed] ◄── Aggregator ◄── [result] ◄── Worker       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 State Management

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STATE MANAGEMENT PATTERNS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Agent State (Ephemeral):                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Stored in: Worker memory (lost on restart)                         │    │
│  │  Contents: Current conversation, active tools, temp variables       │    │
│  │  Pattern: Checkpoint to Redis every N turns or on yield             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Task State (Persistent):                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Stored in: PostgreSQL/SQLite                                       │    │
│  │  Contents: Task definition, status, results, dependencies           │    │
│  │  Pattern: Event-sourced, immutable history                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Swarm State (Distributed):                                                  │
┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Stored in: Redis/ETCD                                              │    │
│  │  Contents: Active agents, topology, health metrics                  │    │
│  │  Pattern: Consensus-based, replicated across nodes                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Conversation State (Vector):                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Stored in: ChromaDB/pgvector                                       │    │
│  │  Contents: Message embeddings, semantic search index                │    │
│  │  Pattern: Incremental updates, approximate nearest neighbor         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  State Transitions:                                                          │
│                                                                              │
│  Task: PENDING ──► ASSIGNED ──► RUNNING ──► [COMPLETED | FAILED | CANCELLED]│
│                         │          │                                        │
│                         ▼          ▼                                        │
│                    TIMEOUT ◄── YIELDED ──► RESUMED                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Scaling Architecture

### 5.1 Deployment Tiers

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
│  │                                                              │    │
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

### 5.2 Auto-Scaling Configuration

```yaml
# Kubernetes HPA Configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: swarm-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: swarm-worker
  minReplicas: 3
  maxReplicas: 100
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: queue_depth
        target:
          type: AverageValue
          averageValue: "10"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
```

---

## 6. Design Decisions & Rationale

### 6.1 Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Message Bus** | Redis → RabbitMQ/NATS | Redis for simple deployments, RabbitMQ for complex routing, NATS for cloud-native |
| **State Storage** | SQLite → PostgreSQL | SQLite for single-node, PostgreSQL for HA and complex queries |
| **Vector DB** | ChromaDB → pgvector | ChromaDB for embedded, pgvector for unified SQL+vector |
| **API Framework** | FastAPI | Native async, automatic OpenAPI, Python-native |
| **Worker Framework** | Celery → Ray | Celery for simple tasks, Ray for distributed ML workloads |
| **Container Orchestration** | Docker Compose → K8s | Compose for dev/small, K8s for production scale |
| **Provider Abstraction** | Adapter Pattern | Clean separation, easy to add new providers |
| **Agent Communication** | Message Bus over RPC | Better decoupling, natural async, easier scaling |

### 6.2 Key Design Rationale

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

#### Why Provider Abstraction Layer?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROVIDER ABSTRACTION BENEFITS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Without Abstraction:                                                        │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                   │
│  │  Kimi   │    │ Claude  │    │ OpenAI  │    │ Ollama  │                   │
│  │ Client  │    │ Client  │    │ Client  │    │ Client  │                   │
│  │ (unique)│    │ (unique)│    │ (unique)│    │ (unique)│                   │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘                   │
│       │              │              │              │                         │
│       └──────────────┴──────────────┴──────────────┘                         │
│                      │                                                       │
│                 [Business Logic]  ← Duplicated provider handling             │
│                                                                              │
│  With Abstraction:                                                           │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                   │
│  │  Kimi   │    │ Claude  │    │ OpenAI  │    │ Ollama  │                   │
│  │ Adapter │    │ Adapter │    │ Adapter │    │ Adapter │                   │
│  │(standard│    │(standard│    │(standard│    │(standard│                   │
│  │interface)│    │interface)│    │interface)│    │interface)│                  │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘                   │
│       │              │              │              │                         │
│       └──────────────┴──────────────┴──────────────┘                         │
│                      │                                                       │
│              [Provider Router]  ← Single point of routing                    │
│                      │                                                       │
│                 [Business Logic]  ← Provider-agnostic code                   │
│                                                                              │
│  Benefits:                                                                   │
│  1. Add new provider = implement 5 methods                                   │
│  2. Swap providers without changing business logic                           │
│  3. A/B testing across providers                                             │
│  4. Automatic failover between providers                                     │
│  5. Cost optimization through intelligent routing                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Why Extend Karpathy's Structure?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KARPATHY'S 3-FILE PHILOSOPHY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Original (llm.c style):                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  train.py     - Training loop, optimization, checkpointing          │    │
│  │  model.py     - Model architecture, forward/backward pass           │    │
│  │  data.py      - Data loading, preprocessing, batching               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  SwarmResearch Extension:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  agent_worker.py      ←→  train.py  (core execution loop)           │    │
│  │  swarm_manager.py     ←→  model.py  (orchestration topology)        │    │
│  │  task_scheduler.py    ←→  data.py   (task flow and distribution)    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Extension Principle:                                                        │
│  - Keep core files under 500 lines each                                      │
│  - Extract complexity into pluggable adapters                                │
│  - Configuration over code for behavior changes                              │
│  - Each file has single, clear responsibility                                │
│                                                                              │
│  Additional Files (infrastructure):                                          │
│  - providers/        - Provider adapters (one per provider)                  │
│  - storage/          - Storage backends (pluggable)                          │
│  - messaging/        - Message bus implementations                           │
│  - schemas/          - Pydantic models for type safety                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Technology Stack Recommendations

### 7.1 Core Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RECOMMENDED TECHNOLOGY STACK                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Layer                    Technology              Purpose                    │
│  ──────────────────────────────────────────────────────────────────────────  │
│  Language                 Python 3.11+            Async-native, rich ecosystem│
│  API Framework            FastAPI                 High-performance async API  │
│  Type Validation          Pydantic v2             Runtime type safety         │
│  Async Runtime            asyncio + uvloop        High-performance event loop │
│  HTTP Client              httpx                   Async HTTP with HTTP/2      │
│  Testing                  pytest + pytest-asyncio Comprehensive testing       │
│                                                                              │
│  Message Queue (Simple)   Redis                   In-memory, fast, familiar   │
│  Message Queue (Complex)  RabbitMQ                Advanced routing, AMQP      │
│  Message Queue (Cloud)    NATS                    Cloud-native, lightweight   │
│                                                                              │
│  Database (Simple)        SQLite                  Zero-config, portable       │
│  Database (Production)    PostgreSQL 15+          Robust, scalable, JSONB     │
│  Database (Distributed)   CockroachDB             Geo-distributed SQL         │
│                                                                              │
│  Vector DB (Embedded)     ChromaDB                Easy setup, good defaults   │
│  Vector DB (Unified)      pgvector                Single database solution    │
│  Vector DB (Scale)        Pinecone/Weaviate       Managed, high performance   │
│                                                                              │
│  Cache                    Redis                   Fast key-value, pub/sub     │
│  Object Storage           MinIO (S3-compatible)   Self-hosted object store    │
│                                                                              │
│  Observability            OpenTelemetry + Jaeger  Distributed tracing         │
│  Metrics                  Prometheus + Grafana    Time-series metrics         │
│  Logging                  structlog + Loki        Structured logging          │
│                                                                              │
│  Containerization         Docker                  Consistent deployments      │
│  Orchestration (Simple)   Docker Compose          Single-node orchestration   │
│  Orchestration (Scale)    Kubernetes              Production orchestration    │
│                                                                              │
│  Local LLM Runtime        Ollama                  Easy local model serving    │
│  Local LLM Runtime (GPU)  vLLM                    High-throughput inference   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Dependency Matrix by Scale

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEPENDENCIES BY DEPLOYMENT TIER                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Package              Dev    Small    Medium    Large    Purpose             │
│  ──────────────────────────────────────────────────────────────────────────  │
│  fastapi              ✓      ✓        ✓         ✓        Web API             │
│  uvicorn[standard]    ✓      ✓        ✓         ✓        ASGI server         │
│  pydantic             ✓      ✓        ✓         ✓        Data validation     │
│  httpx                ✓      ✓        ✓         ✓        HTTP client         │
│  aiosqlite            ✓      ✓        -         -        Async SQLite        │
│  asyncpg              -      -        ✓         ✓        Async PostgreSQL    │
│  redis                -      ✓        ✓         ✓        Redis client        │
│  pika                 -      -        ✓         -        RabbitMQ client     │
│  nats-py              -      -        -         ✓        NATS client         │
│  chromadb             ✓      ✓        -         -        Vector DB (local)   │
│  pgvector             -      -        ✓         ✓        PostgreSQL vectors  │
│  openai               ✓      ✓        ✓         ✓        OpenAI SDK          │
│  anthropic            ✓      ✓        ✓         ✓        Claude SDK          │
│  ollama               ✓      ✓        ✓         ✓        Ollama client       │
│  prometheus-client    -      ✓        ✓         ✓        Metrics             │
│  opentelemetry        -      -        ✓         ✓        Distributed tracing │
│  structlog            ✓      ✓        ✓         ✓        Structured logging  │
│  pytest               ✓      ✓        ✓         ✓        Testing             │
│  ray                  -      -        -         ✓        Distributed compute │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Extension Points

### 8.1 Plugin Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXTENSION POINTS ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. CUSTOM PROVIDERS                                                         │
│  ═══════════════════                                                         │
│                                                                              │
│  Location: providers/custom/                                                 │
│                                                                              │
│  class MyCustomProvider(LLMProvider):                                        │
│      """Example: Grok, Cohere, or internal model"""                          │
│                                                                              │
│      async def complete(self, messages, config):                             │
│          # Implementation                                                    │
│          pass                                                                │
│                                                                              │
│  Registration:                                                               │
│      PROVIDER_REGISTRY.register("mycustom", MyCustomProvider)                │
│                                                                              │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  2. CUSTOM TOOLS                                                             │
│  ═══════════════════                                                         │
│                                                                              │
│  Location: tools/                                                            │
│                                                                              │
│  @tool_registry.register                                                     │
│  async def web_search(query: str, max_results: int = 5) -> SearchResult:     │
│      """Search the web for information"""                                    │
│      # Implementation using DuckDuckGo, SerpAPI, etc.                        │
│      pass                                                                    │
│                                                                              │
│  Built-in Tools:                                                             │
│  - web_search, code_execute, file_read, file_write                           │
│  - git_operations, database_query, api_call                                  │
│  - image_generate, document_parse                                            │
│                                                                              │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  3. CUSTOM AGENT ROLES                                                       │
│  ═══════════════════════                                                     │
│                                                                              │
│  Location: roles/                                                            │
│                                                                              │
│  researcher_role = AgentRole(                                                │
│      name="deep_researcher",                                                 │
│      system_prompt=RESEARCHER_PROMPT,                                        │
│      tools=[web_search, paper_analyzer],                                     │
│      provider_preferences={                                                  │
│          "default": "claude",                                                │
│          "fallback": "kimi"                                                  │
│      },                                                                      │
│      max_iterations=10,                                                      │
│      output_schema=ResearchOutput                                            │
│  )                                                                           │
│                                                                              │
│  Built-in Roles:                                                             │
│  - researcher, coder, reviewer, architect, tester, writer                    │
│  - planner, executor, critic, synthesizer                                    │
│                                                                              │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  4. CUSTOM WORKFLOWS                                                         │
│  ═══════════════════════                                                     │
│                                                                              │
│  Location: workflows/                                                        │
│                                                                              │
│  research_workflow = Workflow(                                               │
│      name="comprehensive_research",                                          │
│      steps=[                                                                 │
│          Step("decompose", role="planner", parallel=False),                  │
│          Step("research", role="researcher", parallel=True, map_input=True), │
│          Step("synthesize", role="synthesizer", parallel=False),             │
│          Step("review", role="reviewer", parallel=False)                     │
│      ]                                                                       │
│  )                                                                           │
│                                                                              │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  5. CUSTOM STORAGE BACKENDS                                                  │
│  ════════════════════════════                                                │
│                                                                              │
│  Location: storage/backends/                                                 │
│                                                                              │
│  class MyStorageBackend(StorageBackend):                                     │
│      """Example: MongoDB, DynamoDB, etc."""                                  │
│                                                                              │
│      async def save(self, key: str, data: dict) -> None:                     │
│          pass                                                                │
│                                                                              │
│      async def load(self, key: str) -> dict:                                 │
│          pass                                                                │
│                                                                              │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  6. CUSTOM AGGREGATION STRATEGIES                                            │
│  ══════════════════════════════════                                          │
│                                                                              │
│  Location: aggregation/strategies/                                           │
│                                                                              │
│  class VotingConsensus(AggregationStrategy):                                 │
│      """Aggregate results through voting"""                                  │
│                                                                              │
│      async def aggregate(self, results: List[Result]) -> Result:             │
│          votes = self._collect_votes(results)                                │
│          winner = self._majority_vote(votes)                                 │
│          return self._build_consensus_result(winner, results)                │
│                                                                              │
│  Built-in Strategies:                                                        │
│  - SimpleConcat, AverageEmbedding, VotingConsensus                           │
│  - IterativeRefinement, TreeOfThought, Debate                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Configuration-Driven Behavior

```yaml
# swarm_config.yaml - Example configuration
version: "1.0"

# Provider Configuration
providers:
  kimi:
    api_key: ${KIMI_API_KEY}
    base_url: "https://api.moonshot.cn/v1"
    default_model: "kimi-k2.5"
    timeout: 120
    max_retries: 3
    
  claude:
    api_key: ${ANTHROPIC_API_KEY}
    default_model: "claude-3-5-sonnet-20241022"
    timeout: 120
    max_retries: 3
    
  openai:
    api_key: ${OPENAI_API_KEY}
    default_model: "gpt-4o"
    timeout: 60
    max_retries: 3
    
  ollama:
    base_url: "http://localhost:11434"
    default_model: "mistral:7b"
    timeout: 300

# Swarm Configuration
swarm:
  max_agents: 100
  default_provider: "kimi"
  fallback_chain: ["claude", "openai", "ollama"]
  
  # Auto-scaling
  scaling:
    enabled: true
    min_workers: 2
    max_workers: 50
    scale_up_threshold: 0.8
    scale_down_threshold: 0.3
    
  # Health monitoring
  health_check:
    interval: 30
    timeout: 10
    unhealthy_threshold: 3

# Task Scheduling
scheduler:
  queue_type: "redis"  # redis | rabbitmq | nats | memory
  default_priority: 5
  max_queue_depth: 10000
  task_timeout: 600
  
  # Load balancing
  strategy: "least_loaded"  # round_robin | least_loaded | random

# Storage
storage:
  state:
    backend: "sqlite"  # sqlite | postgres | cockroachdb
    connection_string: "sqlite:///swarm.db"
    
  vector:
    backend: "chromadb"  # chromadb | pgvector | pinecone
    collection: "swarm_embeddings"
    
  cache:
    backend: "redis"
    url: "redis://localhost:6379"
    ttl: 3600

# Observability
observability:
  logging:
    level: "INFO"
    format: "json"
    
  metrics:
    enabled: true
    port: 9090
    
  tracing:
    enabled: true
    exporter: "jaeger"
    jaeger_endpoint: "http://localhost:14268/api/traces"
```

---

## 9. API Interface Specification

### 9.1 REST API Endpoints

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        REST API SPECIFICATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Research Operations                                                          │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  POST   /api/v1/research              Submit new research task               │
│  GET    /api/v1/research/{id}         Get research status/result             │
│  DELETE /api/v1/research/{id}         Cancel research task                   │
│  GET    /api/v1/research/{id}/stream  Stream research progress               │
│                                                                              │
│  Request Body (POST /api/v1/research):                                       │
│  {                                                                           │
│    "query": "Research quantum computing applications",                       │
│    "depth": "comprehensive",  # quick | standard | comprehensive             │
│    "agents": 10,               # Number of parallel agents                   │
│    "workflow": "research_v2",  # Predefined workflow name                    │
│    "providers": ["kimi", "claude"],  # Preferred providers                   │
│    "output_format": "markdown",  # markdown | json | structured              │
│    "constraints": {                                                          │
│      "max_tokens": 100000,                                                   │
│      "max_cost": 10.00,                                                      │
│      "timeout": 300                                                          │
│    }                                                                         │
│  }                                                                           │
│                                                                              │
│  Response:                                                                   │
│  {                                                                           │
│    "research_id": "uuid",                                                    │
│    "status": "queued",                                                       │
│    "estimated_duration": 120,                                                │
│    "estimated_cost": 5.50,                                                   │
│    "websocket_url": "/ws/research/uuid"                                      │
│  }                                                                           │
│                                                                              │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  Swarm Management                                                             │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  GET    /api/v1/swarm/status          Get swarm health and metrics           │
│  POST   /api/v1/swarm/agents          Spawn new agent                        │
│  DELETE /api/v1/swarm/agents/{id}     Terminate agent                        │
│  POST   /api/v1/swarm/rebalance       Trigger manual rebalancing             │
│                                                                              │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  Provider Management                                                          │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  GET    /api/v1/providers             List configured providers              │
│  GET    /api/v1/providers/{name}/health  Check provider health               │
│  POST   /api/v1/providers/{name}/enable  Enable provider                     │
│  POST   /api/v1/providers/{name}/disable Disable provider                    │
│                                                                              │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  Workflow Management                                                          │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  GET    /api/v1/workflows             List available workflows               │
│  POST   /api/v1/workflows             Create custom workflow                 │
│  GET    /api/v1/workflows/{name}      Get workflow definition                │
│  PUT    /api/v1/workflows/{name}      Update workflow                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 WebSocket Events

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WEBSOCKET EVENT SPECIFICATION                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Connection: ws://api/ws/research/{research_id}                              │
│                                                                              │
│  Client → Server Events:                                                     │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  { "type": "subscribe", "channels": ["progress", "results", "errors"] }      │
│  { "type": "cancel" }                                                        │
│  { "type": "pause" }                                                         │
│  { "type": "resume" }                                                        │
│                                                                              │
│  Server → Client Events:                                                     │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  {                                                                           │
│    "type": "task.started",                                                   │
│    "timestamp": "2024-01-15T10:30:00Z",                                      │
│    "data": {                                                                 │
│      "task_id": "uuid",                                                      │
│      "agent_id": "uuid",                                                     │
│      "description": "Searching for quantum computing papers"                 │
│    }                                                                         │
│  }                                                                           │
│                                                                              │
│  {                                                                           │
│    "type": "task.progress",                                                  │
│    "timestamp": "2024-01-15T10:30:15Z",                                      │
│    "data": {                                                                 │
│      "task_id": "uuid",                                                      │
│      "progress": 0.5,                                                        │
│      "status": "Found 25 relevant papers, analyzing..."                      │
│    }                                                                         │
│  }                                                                           │
│                                                                              │
│  {                                                                           │
│    "type": "task.completed",                                                 │
│    "timestamp": "2024-01-15T10:31:00Z",                                      │
│    "data": {                                                                 │
│      "task_id": "uuid",                                                      │
│      "result": { ... },                                                      │
│      "tokens_used": 5000,                                                    │
│      "cost": 0.15                                                            │
│    }                                                                         │
│  }                                                                           │
│                                                                              │
│  {                                                                           │
│    "type": "research.completed",                                             │
│    "timestamp": "2024-01-15T10:35:00Z",                                      │
│    "data": {                                                                 │
│      "research_id": "uuid",                                                  │
│      "final_result": { ... },                                                │
│      "total_tokens": 50000,                                                  │
│      "total_cost": 1.50,                                                     │
│      "duration": 300                                                         │
│    }                                                                         │
│  }                                                                           │
│                                                                              │
│  {                                                                           │
│    "type": "error",                                                          │
│    "timestamp": "2024-01-15T10:32:00Z",                                      │
│    "data": {                                                                 │
│      "code": "PROVIDER_TIMEOUT",                                             │
│      "message": "Kimi provider timed out after 120s",                        │
│      "recoverable": true,                                                    │
│      "fallback_action": "retrying_with_claude"                               │
│    }                                                                         │
│  }                                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Security Considerations

### 10.1 Security Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SECURITY ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Layer 1: API Security                                                       │
│  ─────────────────────                                                       │
│  • API Key authentication (X-API-Key header)                                 │
│  • Rate limiting per API key (token bucket)                                  │
│  • Request size limits (10MB max)                                            │
│  • Input validation (Pydantic schemas)                                       │
│  • CORS configuration                                                        │
│                                                                              │
│  Layer 2: Provider Security                                                  │
│  ──────────────────────────                                                  │
│  • API keys stored in environment variables or secrets manager               │
│  • Key rotation support                                                      │
│  • Request/response logging (sanitized)                                      │
│  • Circuit breaker for failed providers                                      │
│                                                                              │
│  Layer 3: Agent Sandbox                                                      │
│  ───────────────────────                                                     │
│  • Code execution in isolated containers (gVisor/Firecracker)                │
│  • File system restrictions (read-only root, tmpfs for writes)               │
│  • Network egress controls (whitelist only)                                  │
│  • Resource limits (CPU, memory, time)                                       │
│                                                                              │
│  Layer 4: Data Protection                                                    │
│  ────────────────────────                                                    │
│  • Encryption at rest (database, object storage)                             │
│  • Encryption in transit (TLS 1.3)                                           │
│  • PII detection and redaction                                               │
│  • Data retention policies                                                   │
│                                                                              │
│  Layer 5: Audit & Compliance                                                 │
│  ────────────────────────────                                                │
│  • Complete audit log of all operations                                      │
│  • Non-repudiation through signed logs                                       │
│  • Compliance exports (GDPR, CCPA)                                           │
│  • Anomaly detection                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Deployment Patterns

### 11.1 Docker Compose (Development)

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - KIMI_API_KEY=${KIMI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=sqlite:///data/swarm.db
    volumes:
      - ./data:/app/data
    depends_on:
      - redis
      - ollama

  worker:
    build: .
    command: celery -A swarmresearch worker --loglevel=info
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=sqlite:///data/swarm.db
    volumes:
      - ./data:/app/data
    depends_on:
      - redis
    deploy:
      replicas: 2

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  redis_data:
  ollama_data:
```

### 11.2 Kubernetes (Production)

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: swarm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: swarm-api
  template:
    metadata:
      labels:
        app: swarm-api
    spec:
      containers:
        - name: api
          image: swarmresearch/api:latest
          ports:
            - containerPort: 8000
          envFrom:
            - secretRef:
                name: swarm-secrets
            - configMapRef:
                name: swarm-config
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: swarm-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: swarm-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

---

## 12. Monitoring & Observability

### 12.1 Key Metrics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        KEY METRICS DASHBOARD                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Business Metrics                                                            │
│  ──────────────────────────────────────────────────────────────────────────  │
│  • Research tasks completed per hour                                         │
│  • Average research completion time                                          │
│  • Cost per research task                                                    │
│  • User satisfaction score                                                   │
│                                                                              │
│  System Metrics                                                              │
│  ──────────────────────────────────────────────────────────────────────────  │
│  • API request rate (req/sec)                                                │
│  • API latency (p50, p95, p99)                                               │
│  • Error rate by endpoint                                                    │
│  • Active agent count                                                        │
│  • Queue depth                                                               │
│                                                                              │
│  Provider Metrics                                                            │
│  ──────────────────────────────────────────────────────────────────────────  │
│  • Requests per provider                                                     │
│  • Provider latency (p50, p95, p99)                                          │
│  • Provider error rate                                                       │
│  • Token throughput per provider                                             │
│  • Cost per provider                                                         │
│                                                                              │
│  Resource Metrics                                                            │
│  ──────────────────────────────────────────────────────────────────────────  │
│  • CPU utilization by service                                                │
│  • Memory utilization by service                                             │
│  • GPU utilization (if applicable)                                           │
│  • Disk I/O                                                                  │
│  • Network throughput                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Summary

### 13.1 Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ARCHITECTURE SUMMARY                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  CORE FILES (Karpathy Style)                                        │    │
│  │  ├── swarm_manager.py      # Agent orchestration                    │    │
│  │  ├── task_scheduler.py     # Task distribution                      │    │
│  │  └── agent_worker.py       # Agent execution                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ADAPTER LAYERS (Pluggable)                                         │    │
│  │  ├── providers/            # LLM provider adapters                  │    │
│  │  ├── storage/              # Database backends                      │    │
│  │  ├── messaging/            # Message queue implementations          │    │
│  │  └── tools/                # Agent tool implementations             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  CONFIGURATION                                                      │    │
│  │  ├── swarm_config.yaml     # Main configuration                     │    │
│  │  ├── roles/                # Agent role definitions                 │    │
│  │  └── workflows/            # Research workflow definitions          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  DEPLOYMENT                                                         │    │
│  │  ├── docker-compose.yml    # Development/ small scale               │    │
│  │  ├── k8s/                  # Kubernetes manifests                   │    │
│  │  └── helm/                 # Helm charts                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  KEY FEATURES:                                                               │
│  ✓ Provider-agnostic (Kimi, Claude, OpenAI, Ollama, vLLM)                    │
│  ✓ Scales from Mac Mini M4 to H100 cluster                                   │
│  ✓ Message-driven async architecture                                         │
│  ✓ Extensible plugin system                                                  │
│  ✓ Configuration-driven behavior                                             │
│  ✓ Production-ready observability                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: File Structure

```
swarmresearch/
├── core/                           # Core 3 files (Karpathy style)
│   ├── __init__.py
│   ├── swarm_manager.py           # ~400 lines
│   ├── task_scheduler.py          # ~400 lines
│   └── agent_worker.py            # ~400 lines
│
├── providers/                      # Provider adapters
│   ├── __init__.py
│   ├── base.py                    # LLMProvider abstract class
│   ├── kimi.py                    # Kimi K2.5 adapter
│   ├── claude.py                  # Claude adapter
│   ├── openai.py                  # OpenAI adapter
│   ├── ollama.py                  # Ollama adapter
│   ├── vllm.py                    # vLLM adapter
│   └── router.py                  # Provider routing logic
│
├── storage/                        # Storage backends
│   ├── __init__.py
│   ├── base.py                    # StorageBackend abstract class
│   ├── sqlite.py                  # SQLite implementation
│   ├── postgres.py                # PostgreSQL implementation
│   ├── redis_cache.py             # Redis cache implementation
│   └── vector/
│       ├── base.py
│       ├── chroma.py
│       └── pgvector.py
│
├── messaging/                      # Message bus implementations
│   ├── __init__.py
│   ├── base.py                    # MessageBus abstract class
│   ├── memory.py                  # In-memory (dev)
│   ├── redis_bus.py               # Redis pub/sub
│   ├── rabbitmq.py                # RabbitMQ
│   └── nats.py                    # NATS
│
├── tools/                          # Agent tools
│   ├── __init__.py
│   ├── base.py                    # Tool base class
│   ├── registry.py                # Tool registration
│   ├── web_search.py              # Web search tool
│   ├── code_execute.py            # Code execution tool
│   ├── file_operations.py         # File read/write
│   └── ...
│
├── roles/                          # Agent role definitions
│   ├── __init__.py
│   ├── base.py                    # AgentRole class
│   ├── researcher.py
│   ├── coder.py
│   ├── reviewer.py
│   └── ...
│
├── workflows/                      # Workflow definitions
│   ├── __init__.py
│   ├── base.py                    # Workflow class
│   ├── research.py                # Research workflows
│   └── ...
│
├── aggregation/                    # Result aggregation
│   ├── __init__.py
│   ├── base.py                    # AggregationStrategy
│   ├── simple.py                  # Simple concatenation
│   ├── voting.py                  # Voting consensus
│   └── refinement.py              # Iterative refinement
│
├── api/                            # REST API
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── routes/
│   │   ├── research.py
│   │   ├── swarm.py
│   │   ├── providers.py
│   │   └── workflows.py
│   └── middleware/
│       ├── auth.py
│       ├── rate_limit.py
│       └── logging.py
│
├── schemas/                        # Pydantic models
│   ├── __init__.py
│   ├── task.py
│   ├── agent.py
│   ├── message.py
│   └── result.py
│
├── config/                         # Configuration
│   ├── __init__.py
│   ├── settings.py                # Pydantic Settings
│   └── swarm_config.yaml          # Default config
│
├── observability/                  # Monitoring
│   ├── __init__.py
│   ├── metrics.py                 # Prometheus metrics
│   ├── tracing.py                 # OpenTelemetry
│   └── logging.py                 # Structured logging
│
├── deployment/                     # Deployment configs
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── k8s/
│       ├── deployment.yaml
│       ├── service.yaml
│       └── hpa.yaml
│
├── tests/                          # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── pyproject.toml                  # Project metadata
├── README.md
└── LICENSE
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Agent** | An AI entity with a specific role that can execute tasks |
| **Swarm** | A collection of agents working collaboratively |
| **Provider** | An LLM service (Kimi, Claude, OpenAI, Ollama, etc.) |
| **Adapter** | Code that translates provider-specific APIs to unified interface |
| **Task** | A unit of work assigned to an agent |
| **Workflow** | A predefined sequence of tasks with dependencies |
| **Message Bus** | Infrastructure for async communication between components |
| **Worker** | Process that executes agent tasks |
| **Router** | Component that selects optimal provider for each request |
| **Aggregator** | Component that combines results from multiple agents |

---

*Document Version: 1.0*  
*Last Updated: 2024*  
*Author: SwarmResearch Architecture Team*
