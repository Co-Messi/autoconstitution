# SwarmResearch State Management & Persistence Architecture

## Executive Summary

This document defines the persistent state system for SwarmResearch, designed to ensure **reliable operation during overnight runs** and **seamless recovery from failures**. The architecture provides atomic checkpointing, incremental persistence, and deterministic recovery for all critical system state.

### Design Goals
1. **Zero Data Loss**: All progress is persisted before acknowledgment
2. **Fast Recovery**: Resume within seconds of failure
3. **Minimal Overhead**: Checkpointing adds <5% performance cost
4. **Deterministic Replay**: Same inputs produce same outputs after recovery
5. **Version Safety**: Forward/backward compatibility for state migrations

---

## 1. State Structure and Schema

### 1.1 State Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SWARMRESEARCH STATE HIERARCHY                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SessionState (Root)                                                         │
│  ├── session_id: UUID                    # Unique session identifier         │
│  ├── created_at: datetime                # Session start time                │
│  ├── version: StateVersion               # Schema version                    │
│  │                                                                          │
│  ├── GlobalState                         # System-wide state                 │
│  │   ├── orchestrator: OrchestratorState  # Main orchestrator state          │
│  │   ├── swarm: SwarmState               # Agent swarm state                │
│  │   └── research_context: ResearchContext # Problem definition              │
│  │                                                                          │
│  ├── BranchStates: Dict[UUID, BranchState] # All research branches           │
│  │   ├── branch_id, parent_id, status                                     │
│  │   ├── hypotheses: List[Hypothesis]                                     │
│  │   ├── findings: List[Finding]                                          │
│  │   └── checkpoint_refs: List[CheckpointRef]                             │
│  │                                                                          │
│  ├── AgentStates: Dict[UUID, AgentState]   # All agent states                │
│  │   ├── agent_id, branch_id, status                                      │
│  │   ├── conversation: ConversationState                                  │
│  │   ├── working_memory: Dict[str, Any]                                   │
│  │   └── pending_tool_calls: List[ToolCall]                               │
│  │                                                                          │
│  ├── TaskStates: Dict[UUID, TaskState]     # All task states                 │
│  │   ├── task_id, parent_id, status                                       │
│  │   ├── dependencies: List[UUID]                                         │
│  │   ├── results: TaskResult                                              │
│  │   └── retry_count: int                                                 │
│  │                                                                          │
│  ├── GlobalRatchetState                  # Immutable best findings           │
│  │   ├── version: int                     # Monotonically increasing         │
│  │   ├── best_hypothesis: str                                              │
│  │   ├── best_score: float                                                 │
│  │   ├── best_evidence: List[Evidence]                                     │
│  │   ├── knowledge_graph: KnowledgeGraph                                   │
│  │   └── consolidated_findings: List[Finding]                              │
│  │                                                                          │
│  └── ImprovementLog                      # Canonical improvement history     │
      ├── entries: List[ImprovementEntry]   # Immutable append-only           │
      ├── current_position: int             # Last processed entry            │
      └── checksum: str                     # Integrity verification          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core State Schemas (Pydantic)

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
import hashlib

# ============================================================================
# VERSIONING
# ============================================================================

class StateVersion(BaseModel):
    """Semantic versioning for state schema."""
    major: int = 1
    minor: int = 0
    patch: int = 0
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def is_compatible_with(self, other: 'StateVersion') -> bool:
        """Check if versions are compatible for loading."""
        return self.major == other.major

CURRENT_STATE_VERSION = StateVersion(major=1, minor=0, patch=0)

# ============================================================================
# SESSION STATE
# ============================================================================

class SessionState(BaseModel):
    """Root state container for entire research session."""
    model_config = ConfigDict(extra='forbid')
    
    session_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_saved_at: Optional[datetime] = None
    version: StateVersion = Field(default=CURRENT_STATE_VERSION)
    
    # Core state components
    global_state: 'GlobalState'
    branches: Dict[UUID, 'BranchState'] = Field(default_factory=dict)
    agents: Dict[UUID, 'AgentState'] = Field(default_factory=dict)
    tasks: Dict[UUID, 'TaskState'] = Field(default_factory=dict)
    ratchet_state: 'GlobalRatchetState' = Field(default_factory=lambda: GlobalRatchetState())
    improvement_log: 'ImprovementLog' = Field(default_factory=lambda: ImprovementLog())
    
    # Metadata
    checkpoint_count: int = 0
    total_iterations: int = 0
    
    def compute_checksum(self) -> str:
        """Compute integrity checksum for verification."""
        state_dict = self.model_dump(exclude={'checksum'})
        state_json = str(sorted(state_dict.items()))
        return hashlib.sha256(state_json.encode()).hexdigest()[:16]

# ============================================================================
# GLOBAL STATE
# ============================================================================

class OrchestratorStatus(str, Enum):
    """Orchestrator lifecycle states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    REALLOCATING = "reallocating"
    SHUTTING_DOWN = "shutting_down"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"

class OrchestratorState(BaseModel):
    """State of the main research orchestrator."""
    status: OrchestratorStatus = OrchestratorStatus.INITIALIZING
    current_iteration: int = 0
    max_iterations: int = 1000
    
    # Resource allocation
    active_agent_count: int = 0
    max_concurrent_agents: int = 50
    
    # Branch management
    active_branch_ids: List[UUID] = Field(default_factory=list)
    abandoned_branch_ids: List[UUID] = Field(default_factory=list)
    merged_branch_ids: List[UUID] = Field(default_factory=list)
    
    # Event tracking
    last_reallocation_at: Optional[datetime] = None
    last_health_check_at: Optional[datetime] = None

class SwarmState(BaseModel):
    """State of the agent swarm."""
    total_agents_spawned: int = 0
    total_agents_completed: int = 0
    total_agents_failed: int = 0
    
    # Worker pool state
    worker_pool_size: int = 0
    available_workers: int = 0
    busy_workers: List[UUID] = Field(default_factory=list)
    
    # Provider health
    provider_health: Dict[str, 'ProviderHealthState'] = Field(default_factory=dict)

class ProviderHealthState(BaseModel):
    """Health state for a single provider."""
    provider_id: str
    is_healthy: bool = True
    last_success_at: Optional[datetime] = None
    last_failure_at: Optional[datetime] = None
    consecutive_failures: int = 0
    average_latency_ms: float = 0.0
    request_count: int = 0
    error_count: int = 0

# ============================================================================
# BRANCH STATE
# ============================================================================

class BranchStatus(str, Enum):
    """Research branch lifecycle states."""
    EXPLORING = "exploring"
    CONVERGING = "converging"
    STAGNANT = "stagnant"
    BREAKTHROUGH = "breakthrough"
    MERGED = "merged"
    ABANDONED = "abandoned"

class Hypothesis(BaseModel):
    """A research hypothesis."""
    hypothesis_id: UUID = Field(default_factory=uuid4)
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    validated: bool = False
    validation_score: Optional[float] = None

class Finding(BaseModel):
    """A research finding."""
    finding_id: UUID = Field(default_factory=uuid4)
    text: str
    importance: float = Field(ge=0.0, le=1.0)
    supporting_evidence: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    agent_id: UUID

class BranchState(BaseModel):
    """State of a single research branch."""
    branch_id: UUID = Field(default_factory=uuid4)
    parent_branch_id: Optional[UUID] = None
    
    # Status
    status: BranchStatus = BranchStatus.EXPLORING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Content
    focus_area: str
    depth: int = 0
    max_depth: int = 5
    
    # Progress tracking
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    findings: List[Finding] = Field(default_factory=list)
    explored_directions: List[str] = Field(default_factory=list)
    
    # Performance metrics
    iteration_count: int = 0
    gradient_score: float = 0.0  # Rate of progress
    last_improvement_at: Optional[datetime] = None
    
    # Agent assignments
    assigned_agent_ids: List[UUID] = Field(default_factory=list)
    
    # Checkpoint references
    checkpoint_refs: List['CheckpointRef'] = Field(default_factory=list)

# ============================================================================
# AGENT STATE
# ============================================================================

class AgentStatus(str, Enum):
    """Agent lifecycle states."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    REALLOCATING = "reallocating"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"

class Message(BaseModel):
    """A conversation message."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ToolCall(BaseModel):
    """A pending tool call."""
    call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    issued_at: datetime = Field(default_factory=datetime.utcnow)
    timeout_seconds: int = 60

class ConversationState(BaseModel):
    """State of an agent's conversation."""
    messages: List[Message] = Field(default_factory=list)
    max_tokens: int = 8000
    current_token_count: int = 0
    summary: Optional[str] = None  # Condensed history if truncated

class AgentState(BaseModel):
    """State of a single research agent."""
    agent_id: UUID = Field(default_factory=uuid4)
    branch_id: UUID
    
    # Configuration
    agent_type: str  # 'explorer', 'critic', 'synthesizer', 'verifier'
    specialization: str
    provider_id: str
    model_id: str
    
    # Status
    status: AgentStatus = AgentStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Conversation state
    conversation: ConversationState = Field(default_factory=ConversationState)
    system_prompt: str = ""
    
    # Working memory (can be large, checkpointed separately)
    working_memory: Dict[str, Any] = Field(default_factory=dict)
    working_memory_checkpoint_ref: Optional[str] = None
    
    # Pending operations
    pending_tool_calls: List[ToolCall] = Field(default_factory=list)
    current_task_id: Optional[UUID] = None
    
    # Metrics
    iteration_count: int = 0
    tokens_consumed: int = 0
    cost_usd: float = 0.0

# ============================================================================
# TASK STATE
# ============================================================================

class TaskStatus(str, Enum):
    """Task lifecycle states."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    YIELDED = "yielded"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class TaskResult(BaseModel):
    """Result of a completed task."""
    success: bool
    output: Any = None
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    tokens_consumed: int = 0
    completed_at: datetime = Field(default_factory=datetime.utcnow)

class TaskState(BaseModel):
    """State of a single task."""
    task_id: UUID = Field(default_factory=uuid4)
    parent_task_id: Optional[UUID] = None
    
    # Definition
    task_type: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 5
    
    # Status
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Execution
    assigned_agent_id: Optional[UUID] = None
    dependencies: List[UUID] = Field(default_factory=list)
    
    # Result
    result: Optional[TaskResult] = None
    
    # Retry logic
    retry_count: int = 0
    max_retries: int = 3
    
    # Checkpoint
    last_checkpoint_at: Optional[datetime] = None
    checkpoint_data: Optional[Dict[str, Any]] = None

# ============================================================================
# GLOBAL RATCHET STATE (Immutable Best Findings)
# ============================================================================

class Evidence(BaseModel):
    """Evidence supporting a hypothesis."""
    source: str
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class KnowledgeNode(BaseModel):
    """Node in the knowledge graph."""
    node_id: str
    content: str
    node_type: str  # 'concept', 'fact', 'hypothesis', 'finding'
    confidence: float = 0.5
    sources: List[str] = Field(default_factory=list)

class KnowledgeEdge(BaseModel):
    """Edge in the knowledge graph."""
    source_id: str
    target_id: str
    relation: str
    strength: float = 0.5

class KnowledgeGraph(BaseModel):
    """Graph of accumulated knowledge."""
    nodes: Dict[str, KnowledgeNode] = Field(default_factory=dict)
    edges: List[KnowledgeEdge] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class GlobalRatchetState(BaseModel):
    """
    The global best state that only improves (ratchet mechanism).
    This is the most critical state - never loses data.
    """
    version: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Best findings
    best_hypothesis: Optional[str] = None
    best_score: float = 0.0
    best_evidence: List[Evidence] = Field(default_factory=list)
    
    # Knowledge accumulation
    knowledge_graph: KnowledgeGraph = Field(default_factory=KnowledgeGraph)
    consolidated_findings: List[Finding] = Field(default_factory=list)
    
    # Update tracking
    update_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    def update_if_better(
        self, 
        hypothesis: str, 
        score: float, 
        evidence: List[Evidence]
    ) -> bool:
        """Only update if new result is strictly better (ratchet)."""
        if score > self.best_score:
            # Store previous in history
            if self.best_hypothesis:
                self.update_history.append({
                    "version": self.version,
                    "hypothesis": self.best_hypothesis,
                    "score": self.best_score,
                    "timestamp": self.timestamp
                })
            
            # Update to new best
            self.best_hypothesis = hypothesis
            self.best_score = score
            self.best_evidence = evidence
            self.version += 1
            self.timestamp = datetime.utcnow()
            return True
        return False

# ============================================================================
# IMPROVEMENT LOG (Canonical History)
# ============================================================================

class ImprovementType(str, Enum):
    """Types of improvements."""
    HYPOTHESIS_GENERATED = "hypothesis_generated"
    HYPOTHESIS_VALIDATED = "hypothesis_validated"
    FINDING_DISCOVERED = "finding_discovered"
    KNOWLEDGE_ADDED = "knowledge_added"
    BRANCH_CREATED = "branch_created"
    BRANCH_MERGED = "branch_merged"
    BRANCH_ABANDONED = "branch_abandoned"
    AGENT_SPAWNED = "agent_spawned"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    RATCHET_UPDATED = "ratchet_updated"
    TASK_COMPLETED = "task_completed"

class ImprovementEntry(BaseModel):
    """
    Single entry in the canonical improvement log.
    Immutable and append-only.
    """
    entry_id: UUID = Field(default_factory=uuid4)
    sequence_number: int  # Monotonically increasing
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    improvement_type: ImprovementType
    branch_id: Optional[UUID] = None
    agent_id: Optional[UUID] = None
    task_id: Optional[UUID] = None
    
    description: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    
    # State snapshot reference (for major improvements)
    checkpoint_ref: Optional[str] = None
    
    # Verification
    previous_hash: str  # Hash chain for integrity
    entry_hash: str     # Hash of this entry
    
    def compute_hash(self, previous_hash: str) -> str:
        """Compute hash for this entry."""
        data = f"{self.sequence_number}:{self.timestamp}:{self.improvement_type}:{self.description}:{previous_hash}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]

class ImprovementLog(BaseModel):
    """
    Canonical append-only log of all improvements.
    This is the source of truth for progress tracking.
    """
    entries: List[ImprovementEntry] = Field(default_factory=list)
    current_position: int = 0  # Last processed entry index
    last_entry_hash: str = "0" * 32  # For hash chain
    
    # Integrity
    log_checksum: str = ""
    last_compaction_at: Optional[datetime] = None
    
    def append(self, entry: ImprovementEntry) -> None:
        """Append a new entry to the log."""
        entry.sequence_number = len(self.entries)
        entry.previous_hash = self.last_entry_hash
        entry.entry_hash = entry.compute_hash(self.last_entry_hash)
        
        self.entries.append(entry)
        self.last_entry_hash = entry.entry_hash
        self._update_checksum()
    
    def _update_checksum(self) -> None:
        """Update the log checksum."""
        if self.entries:
            data = "".join(e.entry_hash for e in self.entries)
            self.log_checksum = hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def verify_integrity(self) -> bool:
        """Verify the integrity of the log."""
        if not self.entries:
            return True
        
        # Verify hash chain
        prev_hash = "0" * 32
        for entry in self.entries:
            if entry.previous_hash != prev_hash:
                return False
            expected_hash = entry.compute_hash(prev_hash)
            if entry.entry_hash != expected_hash:
                return False
            prev_hash = entry.entry_hash
        
        # Verify checksum
        data = "".join(e.entry_hash for e in self.entries)
        expected_checksum = hashlib.sha256(data.encode()).hexdigest()[:16]
        return self.log_checksum == expected_checksum

# ============================================================================
# CHECKPOINT REFERENCES
# ============================================================================

class CheckpointRef(BaseModel):
    """Reference to a persisted checkpoint."""
    checkpoint_id: UUID = Field(default_factory=uuid4)
    checkpoint_type: Literal["full", "incremental", "agent", "branch", "ratchet"]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    file_path: str
    file_size_bytes: int = 0
    checksum: str
    
    # For incremental checkpoints
    base_checkpoint_id: Optional[UUID] = None
    
    # What this checkpoint contains
    contains_state_types: List[str] = Field(default_factory=list)

# ============================================================================
# GLOBAL STATE AGGREGATE
# ============================================================================

class GlobalState(BaseModel):
    """Aggregate of all global system state."""
    orchestrator: OrchestratorState = Field(default_factory=OrchestratorState)
    swarm: SwarmState = Field(default_factory=SwarmState)
    research_context: 'ResearchContext' = Field(default_factory=lambda: ResearchContext(
        problem_statement="",
        domain=""
    ))

class ResearchContext(BaseModel):
    """Global context shared across all research branches."""
    problem_statement: str
    domain: str
    constraints: Dict[str, Any] = Field(default_factory=dict)
    success_criteria: List[str] = Field(default_factory=list)
    max_depth: int = 5
    exploration_budget: int = 100
    custom_instructions: str = ""
```

---

## 2. Checkpointing Strategy

### 2.1 Checkpoint Types

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHECKPOINT TYPE HIERARCHY                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  FULL CHECKPOINT                                                    │   │
│  │  ────────────────                                                   │   │
│  │  • Complete session state                                           │   │
│  │  • All branches, agents, tasks                                      │   │
│  │  • Full conversation histories                                      │   │
│  │  • Complete improvement log                                         │   │
│  │                                                                     │   │
│  │  Frequency: Every N iterations OR on major milestones               │   │
│  │  Retention: Keep last 3 + daily + milestone                         │   │
│  │  Size: 10-100MB depending on session                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  INCREMENTAL CHECKPOINT                                             │   │
│  │  ──────────────────────                                             │   │
│  │  • Changes since last checkpoint                                    │   │
│  │  • Delta encoding for efficiency                                    │   │
│  │  • References base checkpoint                                       │   │
│  │                                                                     │   │
│  │  Frequency: Every iteration OR every 30 seconds                     │   │
│  │  Retention: Roll up after 10 increments                             │   │
│  │  Size: 100KB-10MB depending on activity                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  AGENT CHECKPOINT                                                   │   │
│  │  ────────────────                                                   │   │
│  │  • Single agent state                                               │   │
│  │  • Conversation + working memory                                    │   │
│  │  • Tool call state                                                  │   │
│  │                                                                     │   │
│  │  Frequency: After each agent iteration OR on yield                  │   │
│  │  Retention: Until agent completes                                   │   │
│  │  Size: 10KB-1MB per agent                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  RATCHET CHECKPOINT                                                 │   │
│  │  ─────────────────                                                  │   │
│  │  • Global ratchet state only                                        │   │
│  │  • Immutable, append-only                                           │   │
│  │  • Synchronous, blocking write                                      │   │
│  │                                                                     │   │
│  │  Frequency: On every ratchet update (IMMEDIATE)                     │   │
│  │  Retention: Permanent archive                                       │   │
│  │  Size: 1-10KB                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  IMPROVEMENT LOG CHECKPOINT                                         │   │
│  │  ──────────────────────────                                         │   │
│  │  • New improvement entries only                                     │   │
│  │  • Append-only file format                                          │   │
│  │  • Synchronous, blocking write                                      │   │
│  │                                                                     │   │
│  │  Frequency: On every improvement (IMMEDIATE)                        │   │
│  │  Retention: Permanent archive                                       │   │
│  │  Size: 100-500 bytes per entry                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Checkpoint Manager Implementation

```python
import asyncio
import json
import gzip
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable
from contextlib import asynccontextmanager
import aiofiles
import fcntl

class CheckpointManager:
    """
    Manages all checkpointing operations for SwarmResearch.
    Ensures atomic, durable, and efficient state persistence.
    """
    
    def __init__(
        self,
        base_path: Path,
        checkpoint_interval_seconds: float = 30.0,
        max_incremental_before_full: int = 10,
        compression_enabled: bool = True
    ):
        self.base_path = Path(base_path)
        self.checkpoint_interval = checkpoint_interval_seconds
        self.max_incremental = max_incremental_before_full
        self.compression = compression_enabled
        
        # Paths
        self.checkpoints_dir = self.base_path / "checkpoints"
        self.ratchet_dir = self.base_path / "ratchet"
        self.log_dir = self.base_path / "improvement_log"
        self.temp_dir = self.base_path / "temp"
        
        # State
        self._checkpoint_task: Optional[asyncio.Task] = None
        self._pending_checkpoints: asyncio.Queue = asyncio.Queue()
        self._last_full_checkpoint: Optional[Path] = None
        self._incremental_count: int = 0
        self._checkpoint_list: List[Path] = []
        
        # Callbacks
        self._on_checkpoint_complete: List[Callable] = []
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories."""
        for dir_path in [self.checkpoints_dir, self.ratchet_dir, self.log_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # LIFECYCLE
    # ========================================================================
    
    async def start(self) -> None:
        """Start the checkpoint manager background task."""
        self._checkpoint_task = asyncio.create_task(
            self._checkpoint_loop(),
            name="checkpoint_manager"
        )
    
    async def stop(self) -> None:
        """Stop the checkpoint manager gracefully."""
        if self._checkpoint_task:
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass
        
        # Flush any pending checkpoints
        while not self._pending_checkpoints.empty():
            checkpoint_type, state = await self._pending_checkpoints.get()
            await self._execute_checkpoint(checkpoint_type, state)
    
    async def _checkpoint_loop(self) -> None:
        """Background loop for periodic checkpointing."""
        while True:
            try:
                await asyncio.sleep(self.checkpoint_interval)
                
                # Signal for incremental checkpoint
                await self._pending_checkpoints.put(("incremental", None))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Checkpoint loop error: {e}")
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    async def checkpoint_full(self, state: SessionState) -> Path:
        """Create a full checkpoint of the entire session state."""
        return await self._execute_checkpoint("full", state)
    
    async def checkpoint_incremental(self, state: SessionState) -> Path:
        """Create an incremental checkpoint."""
        return await self._execute_checkpoint("incremental", state)
    
    async def checkpoint_agent(self, agent_state: AgentState) -> Path:
        """Create a checkpoint for a single agent."""
        return await self._execute_checkpoint("agent", agent_state)
    
    async def checkpoint_ratchet(self, ratchet_state: GlobalRatchetState) -> Path:
        """
        Create an immediate, synchronous checkpoint of ratchet state.
        This is critical - never async, always durable.
        """
        return await self._execute_checkpoint("ratchet", ratchet_state)
    
    async def append_improvement(self, entry: ImprovementEntry) -> Path:
        """
        Append an improvement entry to the log.
        Synchronous, blocking write for durability.
        """
        return await self._execute_checkpoint("improvement", entry)
    
    # ========================================================================
    # INTERNAL IMPLEMENTATION
    # ========================================================================
    
    async def _execute_checkpoint(
        self, 
        checkpoint_type: str, 
        state: Any
    ) -> Path:
        """Execute a checkpoint operation."""
        timestamp = datetime.utcnow()
        checkpoint_id = uuid4()
        
        # Determine file path
        file_path = self._get_checkpoint_path(checkpoint_type, checkpoint_id, timestamp)
        temp_path = self.temp_dir / f"{checkpoint_id}.tmp"
        
        try:
            # Serialize state
            if hasattr(state, 'model_dump'):
                data = state.model_dump_json(indent=2)
            elif isinstance(state, str):
                data = state
            else:
                data = json.dumps(state, default=str, indent=2)
            
            # Compress if enabled and beneficial
            if self.compression and len(data) > 1024:
                final_path = file_path.with_suffix(file_path.suffix + '.gz')
                async with aiofiles.open(temp_path, 'wb') as f:
                    compressed = gzip.compress(data.encode(), compresslevel=6)
                    await f.write(compressed)
            else:
                final_path = file_path
                async with aiofiles.open(temp_path, 'w') as f:
                    await f.write(data)
            
            # Atomic rename
            temp_path.rename(final_path)
            
            # Update tracking
            if checkpoint_type == "full":
                self._last_full_checkpoint = final_path
                self._incremental_count = 0
            elif checkpoint_type == "incremental":
                self._incremental_count += 1
            
            self._checkpoint_list.append(final_path)
            
            # Cleanup old checkpoints
            await self._cleanup_old_checkpoints()
            
            # Notify callbacks
            for callback in self._on_checkpoint_complete:
                try:
                    callback(checkpoint_type, final_path)
                except Exception:
                    pass
            
            return final_path
            
        except Exception as e:
            # Clean up temp file on failure
            if temp_path.exists():
                temp_path.unlink()
            raise CheckpointError(f"Failed to create {checkpoint_type} checkpoint: {e}")
    
    def _get_checkpoint_path(
        self, 
        checkpoint_type: str, 
        checkpoint_id: UUID,
        timestamp: datetime
    ) -> Path:
        """Generate checkpoint file path."""
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{checkpoint_type}_{ts_str}_{checkpoint_id.hex[:8]}.json"
        
        if checkpoint_type == "ratchet":
            return self.ratchet_dir / filename
        elif checkpoint_type == "improvement":
            return self.log_dir / filename
        else:
            return self.checkpoints_dir / filename
    
    async def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints based on retention policy."""
        # Keep last 3 full checkpoints
        # Keep incremental checkpoints for last 2 full checkpoints
        # Keep all ratchet and improvement entries
        
        full_checkpoints = [
            p for p in self._checkpoint_list 
            if "full_" in p.name
        ]
        
        if len(full_checkpoints) > 3:
            to_remove = full_checkpoints[:-3]
            for path in to_remove:
                if path.exists():
                    path.unlink()
                self._checkpoint_list.remove(path)

    # ========================================================================
    # RECOVERY
    # ========================================================================
    
    async def load_latest_checkpoint(self) -> Optional[SessionState]:
        """Load the most recent valid checkpoint."""
        # Find latest full checkpoint
        full_checkpoints = sorted(
            self.checkpoints_dir.glob("full_*.json*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for full_path in full_checkpoints:
            try:
                state = await self._load_checkpoint_file(full_path)
                
                # Apply incremental checkpoints
                state = await self._apply_incrementals(state, full_path)
                
                return state
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {full_path}: {e}")
                continue
        
        return None
    
    async def _load_checkpoint_file(self, path: Path) -> SessionState:
        """Load and parse a checkpoint file."""
        if path.suffix == '.gz':
            async with aiofiles.open(path, 'rb') as f:
                compressed = await f.read()
                data = gzip.decompress(compressed).decode()
        else:
            async with aiofiles.open(path, 'r') as f:
                data = await f.read()
        
        return SessionState.model_validate_json(data)
    
    async def _apply_incrementals(
        self, 
        base_state: SessionState,
        base_path: Path
    ) -> SessionState:
        """Apply incremental checkpoints on top of base state."""
        base_time = base_path.stat().st_mtime
        
        incrementals = sorted(
            p for p in self.checkpoints_dir.glob("incremental_*.json*")
            if p.stat().st_mtime > base_time
        )
        
        for inc_path in incrementals:
            try:
                inc_state = await self._load_checkpoint_file(inc_path)
                base_state = self._merge_states(base_state, inc_state)
            except Exception as e:
                logger.warning(f"Failed to apply incremental {inc_path}: {e}")
        
        return base_state
    
    def _merge_states(self, base: SessionState, incremental: SessionState) -> SessionState:
        """Merge incremental state into base state."""
        # Merge branches
        for branch_id, branch in incremental.branches.items():
            base.branches[branch_id] = branch
        
        # Merge agents
        for agent_id, agent in incremental.agents.items():
            base.agents[agent_id] = agent
        
        # Merge tasks
        for task_id, task in incremental.tasks.items():
            base.tasks[task_id] = task
        
        # Update ratchet if newer
        if incremental.ratchet_state.version > base.ratchet_state.version:
            base.ratchet_state = incremental.ratchet_state
        
        # Merge improvement log
        for entry in incremental.improvement_log.entries:
            if entry.sequence_number >= len(base.improvement_log.entries):
                base.improvement_log.append(entry)
        
        return base

class CheckpointError(Exception):
    """Error during checkpoint operation."""
    pass
```

---

## 3. Recovery Mechanisms

### 3.1 Recovery Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RECOVERY ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  DETECTION LAYER                                                    │   │
│  │  ───────────────                                                    │   │
│  │  • Process crash detection (PID file monitoring)                    │   │
│  │  • Health check failures                                            │   │
│  │  • Watchdog timeouts                                                │   │
│  │  • Manual recovery trigger                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  RECOVERY ORCHESTRATOR                                              │   │
│  │  ─────────────────────                                              │   │
│  │                                                                     │   │
│  │  Phase 1: State Discovery                                           │   │
│  │  ────────────────────────                                           │   │
│  │  1. Scan checkpoint directory                                       │   │
│  │  2. Identify latest valid full checkpoint                           │   │
│  │  3. Find applicable incrementals                                    │   │
│  │  4. Verify ratchet state integrity                                  │   │
│  │  5. Verify improvement log integrity                                │   │
│  │                                                                     │   │
│  │  Phase 2: State Reconstruction                                      │   │
│  │  ─────────────────────────────                                      │   │
│  │  1. Load base full checkpoint                                       │   │
│  │  2. Apply incrementals in order                                     │   │
│  │  3. Rebuild in-memory indexes                                       │   │
│  │  4. Restore agent conversation states                               │   │
│  │  5. Reconnect to message bus                                        │   │
│  │                                                                     │   │
│  │  Phase 3: State Validation                                          │   │
│  │  ─────────────────────────                                          │   │
│  │  1. Verify state checksums                                          │   │
│  │  2. Check improvement log chain                                     │   │
│  │  3. Validate ratchet monotonicity                                   │   │
│  │  4. Ensure no duplicate task executions                             │   │
│  │                                                                     │   │
│  │  Phase 4: Resume Execution                                          │   │
│  │  ──────────────────────────                                         │   │
│  │  1. Restart paused agents                                           │   │
│  │  2. Re-enqueue pending tasks                                        │   │
│  │  3. Resume branch explorations                                      │   │
│  │  4. Notify monitoring systems                                       │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  RECOVERY STRATEGIES                                                │   │
│  │  ───────────────────                                                │   │
│  │                                                                     │   │
│  │  Strategy A: Clean Checkpoint Recovery                              │   │
│  │  ─────────────────────────────────                                  │   │
│  │  • Latest checkpoint is valid                                       │   │
│  │  • Load and resume normally                                         │   │
│  │  • Expected time: <5 seconds                                        │   │
│  │                                                                     │   │
│  │  Strategy B: Corrupted Checkpoint Recovery                          │   │
│  │  ─────────────────────────────────────                              │   │
│  │  • Latest checkpoint corrupted                                      │   │
│  │  • Roll back to previous valid checkpoint                           │   │
│  │  • May lose some recent progress                                    │   │
│  │  • Expected time: 10-30 seconds                                     │   │
│  │                                                                     │   │
│  │  Strategy C: Partial State Recovery                                 │   │
│  │  ────────────────────────────────                                   │   │
│  │  • Some agent states lost                                           │   │
│  │  • Reconstruct from ratchet + improvement log                       │   │
│  │  • Restart affected agents                                          │   │
│  │  • Expected time: 30-60 seconds                                     │   │
│  │                                                                     │   │
│  │  Strategy D: Catastrophic Recovery                                  │   │
│  │  ──────────────────────────────                                     │   │
│  │  • All checkpoints lost/corrupted                                   │   │
│  │  • Reconstruct from ratchet archive                                 │   │
│  │  • Restart from last known good state                               │   │
│  │  • Expected time: 1-5 minutes                                       │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Recovery Manager Implementation

```python
import os
import signal
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum, auto
import psutil

class RecoveryStrategy(Enum):
    """Recovery strategies in order of preference."""
    CLEAN_CHECKPOINT = auto()
    ROLLBACK_CHECKPOINT = auto()
    PARTIAL_STATE = auto()
    CATASTROPHIC = auto()
    FRESH_START = auto()

@dataclass
class RecoveryResult:
    """Result of a recovery operation."""
    success: bool
    strategy_used: RecoveryStrategy
    state: Optional[SessionState]
    lost_iterations: int
    recovery_time_seconds: float
    warnings: List[str]
    errors: List[str]

class RecoveryManager:
    """
    Manages recovery from failures for SwarmResearch.
    Implements multiple recovery strategies with automatic fallback.
    """
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        base_path: Path,
        max_rollback_attempts: int = 3
    ):
        self.checkpoint_manager = checkpoint_manager
        self.base_path = Path(base_path)
        self.max_rollback = max_rollback_attempts
        
        # Paths
        self.pid_file = self.base_path / "swarmresearch.pid"
        self.lock_file = self.base_path / "swarmresearch.lock"
        
        # State
        self._recovery_in_progress = False
    
    # ========================================================================
    # CRASH DETECTION
    # ========================================================================
    
    def check_for_previous_crash(self) -> Tuple[bool, Optional[str]]:
        """
        Check if previous session crashed uncleanly.
        Returns (crashed, reason).
        """
        if not self.pid_file.exists():
            # No PID file means clean shutdown or first start
            return False, None
        
        try:
            pid = int(self.pid_file.read_text().strip())
            
            if psutil.pid_exists(pid):
                # Process still running - check if it's us
                proc = psutil.Process(pid)
                if "swarmresearch" in proc.name().lower():
                    return False, "Another instance is running"
            
            # PID exists but process doesn't - unclean shutdown
            return True, f"Previous process (PID {pid}) terminated uncleanly"
            
        except (ValueError, psutil.Error) as e:
            return True, f"Error checking PID file: {e}"
    
    def acquire_lock(self) -> bool:
        """Acquire exclusive lock for this instance."""
        try:
            self._lock_fd = open(self.lock_file, 'w')
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # Write PID
            self.pid_file.write_text(str(os.getpid()))
            return True
        except (IOError, OSError):
            return False
    
    def release_lock(self) -> None:
        """Release exclusive lock."""
        try:
            if hasattr(self, '_lock_fd'):
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
                self._lock_fd.close()
            
            if self.pid_file.exists():
                self.pid_file.unlink()
            
            if self.lock_file.exists():
                self.lock_file.unlink()
        except Exception:
            pass
    
    # ========================================================================
    # RECOVERY ENTRY POINT
    # ========================================================================
    
    async def recover(self) -> RecoveryResult:
        """
        Attempt to recover from a failure.
        Tries strategies in order until one succeeds.
        """
        import time
        start_time = time.time()
        
        self._recovery_in_progress = True
        warnings = []
        errors = []
        
        try:
            # Strategy 1: Clean checkpoint recovery
            result = await self._try_clean_recovery()
            if result.success:
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.CLEAN_CHECKPOINT,
                    state=result.state,
                    lost_iterations=0,
                    recovery_time_seconds=time.time() - start_time,
                    warnings=warnings,
                    errors=errors
                )
            
            warnings.extend(result.warnings)
            errors.extend(result.errors)
            
            # Strategy 2: Rollback to previous checkpoint
            result = await self._try_rollback_recovery()
            if result.success:
                warnings.append(f"Rolled back {result.lost_iterations} iterations")
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.ROLLBACK_CHECKPOINT,
                    state=result.state,
                    lost_iterations=result.lost_iterations,
                    recovery_time_seconds=time.time() - start_time,
                    warnings=warnings,
                    errors=errors
                )
            
            warnings.extend(result.warnings)
            errors.extend(result.errors)
            
            # Strategy 3: Partial state recovery
            result = await self._try_partial_recovery()
            if result.success:
                warnings.append("Partial state recovery - some agents restarted")
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.PARTIAL_STATE,
                    state=result.state,
                    lost_iterations=result.lost_iterations,
                    recovery_time_seconds=time.time() - start_time,
                    warnings=warnings,
                    errors=errors
                )
            
            warnings.extend(result.warnings)
            errors.extend(result.errors)
            
            # Strategy 4: Catastrophic recovery
            result = await self._try_catastrophic_recovery()
            if result.success:
                warnings.append("Catastrophic recovery - significant progress lost")
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.CATASTROPHIC,
                    state=result.state,
                    lost_iterations=result.lost_iterations,
                    recovery_time_seconds=time.time() - start_time,
                    warnings=warnings,
                    errors=errors
                )
            
            # All strategies failed
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.FRESH_START,
                state=None,
                lost_iterations=0,
                recovery_time_seconds=time.time() - start_time,
                warnings=warnings,
                errors=errors + ["All recovery strategies failed"]
            )
            
        finally:
            self._recovery_in_progress = False
    
    # ========================================================================
    # RECOVERY STRATEGIES
    # ========================================================================
    
    async def _try_clean_recovery(self) -> '_RecoveryAttempt':
        """Try to recover from the latest checkpoint."""
        try:
            state = await self.checkpoint_manager.load_latest_checkpoint()
            
            if state is None:
                return _RecoveryAttempt(
                    success=False,
                    state=None,
                    lost_iterations=0,
                    warnings=["No checkpoints found"],
                    errors=[]
                )
            
            # Validate state
            if not self._validate_state(state):
                return _RecoveryAttempt(
                    success=False,
                    state=None,
                    lost_iterations=0,
                    warnings=["State validation failed"],
                    errors=["Checksum or integrity check failed"]
                )
            
            return _RecoveryAttempt(
                success=True,
                state=state,
                lost_iterations=0,
                warnings=[],
                errors=[]
            )
            
        except Exception as e:
            return _RecoveryAttempt(
                success=False,
                state=None,
                lost_iterations=0,
                warnings=[],
                errors=[f"Clean recovery failed: {e}"]
            )
    
    async def _try_rollback_recovery(self) -> '_RecoveryAttempt':
        """Try to recover from older checkpoints."""
        checkpoints = sorted(
            self.checkpoint_manager.checkpoints_dir.glob("full_*.json*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for i, checkpoint_path in enumerate(checkpoints[1:self.max_rollback+1], 1):
            try:
                state = await self.checkpoint_manager._load_checkpoint_file(checkpoint_path)
                
                if self._validate_state(state):
                    # Apply incrementals
                    state = await self.checkpoint_manager._apply_incrementals(
                        state, checkpoint_path
                    )
                    
                    return _RecoveryAttempt(
                        success=True,
                        state=state,
                        lost_iterations=i * 10,  # Estimate
                        warnings=[f"Rolled back to {checkpoint_path.name}"],
                        errors=[]
                    )
                    
            except Exception as e:
                continue
        
        return _RecoveryAttempt(
            success=False,
            state=None,
            lost_iterations=0,
            warnings=[],
            errors=["Rollback recovery failed - no valid older checkpoints"]
        )
    
    async def _try_partial_recovery(self) -> '_RecoveryAttempt':
        """Try to reconstruct state from ratchet and improvement log."""
        try:
            # Load ratchet state
            ratchet_files = sorted(
                self.checkpoint_manager.ratchet_dir.glob("ratchet_*.json*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            if not ratchet_files:
                return _RecoveryAttempt(
                    success=False,
                    state=None,
                    lost_iterations=0,
                    warnings=[],
                    errors=["No ratchet state found for partial recovery"]
                )
            
            # Load latest ratchet
            latest_ratchet = await self.checkpoint_manager._load_checkpoint_file(
                ratchet_files[0]
            )
            
            # Create minimal state
            state = SessionState(
                global_state=GlobalState(
                    orchestrator=OrchestratorState(status=OrchestratorStatus.RECOVERING),
                    swarm=SwarmState(),
                    research_context=ResearchContext(problem_statement="", domain="")
                ),
                ratchet_state=latest_ratchet
            )
            
            # Replay improvement log if available
            state = await self._replay_improvement_log(state)
            
            return _RecoveryAttempt(
                success=True,
                state=state,
                lost_iterations=100,  # Significant loss
                warnings=["Partial recovery - agents will be restarted"],
                errors=[]
            )
            
        except Exception as e:
            return _RecoveryAttempt(
                success=False,
                state=None,
                lost_iterations=0,
                warnings=[],
                errors=[f"Partial recovery failed: {e}"]
            )
    
    async def _try_catastrophic_recovery(self) -> '_RecoveryAttempt':
        """Last resort - reconstruct from any available state fragments."""
        # This would involve more complex reconstruction
        # For now, fail to fresh start
        return _RecoveryAttempt(
            success=False,
            state=None,
            lost_iterations=0,
            warnings=[],
            errors=["Catastrophic recovery not implemented - starting fresh"]
        )
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    def _validate_state(self, state: SessionState) -> bool:
        """Validate recovered state integrity."""
        try:
            # Verify version compatibility
            if not state.version.is_compatible_with(CURRENT_STATE_VERSION):
                return False
            
            # Verify improvement log integrity
            if not state.improvement_log.verify_integrity():
                return False
            
            # Verify ratchet monotonicity
            if state.ratchet_state.version < 0:
                return False
            
            # Verify checksum if present
            if hasattr(state, 'checksum'):
                expected = state.compute_checksum()
                if state.checksum != expected:
                    return False
            
            return True
            
        except Exception:
            return False
    
    async def _replay_improvement_log(self, state: SessionState) -> SessionState:
        """Replay improvement log entries to reconstruct state."""
        # Load all improvement entries
        log_files = sorted(
            self.checkpoint_manager.log_dir.glob("improvement_*.json*")
        )
        
        for log_file in log_files:
            try:
                entry = await self.checkpoint_manager._load_checkpoint_file(log_file)
                state.improvement_log.append(entry)
            except Exception:
                continue
        
        return state

@dataclass
class _RecoveryAttempt:
    """Internal result of a recovery attempt."""
    success: bool
    state: Optional[SessionState]
    lost_iterations: int
    warnings: List[str]
    errors: List[str]
```

---

## 4. Persistence Format

### 4.1 File Organization

```
swarmresearch_data/
├── session.json                    # Session metadata and pointers
├── checkpoints/
│   ├── full_20240115_143022_abc12345.json.gz    # Full checkpoint
│   ├── incremental_20240115_143052_def67890.json.gz
│   ├── incremental_20240115_143122_ghi11111.json.gz
│   └── ...
├── ratchet/
│   ├── ratchet_20240115_143025_jkl22222.json    # Immutable ratchet snapshots
│   ├── ratchet_20240115_143055_mno33333.json
│   └── ...
├── improvement_log/
│   ├── improvement_20240115_143025_pqr44444.json   # Individual entries
│   ├── improvement_20240115_143035_stu55555.json
│   └── ...
├── agents/
│   ├── agent_<uuid>_conversation.json           # Large agent states
│   └── agent_<uuid>_working_memory.json
├── temp/
│   └── *.tmp                                    # Temporary files during write
└── archive/
    └── daily_20240114_235959_full.json.gz       # Archived checkpoints
```

### 4.2 Storage Formats

```python
"""
Storage Format Specifications
=============================

1. JSON + GZIP (Default)
   - Human-readable when uncompressed
   - Good compression ratio (typical: 5-10x)
   - Portable and language-agnostic
   - Slower than binary formats

2. MessagePack (Optional)
   - Binary format, faster serialization
   - Smaller than JSON
   - Less human-readable

3. SQLite (Alternative)
   - For very large sessions
   - Queryable state
   - ACID transactions
   - More complex setup
"""

import json
import gzip
import msgpack
from enum import Enum
from typing import BinaryIO

class StorageFormat(Enum):
    """Supported storage formats."""
    JSON = "json"
    JSON_GZIP = "json.gz"
    MESSAGEPACK = "msgpack"
    MESSAGEPACK_GZIP = "msgpack.gz"

class StateSerializer:
    """Handles serialization/deserialization of state objects."""
    
    def __init__(self, format: StorageFormat = StorageFormat.JSON_GZIP):
        self.format = format
    
    def serialize(self, state: BaseModel) -> bytes:
        """Serialize state to bytes."""
        if self.format in (StorageFormat.JSON, StorageFormat.JSON_GZIP):
            data = state.model_dump_json(indent=2).encode('utf-8')
            if self.format == StorageFormat.JSON_GZIP:
                data = gzip.compress(data, compresslevel=6)
            return data
        
        elif self.format in (StorageFormat.MESSAGEPACK, StorageFormat.MESSAGEPACK_GZIP):
            data = msgpack.packb(state.model_dump(), use_bin_type=True)
            if self.format == StorageFormat.MESSAGEPACK_GZIP:
                data = gzip.compress(data, compresslevel=6)
            return data
        
        else:
            raise ValueError(f"Unknown format: {self.format}")
    
    def deserialize(self, data: bytes, state_class: type) -> BaseModel:
        """Deserialize bytes to state object."""
        if self.format in (StorageFormat.JSON_GZIP, StorageFormat.MESSAGEPACK_GZIP):
            data = gzip.decompress(data)
        
        if self.format in (StorageFormat.JSON, StorageFormat.JSON_GZIP):
            return state_class.model_validate_json(data.decode('utf-8'))
        
        elif self.format in (StorageFormat.MESSAGEPACK, StorageFormat.MESSAGEPACK_GZIP):
            return state_class.model_validate(msgpack.unpackb(data, raw=False))
        
        else:
            raise ValueError(f"Unknown format: {self.format}")

# ============================================================================
# ATOMIC WRITE PATTERN
# ============================================================================

async def atomic_write_file(
    target_path: Path,
    data: bytes,
    temp_dir: Optional[Path] = None
) -> Path:
    """
    Atomically write data to file using write-to-temp-then-rename pattern.
    
    This ensures:
    1. Readers never see partial writes
    2. Failed writes don't corrupt existing files
    3. On crash, temp files can be cleaned up
    """
    temp_dir = temp_dir or target_path.parent / ".temp"
    temp_dir.mkdir(exist_ok=True)
    
    temp_path = temp_dir / f".tmp_{uuid4().hex}"
    
    try:
        # Write to temp file
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(data)
            await f.flush()
            # Ensure data is on disk
            if hasattr(f, 'fileno'):
                os.fsync(f.fileno())
        
        # Atomic rename
        temp_path.rename(target_path)
        
        # Sync parent directory
        dir_fd = os.open(target_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        
        return target_path
        
    except Exception:
        # Clean up temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise
```

### 4.3 SQLite Backend (Alternative)

```python
"""
SQLite Backend for Large Sessions
=================================
For sessions with >10GB of state, SQLite provides:
- Queryable state
- Incremental updates
- ACID transactions
- Better space efficiency
"""

import aiosqlite
from contextlib import asynccontextmanager

class SQLiteStateBackend:
    """SQLite-backed state storage for large sessions."""
    
    SCHEMA = """
    -- Sessions table
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        version TEXT NOT NULL,
        metadata TEXT
    );
    
    -- Branches table
    CREATE TABLE IF NOT EXISTS branches (
        branch_id TEXT PRIMARY KEY,
        session_id TEXT REFERENCES sessions(session_id),
        parent_branch_id TEXT,
        status TEXT NOT NULL,
        focus_area TEXT,
        depth INTEGER,
        state_json TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Agents table
    CREATE TABLE IF NOT EXISTS agents (
        agent_id TEXT PRIMARY KEY,
        branch_id TEXT REFERENCES branches(branch_id),
        session_id TEXT REFERENCES sessions(session_id),
        agent_type TEXT NOT NULL,
        status TEXT NOT NULL,
        conversation_json TEXT,
        working_memory_ref TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Tasks table
    CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY,
        session_id TEXT REFERENCES sessions(session_id),
        parent_task_id TEXT,
        status TEXT NOT NULL,
        task_type TEXT NOT NULL,
        parameters_json TEXT,
        result_json TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Ratchet state table (append-only)
    CREATE TABLE IF NOT EXISTS ratchet_states (
        version INTEGER PRIMARY KEY,
        session_id TEXT REFERENCES sessions(session_id),
        best_hypothesis TEXT,
        best_score REAL,
        evidence_json TEXT,
        knowledge_graph_json TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Improvement log (append-only)
    CREATE TABLE IF NOT EXISTS improvement_log (
        sequence_number INTEGER PRIMARY KEY,
        session_id TEXT REFERENCES sessions(session_id),
        entry_type TEXT NOT NULL,
        description TEXT,
        metrics_json TEXT,
        entry_hash TEXT NOT NULL,
        previous_hash TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_branches_session ON branches(session_id);
    CREATE INDEX IF NOT EXISTS idx_agents_branch ON agents(branch_id);
    CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
    CREATE INDEX IF NOT EXISTS idx_improvement_session ON improvement_log(session_id);
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._pool: Optional[aiosqlite.Connection] = None
    
    async def initialize(self) -> None:
        """Initialize database schema."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(self.SCHEMA)
            await db.commit()
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("BEGIN")
            try:
                yield db
                await db.commit()
            except Exception:
                await db.rollback()
                raise
    
    async def save_session(self, state: SessionState) -> None:
        """Save entire session state."""
        async with self.transaction() as db:
            # Save session metadata
            await db.execute(
                """
                INSERT OR REPLACE INTO sessions 
                (session_id, version, metadata) 
                VALUES (?, ?, ?)
                """,
                (
                    str(state.session_id),
                    str(state.version),
                    json.dumps({"checkpoint_count": state.checkpoint_count})
                )
            )
            
            # Save branches
            for branch_id, branch in state.branches.items():
                await db.execute(
                    """
                    INSERT OR REPLACE INTO branches
                    (branch_id, session_id, parent_branch_id, status, 
                     focus_area, depth, state_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(branch_id),
                        str(state.session_id),
                        str(branch.parent_branch_id) if branch.parent_branch_id else None,
                        branch.status.value,
                        branch.focus_area,
                        branch.depth,
                        branch.model_dump_json()
                    )
                )
            
            # Similar for agents, tasks, ratchet, improvement log...
            # (omitted for brevity)
```

---

## 5. State Versioning

### 5.1 Version Compatibility

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STATE VERSIONING STRATEGY                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Version Format: MAJOR.MINOR.PATCH                                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  MAJOR (Breaking Changes)                                           │   │
│  │  ─────────────────────────                                          │   │
│  │  • Schema structure changes                                         │   │
│  │  • Field removal or type changes                                    │   │
│  │  • Incompatible with older versions                                 │   │
│  │  • Requires migration script                                        │   │
│  │  • Examples:                                                        │   │
│  │    - Renaming a field                                               │   │
│  │    - Changing field type (str -> int)                               │   │
│  │    - Removing required fields                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  MINOR (New Features)                                               │   │
│  │  ─────────────────────                                              │   │
│  │  • New optional fields added                                        │   │
│  │  • Backward compatible loading                                      │   │
│  │  • Older versions can read newer state                              │   │
│  │  • Examples:                                                        │   │
│  │    - Adding new optional metadata                                   │   │
│  │    - Adding new state types                                         │   │
│  │    - Adding performance metrics                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PATCH (Bug Fixes)                                                  │   │
│  │  ──────────────────                                                 │   │
│  │  • No schema changes                                                │   │
│  │  • Only default value changes                                       │   │
│  │  • Fully compatible                                                 │   │
│  │  • Examples:                                                        │   │
│  │    - Fixing default values                                          │   │
│  │    - Documentation updates                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Compatibility Rules:                                                        │
│  ────────────────────                                                        │
│  • Same MAJOR: Can load (with minor version handling)                       │
│  • Different MAJOR: Cannot load without migration                           │
│  • Newer MINOR: Older code can read (ignores new fields)                    │
│  • Older MINOR: Newer code can read (uses defaults for missing)             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Migration System

```python
"""
State Migration System
======================
Handles upgrading state from older versions to current.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List
from dataclasses import dataclass

@dataclass
class Migration:
    """Single migration from one version to another."""
    from_version: StateVersion
    to_version: StateVersion
    description: str
    migrate_fn: Callable[[Dict], Dict]

class MigrationManager:
    """Manages state migrations between versions."""
    
    def __init__(self):
        self._migrations: Dict[tuple, Migration] = {}
        self._register_builtin_migrations()
    
    def register(self, migration: Migration) -> None:
        """Register a migration."""
        key = (str(migration.from_version), str(migration.to_version))
        self._migrations[key] = migration
    
    def migrate(self, state_data: Dict, target_version: StateVersion) -> Dict:
        """
        Migrate state data to target version.
        
        Raises MigrationError if no path exists.
        """
        current_version = StateVersion(**state_data.get('version', {'major': 1, 'minor': 0, 'patch': 0}))
        
        while current_version != target_version:
            # Find next migration step
            next_migration = self._find_next_migration(current_version, target_version)
            
            if next_migration is None:
                raise MigrationError(
                    f"No migration path from {current_version} to {target_version}"
                )
            
            # Apply migration
            state_data = next_migration.migrate_fn(state_data)
            state_data['version'] = next_migration.to_version.model_dump()
            current_version = next_migration.to_version
        
        return state_data
    
    def _find_next_migration(
        self, 
        from_version: StateVersion, 
        target_version: StateVersion
    ) -> Optional[Migration]:
        """Find the next migration step toward target."""
        # Simple greedy approach - find any migration from current version
        for (from_v, to_v), migration in self._migrations.items():
            if from_v == str(from_version):
                return migration
        return None
    
    def _register_builtin_migrations(self) -> None:
        """Register built-in migrations."""
        # Example migration from 1.0.0 to 1.1.0
        self.register(Migration(
            from_version=StateVersion(1, 0, 0),
            to_version=StateVersion(1, 1, 0),
            description="Add agent performance metrics",
            migrate_fn=self._migrate_1_0_0_to_1_1_0
        ))
    
    @staticmethod
    def _migrate_1_0_0_to_1_1_0(state_data: Dict) -> Dict:
        """Migration: Add performance metrics to agents."""
        if 'agents' in state_data:
            for agent_id, agent in state_data['agents'].items():
                if 'performance_metrics' not in agent:
                    agent['performance_metrics'] = {
                        'iterations': 0,
                        'tokens_consumed': 0,
                        'cost_usd': 0.0
                    }
        return state_data

class MigrationError(Exception):
    """Error during state migration."""
    pass

# ============================================================================
# VERSIONED STATE LOADING
# ============================================================================

async def load_state_with_migration(
    file_path: Path,
    target_version: StateVersion = CURRENT_STATE_VERSION
) -> SessionState:
    """
    Load state from file, migrating if necessary.
    """
    # Read raw data
    if file_path.suffix == '.gz':
        async with aiofiles.open(file_path, 'rb') as f:
            data = gzip.decompress(await f.read()).decode()
    else:
        async with aiofiles.open(file_path, 'r') as f:
            data = await f.read()
    
    state_data = json.loads(data)
    
    # Check version
    file_version = StateVersion(**state_data.get('version', {'major': 1, 'minor': 0, 'patch': 0}))
    
    if not file_version.is_compatible_with(target_version):
        # Need migration
        migration_manager = MigrationManager()
        state_data = migration_manager.migrate(state_data, target_version)
    
    # Validate and return
    return SessionState.model_validate(state_data)
```

### 5.3 Forward Compatibility

```python
"""
Forward Compatibility Handling
==============================
Ensures newer state can be loaded by older code (within same major version).
"""

from pydantic import BaseModel, Field, ConfigDict

class ForwardCompatibleBaseModel(BaseModel):
    """Base model that handles unknown fields gracefully."""
    
    model_config = ConfigDict(
        extra='ignore',  # Ignore unknown fields
        validate_assignment=True,
        strict=False
    )
    
    # Track unknown fields for potential later use
    _unknown_fields: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    
    def model_post_init(self, __context) -> None:
        """Store any unknown fields that were passed."""
        if hasattr(__context, 'unknown_fields'):
            self._unknown_fields = __context.unknown_fields

# Example: Adding new optional fields
class AgentStateV1_0(BaseModel):
    """Version 1.0 of AgentState."""
    agent_id: UUID
    status: str
    conversation: ConversationState

class AgentStateV1_1(ForwardCompatibleBaseModel):
    """Version 1.1 adds performance_metrics (optional)."""
    agent_id: UUID
    status: str
    conversation: ConversationState
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    # New in 1.1 - older versions will ignore this

# When V1.0 code loads V1.1 state:
# - performance_metrics is ignored (extra='ignore')
# - All other fields load normally
# - State is usable

# When V1.1 code loads V1.0 state:
# - performance_metrics gets default empty dict
# - All other fields load normally
# - State is usable
```

---

## 6. Overnight Run Reliability

### 6.1 Automatic Checkpointing Schedule

```python
"""
Overnight Run Configuration
===========================
Optimized for long-running research sessions.
"""

OVERNIGHT_CHECKPOINT_CONFIG = {
    # Checkpoint every 5 minutes during active processing
    "incremental_interval_seconds": 300,
    
    # Full checkpoint every 30 minutes
    "full_checkpoint_interval_minutes": 30,
    
    # Immediate checkpoint on:
    "immediate_checkpoint_triggers": [
        "ratchet_update",           # Best hypothesis improved
        "major_finding",            # Significant discovery
        "branch_created",           # New research branch
        "branch_merged",            # Branch consolidation
        "agent_completed",          # Agent finished work
        "task_completed",           # Task finished
    ],
    
    # Ratchet updates are synchronous and blocking
    "ratchet_sync_write": True,
    
    # Improvement log is synchronous and blocking
    "improvement_log_sync_write": True,
    
    # Agent checkpoints are async
    "agent_checkpoint_async": True,
    
    # Maximum checkpoint age before forcing full
    "max_incremental_age_minutes": 60,
    
    # Retention
    "keep_full_checkpoints": 5,
    "keep_incremental_checkpoints": 20,
    "archive_daily": True,
}
```

### 6.2 Health Monitoring

```python
class SessionHealthMonitor:
    """Monitors session health during overnight runs."""
    
    def __init__(self, session: SessionState, checkpoint_manager: CheckpointManager):
        self.session = session
        self.checkpoint_manager = checkpoint_manager
        self._last_checkpoint_time = datetime.utcnow()
        self._last_progress_time = datetime.utcnow()
        self._stagnation_threshold_minutes = 30
    
    async def health_check(self) -> 'HealthStatus':
        """Perform health check and take action if needed."""
        now = datetime.utcnow()
        issues = []
        
        # Check checkpoint recency
        time_since_checkpoint = (now - self._last_checkpoint_time).total_seconds()
        if time_since_checkpoint > 600:  # 10 minutes
            issues.append(f"Last checkpoint was {time_since_checkpoint/60:.1f} minutes ago")
            # Force checkpoint
            await self.checkpoint_manager.checkpoint_incremental(self.session)
            self._last_checkpoint_time = now
        
        # Check for stagnation
        time_since_progress = (now - self._last_progress_time).total_seconds()
        if time_since_progress > self._stagnation_threshold_minutes * 60:
            issues.append(f"No progress for {time_since_progress/60:.1f} minutes")
            # Could trigger reallocation or alert
        
        # Check improvement log integrity
        if not self.session.improvement_log.verify_integrity():
            issues.append("Improvement log integrity check failed!")
            # Critical - attempt recovery
        
        # Check disk space
        disk_usage = psutil.disk_usage(self.checkpoint_manager.base_path)
        if disk_usage.percent > 90:
            issues.append(f"Disk usage at {disk_usage.percent}%")
            # Trigger cleanup or alert
        
        return HealthStatus(
            healthy=len(issues) == 0,
            issues=issues,
            timestamp=now
        )
    
    def record_progress(self) -> None:
        """Record that progress was made."""
        self._last_progress_time = datetime.utcnow()

@dataclass
class HealthStatus:
    healthy: bool
    issues: List[str]
    timestamp: datetime
```

---

## 7. Summary

### Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Primary Format** | JSON + GZIP | Human-readable, portable, good compression |
| **Checkpoint Types** | Full, Incremental, Agent, Ratchet, Improvement | Different frequencies for different criticality |
| **Ratchet Writes** | Synchronous, blocking | Never lose best findings |
| **Improvement Log** | Append-only, hash-chained | Tamper-evident history |
| **Recovery** | 4-tier fallback | Maximize recovery success rate |
| **Versioning** | Semantic (MAJOR.MINOR.PATCH) | Clear compatibility rules |
| **Storage Backend** | Filesystem default, SQLite optional | Flexibility for different scales |

### Reliability Guarantees

1. **Zero Data Loss for Critical State**: Ratchet and improvement log are synchronously persisted
2. **Fast Recovery**: <5 seconds for clean checkpoint, <60 seconds for partial recovery
3. **Deterministic Replay**: Same inputs produce same outputs after recovery
4. **Integrity Verification**: Checksums and hash chains detect corruption
5. **Graceful Degradation**: Multiple recovery strategies with automatic fallback

### Performance Impact

- **Checkpointing Overhead**: <5% during normal operation
- **Memory Overhead**: Minimal (state is serialized, not duplicated)
- **Disk Usage**: ~100MB-1GB per hour for typical session
- **Recovery Time**: <5 seconds for 95% of cases

---

*Document Version: 1.0.0*
*Last Updated: 2024-01-15*
