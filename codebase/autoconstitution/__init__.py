"""
autoconstitution — multi-agent autoresearch under an editable constitution.

Extends Karpathy's autoresearch paradigm from a single improving loop into a
multi-agent system where models can critique, revise, judge, and preserve
improvements across student / critic / judge / meta-judge style roles.

Public API surface:
    - CLI app (typer) for running experiments
    - Orchestrator primitives (SwarmOrchestrator, TaskDAG, PerformanceMonitor)
    - Ratchet primitives (Ratchet, MetricConfig, ComparisonMode)
    - Config models (SwarmConfig, ExperimentConfig)
    - Decorators (@task, @retryable)

Example:
    >>> from autoconstitution import SwarmOrchestrator, Ratchet
    >>> orch = SwarmOrchestrator(...)
"""

__version__ = "0.2.0"
__author__ = "autoconstitution contributors"

# --- CLI entry points ----------------------------------------------------
from autoconstitution.cli import (
    ExperimentConfig as CLIExperimentConfig,
    ExperimentState,
    ExperimentType,
    LogLevel as CLILogLevel,
    SwarmConfig as CLISwarmConfig,
    app,
    load_config,
    save_config,
)

# --- Orchestrator primitives --------------------------------------------
from autoconstitution.orchestrator import (
    AgentError,
    AgentMetrics,
    AgentPoolManager,
    AgentStatus,
    BranchError,
    BranchMetrics,
    BranchPriority,
    CircularDependencyError,
    OrchestratorError,
    PerformanceMonitor,
    ReallocationError,
    ResearchBranch,
    SubAgent,
    SwarmOrchestrator,
    TaskDAG,
    TaskDAGError,
    TaskDependency,
    TaskMetrics,
    TaskNode,
    TaskStatus,
    retryable,
    task,
)

# --- Ratchet primitives --------------------------------------------------
from autoconstitution.ratchet import (
    AbstractMetricCalculator,
    ComparisonMode,
    CompositeMetricCalculator,
    ExperimentResult,
    FileSystemPersister,
    InMemoryPersister,
    MetricConfig,
    MultiMetricRatchet,
    Ratchet,
    RatchetError,
    RatchetState,
    RatchetStats,
    SimpleMetricCalculator,
    ValidationDecision,
    ValidationResult,
    create_accuracy_ratchet,
    create_loss_ratchet,
    create_target_ratchet,
)

# --- CAI loop primitives -------------------------------------------------
from autoconstitution.cai import (
    CAIRole,
    CritiqueRevisionLoop,
    CritiqueResult,
    JudgeAgent,
    MetaJudgeAgent,
    PreferencePair,
    PreferencePairBuilder,
    RevisionResult,
    StudentAgent,
)

# --- Config (Pydantic BaseSettings) -------------------------------------
from autoconstitution.config import (
    AgentConfig,
    ExperimentConfig,
    HardwareTarget,
    LogLevel,
    PollinationFrequency,
    Provider,
    RatchetMetric,
    RetryConfig,
    SwarmConfig,
    get_config,
)

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    # CLI
    "app",
    "CLIExperimentConfig",
    "CLISwarmConfig",
    "CLILogLevel",
    "ExperimentState",
    "ExperimentType",
    "load_config",
    "save_config",
    # Orchestrator
    "SwarmOrchestrator",
    "TaskDAG",
    "TaskNode",
    "TaskDependency",
    "TaskStatus",
    "TaskMetrics",
    "ResearchBranch",
    "BranchPriority",
    "BranchMetrics",
    "SubAgent",
    "AgentStatus",
    "AgentMetrics",
    "AgentPoolManager",
    "PerformanceMonitor",
    "OrchestratorError",
    "TaskDAGError",
    "CircularDependencyError",
    "AgentError",
    "BranchError",
    "ReallocationError",
    "task",
    "retryable",
    # Ratchet
    "Ratchet",
    "MultiMetricRatchet",
    "RatchetState",
    "RatchetStats",
    "MetricConfig",
    "ComparisonMode",
    "ValidationDecision",
    "ValidationResult",
    "ExperimentResult",
    "RatchetError",
    "AbstractMetricCalculator",
    "SimpleMetricCalculator",
    "CompositeMetricCalculator",
    "FileSystemPersister",
    "InMemoryPersister",
    "create_accuracy_ratchet",
    "create_loss_ratchet",
    "create_target_ratchet",
    # CAI loop
    "CAIRole",
    "StudentAgent",
    "JudgeAgent",
    "MetaJudgeAgent",
    "CritiqueRevisionLoop",
    "CritiqueResult",
    "RevisionResult",
    "PreferencePair",
    "PreferencePairBuilder",
    # Config
    "SwarmConfig",
    "ExperimentConfig",
    "AgentConfig",
    "RetryConfig",
    "Provider",
    "HardwareTarget",
    "RatchetMetric",
    "PollinationFrequency",
    "LogLevel",
    "get_config",
]
