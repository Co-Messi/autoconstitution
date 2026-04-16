"""
autoconstitution Configuration System.

This module provides Pydantic-based configuration management for the autoconstitution
framework, supporting environment variables, validation, and type-safe access.
"""

from __future__ import annotations

import os
from enum import Enum, auto
from pathlib import Path
from typing import Any, Literal

from typing_extensions import Self  # 3.11+ stdlib, pre-3.11 via typing_extensions

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    """Supported LLM providers for autoconstitution."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    LOCAL = "local"
    CUSTOM = "custom"


class HardwareTarget(str, Enum):
    """Hardware deployment targets for swarm agents."""
    
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    MULTI_GPU = "multi_gpu"
    DISTRIBUTED = "distributed"
    EDGE = "edge"
    AUTO = "auto"


class RatchetMetric(str, Enum):
    """Metrics used for ratchet-based performance tracking."""
    
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    LOSS = "loss"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    CUSTOM = "custom"


class PollinationFrequency(str, Enum):
    """Frequency settings for knowledge pollination between agents."""
    
    REALTIME = "realtime"
    HIGH = "high"  # Every iteration
    MEDIUM = "medium"  # Every 10 iterations
    LOW = "low"  # Every 100 iterations
    ADAPTIVE = "adaptive"
    MANUAL = "manual"


class LogLevel(str, Enum):
    """Logging verbosity levels."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SwarmConfig(BaseSettings):
    """
    Main configuration class for autoconstitution.
    
    This class manages all tunable parameters for the swarm research framework,
    including agent limits, budget constraints, provider settings, and
    performance metrics.
    
    Environment variables are automatically loaded with the SWARM_ prefix.
    For example, SWARM_MAX_AGENTS will set the max_agents field.
    
    Examples:
        >>> config = SwarmConfig()
        >>> config = SwarmConfig(max_agents=10, provider=Provider.OPENAI)
        >>> config = SwarmConfig.from_env_file(".env")
    """
    
    model_config = SettingsConfigDict(
        env_prefix="SWARM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
        frozen=False,
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Core Swarm Parameters
    # ═══════════════════════════════════════════════════════════════════════
    
    max_agents: int = Field(
        default=5,
        ge=1,
        le=1000,
        description="Maximum number of concurrent agents in the swarm",
        examples=[5, 10, 50],
    )
    
    experiment_budget_minutes: float = Field(
        default=60.0,
        gt=0.0,
        description="Total experiment time budget in minutes",
        examples=[30.0, 60.0, 120.0],
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Provider & Hardware Configuration
    # ═══════════════════════════════════════════════════════════════════════
    
    provider: Provider = Field(
        default=Provider.OPENAI,
        description="LLM provider for agent inference",
    )
    
    hardware_target: HardwareTarget = Field(
        default=HardwareTarget.AUTO,
        description="Target hardware for agent deployment",
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Performance & Quality Metrics
    # ═══════════════════════════════════════════════════════════════════════
    
    ratchet_metric: RatchetMetric = Field(
        default=RatchetMetric.ACCURACY,
        description="Primary metric for ratchet-based performance tracking",
    )
    
    ratchet_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum threshold for ratchet metric improvement",
    )
    
    critic_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold for critic agent approval",
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Communication & Synchronization
    # ═══════════════════════════════════════════════════════════════════════
    
    pollination_frequency: PollinationFrequency = Field(
        default=PollinationFrequency.MEDIUM,
        description="Frequency of knowledge sharing between agents",
    )
    
    pollination_batch_size: int = Field(
        default=32,
        ge=1,
        le=10000,
        description="Number of knowledge items to share per pollination event",
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Agent Behavior Configuration
    # ═══════════════════════════════════════════════════════════════════════
    
    agent_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for agent LLM sampling",
    )
    
    agent_max_tokens: int = Field(
        default=2048,
        ge=1,
        le=128000,
        description="Maximum tokens per agent response",
    )
    
    agent_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Timeout for individual agent operations",
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Retry & Resilience Configuration
    # ═══════════════════════════════════════════════════════════════════════
    
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed operations",
    )
    
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.0,
        description="Initial delay between retry attempts",
    )
    
    retry_backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        description="Exponential backoff multiplier for retries",
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Logging & Observability
    # ═══════════════════════════════════════════════════════════════════════
    
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging verbosity level",
    )
    
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection and reporting",
    )
    
    metrics_export_interval_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Interval for metrics export",
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Output & Persistence
    # ═══════════════════════════════════════════════════════════════════════
    
    output_dir: Path = Field(
        default=Path("./swarm_output"),
        description="Directory for experiment outputs and artifacts",
    )
    
    checkpoint_interval_minutes: float = Field(
        default=10.0,
        gt=0.0,
        description="Interval for checkpointing swarm state",
    )
    
    enable_checkpointing: bool = Field(
        default=True,
        description="Enable automatic state checkpointing",
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Advanced Configuration
    # ═══════════════════════════════════════════════════════════════════════
    
    enable_adaptive_scaling: bool = Field(
        default=False,
        description="Enable dynamic agent count adjustment based on workload",
    )
    
    min_agents_for_scaling: int = Field(
        default=2,
        ge=1,
        description="Minimum agents when adaptive scaling is enabled",
    )
    
    max_agents_for_scaling: int = Field(
        default=50,
        ge=1,
        description="Maximum agents when adaptive scaling is enabled",
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Validators
    # ═══════════════════════════════════════════════════════════════════════
    
    @field_validator("max_agents_for_scaling")
    @classmethod
    def validate_scaling_bounds(cls, v: int, info: Any) -> int:
        """Ensure max scaling agents >= min scaling agents."""
        min_val = info.data.get("min_agents_for_scaling", 2)
        if v < min_val:
            raise ValueError(
                f"max_agents_for_scaling ({v}) must be >= "
                f"min_agents_for_scaling ({min_val})"
            )
        return v
    
    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        """Ensure output directory is a valid path."""
        return v.expanduser().resolve()
    
    @model_validator(mode="after")
    def validate_budget_constraints(self) -> Self:
        """Validate that budget constraints are reasonable."""
        if self.experiment_budget_minutes < 1.0 and self.max_agents > 10:
            import warnings
            warnings.warn(
                f"Low budget ({self.experiment_budget_minutes} min) with "
                f"many agents ({self.max_agents}) may cause premature termination",
                UserWarning,
            )
        return self
    
    # ═══════════════════════════════════════════════════════════════════════
    # Public Methods
    # ═══════════════════════════════════════════════════════════════════════
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        return self.model_dump()
    
    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to a JSON string."""
        return self.model_dump_json(indent=indent)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SwarmConfig:
        """Create configuration from a dictionary."""
        return cls.model_validate(data)
    
    @classmethod
    def from_env_file(cls, env_file: str | Path) -> SwarmConfig:
        """Load configuration from an environment file."""
        env_path = Path(env_file)
        if not env_path.exists():
            raise FileNotFoundError(f"Environment file not found: {env_path}")
        
        # Temporarily override env_file in config
        original_config = cls.model_config.get("env_file")
        cls.model_config["env_file"] = str(env_path)
        try:
            return cls()
        finally:
            cls.model_config["env_file"] = original_config
    
    def save_to_file(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json())
    
    @classmethod
    def load_from_file(cls, path: str | Path) -> SwarmConfig:
        """Load configuration from a JSON file."""
        import json
        
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        data = json.loads(file_path.read_text())
        return cls.from_dict(data)
    
    def get_agent_config(self) -> AgentConfig:
        """Extract agent-specific configuration subset."""
        return AgentConfig(
            temperature=self.agent_temperature,
            max_tokens=self.agent_max_tokens,
            timeout_seconds=self.agent_timeout_seconds,
            provider=self.provider,
        )
    
    def get_retry_config(self) -> RetryConfig:
        """Extract retry-specific configuration subset."""
        return RetryConfig(
            max_retries=self.max_retries,
            delay_seconds=self.retry_delay_seconds,
            backoff_factor=self.retry_backoff_factor,
        )
    
    def __repr__(self) -> str:
        return (
            f"SwarmConfig("
            f"max_agents={self.max_agents}, "
            f"budget={self.experiment_budget_minutes}min, "
            f"provider={self.provider.value}, "
            f"hardware={self.hardware_target.value})"
        )


class AgentConfig(BaseModel):
    """Configuration subset for individual agent behavior."""
    
    model_config = ConfigDict(frozen=True)
    
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=128000)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    provider: Provider = Field(default=Provider.OPENAI)


class RetryConfig(BaseModel):
    """Configuration subset for retry behavior."""
    
    model_config = ConfigDict(frozen=True)
    
    max_retries: int = Field(default=3, ge=0, le=10)
    delay_seconds: float = Field(default=1.0, ge=0.0)
    backoff_factor: float = Field(default=2.0, ge=1.0)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a specific retry attempt."""
        return self.delay_seconds * (self.backoff_factor ** attempt)


class ExperimentConfig(BaseModel):
    """Configuration for a specific experiment run."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    name: str = Field(..., min_length=1, max_length=256)
    description: str = Field(default="")
    tags: list[str] = Field(default_factory=list)
    swarm_config: SwarmConfig = Field(default_factory=SwarmConfig)
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure experiment name is valid."""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            if char in v:
                raise ValueError(f"Experiment name contains invalid character: {char}")
        return v.strip()


def get_config(
    env_file: str | Path | None = None,
    **overrides: Any,
) -> SwarmConfig:
    """
    Get SwarmConfig with optional environment file and overrides.
    
    Args:
        env_file: Optional path to environment file
        **overrides: Configuration overrides
        
    Returns:
        Configured SwarmConfig instance
        
    Examples:
        >>> config = get_config()
        >>> config = get_config(env_file=".env.production")
        >>> config = get_config(max_agents=20, provider=Provider.ANTHROPIC)
    """
    if env_file:
        config = SwarmConfig.from_env_file(env_file)
    else:
        config = SwarmConfig()
    
    if overrides:
        config = config.model_copy(update=overrides)
    
    return config


# ═══════════════════════════════════════════════════════════════════════════════
# Default Configuration Instance
# ═══════════════════════════════════════════════════════════════════════════════

default_config = SwarmConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# __all__ Definition
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "SwarmConfig",
    "AgentConfig",
    "RetryConfig",
    "ExperimentConfig",
    "Provider",
    "HardwareTarget",
    "RatchetMetric",
    "PollinationFrequency",
    "LogLevel",
    "get_config",
    "default_config",
]
