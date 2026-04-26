"""
autoconstitution CheckpointManager

A comprehensive checkpointing system for saving and restoring complete autoconstitution
session state, enabling overnight runs to be paused and resumed, with full failure recovery.

Author: autoconstitution Team
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
import re
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    TypeVar,
    Union,
    runtime_checkable,
)
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logger = logging.getLogger(__name__)


class CheckpointError(Exception):
    """Base exception for checkpoint-related errors."""
    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when a requested checkpoint cannot be found."""
    pass


class CheckpointCorruptedError(CheckpointError):
    """Raised when a checkpoint file is corrupted or invalid."""
    pass


class CheckpointVersionError(CheckpointError):
    """Raised when checkpoint version is incompatible."""
    pass


class CheckpointStateError(CheckpointError):
    """Raised when checkpoint state is invalid for an operation."""
    pass


class CheckpointPriority(Enum):
    """Priority levels for checkpoint operations."""
    CRITICAL = auto()
    HIGH = auto()
    NORMAL = auto()
    LOW = auto()


@runtime_checkable
class Checkpointable(Protocol):
    """Protocol for objects that can be checkpointed."""
    
    def to_checkpoint(self) -> Dict[str, Any]:
        """Serialize object state to a dictionary."""
        ...
    
    def from_checkpoint(self, state: Dict[str, Any]) -> None:
        """Restore object state from a dictionary."""
        ...
    
    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        """Return metadata about the checkpointable object."""
        ...


T = TypeVar("T")


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_id: str
    session_id: str
    timestamp: float
    version: str
    priority: CheckpointPriority
    description: str
    tags: Set[str] = field(default_factory=set)
    parent_checkpoint_id: Optional[str] = None
    checksum: Optional[str] = None
    size_bytes: int = 0
    agent_count: int = 0
    iteration: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "version": self.version,
            "priority": self.priority.name,
            "description": self.description,
            "tags": list(self.tags),
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "agent_count": self.agent_count,
            "iteration": self.iteration,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CheckpointMetadata:
        """Create metadata from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            session_id=data["session_id"],
            timestamp=data["timestamp"],
            version=data["version"],
            priority=CheckpointPriority[data["priority"]],
            description=data["description"],
            tags=set(data.get("tags", [])),
            parent_checkpoint_id=data.get("parent_checkpoint_id"),
            checksum=data.get("checksum"),
            size_bytes=data.get("size_bytes", 0),
            agent_count=data.get("agent_count", 0),
            iteration=data.get("iteration", 0),
        )


@dataclass
class AgentState:
    """State container for a single agent."""
    agent_id: str
    agent_type: str
    status: str
    memory_state: Dict[str, Any] = field(default_factory=dict)
    task_queue: List[Dict[str, Any]] = field(default_factory=list)
    completed_tasks: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    custom_state: Dict[str, Any] = field(default_factory=dict)
    last_active: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status,
            "memory_state": self.memory_state,
            "task_queue": self.task_queue,
            "completed_tasks": self.completed_tasks,
            "metrics": self.metrics,
            "custom_state": self.custom_state,
            "last_active": self.last_active,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentState:
        """Create agent state from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            status=data["status"],
            memory_state=data.get("memory_state", {}),
            task_queue=data.get("task_queue", []),
            completed_tasks=data.get("completed_tasks", []),
            metrics=data.get("metrics", {}),
            custom_state=data.get("custom_state", {}),
            last_active=data.get("last_active", time.time()),
        )


@dataclass
class SwarmSessionState:
    """Complete state of a autoconstitution session."""
    session_id: str
    iteration: int
    global_config: Dict[str, Any] = field(default_factory=dict)
    agents: Dict[str, AgentState] = field(default_factory=dict)
    shared_memory: Dict[str, Any] = field(default_factory=dict)
    message_queue: List[Dict[str, Any]] = field(default_factory=list)
    research_goals: List[Dict[str, Any]] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    last_checkpoint_time: Optional[float] = None
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session state to dictionary."""
        return {
            "session_id": self.session_id,
            "iteration": self.iteration,
            "global_config": self.global_config,
            "agents": {k: v.to_dict() for k, v in self.agents.items()},
            "shared_memory": self.shared_memory,
            "message_queue": self.message_queue,
            "research_goals": self.research_goals,
            "results": self.results,
            "metrics": self.metrics,
            "start_time": self.start_time,
            "last_checkpoint_time": self.last_checkpoint_time,
            "custom_data": self.custom_data,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SwarmSessionState:
        """Create session state from dictionary."""
        return cls(
            session_id=data["session_id"],
            iteration=data["iteration"],
            global_config=data.get("global_config", {}),
            agents={
                k: AgentState.from_dict(v) 
                for k, v in data.get("agents", {}).items()
            },
            shared_memory=data.get("shared_memory", {}),
            message_queue=data.get("message_queue", []),
            research_goals=data.get("research_goals", []),
            results=data.get("results", {}),
            metrics=data.get("metrics", {}),
            start_time=data.get("start_time", time.time()),
            last_checkpoint_time=data.get("last_checkpoint_time"),
            custom_data=data.get("custom_data", {}),
        )


@dataclass
class Checkpoint:
    """Complete checkpoint containing state and metadata."""
    metadata: CheckpointMetadata
    state: SwarmSessionState
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "state": self.state.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Checkpoint:
        """Create checkpoint from dictionary."""
        return cls(
            metadata=CheckpointMetadata.from_dict(data["metadata"]),
            state=SwarmSessionState.from_dict(data["state"]),
        )


class CheckpointSerializer(ABC):
    """Abstract base class for checkpoint serializers."""
    
    @abstractmethod
    def serialize(self, checkpoint: Checkpoint) -> bytes:
        """Serialize a checkpoint to bytes."""
        raise NotImplementedError
    
    @abstractmethod
    def deserialize(self, data: bytes) -> Checkpoint:
        """Deserialize bytes to a checkpoint."""
        raise NotImplementedError
    
    @abstractmethod
    def get_extension(self) -> str:
        """Get the file extension for this serializer."""
        raise NotImplementedError


class PickleSerializer(CheckpointSerializer):
    """Serializer using Python's pickle module.

    .. warning::
        Pickle is **not safe** against maliciously constructed data. Loading a
        pickle file crafted by an attacker can execute arbitrary code. Use
        :class:`PickleSerializer` only for checkpoints produced and stored by
        your own process, in a directory you control. If checkpoints may ever
        originate from untrusted sources (imports, shared drives, network
        shares), use :class:`JSONSerializer` instead.
    """

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
        self.protocol = protocol

    def serialize(self, checkpoint: Checkpoint) -> bytes:
        """Serialize checkpoint using pickle."""
        return pickle.dumps(checkpoint, protocol=self.protocol)

    def deserialize(self, data: bytes) -> Checkpoint:
        """Deserialize checkpoint using pickle.

        .. warning::
            Only call on data you generated yourself. Never on attacker-
            controlled bytes.
        """
        try:
            return pickle.loads(data)  # noqa: S301 -- documented risk, see class docstring
        except pickle.UnpicklingError as e:
            raise CheckpointCorruptedError(f"Failed to deserialize checkpoint: {e}")

    def get_extension(self) -> str:
        """Get pickle file extension."""
        return ".pkl"


class JSONSerializer(CheckpointSerializer):
    """Serializer using JSON format (human-readable)."""
    
    def serialize(self, checkpoint: Checkpoint) -> bytes:
        """Serialize checkpoint to JSON bytes."""
        return json.dumps(
            checkpoint.to_dict(), 
            indent=2, 
            default=str,
            sort_keys=True,  # Ensure deterministic serialization
        ).encode("utf-8")
    
    def deserialize(self, data: bytes) -> Checkpoint:
        """Deserialize JSON bytes to checkpoint."""
        try:
            dict_data = json.loads(data.decode("utf-8"))
            return Checkpoint.from_dict(dict_data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise CheckpointCorruptedError(f"Failed to deserialize JSON checkpoint: {e}")
    
    def get_extension(self) -> str:
        """Get JSON file extension."""
        return ".json"


class CheckpointStorage(ABC):
    """Abstract base class for checkpoint storage backends."""
    
    @abstractmethod
    async def save(self, checkpoint_id: str, data: bytes) -> None:
        """Save checkpoint data."""
        raise NotImplementedError
    
    @abstractmethod
    async def load(self, checkpoint_id: str) -> bytes:
        """Load checkpoint data."""
        raise NotImplementedError
    
    @abstractmethod
    async def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint."""
        raise NotImplementedError
    
    @abstractmethod
    async def list_checkpoints(self) -> List[str]:
        """List all checkpoint IDs."""
        raise NotImplementedError
    
    @abstractmethod
    async def exists(self, checkpoint_id: str) -> bool:
        """Check if a checkpoint exists."""
        raise NotImplementedError
    
    @abstractmethod
    async def get_size(self, checkpoint_id: str) -> int:
        """Get checkpoint size in bytes."""
        raise NotImplementedError


#: Allowlist for checkpoint IDs. Keeps filenames safe on every OS we care about.
#: Accepts only ASCII letters, digits, underscore, dash, and dot. No separators,
#: no NUL, no control chars, no unicode tricks.
_SAFE_CHECKPOINT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _validate_checkpoint_id(checkpoint_id: str) -> None:
    """Reject checkpoint IDs that could escape the storage directory.

    Raises:
        CheckpointError: If the ID contains path separators, parent references,
            null bytes, or characters outside the safe allowlist.
    """
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise CheckpointError("Checkpoint ID must be a non-empty string")
    if checkpoint_id in (".", ".."):
        raise CheckpointError("Checkpoint ID must not be '.' or '..'")
    if not _SAFE_CHECKPOINT_ID_RE.fullmatch(checkpoint_id):
        raise CheckpointError(
            f"Unsafe checkpoint ID: {checkpoint_id!r}. "
            "Only [A-Za-z0-9._-], up to 128 chars, are allowed."
        )


class FileSystemStorage(CheckpointStorage):
    """File system based checkpoint storage."""

    def __init__(
        self,
        base_path: Union[str, Path],
        extension: str = ".pkl"
    ) -> None:
        # Resolve once so the root is always absolute. Symlinks in the root itself
        # are intentionally preserved (caller's choice), but we use resolve() for
        # containment checks below.
        self.base_path = Path(base_path).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.extension = extension
        self._lock = asyncio.Lock()

    def _get_path(self, checkpoint_id: str) -> Path:
        """Get the file path for a checkpoint, validating containment."""
        _validate_checkpoint_id(checkpoint_id)
        candidate = (self.base_path / f"{checkpoint_id}{self.extension}").resolve()
        # Defence in depth: even with the allowlist above, make sure the resolved
        # path is still inside base_path. This catches weird cases where the
        # extension itself contains separators, base_path contains symlinks to
        # elsewhere, or the OS normalises paths differently.
        try:
            candidate.relative_to(self.base_path)
        except ValueError as exc:
            raise CheckpointError(
                f"Checkpoint path escapes storage root: {candidate}"
            ) from exc
        return candidate
    
    async def save(self, checkpoint_id: str, data: bytes) -> None:
        """Save checkpoint to file system."""
        async with self._lock:
            path = self._get_path(checkpoint_id)
            try:
                # Write to temp file first, then rename for atomicity
                temp_path = path.with_suffix(".tmp")
                await asyncio.to_thread(self._write_file, temp_path, data)
                await asyncio.to_thread(temp_path.rename, path)
                logger.debug(f"Saved checkpoint to {path}")
            except OSError as e:
                raise CheckpointError(f"Failed to save checkpoint: {e}")
    
    def _write_file(self, path: Path, data: bytes) -> None:
        """Write data to file (sync, for thread pool)."""
        with open(path, "wb") as f:
            f.write(data)
    
    async def load(self, checkpoint_id: str) -> bytes:
        """Load checkpoint from file system."""
        async with self._lock:
            path = self._get_path(checkpoint_id)
            if not path.exists():
                raise CheckpointNotFoundError(
                    f"Checkpoint not found: {checkpoint_id}"
                )
            try:
                return await asyncio.to_thread(self._read_file, path)
            except OSError as e:
                raise CheckpointError(f"Failed to load checkpoint: {e}")
    
    def _read_file(self, path: Path) -> bytes:
        """Read data from file (sync, for thread pool)."""
        with open(path, "rb") as f:
            return f.read()
    
    async def delete(self, checkpoint_id: str) -> None:
        """Delete checkpoint from file system."""
        async with self._lock:
            path = self._get_path(checkpoint_id)
            if path.exists():
                await asyncio.to_thread(path.unlink)
                logger.debug(f"Deleted checkpoint: {checkpoint_id}")
    
    async def list_checkpoints(self) -> List[str]:
        """List all checkpoint IDs."""
        async with self._lock:
            files = await asyncio.to_thread(
                lambda: list(self.base_path.glob(f"*{self.extension}"))
            )
            return [
                f.stem for f in files 
                if f.is_file() and not f.name.endswith(".tmp")
            ]
    
    async def exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists."""
        return self._get_path(checkpoint_id).exists()
    
    async def get_size(self, checkpoint_id: str) -> int:
        """Get checkpoint size in bytes."""
        path = self._get_path(checkpoint_id)
        if not path.exists():
            return 0
        return path.stat().st_size


class RecoveryStrategy(Enum):
    """Strategies for recovery from failures."""
    LATEST = auto()
    LAST_SUCCESSFUL = auto()
    SPECIFIC = auto()
    ROLLBACK = auto()


@dataclass
class RecoveryConfig:
    """Configuration for recovery operations."""
    strategy: RecoveryStrategy = RecoveryStrategy.LATEST
    target_checkpoint_id: Optional[str] = None
    rollback_iterations: int = 1
    max_retries: int = 3
    retry_delay: float = 1.0
    validate_on_restore: bool = True
    notify_on_recovery: bool = True


class CheckpointManager:
    """
    Manager for autoconstitution session checkpointing.
    
    Handles saving and restoring complete session state, enabling overnight runs
    to be paused and resumed, with full failure recovery capabilities.
    
    Example:
        >>> manager = CheckpointManager("/path/to/checkpoints")
        >>> session = SwarmSessionState(session_id="research_001", iteration=0)
        >>> checkpoint_id = await manager.save(session, "Initial checkpoint")
        >>> restored = await manager.restore(checkpoint_id)
    """
    
    # Current checkpoint format version
    VERSION = "2.0.0"
    
    def __init__(
        self,
        storage: Optional[CheckpointStorage] = None,
        serializer: Optional[CheckpointSerializer] = None,
        auto_checkpoint_interval: Optional[float] = None,
        max_checkpoints: int = 100,
        compression: bool = False,
    ) -> None:
        """
        Initialize the checkpoint manager.
        
        Args:
            storage: Storage backend for checkpoints (default: FileSystemStorage)
            serializer: Serializer for checkpoint data (default: PickleSerializer)
            auto_checkpoint_interval: Automatic checkpoint interval in seconds
            max_checkpoints: Maximum number of checkpoints to retain
            compression: Whether to compress checkpoint data
        """
        self.storage = storage or FileSystemStorage(
            "/tmp/autoconstitution/checkpoints"
        )
        self.serializer = serializer or PickleSerializer()
        self.auto_checkpoint_interval = auto_checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.compression = compression
        
        # State tracking
        self._session_id: Optional[str] = None
        self._checkpoints: Dict[str, CheckpointMetadata] = {}
        self._checkpoint_history: List[str] = []
        self._successful_checkpoints: List[str] = []
        self._auto_checkpoint_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()
        
        # Callbacks
        self._pre_save_callbacks: List[Callable[[SwarmSessionState], None]] = []
        self._post_save_callbacks: List[Callable[[CheckpointMetadata], None]] = []
        self._pre_restore_callbacks: List[Callable[[CheckpointMetadata], None]] = []
        self._post_restore_callbacks: List[Callable[[SwarmSessionState], None]] = []
        
        # Recovery tracking
        self._recovery_attempts: Dict[str, int] = {}
        self._last_successful_checkpoint: Optional[str] = None
    
    # ==================== Registration Methods ====================
    
    def register_pre_save_callback(
        self, 
        callback: Callable[[SwarmSessionState], None]
    ) -> None:
        """Register a callback to run before saving a checkpoint."""
        self._pre_save_callbacks.append(callback)
    
    def register_post_save_callback(
        self, 
        callback: Callable[[CheckpointMetadata], None]
    ) -> None:
        """Register a callback to run after saving a checkpoint."""
        self._post_save_callbacks.append(callback)
    
    def register_pre_restore_callback(
        self, 
        callback: Callable[[CheckpointMetadata], None]
    ) -> None:
        """Register a callback to run before restoring a checkpoint."""
        self._pre_restore_callbacks.append(callback)
    
    def register_post_restore_callback(
        self, 
        callback: Callable[[SwarmSessionState], None]
    ) -> None:
        """Register a callback to run after restoring a checkpoint."""
        self._post_restore_callbacks.append(callback)
    
    # ==================== Core Checkpoint Operations ====================
    
    async def save(
        self,
        state: SwarmSessionState,
        description: str = "",
        priority: CheckpointPriority = CheckpointPriority.NORMAL,
        tags: Optional[Set[str]] = None,
        parent_checkpoint_id: Optional[str] = None,
    ) -> str:
        """
        Save a checkpoint of the current session state.
        
        Args:
            state: Current session state to checkpoint
            description: Human-readable description of the checkpoint
            priority: Priority level for this checkpoint
            tags: Optional tags for categorization
            parent_checkpoint_id: ID of parent checkpoint for lineage
            
        Returns:
            The checkpoint ID
            
        Raises:
            CheckpointError: If saving fails
        """
        async with self._lock:
            # Run pre-save callbacks
            for callback in self._pre_save_callbacks:
                try:
                    callback(state)
                except Exception as e:
                    logger.warning(f"Pre-save callback failed: {e}")
            
            # Generate checkpoint ID
            checkpoint_id = self._generate_checkpoint_id()
            
            # Update state timestamp
            state.last_checkpoint_time = time.time()
            
            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                session_id=state.session_id,
                timestamp=time.time(),
                version=self.VERSION,
                priority=priority,
                description=description,
                tags=tags or set(),
                parent_checkpoint_id=parent_checkpoint_id,
                agent_count=len(state.agents),
                iteration=state.iteration,
            )
            
            # Create checkpoint with metadata (without checksum initially)
            checkpoint = Checkpoint(metadata=metadata, state=state)
            
            # Serialize for size calculation
            try:
                temp_data = self.serializer.serialize(checkpoint)
            except Exception as e:
                raise CheckpointError(f"Serialization failed: {e}")
            
            size_bytes = len(temp_data)
            
            # Calculate checksum on the serialized data (without checksum in metadata)
            # This checksum represents the state content, not the final storage format
            content_checksum = hashlib.sha256(temp_data).hexdigest()
            
            # Update metadata with checksum and size
            from dataclasses import replace
            metadata = replace(
                metadata, 
                checksum=content_checksum, 
                size_bytes=size_bytes
            )
            
            # Create final checkpoint with updated metadata
            checkpoint = Checkpoint(metadata=metadata, state=state)
            
            # Final serialization for storage
            data = self.serializer.serialize(checkpoint)
            
            # Save to storage
            await self.storage.save(checkpoint_id, data)
            
            # Update tracking
            self._checkpoints[checkpoint_id] = metadata
            self._checkpoint_history.append(checkpoint_id)
            self._session_id = state.session_id
            
            # Mark as successful
            self._successful_checkpoints.append(checkpoint_id)
            self._last_successful_checkpoint = checkpoint_id
            
            # Cleanup old checkpoints
            await self._cleanup_old_checkpoints()
            
            # Run post-save callbacks
            for callback in self._post_save_callbacks:
                try:
                    callback(metadata)
                except Exception as e:
                    logger.warning(f"Post-save callback failed: {e}")
            
            logger.info(
                f"Saved checkpoint {checkpoint_id} "
                f"(agents: {metadata.agent_count}, iteration: {metadata.iteration})"
            )
            
            return checkpoint_id
    
    async def restore(
        self, 
        checkpoint_id: str,
        validate: bool = True
    ) -> SwarmSessionState:
        """
        Restore session state from a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to restore
            validate: Whether to validate checksum
            
        Returns:
            The restored session state
            
        Raises:
            CheckpointNotFoundError: If checkpoint doesn't exist
            CheckpointCorruptedError: If checkpoint is corrupted
        """
        async with self._lock:
            # Load from storage
            try:
                data = await self.storage.load(checkpoint_id)
            except CheckpointNotFoundError:
                raise
            except Exception as e:
                raise CheckpointError(f"Failed to load checkpoint: {e}")
            
            # Deserialize
            try:
                checkpoint = self.serializer.deserialize(data)
            except CheckpointCorruptedError:
                raise
            except Exception as e:
                raise CheckpointCorruptedError(f"Deserialization failed: {e}")
            
            # Validate checksum by computing content checksum (without checksum field)
            if validate and checkpoint.metadata.checksum:
                # Create a copy of metadata without checksum for validation
                # Use None for checksum (not "") to match original serialization
                from dataclasses import replace
                validation_metadata = replace(
                    checkpoint.metadata, 
                    checksum=None, 
                    size_bytes=0
                )
                validation_checkpoint = Checkpoint(
                    metadata=validation_metadata, 
                    state=checkpoint.state
                )
                validation_data = self.serializer.serialize(validation_checkpoint)
                actual_checksum = hashlib.sha256(validation_data).hexdigest()
                
                if actual_checksum != checkpoint.metadata.checksum:
                    raise CheckpointCorruptedError(
                        "Checksum mismatch - checkpoint may be corrupted"
                    )
            
            # Validate version
            if checkpoint.metadata.version != self.VERSION:
                logger.warning(
                    f"Checkpoint version mismatch: "
                    f"{checkpoint.metadata.version} vs {self.VERSION}"
                )
            
            # Run pre-restore callbacks
            for callback in self._pre_restore_callbacks:
                try:
                    callback(checkpoint.metadata)
                except Exception as e:
                    logger.warning(f"Pre-restore callback failed: {e}")
            
            # Update tracking
            self._session_id = checkpoint.state.session_id
            
            # Run post-restore callbacks
            for callback in self._post_restore_callbacks:
                try:
                    callback(checkpoint.state)
                except Exception as e:
                    logger.warning(f"Post-restore callback failed: {e}")
            
            logger.info(
                f"Restored checkpoint {checkpoint_id} "
                f"(iteration: {checkpoint.state.iteration})"
            )
            
            return checkpoint.state
    
    async def restore_latest(
        self, 
        session_id: Optional[str] = None
    ) -> SwarmSessionState:
        """
        Restore from the most recent checkpoint.
        
        Args:
            session_id: Optional session ID filter
            
        Returns:
            The restored session state
        """
        checkpoints = await self.list_checkpoints(session_id=session_id)
        if not checkpoints:
            raise CheckpointNotFoundError("No checkpoints found")
        
        # Sort by timestamp descending
        latest = max(checkpoints, key=lambda c: c.timestamp)
        return await self.restore(latest.checkpoint_id)
    
    async def recover(
        self, 
        config: Optional[RecoveryConfig] = None
    ) -> SwarmSessionState:
        """
        Recover from failure using specified strategy.
        
        Args:
            config: Recovery configuration
            
        Returns:
            The recovered session state
        """
        config = config or RecoveryConfig()
        
        for attempt in range(config.max_retries):
            try:
                if config.strategy == RecoveryStrategy.LATEST:
                    return await self.restore_latest()
                
                elif config.strategy == RecoveryStrategy.LAST_SUCCESSFUL:
                    if self._last_successful_checkpoint:
                        return await self.restore(self._last_successful_checkpoint)
                    return await self.restore_latest()
                
                elif config.strategy == RecoveryStrategy.SPECIFIC:
                    if not config.target_checkpoint_id:
                        raise CheckpointError(
                            "SPECIFIC strategy requires target_checkpoint_id"
                        )
                    return await self.restore(config.target_checkpoint_id)
                
                elif config.strategy == RecoveryStrategy.ROLLBACK:
                    return await self._rollback(config.rollback_iterations)
                
                else:
                    raise CheckpointError(f"Unknown recovery strategy: {config.strategy}")
                    
            except Exception as e:
                logger.warning(f"Recovery attempt {attempt + 1} failed: {e}")
                if attempt < config.max_retries - 1:
                    await asyncio.sleep(config.retry_delay * (attempt + 1))
                else:
                    raise CheckpointError(f"Recovery failed after {config.max_retries} attempts: {e}")
        
        raise CheckpointError("Recovery failed")
    
    async def _rollback(self, iterations: int) -> SwarmSessionState:
        """Rollback to a checkpoint from N iterations ago."""
        if len(self._checkpoint_history) < iterations:
            raise CheckpointError(
                f"Cannot rollback {iterations} iterations, "
                f"only {len(self._checkpoint_history)} checkpoints available"
            )
        
        target_id = self._checkpoint_history[-iterations]
        return await self.restore(target_id)
    
    # ==================== Agent State Management ====================
    
    async def save_agent_state(
        self,
        session_state: SwarmSessionState,
        agent_id: str,
        description: str = "",
    ) -> str:
        """
        Save checkpoint for a specific agent.
        
        Creates a lightweight checkpoint focused on a single agent's state.
        
        Args:
            session_state: Current session state
            agent_id: ID of agent to checkpoint
            description: Checkpoint description
            
        Returns:
            Checkpoint ID
        """
        if agent_id not in session_state.agents:
            raise CheckpointError(f"Agent {agent_id} not found in session")
        
        # Tag with agent ID for filtering
        tags = {f"agent:{agent_id}"}
        
        return await self.save(
            state=session_state,
            description=f"Agent checkpoint: {description}",
            priority=CheckpointPriority.HIGH,
            tags=tags,
        )
    
    async def restore_agent_state(
        self,
        checkpoint_id: str,
        agent_id: str,
        target_session: SwarmSessionState,
    ) -> None:
        """
        Restore a specific agent's state from a checkpoint.
        
        Args:
            checkpoint_id: Source checkpoint ID
            agent_id: Agent ID to restore
            target_session: Session to update with restored agent state
        """
        checkpoint = await self.restore(checkpoint_id)
        
        if agent_id not in checkpoint.agents:
            raise CheckpointError(
                f"Agent {agent_id} not found in checkpoint {checkpoint_id}"
            )
        
        target_session.agents[agent_id] = checkpoint.agents[agent_id]
        logger.info(f"Restored agent {agent_id} from checkpoint {checkpoint_id}")
    
    # ==================== Auto-Checkpointing ====================
    
    async def start_auto_checkpoint(
        self,
        state_provider: Callable[[], SwarmSessionState],
    ) -> None:
        """
        Start automatic periodic checkpointing.
        
        Args:
            state_provider: Callable that returns current session state
        """
        if self._auto_checkpoint_task is not None:
            logger.warning("Auto-checkpoint already running")
            return
        
        if self.auto_checkpoint_interval is None:
            raise CheckpointError("auto_checkpoint_interval not configured")
        
        self._running = True
        self._auto_checkpoint_task = asyncio.create_task(
            self._auto_checkpoint_loop(state_provider)
        )
        logger.info(f"Started auto-checkpoint (interval: {self.auto_checkpoint_interval}s)")
    
    async def stop_auto_checkpoint(self) -> None:
        """Stop automatic checkpointing."""
        self._running = False
        if self._auto_checkpoint_task:
            self._auto_checkpoint_task.cancel()
            try:
                await self._auto_checkpoint_task
            except asyncio.CancelledError:
                pass
            self._auto_checkpoint_task = None
            logger.info("Stopped auto-checkpoint")
    
    async def _auto_checkpoint_loop(
        self,
        state_provider: Callable[[], SwarmSessionState],
    ) -> None:
        """Background task for automatic checkpointing."""
        while self._running:
            try:
                await asyncio.sleep(self.auto_checkpoint_interval)
                if self._running:
                    state = state_provider()
                    await self.save(
                        state=state,
                        description="Auto-checkpoint",
                        priority=CheckpointPriority.NORMAL,
                        tags={"auto"},
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-checkpoint failed: {e}")
    
    @asynccontextmanager
    async def auto_checkpoint_context(
        self,
        state_provider: Callable[[], SwarmSessionState],
    ):
        """Context manager for auto-checkpointing."""
        await self.start_auto_checkpoint(state_provider)
        try:
            yield self
        finally:
            await self.stop_auto_checkpoint()
    
    # ==================== Checkpoint Management ====================
    
    async def list_checkpoints(
        self,
        session_id: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        min_priority: Optional[CheckpointPriority] = None,
    ) -> List[CheckpointMetadata]:
        """
        List checkpoints with optional filtering.
        
        Args:
            session_id: Filter by session ID
            tags: Filter by tags (all must match)
            min_priority: Minimum priority level
            
        Returns:
            List of matching checkpoint metadata
        """
        # Refresh checkpoint list from storage
        checkpoint_ids = await self.storage.list_checkpoints()
        
        results = []
        for cid in checkpoint_ids:
            if cid in self._checkpoints:
                meta = self._checkpoints[cid]
            else:
                # Load metadata from storage
                try:
                    data = await self.storage.load(cid)
                    checkpoint = self.serializer.deserialize(data)
                    meta = checkpoint.metadata
                    self._checkpoints[cid] = meta
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {cid}: {e}")
                    continue
            
            # Apply filters
            if session_id and meta.session_id != session_id:
                continue
            if tags and not tags.issubset(meta.tags):
                continue
            if min_priority and meta.priority.value < min_priority.value:
                continue
            
            results.append(meta)
        
        return sorted(results, key=lambda m: m.timestamp, reverse=True)
    
    async def delete_checkpoint(self, checkpoint_id: str) -> None:
        """
        Delete a specific checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to delete
        """
        await self.storage.delete(checkpoint_id)
        
        if checkpoint_id in self._checkpoints:
            del self._checkpoints[checkpoint_id]
        
        if checkpoint_id in self._checkpoint_history:
            self._checkpoint_history.remove(checkpoint_id)
        
        if checkpoint_id in self._successful_checkpoints:
            self._successful_checkpoints.remove(checkpoint_id)
        
        if self._last_successful_checkpoint == checkpoint_id:
            self._last_successful_checkpoint = None
        
        logger.info(f"Deleted checkpoint: {checkpoint_id}")
    
    async def delete_session_checkpoints(self, session_id: str) -> int:
        """
        Delete all checkpoints for a session.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            Number of checkpoints deleted
        """
        checkpoints = await self.list_checkpoints(session_id=session_id)
        count = 0
        for meta in checkpoints:
            await self.delete_checkpoint(meta.checkpoint_id)
            count += 1
        logger.info(f"Deleted {count} checkpoints for session {session_id}")
        return count
    
    async def get_checkpoint_info(
        self, 
        checkpoint_id: str
    ) -> Optional[CheckpointMetadata]:
        """Get metadata for a specific checkpoint."""
        if checkpoint_id in self._checkpoints:
            return self._checkpoints[checkpoint_id]
        
        try:
            data = await self.storage.load(checkpoint_id)
            checkpoint = self.serializer.deserialize(data)
            return checkpoint.metadata
        except Exception:
            return None
    
    async def checkpoint_exists(self, checkpoint_id: str) -> bool:
        """Check if a checkpoint exists."""
        return await self.storage.exists(checkpoint_id)
    
    async def get_storage_size(self) -> int:
        """Get total storage size used by checkpoints in bytes."""
        total = 0
        for cid in await self.storage.list_checkpoints():
            total += await self.storage.get_size(cid)
        return total
    
    # ==================== Utility Methods ====================
    
    def _generate_checkpoint_id(self) -> str:
        """Generate a unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:8]
        return f"chk_{timestamp}_{unique}"
    
    async def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if exceeding max_checkpoints."""
        if len(self._checkpoint_history) <= self.max_checkpoints:
            return
        
        # Keep CRITICAL and HIGH priority checkpoints longer
        to_remove = []
        normal_checkpoints = [
            cid for cid in self._checkpoint_history[:-self.max_checkpoints//2]
            if cid in self._checkpoints
            and self._checkpoints[cid].priority in 
            (CheckpointPriority.NORMAL, CheckpointPriority.LOW)
        ]
        
        # Remove oldest normal/low priority checkpoints
        num_to_remove = len(self._checkpoint_history) - self.max_checkpoints
        to_remove = normal_checkpoints[:num_to_remove]
        
        for cid in to_remove:
            try:
                await self.delete_checkpoint(cid)
            except Exception as e:
                logger.warning(f"Failed to cleanup checkpoint {cid}: {e}")
    
    def get_checkpoint_lineage(self, checkpoint_id: str) -> List[str]:
        """
        Get the lineage (parent chain) of a checkpoint.
        
        Args:
            checkpoint_id: Starting checkpoint ID
            
        Returns:
            List of checkpoint IDs from oldest to newest
        """
        lineage = []
        current_id = checkpoint_id
        
        while current_id:
            lineage.append(current_id)
            meta = self._checkpoints.get(current_id)
            if meta:
                current_id = meta.parent_checkpoint_id
            else:
                break
        
        return list(reversed(lineage))
    
    async def export_checkpoint(
        self, 
        checkpoint_id: str, 
        export_path: Union[str, Path]
    ) -> Path:
        """
        Export a checkpoint to a file.
        
        Args:
            checkpoint_id: Checkpoint to export
            export_path: Destination path
            
        Returns:
            Path to exported file
        """
        data = await self.storage.load(checkpoint_id)
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        await asyncio.to_thread(export_path.write_bytes, data)
        logger.info(f"Exported checkpoint {checkpoint_id} to {export_path}")
        return export_path
    
    async def import_checkpoint(self, import_path: Union[str, Path]) -> str:
        """
        Import a checkpoint from a file.
        
        Args:
            import_path: Path to checkpoint file
            
        Returns:
            Imported checkpoint ID
        """
        import_path = Path(import_path)
        if not import_path.exists():
            raise CheckpointNotFoundError(f"Import file not found: {import_path}")
        
        data = await asyncio.to_thread(import_path.read_bytes)

        try:
            checkpoint = self.serializer.deserialize(data)
            checkpoint_id = checkpoint.metadata.checkpoint_id
        except Exception as e:
            raise CheckpointCorruptedError(f"Failed to import checkpoint: {e}")

        # Validate the checkpoint ID from the imported file before trusting it
        # as a filename on disk. Imported files may come from anywhere.
        _validate_checkpoint_id(checkpoint_id)

        await self.storage.save(checkpoint_id, data)
        self._checkpoints[checkpoint_id] = checkpoint.metadata
        self._checkpoint_history.append(checkpoint_id)
        
        logger.info(f"Imported checkpoint {checkpoint_id} from {import_path}")
        return checkpoint_id
    
    async def verify_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Verify a checkpoint's integrity.
        
        Args:
            checkpoint_id: Checkpoint to verify
            
        Returns:
            True if valid, False otherwise
        """
        try:
            data = await self.storage.load(checkpoint_id)
            checkpoint = self.serializer.deserialize(data)
            
            if checkpoint.metadata.checksum:
                actual = hashlib.sha256(
                    self.serializer.serialize(checkpoint)
                ).hexdigest()
                return actual == checkpoint.metadata.checksum
            
            return True
        except Exception as e:
            logger.warning(f"Checkpoint verification failed: {e}")
            return False
    
    async def create_diff(
        self, 
        checkpoint_id_1: str, 
        checkpoint_id_2: str
    ) -> Dict[str, Any]:
        """
        Create a diff between two checkpoints.
        
        Args:
            checkpoint_id_1: First checkpoint
            checkpoint_id_2: Second checkpoint
            
        Returns:
            Dictionary containing differences
        """
        state1 = await self.restore(checkpoint_id_1)
        state2 = await self.restore(checkpoint_id_2)
        
        diff = {
            "iteration_diff": state2.iteration - state1.iteration,
            "agent_changes": {},
            "added_agents": [],
            "removed_agents": [],
            "shared_memory_changes": {},
            "results_changes": {},
        }
        
        # Compare agents
        all_agents = set(state1.agents.keys()) | set(state2.agents.keys())
        for agent_id in all_agents:
            if agent_id in state1.agents and agent_id in state2.agents:
                a1 = state1.agents[agent_id]
                a2 = state2.agents[agent_id]
                if a1.status != a2.status:
                    diff["agent_changes"][agent_id] = {
                        "status": {"from": a1.status, "to": a2.status}
                    }
            elif agent_id in state1.agents:
                diff["removed_agents"].append(agent_id)
            else:
                diff["added_agents"].append(agent_id)
        
        return diff
    
    # ==================== Statistics ====================
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get checkpoint manager statistics."""
        all_checkpoints = await self.list_checkpoints()
        
        if not all_checkpoints:
            return {
                "total_checkpoints": 0,
                "total_size_bytes": 0,
                "sessions": [],
            }
        
        sessions = set(c.session_id for c in all_checkpoints)
        priorities = {}
        for c in all_checkpoints:
            priorities[c.priority.name] = priorities.get(c.priority.name, 0) + 1
        
        return {
            "total_checkpoints": len(all_checkpoints),
            "total_size_bytes": sum(c.size_bytes for c in all_checkpoints),
            "sessions": list(sessions),
            "session_count": len(sessions),
            "by_priority": priorities,
            "oldest_checkpoint": min(c.timestamp for c in all_checkpoints),
            "newest_checkpoint": max(c.timestamp for c in all_checkpoints),
        }
    
    async def close(self) -> None:
        """Clean up resources and stop background tasks."""
        await self.stop_auto_checkpoint()
        logger.info("Checkpoint manager closed")
    
    async def __aenter__(self) -> CheckpointManager:
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


# ==================== Convenience Functions ====================

async def create_checkpoint_manager(
    base_path: Union[str, Path] = "/tmp/autoconstitution/checkpoints",
    auto_checkpoint_interval: Optional[float] = None,
    use_json: bool = False,
    max_checkpoints: int = 100,
) -> CheckpointManager:
    """
    Create a configured CheckpointManager instance.
    
    Args:
        base_path: Directory for checkpoint storage
        auto_checkpoint_interval: Auto-checkpoint interval in seconds
        use_json: Use JSON serializer instead of pickle
        max_checkpoints: Maximum checkpoints to retain
        
    Returns:
        Configured CheckpointManager
    """
    serializer = JSONSerializer() if use_json else PickleSerializer()
    storage = FileSystemStorage(
        base_path=base_path,
        extension=serializer.get_extension()
    )
    
    return CheckpointManager(
        storage=storage,
        serializer=serializer,
        auto_checkpoint_interval=auto_checkpoint_interval,
        max_checkpoints=max_checkpoints,
    )


# ==================== Example Usage ====================

async def example_usage():
    """Example of using the CheckpointManager."""
    # Create manager
    manager = await create_checkpoint_manager(
        base_path="/tmp/autoconstitution/checkpoints",
        auto_checkpoint_interval=300,  # 5 minutes
    )
    
    async with manager:
        # Create initial session state
        session = SwarmSessionState(
            session_id="research_session_001",
            iteration=0,
            global_config={"max_iterations": 1000, "topic": "AI Safety"},
        )
        
        # Add some agents
        session.agents["agent_1"] = AgentState(
            agent_id="agent_1",
            agent_type="researcher",
            status="active",
        )
        
        # Save initial checkpoint
        checkpoint_id = await manager.save(
            state=session,
            description="Initial session setup",
            priority=CheckpointPriority.HIGH,
            tags={"init", "setup"},
        )
        print(f"Created checkpoint: {checkpoint_id}")
        
        # Simulate some work
        session.iteration += 1
        session.agents["agent_1"].completed_tasks.append({"task": "research", "result": "found X"})
        
        # Save progress checkpoint
        await manager.save(
            state=session,
            description=f"Iteration {session.iteration} complete",
        )
        
        # Restore from checkpoint
        restored = await manager.restore(checkpoint_id)
        print(f"Restored to iteration: {restored.iteration}")
        
        # List all checkpoints
        checkpoints = await manager.list_checkpoints()
        print(f"Total checkpoints: {len(checkpoints)}")
        
        # Get statistics
        stats = await manager.get_statistics()
        print(f"Statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(example_usage())
