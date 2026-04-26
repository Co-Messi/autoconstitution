"""
autoconstitution BranchManager

Handles Git operations for multi-agent research workflows including:
- Branch creation and management for parallel agent work
- Async commit operations
- Merge conflict detection and resolution
- Automated PR creation for validated improvements

Author: autoconstitution Team
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
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
)
from contextlib import asynccontextmanager
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BranchManager")


# ==================== Input validation ====================
#
# These helpers reject inputs that could be interpreted as git flags (leading
# dash) or contain characters that git / the filesystem refuse anyway. We pair
# them with ``--`` end-of-options markers in command invocations so that even
# a name that slips through validation cannot be confused for an option.

#: Conservative allowlist for identifiers we interpolate into branch names.
#: Matches the characters GitHub and GitLab accept in slugs: letters, digits,
#: underscore, dash, dot, and forward slash for scoping (``agent/001/...``).
#: Forbids leading dash, leading slash, double slash, trailing slash/dot, and
#: the ``..`` sequence which git rejects in ref names.
_SAFE_REF_RE = re.compile(r"^(?!-)(?!/)[A-Za-z0-9._/-]+(?<!/)(?<!\.)$")


def _validate_ref_component(name: str, *, kind: str = "ref") -> str:
    """Validate a string we are about to splice into a git command.

    Raises:
        GitError: If the name is empty, starts with a dash (flag injection),
            contains shell/path metacharacters, or violates git ref naming
            rules. The caller should still pass ``--`` before user data as a
            second line of defence.
    """
    if not isinstance(name, str) or not name:
        raise GitError(f"{kind} must be a non-empty string")
    if name.startswith("-"):
        raise GitError(f"{kind} must not start with '-' (flag injection risk): {name!r}")
    if ".." in name:
        raise GitError(f"{kind} must not contain '..': {name!r}")
    if "\x00" in name or any(ord(c) < 0x20 for c in name):
        raise GitError(f"{kind} contains control characters: {name!r}")
    if len(name) > 255:
        raise GitError(f"{kind} exceeds 255 characters")
    if not _SAFE_REF_RE.fullmatch(name):
        raise GitError(
            f"{kind} contains disallowed characters: {name!r}. "
            "Only [A-Za-z0-9._/-] are permitted."
        )
    return name


def _validate_commit_ish(value: str) -> str:
    """Validate a commit hash or rev-spec. Accepts hex hashes and ref names."""
    if not isinstance(value, str) or not value:
        raise GitError("commit-ish must be a non-empty string")
    if value.startswith("-"):
        raise GitError(f"commit-ish must not start with '-': {value!r}")
    # Allow the same character set as refs plus ``@`` and ``{`` ``}`` ``~`` ``^``
    # used in rev-spec syntax (e.g. ``HEAD~2``, ``branch@{upstream}``).
    if not re.fullmatch(r"[A-Za-z0-9._/@{}~^-]+", value):
        raise GitError(f"commit-ish contains disallowed characters: {value!r}")
    if len(value) > 255:
        raise GitError("commit-ish exceeds 255 characters")
    return value


def _validate_relative_path(path: str) -> str:
    """Validate a repository-relative file path before passing it to git."""
    if not isinstance(path, str) or not path:
        raise GitError("file path must be a non-empty string")
    if path.startswith("-"):
        raise GitError(f"file path must not start with '-': {path!r}")
    if "\x00" in path:
        raise GitError("file path contains NUL byte")
    # Disallow absolute paths and parent references — staging must be repo-local.
    if path.startswith("/") or path.startswith("\\"):
        raise GitError(f"file path must be repo-relative, got absolute: {path!r}")
    if ".." in Path(path).parts:
        raise GitError(f"file path must not contain '..': {path!r}")
    return path


class GitError(Exception):
    """Base exception for Git-related errors."""
    
    def __init__(self, message: str, command: Optional[str] = None, stderr: Optional[str] = None) -> None:
        super().__init__(message)
        self.command = command
        self.stderr = stderr


class MergeConflictError(GitError):
    """Raised when a merge conflict cannot be auto-resolved."""
    
    def __init__(
        self,
        message: str,
        conflicted_files: List[str],
        branch_a: str,
        branch_b: str,
    ) -> None:
        super().__init__(message)
        self.conflicted_files = conflicted_files
        self.branch_a = branch_a
        self.branch_b = branch_b


class BranchNotFoundError(GitError):
    """Raised when a requested branch does not exist."""
    pass


class ValidationError(GitError):
    """Raised when validation fails before merge."""
    pass


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving merge conflicts."""
    
    OURS = auto()       # Prefer current branch changes
    THEIRS = auto()     # Prefer incoming branch changes
    UNION = auto()      # Combine both changes
    MANUAL = auto()     # Require manual intervention
    SMART = auto()      # Use intelligent conflict resolution


@dataclass(frozen=True)
class GitConfig:
    """Configuration for Git operations."""
    
    repo_path: Path
    user_name: str = "autoconstitution Bot"
    user_email: str = "bot@autoconstitution.ai"
    default_branch: str = "main"
    remote_name: str = "origin"
    gpg_sign: bool = False
    
    def __post_init__(self) -> None:
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {self.repo_path}")


@dataclass
class CommitInfo:
    """Information about a Git commit."""
    
    hash: str
    short_hash: str
    message: str
    author: str
    email: str
    timestamp: datetime
    files_changed: List[str] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)
    
    @classmethod
    def from_git_log(cls, log_line: str) -> CommitInfo:
        """Parse commit info from git log format string."""
        parts = log_line.split("|", 5)
        return cls(
            hash=parts[0],
            short_hash=parts[1],
            message=parts[2],
            author=parts[3],
            email=parts[4],
            timestamp=datetime.fromisoformat(parts[5]) if len(parts) > 5 else datetime.now(),
        )


@dataclass
class BranchInfo:
    """Information about a Git branch."""
    
    name: str
    commit_hash: str
    is_remote: bool
    is_current: bool
    ahead_count: int = 0
    behind_count: int = 0
    last_commit: Optional[CommitInfo] = None
    created_at: Optional[datetime] = None
    
    @property
    def is_local(self) -> bool:
        return not self.is_remote


@dataclass
class MergeConflict:
    """Represents a merge conflict in a file."""
    
    file_path: str
    ours_content: Optional[str] = None
    theirs_content: Optional[str] = None
    base_content: Optional[str] = None
    conflict_markers: List[tuple[int, int, str]] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return f"MergeConflict({self.file_path}, markers={len(self.conflict_markers)})"


@dataclass
class MergeResult:
    """Result of a merge operation."""
    
    success: bool
    branch_merged: str
    target_branch: str
    commit_hash: Optional[str] = None
    conflicts: List[MergeConflict] = field(default_factory=list)
    resolved_conflicts: List[MergeConflict] = field(default_factory=list)
    strategy_used: ConflictResolutionStrategy = ConflictResolutionStrategy.MANUAL
    message: str = ""


@dataclass
class PullRequest:
    """Represents a pull request for automated merging."""
    
    id: str
    title: str
    description: str
    source_branch: str
    target_branch: str
    author: str
    created_at: datetime
    status: PRStatus = field(default_factory=lambda: PRStatus.OPEN)
    labels: List[str] = field(default_factory=list)
    reviewers: List[str] = field(default_factory=list)
    validation_results: Dict[str, bool] = field(default_factory=dict)
    
    @property
    def is_validated(self) -> bool:
        return all(self.validation_results.values())


class PRStatus(Enum):
    """Status of a pull request."""
    
    OPEN = "open"
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    MERGED = "merged"
    CLOSED = "closed"
    CONFLICT = "conflict"


@dataclass
class AgentBranch:
    """Tracks a branch created for a specific agent."""
    
    agent_id: str
    branch_name: str
    base_branch: str
    created_at: datetime
    last_activity: datetime
    commits: List[CommitInfo] = field(default_factory=list)
    status: AgentBranchStatus = field(default_factory=lambda: AgentBranchStatus.ACTIVE)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_stale(self, timeout_hours: float = 24.0) -> bool:
        """Check if branch has been inactive."""
        from datetime import timedelta
        return (datetime.now() - self.last_activity) > timedelta(hours=timeout_hours)


class AgentBranchStatus(Enum):
    """Status of an agent branch."""
    
    ACTIVE = "active"
    MERGING = "merging"
    MERGED = "merged"
    CONFLICT = "conflict"
    STALE = "stale"
    ABANDONED = "abandoned"


# Type variables for generic protocols
T = TypeVar("T")
R = TypeVar("R")


class ConflictResolver(Protocol):
    """Protocol for conflict resolution strategies."""
    
    async def resolve(
        self,
        conflict: MergeConflict,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Attempt to resolve a conflict. Returns resolved content or None."""
        ...


class ValidationHook(Protocol):
    """Protocol for pre-merge validation hooks."""
    
    async def validate(
        self,
        branch: str,
        target: str,
        changes: List[str],
    ) -> tuple[bool, str]:
        """Validate changes before merge. Returns (passed, message)."""
        ...


class NotificationHook(Protocol):
    """Protocol for notification callbacks."""
    
    async def notify(self, event: str, data: Dict[str, Any]) -> None:
        """Send notification about an event."""
        ...


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (GitError,),
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    """Decorator to retry async operations on failure."""
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Optional[Exception] = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                        await asyncio.sleep(delay * (2 ** attempt))
            raise last_error or GitError("All retry attempts failed")
        return wrapper
    return decorator


class AsyncGitExecutor:
    """Async wrapper for Git command execution."""
    
    def __init__(self, config: GitConfig) -> None:
        self.config = config
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(5)  # Limit concurrent git operations
    
    async def execute(
        self,
        command: List[str],
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
        capture_output: bool = True,
    ) -> tuple[int, str, str]:
        """Execute a git command asynchronously.
        
        Args:
            command: Git command and arguments
            cwd: Working directory (defaults to repo path)
            env: Additional environment variables
            timeout: Command timeout in seconds
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            Tuple of (returncode, stdout, stderr)
        """
        async with self._semaphore:
            working_dir = cwd or self.config.repo_path
            
            # Prepare environment
            cmd_env = os.environ.copy()
            cmd_env["GIT_AUTHOR_NAME"] = self.config.user_name
            cmd_env["GIT_AUTHOR_EMAIL"] = self.config.user_email
            cmd_env["GIT_COMMITTER_NAME"] = self.config.user_name
            cmd_env["GIT_COMMITTER_EMAIL"] = self.config.user_email
            if env:
                cmd_env.update(env)
            
            full_command = ["git"] + command
            logger.debug(f"Executing: {' '.join(full_command)} in {working_dir}")
            
            try:
                proc = await asyncio.create_subprocess_exec(
                    *full_command,
                    cwd=working_dir,
                    env=cmd_env,
                    stdout=asyncio.subprocess.PIPE if capture_output else None,
                    stderr=asyncio.subprocess.PIPE if capture_output else None,
                )
                
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
                
                stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
                stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
                
                return proc.returncode or 0, stdout, stderr
                
            except asyncio.TimeoutError:
                if proc:
                    proc.kill()
                raise GitError(f"Command timed out after {timeout}s: {' '.join(full_command)}")
            except Exception as e:
                raise GitError(f"Failed to execute command: {e}")
    
    async def execute_or_raise(
        self,
        command: List[str],
        **kwargs: Any,
    ) -> str:
        """Execute command and raise on non-zero exit."""
        returncode, stdout, stderr = await self.execute(command, **kwargs)
        
        if returncode != 0:
            raise GitError(
                f"Git command failed: {' '.join(command)}",
                command=" ".join(command),
                stderr=stderr,
            )
        
        return stdout.strip()


class ConflictResolutionEngine:
    """Engine for intelligent conflict resolution."""
    
    def __init__(self) -> None:
        self._resolvers: Dict[ConflictResolutionStrategy, ConflictResolver] = {}
        self._setup_default_resolvers()
    
    def _setup_default_resolvers(self) -> None:
        """Register default conflict resolution strategies."""
        self._resolvers[ConflictResolutionStrategy.OURS] = OursResolver()
        self._resolvers[ConflictResolutionStrategy.THEIRS] = TheirsResolver()
        self._resolvers[ConflictResolutionStrategy.UNION] = UnionResolver()
        self._resolvers[ConflictResolutionStrategy.SMART] = SmartResolver()
    
    def register_resolver(
        self,
        strategy: ConflictResolutionStrategy,
        resolver: ConflictResolver,
    ) -> None:
        """Register a custom conflict resolver."""
        self._resolvers[strategy] = resolver
    
    async def resolve_conflicts(
        self,
        conflicts: List[MergeConflict],
        strategy: ConflictResolutionStrategy,
        context: Dict[str, Any],
    ) -> List[MergeConflict]:
        """Attempt to resolve multiple conflicts.
        
        Returns:
            List of conflicts that could not be resolved
        """
        if strategy == ConflictResolutionStrategy.MANUAL:
            return conflicts  # Don't auto-resolve
        
        resolver = self._resolvers.get(strategy)
        if not resolver:
            return conflicts
        
        unresolved: List[MergeConflict] = []
        
        for conflict in conflicts:
            try:
                resolved_content = await resolver.resolve(conflict, context)
                if resolved_content is not None:
                    # Write resolved content
                    repo_path = context.get("repo_path")
                    if repo_path:
                        file_path = Path(repo_path) / conflict.file_path
                        file_path.write_text(resolved_content, encoding="utf-8")
                        # Mark as resolved in git
                        git = context.get("git_executor")
                        if git:
                            await git.execute_or_raise(["add", conflict.file_path])
                else:
                    unresolved.append(conflict)
            except Exception as e:
                logger.error(f"Failed to resolve conflict in {conflict.file_path}: {e}")
                unresolved.append(conflict)
        
        return unresolved


class OursResolver:
    """Resolver that prefers current branch changes."""
    
    async def resolve(
        self,
        conflict: MergeConflict,
        context: Dict[str, Any],
    ) -> Optional[str]:
        return conflict.ours_content


class TheirsResolver:
    """Resolver that prefers incoming branch changes."""
    
    async def resolve(
        self,
        conflict: MergeConflict,
        context: Dict[str, Any],
    ) -> Optional[str]:
        return conflict.theirs_content


class UnionResolver:
    """Resolver that combines both changes."""
    
    async def resolve(
        self,
        conflict: MergeConflict,
        context: Dict[str, Any],
    ) -> Optional[str]:
        if conflict.ours_content and conflict.theirs_content:
            # Simple concatenation with separator
            return (
                conflict.ours_content.rstrip() +
                "\n\n# === Merged from both branches ===\n\n" +
                conflict.theirs_content.lstrip()
            )
        return conflict.ours_content or conflict.theirs_content


class SmartResolver:
    """Intelligent resolver that attempts semantic understanding."""
    
    async def resolve(
        self,
        conflict: MergeConflict,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Attempt smart resolution based on file type and content."""
        file_path = conflict.file_path
        
        # JSON files - try to merge objects
        if file_path.endswith(".json"):
            return await self._resolve_json(conflict)
        
        # Python files - try to merge imports and functions
        if file_path.endswith(".py"):
            return await self._resolve_python(conflict)
        
        # Config files - prefer non-empty values
        if file_path.endswith((".yaml", ".yml", ".toml", ".ini")):
            return await self._resolve_config(conflict)
        
        # Default: try union
        return await UnionResolver().resolve(conflict, context)
    
    async def _resolve_json(self, conflict: MergeConflict) -> Optional[str]:
        """Merge JSON files by combining keys."""
        try:
            import json
            
            ours = json.loads(conflict.ours_content or "{}")
            theirs = json.loads(conflict.theirs_content or "{}")
            
            if isinstance(ours, dict) and isinstance(theirs, dict):
                merged = {**theirs, **ours}  # Ours takes precedence
                return json.dumps(merged, indent=2)
            elif isinstance(ours, list) and isinstance(theirs, list):
                merged = list(dict.fromkeys(theirs + ours))  # Deduplicate
                return json.dumps(merged, indent=2)
        except Exception:
            pass
        return None
    
    async def _resolve_python(self, conflict: MergeConflict) -> Optional[str]:
        """Attempt to merge Python files by sections."""
        # This is a simplified implementation
        # A full implementation would use AST parsing
        ours = conflict.ours_content or ""
        theirs = conflict.theirs_content or ""
        
        # If one is empty, return the other
        if not ours.strip():
            return theirs
        if not theirs.strip():
            return ours
        
        return None  # Fall back to manual resolution
    
    async def _resolve_config(self, conflict: MergeConflict) -> Optional[str]:
        """Merge config files preferring non-default values."""
        # Prefer the more specific/complete configuration
        ours = conflict.ours_content or ""
        theirs = conflict.theirs_content or ""
        
        # Return the longer/more detailed one
        if len(ours) > len(theirs):
            return ours
        return theirs


class BranchManager:
    """
    Manages Git branches for autoconstitution multi-agent workflows.
    
    Handles:
    - Branch creation and lifecycle for individual agents
    - Async commit operations
    - Merge conflict detection and resolution
    - Automated PR creation for validated improvements
    """
    
    def __init__(
        self,
        config: GitConfig,
        conflict_engine: Optional[ConflictResolutionEngine] = None,
    ) -> None:
        self.config = config
        self.git = AsyncGitExecutor(config)
        self.conflict_engine = conflict_engine or ConflictResolutionEngine()
        self._agent_branches: Dict[str, AgentBranch] = {}
        self._validation_hooks: List[ValidationHook] = []
        self._notification_hooks: List[NotificationHook] = []
        self._pr_registry: Dict[str, PullRequest] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the branch manager."""
        if self._initialized:
            return
        
        # Verify we're in a git repository
        try:
            await self.git.execute_or_raise(["rev-parse", "--git-dir"])
        except GitError:
            raise GitError(f"Not a git repository: {self.config.repo_path}")
        
        # Load existing agent branches
        await self._load_agent_branches()
        
        self._initialized = True
        logger.info(f"BranchManager initialized for {self.config.repo_path}")
    
    @asynccontextmanager
    async def agent_session(self, agent_id: str, base_branch: Optional[str] = None):
        """Context manager for agent work session.
        
        Automatically creates branch on enter, handles cleanup on exit.
        
        Usage:
            async with branch_manager.agent_session("agent_001") as branch:
                # Work with branch
                await branch_manager.commit_changes(branch, "message", ["file.py"])
        """
        branch_name = await self.create_agent_branch(agent_id, base_branch)
        try:
            yield branch_name
        finally:
            # Cleanup logic if needed
            await self._notify("agent_session_closed", {
                "agent_id": agent_id,
                "branch": branch_name,
            })
    
    async def create_agent_branch(
        self,
        agent_id: str,
        base_branch: Optional[str] = None,
        branch_prefix: str = "agent",
    ) -> str:
        """Create a new branch for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            base_branch: Branch to base from (defaults to config.default_branch)
            branch_prefix: Prefix for branch name
            
        Returns:
            Name of the created branch
        """
        await self._ensure_initialized()

        # Validate inputs that will be interpolated into git commands.
        _validate_ref_component(agent_id, kind="agent_id")
        _validate_ref_component(branch_prefix, kind="branch_prefix")
        base = base_branch or self.config.default_branch
        _validate_ref_component(base, kind="base_branch")

        branch_name = f"{branch_prefix}/{agent_id}/{self._generate_timestamp()}"
        _validate_ref_component(branch_name, kind="branch_name")

        # Ensure base branch exists and is up to date
        await self._checkout_branch(base)
        await self._pull_latest(base)

        # Create new branch. ``--`` terminates option parsing so the name cannot
        # be interpreted as a flag even if validation somehow slipped.
        await self.git.execute_or_raise(["checkout", "-b", branch_name, "--"])
        
        # Track the branch
        agent_branch = AgentBranch(
            agent_id=agent_id,
            branch_name=branch_name,
            base_branch=base,
            created_at=datetime.now(),
            last_activity=datetime.now(),
        )
        self._agent_branches[branch_name] = agent_branch
        
        await self._notify("branch_created", {
            "agent_id": agent_id,
            "branch": branch_name,
            "base": base,
        })
        
        logger.info(f"Created branch {branch_name} for agent {agent_id}")
        return branch_name
    
    async def commit_changes(
        self,
        branch: str,
        message: str,
        files: Optional[List[str]] = None,
        stage_all: bool = False,
        amend: bool = False,
    ) -> CommitInfo:
        """Commit changes to a branch.
        
        Args:
            branch: Branch to commit to
            message: Commit message
            files: Specific files to commit (None for all staged)
            stage_all: Stage all changes before commit
            amend: Amend previous commit
            
        Returns:
            CommitInfo for the new commit
        """
        await self._ensure_initialized()
        _validate_ref_component(branch, kind="branch")

        # Ensure we're on the correct branch
        await self._checkout_branch(branch)

        # Stage files. Use ``--`` before every user-supplied path so a filename
        # like ``--help`` or ``-p`` cannot be mistaken for a git flag.
        if stage_all:
            await self.git.execute_or_raise(["add", "-A"])
        elif files:
            for file in files:
                _validate_relative_path(file)
                await self.git.execute_or_raise(["add", "--", file])
        
        # Build commit command
        commit_cmd = ["commit", "-m", message]
        if amend:
            commit_cmd.append("--amend")
        if self.config.gpg_sign:
            commit_cmd.append("-S")
        
        # Execute commit
        output = await self.git.execute_or_raise(commit_cmd)
        
        # Get commit hash
        commit_hash = await self.git.execute_or_raise(["rev-parse", "HEAD"])
        
        # Create commit info
        commit_info = CommitInfo(
            hash=commit_hash,
            short_hash=commit_hash[:7],
            message=message,
            author=self.config.user_name,
            email=self.config.user_email,
            timestamp=datetime.now(),
            files_changed=files or [],
        )
        
        # Update agent branch tracking
        if branch in self._agent_branches:
            self._agent_branches[branch].commits.append(commit_info)
            self._agent_branches[branch].last_activity = datetime.now()
        
        await self._notify("commit_created", {
            "branch": branch,
            "commit": commit_hash,
            "message": message,
        })
        
        logger.info(f"Created commit {commit_hash[:7]} on {branch}")
        return commit_info
    
    async def create_pull_request(
        self,
        source_branch: str,
        title: str,
        description: str = "",
        target_branch: Optional[str] = None,
        labels: Optional[List[str]] = None,
        auto_merge: bool = False,
    ) -> PullRequest:
        """Create a pull request for merging changes.
        
        Args:
            source_branch: Branch with changes to merge
            title: PR title
            description: PR description
            target_branch: Target branch (defaults to main)
            labels: Labels to apply
            auto_merge: Whether to auto-merge if validation passes
            
        Returns:
            Created PullRequest object
        """
        await self._ensure_initialized()
        
        target = target_branch or self.config.default_branch
        pr_id = self._generate_pr_id(source_branch, target)
        
        pr = PullRequest(
            id=pr_id,
            title=title,
            description=description,
            source_branch=source_branch,
            target_branch=target,
            author=self.config.user_name,
            created_at=datetime.now(),
            labels=labels or [],
        )
        
        self._pr_registry[pr_id] = pr
        
        await self._notify("pr_created", {
            "pr_id": pr_id,
            "source": source_branch,
            "target": target,
            "title": title,
        })
        
        logger.info(f"Created PR {pr_id}: {source_branch} -> {target}")
        
        # Run validation if auto-merge enabled
        if auto_merge:
            validation_passed = await self._run_validation(pr)
            if validation_passed:
                await self.merge_pr(pr_id)
        
        return pr
    
    async def merge_branch(
        self,
        source_branch: str,
        target_branch: Optional[str] = None,
        strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.SMART,
        squash: bool = False,
        message: Optional[str] = None,
    ) -> MergeResult:
        """Merge one branch into another.
        
        Args:
            source_branch: Branch to merge from
            target_branch: Branch to merge into (defaults to main)
            strategy: Conflict resolution strategy
            squash: Whether to squash commits
            message: Custom merge message
            
        Returns:
            MergeResult with details of the operation
        """
        await self._ensure_initialized()
        _validate_ref_component(source_branch, kind="source_branch")

        target = target_branch or self.config.default_branch
        _validate_ref_component(target, kind="target_branch")

        # Update agent branch status
        if source_branch in self._agent_branches:
            self._agent_branches[source_branch].status = AgentBranchStatus.MERGING

        try:
            # Checkout target branch
            await self._checkout_branch(target)

            # Pull latest changes
            await self._pull_latest(target)

            # Attempt merge
            merge_cmd = ["merge", source_branch]
            if squash:
                merge_cmd.append("--squash")
            if message:
                merge_cmd.extend(["-m", message])
            
            returncode, stdout, stderr = await self.git.execute(merge_cmd)
            
            if returncode == 0:
                # Merge succeeded
                commit_hash = await self.git.execute_or_raise(["rev-parse", "HEAD"])
                
                if source_branch in self._agent_branches:
                    self._agent_branches[source_branch].status = AgentBranchStatus.MERGED
                
                result = MergeResult(
                    success=True,
                    branch_merged=source_branch,
                    target_branch=target,
                    commit_hash=commit_hash,
                    strategy_used=strategy,
                    message=f"Successfully merged {source_branch} into {target}",
                )
                
                await self._notify("merge_succeeded", {
                    "source": source_branch,
                    "target": target,
                    "commit": commit_hash,
                })
                
                logger.info(f"Merged {source_branch} into {target}")
                return result
            
            # Check for conflicts
            if "conflict" in stderr.lower() or returncode != 0:
                conflicts = await self._detect_conflicts()
                
                if conflicts:
                    # Attempt auto-resolution
                    context = {
                        "repo_path": self.config.repo_path,
                        "git_executor": self.git,
                        "source_branch": source_branch,
                        "target_branch": target,
                    }
                    
                    unresolved = await self.conflict_engine.resolve_conflicts(
                        conflicts, strategy, context
                    )
                    
                    if not unresolved:
                        # All conflicts resolved, complete merge
                        await self.git.execute_or_raise(["commit", "-m", f"Merge {source_branch}"])
                        commit_hash = await self.git.execute_or_raise(["rev-parse", "HEAD"])
                        
                        if source_branch in self._agent_branches:
                            self._agent_branches[source_branch].status = AgentBranchStatus.MERGED
                        
                        result = MergeResult(
                            success=True,
                            branch_merged=source_branch,
                            target_branch=target,
                            commit_hash=commit_hash,
                            conflicts=conflicts,
                            resolved_conflicts=conflicts,
                            strategy_used=strategy,
                            message=f"Merged with auto-resolved conflicts",
                        )
                        
                        await self._notify("merge_succeeded", {
                            "source": source_branch,
                            "target": target,
                            "commit": commit_hash,
                            "auto_resolved": len(conflicts),
                        })
                        
                        return result
                    
                    # Could not auto-resolve all conflicts
                    if source_branch in self._agent_branches:
                        self._agent_branches[source_branch].status = AgentBranchStatus.CONFLICT
                    
                    # Abort the merge. No user input here.
                    await self.git.execute(["merge", "--abort"])
                    
                    raise MergeConflictError(
                        f"Merge conflicts in {len(unresolved)} files",
                        conflicted_files=[c.file_path for c in unresolved],
                        branch_a=source_branch,
                        branch_b=target,
                    )
            
            # Other merge failure
            raise GitError(f"Merge failed: {stderr}")
            
        except Exception as e:
            if source_branch in self._agent_branches:
                self._agent_branches[source_branch].status = AgentBranchStatus.CONFLICT
            raise
    
    async def merge_pr(self, pr_id: str) -> MergeResult:
        """Merge a pull request after validation.
        
        Args:
            pr_id: ID of the pull request to merge
            
        Returns:
            MergeResult with details of the operation
        """
        await self._ensure_initialized()
        
        if pr_id not in self._pr_registry:
            raise ValueError(f"Pull request not found: {pr_id}")
        
        pr = self._pr_registry[pr_id]
        
        # Run validation
        validation_passed = await self._run_validation(pr)
        if not validation_passed:
            pr.status = PRStatus.CLOSED
            raise ValidationError(f"Validation failed for PR {pr_id}")
        
        # Perform merge
        result = await self.merge_branch(
            source_branch=pr.source_branch,
            target_branch=pr.target_branch,
        )
        
        if result.success:
            pr.status = PRStatus.MERGED
        
        return result
    
    async def get_branch_info(self, branch_name: str) -> BranchInfo:
        """Get information about a branch.
        
        Args:
            branch_name: Name of the branch
            
        Returns:
            BranchInfo object
        """
        await self._ensure_initialized()
        
        # Check if branch exists
        _validate_ref_component(branch_name, kind="branch_name")
        try:
            commit_hash = await self.git.execute_or_raise(
                ["rev-parse", "--verify", branch_name]
            )
        except GitError:
            raise BranchNotFoundError(f"Branch not found: {branch_name}")
        
        # Get current branch
        current = await self.git.execute_or_raise(["branch", "--show-current"])
        
        # Get ahead/behind counts
        ahead, behind = 0, 0
        try:
            upstream = await self.git.execute(
                ["rev-parse", "--abbrev-ref", f"{branch_name}@{{upstream}}"]
            )
            if upstream[0] == 0:
                count_output = await self.git.execute_or_raise(
                    ["rev-list", "--left-right", "--count", f"{branch_name}...{upstream[1]}"]
                )
                counts = count_output.split()
                ahead, behind = int(counts[0]), int(counts[1])
        except GitError:
            pass
        
        return BranchInfo(
            name=branch_name,
            commit_hash=commit_hash,
            is_remote=branch_name.startswith(self.config.remote_name + "/"),
            is_current=branch_name == current,
            ahead_count=ahead,
            behind_count=behind,
        )
    
    async def list_branches(
        self,
        include_remote: bool = True,
        pattern: Optional[str] = None,
    ) -> List[BranchInfo]:
        """List all branches matching pattern.
        
        Args:
            include_remote: Include remote branches
            pattern: Optional pattern to filter branches
            
        Returns:
            List of BranchInfo objects
        """
        await self._ensure_initialized()
        
        cmd = ["branch", "-a", "--format=%(refname:short)|%(objectname:short)"]
        output = await self.git.execute_or_raise(cmd)
        
        branches: List[BranchInfo] = []
        for line in output.split("\n"):
            if not line:
                continue
            parts = line.split("|")
            name = parts[0]
            commit = parts[1] if len(parts) > 1 else ""
            
            # Filter by pattern
            if pattern and not re.search(pattern, name):
                continue
            
            # Skip remote branches if not included
            if not include_remote and name.startswith("remotes/"):
                continue
            
            info = await self.get_branch_info(name)
            branches.append(info)
        
        return branches
    
    async def get_agent_branches(
        self,
        agent_id: Optional[str] = None,
        status: Optional[AgentBranchStatus] = None,
    ) -> List[AgentBranch]:
        """Get tracked agent branches.
        
        Args:
            agent_id: Filter by agent ID
            status: Filter by status
            
        Returns:
            List of AgentBranch objects
        """
        branches = list(self._agent_branches.values())
        
        if agent_id:
            branches = [b for b in branches if b.agent_id == agent_id]
        
        if status:
            branches = [b for b in branches if b.status == status]
        
        return branches
    
    async def cleanup_stale_branches(
        self,
        max_age_hours: float = 24.0,
        dry_run: bool = True,
    ) -> List[str]:
        """Clean up stale agent branches.
        
        Args:
            max_age_hours: Maximum age before considering stale
            dry_run: If True, only report what would be deleted
            
        Returns:
            List of deleted branch names
        """
        await self._ensure_initialized()
        
        deleted: List[str] = []
        
        for branch_name, agent_branch in list(self._agent_branches.items()):
            if agent_branch.is_stale(max_age_hours):
                if not dry_run:
                    try:
                        # Defence in depth: revalidate before sending to git.
                        _validate_ref_component(branch_name, kind="branch_name")
                        # Delete local branch. ``--`` terminates options.
                        await self.git.execute(["branch", "-D", branch_name, "--"])
                        deleted.append(branch_name)
                        del self._agent_branches[branch_name]
                        
                        await self._notify("branch_deleted", {
                            "branch": branch_name,
                            "agent_id": agent_branch.agent_id,
                            "reason": "stale",
                        })
                    except GitError as e:
                        logger.error(f"Failed to delete branch {branch_name}: {e}")
                else:
                    deleted.append(branch_name)
        
        return deleted
    
    async def sync_with_remote(self, branch: Optional[str] = None) -> None:
        """Sync local branch with remote.
        
        Args:
            branch: Branch to sync (None for all)
        """
        await self._ensure_initialized()
        
        # Fetch all remotes
        await self.git.execute_or_raise(["fetch", "--all"])
        
        if branch:
            # Pull specific branch
            await self._checkout_branch(branch)
            await self._pull_latest(branch)
        
        await self._notify("sync_completed", {
            "branch": branch or "all",
        })
    
    def register_validation_hook(self, hook: ValidationHook) -> None:
        """Register a validation hook for pre-merge checks."""
        self._validation_hooks.append(hook)
    
    def register_notification_hook(self, hook: NotificationHook) -> None:
        """Register a notification hook for events."""
        self._notification_hooks.append(hook)
    
    async def get_diff(
        self,
        branch_a: str,
        branch_b: str,
        stat_only: bool = False,
    ) -> str:
        """Get diff between two branches.
        
        Args:
            branch_a: First branch
            branch_b: Second branch
            stat_only: Return only statistics
            
        Returns:
            Diff output
        """
        await self._ensure_initialized()
        _validate_ref_component(branch_a, kind="branch_a")
        _validate_ref_component(branch_b, kind="branch_b")

        cmd = ["diff"]
        if stat_only:
            cmd.append("--stat")
        cmd.extend([f"{branch_a}...{branch_b}"])

        return await self.git.execute_or_raise(cmd)
    
    async def cherry_pick(
        self,
        commit_hash: str,
        branch: Optional[str] = None,
    ) -> CommitInfo:
        """Cherry-pick a commit onto current or specified branch.
        
        Args:
            commit_hash: Hash of commit to cherry-pick
            branch: Target branch (None for current)
            
        Returns:
            CommitInfo for the new commit
        """
        await self._ensure_initialized()
        _validate_commit_ish(commit_hash)

        if branch:
            await self._checkout_branch(branch)

        await self.git.execute_or_raise(["cherry-pick", commit_hash])

        new_hash = await self.git.execute_or_raise(["rev-parse", "HEAD"])

        return CommitInfo(
            hash=new_hash,
            short_hash=new_hash[:7],
            message=f"Cherry-pick {commit_hash}",
            author=self.config.user_name,
            email=self.config.user_email,
            timestamp=datetime.now(),
        )
    
    async def revert_changes(
        self,
        commit_hash: str,
        branch: Optional[str] = None,
        create_new_commit: bool = True,
    ) -> Optional[CommitInfo]:
        """Revert a commit.
        
        Args:
            commit_hash: Hash of commit to revert
            branch: Target branch (None for current)
            create_new_commit: Whether to create a new commit
            
        Returns:
            CommitInfo if create_new_commit is True
        """
        await self._ensure_initialized()
        _validate_commit_ish(commit_hash)

        if branch:
            await self._checkout_branch(branch)

        cmd = ["revert"]
        if not create_new_commit:
            cmd.append("--no-commit")
        cmd.append(commit_hash)

        await self.git.execute_or_raise(cmd)
        
        if create_new_commit:
            new_hash = await self.git.execute_or_raise(["rev-parse", "HEAD"])
            return CommitInfo(
                hash=new_hash,
                short_hash=new_hash[:7],
                message=f"Revert {commit_hash}",
                author=self.config.user_name,
                email=self.config.user_email,
                timestamp=datetime.now(),
            )
        
        return None
    
    async def stash_changes(
        self,
        message: Optional[str] = None,
        include_untracked: bool = False,
    ) -> str:
        """Stash current changes.
        
        Args:
            message: Stash message
            include_untracked: Include untracked files
            
        Returns:
            Stash reference
        """
        await self._ensure_initialized()
        
        cmd = ["stash", "push"]
        if message:
            cmd.extend(["-m", message])
        if include_untracked:
            cmd.append("-u")
        
        await self.git.execute_or_raise(cmd)
        
        # Get latest stash ref
        stash_list = await self.git.execute_or_raise(["stash", "list", "-1"])
        return stash_list.split(":")[0] if stash_list else "stash@{0}"
    
    async def pop_stash(self, stash_ref: str = "stash@{0}") -> None:
        """Pop a stash.
        
        Args:
            stash_ref: Stash reference to pop
        """
        await self._ensure_initialized()
        await self.git.execute_or_raise(["stash", "pop", stash_ref])
    
    # Private helper methods
    
    async def _ensure_initialized(self) -> None:
        """Ensure the manager is initialized."""
        if not self._initialized:
            await self.initialize()
    
    async def _checkout_branch(self, branch: str) -> None:
        """Checkout a branch."""
        _validate_ref_component(branch, kind="branch")
        await self.git.execute_or_raise(["checkout", branch, "--"])

    async def _pull_latest(self, branch: str) -> None:
        """Pull latest changes for a branch."""
        _validate_ref_component(branch, kind="branch")
        _validate_ref_component(self.config.remote_name, kind="remote_name")
        try:
            await self.git.execute_or_raise(
                ["pull", self.config.remote_name, branch]
            )
        except GitError:
            # Branch might not have upstream, ignore
            pass
    
    async def _detect_conflicts(self) -> List[MergeConflict]:
        """Detect merge conflicts in the working directory."""
        # Get list of conflicted files
        output = await self.git.execute_or_raise(
            ["diff", "--name-only", "--diff-filter=U"]
        )
        
        conflicts: List[MergeConflict] = []
        
        for file_path in output.split("\n"):
            if not file_path:
                continue
            
            # Read conflicted file
            full_path = self.config.repo_path / file_path
            if full_path.exists():
                content = full_path.read_text(encoding="utf-8", errors="replace")
                
                # Parse conflict markers
                conflict_markers = self._parse_conflict_markers(content)
                
                # Extract ours/theirs content
                ours, theirs = self._extract_conflict_sides(content)
                
                conflicts.append(MergeConflict(
                    file_path=file_path,
                    ours_content=ours,
                    theirs_content=theirs,
                    conflict_markers=conflict_markers,
                ))
        
        return conflicts
    
    def _parse_conflict_markers(
        self,
        content: str,
    ) -> List[tuple[int, int, str]]:
        """Parse conflict markers from file content.
        
        Returns:
            List of (start_line, end_line, marker_type) tuples
        """
        markers: List[tuple[int, int, str]] = []
        lines = content.split("\n")
        
        start_idx = None
        for i, line in enumerate(lines):
            if line.startswith("<<<<<<<"):
                start_idx = i
            elif line.startswith(">>>>>>>") and start_idx is not None:
                markers.append((start_idx, i, "conflict"))
                start_idx = None
        
        return markers
    
    def _extract_conflict_sides(self, content: str) -> tuple[Optional[str], Optional[str]]:
        """Extract ours and theirs content from conflicted file."""
        ours_parts: List[str] = []
        theirs_parts: List[str] = []
        
        in_ours = False
        in_theirs = False
        
        for line in content.split("\n"):
            if line.startswith("<<<<<<<"):
                in_ours = True
                continue
            elif line.startswith("======="):
                in_ours = False
                in_theirs = True
                continue
            elif line.startswith(">>>>>>>"):
                in_theirs = False
                continue
            
            if in_ours:
                ours_parts.append(line)
            elif in_theirs:
                theirs_parts.append(line)
        
        ours = "\n".join(ours_parts) if ours_parts else None
        theirs = "\n".join(theirs_parts) if theirs_parts else None
        
        return ours, theirs
    
    async def _run_validation(self, pr: PullRequest) -> bool:
        """Run all validation hooks for a PR."""
        # Get changed files
        diff_output = await self.get_diff(pr.source_branch, pr.target_branch, stat_only=True)
        changed_files = [line.split()[0] for line in diff_output.split("\n") if line.strip()]
        
        results: Dict[str, bool] = {}
        
        for i, hook in enumerate(self._validation_hooks):
            try:
                passed, message = await hook.validate(
                    pr.source_branch,
                    pr.target_branch,
                    changed_files,
                )
                results[f"hook_{i}"] = passed
                
                if not passed:
                    logger.warning(f"Validation hook {i} failed: {message}")
            except Exception as e:
                logger.error(f"Validation hook {i} error: {e}")
                results[f"hook_{i}"] = False
        
        pr.validation_results = results
        return all(results.values())
    
    async def _notify(self, event: str, data: Dict[str, Any]) -> None:
        """Send notifications to all registered hooks."""
        for hook in self._notification_hooks:
            try:
                await hook.notify(event, data)
            except Exception as e:
                logger.error(f"Notification hook error: {e}")
    
    async def _load_agent_branches(self) -> None:
        """Load existing agent branches from git."""
        try:
            branches = await self.list_branches(pattern=r"agent/")
            for branch in branches:
                # Parse agent ID from branch name
                parts = branch.name.split("/")
                if len(parts) >= 2:
                    agent_id = parts[1]
                    if branch.name not in self._agent_branches:
                        self._agent_branches[branch.name] = AgentBranch(
                            agent_id=agent_id,
                            branch_name=branch.name,
                            base_branch=self.config.default_branch,
                            created_at=datetime.now(),
                            last_activity=datetime.now(),
                        )
        except GitError:
            pass
    
    def _generate_timestamp(self) -> str:
        """Generate timestamp string for branch names."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _generate_pr_id(self, source: str, target: str) -> str:
        """Generate unique PR ID."""
        unique = f"{source}:{target}:{datetime.now().isoformat()}"
        return hashlib.sha256(unique.encode()).hexdigest()[:12]


# Example validation hook implementations

class TestValidationHook:
    """Example validation hook that checks for test files."""
    
    async def validate(
        self,
        branch: str,
        target: str,
        changes: List[str],
    ) -> tuple[bool, str]:
        """Validate that changes include tests."""
        has_tests = any("test" in f.lower() for f in changes)
        has_source = any(f.endswith(".py") and "test" not in f.lower() for f in changes)
        
        if has_source and not has_tests:
            return False, "Source changes should include tests"
        
        return True, "Validation passed"


class LintValidationHook:
    """Example validation hook that runs linting."""
    
    def __init__(self, repo_path: Path) -> None:
        self.repo_path = repo_path
    
    async def validate(
        self,
        branch: str,
        target: str,
        changes: List[str],
    ) -> tuple[bool, str]:
        """Run linting on changed Python files."""
        py_files = [f for f in changes if f.endswith(".py")]

        if not py_files:
            return True, "No Python files to lint"

        # Validate paths and guard against filename-as-flag attacks by passing
        # ``--`` to ruff.
        try:
            for f in py_files:
                _validate_relative_path(f)
        except GitError as e:
            return False, f"Refused to lint suspicious filename: {e}"

        # Run ruff or flake8
        try:
            proc = await asyncio.create_subprocess_exec(
                "ruff",
                "check",
                "--",
                *py_files,
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                return False, f"Linting failed: {stdout.decode()}"
            
            return True, "Linting passed"
        except FileNotFoundError:
            return True, "Ruff not installed, skipping"


# Convenience factory functions

async def create_branch_manager(
    repo_path: Union[str, Path],
    user_name: str = "autoconstitution Bot",
    user_email: str = "bot@autoconstitution.ai",
    **kwargs: Any,
) -> BranchManager:
    """Factory function to create and initialize a BranchManager.
    
    Args:
        repo_path: Path to the git repository
        user_name: Git user name for commits
        user_email: Git user email for commits
        **kwargs: Additional config options
        
    Returns:
        Initialized BranchManager instance
    """
    config = GitConfig(
        repo_path=Path(repo_path),
        user_name=user_name,
        user_email=user_email,
        **kwargs,
    )
    
    manager = BranchManager(config)
    await manager.initialize()
    return manager


# Example usage and testing

async def example_usage() -> None:
    """Example usage of BranchManager."""
    # Create manager
    manager = await create_branch_manager(
        repo_path="/path/to/repo",
        user_name="Research Agent",
        user_email="agent@autoconstitution.ai",
    )
    
    # Register validation hooks
    manager.register_validation_hook(TestValidationHook())
    
    # Create agent branch and work
    async with manager.agent_session("agent_001") as branch:
        # Make changes...
        
        # Commit
        commit = await manager.commit_changes(
            branch=branch,
            message="Implement feature X",
            stage_all=True,
        )
        
        # Create PR
        pr = await manager.create_pull_request(
            source_branch=branch,
            title="Feature X Implementation",
            description="Implements feature X with tests",
            auto_merge=True,
        )
        
        print(f"Created PR: {pr.id}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
