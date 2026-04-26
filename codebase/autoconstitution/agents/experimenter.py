"""
ExperimenterAgent for autoconstitution.

This module implements the ExperimenterAgent which is responsible for:
- Taking proposed code changes
- Applying them to train.py
- Running timed experiments
- Capturing metrics
- Reporting results back to the orchestrator

The ExperimenterAgent inherits from ResearchAgent and provides a complete
async implementation with full type hints.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar, Union
from uuid import UUID, uuid4

# Type variables for generic types
T = TypeVar("T")


class ExperimentStatus(Enum):
    """Status of an experiment execution."""
    PENDING = auto()
    SETUP = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    CANCELLED = auto()


class MetricType(Enum):
    """Types of metrics that can be captured."""
    SCALAR = "scalar"
    SERIES = "series"
    HISTOGRAM = "histogram"
    TEXT = "text"


@dataclass
class CodeChange:
    """Represents a proposed code change."""
    change_id: str
    description: str
    file_path: str
    original_code: Optional[str]
    new_code: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate the code change after initialization."""
        if not self.change_id:
            raise ValueError("change_id cannot be empty")
        if not self.file_path:
            raise ValueError("file_path cannot be empty")
        if not self.new_code:
            raise ValueError("new_code cannot be empty")


@dataclass
class MetricValue:
    """A single metric value with metadata."""
    name: str
    value: Union[float, int, str, List[float]]
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    step: Optional[int] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentMetrics:
    """Collection of metrics captured during an experiment."""
    experiment_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    metrics: List[MetricValue] = field(default_factory=list)
    
    def add_scalar(self, name: str, value: Union[float, int], step: Optional[int] = None, tags: Optional[Dict[str, str]] = None) -> None:
        """Add a scalar metric."""
        self.metrics.append(MetricValue(
            name=name,
            value=value,
            metric_type=MetricType.SCALAR,
            step=step,
            tags=tags or {}
        ))
    
    def add_series(self, name: str, values: List[float], step: Optional[int] = None, tags: Optional[Dict[str, str]] = None) -> None:
        """Add a series metric."""
        self.metrics.append(MetricValue(
            name=name,
            value=values,
            metric_type=MetricType.SERIES,
            step=step,
            tags=tags or {}
        ))
    
    def add_text(self, name: str, value: str, step: Optional[int] = None, tags: Optional[Dict[str, str]] = None) -> None:
        """Add a text metric."""
        self.metrics.append(MetricValue(
            name=name,
            value=value,
            metric_type=MetricType.TEXT,
            step=step,
            tags=tags or {}
        ))
    
    def get_scalar(self, name: str) -> Optional[Union[float, int]]:
        """Get the latest scalar metric by name."""
        for metric in reversed(self.metrics):
            if metric.name == name and metric.metric_type == MetricType.SCALAR:
                return metric.value  # type: ignore
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "type": m.metric_type.value,
                    "timestamp": m.timestamp.isoformat(),
                    "step": m.step,
                    "tags": m.tags
                }
                for m in self.metrics
            ]
        }


@dataclass
class ExperimentResult:
    """Result of an experiment execution."""
    experiment_id: str
    change_id: str
    status: ExperimentStatus
    metrics: ExperimentMetrics
    duration_seconds: float
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    error_message: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if the experiment was successful."""
        return self.status == ExperimentStatus.COMPLETED and self.exit_code == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "experiment_id": self.experiment_id,
            "change_id": self.change_id,
            "status": self.status.name,
            "success": self.success,
            "metrics": self.metrics.to_dict(),
            "duration_seconds": self.duration_seconds,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error_message": self.error_message,
            "artifacts": self.artifacts
        }


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    experiment_id: Optional[str] = None
    timeout_seconds: float = 3600.0
    working_directory: Optional[str] = None
    train_script_path: str = "train.py"
    python_executable: str = "python"
    capture_stdout: bool = True
    capture_stderr: bool = True
    environment_variables: Dict[str, str] = field(default_factory=dict)
    command_line_args: List[str] = field(default_factory=list)
    metric_patterns: Dict[str, str] = field(default_factory=dict)
    artifact_paths: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Generate experiment ID if not provided."""
        if self.experiment_id is None:
            self.experiment_id = f"exp_{uuid4().hex[:12]}"


class AgentStatus(Enum):
    """Lifecycle states for agents."""
    PENDING = auto()
    INITIALIZING = auto()
    ACTIVE = auto()
    PAUSED = auto()
    REALLOCATING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TERMINATED = auto()


@dataclass
class AgentConfig:
    """Configuration for spawning an agent."""
    agent_type: str
    specialization: str
    depth: int
    parent_id: Optional[UUID] = None
    hypotheses: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    timeout_seconds: int = 300


class ResearchAgent(ABC):
    """Base class for research agents."""
    
    def __init__(
        self,
        agent_id: UUID,
        config: AgentConfig,
        orchestrator: Optional[Any] = None
    ):
        self.agent_id = agent_id
        self.config = config
        self.orchestrator = orchestrator
        self.status = AgentStatus.PENDING
        
        # State
        self.current_hypothesis: Optional[str] = None
        self.iteration_count = 0
        self.findings: List[Dict[str, Any]] = []
        
        # Communication
        self.message_inbox: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.result_outbox: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
    
    async def initialize(self) -> None:
        """Prepare agent for research."""
        pass
    
    async def research_step(self) -> Dict[str, Any]:
        """Execute one research iteration."""
        return {}
    
    async def handle_message(self, message: Dict[str, Any]) -> None:
        """Process incoming messages from orchestrator or peers."""
        pass
    
    async def synthesize_findings(self) -> Dict[str, Any]:
        """Compile findings into structured output."""
        return {}
    
    async def run(self) -> None:
        """Main agent execution loop."""
        self.status = AgentStatus.ACTIVE
        await self.initialize()
        
        while self.status == AgentStatus.ACTIVE:
            try:
                # Check for messages
                if not self.message_inbox.empty():
                    msg = await asyncio.wait_for(
                        self.message_inbox.get(), 
                        timeout=0.1
                    )
                    await self.handle_message(msg)
                
                # Execute research step
                result = await self.research_step()
                self.iteration_count += 1
                
                # Report results
                await self.result_outbox.put({
                    "agent_id": str(self.agent_id),
                    "iteration": self.iteration_count,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Check for pause/termination
                if self.status != AgentStatus.ACTIVE:
                    break
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.status = AgentStatus.FAILED
                if self.orchestrator:
                    await self._report_error(str(e))
    
    async def pause(self) -> None:
        """Pause agent execution."""
        if self.status == AgentStatus.ACTIVE:
            self.status = AgentStatus.PAUSED
    
    async def resume(self) -> None:
        """Resume agent execution."""
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.ACTIVE
    
    async def terminate(self) -> None:
        """Gracefully terminate agent."""
        self.status = AgentStatus.TERMINATED
    
    async def _report_error(self, error: str) -> None:
        """Report error to orchestrator."""
        if self.orchestrator and hasattr(self.orchestrator, "event_queue"):
            await self.orchestrator.event_queue.put({
                "type": "agent_failed",
                "agent_id": str(self.agent_id),
                "error": error
            })


class ExperimenterAgent(ResearchAgent):
    """
    Agent responsible for running timed training experiments.
    
    The ExperimenterAgent takes proposed code changes, applies them to train.py,
    runs the training experiment with timing, captures metrics, and reports
    results back to the orchestrator.
    
    Attributes:
        agent_id: Unique identifier for this agent
        config: Agent configuration
        orchestrator: Reference to the orchestrator
        experiment_history: List of completed experiment results
    
    Example:
        >>> agent = ExperimenterAgent(
        ...     agent_id=uuid4(),
        ...     config=AgentConfig(agent_type="experimenter", specialization="training", depth=0)
        ... )
        >>> change = CodeChange(
        ...     change_id="change_001",
        ...     description="Increase learning rate",
        ...     file_path="train.py",
        ...     original_code=None,
        ...     new_code="learning_rate = 0.01"
        ... )
        >>> result = await agent.run_experiment(change)
        >>> print(result.success)
    """
    
    def __init__(
        self,
        agent_id: UUID,
        config: AgentConfig,
        orchestrator: Optional[Any] = None,
        default_experiment_config: Optional[ExperimentConfig] = None
    ):
        """
        Initialize the ExperimenterAgent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration
            orchestrator: Reference to the orchestrator
            default_experiment_config: Default configuration for experiments
        """
        super().__init__(agent_id, config, orchestrator)
        self.default_config = default_experiment_config or ExperimentConfig()
        self.experiment_history: List[ExperimentResult] = []
        self._current_experiment: Optional[ExperimentConfig] = None
        self._experiment_task: Optional[asyncio.Task[ExperimentResult]] = None
        self._temp_dir: Optional[str] = None
        
        # Default metric patterns for parsing training output
        self.default_metric_patterns: Dict[str, str] = {
            "loss": r"loss[:\s]+([\d.]+)",
            "accuracy": r"accuracy[:\s]+([\d.]+)",
            "epoch": r"epoch[:\s]+(\d+)",
            "learning_rate": r"lr[:\s]+([\d.e-]+)",
            "val_loss": r"val_loss[:\s]+([\d.]+)",
            "val_accuracy": r"val_accuracy[:\s]+([\d.]+)",
        }
    
    async def initialize(self) -> None:
        """Initialize the experimenter agent."""
        self.status = AgentStatus.INITIALIZING
        # Create temporary working directory if needed
        if self.default_config.working_directory is None:
            self._temp_dir = tempfile.mkdtemp(prefix="experimenter_")
            self.default_config.working_directory = self._temp_dir
        self.status = AgentStatus.ACTIVE
    
    async def research_step(self) -> Dict[str, Any]:
        """Execute one research iteration - not used for experimenter."""
        # Experimenter doesn't use the standard research loop
        return {"status": "idle", "experiments_completed": len(self.experiment_history)}
    
    async def run_experiment(
        self, 
        code_change: CodeChange,
        config: Optional[ExperimentConfig] = None
    ) -> ExperimentResult:
        """
        Run a complete experiment with the given code change.
        
        This is the main entry point for running experiments. It:
        1. Sets up the experiment environment
        2. Applies the code change to train.py
        3. Runs the training script with timing
        4. Captures metrics from output
        5. Cleans up and returns results
        
        Args:
            code_change: The code change to apply
            config: Optional experiment configuration (uses default if not provided)
        
        Returns:
            ExperimentResult containing all experiment data
        """
        experiment_config = config or self._create_config_from_default()
        self._current_experiment = experiment_config
        
        start_time = time.time()
        metrics = ExperimentMetrics(experiment_id=experiment_config.experiment_id or str(uuid4()),
                                     start_time=datetime.now())
        
        try:
            # Setup phase
            await self._update_status(ExperimentStatus.SETUP)
            work_dir = await self._setup_experiment_environment(code_change)
            
            # Apply the code change
            train_path = await self._apply_code_change(work_dir, code_change)
            
            # Run phase
            await self._update_status(ExperimentStatus.RUNNING)
            exit_code, stdout, stderr = await self._run_training_script(
                work_dir, train_path, experiment_config
            )
            
            # Capture metrics
            await self._capture_metrics(stdout, stderr, metrics)
            
            # Collect artifacts
            artifacts = await self._collect_artifacts(work_dir, experiment_config)
            
            # Determine status
            status = ExperimentStatus.COMPLETED if exit_code == 0 else ExperimentStatus.FAILED
            
            duration = time.time() - start_time
            metrics.end_time = datetime.now()
            
            result = ExperimentResult(
                experiment_id=experiment_config.experiment_id or str(uuid4()),
                change_id=code_change.change_id,
                status=status,
                metrics=metrics,
                duration_seconds=duration,
                exit_code=exit_code,
                stdout=stdout if experiment_config.capture_stdout else "",
                stderr=stderr if experiment_config.capture_stderr else "",
                artifacts=artifacts
            )
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            metrics.end_time = datetime.now()
            result = ExperimentResult(
                experiment_id=experiment_config.experiment_id or str(uuid4()),
                change_id=code_change.change_id,
                status=ExperimentStatus.TIMEOUT,
                metrics=metrics,
                duration_seconds=duration,
                error_message=f"Experiment timed out after {experiment_config.timeout_seconds} seconds"
            )
        except Exception as e:
            duration = time.time() - start_time
            metrics.end_time = datetime.now()
            result = ExperimentResult(
                experiment_id=experiment_config.experiment_id or str(uuid4()),
                change_id=code_change.change_id,
                status=ExperimentStatus.FAILED,
                metrics=metrics,
                duration_seconds=duration,
                error_message=str(e)
            )
        finally:
            # Cleanup
            await self._cleanup_experiment(work_dir)
        
        # Store in history
        self.experiment_history.append(result)
        
        # Report to orchestrator
        await self._report_result_to_orchestrator(result)
        
        return result
    
    async def run_experiment_async(
        self, 
        code_change: CodeChange,
        config: Optional[ExperimentConfig] = None
        ) -> asyncio.Task[ExperimentResult]:
        """
        Start an experiment asynchronously and return the task.
        
        Args:
            code_change: The code change to apply
            config: Optional experiment configuration
        
        Returns:
            asyncio.Task that will complete with the ExperimentResult
        """
        self._experiment_task = asyncio.create_task(
            self.run_experiment(code_change, config)
        )
        return self._experiment_task
    
    async def cancel_current_experiment(self) -> bool:
        """
        Cancel the currently running experiment.
        
        Returns:
            True if an experiment was cancelled, False otherwise
        """
        if self._experiment_task and not self._experiment_task.done():
            self._experiment_task.cancel()
            try:
                await self._experiment_task
            except asyncio.CancelledError:
                pass
            return True
        return False
    
    async def _setup_experiment_environment(self, code_change: CodeChange) -> str:
        """
        Set up the experiment working directory.
        
        Args:
            code_change: The code change (used to determine source)
        
        Returns:
            Path to the working directory
        """
        work_dir = self._current_experiment.working_directory if self._current_experiment else None
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix=f"exp_{code_change.change_id}_")
        else:
            work_dir = os.path.join(work_dir, f"exp_{code_change.change_id}")
            os.makedirs(work_dir, exist_ok=True)
        
        # Copy train.py to working directory if it exists
        source_train = self._find_source_train_py()
        if source_train and os.path.exists(source_train):
            shutil.copy2(source_train, os.path.join(work_dir, "train.py"))
        else:
            # Create a minimal train.py
            await self._create_minimal_train_py(work_dir)
        
        return work_dir
    
    async def _apply_code_change(self, work_dir: str, code_change: CodeChange) -> str:
        """
        Apply a code change to train.py in the working directory.
        
        Args:
            work_dir: The working directory
            code_change: The code change to apply
        
        Returns:
            Path to the modified train.py
        """
        train_path = os.path.join(work_dir, "train.py")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"train.py not found in {work_dir}")
        
        # Read the original file
        original_content = await self._read_file_async(train_path)
        
        # Apply the change
        if code_change.original_code is not None:
            # Replace specific code
            if code_change.original_code not in original_content:
                raise ValueError(f"Original code not found in train.py: {code_change.original_code[:100]}...")
            new_content = original_content.replace(code_change.original_code, code_change.new_code)
        elif code_change.line_start is not None and code_change.line_end is not None:
            # Replace by line numbers
            lines = original_content.split("\n")
            new_lines = lines[:code_change.line_start - 1]
            new_lines.extend(code_change.new_code.split("\n"))
            new_lines.extend(lines[code_change.line_end:])
            new_content = "\n".join(new_lines)
        else:
            # Append the new code
            new_content = original_content + "\n" + code_change.new_code
        
        # Write the modified file
        await self._write_file_async(train_path, new_content)
        
        return train_path
    
    async def _run_training_script(
        self, 
        work_dir: str, 
        train_path: str,
        config: ExperimentConfig
    ) -> tuple[int, str, str]:
        """
        Run the training script with timeout.
        
        Args:
            work_dir: Working directory
            train_path: Path to train.py
            config: Experiment configuration
        
        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        cmd = [
            config.python_executable,
            train_path
        ] + config.command_line_args
        
        env = os.environ.copy()
        env.update(config.environment_variables)
        
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE if config.capture_stdout else None,
            stderr=asyncio.subprocess.PIPE if config.capture_stderr else None,
            cwd=work_dir,
            env=env
        )
        
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_seconds
            )
            
            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
            
            return process.returncode or 0, stdout, stderr
            
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise
    
    async def _capture_metrics(
        self, 
        stdout: str, 
        stderr: str,
        metrics: ExperimentMetrics
    ) -> None:
        """
        Capture metrics from training output.
        
        Args:
            stdout: Standard output from training
            stderr: Standard error from training
            metrics: Metrics object to populate
        """
        combined_output = stdout + "\n" + stderr
        
        # Use default patterns + any custom patterns from config
        patterns = self.default_metric_patterns.copy()
        if self._current_experiment and self._current_experiment.metric_patterns:
            patterns.update(self._current_experiment.metric_patterns)
        
        # Parse metrics from output
        for metric_name, pattern in patterns.items():
            matches = re.findall(pattern, combined_output, re.IGNORECASE)
            if matches:
                try:
                    # Try to convert to float
                    values = [float(m) if isinstance(m, str) else float(m[0]) if isinstance(m, tuple) else float(m) for m in matches]
                    
                    if len(values) == 1:
                        metrics.add_scalar(metric_name, values[0])
                    else:
                        metrics.add_series(metric_name, values)
                except (ValueError, IndexError):
                    # Store as text if conversion fails
                    metrics.add_text(metric_name, str(matches[-1]))
        
        # Add basic metrics
        metrics.add_scalar("stdout_lines", len(stdout.split("\n")))
        metrics.add_scalar("stderr_lines", len(stderr.split("\n")))
        metrics.add_text("last_100_chars_stdout", stdout[-100:] if len(stdout) > 100 else stdout)
    
    async def _collect_artifacts(
        self, 
        work_dir: str, 
        config: ExperimentConfig
    ) -> Dict[str, str]:
        """
        Collect artifact files from the experiment.
        
        Args:
            work_dir: Working directory
            config: Experiment configuration
        
        Returns:
            Dictionary mapping artifact names to file paths
        """
        artifacts: Dict[str, str] = {}
        
        for artifact_pattern in config.artifact_paths:
            import glob
            matches = glob.glob(os.path.join(work_dir, artifact_pattern))
            for match in matches:
                artifact_name = os.path.basename(match)
                artifacts[artifact_name] = match
        
        return artifacts
    
    async def _cleanup_experiment(self, work_dir: str) -> None:
        """
        Clean up the experiment working directory.
        
        Args:
            work_dir: Working directory to clean up
        """
        if self._temp_dir and work_dir.startswith(self._temp_dir):
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass
    
    async def _update_status(self, status: ExperimentStatus) -> None:
        """Update the current experiment status."""
        # Could emit events here
        pass
    
    async def _report_result_to_orchestrator(self, result: ExperimentResult) -> None:
        """
        Report experiment results back to the orchestrator.
        
        Args:
            result: The experiment result to report
        """
        if self.orchestrator is None:
            return
        
        report = {
            "type": "experiment_completed",
            "agent_id": str(self.agent_id),
            "agent_type": "experimenter",
            "timestamp": datetime.now().isoformat(),
            "payload": result.to_dict()
        }
        
        # Send to orchestrator's event queue if available
        if hasattr(self.orchestrator, "event_queue"):
            await self.orchestrator.event_queue.put(report)
        
        # Also put in our result outbox
        await self.result_outbox.put(report)
    
    def _create_config_from_default(self) -> ExperimentConfig:
        """Create a new experiment config from the default."""
        return ExperimentConfig(
            experiment_id=None,
            timeout_seconds=self.default_config.timeout_seconds,
            working_directory=self.default_config.working_directory,
            train_script_path=self.default_config.train_script_path,
            python_executable=self.default_config.python_executable,
            capture_stdout=self.default_config.capture_stdout,
            capture_stderr=self.default_config.capture_stderr,
            environment_variables=self.default_config.environment_variables.copy(),
            command_line_args=self.default_config.command_line_args.copy(),
            metric_patterns=self.default_config.metric_patterns.copy(),
            artifact_paths=self.default_config.artifact_paths.copy()
        )
    
    def _find_source_train_py(self) -> Optional[str]:
        """Find the source train.py file."""
        # Check common locations
        candidates = [
            "train.py",
            "./train.py",
            "../train.py",
            "../../train.py",
            "./src/train.py",
        ]
        
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        
        return None
    
    async def _create_minimal_train_py(self, work_dir: str) -> None:
        """Create a minimal train.py for testing purposes."""
        minimal_train = '''#!/usr/bin/env python
"""Minimal training script for testing."""
import time
import random
import sys

def train():
    """Simulate training."""
    print("Starting training...")
    
    epochs = 5
    for epoch in range(epochs):
        loss = 1.0 / (epoch + 1) + random.random() * 0.1
        accuracy = 0.5 + (epoch / epochs) * 0.4 + random.random() * 0.05
        
        print(f"epoch: {epoch + 1}")
        print(f"loss: {loss:.4f}")
        print(f"accuracy: {accuracy:.4f}")
        print(f"lr: {0.01 * (0.9 ** epoch):.6f}")
        
        time.sleep(0.1)
    
    print("Training completed!")
    return 0

if __name__ == "__main__":
    sys.exit(train())
'''
        await self._write_file_async(os.path.join(work_dir, "train.py"), minimal_train)
    
    async def _read_file_async(self, path: str) -> str:
        """Read a file asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._read_file_sync, path)
    
    def _read_file_sync(self, path: str) -> str:
        """Read a file synchronously."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    
    async def _write_file_async(self, path: str, content: str) -> None:
        """Write a file asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_file_sync, path, content)
    
    def _write_file_sync(self, path: str, content: str) -> None:
        """Write a file synchronously."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    
    # Public utility methods
    
    def get_experiment_history(self) -> List[ExperimentResult]:
        """Get the history of all completed experiments."""
        return self.experiment_history.copy()
    
    def get_successful_experiments(self) -> List[ExperimentResult]:
        """Get only successful experiments."""
        return [exp for exp in self.experiment_history if exp.success]
    
    def get_experiment_by_id(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get a specific experiment by ID."""
        for exp in self.experiment_history:
            if exp.experiment_id == experiment_id:
                return exp
        return None
    
    def compare_experiments(
        self, 
        exp_id1: str, 
        exp_id2: str,
        metric_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Compare two experiments on a specific metric.
        
        Args:
            exp_id1: First experiment ID
            exp_id2: Second experiment ID
            metric_name: Name of the metric to compare
        
        Returns:
            Comparison dictionary or None if experiments not found
        """
        exp1 = self.get_experiment_by_id(exp_id1)
        exp2 = self.get_experiment_by_id(exp_id2)
        
        if not exp1 or not exp2:
            return None
        
        val1 = exp1.metrics.get_scalar(metric_name)
        val2 = exp2.metrics.get_scalar(metric_name)
        
        if val1 is None or val2 is None:
            return None
        
        return {
            "metric": metric_name,
            "experiment_1": {"id": exp_id1, "value": val1},
            "experiment_2": {"id": exp_id2, "value": val2},
            "difference": val2 - val1,
            "relative_change": (val2 - val1) / val1 if val1 != 0 else float("inf"),
            "better": exp_id2 if val2 < val1 else exp_id1 if metric_name == "loss" else exp_id1 if val2 < val1 else exp_id2
        }
    
    async def batch_run_experiments(
        self, 
        code_changes: List[CodeChange],
        config: Optional[ExperimentConfig] = None,
        max_concurrent: int = 3
    ) -> List[ExperimentResult]:
        """
        Run multiple experiments in parallel.
        
        Args:
            code_changes: List of code changes to test
            config: Optional experiment configuration
            max_concurrent: Maximum number of concurrent experiments
        
        Returns:
            List of experiment results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(change: CodeChange) -> ExperimentResult:
            async with semaphore:
                return await self.run_experiment(change, config)
        
        tasks = [run_with_semaphore(change) for change in code_changes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results: List[ExperimentResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create a failed result
                failed_result = ExperimentResult(
                    experiment_id=str(uuid4()),
                    change_id=code_changes[i].change_id,
                    status=ExperimentStatus.FAILED,
                    metrics=ExperimentMetrics(experiment_id=str(uuid4()), start_time=datetime.now()),
                    duration_seconds=0.0,
                    error_message=str(result)
                )
                valid_results.append(failed_result)
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def terminate(self) -> None:
        """Gracefully terminate the agent and clean up resources."""
        # Cancel any running experiment
        await self.cancel_current_experiment()
        
        # Clean up temp directory
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception:
                pass
        
        await super().terminate()


# Factory function for creating experimenter agents
async def create_experimenter_agent(
    orchestrator: Optional[Any] = None,
    default_config: Optional[ExperimentConfig] = None,
    specialization: str = "training"
) -> ExperimenterAgent:
    """
    Factory function to create an ExperimenterAgent.
    
    Args:
        orchestrator: Optional orchestrator reference
        default_config: Optional default experiment configuration
        specialization: Agent specialization
    
    Returns:
        Configured ExperimenterAgent instance
    """
    agent_id = uuid4()
    config = AgentConfig(
        agent_type="experimenter",
        specialization=specialization,
        depth=0
    )
    
    agent = ExperimenterAgent(
        agent_id=agent_id,
        config=config,
        orchestrator=orchestrator,
        default_experiment_config=default_config
    )
    
    await agent.initialize()
    return agent


# Export all public classes and functions
__all__ = [
    # Enums
    "ExperimentStatus",
    "MetricType",
    "AgentStatus",
    
    # Data classes
    "CodeChange",
    "MetricValue",
    "ExperimentMetrics",
    "ExperimentResult",
    "ExperimentConfig",
    "AgentConfig",
    
    # Base classes
    "ResearchAgent",
    
    # Main agent
    "ExperimenterAgent",
    
    # Factory
    "create_experimenter_agent",
]
