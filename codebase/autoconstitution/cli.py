"""
autoconstitution CLI.

This module exposes both the legacy experiment-management commands and the
product-facing Constitutional AI loop commands. Public-facing help should frame
the tool as a multi-agent improvement loop rather than a generic swarm runner.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from typing_extensions import Annotated

# Initialize console for rich output
console = Console()

# Create the main Typer app
app = typer.Typer(
    name="autoconstitution",
    help=(
        "autoconstitution CLI - multi-agent autoresearch for critique, revision, "
        "and keep-or-revert improvement loops"
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=True,
)

# Legacy experiment-orchestrator commands live behind `autoconstitution legacy ...`.
# The product CAI loop is the primary surface (`autoconstitution cai run`,
# `autoconstitution demo`); these commands predate that split and stick around
# for existing users / scripts. Each prints a one-line deprecation notice on use.
legacy_app = typer.Typer(
    name="legacy",
    help=(
        "Legacy experiment-orchestrator commands. Prefer `autoconstitution cai run` "
        "for new work; these are kept for backwards compatibility."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(legacy_app, name="legacy")


def _legacy_notice(name: str) -> None:
    """One-line deprecation notice printed at the top of every legacy command."""
    console.print(
        f"[yellow]`autoconstitution legacy {name}` is a legacy command. "
        f"For new work, use `autoconstitution cai run` or `autoconstitution demo`.[/yellow]"
    )


# ============================================================================
# Configuration Classes
# ============================================================================

class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ExperimentType(str, Enum):
    """Type of swarm experiment."""
    CONSENSUS = "consensus"
    OPTIMIZATION = "optimization"
    EXPLORATION = "exploration"
    HYBRID = "hybrid"


@dataclass
class SwarmConfig:
    """Configuration for swarm experiments."""
    num_agents: int = 10
    max_iterations: int = 100
    convergence_threshold: float = 0.001
    communication_range: float = 10.0
    learning_rate: float = 0.01
    random_seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "num_agents": self.num_agents,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "communication_range": self.communication_range,
            "learning_rate": self.learning_rate,
            "random_seed": self.random_seed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwarmConfig":
        """Create config from dictionary."""
        return cls(
            num_agents=data.get("num_agents", 10),
            max_iterations=data.get("max_iterations", 100),
            convergence_threshold=data.get("convergence_threshold", 0.001),
            communication_range=data.get("communication_range", 10.0),
            learning_rate=data.get("learning_rate", 0.01),
            random_seed=data.get("random_seed"),
        )


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "default_experiment"
    experiment_type: ExperimentType = ExperimentType.CONSENSUS
    log_level: LogLevel = LogLevel.INFO
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    checkpoint_interval: int = 10
    save_visualizations: bool = True
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "experiment_type": self.experiment_type.value,
            "log_level": self.log_level.value,
            "output_dir": str(self.output_dir),
            "checkpoint_interval": self.checkpoint_interval,
            "save_visualizations": self.save_visualizations,
            "swarm": self.swarm.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(
            name=data.get("name", "default_experiment"),
            experiment_type=ExperimentType(data.get("experiment_type", "consensus")),
            log_level=LogLevel(data.get("log_level", "info")),
            output_dir=Path(data.get("output_dir", "./outputs")),
            checkpoint_interval=data.get("checkpoint_interval", 10),
            save_visualizations=data.get("save_visualizations", True),
            swarm=SwarmConfig.from_dict(data.get("swarm", {})),
        )


# ============================================================================
# Configuration File Support
# ============================================================================

DEFAULT_CONFIG_PATHS = [
    Path("autoconstitution.yaml"),
    Path("autoconstitution.yml"),
    Path(".autoconstitution.yaml"),
    Path(".autoconstitution.yml"),
    Path.home() / ".config" / "autoconstitution" / "config.yaml",
]


def load_config(config_path: Optional[Path] = None) -> ExperimentConfig:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config file. If None, searches default locations.
    
    Returns:
        Loaded experiment configuration.
    
    Raises:
        FileNotFoundError: If config file not found.
        ValueError: If config file is invalid.
    """
    if config_path is None:
        # Search default locations
        for path in DEFAULT_CONFIG_PATHS:
            if path.exists():
                config_path = path
                break
        else:
            # Return default config if no file found
            return ExperimentConfig()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    content = config_path.read_text()
    
    if config_path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(content)
    elif config_path.suffix == ".json":
        data = json.loads(content)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return ExperimentConfig.from_dict(data)


def save_config(config: ExperimentConfig, path: Path) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save.
        path: Path to save to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix in (".yaml", ".yml"):
        content = yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False)
    elif path.suffix == ".json":
        content = json.dumps(config.to_dict(), indent=2)
    else:
        path = path.with_suffix(".yaml")
        content = yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False)
    
    path.write_text(content)
    console.print(f"[green]Configuration saved to {path}[/green]")


# ============================================================================
# Experiment State Management
# ============================================================================

@dataclass
class ExperimentState:
    """State of a running experiment."""
    experiment_id: str
    name: str
    status: str  # "running", "paused", "completed", "failed"
    current_iteration: int
    total_iterations: int
    start_time: float
    last_checkpoint: Optional[float] = None
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status,
            "current_iteration": self.current_iteration,
            "total_iterations": self.total_iterations,
            "start_time": self.start_time,
            "last_checkpoint": self.last_checkpoint,
            "metrics": self.metrics,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentState":
        """Create state from dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            status=data["status"],
            current_iteration=data["current_iteration"],
            total_iterations=data["total_iterations"],
            start_time=data["start_time"],
            last_checkpoint=data.get("last_checkpoint"),
            metrics=data.get("metrics", {}),
        )


def get_state_file_path(experiment_id: str, output_dir: Path) -> Path:
    """Get path to experiment state file."""
    return output_dir / f"{experiment_id}_state.json"


def save_experiment_state(state: ExperimentState, output_dir: Path) -> None:
    """Save experiment state to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = get_state_file_path(state.experiment_id, output_dir)
    state_path.write_text(json.dumps(state.to_dict(), indent=2))


def load_experiment_state(experiment_id: str, output_dir: Path) -> Optional[ExperimentState]:
    """Load experiment state from file."""
    state_path = get_state_file_path(experiment_id, output_dir)
    if not state_path.exists():
        return None
    data = json.loads(state_path.read_text())
    return ExperimentState.from_dict(data)


def list_experiments(output_dir: Path) -> List[ExperimentState]:
    """List all experiments in output directory."""
    experiments = []
    if not output_dir.exists():
        return experiments
    
    for state_file in output_dir.glob("*_state.json"):
        try:
            data = json.loads(state_file.read_text())
            experiments.append(ExperimentState.from_dict(data))
        except (json.JSONDecodeError, KeyError):
            continue
    
    return sorted(experiments, key=lambda e: e.start_time, reverse=True)


# ============================================================================
# Progress Bar Utilities
# ============================================================================

def create_progress_bar() -> Progress:
    """Create a rich progress bar with custom styling."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


# ============================================================================
# Commands
# ============================================================================

@legacy_app.command()
def run(
    name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help="Name of the experiment")
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to configuration file")
    ] = None,
    experiment_type: Annotated[
        Optional[ExperimentType],
        typer.Option("--type", "-t", help="Type of experiment")
    ] = None,
    agents: Annotated[
        Optional[int],
        typer.Option("--agents", "-a", help="Number of agents")
    ] = None,
    iterations: Annotated[
        Optional[int],
        typer.Option("--iterations", "-i", help="Maximum iterations")
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory")
    ] = None,
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", "-s", help="Random seed for reproducibility")
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show configuration without running")
    ] = False,
) -> None:
    """
    Legacy orchestrator experiment runner.
    
    Examples:
        autoconstitution run
        autoconstitution run --name "consensus_test" --agents 50 --iterations 200
        autoconstitution run --config custom_config.yaml
    """
    # Load configuration
    try:
        exp_config = load_config(config)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    
    # Override with CLI arguments
    if name:
        exp_config.name = name
    if experiment_type:
        exp_config.experiment_type = experiment_type
    if agents is not None:
        exp_config.swarm.num_agents = agents
    if iterations is not None:
        exp_config.swarm.max_iterations = iterations
    if output_dir:
        exp_config.output_dir = output_dir
    if seed is not None:
        exp_config.swarm.random_seed = seed
    
    # Generate experiment ID
    import uuid
    experiment_id = f"{exp_config.name}_{uuid.uuid4().hex[:8]}"
    
    # Display configuration
    config_table = Table(title="Experiment Configuration", show_header=False)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("Experiment ID", experiment_id)
    config_table.add_row("Name", exp_config.name)
    config_table.add_row("Type", exp_config.experiment_type.value)
    config_table.add_row("Agents", str(exp_config.swarm.num_agents))
    config_table.add_row("Max Iterations", str(exp_config.swarm.max_iterations))
    config_table.add_row("Output Directory", str(exp_config.output_dir))
    if exp_config.swarm.random_seed:
        config_table.add_row("Random Seed", str(exp_config.swarm.random_seed))
    
    console.print(config_table)
    
    if dry_run:
        console.print("\n[yellow]Dry run - not executing experiment[/yellow]")
        return
    
    # Create output directory
    exp_config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = exp_config.output_dir / f"{experiment_id}_config.yaml"
    save_config(exp_config, config_path)
    
    # Initialize experiment state
    state = ExperimentState(
        experiment_id=experiment_id,
        name=exp_config.name,
        status="running",
        current_iteration=0,
        total_iterations=exp_config.swarm.max_iterations,
        start_time=time.time(),
        metrics={"convergence": [], "energy": [], "communication": []},
    )
    
    # Run experiment with progress bar
    console.print(f"\n[bold]Starting experiment: {exp_config.name}[/bold]\n")
    
    with create_progress_bar() as progress:
        task = progress.add_task(
            f"[cyan]Running {exp_config.experiment_type.value} experiment...",
            total=exp_config.swarm.max_iterations,
        )
        
        try:
            for iteration in range(exp_config.swarm.max_iterations):
                # Simulate experiment work
                time.sleep(0.05)  # Replace with actual computation
                
                # Update metrics (simulated)
                convergence = 1.0 / (1.0 + iteration * 0.1)
                state.metrics["convergence"].append(convergence)
                state.metrics["energy"].append(100.0 - iteration * 0.5)
                state.metrics["communication"].append(iteration * 2.5)
                
                # Update state
                state.current_iteration = iteration + 1
                
                # Save checkpoint periodically
                if (iteration + 1) % exp_config.checkpoint_interval == 0:
                    state.last_checkpoint = time.time()
                    save_experiment_state(state, exp_config.output_dir)
                
                # Update progress
                progress.update(task, advance=1)
                
                # Check convergence
                if convergence < exp_config.swarm.convergence_threshold:
                    console.print(f"\n[green]Convergence reached at iteration {iteration + 1}[/green]")
                    break
            
            # Mark as completed
            state.status = "completed"
            save_experiment_state(state, exp_config.output_dir)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Experiment interrupted by user[/yellow]")
            state.status = "paused"
            save_experiment_state(state, exp_config.output_dir)
            raise typer.Exit(130)
    
    # Display results
    elapsed = time.time() - state.start_time
    console.print(f"\n[bold green]Experiment completed![/bold green]")
    console.print(f"Total iterations: {state.current_iteration}")
    console.print(f"Elapsed time: {elapsed:.2f}s")
    console.print(f"Final convergence: {state.metrics['convergence'][-1]:.6f}")
    console.print(f"\nResults saved to: {exp_config.output_dir}")


@legacy_app.command()
def resume(
    experiment_id: Annotated[
        str,
        typer.Argument(help="ID of the experiment to resume")
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory")
    ] = None,
    additional_iterations: Annotated[
        Optional[int],
        typer.Option("--iterations", "-i", help="Additional iterations to run")
    ] = None,
) -> None:
    """
    Resume a legacy orchestrator experiment.
    
    Examples:
        autoconstitution resume consensus_test_a1b2c3d4
        autoconstitution resume consensus_test_a1b2c3d4 --iterations 50
    """
    output_dir = output_dir or Path("./outputs")
    
    # Load experiment state
    state = load_experiment_state(experiment_id, output_dir)
    
    if state is None:
        console.print(f"[red]Error: Experiment '{experiment_id}' not found[/red]")
        raise typer.Exit(1)
    
    if state.status == "completed":
        console.print(f"[yellow]Warning: Experiment '{experiment_id}' is already completed[/yellow]")
        if not typer.confirm("Do you want to continue anyway?"):
            raise typer.Exit(0)
    
    # Load configuration
    config_path = output_dir / f"{experiment_id}_config.yaml"
    if config_path.exists():
        exp_config = load_config(config_path)
    else:
        console.print(f"[yellow]Warning: Configuration not found, using defaults[/yellow]")
        exp_config = ExperimentConfig()
    
    # Update total iterations if specified
    if additional_iterations:
        state.total_iterations = state.current_iteration + additional_iterations
    
    console.print(f"[bold]Resuming experiment: {state.name}[/bold]")
    console.print(f"Current iteration: {state.current_iteration}")
    console.print(f"Target iterations: {state.total_iterations}\n")
    
    # Resume experiment
    state.status = "running"
    
    with create_progress_bar() as progress:
        task = progress.add_task(
            f"[cyan]Resuming {state.name}...",
            total=state.total_iterations,
            completed=state.current_iteration,
        )
        
        try:
            for iteration in range(state.current_iteration, state.total_iterations):
                # Simulate experiment work
                time.sleep(0.05)
                
                # Update metrics
                convergence = 1.0 / (1.0 + iteration * 0.1)
                state.metrics["convergence"].append(convergence)
                state.metrics["energy"].append(100.0 - iteration * 0.5)
                state.metrics["communication"].append(iteration * 2.5)
                
                state.current_iteration = iteration + 1
                
                # Save checkpoint
                if (iteration + 1) % exp_config.checkpoint_interval == 0:
                    state.last_checkpoint = time.time()
                    save_experiment_state(state, output_dir)
                
                progress.update(task, advance=1)
                
                if convergence < exp_config.swarm.convergence_threshold:
                    console.print(f"\n[green]Convergence reached at iteration {iteration + 1}[/green]")
                    break
            
            state.status = "completed"
            save_experiment_state(state, output_dir)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Experiment interrupted by user[/yellow]")
            state.status = "paused"
            save_experiment_state(state, output_dir)
            raise typer.Exit(130)
    
    console.print(f"\n[bold green]Experiment completed![/bold green]")
    console.print(f"Total iterations: {state.current_iteration}")


@legacy_app.command()
def status(
    experiment_id: Annotated[
        Optional[str],
        typer.Argument(help="ID of specific experiment (optional)")
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory")
    ] = None,
    watch: Annotated[
        bool,
        typer.Option("--watch", "-w", help="Watch mode - continuously update")
    ] = False,
    interval: Annotated[
        int,
        typer.Option("--interval", help="Update interval in seconds (watch mode)")
    ] = 2,
) -> None:
    """
    Check the status of legacy orchestrator experiments.
    
    Examples:
        autoconstitution status
        autoconstitution status consensus_test_a1b2c3d4
        autoconstitution status --watch
    """
    output_dir = output_dir or Path("./outputs")
    
    def display_status():
        console.clear()
        
        if experiment_id:
            # Show specific experiment
            state = load_experiment_state(experiment_id, output_dir)
            if state is None:
                console.print(f"[red]Error: Experiment '{experiment_id}' not found[/red]")
                return False
            
            # Status color based on state
            status_colors = {
                "running": "yellow",
                "paused": "blue",
                "completed": "green",
                "failed": "red",
            }
            color = status_colors.get(state.status, "white")
            
            # Create status panel
            status_text = Text()
            status_text.append(f"Experiment ID: ", style="cyan")
            status_text.append(f"{state.experiment_id}\n", style="white")
            status_text.append(f"Name: ", style="cyan")
            status_text.append(f"{state.name}\n", style="white")
            status_text.append(f"Status: ", style="cyan")
            status_text.append(f"{state.status.upper()}\n", style=f"bold {color}")
            status_text.append(f"Progress: ", style="cyan")
            status_text.append(f"{state.current_iteration}/{state.total_iterations}\n", style="white")
            
            elapsed = time.time() - state.start_time
            status_text.append(f"Elapsed Time: ", style="cyan")
            status_text.append(f"{elapsed:.2f}s\n", style="white")
            
            if state.last_checkpoint:
                status_text.append(f"Last Checkpoint: ", style="cyan")
                status_text.append(f"{time.time() - state.last_checkpoint:.2f}s ago\n", style="white")
            
            console.print(Panel(status_text, title="Experiment Status", border_style=color))
            
            # Show metrics if available
            if state.metrics:
                metrics_table = Table(title="Metrics")
                metrics_table.add_column("Metric", style="cyan")
                metrics_table.add_column("Current", style="green")
                metrics_table.add_column("Min", style="blue")
                metrics_table.add_column("Max", style="magenta")
                
                for metric_name, values in state.metrics.items():
                    if values:
                        metrics_table.add_row(
                            metric_name,
                            f"{values[-1]:.4f}",
                            f"{min(values):.4f}",
                            f"{max(values):.4f}",
                        )
                
                console.print(metrics_table)
        
        else:
            # List all experiments
            experiments = list_experiments(output_dir)
            
            if not experiments:
                console.print("[yellow]No experiments found[/yellow]")
                return False
            
            table = Table(title="All Experiments")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Name", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Progress", style="blue")
            table.add_column("Elapsed", style="magenta")
            
            status_colors = {
                "running": "[yellow]",
                "paused": "[blue]",
                "completed": "[green]",
                "failed": "[red]",
            }
            
            for exp in experiments:
                color = status_colors.get(exp.status, "[white]")
                elapsed = time.time() - exp.start_time
                progress_pct = (exp.current_iteration / exp.total_iterations * 100) if exp.total_iterations > 0 else 0
                
                table.add_row(
                    exp.experiment_id,
                    exp.name,
                    f"{color}{exp.status.upper()}[/]",
                    f"{exp.current_iteration}/{exp.total_iterations} ({progress_pct:.1f}%)",
                    f"{elapsed:.1f}s",
                )
            
            console.print(table)
        
        return True
    
    if watch:
        try:
            while True:
                if not display_status():
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[dim]Watch mode stopped[/dim]")
    else:
        display_status()


@legacy_app.command()
def benchmark(
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to benchmark configuration")
    ] = None,
    agents: Annotated[
        List[int],
        typer.Option("--agents", "-a", help="Number of agents to benchmark")
    ] = [10, 50, 100],
    iterations: Annotated[
        int,
        typer.Option("--iterations", "-i", help="Iterations per benchmark")
    ] = 100,
    runs: Annotated[
        int,
        typer.Option("--runs", "-r", help="Number of runs per configuration")
    ] = 3,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file for results")
    ] = None,
) -> None:
    """
    Run legacy orchestrator benchmarks.
    
    Examples:
        autoconstitution benchmark
        autoconstitution benchmark --agents 10 50 100 500 --iterations 200
        autoconstitution benchmark --runs 5 --output benchmark_results.json
    """
    console.print("[bold]autoconstitution Benchmark Suite[/bold]\n")
    
    # Results storage
    results: Dict[str, List[Dict[str, Any]]] = {}
    
    # Create results table
    results_table = Table(title="Benchmark Results")
    results_table.add_column("Agents", style="cyan", justify="right")
    results_table.add_column("Run", style="blue", justify="right")
    results_table.add_column("Iterations", style="green", justify="right")
    results_table.add_column("Time (s)", style="magenta", justify="right")
    results_table.add_column("Throughput (it/s)", style="yellow", justify="right")
    
    total_benchmarks = len(agents) * runs
    current_benchmark = 0
    
    with create_progress_bar() as progress:
        overall_task = progress.add_task(
            "[bold cyan]Running benchmarks...",
            total=total_benchmarks,
        )
        
        for num_agents in agents:
            agent_results = []
            
            for run in range(1, runs + 1):
                current_benchmark += 1
                
                # Create benchmark task
                benchmark_task = progress.add_task(
                    f"[cyan]Agents={num_agents}, Run={run}/{runs}",
                    total=iterations,
                    visible=False,
                )
                
                start_time = time.time()
                
                # Simulate benchmark run
                for i in range(iterations):
                    time.sleep(0.01 + num_agents * 0.0001)  # Simulate work
                    if i % 10 == 0:
                        progress.update(benchmark_task, advance=10)
                
                elapsed = time.time() - start_time
                throughput = iterations / elapsed
                
                # Store result
                result = {
                    "agents": num_agents,
                    "run": run,
                    "iterations": iterations,
                    "time": elapsed,
                    "throughput": throughput,
                }
                agent_results.append(result)
                
                # Update table
                results_table.add_row(
                    str(num_agents),
                    str(run),
                    str(iterations),
                    f"{elapsed:.3f}",
                    f"{throughput:.2f}",
                )
                
                progress.update(overall_task, advance=1)
                progress.remove_task(benchmark_task)
            
            results[f"agents_{num_agents}"] = agent_results
    
    # Display results
    console.print("\n")
    console.print(results_table)
    
    # Summary statistics
    console.print("\n[bold]Summary Statistics:[/bold]")
    
    summary_table = Table()
    summary_table.add_column("Agents", style="cyan")
    summary_table.add_column("Avg Time (s)", style="magenta")
    summary_table.add_column("Avg Throughput (it/s)", style="yellow")
    summary_table.add_column("Speedup", style="green")
    
    baseline_throughput = None
    
    for num_agents in agents:
        key = f"agents_{num_agents}"
        agent_results = results[key]
        
        avg_time = sum(r["time"] for r in agent_results) / len(agent_results)
        avg_throughput = sum(r["throughput"] for r in agent_results) / len(agent_results)
        
        if baseline_throughput is None:
            baseline_throughput = avg_throughput
            speedup = 1.0
        else:
            speedup = avg_throughput / baseline_throughput
        
        summary_table.add_row(
            str(num_agents),
            f"{avg_time:.3f}",
            f"{avg_throughput:.2f}",
            f"{speedup:.2f}x",
        )
    
    console.print(summary_table)
    
    # Save results if output specified
    if output:
        output_data = {
            "timestamp": time.time(),
            "config": {
                "agents": agents,
                "iterations": iterations,
                "runs": runs,
            },
            "results": results,
        }
        output.write_text(json.dumps(output_data, indent=2))
        console.print(f"\n[green]Results saved to: {output}[/green]")


# ============================================================================
# Additional Utility Commands
# ============================================================================

@app.command(name="config")
def config_command(
    action: Annotated[
        str,
        typer.Argument(help="Action: init, show, validate")
    ] = "show",
    path: Annotated[
        Optional[Path],
        typer.Option("--path", "-p", help="Path to configuration file")
    ] = None,
) -> None:
    """
    Manage autoconstitution configuration.
    
    Examples:
        autoconstitution config init
        autoconstitution config show
        autoconstitution config validate --path config.yaml
    """
    if action == "init":
        # Create default configuration
        config_path = path or Path("autoconstitution.yaml")
        default_config = ExperimentConfig()
        save_config(default_config, config_path)
        console.print(f"[green]Created default configuration at {config_path}[/green]")
    
    elif action == "show":
        # Show current configuration
        try:
            config = load_config(path)
            console.print(yaml.dump(config.to_dict(), default_flow_style=False))
        except FileNotFoundError:
            console.print("[yellow]No configuration file found. Using defaults:[/yellow]")
            default_config = ExperimentConfig()
            console.print(yaml.dump(default_config.to_dict(), default_flow_style=False))
    
    elif action == "validate":
        # Validate configuration file
        if not path:
            console.print("[red]Error: --path is required for validation[/red]")
            raise typer.Exit(1)
        
        try:
            config = load_config(path)
            console.print(f"[green]Configuration is valid: {path}[/green]")
            
            # Show validation details
            validation_table = Table(title="Configuration Validation")
            validation_table.add_column("Parameter", style="cyan")
            validation_table.add_column("Value", style="green")
            validation_table.add_column("Status", style="blue")
            
            validation_table.add_row("Name", config.name, "✓ Valid")
            validation_table.add_row("Type", config.experiment_type.value, "✓ Valid")
            validation_table.add_row("Agents", str(config.swarm.num_agents), 
                "✓ Valid" if config.swarm.num_agents > 0 else "✗ Invalid")
            validation_table.add_row("Iterations", str(config.swarm.max_iterations),
                "✓ Valid" if config.swarm.max_iterations > 0 else "✗ Invalid")
            
            console.print(validation_table)
            
        except Exception as e:
            console.print(f"[red]Configuration validation failed: {e}[/red]")
            raise typer.Exit(1)
    
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available actions: init, show, validate")
        raise typer.Exit(1)


@app.command()
def clean(
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory to clean")
    ] = None,
    keep_recent: Annotated[
        int,
        typer.Option("--keep-recent", help="Number of recent experiments to keep")
    ] = 0,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation")
    ] = False,
) -> None:
    """
    Clean up legacy experiment files.
    
    Examples:
        autoconstitution clean
        autoconstitution clean --keep-recent 5
        autoconstitution clean --force
    """
    output_dir = output_dir or Path("./outputs")
    
    if not output_dir.exists():
        console.print("[yellow]Output directory does not exist[/yellow]")
        return
    
    # List all experiment files
    state_files = list(output_dir.glob("*_state.json"))
    config_files = list(output_dir.glob("*_config.yaml"))
    
    total_files = len(state_files) + len(config_files)
    
    if total_files == 0:
        console.print("[yellow]No experiment files to clean[/yellow]")
        return
    
    # Sort by modification time
    all_files = sorted(
        state_files + config_files,
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    
    # Determine files to remove
    files_to_keep = all_files[:keep_recent * 2]  # 2 files per experiment
    files_to_remove = all_files[keep_recent * 2:]
    
    if not files_to_remove:
        console.print(f"[green]Nothing to clean (keeping {keep_recent} most recent)[/green]")
        return
    
    # Show what will be removed
    console.print(f"Files to remove: {len(files_to_remove)}")
    for f in files_to_remove[:10]:
        console.print(f"  - {f.name}")
    if len(files_to_remove) > 10:
        console.print(f"  ... and {len(files_to_remove) - 10} more")
    
    # Confirm deletion
    if not force:
        if not typer.confirm("\nDo you want to proceed?"):
            console.print("[dim]Clean cancelled[/dim]")
            return
    
    # Remove files
    removed_count = 0
    for f in files_to_remove:
        try:
            f.unlink()
            removed_count += 1
        except OSError as e:
            console.print(f"[red]Failed to remove {f}: {e}[/red]")
    
    console.print(f"[green]Removed {removed_count} files[/green]")


# ============================================================================
# CAI Subcommand — real critique/revision loop (not a simulation)
# ============================================================================

cai_app = typer.Typer(
    name="cai",
    help=(
        "Constitutional multi-agent loop: agents propose, critique, revise, and "
        "judge under an editable constitution."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(cai_app, name="cai")


@cai_app.command("run")
def cai_run(
    prompt: Annotated[
        Optional[str],
        typer.Option("--prompt", "-p", help="Single prompt to run the CAI loop on."),
    ] = None,
    prompts_file: Annotated[
        Optional[Path],
        typer.Option("--prompts-file", "-f", help="JSONL/TXT file, one prompt per line."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Where to write the preference-pair JSONL."),
    ] = Path("outputs/preference_pairs.jsonl"),
    max_rounds: Annotated[
        int,
        typer.Option("--max-rounds", help="Max Student/Judge rounds per prompt."),
    ] = 3,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", help="Parallel prompts when batching."),
    ] = 4,
    provider_name: Annotated[
        Optional[str],
        typer.Option("--provider", help="Force a provider (ollama/kimi/anthropic/openai)."),
    ] = None,
    constitution: Annotated[
        Optional[Path],
        typer.Option("--constitution", help="Path to a custom constitution.md."),
    ] = None,
    ui: Annotated[
        str,
        typer.Option(
            "--ui",
            help=(
                "Rendering mode: 'live' (Rich dashboard with role panels), "
                "'plain' (one line per event, pipe-friendly), 'json' (JSONL "
                "events on stdout for programmatic consumers), or 'auto' "
                "(live when running a single prompt in a TTY, plain otherwise)."
            ),
        ),
    ] = "auto",
    live: Annotated[
        Optional[bool],
        typer.Option(
            "--live/--no-live",
            hidden=True,
            help="Deprecated: use --ui=live / --ui=plain instead.",
        ),
    ] = None,
) -> None:
    """Run the Constitutional AI critique-revision loop for real.

    Requires a working provider. Ollama is tried first (no key needed).
    """
    import asyncio as _asyncio
    import sys as _sys

    from autoconstitution.cai import (
        CritiqueRevisionLoop,
        JudgeAgent,
        StudentAgent,
    )
    from autoconstitution.cai.preference_pairs import PreferencePairBuilder
    from autoconstitution.providers import pick_provider

    # ---- Resolve UI mode -----------------------------------------------
    ui_normalized = ui.lower().strip()
    if ui_normalized not in ("live", "plain", "json", "auto"):
        console.print(
            f"[red]Invalid --ui value: {ui!r}. "
            f"Pick one of live/plain/json/auto.[/red]"
        )
        raise typer.Exit(code=2)
    # Back-compat: --live/--no-live overrides --ui when explicitly set.
    if live is True:
        ui_normalized = "live"
    elif live is False:
        ui_normalized = "plain"

    # ---- Collect prompts ------------------------------------------------
    prompts: List[str] = []
    if prompt:
        prompts.append(prompt)
    if prompts_file:
        if not prompts_file.exists():
            console.print(f"[red]prompts file not found: {prompts_file}[/red]")
            raise typer.Exit(code=1)
        for line in prompts_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            # Accept JSONL with {"prompt": "..."} or plain lines.
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    prompts.append(obj.get("prompt", line))
                    continue
                except json.JSONDecodeError:
                    pass
            prompts.append(line)

    if not prompts:
        console.print("[red]No prompts given. Use --prompt or --prompts-file.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Loaded {len(prompts)} prompt(s)[/cyan]")

    # ---- Spin up the loop ----------------------------------------------
    async def _go() -> None:
        prefer = [provider_name] if provider_name else None
        choice = await pick_provider(prefer=prefer)
        console.print(
            f"[green]Using provider: {choice.name} (model={choice.model})[/green] "
            f"[dim]— {choice.reason}[/dim]"
        )

        student = StudentAgent(provider=choice.provider)
        judge = JudgeAgent(provider=choice.provider, constitution_path=constitution)
        loop = CritiqueRevisionLoop(student=student, judge=judge, max_rounds=max_rounds)

        # Resolve 'auto' against the actual runtime (TTY, batch vs single).
        effective_ui = ui_normalized
        if effective_ui == "auto":
            effective_ui = (
                "live"
                if len(prompts) == 1 and _sys.stdout.isatty()
                else "plain"
            )
        # 'live' only makes sense for a single prompt — downgrade with a notice.
        if effective_ui == "live" and len(prompts) != 1:
            console.print(
                "[yellow]--ui=live ignored for batch runs; "
                "falling back to plain.[/yellow]"
            )
            effective_ui = "plain"

        if effective_ui == "live":
            from autoconstitution.ui.live import LiveRenderer

            renderer = LiveRenderer(console=console, max_rounds=max_rounds)
            try:
                result = await loop.run(prompts[0], renderer=renderer)
                results = [result]
            finally:
                await renderer.aclose()
        elif effective_ui == "json":
            from autoconstitution.ui.json_stream import JSONRenderer

            json_renderer = JSONRenderer()
            try:
                results = []
                for p in prompts:
                    results.append(await loop.run(p, renderer=json_renderer))
            finally:
                await json_renderer.aclose()
        else:  # plain
            from autoconstitution.ui.plain import PlainRenderer

            plain_renderer = PlainRenderer()
            try:
                if len(prompts) == 1:
                    results = [await loop.run(prompts[0], renderer=plain_renderer)]
                else:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        TimeElapsedColumn(),
                        console=console,
                    ) as progress:
                        task_id = progress.add_task(
                            "Critique/Revision", total=len(prompts)
                        )

                        async def _one(p: str):
                            r = await loop.run(p, renderer=plain_renderer)
                            progress.advance(task_id)
                            return r

                        results = await _asyncio.gather(*(_one(p) for p in prompts))
            finally:
                await plain_renderer.aclose()

        # ---- Export preference pairs -----------------------------------
        builder = PreferencePairBuilder()
        added = builder.add_results(results)
        output.parent.mkdir(parents=True, exist_ok=True)
        count = builder.export_jsonl(output)

        # ---- Summary ----------------------------------------------------
        converged = sum(1 for r in results if r.converged)
        console.print(
            Panel.fit(
                f"[green]✓ Wrote {count} preference pairs to {output}[/green]\n"
                f"Converged: {converged}/{len(results)}\n"
                f"Provider: {choice.name} ({choice.model})\n"
                f"Added to dataset: {added}",
                title="CAI Run Complete",
            )
        )

    try:
        _asyncio.run(_go())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(code=130) from None
    except Exception as exc:  # noqa: BLE001 - boundary: surface to CLI user
        console.print(f"\n[red]✗ CAI run failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc


@cai_app.command("providers")
def cai_providers(
    timeout: float = typer.Option(
        5.0, "--timeout", help="Per-provider probe timeout in seconds."
    ),
) -> None:
    """Probe every provider in parallel and report which ones actually work."""
    import asyncio as _asyncio

    from autoconstitution.providers.probe import any_ready, probe_all
    from autoconstitution.ui.probe_view import (
        render_no_provider_panel,
        render_probe_table,
    )

    async def _check() -> int:
        results = await probe_all(timeout_s=timeout)
        console.print(render_probe_table(results))
        if not any_ready(results):
            console.print()
            console.print(render_no_provider_panel())
            return 1
        return 0

    raise typer.Exit(code=_asyncio.run(_check()))


# ============================================================================
# Demo Command — hero experience for new users
# ============================================================================

_DEMO_PROMPT = (
    "Design a one-month remote-work onboarding plan for a senior engineer "
    "who needs to ramp up on a large codebase without burning out. Focus on "
    "concrete weekly objectives and measurable checkpoints."
)


@app.command("demo")
def demo() -> None:
    """Run the Constitutional AI loop on a canned prompt with zero configuration.

    Detects an available provider (Ollama first, then cloud keys), renders the
    live role-panel dashboard, and walks through three critique/revision
    rounds. Intended as the first thing a new user runs.
    """
    import asyncio as _asyncio

    from autoconstitution.cai import (
        CritiqueRevisionLoop,
        JudgeAgent,
        StudentAgent,
    )
    from autoconstitution.providers import pick_provider
    from autoconstitution.providers.probe import any_ready, probe_all
    from autoconstitution.ui.live import LiveRenderer
    from autoconstitution.ui.probe_view import (
        render_no_provider_panel,
        render_probe_table,
    )

    async def _go() -> int:
        console.print(
            Panel.fit(
                Text.from_markup(
                    "[bold]autoconstitution demo[/bold]\n"
                    "[dim]Watching constitutional AI critique and revise itself "
                    "in three rounds.[/dim]"
                ),
                border_style="cyan",
            )
        )

        # 1. Probe providers — same widget as `cai providers` so the user learns
        #    both commands at once.
        probe_results = await probe_all(timeout_s=5.0)
        console.print(render_probe_table(probe_results))
        if not any_ready(probe_results):
            console.print()
            console.print(render_no_provider_panel())
            return 1

        # 2. Pick a provider. pick_provider() knows how to wire the adapter.
        choice = await pick_provider()
        console.print(
            f"\n[green]✓ Provider ready:[/green] {choice.name} "
            f"[dim]({choice.model} — {choice.reason})[/dim]"
        )
        console.print(f"\n[bold]Prompt:[/bold] {_DEMO_PROMPT}\n")

        # 3. Spin up Student + Judge and run the loop with the live dashboard.
        student = StudentAgent(provider=choice.provider)
        judge = JudgeAgent(provider=choice.provider)
        loop = CritiqueRevisionLoop(student=student, judge=judge, max_rounds=3)

        renderer = LiveRenderer(console=console, max_rounds=3)
        try:
            result = await loop.run(_DEMO_PROMPT, renderer=renderer)
        finally:
            await renderer.aclose()

        # 4. Summary + nudge toward real usage.
        converged_note = (
            "[green]✓ Judge signed off on the revision[/green]"
            if result.converged
            else "[yellow]⚠ Still critiques outstanding — would benefit from "
            "more rounds[/yellow]"
        )
        console.print(
            Panel.fit(
                f"{converged_note}\n"
                f"Rounds used: {result.rounds_used}\n"
                f"Critiques collected: {len(result.critiques)}\n\n"
                "[bold]Next:[/bold]\n"
                "  autoconstitution cai run --prompt \"your task here\"\n"
                "  autoconstitution cai run --prompts-file prompts.txt -o pairs.jsonl",
                title="demo complete",
                border_style="green",
            )
        )
        return 0

    try:
        raise typer.Exit(code=_asyncio.run(_go()))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(code=130) from None
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001 - boundary: surface to CLI user
        console.print(f"\n[red]✗ Demo failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
