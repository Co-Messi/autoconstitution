"""
autoconstitution Terminal Dashboard

A real-time terminal dashboard for monitoring swarm research operations,
agent statuses, ratchet scores, and cross-pollination activities.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Protocol, Tuple

from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


# =============================================================================
# Data Models
# =============================================================================

class AgentStatus(Enum):
    """Status of a research agent."""
    IDLE = auto()
    EXPLORING = auto()
    EVALUATING = auto()
    CROSS_POLLINATING = auto()
    WAITING = auto()


@dataclass
class ResearchAgent:
    """Represents an active research agent in the swarm."""
    agent_id: str
    name: str
    status: AgentStatus
    research_direction: str
    current_score: float
    iterations_completed: int
    last_update: datetime = field(default_factory=datetime.now)
    
    def get_status_color(self) -> str:
        """Get color based on agent status."""
        color_map = {
            AgentStatus.IDLE: "dim",
            AgentStatus.EXPLORING: "blue",
            AgentStatus.EVALUATING: "yellow",
            AgentStatus.CROSS_POLLINATING: "magenta",
            AgentStatus.WAITING: "dim",
        }
        return color_map.get(self.status, "white")


@dataclass
class RatchetScore:
    """Represents a ratchet score with improvement history."""
    current_score: float
    best_score: float
    baseline_score: float
    improvement_history: List[Tuple[datetime, float]] = field(default_factory=list)
    
    @property
    def total_improvement(self) -> float:
        """Calculate total improvement from baseline."""
        return self.best_score - self.baseline_score
    
    @property
    def improvement_percentage(self) -> float:
        """Calculate improvement percentage."""
        if self.baseline_score == 0:
            return 0.0
        return (self.total_improvement / self.baseline_score) * 100


@dataclass
class CrossPollinationEvent:
    """Represents a cross-pollination broadcast event."""
    timestamp: datetime
    source_agent: str
    target_agents: List[str]
    insight_type: str
    impact_score: float
    description: str


@dataclass
class BranchPerformance:
    """Performance metrics for a research branch."""
    branch_id: str
    branch_name: str
    agents_count: int
    total_iterations: int
    best_score: float
    avg_score: float
    improvement_rate: float  # improvements per hour
    
    def get_performance_rating(self) -> str:
        """Get performance rating based on improvement rate."""
        if self.improvement_rate >= 5.0:
            return "[green]Excellent[/green]"
        elif self.improvement_rate >= 2.0:
            return "[blue]Good[/blue]"
        elif self.improvement_rate >= 0.5:
            return "[yellow]Fair[/yellow]"
        return "[red]Poor[/red]"


@dataclass
class SwarmState:
    """Complete state of the swarm research system."""
    agents: Dict[str, ResearchAgent] = field(default_factory=dict)
    ratchet: RatchetScore = field(default_factory=lambda: RatchetScore(0.0, 0.0, 0.0))
    cross_pollinations: List[CrossPollinationEvent] = field(default_factory=list)
    branches: Dict[str, BranchPerformance] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None
    total_iterations: int = 0
    
    def get_active_agents_count(self) -> int:
        """Count agents that are currently active."""
        return sum(
            1 for a in self.agents.values()
            if a.status in (AgentStatus.EXPLORING, AgentStatus.EVALUATING, AgentStatus.CROSS_POLLINATING)
        )


# =============================================================================
# Dashboard Components
# =============================================================================

class DashboardComponent(Protocol):
    """Protocol for dashboard components."""
    
    def render(self, state: SwarmState) -> RenderableType:
        """Render the component given the current state."""
        ...


class AgentsPanel:
    """Panel displaying active agents and their research directions."""
    
    def render(self, state: SwarmState) -> Panel:
        """Render the agents panel."""
        table = Table(
            title="",
            box=None,
            expand=True,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Agent", style="cyan", width=12)
        table.add_column("Status", width=10)
        table.add_column("Direction", style="green", min_width=20)
        table.add_column("Score", justify="right", width=10)
        table.add_column("Iter", justify="right", width=6)
        table.add_column("Last Update", width=12)
        
        # Sort agents by status (active first) then by score
        sorted_agents = sorted(
            state.agents.values(),
            key=lambda a: (
                a.status not in (AgentStatus.EXPLORING, AgentStatus.EVALUATING),
                -a.current_score,
            ),
        )
        
        for agent in sorted_agents[:15]:  # Show top 15 agents
            status_color = agent.get_status_color()
            status_text = f"[{status_color}]{agent.status.name}[/{status_color}]"
            time_ago = self._format_time_ago(agent.last_update)
            
            table.add_row(
                agent.name,
                status_text,
                agent.research_direction[:28],
                f"{agent.current_score:.4f}",
                str(agent.iterations_completed),
                time_ago,
            )
        
        active_count = state.get_active_agents_count()
        total_count = len(state.agents)
        
        return Panel(
            table,
            title=f"[bold cyan]Research Agents[/bold cyan] ([green]{active_count}[/green]/{total_count} active)",
            border_style="cyan",
            padding=(0, 1),
        )
    
    @staticmethod
    def _format_time_ago(dt: datetime) -> str:
        """Format datetime as relative time."""
        delta = datetime.now() - dt
        if delta < timedelta(seconds=60):
            return f"{delta.seconds}s ago"
        elif delta < timedelta(minutes=60):
            return f"{delta.seconds // 60}m ago"
        return f"{delta.seconds // 3600}h ago"


class RatchetPanel:
    """Panel displaying ratchet score and improvement history."""
    
    def render(self, state: SwarmState) -> Panel:
        """Render the ratchet panel."""
        ratchet = state.ratchet
        
        # Main score display
        score_text = Text()
        score_text.append("Current Best: ", style="dim")
        score_text.append(f"{ratchet.best_score:.6f}", style="bold green")
        score_text.append("\n")
        score_text.append("Baseline: ", style="dim")
        score_text.append(f"{ratchet.baseline_score:.6f}", style="dim")
        score_text.append("\n")
        score_text.append("Improvement: ", style="dim")
        
        improvement = ratchet.total_improvement
        improvement_pct = ratchet.improvement_percentage
        
        if improvement > 0:
            score_text.append(f"+{improvement:.6f}", style="bold green")
            score_text.append(f" (+{improvement_pct:.2f}%)", style="green")
        else:
            score_text.append(f"{improvement:.6f}", style="yellow")
        
        # Improvement history sparkline
        history_text = Text("\n\nImprovement History:\n", style="dim")
        
        if ratchet.improvement_history:
            # Create simple bar chart of improvements
            recent_history = ratchet.improvement_history[-20:]
            values = [h[1] for h in recent_history]
            
            if values:
                max_val = max(values) if max(values) > 0 else 1
                min_val = min(values) if min(values) < 0 else 0
                
                bars = []
                for val in values:
                    if max_val == min_val:
                        height = 4
                    else:
                        height = int(((val - min_val) / (max_val - min_val)) * 7) + 1
                    
                    if val >= 0:
                        bars.append(f"[green]{'█' * height}[/green]")
                    else:
                        bars.append(f"[red]{'█' * height}[/red]")
                
                history_text.append(" ".join(bars))
        else:
            history_text.append("[dim]No improvements recorded yet[/dim]")
        
        content = Group(score_text, history_text)
        
        return Panel(
            content,
            title="[bold yellow]Ratchet Score[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )


class CrossPollinationPanel:
    """Panel displaying recent cross-pollination broadcasts."""
    
    def render(self, state: SwarmState) -> Panel:
        """Render the cross-pollination panel."""
        table = Table(
            box=None,
            expand=True,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Time", style="dim", width=8)
        table.add_column("Source", style="cyan", width=12)
        table.add_column("Targets", style="blue", min_width=15)
        table.add_column("Type", style="green", width=12)
        table.add_column("Impact", justify="right", width=8)
        
        # Show recent events (most recent first)
        recent_events = sorted(
            state.cross_pollinations,
            key=lambda e: e.timestamp,
            reverse=True,
        )[:10]
        
        for event in recent_events:
            time_str = event.timestamp.strftime("%H:%M:%S")
            targets_str = f"{len(event.target_agents)} agents"
            
            # Color impact based on score
            if event.impact_score >= 0.8:
                impact_str = f"[green]{event.impact_score:.2f}[/green]"
            elif event.impact_score >= 0.5:
                impact_str = f"[yellow]{event.impact_score:.2f}[/yellow]"
            else:
                impact_str = f"[dim]{event.impact_score:.2f}[/dim]"
            
            table.add_row(
                time_str,
                event.source_agent,
                targets_str,
                event.insight_type,
                impact_str,
            )
        
        if not recent_events:
            table.add_row("", "[dim]No cross-pollination events yet[/dim]", "", "", "")
        
        return Panel(
            table,
            title="[bold magenta]Cross-Pollination Broadcasts[/bold magenta]",
            border_style="magenta",
            padding=(0, 1),
        )


class BranchPerformancePanel:
    """Panel displaying branch performance comparison."""
    
    def render(self, state: SwarmState) -> Panel:
        """Render the branch performance panel."""
        table = Table(
            box=None,
            expand=True,
            show_header=True,
            header_style="bold blue",
        )
        table.add_column("Branch", style="cyan", min_width=15)
        table.add_column("Agents", justify="right", width=6)
        table.add_column("Iterations", justify="right", width=10)
        table.add_column("Best Score", justify="right", width=12)
        table.add_column("Avg Score", justify="right", width=12)
        table.add_column("Rate/hr", justify="right", width=8)
        table.add_column("Rating", width=12)
        
        # Sort branches by best score
        sorted_branches = sorted(
            state.branches.values(),
            key=lambda b: b.best_score,
            reverse=True,
        )
        
        for branch in sorted_branches:
            table.add_row(
                branch.branch_name,
                str(branch.agents_count),
                f"{branch.total_iterations:,}",
                f"{branch.best_score:.6f}",
                f"{branch.avg_score:.6f}",
                f"{branch.improvement_rate:.2f}",
                branch.get_performance_rating(),
            )
        
        if not sorted_branches:
            table.add_row(
                "[dim]No branches active[/dim]", "", "", "", "", "", ""
            )
        
        return Panel(
            table,
            title="[bold blue]Branch Performance[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        )


class TimeRemainingPanel:
    """Panel displaying time statistics and progress."""
    
    def render(self, state: SwarmState) -> Panel:
        """Render the time remaining panel."""
        elapsed = datetime.now() - state.start_time
        
        # Create progress bar
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            expand=True,
        )
        
        # Estimate progress (simulated)
        estimated_total_iterations = 10000
        progress_percent = min(100, (state.total_iterations / estimated_total_iterations) * 100)
        
        task = progress.add_task(
            "[cyan]Overall Progress",
            total=estimated_total_iterations,
            completed=state.total_iterations,
        )
        
        # Time statistics
        time_stats = Table(box=None, show_header=False, padding=(0, 2))
        time_stats.add_column(style="dim")
        time_stats.add_column()
        
        time_stats.add_row("Elapsed:", self._format_duration(elapsed))
        
        if state.estimated_completion:
            remaining = state.estimated_completion - datetime.now()
            if remaining.total_seconds() > 0:
                time_stats.add_row("Remaining:", self._format_duration(remaining))
                time_stats.add_row(
                    "ETA:",
                    f"[green]{state.estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}[/green]"
                )
        
        time_stats.add_row("Iterations:", f"{state.total_iterations:,}")
        time_stats.add_row("Rate:", f"{self._calculate_rate(state):,.0f} iter/hr")
        
        content = Group(progress, Text("\n"), time_stats)
        
        return Panel(
            content,
            title="[bold green]Time & Progress[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    
    @staticmethod
    def _format_duration(delta: timedelta) -> str:
        """Format timedelta as human-readable duration."""
        total_seconds = int(delta.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"
    
    @staticmethod
    def _calculate_rate(state: SwarmState) -> float:
        """Calculate iteration rate per hour."""
        elapsed = datetime.now() - state.start_time
        hours = elapsed.total_seconds() / 3600
        if hours > 0:
            return state.total_iterations / hours
        return 0.0


class HeaderComponent:
    """Dashboard header with title and system status."""
    
    def render(self, state: SwarmState) -> Panel:
        """Render the header."""
        title = Text("🐝 autoconstitution Dashboard", style="bold cyan")
        subtitle = Text(f"Real-time Monitoring • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
        
        # System status indicators
        status_items = [
            f"[green]●[/green] {state.get_active_agents_count()} Active",
            f"[blue]●[/blue] {len(state.branches)} Branches",
            f"[yellow]●[/yellow] Best: {state.ratchet.best_score:.4f}",
        ]
        
        if state.estimated_completion:
            remaining = state.estimated_completion - datetime.now()
            if remaining.total_seconds() > 0:
                status_items.append(f"[magenta]●[/magenta] ETA: {self._format_short_duration(remaining)}")
        
        status_line = Text("  |  ").join([Text.from_markup(s) for s in status_items])
        
        content = Group(
            Align.center(title),
            Align.center(subtitle),
            Text("\n"),
            Align.center(status_line),
        )
        
        return Panel(
            content,
            border_style="cyan",
            padding=(1, 2),
        )
    
    @staticmethod
    def _format_short_duration(delta: timedelta) -> str:
        """Format timedelta as short duration."""
        total_seconds = int(delta.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        
        if hours > 0:
            return f"{hours}h{minutes}m"
        return f"{minutes}m"


# =============================================================================
# Main Dashboard
# =============================================================================

class SwarmDashboard:
    """
    Main autoconstitution dashboard with real-time updates.
    
    Usage:
        dashboard = SwarmDashboard()
        dashboard.run(state_provider)
    """
    
    def __init__(self, refresh_rate: float = 1.0) -> None:
        """
        Initialize the dashboard.
        
        Args:
            refresh_rate: Update interval in seconds
        """
        self.console = Console()
        self.refresh_rate = refresh_rate
        
        # Initialize components
        self.header = HeaderComponent()
        self.agents_panel = AgentsPanel()
        self.ratchet_panel = RatchetPanel()
        self.cross_pollination_panel = CrossPollinationPanel()
        self.branch_panel = BranchPerformancePanel()
        self.time_panel = TimeRemainingPanel()
        
        # Layout configuration
        self.layout = self._create_layout()
    
    def _create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout(name="root")
        
        # Split into header and body
        layout.split(
            Layout(name="header", size=8),
            Layout(name="body"),
        )
        
        # Split body into left and right columns
        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )
        
        # Split left column into agents and cross-pollination
        layout["body"]["left"].split(
            Layout(name="agents", ratio=2),
            Layout(name="cross_pollination", ratio=1),
        )
        
        # Split right column into ratchet, branches, and time
        layout["body"]["right"].split(
            Layout(name="ratchet", ratio=1),
            Layout(name="branches", ratio=1),
            Layout(name="time", ratio=1),
        )
        
        return layout
    
    def render(self, state: SwarmState) -> Layout:
        """Render the complete dashboard with current state."""
        self.layout["header"].update(self.header.render(state))
        self.layout["body"]["left"]["agents"].update(self.agents_panel.render(state))
        self.layout["body"]["left"]["cross_pollination"].update(
            self.cross_pollination_panel.render(state)
        )
        self.layout["body"]["right"]["ratchet"].update(self.ratchet_panel.render(state))
        self.layout["body"]["right"]["branches"].update(self.branch_panel.render(state))
        self.layout["body"]["right"]["time"].update(self.time_panel.render(state))
        
        return self.layout
    
    def run(
        self,
        state_provider: Callable[[], SwarmState],
        duration: Optional[float] = None,
    ) -> None:
        """
        Run the dashboard with live updates.
        
        Args:
            state_provider: Function that returns current SwarmState
            duration: Optional duration to run in seconds
        """
        start_time = time.time()
        
        with Live(
            self.render(state_provider()),
            console=self.console,
            refresh_per_second=1 / self.refresh_rate,
            screen=True,
        ) as live:
            try:
                while True:
                    # Check duration
                    if duration and (time.time() - start_time) >= duration:
                        break
                    
                    # Update display
                    state = state_provider()
                    live.update(self.render(state))
                    
                    # Sleep
                    time.sleep(self.refresh_rate)
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Dashboard stopped by user[/yellow]")
    
    async def run_async(
        self,
        state_provider: Callable[[], SwarmState],
    ) -> None:
        """
        Run the dashboard asynchronously with live updates.
        
        Args:
            state_provider: Function that returns current SwarmState
        """
        with Live(
            self.render(state_provider()),
            console=self.console,
            refresh_per_second=1 / self.refresh_rate,
            screen=True,
        ) as live:
            try:
                while True:
                    state = state_provider()
                    live.update(self.render(state))
                    await asyncio.sleep(self.refresh_rate)
            except asyncio.CancelledError:
                self.console.print("\n[yellow]Dashboard stopped[/yellow]")


# =============================================================================
# Demo / Simulation
# =============================================================================

def create_demo_state() -> SwarmState:
    """Create a demo swarm state for testing."""
    state = SwarmState()
    
    # Add demo agents
    research_directions = [
        "Neural Architecture Search",
        "Hyperparameter Optimization",
        "Ensemble Methods",
        "Feature Engineering",
        "Data Augmentation",
        "Transfer Learning",
        "Meta-Learning",
        "Federated Learning",
    ]
    
    statuses = list(AgentStatus)
    
    for i in range(12):
        agent = ResearchAgent(
            agent_id=f"agent_{i:03d}",
            name=f"Agent-{i+1:02d}",
            status=random.choice(statuses),
            research_direction=random.choice(research_directions),
            current_score=0.5 + random.random() * 0.4,
            iterations_completed=random.randint(100, 5000),
        )
        state.agents[agent.agent_id] = agent
    
    # Add ratchet score with history
    history = []
    base_score = 0.5
    for i in range(50):
        timestamp = datetime.now() - timedelta(minutes=50-i)
        score = base_score + (i * 0.005) + random.random() * 0.01
        history.append((timestamp, score))
    
    state.ratchet = RatchetScore(
        current_score=history[-1][1] if history else 0.5,
        best_score=history[-1][1] if history else 0.75,
        baseline_score=0.5,
        improvement_history=history,
    )
    
    # Add cross-pollination events
    insight_types = ["Architecture", "Hyperparams", "Features", "Training", "Data"]
    for i in range(15):
        event = CrossPollinationEvent(
            timestamp=datetime.now() - timedelta(minutes=i*3),
            source_agent=f"Agent-{random.randint(1, 12):02d}",
            target_agents=[f"Agent-{random.randint(1, 12):02d}" for _ in range(random.randint(2, 5))],
            insight_type=random.choice(insight_types),
            impact_score=random.random(),
            description="Shared optimization insight",
        )
        state.cross_pollinations.append(event)
    
    # Add branch performance
    branch_names = ["Main Branch", "Exploration-A", "Exploration-B", "Fine-tuning"]
    for i, name in enumerate(branch_names):
        branch = BranchPerformance(
            branch_id=f"branch_{i}",
            branch_name=name,
            agents_count=random.randint(2, 5),
            total_iterations=random.randint(1000, 10000),
            best_score=0.7 + random.random() * 0.2,
            avg_score=0.6 + random.random() * 0.15,
            improvement_rate=random.random() * 8,
        )
        state.branches[branch.branch_id] = branch
    
    state.start_time = datetime.now() - timedelta(hours=2)
    state.estimated_completion = datetime.now() + timedelta(hours=3)
    state.total_iterations = sum(a.iterations_completed for a in state.agents.values())
    
    return state


def demo_state_provider() -> SwarmState:
    """Provider that returns a slightly modified demo state for animation."""
    state = create_demo_state()
    
    # Randomly update some values to simulate activity
    for agent in state.agents.values():
        if random.random() < 0.3:
            agent.iterations_completed += random.randint(1, 10)
            agent.current_score += random.random() * 0.001 - 0.0005
            agent.last_update = datetime.now()
    
    state.total_iterations = sum(a.iterations_completed for a in state.agents.values())
    
    return state


def main() -> None:
    """Run the demo dashboard."""
    dashboard = SwarmDashboard(refresh_rate=1.0)
    dashboard.run(demo_state_provider)


if __name__ == "__main__":
    main()
