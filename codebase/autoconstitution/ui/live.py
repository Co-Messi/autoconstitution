"""The live dashboard — a Rich ``Live``/``Layout`` that renders the CAI loop.

Watch constitutional AI happen in your terminal:

* Header with the task prompt, round number, elapsed timer.
* Role panels (Student / Judge / Meta-Judge) in a grid, each one showing
  the latest output for that role. The active panel's border pulses;
  inactive panels dim.
* Ratchet scoreboard in the footer, re-rendered whenever a
  :class:`~autoconstitution.ui.events.RatchetDecision` arrives.

Design notes:

* Layout width is measured at start-up. Terminals narrower than 120 columns
  collapse to a single "current role" panel plus a tape of prior outputs.
* Streamed tokens (:class:`~autoconstitution.ui.events.Token`) append to the
  role's buffer. Providers that don't stream still get a single
  :class:`~autoconstitution.ui.events.RoleEnd` event and the panel swaps to
  the finished output at once — no fake spinner.
* The renderer is safe to use outside a TTY: pass ``force_terminal=False``
  to the Console, or use the plain renderer from ``ui.plain`` (owned by the
  consultant) for piped stdout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from autoconstitution.ratchet import ValidationDecision, ValidationResult
from autoconstitution.ui.events import (
    Critique,
    Event,
    LoopError,
    RatchetDecision,
    Revision,
    Role,
    RoleEnd,
    RoleStart,
    RoundEnd,
    RoundStart,
    Token,
)
from autoconstitution.ui.scoreboard import render_scoreboard

_ROLE_ORDER: tuple[Role, ...] = ("student", "judge", "meta_judge")

_ROLE_COLOR: dict[Role, str] = {
    "student": "bright_cyan",
    "judge": "bright_magenta",
    "meta_judge": "bright_yellow",
    # Aliases used by hierarchy-style runs; rendered identically.
    "critic": "bright_magenta",
    "teacher": "bright_blue",
    "synthesizer": "bright_green",
}

_ROLE_TITLE: dict[Role, str] = {
    "student": "Student",
    "judge": "Judge",
    "meta_judge": "Meta-Judge",
    "critic": "Critic",
    "teacher": "Teacher",
    "synthesizer": "Synthesizer",
}

_NARROW_THRESHOLD = 120
"""Below this width we collapse the grid to a single active panel."""


@dataclass
class _RolePanelState:
    """Per-role scratch state that feeds the renderer."""

    buffer: str = ""
    output: str | None = None
    last_activity: float = 0.0

    def reset(self) -> None:
        self.buffer = ""
        self.output = None
        self.last_activity = time.monotonic()


@dataclass
class _LiveState:
    """Loop-wide state kept in one place so render passes are pure."""

    prompt: str = ""
    round: int = 0
    max_rounds: int = 0
    started: float = field(default_factory=time.monotonic)
    active_role: Role | None = None
    panels: dict[Role, _RolePanelState] = field(default_factory=dict)
    scoreboard: RenderableType | None = None
    latest_revision: Revision | None = None
    error: str | None = None
    converged: bool = False

    def panel(self, role: Role) -> _RolePanelState:
        return self.panels.setdefault(role, _RolePanelState())


class LiveRenderer:
    """Rich ``Live`` renderer. Implements :class:`~autoconstitution.ui.protocol.Renderer`."""

    supports_streaming: bool = True

    def __init__(
        self,
        *,
        console: Console | None = None,
        max_rounds: int = 3,
        refresh_per_second: int = 8,
    ) -> None:
        self._console = console or Console()
        self._state = _LiveState(max_rounds=max_rounds)
        self._live: Live | None = None
        self._refresh_per_second = refresh_per_second

    # ------------------------------------------------------------------
    # Renderer protocol
    # ------------------------------------------------------------------

    def on_event(self, event: Event) -> None:
        s = self._state
        match event:
            case RoundStart():
                s.prompt = event.prompt
                s.round = event.round
                s.error = None
                s.converged = False
                for panel in s.panels.values():
                    panel.reset()
                self._ensure_live()
            case RoleStart():
                s.active_role = event.role
                panel = s.panel(event.role)
                panel.buffer = ""
                panel.output = None
                panel.last_activity = time.monotonic()
            case Token():
                panel = s.panel(event.role)
                panel.buffer += event.text
                panel.last_activity = time.monotonic()
            case RoleEnd():
                panel = s.panel(event.role)
                panel.output = event.output
                panel.buffer = event.output
                panel.last_activity = time.monotonic()
            case Critique():
                # Surface verdict in the judge panel header via state; no direct change.
                pass
            case Revision():
                s.latest_revision = event
            case RatchetDecision():
                s.scoreboard = _scoreboard_from_decision(event)
            case RoundEnd():
                s.converged = event.converged
                s.active_role = None
            case LoopError():
                s.error = event.message
                s.active_role = None
        self._refresh()

    async def aclose(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def _ensure_live(self) -> None:
        if self._live is None:
            self._live = Live(
                self._render(),
                console=self._console,
                refresh_per_second=self._refresh_per_second,
                transient=False,
            )
            self._live.start()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> RenderableType:
        width = self._console.size.width
        if width < _NARROW_THRESHOLD:
            return self._render_narrow()
        return self._render_wide()

    def _render_wide(self) -> RenderableType:
        layout = Layout()
        layout.split_column(
            Layout(self._header(), size=3, name="header"),
            Layout(name="body", ratio=1),
            Layout(self._footer(), size=7, name="footer"),
        )
        panels = [
            Layout(self._render_panel(role), name=role) for role in _ROLE_ORDER
        ]
        layout["body"].split_row(*panels)
        return layout

    def _render_narrow(self) -> RenderableType:
        # Show only the active panel and a tape of the other roles' latest output.
        active = self._state.active_role or "student"
        active_panel = self._render_panel(active)
        tape = _tape_of_other_roles(self._state, exclude=active)
        group = Group(self._header(), active_panel, tape, self._footer())
        return group

    def _header(self) -> RenderableType:
        elapsed = time.monotonic() - self._state.started
        round_label = (
            f"Round {self._state.round}/{self._state.max_rounds}"
            if self._state.max_rounds
            else f"Round {self._state.round}"
        )
        prompt_line = Text.from_markup(
            f"[bold]autoconstitution[/bold] · {round_label}"
            f" · [dim]{elapsed:.1f}s[/dim]"
        )
        prompt_preview = Text(_truncate(self._state.prompt, 120), style="italic")
        group = Group(prompt_line, prompt_preview)
        return Panel(group, border_style="cyan", padding=(0, 1))

    def _footer(self) -> RenderableType:
        parts: list[RenderableType] = []
        if self._state.scoreboard is not None:
            parts.append(self._state.scoreboard)
        else:
            parts.append(Align.center(Text("(no ratchet decision yet)", style="dim")))
        parts.append(self._progress_bar())
        if self._state.error:
            parts.append(
                Panel(
                    Text(self._state.error, style="bold red"),
                    title="error",
                    border_style="red",
                )
            )
        return Group(*parts)

    def _progress_bar(self) -> RenderableType:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TimeElapsedColumn(),
            expand=True,
        )
        total = max(self._state.max_rounds, 1)
        task = progress.add_task("round progress", total=total)
        progress.update(task, completed=min(self._state.round, total))
        return progress

    def _render_panel(self, role: Role) -> RenderableType:
        state = self._state.panel(role)
        color = _ROLE_COLOR.get(role, "white")
        title = _ROLE_TITLE.get(role, role.title())
        is_active = self._state.active_role == role
        border = f"bold {color}" if is_active else "dim"
        body = state.output if state.output is not None else state.buffer
        if not body:
            body_renderable: RenderableType = Text("(waiting…)", style="dim")
        else:
            body_renderable = Text(body)
        status_marker = "●" if is_active else "○"
        return Panel(
            body_renderable,
            title=f"{status_marker} {title}",
            border_style=border,
            padding=(0, 1),
        )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _tape_of_other_roles(state: _LiveState, *, exclude: Role) -> RenderableType:
    rows: list[tuple[str, str]] = []
    for role in _ROLE_ORDER:
        if role == exclude:
            continue
        panel = state.panels.get(role)
        if panel is None or panel.output is None:
            continue
        label = _ROLE_TITLE.get(role) or role
        rows.append((label, _truncate(panel.output, 80)))
    if not rows:
        return Text("", style="dim")
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="dim")
    table.add_column()
    for label, body in rows:
        table.add_row(label, body)
    return table


def _scoreboard_from_decision(decision: RatchetDecision) -> RenderableType:
    """Adapter: build the ratchet scoreboard Table from a RatchetDecision event."""
    verdict_map = {
        "keep": ValidationDecision.KEEP,
        "discard": ValidationDecision.DISCARD,
        "tie": ValidationDecision.TIE,
        "first": ValidationDecision.FIRST,
    }
    result = ValidationResult(
        experiment_id=f"round_{decision.round}",
        score=decision.score,
        decision=verdict_map[decision.decision],
        is_improvement=decision.decision == "keep",
        previous_best=decision.previous_best,
        improvement_delta=decision.improvement_delta,
        improvement_pct=0.0,
        message="",
    )
    return render_scoreboard(
        decision.metric_name,
        result,
        round_label=f"Round {decision.round}",
    )


__all__ = ["LiveRenderer"]
