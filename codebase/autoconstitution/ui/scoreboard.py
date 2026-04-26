"""Ratchet scoreboard widget.

Renders the ratchet's decision after each round as a compact Rich table:
metric name, best score, last score, delta, and decision. The decision cell
is colour-coded so a glance at the terminal tells you whether the round was
kept, discarded, or tied.

Two entry points:

- :func:`render_scoreboard` returns a Rich renderable (``Table``) for use
  inside live dashboards and panels.
- :func:`scoreboard_line` returns a plain string for line-based logging when
  the terminal isn't a TTY.

The widget is intentionally dumb: it reads a :class:`ValidationResult` and a
metric name, formats, and returns. No side effects, no I/O.
"""

from __future__ import annotations

from rich.table import Table
from rich.text import Text

from autoconstitution.ratchet import ValidationDecision, ValidationResult

_DECISION_STYLE: dict[ValidationDecision, str] = {
    ValidationDecision.KEEP: "bold green",
    ValidationDecision.DISCARD: "dim red",
    ValidationDecision.TIE: "bold yellow",
    ValidationDecision.FIRST: "bold cyan",
}

_DECISION_GLYPH: dict[ValidationDecision, str] = {
    ValidationDecision.KEEP: "▲ KEEP",
    ValidationDecision.DISCARD: "▼ DISCARD",
    ValidationDecision.TIE: "= TIE",
    ValidationDecision.FIRST: "• FIRST",
}


def _format_score(score: float | None) -> str:
    """Format a score for display, or ``—`` for ``None``."""
    if score is None:
        return "—"
    return f"{score:.4f}"


def _format_delta(delta: float, decision: ValidationDecision) -> str:
    """Format the delta with a leading sign, or ``—`` when there's no prior best."""
    if decision == ValidationDecision.FIRST:
        return "—"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def render_scoreboard(
    metric_name: str,
    result: ValidationResult,
    *,
    round_label: str | None = None,
) -> Table:
    """Build a Rich ``Table`` summarising the latest ratchet decision.

    Args:
        metric_name: Name of the metric being ratcheted (e.g. ``"accuracy"``).
        result: The :class:`ValidationResult` returned by the ratchet.
        round_label: Optional label for the round (e.g. ``"Round 2"``). When
            provided, it appears in the title above the table.

    Returns:
        A Rich ``Table`` ready to print or embed in a panel/layout.
    """
    title = f"Ratchet — {metric_name}"
    if round_label:
        title = f"{round_label} · {title}"

    table = Table(title=title, title_style="bold", show_lines=False, expand=False)
    table.add_column("best", justify="right", style="cyan")
    table.add_column("last", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("decision", justify="center")

    best_score = (
        result.score if result.decision == ValidationDecision.KEEP else result.previous_best
    )
    decision_style = _DECISION_STYLE[result.decision]
    decision_text = Text(_DECISION_GLYPH[result.decision], style=decision_style)

    delta_text = Text(
        _format_delta(result.improvement_delta, result.decision),
        style=decision_style if result.decision != ValidationDecision.TIE else "yellow",
    )

    table.add_row(
        _format_score(best_score),
        _format_score(result.score),
        delta_text,
        decision_text,
    )
    return table


def scoreboard_line(metric_name: str, result: ValidationResult) -> str:
    """Return a single-line plain-text scoreboard for line-based logging."""
    best_score = (
        result.score if result.decision == ValidationDecision.KEEP else result.previous_best
    )
    return (
        f"[ratchet:{metric_name}] "
        f"best={_format_score(best_score)} "
        f"last={_format_score(result.score)} "
        f"Δ={_format_delta(result.improvement_delta, result.decision)} "
        f"→ {_DECISION_GLYPH[result.decision]}"
    )
