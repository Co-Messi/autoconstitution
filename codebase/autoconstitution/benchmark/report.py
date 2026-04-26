"""Rich renderers for a :class:`BenchReport`.

Two artifacts, one call each:

* :func:`render_report_table` — per-case row (id, before, after, Δ, verdict, time).
* :func:`render_report_summary` — aggregate panel (n, delta, win/tie/loss, 95% CI).

Separated from the runner so the runner is pure data. The CLI composes them.
"""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from autoconstitution.benchmark.protocol import BenchReport, CaseOutcome

_VERDICT_STYLE: dict[str, str] = {
    "win": "bold green",
    "tie": "yellow",
    "loss": "dim red",
}

_VERDICT_GLYPH: dict[str, str] = {
    "win": "▲",
    "tie": "=",
    "loss": "▼",
}


def render_report_table(
    report: BenchReport,
    *,
    max_rows: int | None = None,
) -> Table:
    """Render the per-case table.

    Args:
        report: The report to render.
        max_rows: If set, show at most this many rows (best + worst halves).
            ``None`` shows every row.
    """
    table = Table(
        title=f"Benchmark — {report.scorer_name} · n={report.n}",
        title_style="bold",
        show_lines=False,
        expand=False,
    )
    table.add_column("case", style="cyan", no_wrap=True)
    table.add_column("before", justify="right")
    table.add_column("after", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("verdict", justify="center")
    table.add_column("time", justify="right", style="dim")

    rows = _select_rows(report.outcomes, max_rows)
    for outcome in rows:
        style = _VERDICT_STYLE[outcome.verdict]
        glyph = _VERDICT_GLYPH[outcome.verdict]
        table.add_row(
            outcome.case.id,
            f"{outcome.before_score.score:.3f}",
            f"{outcome.after_score.score:.3f}",
            Text(f"{outcome.delta:+.3f}", style=style),
            Text(f"{glyph} {outcome.verdict}", style=style),
            f"{outcome.elapsed_s:.1f}s",
        )
    return table


def render_report_summary(report: BenchReport) -> Panel:
    """Render the aggregate summary panel — the quotable numbers."""
    lo, hi = report.ci95
    delta_sign = "+" if report.delta >= 0 else ""
    header = Text.from_markup(
        f"[bold]aggregate Δ:[/bold] {delta_sign}{report.delta:.4f}  "
        f"[dim](before {report.aggregate_before:.4f} → after {report.aggregate_after:.4f})[/dim]"
    )
    ci = Text.from_markup(
        f"[bold]95% CI[/bold] (bootstrap, n={report.n}): "
        f"[{lo:+.4f}, {hi:+.4f}]"
    )
    breakdown = Text.from_markup(
        f"[green]▲ wins[/green]: {report.wins}   "
        f"[yellow]= ties[/yellow]: {report.ties}   "
        f"[dim red]▼ losses[/dim red]: {report.losses}"
    )

    if report.n == 0:
        body: RenderableType = Text("(no cases ran)", style="dim")
    else:
        body = Group(header, ci, breakdown)

    return Panel(
        body,
        title="Benchmark summary",
        border_style="green" if report.delta > 0 else "yellow",
        padding=(0, 1),
    )


def _select_rows(
    outcomes: list[CaseOutcome], max_rows: int | None
) -> list[CaseOutcome]:
    """Return up to ``max_rows`` outcomes — best half + worst half if clipping."""
    if max_rows is None or len(outcomes) <= max_rows:
        return outcomes
    half = max_rows // 2
    by_delta = sorted(outcomes, key=lambda o: o.delta)
    worst = by_delta[:half]
    best = by_delta[-half:]
    # Preserve input order within each slice so the reader can map back.
    keep = {id(o) for o in worst} | {id(o) for o in best}
    return [o for o in outcomes if id(o) in keep]


__all__ = ["render_report_summary", "render_report_table"]
