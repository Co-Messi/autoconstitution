"""Rich rendering for the provider startup probe.

Two views:

* :func:`render_probe_table` — compact status table, one row per provider.
* :func:`render_no_provider_panel` — friendly first-run panel shown when
  every probe came back unavailable. Gives the exact command to fix it.

Both views are pure functions of the probe results; neither prints directly.
Callers own the ``Console``.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from autoconstitution.providers.probe import ProbeResult, Status

_GLYPH: dict[Status, str] = {
    "ready": "[green]✓[/green]",
    "no_key": "[yellow]—[/yellow]",
    "unreachable": "[red]✗[/red]",
    "no_model": "[yellow]⚠[/yellow]",
    "sdk_missing": "[yellow]⚠[/yellow]",
    "error": "[red]✗[/red]",
}

_STATUS_LABEL: dict[Status, str] = {
    "ready": "[green]ready[/green]",
    "no_key": "[dim]no key[/dim]",
    "unreachable": "[red]unreachable[/red]",
    "no_model": "[yellow]no model[/yellow]",
    "sdk_missing": "[yellow]sdk missing[/yellow]",
    "error": "[red]error[/red]",
}


def render_probe_table(results: list[ProbeResult], *, title: str = "Provider probe") -> Table:
    """Compact Rich table summarizing a :func:`probe_all` result."""
    table = Table(title=title, header_style="bold", expand=False)
    table.add_column("", width=2, no_wrap=True)
    table.add_column("Provider", style="bold")
    table.add_column("Status")
    table.add_column("Latency", justify="right")
    table.add_column("Detail", overflow="fold")

    for r in results:
        latency = f"{r.latency_ms:.0f} ms" if r.latency_ms is not None else "—"
        table.add_row(_GLYPH[r.status], r.name, _STATUS_LABEL[r.status], latency, r.detail)

    return table


def render_no_provider_panel() -> Panel:
    """Shown when no provider is available — exact commands to fix it."""
    body = Text.from_markup(
        "[bold]No LLM provider is available.[/bold] Pick one of:\n\n"
        "  [cyan]Local, free[/cyan]        "
        "brew install ollama && ollama serve && ollama pull llama3.2\n"
        "  [cyan]Moonshot Kimi[/cyan]     export MOONSHOT_API_KEY=...\n"
        "  [cyan]Anthropic Claude[/cyan]  export ANTHROPIC_API_KEY=...\n"
        "  [cyan]OpenAI[/cyan]            export OPENAI_API_KEY=...\n\n"
        "Then re-run [bold]autoconstitution cai providers[/bold] to verify."
    )
    return Panel(body, title="Getting started", border_style="yellow", expand=False)


__all__ = ["render_no_provider_panel", "render_probe_table"]
