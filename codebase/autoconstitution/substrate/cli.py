"""
autoconstitution.substrate.cli
================================

Substrate subgroup CLI: ``autoconstitution substrate <command>``.

Commands:
    run       <prompt>  — Run the SubstrateLoop on a prompt (or JSONL file).
    status              — Manifold stats + self-map summary.
    capabilities        — Rich table: domain × difficulty × kind.
    curriculum list|generate|next — Curriculum management.
    forget --id <id>    — Revoke a packet (refuses active SKILL without --force).
    demo                — Built-in mock demo; works with no network/Ollama.

MANIFOLD integration: the substrate Typer app is registered into the main
autoconstitution CLI via ``app.add_typer(substrate_app, name="substrate")``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from typing_extensions import Annotated

console = Console()
logger = logging.getLogger(__name__)

substrate_app = typer.Typer(
    name="substrate",
    help=(
        "MANIFOLD shared-state substrate: persistent packets, causal graph, "
        "capability self-map, curriculum, and proof artifacts."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def _get_manifold(db_path: str | None = None):
    from autoconstitution.substrate.manifold import Manifold
    return Manifold(db_path=db_path)


def _get_self_map(manifold):
    from autoconstitution.substrate.capability_self_map import CapabilitySelfMap
    return CapabilitySelfMap(manifold)


# ─────────────────────────────────────────────
# run
# ─────────────────────────────────────────────


@substrate_app.command("run")
def substrate_run(
    prompt: Annotated[
        Optional[str],
        typer.Option("--prompt", "-p", help="Prompt string to run."),
    ] = None,
    prompts_file: Annotated[
        Optional[Path],
        typer.Option("-f", "--file", help="JSONL file with task dicts (one per line)."),
    ] = None,
    db_path: Annotated[
        Optional[str],
        typer.Option("--db", help="Path to substrate SQLite db. Default: ~/.autoconstitution/substrate.db"),
    ] = None,
    domain: Annotated[str, typer.Option("--domain", help="Task domain.")] = "general",
    difficulty: Annotated[str, typer.Option("--difficulty", help="easy|medium|hard.")] = "medium",
    kind: Annotated[str, typer.Option("--kind", help="code|math|reasoning|extraction|free.")] = "free",
) -> None:
    """Run the SubstrateLoop on one prompt or a JSONL file of tasks."""
    import uuid

    from autoconstitution.providers.fake import FakeProvider

    tasks: list[dict] = []
    if prompt:
        tasks.append({"prompt": prompt, "domain": domain, "difficulty": difficulty, "kind": kind})
    if prompts_file:
        for line in prompts_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                tasks.append(obj)
            except json.JSONDecodeError:
                tasks.append({"prompt": line, "domain": domain, "difficulty": difficulty, "kind": kind})

    if not tasks:
        console.print("[red]No prompts provided. Use --prompt or --file.[/red]")
        raise typer.Exit(1)

    manifold = _get_manifold(db_path)

    # Try real provider; fall back to FakeProvider
    try:
        from autoconstitution.providers import pick_provider
        provider_obj = asyncio.run(_async_pick_provider())
        provider = provider_obj.provider
        console.print(f"[green]Provider: {provider_obj.name} ({provider_obj.model})[/green]")
    except Exception:
        provider = FakeProvider(responses=lambda p: f"[mock] Response to: {p[:60]}")
        console.print("[yellow]No provider available — using mock responses.[/yellow]")

    from autoconstitution.substrate.capability_self_map import CapabilitySelfMap
    from autoconstitution.substrate.curriculum import CurriculumGenerator
    from autoconstitution.substrate.loop import SubstrateLoop
    from autoconstitution.substrate.shadow_validator import ShadowValidator
    from autoconstitution.substrate.skill_compiler import SkillCompiler

    self_map = CapabilitySelfMap(manifold)
    compiler = SkillCompiler(manifold, provider=provider)
    validator = ShadowValidator(manifold, protected_tasks={})
    curriculum = CurriculumGenerator(provider, manifold, self_map)
    loop = SubstrateLoop(
        provider=provider,
        manifold=manifold,
        self_map=self_map,
        skill_compiler=compiler,
        shadow_validator=validator,
        curriculum_gen=curriculum,
    )

    for task in tasks:
        run_id = str(uuid.uuid4())
        console.print(f"\n[bold cyan]Running:[/bold cyan] {task.get('prompt', '')[:80]}")
        result = asyncio.run(loop.run(task, run_id=run_id))
        console.print(Panel.fit(
            f"[green]Chosen:[/green] {result.chosen_text[:200]}\n"
            f"Packets created: {len(result.all_packet_ids)}\n"
            f"Proof: {result.proof.verdict if result.proof else 'n/a'}\n"
            f"Skill compiled: {result.skill_id or 'none'}",
            title=f"Result [{run_id[:8]}]",
        ))

    manifold.close()


async def _async_pick_provider():
    from autoconstitution.providers import pick_provider
    return await pick_provider()


# ─────────────────────────────────────────────
# status
# ─────────────────────────────────────────────


@substrate_app.command("status")
def substrate_status(
    db_path: Annotated[
        Optional[str],
        typer.Option("--db", help="Path to substrate SQLite db."),
    ] = None,
) -> None:
    """Show Manifold stats and capability self-map summary."""
    manifold = _get_manifold(db_path)
    self_map = _get_self_map(manifold)

    # Stats table
    stats = manifold.stats()
    stats_table = Table(title="Manifold Stats")
    stats_table.add_column("Type", style="cyan")
    stats_table.add_column("Count", style="green", justify="right")
    stats_table.add_column("Revoked", style="yellow", justify="right")
    stats_table.add_column("Avg Confidence", style="magenta", justify="right")

    for ptype, data in sorted(stats.items()):
        if ptype.startswith("_"):
            continue
        stats_table.add_row(
            ptype,
            str(data["count"]),
            str(data["revoked"]),
            f"{data['avg_confidence']:.3f}",
        )
    total = stats.get("_total", {})
    stats_table.add_row(
        "[bold]TOTAL[/bold]",
        str(total.get("count", 0)),
        f"{total.get('revoked', 0)} ({total.get('pct_revoked', 0):.1f}%)",
        "",
    )
    console.print(stats_table)
    console.print(f"[dim]Edges: {stats.get('_edges', {}).get('count', 0)}[/dim]")

    # Self-map summary
    summary = self_map.summary()
    if summary["total"] == 0:
        console.print("\n[dim]Capability self-map: empty (no outcomes recorded yet)[/dim]")
    else:
        sm_table = Table(title="Capability Self-Map")
        sm_table.add_column("Domain", style="cyan")
        sm_table.add_column("Difficulty", style="blue")
        sm_table.add_column("Kind", style="green")
        sm_table.add_column("N", style="white", justify="right")
        sm_table.add_column("Success %", style="magenta", justify="right")
        sm_table.add_column("EWMA Score", style="yellow", justify="right")
        for row in summary["signatures"]:
            sm_table.add_row(
                row["domain"],
                row["difficulty"],
                row["kind"],
                str(row["n"]),
                f"{row['success_rate']*100:.1f}%",
                f"{row['ewma_score']:.3f}",
            )
        console.print(sm_table)

    manifold.close()


# ─────────────────────────────────────────────
# capabilities
# ─────────────────────────────────────────────


@substrate_app.command("capabilities")
def substrate_capabilities(
    db_path: Annotated[
        Optional[str],
        typer.Option("--db", help="Path to substrate SQLite db."),
    ] = None,
) -> None:
    """Show a Rich table of domain × difficulty × kind competence scores."""
    manifold = _get_manifold(db_path)
    self_map = _get_self_map(manifold)
    summary = self_map.summary()

    if summary["total"] == 0:
        console.print("[dim]No capability data recorded yet.[/dim]")
        manifold.close()
        return

    table = Table(title="Capabilities (EWMA Score)")
    table.add_column("Domain", style="cyan")
    table.add_column("Difficulty", style="blue")
    table.add_column("Kind", style="green")
    table.add_column("N", style="white", justify="right")
    table.add_column("Score", style="magenta", justify="right")

    for row in sorted(summary["signatures"], key=lambda r: r["ewma_score"]):
        score = row["ewma_score"]
        color = "red" if score < 0.4 else ("yellow" if score < 0.7 else "green")
        table.add_row(
            row["domain"],
            row["difficulty"],
            row["kind"],
            str(row["n"]),
            f"[{color}]{score:.3f}[/{color}]",
        )

    console.print(table)
    manifold.close()


# ─────────────────────────────────────────────
# curriculum
# ─────────────────────────────────────────────

curriculum_app = typer.Typer(
    name="curriculum",
    help="Manage the curriculum self-generation system.",
    no_args_is_help=True,
)
substrate_app.add_typer(curriculum_app, name="curriculum")


@curriculum_app.command("list")
def curriculum_list(
    db_path: Annotated[Optional[str], typer.Option("--db")] = None,
) -> None:
    """List unresolved GOAL packets."""
    from autoconstitution.substrate.packet import PacketType

    manifold = _get_manifold(db_path)
    goals = manifold.query(type=PacketType.GOAL)
    if not goals:
        console.print("[dim]No active curriculum goals.[/dim]")
    else:
        table = Table(title=f"Active Goals ({len(goals)})")
        table.add_column("ID", style="dim", no_wrap=True)
        table.add_column("Content", style="cyan")
        for g in goals:
            table.add_row(g.id[:12] + "…", g.content[:80])
        console.print(table)
    manifold.close()


@curriculum_app.command("generate")
def curriculum_generate(
    n: Annotated[int, typer.Option("--n", help="Number of goals to generate.")] = 3,
    db_path: Annotated[Optional[str], typer.Option("--db")] = None,
) -> None:
    """Generate n GOAL packets targeting weak capability spots."""
    manifold = _get_manifold(db_path)
    self_map = _get_self_map(manifold)
    from autoconstitution.substrate.curriculum import CurriculumGenerator

    gen = CurriculumGenerator(None, manifold, self_map)
    ids = asyncio.run(gen.generate(n=n))
    console.print(f"[green]Generated {len(ids)} goal(s): {ids}[/green]")
    manifold.close()


@curriculum_app.command("next")
def curriculum_next(
    db_path: Annotated[Optional[str], typer.Option("--db")] = None,
) -> None:
    """Show the next unresolved GOAL packet."""
    manifold = _get_manifold(db_path)
    self_map = _get_self_map(manifold)
    from autoconstitution.substrate.curriculum import CurriculumGenerator

    gen = CurriculumGenerator(None, manifold, self_map)
    pkt = gen.next_practice()
    if pkt is None:
        console.print("[dim]No pending practice goals.[/dim]")
    else:
        console.print(Panel.fit(pkt.content, title=f"Next Practice [{pkt.id[:8]}]"))
    manifold.close()


# ─────────────────────────────────────────────
# forget
# ─────────────────────────────────────────────


@substrate_app.command("forget")
def substrate_forget(
    packet_id: Annotated[str, typer.Option("--id", help="Packet ID to revoke.")],
    reason: Annotated[
        str, typer.Option("--reason", help="Reason for revocation.")
    ] = "user request",
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force revocation even for active SKILL packets."),
    ] = False,
    db_path: Annotated[Optional[str], typer.Option("--db")] = None,
) -> None:
    """Revoke a packet by ID. Refuses active SKILL packets without --force."""
    from autoconstitution.substrate.packet import PacketType

    manifold = _get_manifold(db_path)
    pkt = manifold.read(packet_id)
    if pkt is None:
        console.print(f"[red]Packet {packet_id!r} not found.[/red]")
        manifold.close()
        raise typer.Exit(1)

    if pkt.type == PacketType.SKILL and not pkt.metadata.get("quarantined", False) and not force:
        console.print(
            f"[red]Packet {packet_id} is an active SKILL. "
            f"Use --force to revoke it anyway.[/red]"
        )
        manifold.close()
        raise typer.Exit(1)

    manifold.revoke(packet_id, reason=reason)
    console.print(f"[green]Revoked packet {packet_id}: {reason}[/green]")
    manifold.close()


# ─────────────────────────────────────────────
# demo
# ─────────────────────────────────────────────

_DEMO_TASKS = [
    {
        "id": "demo-1",
        "prompt": "Write a Python function that returns the Fibonacci sequence up to n terms.",
        "domain": "code",
        "difficulty": "easy",
        "kind": "code",
        "code": "def fibonacci(n):\n    seq = []\n    a, b = 0, 1\n    for _ in range(n):\n        seq.append(a)\n        a, b = b, a + b\n    return seq\n",
        "tests": (
            "from solution import fibonacci\n"
            "def test_fib_first_8():\n"
            "    assert fibonacci(8) == [0, 1, 1, 2, 3, 5, 8, 13]\n"
            "def test_fib_empty():\n"
            "    assert fibonacci(0) == []\n"
        ),
    },
    {
        "id": "demo-2",
        "prompt": "Explain the difference between a list and a tuple in Python in one concise paragraph.",
        "domain": "reasoning",
        "difficulty": "easy",
        "kind": "free",
    },
    {
        "id": "demo-3",
        "prompt": "Write a function that checks if a string is a palindrome.",
        "domain": "code",
        "difficulty": "medium",
        "kind": "code",
        "code": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]\n",
        "tests": (
            "from solution import is_palindrome\n"
            "def test_palindrome_true():\n"
            "    assert is_palindrome('racecar') is True\n"
            "def test_palindrome_false():\n"
            "    assert is_palindrome('hello') is False\n"
            "def test_palindrome_spaces():\n"
            "    assert is_palindrome('A man a plan a canal Panama') is True\n"
        ),
    },
]


@substrate_app.command("demo")
def substrate_demo(
    db_path: Annotated[
        Optional[str],
        typer.Option("--db", help="Path to substrate SQLite db for demo."),
    ] = None,
) -> None:
    """Run a built-in mini-suite of coding problems with a mock provider.

    Works without any network connection or Ollama installation.
    Demonstrates: packet creation, proof artifacts, self-map updates,
    counterfactual shadow execution, and capability tracking.
    """
    import uuid

    from autoconstitution.providers.fake import FakeProvider
    from autoconstitution.substrate.capability_self_map import CapabilitySelfMap
    from autoconstitution.substrate.curriculum import CurriculumGenerator
    from autoconstitution.substrate.loop import SubstrateLoop
    from autoconstitution.substrate.shadow_validator import ShadowValidator
    from autoconstitution.substrate.skill_compiler import SkillCompiler

    console.print(
        Panel.fit(
            Text.from_markup(
                "[bold cyan]autoconstitution substrate demo[/bold cyan]\n"
                "[dim]MANIFOLD shared-state substrate — no network required.[/dim]\n\n"
                f"Running {len(_DEMO_TASKS)} tasks with a deterministic mock provider.\n"
                "Each task: claim packet → critique/revise → proof → self-map update."
            ),
            border_style="cyan",
        )
    )

    # Wire a deterministic mock provider — domain-specific checks FIRST,
    # then meta-level checks (critique/revision/lesson) so skill-augmented
    # prompts still return good domain answers.
    def _mock_respond(prompt: str) -> str:
        p = prompt.lower()
        # Domain-specific responses — check most-specific first so that
        # skill-augmented prompts (which may contain prior task keywords) still
        # return the right answer for the current task.
        if "palindrome" in p:
            return "A palindrome reads the same forwards and backwards after normalising case and spaces."
        if "list" in p and "tuple" in p:
            return "Lists are mutable ordered sequences; tuples are immutable. Use tuples for fixed data."
        if "fibonacci" in p:
            return "The Fibonacci sequence starts with 0 and 1; each subsequent term is the sum of the previous two."
        # Meta-level responses
        if "critique" in p or '"verdict"' in p or "return json" in p:
            return '{"verdict": "compliant", "critiques": []}'
        if "improved" in p or "revise" in p or "revision" in p:
            return "Here is the improved answer after revision."
        if "extract" in p or "lesson" in p:
            return "Lesson: keep functions focused on a single responsibility."
        return "Mock response for the demo task."

    mock_provider = FakeProvider(responses=_mock_respond)

    # Use a temp DB for the demo unless user specified one
    import tempfile
    if db_path is None:
        tmpdir = tempfile.mkdtemp(prefix="substrate_demo_")
        db_path = str(Path(tmpdir) / "demo.db")

    from autoconstitution.substrate.manifold import Manifold

    manifold = Manifold(db_path=db_path)
    self_map = CapabilitySelfMap(manifold)
    compiler = SkillCompiler(manifold, provider=mock_provider)
    validator = ShadowValidator(manifold, protected_tasks={})
    curriculum = CurriculumGenerator(mock_provider, manifold, self_map)
    loop = SubstrateLoop(
        provider=mock_provider,
        manifold=manifold,
        self_map=self_map,
        skill_compiler=compiler,
        shadow_validator=validator,
        curriculum_gen=curriculum,
    )

    results = []
    for task in _DEMO_TASKS:
        run_id = str(uuid.uuid4())
        console.print(f"\n[bold]Task:[/bold] [cyan]{task['prompt'][:70]}[/cyan]")
        try:
            result = asyncio.run(loop.run(task, run_id=run_id))
            results.append((task, result))

            proof_str = f"proof={result.proof.verdict}" if result.proof else "no proof"
            console.print(
                f"  [green]✓[/green] {proof_str} | "
                f"packets={len(result.all_packet_ids)} | "
                f"shadows={len(result.counterfactual_ids)} | "
                f"skill={'compiled' if result.skill_id else 'none'}"
            )
            console.print(f"  [dim]output: {result.chosen_text[:80]}[/dim]")
        except Exception as exc:
            console.print(f"  [red]✗ failed: {exc}[/red]")

    # Final summary
    stats = manifold.stats()
    sm_summary = self_map.summary()

    console.print("\n")
    summary_table = Table(title="Demo Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Tasks completed", str(len(results)))
    summary_table.add_row("Total packets", str(stats.get("_total", {}).get("count", 0)))
    summary_table.add_row("Causal edges", str(stats.get("_edges", {}).get("count", 0)))
    summary_table.add_row("Capability signatures tracked", str(sm_summary["total"]))

    if sm_summary["total"] > 0:
        avg_ewma = sum(r["ewma_score"] for r in sm_summary["signatures"]) / sm_summary["total"]
        summary_table.add_row("Avg EWMA competence", f"{avg_ewma:.3f}")

    console.print(summary_table)

    console.print(
        Panel.fit(
            "[bold]Next steps:[/bold]\n"
            "  autoconstitution substrate status\n"
            "  autoconstitution substrate capabilities\n"
            "  autoconstitution substrate curriculum list\n"
            "  autoconstitution substrate run --prompt 'your task'",
            title="demo complete",
            border_style="green",
        )
    )

    manifold.close()


__all__ = ["substrate_app"]
