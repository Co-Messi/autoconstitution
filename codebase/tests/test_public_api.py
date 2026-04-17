from __future__ import annotations

from typer.testing import CliRunner

import autoconstitution
from autoconstitution.cli import app


def test_root_package_exports_cai_primitives() -> None:
    assert autoconstitution.StudentAgent is not None
    assert autoconstitution.JudgeAgent is not None
    assert autoconstitution.MetaJudgeAgent is not None
    assert autoconstitution.CritiqueRevisionLoop is not None
    assert autoconstitution.PreferencePairBuilder is not None


def test_cli_help_uses_product_language() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    output = result.stdout.lower()
    assert "multi-agent" in output
    assert "swarm research experiments" not in output
    # Orchestrator commands are hidden behind a `legacy` subgroup so the
    # top-level help surfaces product commands (cai, demo) over legacy ones.
    assert "legacy" in output
    assert "cai" in output
    assert "demo" in output


def test_legacy_subgroup_exposes_orchestrator_commands() -> None:
    """The legacy subgroup keeps backwards compatibility with old scripts."""
    runner = CliRunner()

    result = runner.invoke(app, ["legacy", "--help"])

    assert result.exit_code == 0
    output = result.stdout.lower()
    for cmd in ("run", "resume", "status", "benchmark"):
        assert cmd in output


def test_top_level_no_longer_exposes_run_resume_status_benchmark() -> None:
    """Orchestrator verbs must not appear as root-level commands anymore."""
    runner = CliRunner()

    # Each of these used to be a root command; they should now fail at the
    # top level (typer exits 2 for unknown commands).
    for cmd in ("run", "resume", "status", "benchmark"):
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code != 0, f"{cmd!r} should no longer be a top-level command"
