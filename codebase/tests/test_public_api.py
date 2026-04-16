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
    assert "multi-agent" in result.stdout.lower()
    assert "swarm research experiments" not in result.stdout.lower()
    assert "legacy orchestrator experiment runner" in result.stdout.lower()
