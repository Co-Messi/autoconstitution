"""CLI exit-code contract.

Slice 2 requires non-zero exit on provider failure, Ctrl+C, or validation
failure. These tests exercise the public CLI surface via typer's test
runner — nothing hits a real network.
"""

from __future__ import annotations

from typer.testing import CliRunner

from autoconstitution.cli import app


def test_cai_run_without_prompts_exits_nonzero() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["cai", "run"])
    assert result.exit_code == 1
    assert "no prompts" in result.output.lower()


def test_cai_run_missing_prompts_file_exits_nonzero() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["cai", "run", "-f", "/no/such/file.txt"])
    assert result.exit_code == 1


def test_cai_run_unknown_provider_exits_nonzero() -> None:
    runner = CliRunner()
    # Prompt provided so we don't short-circuit on validation; then pick_provider
    # rejects the unknown name with ValueError, which our CLI wraps to exit 1.
    result = runner.invoke(
        app, ["cai", "run", "--prompt", "hi", "--provider", "definitely-not-a-provider"]
    )
    assert result.exit_code == 1


def test_cai_providers_runs_without_error() -> None:
    runner = CliRunner()
    # The command probes real providers; with none available it still exits
    # gracefully (exit 1 per consultant's design) but must not raise.
    result = runner.invoke(app, ["cai", "providers", "--timeout", "0.1"])
    assert result.exit_code in (0, 1)


def test_help_exits_zero() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_demo_command_registered() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "demo" in result.output.lower()


def test_demo_command_help_exits_zero() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["demo", "--help"])
    assert result.exit_code == 0
    assert "canned prompt" in result.output.lower()


def test_bench_command_registered() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "bench" in result.output.lower()


def test_bench_command_help_exits_zero() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["bench", "--help"])
    assert result.exit_code == 0
    assert "benchmark" in result.output.lower()
    assert "--scorer" in result.output


def test_bench_unknown_scorer_exits_2() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["bench", "--scorer", "nope", "--yes"])
    assert result.exit_code == 2
    assert "unknown scorer" in result.output.lower()


def test_bench_missing_dataset_exits_2() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["bench", "--dataset", "/no/such/file.jsonl", "--scorer", "judge"]
    )
    assert result.exit_code == 2
