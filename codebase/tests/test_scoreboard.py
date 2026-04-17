"""Tests for the ratchet scoreboard widget."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from autoconstitution.ratchet import ValidationDecision, ValidationResult
from autoconstitution.ui import render_scoreboard, scoreboard_line


def _make_result(
    decision: ValidationDecision,
    score: float = 0.85,
    previous_best: float | None = 0.75,
    delta: float = 0.10,
) -> ValidationResult:
    return ValidationResult(
        experiment_id="exp_test",
        score=score,
        decision=decision,
        is_improvement=decision == ValidationDecision.KEEP,
        previous_best=previous_best,
        improvement_delta=delta,
        improvement_pct=delta * 100,
        message="test",
    )


def _render_to_string(table_or_text: object) -> str:
    buffer = StringIO()
    console = Console(file=buffer, width=80, record=True, force_terminal=False)
    console.print(table_or_text)
    return buffer.getvalue()


class TestRenderScoreboard:
    def test_keep_is_green(self) -> None:
        table = render_scoreboard("accuracy", _make_result(ValidationDecision.KEEP))
        output = _render_to_string(table)
        assert "KEEP" in output
        assert "accuracy" in output
        assert "+0.1000" in output

    def test_discard_is_red(self) -> None:
        result = _make_result(
            ValidationDecision.DISCARD,
            score=0.70,
            previous_best=0.75,
            delta=-0.05,
        )
        output = _render_to_string(render_scoreboard("accuracy", result))
        assert "DISCARD" in output
        assert "-0.0500" in output

    def test_tie(self) -> None:
        result = _make_result(
            ValidationDecision.TIE,
            score=0.75,
            previous_best=0.75,
            delta=0.0,
        )
        output = _render_to_string(render_scoreboard("accuracy", result))
        assert "TIE" in output

    def test_first_has_no_previous_best(self) -> None:
        result = _make_result(
            ValidationDecision.FIRST,
            score=0.50,
            previous_best=None,
            delta=0.0,
        )
        output = _render_to_string(render_scoreboard("accuracy", result))
        assert "FIRST" in output
        # Delta shows em-dash for FIRST since there's no prior score
        assert "—" in output

    def test_round_label_in_title(self) -> None:
        table = render_scoreboard(
            "accuracy",
            _make_result(ValidationDecision.KEEP),
            round_label="Round 3",
        )
        output = _render_to_string(table)
        assert "Round 3" in output
        assert "accuracy" in output


class TestScoreboardLine:
    def test_keep_line(self) -> None:
        line = scoreboard_line("accuracy", _make_result(ValidationDecision.KEEP))
        assert "[ratchet:accuracy]" in line
        assert "KEEP" in line
        assert "last=0.8500" in line
        assert "Δ=+0.1000" in line

    def test_discard_line(self) -> None:
        result = _make_result(
            ValidationDecision.DISCARD,
            score=0.70,
            previous_best=0.75,
            delta=-0.05,
        )
        line = scoreboard_line("accuracy", result)
        assert "DISCARD" in line
        assert "best=0.7500" in line
        assert "Δ=-0.0500" in line

    def test_first_line_shows_dash_for_best(self) -> None:
        result = _make_result(
            ValidationDecision.FIRST,
            score=0.50,
            previous_best=None,
            delta=0.0,
        )
        line = scoreboard_line("accuracy", result)
        assert "FIRST" in line
        assert "best=—" in line
        assert "Δ=—" in line

    def test_line_is_single_line(self) -> None:
        line = scoreboard_line("accuracy", _make_result(ValidationDecision.KEEP))
        assert "\n" not in line
