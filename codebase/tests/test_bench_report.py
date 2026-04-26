"""Tests for render_report_table + render_report_summary."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from autoconstitution.benchmark.protocol import (
    BenchCase,
    BenchReport,
    CaseOutcome,
    ScoreResult,
)
from autoconstitution.benchmark.report import (
    render_report_summary,
    render_report_table,
)


def _render(obj: object) -> str:
    buf = StringIO()
    console = Console(file=buf, width=120, record=True, force_terminal=False)
    console.print(obj)
    return buf.getvalue()


def _mk_outcome(
    case_id: str,
    before: float,
    after: float,
    *,
    elapsed: float = 0.5,
) -> CaseOutcome:
    return CaseOutcome(
        case=BenchCase(id=case_id, prompt="p"),
        before_answer="before",
        after_answer="after",
        before_score=ScoreResult(score=before, detail="", passed=None),
        after_score=ScoreResult(score=after, detail="", passed=None),
        rounds_used=1,
        converged=True,
        elapsed_s=elapsed,
    )


def _mk_report(outcomes: list[CaseOutcome]) -> BenchReport:
    n = len(outcomes)
    before_avg = sum(o.before_score.score for o in outcomes) / n if n else 0.0
    after_avg = sum(o.after_score.score for o in outcomes) / n if n else 0.0
    wins = sum(1 for o in outcomes if o.verdict == "win")
    ties = sum(1 for o in outcomes if o.verdict == "tie")
    losses = sum(1 for o in outcomes if o.verdict == "loss")
    return BenchReport(
        outcomes=outcomes,
        scorer_name="test_scorer",
        aggregate_before=before_avg,
        aggregate_after=after_avg,
        delta=after_avg - before_avg,
        wins=wins,
        ties=ties,
        losses=losses,
        ci95=(-0.1, 0.2),
    )


class TestRenderReportTable:
    def test_includes_all_cases_when_no_max(self) -> None:
        report = _mk_report([_mk_outcome(f"c{i}", 0.1, 0.5) for i in range(5)])
        out = _render(render_report_table(report))
        for i in range(5):
            assert f"c{i}" in out

    def test_shows_verdict_glyphs(self) -> None:
        report = _mk_report(
            [
                _mk_outcome("win", 0.1, 0.9),
                _mk_outcome("tie", 0.5, 0.5),
                _mk_outcome("loss", 0.9, 0.1),
            ]
        )
        out = _render(render_report_table(report))
        assert "win" in out
        assert "tie" in out
        assert "loss" in out
        assert "▲" in out
        assert "▼" in out

    def test_max_rows_clips(self) -> None:
        outcomes = [_mk_outcome(f"c{i}", 0.1, 0.1 + i * 0.1) for i in range(20)]
        report = _mk_report(outcomes)
        out = _render(render_report_table(report, max_rows=4))
        assert "c0" in out   # one of the worst (smallest delta)
        assert "c19" in out  # one of the best (largest delta)
        # Middle of the distribution drops out.
        assert "c10" not in out


class TestRenderReportSummary:
    def test_shows_delta_and_ci(self) -> None:
        report = _mk_report([_mk_outcome("c", 0.3, 0.7)])
        out = _render(render_report_summary(report))
        assert "+0.4" in out
        assert "95% CI" in out
        assert "n=1" in out

    def test_includes_win_loss_breakdown(self) -> None:
        report = _mk_report(
            [
                _mk_outcome("w", 0.1, 0.9),
                _mk_outcome("t", 0.5, 0.5),
                _mk_outcome("l", 0.9, 0.1),
            ]
        )
        out = _render(render_report_summary(report))
        assert "wins" in out and "1" in out
        assert "ties" in out
        assert "losses" in out

    def test_empty_report_does_not_crash(self) -> None:
        report = _mk_report([])
        out = _render(render_report_summary(report))
        assert "no cases" in out
