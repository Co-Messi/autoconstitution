"""Benchmark harness for the CAI loop.

``autoconstitution bench`` answers the question autoresearch asks first:
is the critique/revise loop actually making answers better?

The public surface:

- :class:`Scorer` — Protocol for judging an answer against a benchmark case.
- :class:`BenchCase` / :class:`ScoreResult` / :class:`CaseOutcome` / :class:`BenchReport`
  — typed data flowing through the runner.
- :func:`run_benchmark` — wires a baseline provider + a CAI loop + a scorer,
  returns a structured report.

Scorers ship in :mod:`autoconstitution.benchmark.scorers`. Datasets live in
:mod:`autoconstitution.benchmark.datasets`.
"""

from autoconstitution.benchmark.protocol import (
    BenchCase,
    BenchReport,
    CaseOutcome,
    Scorer,
    ScoreResult,
)
from autoconstitution.benchmark.runner import run_benchmark

__all__ = [
    "BenchCase",
    "BenchReport",
    "CaseOutcome",
    "ScoreResult",
    "Scorer",
    "run_benchmark",
]
