"""Data shapes and the ``Scorer`` Protocol for the benchmark runner.

Every benchmark run flows one thing through the system: a list of
:class:`BenchCase` in, a :class:`BenchReport` out. The runner invokes a
:class:`Scorer` on both the baseline answer and the CAI-revised answer,
so the report always contrasts before/after at the same metric.

Scorers must never raise on a bad answer. If they can't score, they
return ``ScoreResult(score=0.0, passed=False, detail="...")``. This keeps
the aggregate math simple: every case contributes one number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class BenchCase:
    """One input to the benchmark.

    Attributes:
        id: Stable identifier for the case (used in the report row).
        prompt: The user-facing prompt fed to Student and to the CAI loop.
        metadata: Scorer-specific fields. The ``coding`` scorer reads
            ``hidden_tests``; the ``judge`` scorer reads ``rubric``.
    """

    id: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Output of a :class:`Scorer` for one answer.

    Attributes:
        score: 0.0 – 1.0. Higher is better.
        detail: Human-readable explanation of why this score was given.
            Surfaced in the report so users can debug failures.
        passed: True iff the answer met a hard binary criterion (e.g., all
            hidden tests passed). ``None`` when the scorer is continuous and
            has no natural threshold — the report includes those cases in
            delta statistics but not in pass/fail tallies.
    """

    score: float
    detail: str
    passed: bool | None = None


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """The full result of running one :class:`BenchCase` through before+after."""

    case: BenchCase
    before_answer: str
    after_answer: str
    before_score: ScoreResult
    after_score: ScoreResult
    rounds_used: int
    converged: bool
    elapsed_s: float

    @property
    def delta(self) -> float:
        """``after_score.score - before_score.score``. Positive means CAI helped."""
        return self.after_score.score - self.before_score.score

    @property
    def verdict(self) -> str:
        """``"win"`` / ``"tie"`` / ``"loss"`` — whether CAI improved this case."""
        if self.delta > 1e-9:
            return "win"
        if self.delta < -1e-9:
            return "loss"
        return "tie"


@dataclass(frozen=True, slots=True)
class BenchReport:
    """Aggregated result of a benchmark run.

    Attributes:
        outcomes: Per-case results in input order.
        scorer_name: Which scorer produced the numbers.
        aggregate_before: Mean of ``before_score.score`` across outcomes.
        aggregate_after: Mean of ``after_score.score`` across outcomes.
        delta: ``aggregate_after - aggregate_before``.
        wins: Count of outcomes where CAI improved the answer.
        ties: Count of outcomes with no change.
        losses: Count of outcomes where CAI made it worse.
        ci95: 95% bootstrap confidence interval on the mean delta.
    """

    outcomes: list[CaseOutcome]
    scorer_name: str
    aggregate_before: float
    aggregate_after: float
    delta: float
    wins: int
    ties: int
    losses: int
    ci95: tuple[float, float]

    @property
    def n(self) -> int:
        return len(self.outcomes)


@runtime_checkable
class Scorer(Protocol):
    """Judges an answer against a :class:`BenchCase`.

    Implementations must:

    - be safe to call concurrently on independent ``case`` / ``answer`` pairs;
    - never raise — always return a :class:`ScoreResult`, using ``score=0.0``
      and a ``detail`` field for failures;
    - return ``score`` in ``[0.0, 1.0]``, clamped if necessary.
    """

    name: str

    async def score(self, case: BenchCase, answer: str) -> ScoreResult:
        """Return a score in ``[0.0, 1.0]`` for ``answer`` given ``case``."""
        ...

    async def close(self) -> None:
        """Release resources (SDK clients, tmp dirs). Safe to call multiple times."""
        ...


__all__ = [
    "BenchCase",
    "BenchReport",
    "CaseOutcome",
    "ScoreResult",
    "Scorer",
]
