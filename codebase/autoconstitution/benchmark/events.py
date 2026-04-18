"""Events emitted by the benchmark runner.

Kept separate from :mod:`autoconstitution.ui.events` so the CAI-loop Event
union stays focused on single-prompt concerns. The benchmark runner has
its own lifecycle (case start, pre-scored, post-scored, case end, bench
start/end) that doesn't map onto the per-round CAI events.

Renderers that already consume CAI events can opt into these too by
implementing ``on_bench_event``. The built-in plain/json renderers do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from autoconstitution.benchmark.protocol import BenchCase, ScoreResult


@dataclass(frozen=True, slots=True)
class BenchStart:
    """Emitted once before the first case runs."""

    n_cases: int
    scorer_name: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class BenchCaseStart:
    """One benchmark case is about to run."""

    index: int
    case: BenchCase
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class BenchPreScored:
    """Baseline (Student-only) answer has been generated and scored."""

    index: int
    case_id: str
    answer: str
    score: ScoreResult
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class BenchPostScored:
    """Post-CAI-loop answer has been generated and scored."""

    index: int
    case_id: str
    answer: str
    score: ScoreResult
    rounds_used: int
    converged: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class BenchCaseEnd:
    """One case is fully done — use this to advance a progress bar."""

    index: int
    case_id: str
    delta: float
    verdict: str  # "win" | "tie" | "loss"
    elapsed_s: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class BenchEnd:
    """Emitted once after the last case, before the report is returned."""

    n_cases: int
    wins: int
    ties: int
    losses: int
    aggregate_before: float
    aggregate_after: float
    delta: float
    timestamp: datetime = field(default_factory=datetime.now)


BenchEvent = (
    BenchStart
    | BenchCaseStart
    | BenchPreScored
    | BenchPostScored
    | BenchCaseEnd
    | BenchEnd
)
"""Tagged union of every event the benchmark runner emits."""


__all__ = [
    "BenchCaseEnd",
    "BenchCaseStart",
    "BenchEnd",
    "BenchEvent",
    "BenchPostScored",
    "BenchPreScored",
    "BenchStart",
]
