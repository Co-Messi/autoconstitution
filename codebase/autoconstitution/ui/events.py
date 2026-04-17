"""Event dataclasses emitted by the CAI loop.

The loop emits a small set of typed events as it runs. Renderers (live
dashboard, line logger, JSON emitter) consume these events without the loop
needing to know how they're displayed.

Keeping the events in their own module means the CAI loop doesn't depend on
Rich or any rendering library — it just calls ``on_event(SomeEvent(...))``
and moves on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Role = Literal["student", "critic", "teacher", "judge", "synthesizer", "meta_judge"]
"""The canonical role names emitted by the loop.

``critic``/``teacher``/``synthesizer`` are aliases for hierarchy-style runs
that group Judge outputs into critique/teacher/synthesizer perspectives.
"""


@dataclass(frozen=True, slots=True)
class RoundStart:
    """Emitted when a new critique-revision round begins."""

    round: int
    prompt: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class RoleStart:
    """Emitted when a specific role begins generating output for a round."""

    role: Role
    round: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class Token:
    """Emitted once per streamed token. Providers without streaming skip this."""

    role: Role
    round: int
    text: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class RoleEnd:
    """Emitted when a role finishes producing its output for a round."""

    role: Role
    round: int
    output: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class Critique:
    """A parsed critique verdict from the Judge."""

    round: int
    verdict: Literal["compliant", "needs_revision", "parse_error"]
    critique_count: int
    raw: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class Revision:
    """Emitted after the Student revises in response to a critique."""

    round: int
    before: str
    after: str
    identical: bool
    """True when revision equals previous answer (loop will halt)."""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class RatchetDecision:
    """Result of the ratchet's decision after a round."""

    round: int
    metric_name: str
    decision: Literal["keep", "discard", "tie", "first"]
    score: float
    previous_best: float | None
    improvement_delta: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class RoundEnd:
    """Emitted when a round finishes, regardless of the ratchet decision."""

    round: int
    converged: bool
    """True when Judge returned ``compliant`` and we can stop iterating."""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class LoopError:
    """Emitted when a provider or loop-level error aborts the run."""

    round: int
    role: Role | None
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


Event = (
    RoundStart
    | RoleStart
    | Token
    | RoleEnd
    | Critique
    | Revision
    | RatchetDecision
    | RoundEnd
    | LoopError
)
"""Tagged union of every event type the loop emits."""


__all__ = [
    "Critique",
    "Event",
    "LoopError",
    "RatchetDecision",
    "Revision",
    "Role",
    "RoleEnd",
    "RoleStart",
    "RoundEnd",
    "RoundStart",
    "Token",
]
