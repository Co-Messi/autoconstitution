"""
autoconstitution.substrate.capability_self_map
================================================

CapabilitySelfMap: the meta-controller's explicit model of what the system
can and cannot reliably do, updated from success and failure statistics.

MANIFOLD §2 feature 3 ("Capability self-map"): "The meta-controller maintains
an explicit typed model of what the system can and can't reliably do, updated
from success and failure statistics. When asked to do something outside the
map, it does not hallucinate competence — it escalates to tool use, requests
clarification, or declines with a calibrated explanation."

MANIFOLD §3.7 ("Metacognitive Controller"): "Maintains the capability self-map,
allocates compute across the other subsystems, decides when to escalate."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from autoconstitution.substrate.manifold import Manifold
from autoconstitution.substrate.packet import make_fact

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────


@dataclass
class TaskSignature:
    """A typed key identifying a class of tasks.

    Used to index into the capability self-map so the system knows its
    historical win/loss record for tasks of this shape.

    Attributes:
        domain:     High-level domain (e.g. "math", "code", "reasoning").
        difficulty: Rough difficulty level ("easy", "medium", "hard").
        kind:       Fine-grained task kind ("code", "math", "reasoning",
                    "extraction", "free").
    """

    domain: str
    difficulty: str  # easy | medium | hard
    kind: str  # code | math | reasoning | extraction | free

    def __hash__(self) -> int:
        return hash((self.domain, self.difficulty, self.kind))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaskSignature):
            return NotImplemented
        return (
            self.domain == other.domain
            and self.difficulty == other.difficulty
            and self.kind == other.kind
        )

    def to_dict(self) -> dict[str, str]:
        return {"domain": self.domain, "difficulty": self.difficulty, "kind": self.kind}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "TaskSignature":
        return cls(domain=d["domain"], difficulty=d["difficulty"], kind=d["kind"])


@dataclass
class Outcome:
    """A single task outcome recorded in the self-map.

    Attributes:
        signature:   Which class of task this outcome is for.
        success:     Whether the task was completed successfully.
        score:       Numeric quality score in [0, 1].
        rounds_used: Number of critique/revise rounds consumed.
        ts:          UTC timestamp of the outcome.
    """

    signature: TaskSignature
    success: bool
    score: float
    rounds_used: int
    ts: datetime


@dataclass
class TrackRecord:
    """Aggregated statistics for a single TaskSignature.

    Attributes:
        n:           Number of attempts.
        success_rate: Fraction that succeeded.
        avg_rounds:  Average rounds used.
        last_seen:   Most recent attempt timestamp.
        ewma_score:  Exponentially weighted moving average of score (alpha=0.3).
    """

    n: int
    success_rate: float
    avg_rounds: float
    last_seen: datetime
    ewma_score: float


# ─────────────────────────────────────────────
# CapabilitySelfMap
# ─────────────────────────────────────────────


class CapabilitySelfMap:
    """Persistent capability self-model backed by the Manifold.

    Records Outcome objects as FACT packets tagged with kind="outcome", then
    aggregates them on demand to answer competence and calibration queries.

    MANIFOLD §2 feature 3 and §3.7: the metacognitive controller uses this
    to allocate compute (weak signatures get more rounds), detect blind spots
    (blank regions), and generate curriculum practice (via CurriculumGenerator).

    Args:
        manifold: The Manifold instance to persist outcomes into.
    """

    _EWMA_ALPHA = 0.3  # smoothing factor for EWMA score

    def __init__(self, manifold: Manifold) -> None:
        self._m = manifold

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    def record(self, outcome: Outcome) -> None:
        """Persist an Outcome as a FACT packet in the Manifold."""
        content = (
            f"Outcome: domain={outcome.signature.domain} "
            f"difficulty={outcome.signature.difficulty} "
            f"kind={outcome.signature.kind} "
            f"success={outcome.success} "
            f"score={outcome.score:.4f} "
            f"rounds={outcome.rounds_used}"
        )
        packet = make_fact(
            content,
            metadata={
                "kind": "outcome",
                "signature": outcome.signature.to_dict(),
                "success": outcome.success,
                "score": outcome.score,
                "rounds_used": outcome.rounds_used,
                "ts": outcome.ts.isoformat(),
            },
        )
        self._m.write(packet)
        logger.debug("recorded outcome: %s success=%s score=%.3f", outcome.signature, outcome.success, outcome.score)

    # ------------------------------------------------------------------
    # Track record
    # ------------------------------------------------------------------

    def track_record(self, signature: TaskSignature) -> TrackRecord | None:
        """Aggregate all recorded outcomes for the given signature.

        Returns None if no outcomes have been recorded for this signature.
        The ewma_score is computed by processing outcomes in chronological
        order with alpha=0.3 (more weight on recent outcomes).
        """
        outcomes = self._load_outcomes(signature)
        if not outcomes:
            return None

        n = len(outcomes)
        successes = sum(1 for o in outcomes if o.success)
        rounds_list = [o.rounds_used for o in outcomes]
        last_seen = max(o.ts for o in outcomes)

        # EWMA score — process in chronological order
        sorted_outcomes = sorted(outcomes, key=lambda o: o.ts)
        ewma = sorted_outcomes[0].score
        for o in sorted_outcomes[1:]:
            ewma = self._EWMA_ALPHA * o.score + (1 - self._EWMA_ALPHA) * ewma

        return TrackRecord(
            n=n,
            success_rate=successes / n,
            avg_rounds=sum(rounds_list) / n,
            last_seen=last_seen,
            ewma_score=ewma,
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, signature: TaskSignature) -> tuple[float, float]:
        """Return (competence, calibration) for a task signature.

        competence:  EWMA score in [0, 1]. 0.5 if never tried.
        calibration: Confidence in the competence estimate, saturates at 1.0
                     after min(n, 5) = 5 samples (MANIFOLD §2 feature 3).

        Returns:
            (competence, calibration) both in [0, 1].
        """
        record = self.track_record(signature)
        if record is None:
            return 0.5, 0.0  # no data → maximum uncertainty
        calibration = min(1.0, record.n / 5)
        return record.ewma_score, calibration

    # ------------------------------------------------------------------
    # Weak spots
    # ------------------------------------------------------------------

    def weak_spots(
        self, threshold: float = 0.4, min_n: int = 3
    ) -> list[TaskSignature]:
        """Return signatures with ewma_score below threshold and >= min_n attempts.

        MANIFOLD §2 feature 8 ("Curriculum self-generation"): "The meta-
        controller monitors the capability self-map for weak spots and
        synthesizes practice problems targeting them."
        """
        all_sigs = self._all_signatures()
        weak: list[TaskSignature] = []
        for sig in all_sigs:
            record = self.track_record(sig)
            if record and record.n >= min_n and record.ewma_score < threshold:
                weak.append(sig)
        return weak

    # ------------------------------------------------------------------
    # Blank regions
    # ------------------------------------------------------------------

    def blank_regions(
        self, all_signatures: list[TaskSignature]
    ) -> list[TaskSignature]:
        """Return signatures that have never been attempted.

        MANIFOLD §7 open problem 6: "The self-map is built from success/failure
        on attempted tasks; novel out-of-distribution regions remain blank."
        This method makes those blank regions explicit.
        """
        tried = self._all_signatures()
        return [s for s in all_signatures if s not in tried]

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a dict table of all known signatures and their stats."""
        all_sigs = self._all_signatures()
        rows: list[dict[str, Any]] = []
        for sig in all_sigs:
            record = self.track_record(sig)
            if record:
                rows.append(
                    {
                        "domain": sig.domain,
                        "difficulty": sig.difficulty,
                        "kind": sig.kind,
                        "n": record.n,
                        "success_rate": round(record.success_rate, 3),
                        "ewma_score": round(record.ewma_score, 3),
                        "avg_rounds": round(record.avg_rounds, 2),
                        "last_seen": record.last_seen.isoformat(),
                    }
                )
        return {"signatures": rows, "total": len(rows)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_outcomes(self, signature: TaskSignature) -> list[Outcome]:
        """Load all outcome FACT packets matching the given signature."""
        packets = self._m.query(type=None)  # query all non-revoked
        results: list[Outcome] = []
        for p in packets:
            meta = p.metadata
            if meta.get("kind") != "outcome":
                continue
            sig_dict = meta.get("signature", {})
            if (
                sig_dict.get("domain") == signature.domain
                and sig_dict.get("difficulty") == signature.difficulty
                and sig_dict.get("kind") == signature.kind
            ):
                ts_str = meta.get("ts", p.valid_from.isoformat())
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                results.append(
                    Outcome(
                        signature=signature,
                        success=bool(meta.get("success", False)),
                        score=float(meta.get("score", 0.0)),
                        rounds_used=int(meta.get("rounds_used", 1)),
                        ts=ts,
                    )
                )
        return results

    def _all_signatures(self) -> list[TaskSignature]:
        """Return the set of unique signatures that have outcome records."""
        packets = self._m.query(type=None)
        seen: set[TaskSignature] = set()
        for p in packets:
            meta = p.metadata
            if meta.get("kind") != "outcome":
                continue
            sig_dict = meta.get("signature", {})
            sig = TaskSignature.from_dict(sig_dict)
            seen.add(sig)
        return list(seen)


__all__ = [
    "TaskSignature",
    "Outcome",
    "TrackRecord",
    "CapabilitySelfMap",
]
