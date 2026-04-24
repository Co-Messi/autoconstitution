"""
autoconstitution.substrate.shadow_validator
=============================================

Two-phase adaptation with rollback: validates candidate skills against
protected tasks before committing; rolls back on regression.

MANIFOLD §2 feature 6 ("Two-phase adaptation with rollback"): "Any parameter
update (online synaptic consolidation, skill commit, memory promotion) runs in
a shadow copy first. The shadow is validated against held-out tasks drawn from
the capability self-map. If it degrades performance on any protected capability,
the update is rolled back. This is a circuit breaker against catastrophic
forgetting and adversarial data poisoning."

MANIFOLD §3.7 (Metacognitive Controller): "the only subsystem authorized to
commit synaptic updates to the backbone" — analogous role here for skill commits.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from autoconstitution.substrate.manifold import Manifold
from autoconstitution.substrate.packet import PacketType
from autoconstitution.substrate.skill_compiler import Skill

logger = logging.getLogger(__name__)


class ShadowValidator:
    """Validates candidate skills against protected tasks before committing.

    The two-phase protocol:
        1. validate(candidate) — run the candidate against protected_tasks,
           compare each result to baseline stored in manifold.
        2a. commit(candidate) — mark skill un-quarantined in the Manifold.
        2b. rollback(candidate, reason) — revoke the SKILL packet.

    MANIFOLD §2 feature 6: the shadow copy runs first; commit only happens
    if all protected tasks remain at or above their baseline performance.

    Args:
        manifold:        The backing Manifold instance.
        protected_tasks: Mapping of task_id → baseline_score (float 0..1).
                         Tasks whose performance must not regress.
    """

    def __init__(
        self,
        manifold: Manifold,
        protected_tasks: dict[str, float] | None = None,
    ) -> None:
        self._m = manifold
        self._protected = protected_tasks or {}

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    async def validate(
        self,
        candidate_skill: Skill,
        run_fn: Callable[[str, str], Awaitable[float]],
    ) -> tuple[bool, dict[str, Any]]:
        """Run candidate against protected tasks; compare to baselines.

        MANIFOLD §2 feature 6: "The shadow is validated against held-out
        tasks drawn from the capability self-map."

        Args:
            candidate_skill: The skill to validate (still quarantined).
            run_fn:          Async callable ``(task_id, skill_prompt) -> score``.
                             score should be in [0, 1].

        Returns:
            ``(passed, report)``
            - passed: True iff every protected task's score >= baseline.
            - report: Per-task scores and deltas.
        """
        report: dict[str, Any] = {
            "skill_id": candidate_skill.id,
            "tasks": {},
            "passed": True,
            "regressions": [],
        }

        for task_id, baseline in self._protected.items():
            try:
                score = await run_fn(task_id, candidate_skill.transformation_prompt)
            except Exception as exc:
                logger.warning("shadow validate task %s raised: %s", task_id, exc)
                score = 0.0

            delta = score - baseline
            regressed = score < baseline
            report["tasks"][task_id] = {
                "baseline": baseline,
                "score": score,
                "delta": delta,
                "regressed": regressed,
            }
            if regressed:
                report["passed"] = False
                report["regressions"].append(task_id)
                logger.warning(
                    "shadow validate: task %s regressed %.3f → %.3f (delta=%.3f)",
                    task_id, baseline, score, delta,
                )

        logger.debug(
            "shadow validate skill %s: passed=%s regressions=%s",
            candidate_skill.id, report["passed"], report["regressions"],
        )
        return bool(report["passed"]), report

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def commit(self, candidate_skill: Skill) -> None:
        """Mark skill as active (un-quarantined) in the Manifold.

        MANIFOLD §2 feature 6: after shadow validation passes, the update
        is committed. Only ShadowValidator may commit — all other code
        creates skills quarantined=True.
        """
        import json

        row = self._m._conn.execute(
            "SELECT metadata_json FROM packets WHERE id = ?",
            (candidate_skill.id,),
        ).fetchone()
        if row is None:
            logger.warning("commit: skill packet %s not found", candidate_skill.id)
            return

        meta = json.loads(row[0])
        meta["quarantined"] = False

        with self._m._conn:
            self._m._conn.execute(
                "UPDATE packets SET metadata_json = ? WHERE id = ?",
                (json.dumps(meta), candidate_skill.id),
            )
        logger.debug("committed skill %s", candidate_skill.id)

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self, candidate_skill: Skill, reason: str) -> None:
        """Revoke the candidate SKILL packet — rollback on regression.

        MANIFOLD §2 feature 6: "If it degrades performance on any protected
        capability, the update is rolled back."
        """
        self._m.revoke(candidate_skill.id, reason=reason)
        logger.warning("rolled back skill %s: %s", candidate_skill.id, reason)


__all__ = ["ShadowValidator"]
