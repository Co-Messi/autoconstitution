"""
autoconstitution.substrate.curriculum
=======================================

CurriculumGenerator: meta-controller component that synthesizes targeted
practice problems for weak capability spots, closes the self-improvement loop.

MANIFOLD §2 feature 8 ("Curriculum self-generation"): "The meta-controller
monitors the capability self-map for weak spots and synthesizes practice
problems targeting them. Successful solutions feed the skill compiler;
failures refine the self-map. This closes the self-improvement loop
internally, without requiring an external training corpus."
"""

from __future__ import annotations

import logging
from typing import Any

from autoconstitution.substrate.capability_self_map import (
    CapabilitySelfMap,
    TaskSignature,
)
from autoconstitution.substrate.manifold import Manifold
from autoconstitution.substrate.packet import (
    PacketType,
    make_goal,
    make_verdict,
)

logger = logging.getLogger(__name__)


class CurriculumGenerator:
    """Generates GOAL packets targeting weak capability regions.

    Works with or without a provider: when no provider is available (or on
    failure), falls back to a generic practice string derived from the
    task signature.

    MANIFOLD §3.7 (Metacognitive Controller): "manages the skill compiler
    and curriculum generator (feature 8)."

    Args:
        provider:  Optional LLM provider (must support ``complete(prompt)``).
        manifold:  Backing Manifold for persisting GOAL packets.
        self_map:  The CapabilitySelfMap to query for weak spots.
    """

    def __init__(
        self,
        provider: Any | None,
        manifold: Manifold,
        self_map: CapabilitySelfMap,
    ) -> None:
        self._provider = provider
        self._m = manifold
        self._self_map = self_map

    # ------------------------------------------------------------------
    # Generate practice goals
    # ------------------------------------------------------------------

    async def generate(self, n: int = 5) -> list[str]:
        """Generate up to n GOAL packets targeting weak capability spots.

        For each weak spot, asks the provider to synthesize a targeted
        practice problem. On provider failure or missing provider, uses a
        generic template. Creates GOAL packets in the Manifold.

        Returns:
            List of created GOAL packet ids.
        """
        weak = self._self_map.weak_spots()
        if not weak:
            logger.debug("curriculum.generate: no weak spots found")
            return []

        targets = weak[:n]
        created_ids: list[str] = []
        for sig in targets:
            content = await self._synthesize_problem(sig)
            # Find parent FACT packets for this signature
            parent_ids = self._find_outcome_packet_ids(sig)
            goal = make_goal(
                content,
                source="curriculum_generator",
                parent_ids=parent_ids,
                metadata={"signature": sig.to_dict(), "generated_by": "curriculum"},
            )
            self._m.write(goal)
            created_ids.append(goal.id)
            logger.debug("curriculum goal created: %s for %s", goal.id, sig)

        return created_ids

    async def _synthesize_problem(self, sig: TaskSignature) -> str:
        """Ask provider to synthesize a targeted practice problem, with fallback."""
        if self._provider is None:
            return self._generic_problem(sig)
        try:
            prompt = (
                f"synthesize a targeted practice problem for a student who struggles "
                f"with {sig.kind} tasks in the {sig.domain} domain at {sig.difficulty} "
                f"difficulty. Return only the problem statement, no preamble."
            )
            response = await self._provider.complete(prompt)
            return response.strip() or self._generic_problem(sig)
        except Exception as exc:
            logger.warning("curriculum synthesis failed: %s — using fallback", exc)
            return self._generic_problem(sig)

    @staticmethod
    def _generic_problem(sig: TaskSignature) -> str:
        return (
            f"Practice problem [{sig.difficulty}] {sig.domain}/{sig.kind}: "
            f"Demonstrate your ability to perform a {sig.difficulty}-level "
            f"{sig.kind} task in the {sig.domain} domain."
        )

    # ------------------------------------------------------------------
    # Next practice
    # ------------------------------------------------------------------

    def next_practice(self) -> Any | None:
        """Return the oldest unresolved GOAL packet, or None.

        "Unresolved" means not revoked. The oldest is determined by valid_from.
        """
        goals = self._m.query(type=PacketType.GOAL)
        if not goals:
            return None
        # sort ascending by valid_from (oldest first)
        return min(goals, key=lambda p: p.valid_from)

    # ------------------------------------------------------------------
    # Mark resolved
    # ------------------------------------------------------------------

    def mark_resolved(self, goal_id: str, outcome: Any) -> None:
        """Revoke a GOAL packet and link it to a resolving VERDICT.

        MANIFOLD §2 feature 8: successful solutions feed back to the system.
        Creates a VERDICT packet and adds a SUPERSEDES edge from the verdict
        to the goal.

        Args:
            goal_id: The GOAL packet to mark done.
            outcome: A dict or string describing the resolution outcome.
        """
        from autoconstitution.substrate.packet import EdgeType

        verdict_content = (
            str(outcome) if not isinstance(outcome, str) else outcome
        )
        verdict = make_verdict(
            f"Goal resolved: {verdict_content}",
            source="curriculum_generator",
            parent_ids=[goal_id],
            metadata={"resolved_goal_id": goal_id, "outcome": str(outcome)},
        )
        self._m.write(verdict)
        self._m.revoke(goal_id, reason=f"resolved by verdict {verdict.id}", by_packet_id=verdict.id)
        self._m.add_edge(verdict.id, goal_id, EdgeType.SUPERSEDES)
        logger.debug("goal %s resolved by verdict %s", goal_id, verdict.id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_outcome_packet_ids(self, sig: TaskSignature) -> list[str]:
        """Return ids of FACT outcome packets matching the signature."""
        packets = self._m.query(type=None)
        ids: list[str] = []
        for p in packets:
            meta = p.metadata
            if meta.get("kind") != "outcome":
                continue
            sd = meta.get("signature", {})
            if (
                sd.get("domain") == sig.domain
                and sd.get("difficulty") == sig.difficulty
                and sd.get("kind") == sig.kind
            ):
                ids.append(p.id)
        return ids


__all__ = ["CurriculumGenerator"]
