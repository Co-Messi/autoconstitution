"""
autoconstitution.substrate.skill_compiler
==========================================

Skill compiler: converts successful critique/revise traces into reusable
templated skills, quarantines failing skills.

MANIFOLD §2 feature 5 ("Skill deprecation"): "The skill compiler tracks each
compiled skill's usage and success. When environment drift causes a skill to
fail consistently, the skill is quarantined, not silently reused."

MANIFOLD §3.4: "Procedural graph: compiled skills, tool recipes, partial
programs, option-like policies." Writes to the procedural sub-graph of the
Manifold.

MANIFOLD §6: "Successful traces are distilled back into the skill compiler."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from autoconstitution.substrate.capability_self_map import TaskSignature
from autoconstitution.substrate.manifold import Manifold
from autoconstitution.substrate.packet import (
    Packet,
    PacketType,
    make_skill,
)

logger = logging.getLogger(__name__)

# Quarantine threshold: if rolling success rate over last N uses < this, quarantine.
_QUARANTINE_THRESHOLD = 0.3
_QUARANTINE_WINDOW = 5  # uses to look back


# ─────────────────────────────────────────────
# Skill dataclass
# ─────────────────────────────────────────────


@dataclass
class Skill:
    """A compiled reusable transformation extracted from a successful trace.

    MANIFOLD §3.4 procedural graph: compiled skills are stored as SKILL packets
    in the Manifold so they persist across sessions and can be retrieved by
    task signature.

    Attributes:
        id:                    UUID string matching the underlying SKILL packet.
        trigger_signature:     The TaskSignature this skill applies to.
        transformation_prompt: The templated prompt/instruction encoding the
                               lesson from the successful trace.
        success_rate:          Historical success rate across all uses.
        uses:                  Total use count.
        last_used_at:          UTC timestamp of most recent use.
        quarantined:           True if skill is suspended due to poor recent
                               performance.
    """

    id: str
    trigger_signature: TaskSignature
    transformation_prompt: str
    success_rate: float
    uses: int
    last_used_at: datetime
    quarantined: bool


# ─────────────────────────────────────────────
# SkillCompiler
# ─────────────────────────────────────────────


class SkillCompiler:
    """Extracts and manages reusable skills from critique/revise traces.

    MANIFOLD §3.4 and §6: successful revision trajectories are compiled into
    templated skills; stale or failing skills are quarantined (feature 5).

    Args:
        manifold: The backing Manifold for SKILL packet storage.
        provider: Optional LLM provider for lesson extraction. When None,
                  falls back to extracting the last critique content.
    """

    def __init__(
        self,
        manifold: Manifold,
        provider: Any | None = None,
    ) -> None:
        self._m = manifold
        self._provider = provider

    # ------------------------------------------------------------------
    # Compile from trace
    # ------------------------------------------------------------------

    async def compile_from_trace(
        self,
        trace_packets: list[Packet],
        signature: TaskSignature | None = None,
    ) -> Skill | None:
        """Attempt to compile a skill from a list of trace packets.

        A trace is "compilable" if it contains at least one CRITIQUE followed
        by a REVISION — i.e., the system identified a problem and fixed it.

        MANIFOLD §6: "Successful traces are distilled back into the skill
        compiler."

        Args:
            trace_packets: Ordered list of packets from a critique/revise run
                           (CLAIM, CRITIQUE, REVISION, VERDICT, etc.).
            signature:     Optional TaskSignature override. If None, inferred
                           from trace metadata where possible.

        Returns:
            A Skill object (and persisted SKILL packet), or None if the trace
            doesn't meet the compilable threshold.
        """
        critiques = [p for p in trace_packets if p.type == PacketType.CRITIQUE]
        revisions = [p for p in trace_packets if p.type == PacketType.REVISION]

        if not critiques or not revisions:
            logger.debug("compile_from_trace: no critique→revision pattern; skipping")
            return None

        lesson = await self._extract_lesson(critiques, revisions)

        sig = signature or TaskSignature(domain="general", difficulty="medium", kind="free")

        skill_content = f"Skill [{sig.domain}/{sig.difficulty}/{sig.kind}]: {lesson}"
        now = datetime.now(timezone.utc)

        skill_packet = make_skill(
            skill_content,
            source="skill_compiler",
            metadata={
                "trigger_signature": sig.to_dict(),
                "transformation_prompt": lesson,
                "success_rate": 1.0,
                "uses": 0,
                "quarantined": False,
                "use_history": [],  # list of bool (True=success)
                "last_used_at": now.isoformat(),
            },
        )
        self._m.write(skill_packet)

        # Link to the trace packets
        from autoconstitution.substrate.packet import EdgeType
        for p in trace_packets:
            self._m.add_edge(skill_packet.id, p.id, EdgeType.DERIVES_FROM)

        logger.debug("compiled skill %s: %s", skill_packet.id, lesson[:60])

        return Skill(
            id=skill_packet.id,
            trigger_signature=sig,
            transformation_prompt=lesson,
            success_rate=1.0,
            uses=0,
            last_used_at=now,
            quarantined=False,
        )

    async def _extract_lesson(
        self, critiques: list[Packet], revisions: list[Packet]
    ) -> str:
        """Ask provider to extract the lesson; fallback to last critique content."""
        if self._provider is None:
            return self._fallback_lesson(critiques, revisions)
        try:
            last_critique = critiques[-1].content
            last_revision = revisions[-1].content
            prompt = (
                f"extract the lesson from this revision pattern as a single concise "
                f"actionable rule (one sentence). "
                f"Critique: {last_critique[:300]} "
                f"Revised output: {last_revision[:300]}"
            )
            response = await self._provider.complete(prompt)
            return response.strip() or self._fallback_lesson(critiques, revisions)
        except Exception as exc:
            logger.warning("skill lesson extraction failed: %s", exc)
            return self._fallback_lesson(critiques, revisions)

    @staticmethod
    def _fallback_lesson(critiques: list[Packet], revisions: list[Packet]) -> str:
        """Use last critique content as the lesson when no provider."""
        last_critique = critiques[-1].content
        # Trim to a single sentence
        first_sentence = last_critique.split(".")[0]
        return f"Lesson from critique: {first_sentence.strip()}"

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, signature: TaskSignature) -> list[Skill]:
        """Return active (non-quarantined) skills matching kind and domain.

        MANIFOLD §3.4: skills are retrieved by "skill retrieval" mode.
        """
        skill_packets = self._m.query(type=PacketType.SKILL)
        results: list[Skill] = []
        for pkt in skill_packets:
            meta = pkt.metadata
            if meta.get("quarantined", False):
                continue
            sig_dict = meta.get("trigger_signature", {})
            # Match on domain + kind (difficulty is a hint, not hard constraint)
            if (
                sig_dict.get("domain") == signature.domain
                and sig_dict.get("kind") == signature.kind
            ):
                results.append(self._packet_to_skill(pkt))
        return results

    # ------------------------------------------------------------------
    # Update stats
    # ------------------------------------------------------------------

    def update_stats(self, skill_id: str, success: bool) -> None:
        """Record the result of using a skill and quarantine if poor performer.

        MANIFOLD §2 feature 5: "When environment drift causes a skill to fail
        consistently, the skill is quarantined, not silently reused."

        Quarantine is triggered if the last _QUARANTINE_WINDOW uses have a
        success rate below _QUARANTINE_THRESHOLD.
        """
        import json

        row = self._m._conn.execute(
            "SELECT metadata_json FROM packets WHERE id = ?", (skill_id,)
        ).fetchone()
        if row is None:
            logger.warning("update_stats: skill %s not found", skill_id)
            return

        meta = json.loads(row[0])
        use_history: list[bool] = meta.get("use_history", [])
        use_history.append(success)
        uses = meta.get("uses", 0) + 1
        success_rate = sum(use_history) / len(use_history)

        # Rolling quarantine check over last window
        recent = use_history[-_QUARANTINE_WINDOW:]
        quarantined = (
            len(recent) >= _QUARANTINE_WINDOW
            and (sum(recent) / len(recent)) < _QUARANTINE_THRESHOLD
        )

        if quarantined and not meta.get("quarantined", False):
            logger.warning("skill %s quarantined: recent success rate %.2f", skill_id, sum(recent) / len(recent))

        meta.update(
            {
                "uses": uses,
                "success_rate": success_rate,
                "use_history": use_history,
                "quarantined": quarantined,
                "last_used_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        with self._m._conn:
            self._m._conn.execute(
                "UPDATE packets SET metadata_json = ? WHERE id = ?",
                (json.dumps(meta), skill_id),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _packet_to_skill(pkt: Packet) -> Skill:
        meta = pkt.metadata
        last_used_str = meta.get("last_used_at", pkt.valid_from.isoformat())
        last_used = datetime.fromisoformat(last_used_str)
        if last_used.tzinfo is None:
            last_used = last_used.replace(tzinfo=timezone.utc)
        sig_dict = meta.get("trigger_signature", {})
        sig = TaskSignature.from_dict(sig_dict) if sig_dict else TaskSignature("general", "medium", "free")
        return Skill(
            id=pkt.id,
            trigger_signature=sig,
            transformation_prompt=meta.get("transformation_prompt", pkt.content),
            success_rate=meta.get("success_rate", 1.0),
            uses=meta.get("uses", 0),
            last_used_at=last_used,
            quarantined=meta.get("quarantined", False),
        )


__all__ = ["Skill", "SkillCompiler"]
