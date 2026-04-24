"""Tests for autoconstitution.substrate.curriculum."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from autoconstitution.substrate.capability_self_map import (
    CapabilitySelfMap,
    Outcome,
    TaskSignature,
)
from autoconstitution.substrate.curriculum import CurriculumGenerator
from autoconstitution.substrate.manifold import Manifold
from autoconstitution.substrate.packet import PacketType


def _outcome(sig: TaskSignature, *, success: bool = False, score: float = 0.1) -> Outcome:
    return Outcome(
        signature=sig,
        success=success,
        score=score,
        rounds_used=2,
        ts=datetime.now(timezone.utc),
    )


# ─────────────────────────────────────────────
# generate
# ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_no_weak_spots(tmp_manifold: Manifold, fake_provider: object) -> None:
    """With no weak spots, generate should return an empty list."""
    csm = CapabilitySelfMap(tmp_manifold)
    gen = CurriculumGenerator(fake_provider, tmp_manifold, csm)
    ids = await gen.generate(n=3)
    assert ids == []


@pytest.mark.asyncio
async def test_generate_creates_goal_packets(tmp_manifold: Manifold, fake_provider: object) -> None:
    """With weak spots, generate should create GOAL packets in the manifold."""
    csm = CapabilitySelfMap(tmp_manifold)
    sig = TaskSignature("math", "hard", "math")
    for _ in range(3):
        csm.record(_outcome(sig, success=False, score=0.1))

    gen = CurriculumGenerator(fake_provider, tmp_manifold, csm)
    ids = await gen.generate(n=3)
    assert len(ids) >= 1

    for gid in ids:
        pkt = tmp_manifold.read(gid)
        assert pkt is not None
        assert pkt.type == PacketType.GOAL
        assert pkt.metadata.get("generated_by") == "curriculum"


@pytest.mark.asyncio
async def test_generate_with_no_provider(tmp_manifold: Manifold) -> None:
    """Without a provider, generate should use the generic fallback."""
    csm = CapabilitySelfMap(tmp_manifold)
    sig = TaskSignature("reasoning", "hard", "free")
    for _ in range(4):
        csm.record(_outcome(sig, success=False, score=0.2))

    gen = CurriculumGenerator(None, tmp_manifold, csm)
    ids = await gen.generate(n=2)
    assert len(ids) == 1  # only one weak spot
    pkt = tmp_manifold.read(ids[0])
    assert pkt is not None
    assert "reasoning" in pkt.content.lower() or "Practice" in pkt.content


# ─────────────────────────────────────────────
# next_practice
# ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_next_practice_empty(tmp_manifold: Manifold) -> None:
    csm = CapabilitySelfMap(tmp_manifold)
    gen = CurriculumGenerator(None, tmp_manifold, csm)
    assert gen.next_practice() is None


@pytest.mark.asyncio
async def test_next_practice_returns_oldest_goal(tmp_manifold: Manifold) -> None:
    """next_practice should return the oldest unresolved GOAL."""
    csm = CapabilitySelfMap(tmp_manifold)
    sig = TaskSignature("code", "easy", "code")
    for _ in range(3):
        csm.record(_outcome(sig, success=False, score=0.1))

    gen = CurriculumGenerator(None, tmp_manifold, csm)
    ids = await gen.generate(n=2)
    assert len(ids) >= 1

    practice = gen.next_practice()
    assert practice is not None
    assert practice.type == PacketType.GOAL


# ─────────────────────────────────────────────
# mark_resolved
# ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mark_resolved_revokes_goal(tmp_manifold: Manifold) -> None:
    csm = CapabilitySelfMap(tmp_manifold)
    sig = TaskSignature("code", "medium", "code")
    for _ in range(3):
        csm.record(_outcome(sig, success=False, score=0.2))

    gen = CurriculumGenerator(None, tmp_manifold, csm)
    ids = await gen.generate(n=1)
    goal_id = ids[0]

    gen.mark_resolved(goal_id, outcome="scored 0.85")

    # Goal should now be excluded from active query
    active_goals = tmp_manifold.query(type=PacketType.GOAL)
    active_ids = [p.id for p in active_goals]
    assert goal_id not in active_ids


@pytest.mark.asyncio
async def test_mark_resolved_creates_verdict(tmp_manifold: Manifold) -> None:
    csm = CapabilitySelfMap(tmp_manifold)
    sig = TaskSignature("code", "medium", "code")
    for _ in range(3):
        csm.record(_outcome(sig, success=False, score=0.2))

    gen = CurriculumGenerator(None, tmp_manifold, csm)
    ids = await gen.generate(n=1)
    goal_id = ids[0]
    gen.mark_resolved(goal_id, outcome="pass")

    verdicts = tmp_manifold.query(type=PacketType.VERDICT)
    assert len(verdicts) >= 1
    v = verdicts[0]
    assert v.metadata.get("resolved_goal_id") == goal_id
