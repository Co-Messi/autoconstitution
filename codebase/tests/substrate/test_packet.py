"""Tests for autoconstitution.substrate.packet."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from autoconstitution.substrate.packet import (
    EdgeType,
    Packet,
    PacketType,
    Provenance,
    _pseudo_embed,
    make_claim,
    make_critique,
    make_episode,
    make_fact,
    make_goal,
    make_revision,
    make_shadow,
    make_skill,
    make_verdict,
)


# ─────────────────────────────────────────────
# PacketType / EdgeType enum coverage
# ─────────────────────────────────────────────


def test_packet_type_values() -> None:
    assert PacketType.CLAIM == "claim"
    assert PacketType.SHADOW == "shadow"
    assert len(PacketType) == 9


def test_edge_type_values() -> None:
    assert EdgeType.SUPPORTS == "supports"
    assert len(EdgeType) == 6


# ─────────────────────────────────────────────
# Provenance serialisation
# ─────────────────────────────────────────────


def test_provenance_roundtrip() -> None:
    now = datetime.now(timezone.utc)
    prov = Provenance(
        source="unit-test",
        created_at=now,
        parent_ids=["abc", "def"],
        run_id="run-1",
        trace=["step1", "step2"],
    )
    d = prov.to_dict()
    prov2 = Provenance.from_dict(d)
    assert prov2.source == "unit-test"
    assert prov2.parent_ids == ["abc", "def"]
    assert prov2.run_id == "run-1"
    assert prov2.trace == ["step1", "step2"]
    assert prov2.created_at.isoformat() == now.isoformat()


# ─────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────


def test_make_claim_has_correct_type() -> None:
    p = make_claim("Why is the sky blue?")
    assert p.type == PacketType.CLAIM
    assert "sky blue" in p.content
    assert 0.0 <= p.confidence <= 1.0


def test_make_critique() -> None:
    p = make_critique("Critique: too vague.")
    assert p.type == PacketType.CRITIQUE


def test_make_revision() -> None:
    p = make_revision("Here is my revised answer.")
    assert p.type == PacketType.REVISION


def test_make_verdict() -> None:
    p = make_verdict("Pass — all tests green.")
    assert p.type == PacketType.VERDICT
    # Verdicts have no half-life (permanent)
    assert p.half_life_seconds is None


def test_make_goal() -> None:
    p = make_goal("Write a binary search implementation.")
    assert p.type == PacketType.GOAL


def test_make_fact() -> None:
    p = make_fact("The earth orbits the sun.")
    assert p.type == PacketType.FACT
    assert p.half_life_seconds is None


def test_make_skill() -> None:
    p = make_skill("Use early returns to reduce nesting.")
    assert p.type == PacketType.SKILL


def test_make_episode() -> None:
    p = make_episode("Run 42 completed in 3 rounds.")
    assert p.type == PacketType.EPISODE


def test_make_shadow_default_confidence() -> None:
    p = make_shadow("Alternative approach using recursion.")
    assert p.type == PacketType.SHADOW
    assert p.confidence == pytest.approx(0.3, abs=1e-6)


def test_make_shadow_custom_confidence() -> None:
    p = make_shadow("Another shadow.", confidence=0.15)
    assert p.confidence == pytest.approx(0.15, abs=1e-6)


# ─────────────────────────────────────────────
# UUID uniqueness
# ─────────────────────────────────────────────


def test_packets_have_unique_ids() -> None:
    ids = {make_claim("hello").id for _ in range(100)}
    assert len(ids) == 100


# ─────────────────────────────────────────────
# Pseudo-embedding
# ─────────────────────────────────────────────


def test_pseudo_embed_deterministic() -> None:
    v1 = _pseudo_embed("the sky is blue")
    v2 = _pseudo_embed("the sky is blue")
    assert v1 == v2


def test_pseudo_embed_unit_length() -> None:
    import math

    v = _pseudo_embed("test content")
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-5


def test_pseudo_embed_different_for_different_text() -> None:
    v1 = _pseudo_embed("apple")
    v2 = _pseudo_embed("banana")
    assert v1 != v2


def test_pseudo_embed_dimension() -> None:
    v = _pseudo_embed("anything")
    assert len(v) == 256


# ─────────────────────────────────────────────
# current_confidence with decay
# ─────────────────────────────────────────────


def test_confidence_no_decay() -> None:
    p = make_fact("permanent fact", confidence=0.95)
    assert p.current_confidence() == pytest.approx(0.95)


def test_confidence_decay_half_life() -> None:
    now = datetime.now(timezone.utc)
    p = make_claim("decaying claim", confidence=1.0)
    # Manually set valid_from in the past by half_life seconds
    hl = p.half_life_seconds
    assert hl is not None
    past = now - timedelta(seconds=hl)
    p.valid_from = past
    conf = p.current_confidence(now)
    # After one half-life, confidence should be ~0.5
    assert abs(conf - 0.5) < 0.01


def test_confidence_revoked_is_zero() -> None:
    p = make_claim("revoked claim", confidence=1.0)
    p.revoked_by = "some-other-packet-id"
    assert p.current_confidence() == 0.0


def test_confidence_past_valid_until_is_zero() -> None:
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    p = make_claim("expired claim", confidence=0.9, valid_until=past)
    assert p.current_confidence() == 0.0


def test_confidence_future_valid_from_unchanged() -> None:
    """A packet from the future should return its full confidence."""
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    p = make_claim("future claim", confidence=0.8)
    p.valid_from = future
    # elapsed < 0 → full confidence
    assert p.current_confidence() == pytest.approx(0.8)


# ─────────────────────────────────────────────
# Packet dict roundtrip
# ─────────────────────────────────────────────


def test_packet_dict_roundtrip() -> None:
    original = make_claim(
        "Roundtrip test claim",
        confidence=0.85,
        source="test",
        parent_ids=["parent-1"],
        run_id="run-xyz",
        trace=["step-a"],
        metadata={"key": "value"},
    )
    d = original.to_dict()
    restored = Packet.from_dict(d)

    assert restored.id == original.id
    assert restored.type == original.type
    assert restored.content == original.content
    assert restored.confidence == pytest.approx(original.confidence)
    assert restored.half_life_seconds == original.half_life_seconds
    assert restored.revoked_by is None
    assert restored.provenance.source == "test"
    assert restored.provenance.parent_ids == ["parent-1"]
    assert restored.provenance.run_id == "run-xyz"
    assert restored.provenance.trace == ["step-a"]
    assert restored.metadata["key"] == "value"
    assert restored.vector == original.vector
