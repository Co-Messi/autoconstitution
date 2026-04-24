"""
autoconstitution.substrate.packet
==================================

Typed latent state packets: the universal message format on MANIFOLD's shared
substrate. Every piece of information the substrate stores (claims, critiques,
revisions, verdicts, facts, skills, shadow alternatives) is a Packet with
provenance, confidence, temporal validity, and a learned decay function.

This maps to MANIFOLD §3.3 "Persistent Causal State Graph" and §4 "The shared
substrate" — specifically the typed packet format with content vector, type tag,
provenance pointer, confidence, temporal validity, decay rate, and revocation
link.
"""

from __future__ import annotations

import hashlib
import math
import struct
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────


class PacketType(str, Enum):
    """Typed tags that identify a packet's semantic role on the manifold.

    MANIFOLD §3.1 specifies entities, relations, actions, observations,
    affordances, uncertainty, and provenance tags. We flatten these into
    the packet types most relevant to the autoconstitution critique/revise loop.
    """

    CLAIM = "claim"
    CRITIQUE = "critique"
    REVISION = "revision"
    VERDICT = "verdict"
    GOAL = "goal"
    FACT = "fact"
    EPISODE = "episode"
    SKILL = "skill"
    SHADOW = "shadow"


class EdgeType(str, Enum):
    """Typed causal edges between packets in the state graph.

    MANIFOLD §3.3: "Every node and edge carries a confidence estimate, a
    provenance pointer, a temporal validity window, and a learned decay rate."
    """

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DERIVES_FROM = "derives_from"
    REVISES = "revises"
    SUPERSEDES = "supersedes"
    CITES = "cites"


# ─────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────


@dataclass
class Provenance:
    """Tracks the origin and lineage of every packet.

    MANIFOLD §4: "Provenance pointer (back to source: raw stream timestamp,
    memory node ID, inference trace)."

    Attributes:
        source:     Human-readable source label (e.g. "cai_loop", "user").
        created_at: Wall-clock UTC instant this provenance was recorded.
        parent_ids: IDs of packets this one was derived from.
        run_id:     Opaque string tying this packet to a specific execution run.
        trace:      Free-form audit trail entries (most recent last).
    """

    source: str
    created_at: datetime
    parent_ids: list[str] = field(default_factory=list)
    run_id: str | None = None
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "parent_ids": self.parent_ids,
            "run_id": self.run_id,
            "trace": self.trace,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Provenance":
        return cls(
            source=d["source"],
            created_at=datetime.fromisoformat(d["created_at"]),
            parent_ids=d.get("parent_ids", []),
            run_id=d.get("run_id"),
            trace=d.get("trace", []),
        )


# ─────────────────────────────────────────────
# Packet
# ─────────────────────────────────────────────

# Default half-lives by type (seconds) — tuned per MANIFOLD's belief half-life
# feature: "facts about fast-changing state decay quickly; facts about slow-
# changing state decay slowly or not at all." (MANIFOLD §2, feature 1)
_DEFAULT_HALF_LIFE: dict[PacketType, float | None] = {
    PacketType.CLAIM: 86_400,       # 1 day
    PacketType.CRITIQUE: 43_200,    # 12 hours
    PacketType.REVISION: 86_400,    # 1 day
    PacketType.VERDICT: None,       # permanent — a verdict is a verdict
    PacketType.GOAL: 604_800,       # 1 week
    PacketType.FACT: None,          # facts don't decay by default
    PacketType.EPISODE: None,       # episodes are archival
    PacketType.SKILL: None,         # skills persist until quarantined
    PacketType.SHADOW: 300,         # 5 minutes — shadows evaporate quickly
}


@dataclass
class Packet:
    """A typed latent state packet on the shared manifold.

    Maps directly to MANIFOLD §4's per-packet attributes: content vector, type
    tag, provenance pointer, confidence, temporal validity, decay rate, and
    revocation link. The ``vector`` is the semantic embedding; if not provided, a
    deterministic pseudo-embedding is computed from the content hash.

    Attributes:
        id:               UUID string, stable across writes.
        type:             Semantic role on the manifold.
        content:          Natural-language or structured string payload.
        vector:           256-d unit-normalized embedding (or None to use hash).
        confidence:       Calibrated probability in [0, 1].
        valid_from:       Earliest instant this packet is valid.
        valid_until:      Optional expiry (None = no expiry).
        half_life_seconds: Optional exponential decay constant in seconds.
        revoked_by:       ID of the superseding packet, or None.
        provenance:       Source, parent chain, run ID, trace.
        metadata:         Arbitrary key/value annotations.
    """

    id: str
    type: PacketType
    content: str
    vector: list[float] | None
    confidence: float
    valid_from: datetime
    valid_until: datetime | None
    half_life_seconds: float | None
    revoked_by: str | None
    provenance: Provenance
    metadata: dict[str, Any]

    # ------------------------------------------------------------------
    # Confidence with exponential decay
    # ------------------------------------------------------------------

    def current_confidence(self, now: datetime | None = None) -> float:
        """Return the effective confidence after applying decay and validity checks.

        MANIFOLD §2 feature 1 ("Belief half-life"): "Facts about fast-changing
        state decay quickly; new evidence refreshes the decay clock."

        Returns 0.0 if the packet is revoked or past its valid_until window.
        Applies ``conf * 0.5^(elapsed / half_life)`` if half_life is set.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        if self.revoked_by is not None:
            return 0.0

        if self.valid_until is not None and now > self.valid_until:
            return 0.0

        if self.half_life_seconds is None:
            return self.confidence

        elapsed = (now - self.valid_from).total_seconds()
        if elapsed < 0:
            return self.confidence  # packet from the future — full confidence

        decay_factor = 0.5 ** (elapsed / self.half_life_seconds)
        return self.confidence * decay_factor

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary representation."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "vector": self.vector,
            "confidence": self.confidence,
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "half_life_seconds": self.half_life_seconds,
            "revoked_by": self.revoked_by,
            "provenance": self.provenance.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Packet":
        """Reconstruct a Packet from its dictionary representation."""
        return cls(
            id=d["id"],
            type=PacketType(d["type"]),
            content=d["content"],
            vector=d.get("vector"),
            confidence=d["confidence"],
            valid_from=datetime.fromisoformat(d["valid_from"]),
            valid_until=(
                datetime.fromisoformat(d["valid_until"]) if d.get("valid_until") else None
            ),
            half_life_seconds=d.get("half_life_seconds"),
            revoked_by=d.get("revoked_by"),
            provenance=Provenance.from_dict(d["provenance"]),
            metadata=d.get("metadata", {}),
        )


# ─────────────────────────────────────────────
# Pseudo-embedding
# ─────────────────────────────────────────────

_EMBED_DIM = 256


def _pseudo_embed(text: str) -> list[float]:
    """Deterministic 256-d unit-normalised pseudo-embedding from content hash.

    Uses SHA-256 of the UTF-8 text repeated over 256 floats via struct
    unpacking, then L2-normalises. Bit-reproducible across runs; no network
    or model required. Used when no real embedder is wired in.
    """
    seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
    # Repeat seed bytes to fill 256 floats (4 bytes each = 1024 bytes total).
    raw_bytes = (seed_bytes * math.ceil(_EMBED_DIM * 4 / len(seed_bytes)))[: _EMBED_DIM * 4]
    floats = list(struct.unpack(f"{_EMBED_DIM}f", raw_bytes))
    # Normalize to unit sphere.
    norm = math.sqrt(sum(x * x for x in floats)) or 1.0
    return [x / norm for x in floats]


# ─────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────


def _make_packet(
    ptype: PacketType,
    content: str,
    *,
    confidence: float = 0.9,
    source: str = "autoconstitution",
    parent_ids: list[str] | None = None,
    run_id: str | None = None,
    trace: list[str] | None = None,
    valid_until: datetime | None = None,
    half_life_seconds: float | None = None,
    metadata: dict[str, Any] | None = None,
    embed: bool = True,
) -> Packet:
    now = datetime.now(timezone.utc)
    hl = half_life_seconds if half_life_seconds is not None else _DEFAULT_HALF_LIFE[ptype]
    return Packet(
        id=str(uuid.uuid4()),
        type=ptype,
        content=content,
        vector=_pseudo_embed(content) if embed else None,
        confidence=max(0.0, min(1.0, confidence)),
        valid_from=now,
        valid_until=valid_until,
        half_life_seconds=hl,
        revoked_by=None,
        provenance=Provenance(
            source=source,
            created_at=now,
            parent_ids=parent_ids or [],
            run_id=run_id,
            trace=trace or [],
        ),
        metadata=metadata or {},
    )


def make_claim(content: str, **kw: Any) -> Packet:
    """Create a CLAIM packet (initial user prompt or assertion)."""
    return _make_packet(PacketType.CLAIM, content, **kw)


def make_critique(content: str, **kw: Any) -> Packet:
    """Create a CRITIQUE packet (judge output)."""
    return _make_packet(PacketType.CRITIQUE, content, **kw)


def make_revision(content: str, **kw: Any) -> Packet:
    """Create a REVISION packet (student revised answer)."""
    return _make_packet(PacketType.REVISION, content, **kw)


def make_verdict(content: str, **kw: Any) -> Packet:
    """Create a VERDICT packet (proof or final judgment)."""
    return _make_packet(PacketType.VERDICT, content, **kw)


def make_goal(content: str, **kw: Any) -> Packet:
    """Create a GOAL packet (curriculum practice problem)."""
    return _make_packet(PacketType.GOAL, content, **kw)


def make_fact(content: str, **kw: Any) -> Packet:
    """Create a FACT packet (persistent world knowledge)."""
    return _make_packet(PacketType.FACT, content, **kw)


def make_skill(content: str, **kw: Any) -> Packet:
    """Create a SKILL packet (compiled reusable transformation)."""
    return _make_packet(PacketType.SKILL, content, **kw)


def make_episode(content: str, **kw: Any) -> Packet:
    """Create an EPISODE packet (temporal trace entry)."""
    return _make_packet(PacketType.EPISODE, content, **kw)


def make_shadow(content: str, **kw: Any) -> Packet:
    """Create a SHADOW packet (counterfactual alternative with reduced confidence).

    MANIFOLD §2 feature 2 ("Counterfactual shadow execution"): "Rejected
    plans are retained as 'what would have happened' memories with lower
    confidence."
    """
    kw.setdefault("confidence", 0.3)
    return _make_packet(PacketType.SHADOW, content, **kw)


__all__ = [
    "PacketType",
    "EdgeType",
    "Provenance",
    "Packet",
    "make_claim",
    "make_critique",
    "make_revision",
    "make_verdict",
    "make_goal",
    "make_fact",
    "make_skill",
    "make_episode",
    "make_shadow",
    "_pseudo_embed",
    "_EMBED_DIM",
]
