"""Tests for autoconstitution.substrate.manifold."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from autoconstitution.substrate.manifold import Manifold
from autoconstitution.substrate.packet import (
    EdgeType,
    PacketType,
    make_claim,
    make_critique,
    make_fact,
    make_revision,
    make_shadow,
    _pseudo_embed,
)


# ─────────────────────────────────────────────
# Schema init idempotency
# ─────────────────────────────────────────────


def test_schema_init_idempotent(tmp_path: object) -> None:
    """Re-opening the same db file should not raise or duplicate version rows."""
    db = tmp_path / "test.db"
    m1 = Manifold(db_path=db)
    m1.close()
    m2 = Manifold(db_path=db)
    # Should still work
    row = m2._conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
    assert row == 1
    m2.close()


# ─────────────────────────────────────────────
# Write / Read
# ─────────────────────────────────────────────


def test_write_and_read(tmp_manifold: Manifold) -> None:
    p = make_claim("Hello world")
    pid = tmp_manifold.write(p)
    assert pid == p.id
    restored = tmp_manifold.read(pid)
    assert restored is not None
    assert restored.id == p.id
    assert restored.content == "Hello world"
    assert restored.type == PacketType.CLAIM


def test_write_idempotent(tmp_manifold: Manifold) -> None:
    """Writing the same packet twice should not raise or duplicate."""
    p = make_claim("Idempotent write")
    tmp_manifold.write(p)
    tmp_manifold.write(p)  # second write → INSERT OR IGNORE
    rows = tmp_manifold._conn.execute(
        "SELECT COUNT(*) FROM packets WHERE id = ?", (p.id,)
    ).fetchone()[0]
    assert rows == 1


def test_read_missing_returns_none(tmp_manifold: Manifold) -> None:
    assert tmp_manifold.read("nonexistent-id") is None


# ─────────────────────────────────────────────
# Query
# ─────────────────────────────────────────────


def test_query_by_type(tmp_manifold: Manifold) -> None:
    tmp_manifold.write(make_claim("c1"))
    tmp_manifold.write(make_claim("c2"))
    tmp_manifold.write(make_critique("cr1"))
    claims = tmp_manifold.query(type=PacketType.CLAIM)
    critiques = tmp_manifold.query(type=PacketType.CRITIQUE)
    assert len(claims) == 2
    assert len(critiques) == 1


def test_query_since(tmp_manifold: Manifold) -> None:
    past = datetime.now(timezone.utc) - timedelta(hours=2)
    old = make_claim("old claim")
    old.valid_from = past
    tmp_manifold.write(old)
    new = make_claim("new claim")  # valid_from is now
    tmp_manifold.write(new)

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
    recent = tmp_manifold.query(since=cutoff)
    ids = [p.id for p in recent]
    assert new.id in ids
    assert old.id not in ids


def test_query_min_confidence(tmp_manifold: Manifold) -> None:
    lo = make_claim("low conf", confidence=0.1)
    hi = make_claim("high conf", confidence=0.9)
    tmp_manifold.write(lo)
    tmp_manifold.write(hi)
    results = tmp_manifold.query(min_confidence=0.5)
    ids = [p.id for p in results]
    assert hi.id in ids
    assert lo.id not in ids


def test_query_limit(tmp_manifold: Manifold) -> None:
    for i in range(5):
        tmp_manifold.write(make_claim(f"claim {i}"))
    results = tmp_manifold.query(limit=3)
    assert len(results) == 3


# ─────────────────────────────────────────────
# Neighbors (vector similarity)
# ─────────────────────────────────────────────


def test_neighbors_returns_k_results(tmp_manifold: Manifold) -> None:
    texts = ["apple", "apricot", "avocado", "banana", "blueberry"]
    for t in texts:
        tmp_manifold.write(make_claim(t))
    query_vec = _pseudo_embed("apple")
    results = tmp_manifold.neighbors(query_vec, k=3)
    assert len(results) == 3
    # Top result should be "apple" (exact match → cosine 1.0)
    assert results[0][0].content == "apple"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_neighbors_type_filter(tmp_manifold: Manifold) -> None:
    tmp_manifold.write(make_claim("test claim"))
    tmp_manifold.write(make_critique("test critique"))
    vec = _pseudo_embed("test")
    results = tmp_manifold.neighbors(vec, k=5, type_filter=PacketType.CRITIQUE)
    assert all(p.type == PacketType.CRITIQUE for p, _ in results)


# ─────────────────────────────────────────────
# Revocation
# ─────────────────────────────────────────────


def test_revoke(tmp_manifold: Manifold) -> None:
    p = make_claim("To be revoked")
    tmp_manifold.write(p)
    tmp_manifold.revoke(p.id, reason="outdated", by_packet_id=None)
    # Read raw from DB to check sentinel — read() filters revoked packets out of query
    # but read(id) still returns it
    row = tmp_manifold._conn.execute(
        "SELECT revoked_by, metadata_json FROM packets WHERE id = ?", (p.id,)
    ).fetchone()
    assert row is not None
    assert row["revoked_by"] == "__revoked__"
    import json
    meta = json.loads(row["metadata_json"])
    assert meta.get("revocation_reason") == "outdated"


def test_revoke_with_by_packet_id(tmp_manifold: Manifold) -> None:
    old = make_claim("old")
    new = make_claim("new")
    tmp_manifold.write(old)
    tmp_manifold.write(new)
    tmp_manifold.revoke(old.id, reason="superseded", by_packet_id=new.id)
    restored = tmp_manifold.read(old.id)
    assert restored is not None
    assert restored.revoked_by == new.id


def test_revoke_excluded_from_query(tmp_manifold: Manifold) -> None:
    p = make_claim("will be revoked")
    tmp_manifold.write(p)
    tmp_manifold.revoke(p.id, reason="test")
    results = tmp_manifold.query(type=PacketType.CLAIM)
    assert all(r.id != p.id for r in results)


def test_revoke_missing_packet_is_noop(tmp_manifold: Manifold) -> None:
    """Should not raise when revoking a non-existent packet."""
    tmp_manifold.revoke("does-not-exist", reason="oops")  # no exception


# ─────────────────────────────────────────────
# Supersede
# ─────────────────────────────────────────────


def test_supersede(tmp_manifold: Manifold) -> None:
    old = make_claim("old claim")
    new = make_revision("improved revision")
    tmp_manifold.write(old)
    tmp_manifold.write(new)
    tmp_manifold.supersede(old.id, new.id)

    old_restored = tmp_manifold.read(old.id)
    assert old_restored is not None
    assert old_restored.revoked_by == new.id

    # Should have a SUPERSEDES edge from new → old
    edges = tmp_manifold.edges_from(new.id, edge_type=EdgeType.SUPERSEDES)
    assert any(e["dst_id"] == old.id for e in edges)


# ─────────────────────────────────────────────
# Edges
# ─────────────────────────────────────────────


def test_add_edge_and_query(tmp_manifold: Manifold) -> None:
    src = make_claim("source")
    dst = make_fact("destination")
    tmp_manifold.write(src)
    tmp_manifold.write(dst)
    tmp_manifold.add_edge(src.id, dst.id, EdgeType.SUPPORTS, weight=0.8)

    from_edges = tmp_manifold.edges_from(src.id, edge_type=EdgeType.SUPPORTS)
    assert len(from_edges) == 1
    assert from_edges[0]["dst_id"] == dst.id
    assert from_edges[0]["weight"] == pytest.approx(0.8)

    to_edges = tmp_manifold.edges_to(dst.id, edge_type=EdgeType.SUPPORTS)
    assert len(to_edges) == 1
    assert to_edges[0]["src_id"] == src.id


def test_add_edge_replace_on_same_key(tmp_manifold: Manifold) -> None:
    a = make_claim("a")
    b = make_claim("b")
    tmp_manifold.write(a)
    tmp_manifold.write(b)
    tmp_manifold.add_edge(a.id, b.id, EdgeType.SUPPORTS, weight=0.5)
    tmp_manifold.add_edge(a.id, b.id, EdgeType.SUPPORTS, weight=1.0)  # replace
    edges = tmp_manifold.edges_from(a.id)
    # Should still be exactly 1 SUPPORTS edge
    supports = [e for e in edges if e["edge_type"] == EdgeType.SUPPORTS.value]
    assert len(supports) == 1
    assert supports[0]["weight"] == pytest.approx(1.0)


def test_edges_from_no_type_filter(tmp_manifold: Manifold) -> None:
    a = make_claim("a")
    b = make_claim("b")
    c = make_claim("c")
    tmp_manifold.write(a)
    tmp_manifold.write(b)
    tmp_manifold.write(c)
    tmp_manifold.add_edge(a.id, b.id, EdgeType.SUPPORTS)
    tmp_manifold.add_edge(a.id, c.id, EdgeType.CONTRADICTS)
    edges = tmp_manifold.edges_from(a.id)
    assert len(edges) == 2


# ─────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────


def test_stats_empty(tmp_manifold: Manifold) -> None:
    s = tmp_manifold.stats()
    assert s["_total"]["count"] == 0
    assert s["_edges"]["count"] == 0


def test_stats_populated(tmp_manifold: Manifold) -> None:
    c1 = make_claim("c1")
    c2 = make_claim("c2")
    f1 = make_fact("f1")
    tmp_manifold.write(c1)
    tmp_manifold.write(c2)
    tmp_manifold.write(f1)
    tmp_manifold.revoke(c2.id, reason="test")
    tmp_manifold.add_edge(c1.id, f1.id, EdgeType.CITES)

    s = tmp_manifold.stats()
    assert s["_total"]["count"] == 3
    assert s["_total"]["revoked"] == 1
    assert s["_edges"]["count"] == 1
    assert "claim" in s
    assert s["claim"]["count"] == 2
