"""Tests for autoconstitution.substrate.state_graph."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from autoconstitution.substrate.manifold import Manifold
from autoconstitution.substrate.packet import (
    EdgeType,
    make_claim,
    make_critique,
    make_fact,
    make_revision,
)
from autoconstitution.substrate.state_graph import StateGraph


def test_justification_graph_single_node(tmp_manifold: Manifold) -> None:
    """A claim with no parents should produce a graph with just itself."""
    p = make_claim("isolated claim")
    tmp_manifold.write(p)
    sg = StateGraph(tmp_manifold)
    jg = sg.justification_graph(p.id)
    assert jg["root"] == p.id
    assert p.id in jg["nodes"]
    assert jg["nodes"][p.id]["parent_ids"] == []


def test_justification_graph_walks_derives_from(tmp_manifold: Manifold) -> None:
    root = make_fact("base fact")
    child = make_claim("derived claim")
    tmp_manifold.write(root)
    tmp_manifold.write(child)
    tmp_manifold.add_edge(child.id, root.id, EdgeType.DERIVES_FROM)

    sg = StateGraph(tmp_manifold)
    jg = sg.justification_graph(child.id)
    assert child.id in jg["nodes"]
    assert root.id in jg["nodes"]
    assert root.id in jg["nodes"][child.id]["parent_ids"]


def test_justification_graph_walks_supports(tmp_manifold: Manifold) -> None:
    evidence = make_fact("supporting evidence")
    claim = make_claim("claim")
    tmp_manifold.write(evidence)
    tmp_manifold.write(claim)
    tmp_manifold.add_edge(claim.id, evidence.id, EdgeType.SUPPORTS)

    sg = StateGraph(tmp_manifold)
    jg = sg.justification_graph(claim.id)
    assert evidence.id in jg["nodes"]


def test_justification_graph_transitive(tmp_manifold: Manifold) -> None:
    root = make_fact("root fact")
    mid = make_claim("middle claim")
    leaf = make_revision("leaf revision")
    tmp_manifold.write(root)
    tmp_manifold.write(mid)
    tmp_manifold.write(leaf)
    tmp_manifold.add_edge(mid.id, root.id, EdgeType.DERIVES_FROM)
    tmp_manifold.add_edge(leaf.id, mid.id, EdgeType.DERIVES_FROM)

    sg = StateGraph(tmp_manifold)
    jg = sg.justification_graph(leaf.id)
    assert root.id in jg["nodes"]
    assert mid.id in jg["nodes"]
    assert leaf.id in jg["nodes"]


def test_contradictions_empty(tmp_manifold: Manifold) -> None:
    sg = StateGraph(tmp_manifold)
    assert sg.contradictions() == []


def test_contradictions_detected(tmp_manifold: Manifold) -> None:
    a = make_claim("the sky is blue")
    b = make_claim("the sky is not blue")
    tmp_manifold.write(a)
    tmp_manifold.write(b)
    tmp_manifold.add_edge(a.id, b.id, EdgeType.CONTRADICTS)

    sg = StateGraph(tmp_manifold)
    pairs = sg.contradictions()
    assert len(pairs) == 1
    ids = {pairs[0][0].id, pairs[0][1].id}
    assert a.id in ids
    assert b.id in ids


def test_stale_nodes_empty_manifold(tmp_manifold: Manifold) -> None:
    sg = StateGraph(tmp_manifold)
    stale = sg.stale_nodes()
    assert stale == []


def test_stale_nodes_detected(tmp_manifold: Manifold) -> None:
    """A claim with short half_life written long ago should appear stale."""
    p = make_claim("stale claim", confidence=1.0, half_life_seconds=1.0)
    # Set valid_from far in the past
    p.valid_from = datetime.now(timezone.utc) - timedelta(hours=2)
    tmp_manifold.write(p)

    sg = StateGraph(tmp_manifold)
    stale = sg.stale_nodes(threshold=0.5)
    ids = [s.id for s in stale]
    assert p.id in ids


def test_stale_nodes_fresh_not_included(tmp_manifold: Manifold) -> None:
    p = make_fact("fresh permanent fact", confidence=0.95)
    tmp_manifold.write(p)
    sg = StateGraph(tmp_manifold)
    stale = sg.stale_nodes(threshold=0.5)
    ids = [s.id for s in stale]
    assert p.id not in ids


def test_lineage_single_packet(tmp_manifold: Manifold) -> None:
    p = make_claim("standalone")
    tmp_manifold.write(p)
    sg = StateGraph(tmp_manifold)
    chain = sg.lineage(p.id)
    assert chain == [p.id]


def test_lineage_two_step_chain(tmp_manifold: Manifold) -> None:
    root = make_claim("original")
    rev = make_revision("revised version")
    tmp_manifold.write(root)
    tmp_manifold.write(rev)
    # rev REVISES root: edge goes root → rev (src=root, dst=rev direction)
    # lineage walks edges_to current via REVISES
    tmp_manifold.add_edge(root.id, rev.id, EdgeType.REVISES)

    sg = StateGraph(tmp_manifold)
    chain = sg.lineage(rev.id)
    assert chain[0] == root.id
    assert chain[-1] == rev.id
    assert len(chain) == 2
