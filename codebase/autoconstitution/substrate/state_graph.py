"""
autoconstitution.substrate.state_graph
=========================================

StateGraph: graph-level queries over the Manifold, implementing the
"Persistent Causal State Graph" from MANIFOLD §3.3.

Provides justification walks, contradiction detection, staleness analysis,
and lineage chains — without requiring networkx. All graph traversal is done
with pure adjacency dicts built on the fly from Manifold edge queries.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from autoconstitution.substrate.manifold import Manifold
from autoconstitution.substrate.packet import EdgeType, Packet

logger = logging.getLogger(__name__)


class StateGraph:
    """Graph-level queries over the Manifold.

    MANIFOLD §3.3: "This graph is the object of reasoning. The deliberation
    engine operates over it, not over raw tokens. Perception updates it,
    memory retrieval populates it, planning simulates futures against it,
    action commits to it."

    All traversal is done with pure Python BFS/DFS; networkx is not required.

    Args:
        manifold: The backing Manifold instance.
    """

    def __init__(self, manifold: Manifold) -> None:
        self._m = manifold

    # ------------------------------------------------------------------
    # Justification graph
    # ------------------------------------------------------------------

    def justification_graph(self, packet_id: str) -> dict[str, Any]:
        """Walk DERIVES_FROM / SUPPORTS / CITES edges backward to roots.

        Returns an adjacency-dict representation suitable for JSON export or
        display. Each node is keyed by packet id; its value includes the
        packet's type, content snippet, confidence, and a list of parent ids.

        MANIFOLD §2 feature 4 ("Proof-carrying outputs"): "For non-verifiable
        domains it is a structured justification graph citing the evidence nodes
        in memory that support each claim."
        """
        walk_types = {EdgeType.DERIVES_FROM, EdgeType.SUPPORTS, EdgeType.CITES}
        visited: set[str] = set()
        nodes: dict[str, Any] = {}
        queue = [packet_id]

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            pkt = self._m.read(current_id)
            if pkt is None:
                continue

            parents: list[str] = []
            for et in walk_types:
                for edge in self._m.edges_from(current_id, edge_type=et):
                    dst = edge["dst_id"]
                    parents.append(dst)
                    if dst not in visited:
                        queue.append(dst)

            nodes[current_id] = {
                "id": current_id,
                "type": pkt.type.value,
                "content_snippet": pkt.content[:120],
                "confidence": pkt.current_confidence(),
                "parent_ids": parents,
            }

        return {
            "root": packet_id,
            "nodes": nodes,
        }

    # ------------------------------------------------------------------
    # Contradictions
    # ------------------------------------------------------------------

    def contradictions(self) -> list[tuple[Packet, Packet]]:
        """Return pairs of packets linked by a CONTRADICTS edge.

        MANIFOLD §3.3: nodes and edges carry confidence and provenance so the
        system can reason about conflicting beliefs rather than silently
        accumulating them.
        """
        rows = self._m._conn.execute(
            "SELECT src_id, dst_id FROM edges WHERE edge_type = ?",
            (EdgeType.CONTRADICTS.value,),
        ).fetchall()

        pairs: list[tuple[Packet, Packet]] = []
        for row in rows:
            src = self._m.read(row["src_id"])
            dst = self._m.read(row["dst_id"])
            if src is not None and dst is not None:
                pairs.append((src, dst))
        return pairs

    # ------------------------------------------------------------------
    # Stale nodes
    # ------------------------------------------------------------------

    def stale_nodes(
        self, now: datetime | None = None, threshold: float = 0.2
    ) -> list[Packet]:
        """Return packets whose current_confidence has fallen below threshold.

        MANIFOLD §2 feature 1 ("Belief half-life"): beliefs with short half-lives
        eventually become stale and should be quarantined or refreshed.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        packets = self._m.query()  # all non-revoked
        return [p for p in packets if p.current_confidence(now) < threshold]

    # ------------------------------------------------------------------
    # Lineage chain
    # ------------------------------------------------------------------

    def lineage(self, packet_id: str) -> list[str]:
        """Return the chain of REVISES edges from root to packet_id.

        Walks backward (packet_id → ancestors) following REVISES edges
        until a root is reached. Returns the chain in root-first order.

        MANIFOLD §3.3: "Every node and edge carries a provenance pointer."
        """
        chain = [packet_id]
        current = packet_id
        seen: set[str] = {current}

        while True:
            # Look for incoming REVISES edges (who revised into current?)
            edges = self._m.edges_to(current, edge_type=EdgeType.REVISES)
            if not edges:
                break
            parent_id = edges[0]["src_id"]
            if parent_id in seen:
                break  # cycle guard
            seen.add(parent_id)
            chain.append(parent_id)
            current = parent_id

        chain.reverse()
        return chain


__all__ = ["StateGraph"]
