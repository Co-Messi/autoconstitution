"""
autoconstitution.substrate.manifold
=====================================

The Manifold class: SQLite-backed persistent store for Packets and typed
causal edges. All subsystems read and write to this shared store.

Maps to MANIFOLD §3.3 "Persistent Causal State Graph" and §3.4
"Multi-Graph Memory System" — the concrete implementation of the shared
substrate for the autoconstitution critique/revise loop.

Schema version: 1 (inserted on first connect; migrations are additive).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from autoconstitution.substrate.packet import (
    EdgeType,
    Packet,
    PacketType,
    _EMBED_DIM,
    _pseudo_embed,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS packets (
    id              TEXT PRIMARY KEY,
    type            TEXT NOT NULL,
    content         TEXT NOT NULL,
    confidence      REAL NOT NULL,
    valid_from      TEXT NOT NULL,
    valid_until     TEXT,
    half_life       REAL,
    revoked_by      TEXT,
    provenance_json TEXT NOT NULL,
    metadata_json   TEXT NOT NULL,
    vector_blob     BLOB
);

CREATE TABLE IF NOT EXISTS edges (
    src_id      TEXT NOT NULL,
    dst_id      TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (src_id, dst_id, edge_type)
);
"""


def _pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_vector(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine_similarity_np(a: list[float], b: list[float]) -> float:
    """Cosine similarity using numpy (base dep)."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-10:
        return 0.0
    return float(np.dot(va, vb) / denom)


class Manifold:
    """Persistent shared-state substrate backed by SQLite.

    Thread-safe writes via ``with self._conn:`` context manager (SQLite
    isolation level = None means autocommit off; context manager issues
    BEGIN / COMMIT / ROLLBACK).

    MANIFOLD §3.3: "Every node and edge carries a confidence estimate,
    a provenance pointer, a temporal validity window, and a learned decay
    rate."

    Args:
        db_path: Path to the SQLite file. Defaults to
                 ``~/.autoconstitution/substrate.db``.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            default_dir = Path.home() / ".autoconstitution"
            default_dir.mkdir(parents=True, exist_ok=True)
            db_path = default_dir / "substrate.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema init
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(_DDL)
            row = self._conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)", (_SCHEMA_VERSION,)
                )
                logger.debug("substrate schema_version %d initialized", _SCHEMA_VERSION)
            else:
                logger.debug("substrate schema_version %d loaded", row[0])

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Write / Read
    # ------------------------------------------------------------------

    def write(self, packet: Packet) -> str:
        """Persist a packet. Idempotent on id (INSERT OR IGNORE).

        Returns:
            The packet's id.
        """
        vec_blob = _pack_vector(packet.vector) if packet.vector else None
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO packets
                (id, type, content, confidence, valid_from, valid_until,
                 half_life, revoked_by, provenance_json, metadata_json, vector_blob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    packet.id,
                    packet.type.value,
                    packet.content,
                    packet.confidence,
                    packet.valid_from.isoformat(),
                    packet.valid_until.isoformat() if packet.valid_until else None,
                    packet.half_life_seconds,
                    packet.revoked_by,
                    json.dumps(packet.provenance.to_dict()),
                    json.dumps(packet.metadata),
                    vec_blob,
                ),
            )
        return packet.id

    def read(self, packet_id: str) -> Packet | None:
        """Retrieve a packet by id, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM packets WHERE id = ?", (packet_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_packet(row)

    def query(
        self,
        type: PacketType | None = None,
        since: datetime | None = None,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> list[Packet]:
        """Retrieve packets with optional type, time, and confidence filters."""
        clauses: list[str] = ["revoked_by IS NULL"]
        params: list[Any] = []

        if type is not None:
            clauses.append("type = ?")
            params.append(type.value)

        if since is not None:
            clauses.append("valid_from >= ?")
            params.append(since.isoformat())

        if min_confidence is not None:
            clauses.append("confidence >= ?")
            params.append(min_confidence)

        sql = "SELECT * FROM packets"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY valid_from DESC"

        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_packet(r) for r in rows]

    # ------------------------------------------------------------------
    # Vector neighbours (cosine similarity)
    # ------------------------------------------------------------------

    def neighbors(
        self,
        vector: list[float],
        k: int = 5,
        type_filter: PacketType | None = None,
    ) -> list[tuple[Packet, float]]:
        """Return up to k nearest packets by cosine similarity.

        MANIFOLD §3.4: "Each store is queried by the graph's type-appropriate
        mode (spreading activation, range query, identity resolution, skill
        retrieval)."

        Similarity is computed in pure Python / numpy over all non-revoked
        packets that have a stored vector. This is O(n) — fine for MVP scale.
        """
        sql = "SELECT * FROM packets WHERE vector_blob IS NOT NULL AND revoked_by IS NULL"
        params: list[Any] = []
        if type_filter is not None:
            sql += " AND type = ?"
            params.append(type_filter.value)

        rows = self._conn.execute(sql, params).fetchall()
        scored: list[tuple[Packet, float]] = []
        for row in rows:
            pkt = self._row_to_packet(row)
            if pkt.vector:
                sim = _cosine_similarity_np(vector, pkt.vector)
                scored.append((pkt, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ------------------------------------------------------------------
    # Revocation / supersession
    # ------------------------------------------------------------------

    def revoke(
        self,
        packet_id: str,
        reason: str,
        by_packet_id: str | None = None,
    ) -> None:
        """Mark a packet as revoked.

        MANIFOLD §1: "Every stored fact has a source, a confidence, a time
        validity, and a revocation link. An online-learning system that cannot
        unlearn is a system that accumulates error."

        The ``reason`` is stored in metadata["revocation_reason"]; ``revoked_by``
        holds the superseding packet's id (if any). When ``by_packet_id`` is None
        the sentinel ``"__revoked__"`` is used so the query filter
        (``revoked_by IS NULL``) correctly excludes the packet.
        """
        row = self._conn.execute(
            "SELECT metadata_json FROM packets WHERE id = ?", (packet_id,)
        ).fetchone()
        if row is None:
            logger.warning("revoke: packet %s not found", packet_id)
            return
        meta = json.loads(row[0])
        meta["revocation_reason"] = reason

        effective_by = by_packet_id if by_packet_id is not None else "__revoked__"
        with self._conn:
            self._conn.execute(
                "UPDATE packets SET revoked_by = ?, metadata_json = ? WHERE id = ?",
                (effective_by, json.dumps(meta), packet_id),
            )

    def supersede(self, old_id: str, new_id: str) -> None:
        """Mark old_id as superseded by new_id and add a SUPERSEDES edge."""
        self.revoke(old_id, reason=f"superseded by {new_id}", by_packet_id=new_id)
        self.add_edge(new_id, old_id, EdgeType.SUPERSEDES)

    # ------------------------------------------------------------------
    # Graph edges
    # ------------------------------------------------------------------

    def add_edge(
        self,
        src: str,
        dst: str,
        edge_type: EdgeType,
        weight: float = 1.0,
    ) -> None:
        """Add or replace a typed causal edge between two packets."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO edges (src_id, dst_id, edge_type, weight, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (src, dst, edge_type.value, weight, now),
            )

    def edges_from(
        self, packet_id: str, edge_type: EdgeType | None = None
    ) -> list[dict[str, Any]]:
        """Return all edges originating from packet_id."""
        sql = "SELECT * FROM edges WHERE src_id = ?"
        params: list[Any] = [packet_id]
        if edge_type is not None:
            sql += " AND edge_type = ?"
            params.append(edge_type.value)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def edges_to(
        self, packet_id: str, edge_type: EdgeType | None = None
    ) -> list[dict[str, Any]]:
        """Return all edges terminating at packet_id."""
        sql = "SELECT * FROM edges WHERE dst_id = ?"
        params: list[Any] = [packet_id]
        if edge_type is not None:
            sql += " AND edge_type = ?"
            params.append(edge_type.value)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return aggregate counts per packet type, revocation rate, avg confidence."""
        rows = self._conn.execute(
            "SELECT type, COUNT(*) as n, "
            "SUM(CASE WHEN revoked_by IS NOT NULL THEN 1 ELSE 0 END) as revoked, "
            "AVG(confidence) as avg_conf "
            "FROM packets GROUP BY type"
        ).fetchall()
        result: dict[str, Any] = {}
        total = 0
        total_revoked = 0
        for row in rows:
            t, n, revoked, avg_conf = row["type"], row["n"], row["revoked"], row["avg_conf"]
            result[t] = {"count": n, "revoked": revoked, "avg_confidence": round(avg_conf, 4)}
            total += n
            total_revoked += revoked

        result["_total"] = {
            "count": total,
            "revoked": total_revoked,
            "pct_revoked": round(total_revoked / total * 100, 2) if total else 0.0,
        }
        edge_count = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        result["_edges"] = {"count": edge_count}
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_packet(self, row: sqlite3.Row) -> Packet:
        from autoconstitution.substrate.packet import Provenance

        provenance = Provenance.from_dict(json.loads(row["provenance_json"]))
        meta = json.loads(row["metadata_json"])
        vector: list[float] | None = None
        if row["vector_blob"]:
            vector = _unpack_vector(bytes(row["vector_blob"]))

        valid_from = datetime.fromisoformat(row["valid_from"])
        if valid_from.tzinfo is None:
            valid_from = valid_from.replace(tzinfo=timezone.utc)

        valid_until: datetime | None = None
        if row["valid_until"]:
            valid_until = datetime.fromisoformat(row["valid_until"])
            if valid_until.tzinfo is None:
                valid_until = valid_until.replace(tzinfo=timezone.utc)

        return Packet(
            id=row["id"],
            type=PacketType(row["type"]),
            content=row["content"],
            vector=vector,
            confidence=row["confidence"],
            valid_from=valid_from,
            valid_until=valid_until,
            half_life_seconds=row["half_life"],
            revoked_by=row["revoked_by"],
            provenance=provenance,
            metadata=meta,
        )


__all__ = ["Manifold"]
