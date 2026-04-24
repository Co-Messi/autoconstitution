"""Tests for autoconstitution.substrate.proof_artifact."""

from __future__ import annotations

import pytest

from autoconstitution.substrate.proof_artifact import (
    ProofArtifact,
    justification_proof,
    run_pytest_proof,
)


# ─────────────────────────────────────────────
# run_pytest_proof
# ─────────────────────────────────────────────

_PASSING_CODE = """
def add(a, b):
    return a + b
"""

_PASSING_TESTS = """
from solution import add

def test_add_positive():
    assert add(1, 2) == 3

def test_add_zero():
    assert add(0, 0) == 0
"""

_FAILING_CODE = """
def add(a, b):
    return a - b  # intentional bug
"""

_FAILING_TESTS = """
from solution import add

def test_add_fails():
    assert add(1, 2) == 3
"""


def test_pytest_proof_passing() -> None:
    proof = run_pytest_proof(_PASSING_CODE, _PASSING_TESTS)
    assert isinstance(proof, ProofArtifact)
    assert proof.kind == "pytest"
    assert proof.verdict == "pass"
    assert proof.checkable is True
    assert proof.payload["exit_code"] == 0
    assert proof.payload["passed"] >= 2


def test_pytest_proof_failing() -> None:
    proof = run_pytest_proof(_FAILING_CODE, _FAILING_TESTS)
    assert proof.kind == "pytest"
    assert proof.verdict == "fail"
    assert proof.checkable is True
    assert proof.payload["exit_code"] == 1
    assert proof.payload["failed"] >= 1


def test_pytest_proof_payload_has_stdout() -> None:
    proof = run_pytest_proof(_PASSING_CODE, _PASSING_TESTS)
    assert "stdout" in proof.payload
    assert len(proof.payload["stdout"]) > 0


# ─────────────────────────────────────────────
# justification_proof
# ─────────────────────────────────────────────


def test_justification_proof_single_node(tmp_manifold: object) -> None:
    from autoconstitution.substrate.packet import make_claim
    from autoconstitution.substrate.state_graph import StateGraph

    p = make_claim("Sky is blue", confidence=0.9)
    tmp_manifold.write(p)
    sg = StateGraph(tmp_manifold)

    proof = justification_proof(p.id, sg)
    assert proof.kind == "justification"
    assert proof.checkable is False
    assert p.id in proof.payload["cited_packet_ids"]
    assert proof.payload["root"] == p.id


def test_justification_proof_verdict_pass(tmp_manifold: object) -> None:
    """High-confidence root → verdict pass."""
    from autoconstitution.substrate.packet import make_fact
    from autoconstitution.substrate.state_graph import StateGraph

    p = make_fact("Well-established fact", confidence=0.95)
    tmp_manifold.write(p)
    sg = StateGraph(tmp_manifold)
    proof = justification_proof(p.id, sg)
    assert proof.verdict == "pass"


def test_justification_proof_verdict_fail(tmp_manifold: object) -> None:
    """Revoked packet → confidence 0 → verdict fail."""
    from autoconstitution.substrate.packet import make_claim
    from autoconstitution.substrate.state_graph import StateGraph

    p = make_claim("Dubious claim", confidence=0.1)
    tmp_manifold.write(p)
    sg = StateGraph(tmp_manifold)
    proof = justification_proof(p.id, sg)
    assert proof.verdict == "fail"
