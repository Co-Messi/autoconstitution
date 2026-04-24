"""
autoconstitution.substrate.proof_artifact
==========================================

Proof-carrying outputs: every externalizable answer carries a justification
artifact that consumers can verify without re-running the computation.

MANIFOLD §2 feature 4 ("Proof-carrying outputs"): "Every externalizable answer
carries a justification artifact. For formally verifiable domains (math, code,
constraint-satisfaction, database queries) this is a checkable proof
certificate. For non-verifiable domains it is a structured justification graph
citing the evidence nodes in memory that support each claim."

MANIFOLD §3.5: "When a subproblem is formalizable, the engine compiles outward —
emitting code, a theorem-prover statement, a SQL query, a constraint problem —
executes it in an external verifier, and re-embeds the result back into the
manifold. The verifier's output becomes a proof artifact attached to the
conclusion."
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ProofArtifact
# ─────────────────────────────────────────────


@dataclass
class ProofArtifact:
    """A verifiable proof certificate attached to an output.

    MANIFOLD §4 feature 4: proof artifacts are *transportable* — they travel
    with the answer and can be re-verified by downstream consumers.

    Attributes:
        kind:      "pytest" for code proofs, "justification" for reasoning.
        payload:   Structured data: pytest stdout, assertion counts, or
                   justification graph node dict.
        verdict:   "pass" | "fail" | "inconclusive".
        checkable: True if a consumer can re-run verification independently.
    """

    kind: str  # "pytest" | "justification"
    payload: dict[str, Any]
    verdict: str  # "pass" | "fail" | "inconclusive"
    checkable: bool


# ─────────────────────────────────────────────
# Pytest proof
# ─────────────────────────────────────────────


def run_pytest_proof(
    code: str,
    tests: str,
    timeout: int = 15,
) -> ProofArtifact:
    """Run code + tests in a subprocess and return a ProofArtifact.

    MANIFOLD §3.5: "compile outward — emitting code … executes it in an
    external verifier." This is the code-domain verifier.

    The code and tests are written to a temp directory, then:
        python -m pytest <test_file> -v --tb=short --timeout=<timeout>

    Never imports pytest into this process — subprocess isolation prevents
    test state pollution.

    Args:
        code:    Python source to write to ``solution.py``.
        tests:   Pytest test source to write to ``test_solution.py``.
        timeout: Hard wall-clock timeout in seconds.

    Returns:
        ProofArtifact with kind="pytest", checkable=True.
    """
    with tempfile.TemporaryDirectory(prefix="autoconstitution_proof_") as tmpdir:
        tmp = Path(tmpdir)
        solution_path = tmp / "solution.py"
        test_path = tmp / "test_solution.py"
        solution_path.write_text(textwrap.dedent(code), encoding="utf-8")
        test_path.write_text(textwrap.dedent(tests), encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(test_path),
            "-v",
            "--tb=short",
            "-o", "addopts=",  # strip project addopts (timeout, cov, etc.)
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout + 5,
            )
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            return ProofArtifact(
                kind="pytest",
                payload={"error": "timed out", "timeout_s": timeout},
                verdict="inconclusive",
                checkable=False,
            )
        except Exception as exc:
            return ProofArtifact(
                kind="pytest",
                payload={"error": str(exc)},
                verdict="inconclusive",
                checkable=False,
            )

    # Parse the pytest output for summary lines
    passed, failed, errors = _parse_pytest_summary(stdout)

    verdict: str
    if exit_code == 0:
        verdict = "pass"
    elif exit_code == 1:
        verdict = "fail"
    else:
        verdict = "inconclusive"

    logger.debug(
        "pytest proof: exit=%d passed=%d failed=%d", exit_code, passed, failed
    )

    return ProofArtifact(
        kind="pytest",
        payload={
            "exit_code": exit_code,
            "stdout": stdout[:4000],  # cap for storage
            "stderr": stderr[:1000],
            "passed": passed,
            "failed": failed,
            "errors": errors,
        },
        verdict=verdict,
        checkable=True,
    )


def _parse_pytest_summary(output: str) -> tuple[int, int, int]:
    """Extract passed / failed / error counts from pytest's final summary line."""
    passed = failed = errors = 0
    for line in reversed(output.splitlines()):
        line = line.lower()
        if "passed" in line or "failed" in line or "error" in line:
            parts = line.split(",")
            for part in parts:
                part = part.strip()
                if "passed" in part:
                    passed = int("".join(c for c in part if c.isdigit()) or "0")
                elif "failed" in part:
                    failed = int("".join(c for c in part if c.isdigit()) or "0")
                elif "error" in part:
                    errors = int("".join(c for c in part if c.isdigit()) or "0")
            break
    return passed, failed, errors


# ─────────────────────────────────────────────
# Justification proof
# ─────────────────────────────────────────────


def justification_proof(claim_id: str, state_graph: Any) -> ProofArtifact:
    """Serialise a justification graph as a proof artifact.

    MANIFOLD §2 feature 4: "For non-verifiable domains it is a structured
    justification graph citing the evidence nodes in memory that support
    each claim."

    Args:
        claim_id:    The root packet whose justification to walk.
        state_graph: A StateGraph instance.

    Returns:
        ProofArtifact with kind="justification", checkable=False (no re-run).
    """
    jg = state_graph.justification_graph(claim_id)
    nodes = jg.get("nodes", {})
    cited_ids = list(nodes.keys())
    confidences = {nid: nodes[nid].get("confidence", 0.0) for nid in cited_ids}
    min_conf = min(confidences.values()) if confidences else 0.0

    verdict: str
    if min_conf >= 0.7:
        verdict = "pass"
    elif min_conf >= 0.3:
        verdict = "inconclusive"
    else:
        verdict = "fail"

    return ProofArtifact(
        kind="justification",
        payload={
            "root": claim_id,
            "cited_packet_ids": cited_ids,
            "confidences": confidences,
            "min_confidence": min_conf,
            "graph": jg,
        },
        verdict=verdict,
        checkable=False,
    )


__all__ = [
    "ProofArtifact",
    "run_pytest_proof",
    "justification_proof",
]
