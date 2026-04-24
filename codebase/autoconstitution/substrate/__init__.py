"""
autoconstitution.substrate
===========================

MANIFOLD shared-state substrate for the autoconstitution critique/revise loop.

This subpackage implements the concrete realization of MANIFOLD's seven
subsystems and eight features, specialized for autoconstitution:

Seven subsystems realized:
    Event Compiler          → packet.py (typed latent state packets)
    Persistent State Graph  → manifold.py + state_graph.py (SQLite-backed)
    Multi-Graph Memory      → manifold.py (MAGMA schema in single DB)
    Latent Deliberation     → loop.py (critique/revise + counterfactuals)
    Neuro-Symbolic Verif.   → proof_artifact.py (pytest + justification graphs)
    Metacognitive Controller → capability_self_map.py + curriculum.py
    (Inter-agent protocol)  → DEFERRED (needs cryptographic signing)

Eight features realized:
    1. Belief half-life         → packet.current_confidence() + half_life_seconds
    2. Counterfactual shadow    → counterfactual.run_counterfactuals()
    3. Capability self-map      → capability_self_map.CapabilitySelfMap
    4. Proof-carrying outputs   → proof_artifact.{run_pytest_proof, justification_proof}
    5. Skill deprecation        → skill_compiler.SkillCompiler.update_stats() quarantine
    6. Two-phase adaptation     → shadow_validator.ShadowValidator
    7. Inter-agent protocol     → DEFERRED
    8. Curriculum self-generation → curriculum.CurriculumGenerator

Public API:
    Packet, PacketType, EdgeType, Provenance
    make_claim, make_critique, make_revision, make_verdict, make_goal,
    make_fact, make_skill, make_episode, make_shadow
    Manifold
    StateGraph
    TaskSignature, Outcome, TrackRecord, CapabilitySelfMap
    CurriculumGenerator
    ProofArtifact, run_pytest_proof, justification_proof
    run_counterfactuals
    Skill, SkillCompiler
    ShadowValidator
    SubstrateRunResult, SubstrateLoop
"""

from autoconstitution.substrate.packet import (
    PacketType,
    EdgeType,
    Provenance,
    Packet,
    make_claim,
    make_critique,
    make_revision,
    make_verdict,
    make_goal,
    make_fact,
    make_skill,
    make_episode,
    make_shadow,
)
from autoconstitution.substrate.manifold import Manifold
from autoconstitution.substrate.state_graph import StateGraph
from autoconstitution.substrate.capability_self_map import (
    TaskSignature,
    Outcome,
    TrackRecord,
    CapabilitySelfMap,
)
from autoconstitution.substrate.curriculum import CurriculumGenerator
from autoconstitution.substrate.proof_artifact import (
    ProofArtifact,
    run_pytest_proof,
    justification_proof,
)
from autoconstitution.substrate.counterfactual import run_counterfactuals
from autoconstitution.substrate.skill_compiler import Skill, SkillCompiler
from autoconstitution.substrate.shadow_validator import ShadowValidator
from autoconstitution.substrate.loop import SubstrateRunResult, SubstrateLoop

__all__ = [
    # Packet types
    "PacketType",
    "EdgeType",
    "Provenance",
    "Packet",
    # Factories
    "make_claim",
    "make_critique",
    "make_revision",
    "make_verdict",
    "make_goal",
    "make_fact",
    "make_skill",
    "make_episode",
    "make_shadow",
    # Core substrate
    "Manifold",
    "StateGraph",
    # Capability self-map
    "TaskSignature",
    "Outcome",
    "TrackRecord",
    "CapabilitySelfMap",
    # Curriculum
    "CurriculumGenerator",
    # Proof artifacts
    "ProofArtifact",
    "run_pytest_proof",
    "justification_proof",
    # Counterfactuals
    "run_counterfactuals",
    # Skills
    "Skill",
    "SkillCompiler",
    # Shadow validation
    "ShadowValidator",
    # Loop
    "SubstrateRunResult",
    "SubstrateLoop",
]
