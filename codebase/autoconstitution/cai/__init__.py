"""
autoconstitution.cai — Constitutional AI loop primitives.

Implements the four pillars of the CAI training recipe:

    critique_revision  — Student produces, Judge critiques, Student revises.
    hierarchy          — Student / Judge / Meta-Judge role abstractions.
    preference_pairs   — Convert (chosen, rejected) traces into DPO training data.
    trl_trainer        — Thin wrapper over HuggingFace TRL's DPOTrainer.

Reference: Bai et al. 2022, "Constitutional AI: Harmlessness from AI Feedback".
"""

from autoconstitution.cai.hierarchy import (
    CAIRole,
    JudgeAgent,
    MetaJudgeAgent,
    StudentAgent,
)
from autoconstitution.cai.critique_revision import (
    CritiqueRevisionLoop,
    CritiqueResult,
    RevisionResult,
)
from autoconstitution.cai.preference_pairs import (
    PreferencePair,
    PreferencePairBuilder,
)

__all__ = [
    "CAIRole",
    "StudentAgent",
    "JudgeAgent",
    "MetaJudgeAgent",
    "CritiqueRevisionLoop",
    "CritiqueResult",
    "RevisionResult",
    "PreferencePair",
    "PreferencePairBuilder",
]
