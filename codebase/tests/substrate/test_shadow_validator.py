"""Tests for autoconstitution.substrate.shadow_validator."""

from __future__ import annotations

import pytest

from autoconstitution.substrate.capability_self_map import TaskSignature
from autoconstitution.substrate.manifold import Manifold
from autoconstitution.substrate.packet import PacketType, make_critique, make_revision
from autoconstitution.substrate.shadow_validator import ShadowValidator
from autoconstitution.substrate.skill_compiler import Skill, SkillCompiler


async def _score_always(task_id: str, prompt: str) -> float:
    """Always returns 0.9 — no regression."""
    return 0.9


async def _score_low(task_id: str, prompt: str) -> float:
    """Always returns 0.2 — forces regression."""
    return 0.2


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


async def _make_skill(tmp_manifold: Manifold) -> Skill:
    compiler = SkillCompiler(tmp_manifold)
    trace = [make_critique("Be concise."), make_revision("Better answer.")]
    skill = await compiler.compile_from_trace(
        trace, signature=TaskSignature("code", "easy", "code")
    )
    assert skill is not None
    return skill


# ─────────────────────────────────────────────
# validate
# ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_validate_passes_when_no_regression(tmp_manifold: Manifold) -> None:
    skill = await _make_skill(tmp_manifold)
    validator = ShadowValidator(tmp_manifold, protected_tasks={"task-1": 0.8, "task-2": 0.7})
    passed, report = await validator.validate(skill, _score_always)
    assert passed is True
    assert report["regressions"] == []


@pytest.mark.asyncio
async def test_validate_fails_on_regression(tmp_manifold: Manifold) -> None:
    skill = await _make_skill(tmp_manifold)
    validator = ShadowValidator(tmp_manifold, protected_tasks={"task-1": 0.8})
    passed, report = await validator.validate(skill, _score_low)
    assert passed is False
    assert "task-1" in report["regressions"]


@pytest.mark.asyncio
async def test_validate_report_has_delta(tmp_manifold: Manifold) -> None:
    skill = await _make_skill(tmp_manifold)
    validator = ShadowValidator(tmp_manifold, protected_tasks={"task-1": 0.8})
    _, report = await validator.validate(skill, _score_always)
    task_report = report["tasks"]["task-1"]
    assert "baseline" in task_report
    assert "score" in task_report
    assert "delta" in task_report


@pytest.mark.asyncio
async def test_validate_empty_protected_tasks_always_passes(tmp_manifold: Manifold) -> None:
    skill = await _make_skill(tmp_manifold)
    validator = ShadowValidator(tmp_manifold, protected_tasks={})
    passed, report = await validator.validate(skill, _score_low)
    assert passed is True


@pytest.mark.asyncio
async def test_validate_handles_run_fn_exception(tmp_manifold: Manifold) -> None:
    """run_fn raising should be treated as score=0 → regression."""
    skill = await _make_skill(tmp_manifold)

    async def _raise(task_id: str, prompt: str) -> float:
        raise RuntimeError("simulated failure")

    validator = ShadowValidator(tmp_manifold, protected_tasks={"task-1": 0.5})
    passed, report = await validator.validate(skill, _raise)
    assert passed is False
    assert report["tasks"]["task-1"]["score"] == 0.0


# ─────────────────────────────────────────────
# commit
# ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_commit_un_quarantines_skill(tmp_manifold: Manifold) -> None:
    skill = await _make_skill(tmp_manifold)
    # Force quarantine
    import json
    pkt = tmp_manifold.read(skill.id)
    assert pkt is not None
    meta = pkt.metadata
    meta["quarantined"] = True
    with tmp_manifold._conn:
        tmp_manifold._conn.execute(
            "UPDATE packets SET metadata_json = ? WHERE id = ?",
            (json.dumps(meta), skill.id),
        )

    validator = ShadowValidator(tmp_manifold)
    validator.commit(skill)

    updated = tmp_manifold.read(skill.id)
    assert updated is not None
    assert updated.metadata.get("quarantined") is False


# ─────────────────────────────────────────────
# rollback
# ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rollback_revokes_skill_packet(tmp_manifold: Manifold) -> None:
    skill = await _make_skill(tmp_manifold)
    validator = ShadowValidator(tmp_manifold)
    validator.rollback(skill, reason="regression detected")

    # Packet should now be excluded from active queries
    active = tmp_manifold.query(type=PacketType.SKILL)
    assert not any(p.id == skill.id for p in active)


@pytest.mark.asyncio
async def test_full_shadow_validate_and_rollback_cycle(tmp_manifold: Manifold) -> None:
    """Full two-phase cycle: validate fails → rollback."""
    skill = await _make_skill(tmp_manifold)
    validator = ShadowValidator(tmp_manifold, protected_tasks={"task-A": 0.9})
    passed, report = await validator.validate(skill, _score_low)
    assert not passed
    validator.rollback(skill, reason=f"regressions: {report['regressions']}")

    active = tmp_manifold.query(type=PacketType.SKILL)
    assert not any(p.id == skill.id for p in active)


@pytest.mark.asyncio
async def test_full_shadow_validate_and_commit_cycle(tmp_manifold: Manifold) -> None:
    """Full two-phase cycle: validate passes → commit."""
    skill = await _make_skill(tmp_manifold)
    validator = ShadowValidator(tmp_manifold, protected_tasks={"task-B": 0.5})
    passed, _ = await validator.validate(skill, _score_always)
    assert passed
    validator.commit(skill)

    updated = tmp_manifold.read(skill.id)
    assert updated is not None
    assert updated.metadata.get("quarantined") is False
