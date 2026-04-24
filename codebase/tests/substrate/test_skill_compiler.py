"""Tests for autoconstitution.substrate.skill_compiler."""

from __future__ import annotations

import pytest

from autoconstitution.substrate.capability_self_map import TaskSignature
from autoconstitution.substrate.manifold import Manifold
from autoconstitution.substrate.packet import (
    PacketType,
    make_claim,
    make_critique,
    make_revision,
    make_verdict,
)
from autoconstitution.substrate.skill_compiler import Skill, SkillCompiler


def _sig(domain: str = "code", kind: str = "code") -> TaskSignature:
    return TaskSignature(domain=domain, difficulty="medium", kind=kind)


@pytest.fixture()
def compiler(tmp_manifold: Manifold) -> SkillCompiler:
    return SkillCompiler(tmp_manifold)


@pytest.fixture()
def compiler_with_provider(tmp_manifold: Manifold, fake_provider: object) -> SkillCompiler:
    return SkillCompiler(tmp_manifold, provider=fake_provider)


# ─────────────────────────────────────────────
# compile_from_trace
# ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compile_from_trace_empty_returns_none(compiler: SkillCompiler) -> None:
    result = await compiler.compile_from_trace([])
    assert result is None


@pytest.mark.asyncio
async def test_compile_from_trace_no_critique_returns_none(compiler: SkillCompiler) -> None:
    trace = [make_claim("just a claim"), make_revision("just a revision")]
    # No CRITIQUE → not compilable (claim + revision alone insufficient)
    result = await compiler.compile_from_trace(trace)
    assert result is None


@pytest.mark.asyncio
async def test_compile_from_trace_no_revision_returns_none(compiler: SkillCompiler) -> None:
    trace = [make_claim("claim"), make_critique("just critiques")]
    result = await compiler.compile_from_trace(trace)
    assert result is None


@pytest.mark.asyncio
async def test_compile_from_trace_produces_skill(
    tmp_manifold: Manifold, compiler: SkillCompiler
) -> None:
    sig = _sig()
    trace = [
        make_claim("Write a sorting function."),
        make_critique("Critique: use early returns to simplify logic."),
        make_revision("def sort(lst): lst.sort(); return lst"),
    ]
    skill = await compiler.compile_from_trace(trace, signature=sig)
    assert skill is not None
    assert isinstance(skill, Skill)
    assert skill.trigger_signature == sig
    assert len(skill.transformation_prompt) > 0


@pytest.mark.asyncio
async def test_compile_from_trace_persists_skill_packet(
    tmp_manifold: Manifold, compiler: SkillCompiler
) -> None:
    trace = [make_critique("Be concise."), make_revision("Short answer.")]
    skill = await compiler.compile_from_trace(trace)
    assert skill is not None
    pkt = tmp_manifold.read(skill.id)
    assert pkt is not None
    assert pkt.type == PacketType.SKILL


@pytest.mark.asyncio
async def test_compile_with_provider_uses_extraction(
    tmp_manifold: Manifold, compiler_with_provider: SkillCompiler
) -> None:
    trace = [make_critique("Use early returns."), make_revision("Improved.")]
    skill = await compiler_with_provider.compile_from_trace(trace)
    # With fake_provider keyed on "extract"/"lesson"/"skill" → should return lesson text
    assert skill is not None
    assert len(skill.transformation_prompt) > 0


# ─────────────────────────────────────────────
# retrieve
# ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieve_returns_matching_skill(
    tmp_manifold: Manifold, compiler: SkillCompiler
) -> None:
    sig = TaskSignature("code", "medium", "code")
    trace = [make_critique("Be concise."), make_revision("Better.")]
    skill = await compiler.compile_from_trace(trace, signature=sig)
    assert skill is not None

    results = compiler.retrieve(sig)
    assert len(results) >= 1
    assert any(s.id == skill.id for s in results)


@pytest.mark.asyncio
async def test_retrieve_ignores_different_domain(
    tmp_manifold: Manifold, compiler: SkillCompiler
) -> None:
    sig_code = TaskSignature("code", "easy", "code")
    sig_math = TaskSignature("math", "easy", "math")
    trace = [make_critique("critique"), make_revision("rev")]
    await compiler.compile_from_trace(trace, signature=sig_code)
    results = compiler.retrieve(sig_math)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_retrieve_ignores_quarantined(
    tmp_manifold: Manifold, compiler: SkillCompiler
) -> None:
    sig = TaskSignature("code", "medium", "code")
    trace = [make_critique("crit"), make_revision("rev")]
    skill = await compiler.compile_from_trace(trace, signature=sig)
    assert skill is not None

    # Force quarantine via update_stats
    for _ in range(5):
        compiler.update_stats(skill.id, success=False)

    results = compiler.retrieve(sig)
    # Quarantined skill should not appear
    assert not any(s.id == skill.id for s in results)


# ─────────────────────────────────────────────
# update_stats / quarantine
# ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_stats_success_increments_uses(
    tmp_manifold: Manifold, compiler: SkillCompiler
) -> None:
    trace = [make_critique("crit"), make_revision("rev")]
    skill = await compiler.compile_from_trace(trace)
    assert skill is not None
    compiler.update_stats(skill.id, success=True)
    compiler.update_stats(skill.id, success=True)
    updated = tmp_manifold.read(skill.id)
    assert updated is not None
    assert updated.metadata["uses"] == 2


@pytest.mark.asyncio
async def test_quarantine_after_failure_streak(
    tmp_manifold: Manifold, compiler: SkillCompiler
) -> None:
    """After 5 consecutive failures, skill should be quarantined."""
    trace = [make_critique("crit"), make_revision("rev")]
    skill = await compiler.compile_from_trace(trace)
    assert skill is not None

    for _ in range(5):
        compiler.update_stats(skill.id, success=False)

    updated = tmp_manifold.read(skill.id)
    assert updated is not None
    assert updated.metadata.get("quarantined") is True


@pytest.mark.asyncio
async def test_no_quarantine_with_mixed_results(
    tmp_manifold: Manifold, compiler: SkillCompiler
) -> None:
    """Mixed success/failure should not trigger quarantine."""
    trace = [make_critique("crit"), make_revision("rev")]
    skill = await compiler.compile_from_trace(trace)
    assert skill is not None

    for _ in range(3):
        compiler.update_stats(skill.id, success=True)
    for _ in range(2):
        compiler.update_stats(skill.id, success=False)

    updated = tmp_manifold.read(skill.id)
    assert updated is not None
    # 3/5 successes = 0.6 > 0.3 threshold → not quarantined
    assert updated.metadata.get("quarantined") is False
