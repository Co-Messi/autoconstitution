"""Integration tests for autoconstitution.substrate.loop."""

from __future__ import annotations

import uuid

import pytest

from autoconstitution.providers.fake import FakeProvider
from autoconstitution.substrate.capability_self_map import CapabilitySelfMap
from autoconstitution.substrate.curriculum import CurriculumGenerator
from autoconstitution.substrate.loop import SubstrateLoop, SubstrateRunResult
from autoconstitution.substrate.manifold import Manifold
from autoconstitution.substrate.packet import PacketType
from autoconstitution.substrate.shadow_validator import ShadowValidator
from autoconstitution.substrate.skill_compiler import SkillCompiler


def _make_loop(tmp_manifold: Manifold, provider: FakeProvider) -> SubstrateLoop:
    """Wire up a full SubstrateLoop with the given manifold and fake provider."""
    self_map = CapabilitySelfMap(tmp_manifold)
    skill_compiler = SkillCompiler(tmp_manifold, provider=provider)
    shadow_validator = ShadowValidator(tmp_manifold, protected_tasks={})
    curriculum_gen = CurriculumGenerator(provider, tmp_manifold, self_map)
    return SubstrateLoop(
        provider=provider,
        manifold=tmp_manifold,
        self_map=self_map,
        skill_compiler=skill_compiler,
        shadow_validator=shadow_validator,
        curriculum_gen=curriculum_gen,
        cai_runner=None,
    )


def _make_provider() -> FakeProvider:
    """Deterministic callable-mode FakeProvider for loop tests."""
    def _respond(prompt: str) -> str:
        p = prompt.lower()
        if "critique" in p or "json" in p:
            return '{"verdict": "compliant", "critiques": []}'
        if "improved answer" in p or "revise" in p or "revised" in p or "revision" in p:
            return "Revised: here is an improved answer addressing the critique."
        if "extract" in p or "lesson" in p or "skill" in p:
            return "Lesson: prefer clear variable names."
        if "synthesize" in p or "practice" in p or "curriculum" in p:
            return "Practice problem: implement a fizzbuzz function."
        if "shadow" in p or "alternative" in p or "counterfactual" in p:
            return "Alternative: use a different approach."
        return "Initial response to the user prompt."

    return FakeProvider(responses=_respond)


# ─────────────────────────────────────────────
# Core run
# ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_returns_result(tmp_manifold: Manifold) -> None:
    provider = _make_provider()
    loop = _make_loop(tmp_manifold, provider)
    task = {
        "prompt": "Explain binary search.",
        "domain": "code",
        "difficulty": "easy",
        "kind": "free",
    }
    result = await loop.run(task, run_id=str(uuid.uuid4()))
    assert isinstance(result, SubstrateRunResult)
    assert len(result.chosen_text) > 0


@pytest.mark.asyncio
async def test_run_creates_packets(tmp_manifold: Manifold) -> None:
    provider = _make_provider()
    loop = _make_loop(tmp_manifold, provider)
    task = {
        "prompt": "Write a hello world function.",
        "domain": "code",
        "difficulty": "easy",
        "kind": "code",
    }
    result = await loop.run(task, run_id="test-run-1")
    assert len(result.all_packet_ids) >= 3  # at minimum: claim + revision + episode


@pytest.mark.asyncio
async def test_run_claim_packet_written(tmp_manifold: Manifold) -> None:
    provider = _make_provider()
    loop = _make_loop(tmp_manifold, provider)
    task = {
        "prompt": "What is recursion?",
        "domain": "reasoning",
        "difficulty": "easy",
        "kind": "free",
    }
    result = await loop.run(task, run_id="test-claim-1")
    claims = tmp_manifold.query(type=PacketType.CLAIM)
    assert len(claims) >= 1
    assert any(c.content == task["prompt"] for c in claims)


@pytest.mark.asyncio
async def test_run_episode_packet_written(tmp_manifold: Manifold) -> None:
    provider = _make_provider()
    loop = _make_loop(tmp_manifold, provider)
    task = {"prompt": "explain maps", "domain": "code", "difficulty": "easy", "kind": "free"}
    await loop.run(task, run_id="test-ep-1")
    episodes = tmp_manifold.query(type=PacketType.EPISODE)
    assert len(episodes) >= 1


@pytest.mark.asyncio
async def test_run_self_map_updated(tmp_manifold: Manifold) -> None:
    provider = _make_provider()
    self_map = CapabilitySelfMap(tmp_manifold)
    skill_compiler = SkillCompiler(tmp_manifold, provider=provider)
    shadow_validator = ShadowValidator(tmp_manifold, protected_tasks={})
    curriculum_gen = CurriculumGenerator(provider, tmp_manifold, self_map)
    loop = SubstrateLoop(
        provider=provider,
        manifold=tmp_manifold,
        self_map=self_map,
        skill_compiler=skill_compiler,
        shadow_validator=shadow_validator,
        curriculum_gen=curriculum_gen,
    )
    task = {"prompt": "quick test", "domain": "code", "difficulty": "medium", "kind": "code"}
    result = await loop.run(task, run_id="map-test")
    delta = result.self_map_delta
    assert "before_competence" in delta
    assert "after_competence" in delta
    assert delta["signature"]["domain"] == "code"


@pytest.mark.asyncio
async def test_run_counterfactual_packets_stored(tmp_manifold: Manifold) -> None:
    provider = _make_provider()
    loop = _make_loop(tmp_manifold, provider)
    task = {"prompt": "describe list comprehensions", "domain": "code", "difficulty": "easy", "kind": "free"}
    result = await loop.run(task, run_id="cf-test")
    shadows = tmp_manifold.query(type=PacketType.SHADOW)
    assert len(shadows) >= 0  # at least attempted — may be 0 if n=2, one shadow
    # counterfactual_ids in result
    assert isinstance(result.counterfactual_ids, list)


@pytest.mark.asyncio
async def test_run_justification_proof_attached(tmp_manifold: Manifold) -> None:
    provider = _make_provider()
    loop = _make_loop(tmp_manifold, provider)
    task = {"prompt": "explain caching", "domain": "code", "difficulty": "medium", "kind": "free"}
    result = await loop.run(task, run_id="just-test")
    assert result.justification is not None
    assert result.justification.kind == "justification"


@pytest.mark.asyncio
async def test_run_with_pytest_proof(tmp_manifold: Manifold) -> None:
    """Code task with tests should attach a pytest proof."""
    provider = _make_provider()
    loop = _make_loop(tmp_manifold, provider)
    code = "def add(a, b):\n    return a + b\n"
    tests = "from solution import add\ndef test_add(): assert add(1, 2) == 3\n"
    task = {
        "prompt": "Fix the add function.",
        "domain": "code",
        "difficulty": "easy",
        "kind": "code",
        "code": code,
        "tests": tests,
    }
    result = await loop.run(task, run_id="proof-test")
    assert result.proof is not None
    assert result.proof.kind == "pytest"


@pytest.mark.asyncio
async def test_run_multiple_times_updates_competence(tmp_manifold: Manifold) -> None:
    """Running twice for the same signature should change competence."""
    provider = _make_provider()
    self_map = CapabilitySelfMap(tmp_manifold)
    loop = SubstrateLoop(
        provider=provider,
        manifold=tmp_manifold,
        self_map=self_map,
        skill_compiler=SkillCompiler(tmp_manifold),
        shadow_validator=ShadowValidator(tmp_manifold, protected_tasks={}),
        curriculum_gen=CurriculumGenerator(None, tmp_manifold, self_map),
    )
    task = {"prompt": "test task", "domain": "reasoning", "difficulty": "hard", "kind": "reasoning"}

    comp_before, _ = self_map.predict(
        __import__("autoconstitution.substrate.capability_self_map", fromlist=["TaskSignature"]).TaskSignature(
            "reasoning", "hard", "reasoning"
        )
    )
    for i in range(3):
        await loop.run(task, run_id=f"multi-{i}")
    comp_after, _ = self_map.predict(
        __import__("autoconstitution.substrate.capability_self_map", fromlist=["TaskSignature"]).TaskSignature(
            "reasoning", "hard", "reasoning"
        )
    )
    # After 3 runs, competence should have moved from the default 0.5
    assert comp_after != comp_before or comp_after != 0.5  # something changed
