"""Tests for autoconstitution.substrate.counterfactual."""

from __future__ import annotations

import asyncio

import pytest

from autoconstitution.substrate.counterfactual import run_counterfactuals
from autoconstitution.substrate.packet import PacketType


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


async def _fixed_loop(prompt: str) -> str:
    """A loop_fn that always returns a fixed response."""
    return f"Response to: {prompt[:20]}"


async def _indexed_loop(prompt: str, _counter: list = [0]) -> str:
    """A loop_fn that returns unique responses per call."""
    _counter[0] += 1
    return f"Alternative-{_counter[0]}: response"


async def _failing_loop(prompt: str) -> str:
    raise RuntimeError("simulated branch failure")


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_winner_is_revision_packet() -> None:
    winner, shadows = await run_counterfactuals(_fixed_loop, "test prompt", n=3)
    assert winner.type == PacketType.REVISION


@pytest.mark.asyncio
async def test_shadows_are_shadow_packets() -> None:
    winner, shadows = await run_counterfactuals(_fixed_loop, "test prompt", n=3)
    assert all(s.type == PacketType.SHADOW for s in shadows)


@pytest.mark.asyncio
async def test_correct_number_of_shadows() -> None:
    winner, shadows = await run_counterfactuals(_fixed_loop, "test prompt", n=4)
    # n=4 → winner + 3 shadows
    assert len(shadows) == 3


@pytest.mark.asyncio
async def test_shadow_confidence_reduced() -> None:
    winner, shadows = await run_counterfactuals(_fixed_loop, "test prompt", n=3, rank_penalty=0.3)
    # All shadows must have confidence < winner confidence
    assert all(s.confidence < winner.confidence for s in shadows)


@pytest.mark.asyncio
async def test_shadow_confidence_decreases_with_rank() -> None:
    """Shadows at higher rank should have lower confidence."""
    async def _unique_loop(prompt: str, _c: list = [0]) -> str:
        _c[0] += 1
        return f"option-{_c[0]}"

    winner, shadows = await run_counterfactuals(_unique_loop, "prompt", n=4, rank_penalty=0.3)
    assert len(shadows) == 3
    for i in range(len(shadows) - 1):
        assert shadows[i].confidence >= shadows[i + 1].confidence


@pytest.mark.asyncio
async def test_all_branches_fail_returns_fallback() -> None:
    winner, shadows = await run_counterfactuals(_failing_loop, "doomed", n=2)
    assert winner.type == PacketType.REVISION
    assert winner.confidence <= 0.2
    assert shadows == []


@pytest.mark.asyncio
async def test_partial_failure_shadows_from_successes() -> None:
    """If some branches fail, successful ones still produce winner + shadows."""
    call_count = [0]

    async def _sometimes_fail(prompt: str) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("first branch fails")
        return f"success-{call_count[0]}"

    winner, shadows = await run_counterfactuals(_sometimes_fail, "prompt", n=3)
    assert winner.type == PacketType.REVISION
    assert winner.confidence > 0.1


@pytest.mark.asyncio
async def test_winner_metadata_set() -> None:
    winner, _ = await run_counterfactuals(_fixed_loop, "prompt", n=2)
    assert winner.metadata.get("counterfactual") is True
    assert winner.metadata.get("rank") == 0


@pytest.mark.asyncio
async def test_shadow_metadata_has_winner_id() -> None:
    winner, shadows = await run_counterfactuals(_fixed_loop, "prompt", n=3)
    for s in shadows:
        assert s.metadata.get("winner_id") == winner.id


@pytest.mark.asyncio
async def test_n_equals_one_no_shadows() -> None:
    winner, shadows = await run_counterfactuals(_fixed_loop, "prompt", n=1)
    assert winner.type == PacketType.REVISION
    assert shadows == []
