"""
autoconstitution.substrate.counterfactual
==========================================

Counterfactual shadow execution: runs N alternative revisions in parallel,
keeps the best winner, stores rejected alternatives as SHADOW packets.

MANIFOLD §2 feature 2 ("Counterfactual shadow execution"): "When the planner
picks an action, it simulates that action plus N counterfactual alternatives in
the world model. The counterfactuals are retained as 'what would have happened'
memories with lower confidence. This is the mechanism that lets MANIFOLD learn
from roads not taken and converts planning waste into training signal."

MANIFOLD §3.5: "The deliberation engine branches — exploring candidate latent
trajectories in parallel, scoring them by expected free energy, and retaining
the best."
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Awaitable
from typing import Any

from autoconstitution.substrate.packet import (
    Packet,
    make_revision,
    make_shadow,
)

logger = logging.getLogger(__name__)


async def run_counterfactuals(
    loop_fn: Callable[[str], Awaitable[str]],
    prompt: str,
    n: int = 3,
    rank_penalty: float = 0.3,
) -> tuple[Packet, list[Packet]]:
    """Run N alternative responses in parallel and return the winner + shadows.

    MANIFOLD §3.5: parallel candidate trajectory exploration with the best
    selected by scoring; rejected alternatives become shadow memories.

    All N calls run via asyncio.gather — no sequential bottleneck. The winner
    is the first result (we cannot rank without a scorer; callers can override
    by post-processing shadows). Shadows get base confidence reduced by
    ``rank_penalty`` per rank below winner.

    Args:
        loop_fn:      Async callable that takes a prompt and returns a string.
        prompt:       The prompt string to run N times.
        n:            Number of alternatives to generate.
        rank_penalty: Per-rank confidence penalty applied to shadows.

    Returns:
        ``(winner_packet, shadow_packets)``
        - winner_packet: REVISION packet with confidence 0.9.
        - shadow_packets: SHADOW packets, one per non-winner result.
    """
    tasks = [loop_fn(prompt) for _ in range(n)]
    results: list[str | BaseException] = await asyncio.gather(
        *tasks, return_exceptions=True
    )

    # Filter out exceptions — log and skip
    ok_results: list[str] = []
    for r in results:
        if isinstance(r, BaseException):
            logger.warning("counterfactual branch raised: %s", r)
        else:
            ok_results.append(r)

    if not ok_results:
        # All branches failed — return a shell revision packet
        fallback = make_revision(
            f"[counterfactual fallback] No branches succeeded for prompt: {prompt[:80]}",
            confidence=0.1,
            metadata={"counterfactual": True, "n_tried": n, "n_ok": 0},
        )
        return fallback, []

    # First OK result is the "winner" — simplest possible selection strategy.
    # Real MANIFOLD would score by expected free energy; here caller can
    # post-process or supply a ranked loop_fn that returns sorted results.
    winner_text = ok_results[0]
    winner = make_revision(
        winner_text,
        confidence=0.9,
        metadata={"counterfactual": True, "n_tried": n, "n_ok": len(ok_results), "rank": 0},
    )

    shadows: list[Packet] = []
    for rank, alt_text in enumerate(ok_results[1:], start=1):
        shadow_confidence = max(0.05, 0.9 - rank * rank_penalty)
        shadow = make_shadow(
            alt_text,
            confidence=shadow_confidence,
            metadata={
                "counterfactual": True,
                "n_tried": n,
                "rank": rank,
                "winner_id": winner.id,
            },
        )
        shadows.append(shadow)

    logger.debug(
        "counterfactuals: %d tried, %d ok, winner len=%d, shadows=%d",
        n, len(ok_results), len(winner_text), len(shadows),
    )

    return winner, shadows


__all__ = ["run_counterfactuals"]
