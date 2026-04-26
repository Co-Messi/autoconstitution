"""Manual smoke test for LiveRenderer.

Run with: python -m tests.smoke_live (from codebase/) to watch the dashboard
unfold in your actual terminal. Not a pytest — unit tests can't see Live
output. Use this when visually verifying layout, colors, pulse, and
narrow/wide mode fallback.
"""

from __future__ import annotations

import asyncio
import time

from autoconstitution.cai import CritiqueRevisionLoop, JudgeAgent, StudentAgent
from autoconstitution.providers.fake import FakeProvider
from autoconstitution.ui.events import RatchetDecision
from autoconstitution.ui.live import LiveRenderer


async def main() -> None:
    student = StudentAgent(
        provider=FakeProvider(
            responses=[
                "Draft 1: the sky scatters short-wavelength light.",
                "Draft 2: the sky appears blue due to Rayleigh scattering; "
                "shorter wavelengths scatter more in the atmosphere.",
            ],
            chunk_delay_ms=40,
        )
    )
    judge = JudgeAgent(
        provider=FakeProvider(
            responses=[
                '{"verdict":"needs_revision","critiques":[{"principle":"P4",'
                '"quote":"Draft 1","fix":"cite Rayleigh scattering explicitly",'
                '"severity":"moderate"}]}',
                '{"verdict":"compliant","critiques":[]}',
            ],
            chunk_delay_ms=40,
        )
    )
    loop = CritiqueRevisionLoop(student=student, judge=judge, max_rounds=3)
    renderer = LiveRenderer(max_rounds=3, refresh_per_second=10)

    try:
        await loop.run("Why is the sky blue?", renderer=renderer)
        # Show a scoreboard in the footer by emitting a RatchetDecision.
        renderer.on_event(
            RatchetDecision(
                round=2,
                metric_name="rayleigh_alignment",
                decision="keep",
                score=0.91,
                previous_best=0.74,
                improvement_delta=0.17,
            )
        )
        # Let the user see the final state for a moment.
        time.sleep(1.5)
    finally:
        await renderer.aclose()


if __name__ == "__main__":
    asyncio.run(main())
