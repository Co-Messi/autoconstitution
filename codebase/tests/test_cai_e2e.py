"""End-to-end: real StudentAgent + JudgeAgent driven by FakeProvider.

These tests run the full CAI loop on the public API surface, proving the
pieces work together. They use consultant's FakeProvider rather than a
private test double so we're exercising the same provider surface real
users would.
"""

from __future__ import annotations

import pytest

from autoconstitution.cai import (
    CritiqueRevisionLoop,
    JudgeAgent,
    StudentAgent,
)
from autoconstitution.providers.fake import FakeProvider
from autoconstitution.ui import CapturingRenderer
from autoconstitution.ui.events import (
    Critique,
    RatchetDecision,
    Revision,
    RoleEnd,
    RoleStart,
    RoundEnd,
    RoundStart,
    Token,
)

_COMPLIANT = '{"verdict": "compliant", "critiques": []}'
_NEEDS_REVISION = (
    '{"verdict": "needs_revision", "critiques": '
    '[{"principle":"P5","quote":"bad","fix":"be concise","severity":"minor"}]}'
)


class TestFakeProviderDrivesLoop:
    @pytest.mark.asyncio
    async def test_compliant_first_round(self) -> None:
        student = StudentAgent(provider=FakeProvider(responses=["one good answer"]))
        judge = JudgeAgent(provider=FakeProvider(responses=[_COMPLIANT]))
        loop = CritiqueRevisionLoop(student=student, judge=judge, max_rounds=3)

        renderer = CapturingRenderer()
        result = await loop.run("Why is the sky blue?", renderer=renderer)

        assert result.converged is True
        assert result.rounds_used == 1
        assert result.initial_answer == "one good answer"
        assert result.chosen == "one good answer"

        # Event timeline on the happy path (filtering streamed Tokens which are
        # interleaved between RoleStart and RoleEnd when providers stream):
        non_token = [type(e) for e in renderer.events if not isinstance(e, Token)]
        assert non_token == [
            RoundStart,
            RoleStart,
            RoleEnd,
            RoleStart,
            RoleEnd,
            Critique,
            RoundEnd,
        ]

    @pytest.mark.asyncio
    async def test_revision_cycle(self) -> None:
        student = StudentAgent(
            provider=FakeProvider(responses=["first draft", "second draft"])
        )
        judge = JudgeAgent(
            provider=FakeProvider(responses=[_NEEDS_REVISION, _COMPLIANT])
        )
        loop = CritiqueRevisionLoop(student=student, judge=judge, max_rounds=3)

        renderer = CapturingRenderer()
        result = await loop.run("prompt", renderer=renderer)

        assert result.converged is True
        assert result.rounds_used == 2
        assert result.rejected == "first draft"
        assert result.chosen == "second draft"

        revisions = [e for e in renderer.events if isinstance(e, Revision)]
        assert len(revisions) == 1
        assert revisions[0].before == "first draft"
        assert revisions[0].after == "second draft"
        assert revisions[0].identical is False

    @pytest.mark.asyncio
    async def test_dpo_record_shape(self) -> None:
        student = StudentAgent(
            provider=FakeProvider(responses=["draft-v1", "draft-v2"])
        )
        judge = JudgeAgent(
            provider=FakeProvider(responses=[_NEEDS_REVISION, _COMPLIANT])
        )
        loop = CritiqueRevisionLoop(student=student, judge=judge, max_rounds=2)
        result = await loop.run("the prompt")

        record = result.to_dpo_record()
        assert record == {
            "prompt": "the prompt",
            "chosen": "draft-v2",
            "rejected": "draft-v1",
        }


class TestFakeProviderKeyedMode:
    @pytest.mark.asyncio
    async def test_keyed_responses_drive_both_agents(self) -> None:
        # Student has one prompt, so keyed by prompt substring.
        student_provider = FakeProvider(
            responses={"sky": "because Rayleigh scattering"},
            default="unknown domain",
        )
        # Judge gets a keyed response tied to a substring of its critique_prompt.
        judge_provider = FakeProvider(
            responses={"Rayleigh": _COMPLIANT},
            default='{"verdict": "parse_error"}',
        )
        student = StudentAgent(provider=student_provider)
        judge = JudgeAgent(provider=judge_provider)
        loop = CritiqueRevisionLoop(student=student, judge=judge, max_rounds=3)

        result = await loop.run("Why is the sky blue?")
        assert result.converged is True
        assert result.chosen == "because Rayleigh scattering"


class TestRendererOrchestration:
    @pytest.mark.asyncio
    async def test_renderer_sees_full_event_timeline(self) -> None:
        student = StudentAgent(
            provider=FakeProvider(responses=["v0", "v1", "v2"])
        )
        judge = JudgeAgent(
            provider=FakeProvider(
                responses=[_NEEDS_REVISION, _NEEDS_REVISION, _COMPLIANT]
            )
        )
        loop = CritiqueRevisionLoop(student=student, judge=judge, max_rounds=3)

        renderer = CapturingRenderer()
        await loop.run("prompt", renderer=renderer)

        # One RoundStart ever (emitted at loop kickoff, not per round).
        assert sum(1 for e in renderer.events if isinstance(e, RoundStart)) == 1
        # Critique emitted once per round.
        critiques = [e for e in renderer.events if isinstance(e, Critique)]
        assert len(critiques) == 3
        assert critiques[0].verdict == "needs_revision"
        assert critiques[1].verdict == "needs_revision"
        assert critiques[2].verdict == "compliant"
        # RoundEnd fires per round.
        round_ends = [e for e in renderer.events if isinstance(e, RoundEnd)]
        assert len(round_ends) == 3
        assert round_ends[-1].converged is True

    @pytest.mark.asyncio
    async def test_renderer_can_publish_ratchet_decision_externally(self) -> None:
        """Outside code pushing RatchetDecision events still reaches the renderer.

        The loop itself doesn't emit RatchetDecision — callers that plug a
        ratchet *around* the loop do. We verify the event type round-trips
        through a renderer so external wiring stays a one-liner.
        """
        from autoconstitution.ratchet import (
            ComparisonMode,
            Ratchet,
        )

        renderer = CapturingRenderer()
        ratchet = Ratchet(
            metric_name="length",
            comparison_mode=ComparisonMode.HIGHER_IS_BETTER,
            auto_persist=False,
        )
        result = await ratchet.commit_experiment("exp_1", 0.5)

        renderer.on_event(
            RatchetDecision(
                round=1,
                metric_name="length",
                decision=result.decision.value,  # type: ignore[arg-type]
                score=result.score,
                previous_best=result.previous_best,
                improvement_delta=result.improvement_delta,
            )
        )

        ratchet_decisions = [
            e for e in renderer.events if isinstance(e, RatchetDecision)
        ]
        assert len(ratchet_decisions) == 1
        assert ratchet_decisions[0].decision == "first"
