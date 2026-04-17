"""Tests for the CAI critique/revision loop.

These use a minimal inline ``ScriptedProvider`` so the loop is exercised
deterministically without any network. When ``providers/fake.py`` lands
(slice 3-4), these tests can migrate to the richer fake.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Any, Optional

import pytest

from autoconstitution.cai import (
    CritiqueRevisionLoop,
    JudgeAgent,
    StudentAgent,
)
from autoconstitution.ui import CapturingRenderer
from autoconstitution.ui.events import (
    Event,
    LoopError,
    Revision,
    RoundStart,
)


class ScriptedProvider:
    """Pops one scripted response per ``complete()`` call. Raises if exhausted."""

    def __init__(self, responses: Iterable[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self._responses:
            raise RuntimeError("ScriptedProvider exhausted")
        return self._responses.pop(0)

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):  # pragma: no cover - unused in the non-streaming tests in this file
        text = await self.complete(prompt, system, temperature, max_tokens)
        yield text


class ExplodingProvider:
    """Raises on the N-th call. Useful for testing error propagation."""

    def __init__(self, fail_on_call: int = 1, exc: Exception | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.exc = exc or RuntimeError("provider boom")
        self.call_count = 0

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        self.call_count += 1
        if self.call_count == self.fail_on_call:
            raise self.exc
        return f"ok#{self.call_count}"

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):  # pragma: no cover - unused
        text = await self.complete(prompt, system, temperature, max_tokens)
        yield text


def _make_loop(
    student_responses: list[str],
    judge_responses: list[str],
    *,
    max_rounds: int = 3,
) -> tuple[CritiqueRevisionLoop, ScriptedProvider, ScriptedProvider]:
    student_provider = ScriptedProvider(student_responses)
    judge_provider = ScriptedProvider(judge_responses)
    student = StudentAgent(provider=student_provider)
    judge = JudgeAgent(provider=judge_provider)
    loop = CritiqueRevisionLoop(student=student, judge=judge, max_rounds=max_rounds)
    return loop, student_provider, judge_provider


class TestRunConverges:
    @pytest.mark.asyncio
    async def test_first_round_compliant_exits_immediately(self) -> None:
        loop, sp, jp = _make_loop(
            student_responses=["The sky is blue because of Rayleigh scattering."],
            judge_responses=['{"verdict": "compliant", "critiques": []}'],
        )
        result = await loop.run("Why is the sky blue?")
        assert result.converged is True
        assert result.rounds_used == 1
        assert result.initial_answer == result.final_answer
        assert len(result.critiques) == 1
        assert result.critiques[0].verdict == "compliant"

    @pytest.mark.asyncio
    async def test_converges_on_second_round(self) -> None:
        loop, sp, jp = _make_loop(
            student_responses=["bad answer", "better answer"],
            judge_responses=[
                '{"verdict": "needs_revision", "critiques":'
                ' [{"principle":"P1","quote":"bad answer","fix":"be correct",'
                '"severity":"major"}]}',
                '{"verdict": "compliant", "critiques": []}',
            ],
        )
        result = await loop.run("prompt")
        assert result.converged is True
        assert result.rounds_used == 2
        assert result.initial_answer == "bad answer"
        assert result.final_answer == "better answer"


class TestRunHitsMaxRounds:
    @pytest.mark.asyncio
    async def test_stops_at_max_rounds_without_converging(self) -> None:
        # max_rounds=3 → 1 initial + 3 revisions = 4 student calls,
        #                 3 judge calls.
        needs_revision = (
            '{"verdict": "needs_revision", "critiques":'
            ' [{"principle":"p","quote":"q","fix":"f","severity":"minor"}]}'
        )
        loop, _, _ = _make_loop(
            student_responses=["v0", "v1", "v2", "v3"],
            judge_responses=[needs_revision] * 3,
            max_rounds=3,
        )
        result = await loop.run("prompt")
        assert result.converged is False
        assert result.rounds_used == 3
        assert result.initial_answer == "v0"
        assert result.final_answer == "v3"


class TestParseErrors:
    @pytest.mark.asyncio
    async def test_parse_error_halts_loop(self) -> None:
        loop, _, _ = _make_loop(
            student_responses=["answer"],
            judge_responses=["not valid json and no braces either"],
        )
        result = await loop.run("prompt")
        assert result.converged is False
        assert len(result.critiques) == 1
        assert result.critiques[0].verdict == "parse_error"

    @pytest.mark.asyncio
    async def test_markdown_fenced_json_parses(self) -> None:
        loop, _, _ = _make_loop(
            student_responses=["answer"],
            judge_responses=['```json\n{"verdict": "compliant", "critiques": []}\n```'],
        )
        result = await loop.run("prompt")
        assert result.converged is True


class TestIdenticalRevision:
    @pytest.mark.asyncio
    async def test_identical_revision_stops_loop(self) -> None:
        loop, _, _ = _make_loop(
            student_responses=["same answer", "same answer"],
            judge_responses=[
                '{"verdict": "needs_revision", "critiques": []}',
            ],
        )
        result = await loop.run("prompt")
        assert result.converged is False
        assert result.rounds_used == 1
        assert result.final_answer == "same answer"


class TestEventEmission:
    @pytest.mark.asyncio
    async def test_events_emitted_in_order(self) -> None:
        loop, _, _ = _make_loop(
            student_responses=["answer"],
            judge_responses=['{"verdict": "compliant", "critiques": []}'],
        )
        renderer = CapturingRenderer()
        await loop.run("prompt", renderer=renderer)

        # Expected order (ignoring streamed Tokens which CapturingRenderer
        # flags as supported): RoundStart, RoleStart(student), RoleEnd(student),
        # RoleStart(judge), RoleEnd(judge), Critique, RoundEnd.
        kinds = [type(e).__name__ for e in renderer.events if type(e).__name__ != "Token"]
        assert kinds == [
            "RoundStart",
            "RoleStart",
            "RoleEnd",
            "RoleStart",
            "RoleEnd",
            "Critique",
            "RoundEnd",
        ]

    @pytest.mark.asyncio
    async def test_revision_event_carries_before_after(self) -> None:
        loop, _, _ = _make_loop(
            student_responses=["v1", "v2"],
            judge_responses=[
                '{"verdict": "needs_revision", "critiques":'
                ' [{"principle":"P1","quote":"v1","fix":"do better","severity":"major"}]}',
                '{"verdict": "compliant", "critiques": []}',
            ],
        )
        renderer = CapturingRenderer()
        await loop.run("prompt", renderer=renderer)
        revisions = [e for e in renderer.events if isinstance(e, Revision)]
        assert len(revisions) == 1
        assert revisions[0].before == "v1"
        assert revisions[0].after == "v2"
        assert revisions[0].identical is False

    @pytest.mark.asyncio
    async def test_buggy_renderer_does_not_break_loop(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        loop, _, _ = _make_loop(
            student_responses=["answer"],
            judge_responses=['{"verdict": "compliant", "critiques": []}'],
        )

        class BadRenderer:
            supports_streaming = False

            def on_event(self, event: Event) -> None:
                raise ValueError("renderer blew up")

            async def aclose(self) -> None:
                return None

        caplog.set_level(logging.ERROR)
        result = await loop.run("prompt", renderer=BadRenderer())
        assert result.converged is True
        assert any("event sink raised" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_first_event_is_round_start(self) -> None:
        loop, _, _ = _make_loop(
            student_responses=["answer"],
            judge_responses=['{"verdict": "compliant", "critiques": []}'],
        )
        renderer = CapturingRenderer()
        await loop.run("prompt", renderer=renderer)
        first = renderer.events[0]
        assert isinstance(first, RoundStart)
        assert first.prompt == "prompt"
        assert first.round == 1


class TestErrorPropagation:
    @pytest.mark.asyncio
    async def test_student_error_emits_loop_error_and_reraises(self) -> None:
        exploding = ExplodingProvider(fail_on_call=1, exc=RuntimeError("student down"))
        student = StudentAgent(provider=exploding)
        judge = JudgeAgent(provider=ScriptedProvider([]))
        loop = CritiqueRevisionLoop(student=student, judge=judge, max_rounds=3)

        renderer = CapturingRenderer()
        with pytest.raises(RuntimeError, match="student down"):
            await loop.run("prompt", renderer=renderer)

        errors = [e for e in renderer.events if isinstance(e, LoopError)]
        assert len(errors) == 1
        assert errors[0].role == "student"
        assert "student down" in errors[0].message


class TestRunBatch:
    @pytest.mark.asyncio
    async def test_batch_preserves_order(self) -> None:
        # Each prompt runs a single-round compliant loop.
        student = StudentAgent(
            provider=ScriptedProvider([f"ans-{i}" for i in range(5)])
        )
        judge = JudgeAgent(
            provider=ScriptedProvider(
                ['{"verdict": "compliant", "critiques": []}'] * 5
            )
        )
        loop = CritiqueRevisionLoop(student=student, judge=judge, max_rounds=3)

        prompts = [f"prompt-{i}" for i in range(5)]
        results = await loop.run_batch(prompts, concurrency=1)
        assert len(results) == 5
        for i, r in enumerate(results):
            assert not isinstance(r, BaseException)
            assert r.prompt == f"prompt-{i}"  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_batch_return_exceptions_keeps_good_results(self) -> None:
        # Inject an exploding student so one prompt fails, the rest succeed.
        responses = ["ok-0", "BOOM", "ok-2"]

        class OneFailProvider:
            call_count = 0

            async def complete(
                self,
                prompt: str,
                system: Optional[str] = None,
                temperature: float = 0.7,
                max_tokens: int = 2048,
            ) -> str:
                OneFailProvider.call_count += 1
                r = responses.pop(0)
                if r == "BOOM":
                    raise RuntimeError("middle prompt broke")
                return r

            async def stream(
                self,
                prompt: str,
                system: Optional[str] = None,
                temperature: float = 0.7,
                max_tokens: int = 2048,
            ):  # pragma: no cover
                yield await self.complete(prompt, system, temperature, max_tokens)

        student = StudentAgent(provider=OneFailProvider())
        judge = JudgeAgent(
            provider=ScriptedProvider(
                ['{"verdict": "compliant", "critiques": []}'] * 3
            )
        )
        loop = CritiqueRevisionLoop(student=student, judge=judge, max_rounds=3)

        results = await loop.run_batch(
            ["p0", "p1", "p2"],
            concurrency=1,
            return_exceptions=True,
        )
        assert len(results) == 3
        assert not isinstance(results[0], BaseException)
        assert isinstance(results[1], BaseException)
        assert not isinstance(results[2], BaseException)

    @pytest.mark.asyncio
    async def test_batch_respects_concurrency(self) -> None:
        max_in_flight = 0
        current = 0
        lock = asyncio.Lock()

        class CountingProvider:
            async def complete(
                self,
                prompt: str,
                system: Optional[str] = None,
                temperature: float = 0.7,
                max_tokens: int = 2048,
            ) -> str:
                nonlocal max_in_flight, current
                async with lock:
                    current += 1
                    max_in_flight = max(max_in_flight, current)
                await asyncio.sleep(0.01)
                async with lock:
                    current -= 1
                return '{"verdict": "compliant", "critiques": []}'

            async def stream(
                self,
                prompt: str,
                system: Optional[str] = None,
                temperature: float = 0.7,
                max_tokens: int = 2048,
            ):  # pragma: no cover
                yield await self.complete(prompt, system, temperature, max_tokens)

        student = StudentAgent(provider=CountingProvider())
        judge = JudgeAgent(provider=CountingProvider())
        loop = CritiqueRevisionLoop(student=student, judge=judge, max_rounds=1)

        await loop.run_batch([f"p{i}" for i in range(10)], concurrency=3)
        # With concurrency=3, at most 3 (student or judge) calls concurrent per
        # prompt; across prompts capped by semaphore. Safety margin: 6 = 3*2.
        assert max_in_flight <= 6
