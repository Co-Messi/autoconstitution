"""
Critique → Revision loop.

The inner loop of Constitutional AI Phase 1 (SL-CAI):

    for prompt in batch:
        answer        = Student.respond(prompt)
        while True:
            critique  = Judge.critique(prompt, answer)
            if critique.compliant or round >= max_rounds:
                break
            answer    = Student.revise(prompt, answer, critique)

Every loop produces:
    - an initial answer   (`rejected` candidate for DPO)
    - a final answer      (`chosen` candidate for DPO)
    - a trail of critiques (for dataset audit / constitution tuning)

Example:
    >>> loop = CritiqueRevisionLoop(student=student, judge=judge, max_rounds=3)
    >>> result = await loop.run("Why is the sky blue?")
    >>> result.chosen       # final, Judge-approved answer
    >>> result.rejected     # initial answer
    >>> result.critiques    # list of Judge verdicts
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from autoconstitution.cai.hierarchy import JudgeAgent, StudentAgent
from autoconstitution.ui.events import (
    Critique,
    Event,
    LoopError,
    Revision,
    RoleEnd,
    RoleStart,
    RoundEnd,
    RoundStart,
)
from autoconstitution.ui.protocol import Renderer

logger = logging.getLogger(__name__)


@dataclass
class CritiqueResult:
    """A single Judge verdict on a Student answer."""

    round: int
    verdict: str  # "compliant" | "needs_revision" | "parse_error"
    critiques: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""

    @classmethod
    def from_judge_output(cls, round_num: int, raw: str) -> CritiqueResult:
        """Parse a Judge's JSON-ish output into a structured verdict.

        The Judge is instructed to return JSON, but real models sometimes wrap
        it in markdown code fences or prose. We try to be forgiving.
        """
        # Strip common markdown fences.
        stripped = raw.strip()
        if stripped.startswith("```"):
            # Remove first line and possible trailing ```.
            lines = stripped.split("\n")
            stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            # Try to find the first {...} block.
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(stripped[start : end + 1])
                except json.JSONDecodeError:
                    return cls(round=round_num, verdict="parse_error", raw_response=raw)
            else:
                return cls(round=round_num, verdict="parse_error", raw_response=raw)

        return cls(
            round=round_num,
            verdict=parsed.get("verdict", "parse_error"),
            critiques=parsed.get("critiques", []),
            raw_response=raw,
        )


@dataclass
class RevisionResult:
    """Full trace of one critique-revision run."""

    prompt: str
    initial_answer: str
    final_answer: str
    critiques: list[CritiqueResult]
    rounds_used: int
    converged: bool  # True if Judge returned "compliant" before hitting max_rounds

    @property
    def rejected(self) -> str:
        """For DPO training: the initial (pre-revision) answer."""
        return self.initial_answer

    @property
    def chosen(self) -> str:
        """For DPO training: the final (post-revision) answer."""
        return self.final_answer

    def to_dpo_record(self) -> dict[str, str]:
        """Return a dict in the shape TRL's DPOTrainer expects."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }


class CritiqueRevisionLoop:
    """Drives Student ↔ Judge until convergence or max_rounds."""

    def __init__(
        self,
        student: StudentAgent,
        judge: JudgeAgent,
        *,
        max_rounds: int = 3,
        skip_identical_revisions: bool = True,
    ) -> None:
        self.student = student
        self.judge = judge
        self.max_rounds = max_rounds
        self.skip_identical_revisions = skip_identical_revisions

    async def run(
        self,
        prompt: str,
        *,
        renderer: Renderer | None = None,
    ) -> RevisionResult:
        """Run the loop for a single prompt. Returns full trace.

        Args:
            prompt: The user prompt to answer.
            on_event: Optional callback invoked for every lifecycle event.
                Renderers (live dashboard, line logger, JSON emitter) plug in
                here without the loop needing to know how they display.
        """
        emit = _make_emitter(renderer)

        emit(RoundStart(round=1, prompt=prompt))
        emit(RoleStart(role="student", round=1))
        try:
            initial_answer = await self.student.respond(prompt)
        except Exception as exc:  # noqa: BLE001 - surfaced to caller via LoopError
            emit(LoopError(round=1, role="student", message=str(exc)))
            raise
        emit(RoleEnd(role="student", round=1, output=initial_answer))

        current_answer = initial_answer
        critiques: list[CritiqueResult] = []
        converged = False
        rounds_used = 0

        for round_num in range(1, self.max_rounds + 1):
            rounds_used = round_num
            emit(RoleStart(role="judge", round=round_num))
            try:
                raw_critique = await self.judge.critique(prompt, current_answer)
            except Exception as exc:  # noqa: BLE001
                emit(LoopError(round=round_num, role="judge", message=str(exc)))
                raise
            critique = CritiqueResult.from_judge_output(round_num, raw_critique)
            critiques.append(critique)
            emit(RoleEnd(role="judge", round=round_num, output=raw_critique))
            emit(
                Critique(
                    round=round_num,
                    verdict=_normalize_verdict(critique.verdict),
                    critique_count=len(critique.critiques),
                    raw=raw_critique,
                )
            )

            if critique.verdict == "compliant":
                converged = True
                logger.info("converged after %d rounds", round_num)
                emit(RoundEnd(round=round_num, converged=True))
                break

            if critique.verdict == "parse_error":
                logger.warning("judge output failed to parse on round %d", round_num)
                emit(RoundEnd(round=round_num, converged=False))
                break

            # Ask Student to revise given critiques.
            critique_text = self._format_critiques(critique.critiques)
            emit(RoleStart(role="student", round=round_num))
            try:
                revised = await self.student.revise(
                    prompt=prompt,
                    previous_answer=current_answer,
                    critique=critique_text,
                )
            except Exception as exc:  # noqa: BLE001
                emit(LoopError(round=round_num, role="student", message=str(exc)))
                raise
            emit(RoleEnd(role="student", round=round_num, output=revised))

            identical = revised.strip() == current_answer.strip()
            emit(
                Revision(
                    round=round_num,
                    before=current_answer,
                    after=revised,
                    identical=identical,
                )
            )

            if self.skip_identical_revisions and identical:
                logger.info("revision identical to previous answer; stopping")
                emit(RoundEnd(round=round_num, converged=False))
                break

            current_answer = revised
            emit(RoundEnd(round=round_num, converged=False))

        return RevisionResult(
            prompt=prompt,
            initial_answer=initial_answer,
            final_answer=current_answer,
            critiques=critiques,
            rounds_used=rounds_used,
            converged=converged,
        )

    async def run_batch(
        self,
        prompts: list[str],
        *,
        concurrency: int = 4,
        renderer: Renderer | None = None,
        return_exceptions: bool = False,
    ) -> list[RevisionResult] | list[RevisionResult | BaseException]:
        """Run the loop across a batch of prompts with bounded concurrency.

        Args:
            prompts: Prompts to process.
            concurrency: Maximum parallel loops.
            renderer: Optional renderer, invoked per-event across all prompts.
                Note: events from concurrent runs are interleaved, so filter by
                ``prompt`` / ``round`` if you need per-run timelines.
            return_exceptions: When True, failed runs appear in the result list
                as ``BaseException`` instances (matches ``asyncio.gather``).
                When False, the first exception aborts the batch.

        Returns:
            A list the same length as ``prompts``. Each entry is either a
            :class:`RevisionResult` (success) or a ``BaseException`` (only when
            ``return_exceptions=True``).
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded(p: str) -> RevisionResult:
            async with semaphore:
                return await self.run(p, renderer=renderer)

        if return_exceptions:
            return await asyncio.gather(
                *(_bounded(p) for p in prompts),
                return_exceptions=True,
            )
        return await asyncio.gather(*(_bounded(p) for p in prompts))

    @staticmethod
    def _format_critiques(critiques: list[dict[str, Any]]) -> str:
        if not critiques:
            return "(no specific critiques provided)"
        lines = []
        for c in critiques:
            principle = c.get("principle", "?")
            quote = c.get("quote", "")
            fix = c.get("fix", "")
            severity = c.get("severity", "moderate")
            lines.append(f"- [{principle}] ({severity}) Offending: {quote!r} → Fix: {fix}")
        return "\n".join(lines)


def _make_emitter(renderer: Renderer | None) -> _Emitter:
    """Return a safe event emitter that swallows renderer exceptions.

    A buggy renderer must never break the loop — this wraps
    ``renderer.on_event`` in a try/except and logs rather than re-raises. If
    ``renderer`` is None we return a no-op, so the fast path has zero overhead
    and no branching on every event.
    """
    if renderer is None:
        return _noop_emit

    def _emit(event: Event) -> None:
        try:
            renderer.on_event(event)
        except Exception:  # noqa: BLE001 - isolate renderer bugs from loop
            logger.exception("event sink raised for %s", type(event).__name__)

    return _emit


_Emitter = Callable[[Event], None]


def _noop_emit(event: Event) -> None:  # noqa: ARG001 - sink interface
    """Null emitter used when no renderer is registered."""


def _normalize_verdict(
    verdict: str,
) -> Literal["compliant", "needs_revision", "parse_error"]:
    """Coerce arbitrary verdict strings into the three expected values."""
    if verdict == "compliant":
        return "compliant"
    if verdict == "needs_revision":
        return "needs_revision"
    return "parse_error"
