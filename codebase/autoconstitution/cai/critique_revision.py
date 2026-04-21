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
import inspect
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Union

# A scorer is a user-supplied oracle that says "keep this revision?" (True)
# or "reject it, it's worse" (False). Sync or async. Called with the previous
# answer and the revised answer — returning False causes the loop to roll
# back to the previous answer and halt, preventing a hallucinating Judge
# from degrading the Student's output into the DPO pair.
Scorer = Union[
    Callable[[str, str], bool],
    Callable[[str, str], Awaitable[bool]],
]

from autoconstitution.cai.hierarchy import JudgeAgent, StudentAgent
from autoconstitution.ui.events import (
    Critique,
    Event,
    LoopError,
    Revision,
    Role,
    RoleEnd,
    RoleStart,
    RoundEnd,
    RoundStart,
    Token,
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
        it in markdown code fences, ``<think>`` reasoning tags (MiniMax, R1),
        or prose. We try to be forgiving.
        """
        import re as _re
        # Strip <think>...</think> reasoning blocks that precede the JSON.
        # MiniMax-M2.7 and DeepSeek-R1-style models emit these.
        stripped = _re.sub(
            r"<think>.*?</think>", "", raw, flags=_re.DOTALL,
        ).strip()
        # Strip common markdown fences.
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
        scorer: Scorer | None = None,
    ) -> RevisionResult:
        """Run the loop for a single prompt. Returns full trace.

        Args:
            prompt: The user prompt to answer.
            renderer: Optional callback invoked for every lifecycle event.
                Renderers (live dashboard, line logger, JSON emitter) plug in
                here without the loop needing to know how they display.
            scorer: Optional ground-truth oracle called on every accepted
                revision as ``scorer(previous_answer, revised_answer)``.
                Sync or async. If it returns ``False``, the loop rolls the
                revision back and halts — the last-known-good answer stays
                in ``final_answer``, protecting DPO pairs from regressions
                that a hallucinating Judge would have quietly accepted.
                Pass a pytest wrapper, a rules engine, a rubric-checker —
                anything that answers "is this revision actually better?"
                For code tasks, ``autoconstitution.benchmark.tdd_loop`` is
                a pre-built scorer that uses hidden tests directly.
        """
        emit = _make_emitter(renderer)

        streaming = bool(renderer and renderer.supports_streaming)

        emit(RoundStart(round=1, prompt=prompt))

        async def _student_respond() -> str:
            return await self.student.respond(prompt)

        def _student_respond_stream() -> AsyncIterator[str]:
            return self.student.respond_stream(prompt)

        initial_answer = await _run_role(
            emit,
            role="student",
            round_num=1,
            streaming=streaming,
            complete=_student_respond,
            stream=_student_respond_stream,
        )

        current_answer = initial_answer
        critiques: list[CritiqueResult] = []
        converged = False
        rounds_used = 0

        for round_num in range(1, self.max_rounds + 1):
            rounds_used = round_num

            async def _judge_complete(ca: str = current_answer) -> str:
                return await self.judge.critique(prompt, ca)

            def _judge_stream(ca: str = current_answer) -> AsyncIterator[str]:
                return self.judge.critique_stream(prompt, ca)

            raw_critique = await _run_role(
                emit,
                role="judge",
                round_num=round_num,
                streaming=streaming,
                complete=_judge_complete,
                stream=_judge_stream,
            )
            critique = CritiqueResult.from_judge_output(round_num, raw_critique)
            critiques.append(critique)
            norm_verdict = _normalize_verdict(critique.verdict)
            emit(
                Critique(
                    round=round_num,
                    verdict=norm_verdict,
                    critique_count=len(critique.critiques),
                    raw=raw_critique,
                )
            )

            if norm_verdict == "compliant":
                converged = True
                logger.info("converged after %d rounds", round_num)
                emit(RoundEnd(round=round_num, converged=True))
                break

            if norm_verdict == "parse_error":
                # Don't abandon the case on one bad JSON — small judges often
                # recover on retry. Retry the judge next round against the same
                # answer; max_rounds still bounds total work.
                logger.warning("judge output failed to parse on round %d; retrying", round_num)
                emit(RoundEnd(round=round_num, converged=False))
                continue

            # Ask Student to revise given critiques.
            critique_text = self._format_critiques(critique.critiques)

            async def _student_revise(
                ca: str = current_answer, ct: str = critique_text
            ) -> str:
                return await self.student.revise(
                    prompt=prompt, previous_answer=ca, critique=ct,
                )

            def _student_revise_stream(
                ca: str = current_answer, ct: str = critique_text
            ) -> AsyncIterator[str]:
                return self.student.revise_stream(prompt, ca, ct)

            revised = await _run_role(
                emit,
                role="student",
                round_num=round_num,
                streaming=streaming,
                complete=_student_revise,
                stream=_student_revise_stream,
            )

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

            # Optional ground-truth gate. If the caller supplied a scorer
            # (pytest wrapper, rules engine, whatever), ask it whether the
            # revision is actually an improvement before we accept it. A
            # False verdict means the Judge's critique led the Student
            # astray — we keep the previous answer and halt the loop so
            # the DPO pair stays clean.
            if scorer is not None:
                accepted = await _run_scorer(scorer, current_answer, revised)
                if not accepted:
                    logger.info(
                        "scorer rejected revision on round %d; rolling back",
                        round_num,
                    )
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
        scorer: Scorer | None = None,
        return_exceptions: bool = False,
    ) -> list[RevisionResult] | list[RevisionResult | BaseException]:
        """Run the loop across a batch of prompts with bounded concurrency.

        Args:
            prompts: Prompts to process.
            concurrency: Maximum parallel loops.
            renderer: Optional renderer, invoked per-event across all prompts.
                Note: events from concurrent runs are interleaved, so filter by
                ``prompt`` / ``round`` if you need per-run timelines.
            scorer: Optional ground-truth oracle, same contract as
                :meth:`run`. Applied independently to each prompt.
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
                return await self.run(p, renderer=renderer, scorer=scorer)

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


async def _run_scorer(scorer: Scorer, previous: str, revised: str) -> bool:
    """Invoke a user scorer, auto-detecting sync vs async callables.

    Caller passes either ``(prev, rev) -> bool`` or ``(prev, rev) -> Awaitable[bool]``.
    A raised exception is treated as "reject" (don't propagate — a buggy
    scorer must not crash the loop); the scorer's return value is coerced
    to bool so truthy/falsy heuristics work.
    """
    try:
        result = scorer(previous, revised)
        if inspect.isawaitable(result):
            result = await result
    except Exception:  # noqa: BLE001 — user code; isolate failures
        logger.exception("scorer raised; treating as reject")
        return False
    return bool(result)


async def _run_role(
    emit: _Emitter,
    *,
    role: Role,
    round_num: int,
    streaming: bool,
    complete: Callable[[], Awaitable[str]],
    stream: Callable[[], AsyncIterator[str]],
) -> str:
    """Run one role's turn. Emits RoleStart, per-token events (if streaming),
    and RoleEnd. Provider errors are surfaced as LoopError before re-raising.
    """
    emit(RoleStart(role=role, round=round_num))
    try:
        if streaming:
            chunks: list[str] = []
            async for chunk in stream():
                if chunk:
                    emit(Token(role=role, round=round_num, text=chunk))
                    chunks.append(chunk)
            output = "".join(chunks)
        else:
            output = await complete()
    except Exception as exc:  # noqa: BLE001
        emit(LoopError(round=round_num, role=role, message=str(exc)))
        raise
    emit(RoleEnd(role=role, round=round_num, output=output))
    return output


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
    """Coerce arbitrary verdict strings into the three expected values.

    Local judges emit casing/spacing variants ("Compliant", "needs revision",
    "needs-revision"); normalize before comparing so we don't silently drop
    valid verdicts as parse_error.
    """
    key = verdict.strip().lower().replace(" ", "_").replace("-", "_")
    if key == "compliant":
        return "compliant"
    if key == "needs_revision":
        return "needs_revision"
    return "parse_error"
