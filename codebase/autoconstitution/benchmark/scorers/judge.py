"""Judge scorer: rate an answer against a rubric using a separate LLM.

Given a :class:`BenchCase` with ``metadata['rubric']`` and the student's
answer, the judge provider returns a JSON object
``{"score": float, "reasoning": str}`` that the scorer clamps to
``[0.0, 1.0]``.

The judge is a *separate* provider from the one driving the CAI loop —
you don't want the Student grading its own homework. Pass it in at
construction time.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Protocol

from autoconstitution.benchmark.protocol import BenchCase, ScoreResult


class _JudgeProvider(Protocol):
    """Structural subset of ``LLMProvider`` that judge scorer needs."""

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str: ...


_DEFAULT_SYSTEM = """You are a careful grader. Evaluate the answer against the rubric.

You MUST reply with a single JSON object, and nothing else, with exactly these fields:

    {"score": <float between 0.0 and 1.0>, "reasoning": "<one-sentence justification>"}

Score 1.0 means the answer fully satisfies the rubric; 0.0 means it doesn't address
the rubric at all. Be honest — overly generous grades make the benchmark useless.
Do NOT wrap the JSON in markdown fences. Do NOT add commentary outside the JSON."""


_JSON_OBJECT = re.compile(r"\{[^{}]*\}", flags=re.DOTALL)


class JudgeScorer:
    """Rate an answer against a rubric using a separate LLM."""

    name: str = "judge"

    def __init__(
        self,
        *,
        provider: _JudgeProvider,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> None:
        self._provider = provider
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def score(self, case: BenchCase, answer: str) -> ScoreResult:
        rubric = case.metadata.get("rubric")
        if not isinstance(rubric, str) or not rubric.strip():
            return ScoreResult(
                score=0.0,
                detail=f"case {case.id!r} has no 'rubric' in metadata",
                passed=None,
            )

        user_prompt = _build_prompt(case.prompt, rubric, answer)

        try:
            raw = await self._provider.complete(
                prompt=user_prompt,
                system=self._system_prompt,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            return ScoreResult(
                score=0.0,
                detail=f"judge provider raised: {type(exc).__name__}: {exc}",
                passed=None,
            )

        score, reasoning, parse_error = _parse_judge_output(raw)
        if parse_error is not None:
            return ScoreResult(
                score=0.0,
                detail=f"couldn't parse judge output: {parse_error}. Raw: {raw[:200]!r}",
                passed=None,
            )

        clamped = max(0.0, min(1.0, score))
        detail = reasoning or "(no reasoning provided)"
        if clamped != score:
            detail = f"{detail} [clamped from {score:.3f}]"
        return ScoreResult(score=clamped, detail=detail, passed=None)

    async def close(self) -> None:
        # If the provider exposes close(), call it. Otherwise no-op.
        close_fn = getattr(self._provider, "close", None)
        if callable(close_fn):
            result = close_fn()
            if asyncio.iscoroutine(result):
                await result


def _build_prompt(question: str, rubric: str, answer: str) -> str:
    return (
        "=== QUESTION ===\n"
        f"{question.strip()}\n\n"
        "=== RUBRIC ===\n"
        f"{rubric.strip()}\n\n"
        "=== ANSWER TO GRADE ===\n"
        f"{answer.strip()}\n\n"
        "Return the grading JSON now."
    )


def _parse_judge_output(raw: str) -> tuple[float, str, str | None]:
    """Return ``(score, reasoning, parse_error)``.

    We try increasingly forgiving extraction strategies because real models
    often wrap JSON in prose or markdown fences despite explicit instructions.
    """
    stripped = raw.strip()

    # Strip a single leading markdown fence pair if present.
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        if lines[-1].strip().startswith("```"):
            stripped = "\n".join(lines[1:-1])
        else:
            stripped = "\n".join(lines[1:])

    parsed = _try_load_json(stripped)
    if parsed is None:
        # Find the first {...} block.
        match = _JSON_OBJECT.search(stripped)
        if match:
            parsed = _try_load_json(match.group(0))

    if not isinstance(parsed, dict):
        return 0.0, "", "no JSON object found"

    score_value = parsed.get("score")
    if not isinstance(score_value, (int, float)):
        return 0.0, "", f"'score' missing or non-numeric: {score_value!r}"

    reasoning_raw = parsed.get("reasoning", "")
    reasoning = reasoning_raw if isinstance(reasoning_raw, str) else str(reasoning_raw)
    return float(score_value), reasoning, None


def _try_load_json(text: str) -> object | None:
    try:
        parsed: object = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed


__all__ = ["JudgeScorer"]
