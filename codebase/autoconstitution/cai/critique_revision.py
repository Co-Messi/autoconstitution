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
from dataclasses import dataclass, field
from typing import Any, Optional

from autoconstitution.cai.hierarchy import JudgeAgent, StudentAgent

logger = logging.getLogger(__name__)


@dataclass
class CritiqueResult:
    """A single Judge verdict on a Student answer."""

    round: int
    verdict: str  # "compliant" | "needs_revision" | "parse_error"
    critiques: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""

    @classmethod
    def from_judge_output(cls, round_num: int, raw: str) -> "CritiqueResult":
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

    async def run(self, prompt: str) -> RevisionResult:
        """Run the loop for a single prompt. Returns full trace."""
        initial_answer = await self.student.respond(prompt)
        current_answer = initial_answer
        critiques: list[CritiqueResult] = []
        converged = False
        rounds_used = 0

        for round_num in range(1, self.max_rounds + 1):
            rounds_used = round_num
            raw_critique = await self.judge.critique(prompt, current_answer)
            critique = CritiqueResult.from_judge_output(round_num, raw_critique)
            critiques.append(critique)

            if critique.verdict == "compliant":
                converged = True
                logger.info("converged after %d rounds", round_num)
                break

            if critique.verdict == "parse_error":
                logger.warning("judge output failed to parse on round %d", round_num)
                break

            # Ask Student to revise given critiques.
            critique_text = self._format_critiques(critique.critiques)
            revised = await self.student.revise(
                prompt=prompt,
                previous_answer=current_answer,
                critique=critique_text,
            )

            if self.skip_identical_revisions and revised.strip() == current_answer.strip():
                logger.info("revision identical to previous answer; stopping")
                break

            current_answer = revised

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
    ) -> list[RevisionResult]:
        """Run the loop across a batch of prompts with bounded concurrency."""
        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded(p: str) -> RevisionResult:
            async with semaphore:
                return await self.run(p)

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
