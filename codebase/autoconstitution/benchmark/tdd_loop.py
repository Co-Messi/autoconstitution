"""Test-driven revision loop: ground critique on actual pytest output.

The standard CAI loop asks a Judge LLM to critique a Student LLM's answer
against a constitution. Small-model Judges hallucinate critiques that break
working code — we measured this: 1b Student + 3b Judge on hard algo problems
produced a negative Δ because the Judge flagged "issues" that weren't real.

This module replaces the Judge with the actual failing tests. After the
Student produces an answer, we run the hidden tests; if any fail, we feed
the failure text (assertion message, traceback one-liner) into the next
Student prompt and ask for a revision. No Judge in the loop — the tests
ARE the ground truth.

Same ``BenchReport`` shape out the other side, so CLI + report rendering
work unchanged.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from autoconstitution.benchmark.events import (
    BenchCaseEnd,
    BenchCaseStart,
    BenchEnd,
    BenchEvent,
    BenchPostScored,
    BenchPreScored,
    BenchStart,
)
from autoconstitution.benchmark.protocol import (
    BenchCase,
    BenchReport,
    CaseOutcome,
    ScoreResult,
)
from autoconstitution.benchmark.runner import _aggregate
from autoconstitution.benchmark.scorers.coding import (
    _CODE_FENCE,
    _compose_test_module,
    _sanitized_env,
)

logger = logging.getLogger(__name__)

BenchEventSink = Callable[[BenchEvent], None]


class _StudentProvider:
    """Structural shape the TDD loop needs from an ``LLMProvider``."""

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str: ...


async def run_tdd_benchmark(
    cases: list[BenchCase],
    student: _StudentProvider,
    *,
    max_rounds: int = 5,
    timeout_s: float = 15.0,
    on_event: BenchEventSink | None = None,
) -> BenchReport:
    """Test-driven revision loop.

    Args:
        cases: Benchmark cases with ``metadata['hidden_tests']``.
        student: LLM provider used for both baseline and revisions.
        max_rounds: Maximum revision rounds per case. First round is the
            baseline; subsequent rounds are revisions with test feedback.
        timeout_s: pytest timeout per run.
        on_event: Optional lifecycle callback.

    Returns:
        :class:`BenchReport` — baseline score vs final score after test-driven
        revision. Aggregates, win/tie/loss, and bootstrap CI are computed
        identically to the judge-based runner.
    """
    emit = _make_emitter(on_event)
    emit(BenchStart(n_cases=len(cases), scorer_name="tdd"))

    outcomes: list[CaseOutcome] = []
    for index, case in enumerate(cases):
        emit(BenchCaseStart(index=index, case=case))
        started = time.monotonic()

        hidden_tests = case.metadata.get("hidden_tests", "")
        setup = case.metadata.get("setup")
        setup_str = setup if isinstance(setup, str) else None

        # Round 0: baseline answer.
        baseline = await student.complete(
            prompt=case.prompt, temperature=0.7, max_tokens=2048
        )
        baseline_score, baseline_fail = _score_answer(
            baseline, hidden_tests, setup_str, timeout_s
        )
        emit(
            BenchPreScored(
                index=index,
                case_id=case.id,
                answer=baseline,
                score=baseline_score,
            )
        )

        # Rounds 1..N: revise with feedback when tests fail.
        current_answer = baseline
        current_score = baseline_score
        current_fail = baseline_fail
        rounds_used = 1
        converged = current_score.score >= 1.0

        for round_num in range(1, max_rounds + 1):
            if current_score.score >= 1.0:
                break
            revision_prompt = _build_revision_prompt(
                original_prompt=case.prompt,
                previous_answer=current_answer,
                failure_text=current_fail,
            )
            revised = await student.complete(
                prompt=revision_prompt, temperature=0.7, max_tokens=2048
            )
            revised_score, revised_fail = _score_answer(
                revised, hidden_tests, setup_str, timeout_s
            )
            rounds_used = round_num + 1
            if revised_score.score > current_score.score:
                # Accept improvement.
                current_answer = revised
                current_score = revised_score
                current_fail = revised_fail
            else:
                # Don't regress — keep the better answer but keep iterating
                # against the original failure in case the next try lands it.
                pass
            if current_score.score >= 1.0:
                converged = True
                break

        emit(
            BenchPostScored(
                index=index,
                case_id=case.id,
                answer=current_answer,
                score=current_score,
                rounds_used=rounds_used,
                converged=converged,
            )
        )

        elapsed = time.monotonic() - started
        outcome = CaseOutcome(
            case=case,
            before_answer=baseline,
            after_answer=current_answer,
            before_score=baseline_score,
            after_score=current_score,
            rounds_used=rounds_used,
            converged=converged,
            elapsed_s=elapsed,
        )
        outcomes.append(outcome)
        emit(
            BenchCaseEnd(
                index=index,
                case_id=case.id,
                delta=outcome.delta,
                verdict=outcome.verdict,
                elapsed_s=elapsed,
            )
        )

    report = _aggregate(outcomes, scorer_name="tdd")
    emit(
        BenchEnd(
            n_cases=report.n,
            wins=report.wins,
            ties=report.ties,
            losses=report.losses,
            aggregate_before=report.aggregate_before,
            aggregate_after=report.aggregate_after,
            delta=report.delta,
        )
    )
    return report


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _score_answer(
    answer: str,
    hidden_tests: str,
    setup: str | None,
    timeout_s: float,
) -> tuple[ScoreResult, str]:
    """Score ``answer`` with pytest; return the score and a terse failure dump.

    The failure dump is what gets fed into the next Student prompt — it's
    truncated and cleaned up so a small model isn't drowned in traceback
    noise.
    """
    if not hidden_tests or not hidden_tests.strip():
        return (
            ScoreResult(
                score=0.0,
                detail="no hidden_tests in metadata",
                passed=False,
            ),
            "",
        )

    code, had_fence = _extract_python(answer)
    if not code.strip():
        return (
            ScoreResult(
                score=0.0, detail="no python code extracted from answer", passed=False
            ),
            "answer contained no Python code",
        )

    test_module = _compose_test_module(
        setup=setup, code=code, hidden_tests=hidden_tests
    )

    tmpdir = tempfile.mkdtemp(prefix="autoconstitution-tdd-")
    try:
        tmp_path = Path(tmpdir)
        test_file = tmp_path / "test_case.py"
        test_file.write_text(test_module, encoding="utf-8")
        env = _sanitized_env(tmp_path)

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "-v",
            "--tb=short",
            "--color=no",
            "-p",
            "no:cacheprovider",
        ]
        try:
            completed = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                timeout=timeout_s,
                cwd=tmpdir,
                env=env,
                check=False,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return (
                ScoreResult(
                    score=0.0,
                    detail=f"pytest timed out after {timeout_s:.1f}s",
                    passed=False,
                ),
                "the solution hung or timed out; ensure no infinite loops",
            )

        output = (completed.stdout or "") + (completed.stderr or "")
        passed_count, failed_count, error_count = _count_from_summary(output)
        total = passed_count + failed_count + error_count

        if total == 0:
            return (
                ScoreResult(
                    score=0.0,
                    detail="no tests collected — likely a syntax error",
                    passed=False,
                ),
                _truncate(output, 400),
            )

        score = passed_count / total
        passed = failed_count == 0 and error_count == 0
        failure_text = _extract_failure_text(output) if not passed else ""
        detail = f"{passed_count}/{total} tests passed"
        if failed_count or error_count:
            detail += f"; {failed_count} failed, {error_count} errors"
        return (
            ScoreResult(score=score, detail=detail, passed=passed),
            failure_text,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _extract_python(answer: str) -> tuple[str, bool]:
    match = _CODE_FENCE.search(answer)
    if match:
        return match.group(1), True
    return answer, False


def _count_from_summary(output: str) -> tuple[int, int, int]:
    """Read the last pytest summary line."""
    for line in output.splitlines()[::-1]:
        if "passed" in line or "failed" in line or "error" in line:
            passed = _extract_count(line, "passed")
            failed = _extract_count(line, "failed")
            errors = _extract_count(line, "error")
            if passed or failed or errors:
                return passed, failed, errors
    return 0, 0, 0


def _extract_count(line: str, keyword: str) -> int:
    match = re.search(rf"(\d+)\s+{keyword}", line)
    return int(match.group(1)) if match else 0


def _extract_failure_text(output: str) -> str:
    """Pull the FAILURES section out of pytest -v --tb=short output.

    Small models do better with tight, structured feedback than with a
    full traceback, so we return the tail of the FAILURES block capped at
    ~1200 chars.
    """
    lines = output.splitlines()
    start: int | None = None
    for i, line in enumerate(lines):
        if "FAILURES" in line and line.strip().startswith("="):
            start = i
            break
    if start is None:
        # No structured failures — fall back to a tail of stdout.
        return _truncate(output, 800)
    # End at the short summary section or end of file.
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].strip().startswith("="):
            end = i
            break
    block = "\n".join(lines[start:end]).strip()
    return _truncate(block, 1200)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _build_revision_prompt(
    *, original_prompt: str, previous_answer: str, failure_text: str
) -> str:
    """Compose the revision prompt with grounded test feedback."""
    return (
        "You previously tried to solve this problem and your solution failed "
        "some tests. Fix the implementation.\n\n"
        "=== ORIGINAL PROBLEM ===\n"
        f"{original_prompt}\n\n"
        "=== YOUR PREVIOUS SOLUTION ===\n"
        f"{previous_answer}\n\n"
        "=== FAILING TESTS (pytest output) ===\n"
        f"{failure_text}\n\n"
        "Return ONLY a corrected ```python``` code block. Do not add "
        "commentary. Address every failing test."
    )


def _make_emitter(sink: BenchEventSink | None) -> BenchEventSink:
    if sink is None:
        return _noop_emit

    def _emit(event: BenchEvent) -> None:
        try:
            sink(event)
        except Exception:  # noqa: BLE001
            logger.exception("tdd event sink raised for %s", type(event).__name__)

    return _emit


def _noop_emit(event: BenchEvent) -> None:  # noqa: ARG001
    pass


# Keep the pylint-unused-imports check happy for the stdlib modules we pulled
# in above; they're all used inside this module.
_ = os  # noqa: F841


__all__ = ["run_tdd_benchmark"]
