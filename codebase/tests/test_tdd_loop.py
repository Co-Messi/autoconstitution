"""Unit tests for tdd_loop helpers and the lateral-move acceptance rule.

Integration-level tests would need a real pytest subprocess; here we test the
two behaviours the adversarial audit flagged:

1. ``_truncate_tail`` returns the last ``limit`` chars (not the first), so the
   Student sees the error at the bottom of pytest stdout instead of the
   platform/plugin banner at the top.
2. ``run_tdd_benchmark`` accepts lateral moves (same score, different failure
   signature) so the revision prompt doesn't loop on identical context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from autoconstitution.benchmark.protocol import BenchCase
from autoconstitution.benchmark.tdd_loop import (
    _truncate,
    _truncate_tail,
    run_tdd_benchmark,
)


class TestTruncateTail:
    def test_short_text_unchanged(self) -> None:
        assert _truncate_tail("hello", 10) == "hello"

    def test_long_text_keeps_tail(self) -> None:
        text = "HEADER" * 100 + "ERROR_AT_END"
        out = _truncate_tail(text, 20)
        assert out.endswith("ERROR_AT_END")
        assert len(out) == 20
        assert out.startswith("…")

    def test_tail_differs_from_head_on_long_input(self) -> None:
        # This is the bug the audit flagged: _truncate returns head, we need
        # both behaviors depending on where the signal lives.
        text = "START" + "x" * 1000 + "END"
        head = _truncate(text, 20)
        tail = _truncate_tail(text, 20)
        assert head.startswith("START")
        assert tail.endswith("END")
        assert head != tail


class _ScriptedStudent:
    """Returns pre-queued answers in order. Raises when exhausted."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.calls = 0

    async def complete(
        self,
        prompt: str,  # noqa: ARG002
        system: str | None = None,  # noqa: ARG002
        temperature: float = 0.7,  # noqa: ARG002
        max_tokens: int = 2048,  # noqa: ARG002
    ) -> str:
        if not self._answers:
            raise RuntimeError("scripted student exhausted")
        self.calls += 1
        return self._answers.pop(0)


@dataclass
class _Score:
    score: float
    detail: str = ""
    passed: bool | None = None


def _score_sequence(outputs: list[tuple[float, str]]) -> Any:
    """Monkeypatch target for ``_score_answer`` — hands back queued (score, fail)
    tuples in order. We don't exercise the real pytest path in these unit tests.
    """
    queue = list(outputs)

    def _fake(answer: str, hidden_tests: str, setup: str | None, timeout_s: float) -> tuple[Any, str]:  # noqa: ARG001
        score, fail = queue.pop(0)
        return (_Score(score=score, passed=score >= 1.0, detail=f"score={score}"), fail)

    return _fake


@pytest.mark.asyncio
async def test_lateral_move_accepted_refreshes_failure_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same score, different failure → accept and swap the failure text so the
    next prompt carries the new signature. Without this fix, the Student keeps
    seeing the original failure on every round and regenerates the same bug.
    """
    # Baseline + 3 revisions. Scores: 0.5 baseline, 0.5 revise-1 (diff fail),
    # 0.5 revise-2 (diff fail again), 1.0 revise-3 (solved). All lateral moves
    # must propagate the new failure_text; otherwise the final round never
    # reaches the 1.0 input.
    score_queue = [
        (0.5, "FAIL_A"),
        (0.5, "FAIL_B"),
        (0.5, "FAIL_C"),
        (1.0, ""),
    ]
    monkeypatch.setattr(
        "autoconstitution.benchmark.tdd_loop._score_answer",
        _score_sequence(score_queue),
    )

    student = _ScriptedStudent(
        ["baseline_code", "revision_1", "revision_2", "revision_3"]
    )
    cases = [
        BenchCase(
            id="t1",
            prompt="solve",
            metadata={"hidden_tests": "def test_x(): assert True"},
        )
    ]
    report = await run_tdd_benchmark(cases, student, max_rounds=3, timeout_s=5)

    outcome = report.outcomes[0]
    assert outcome.after_score.score == 1.0
    assert outcome.converged is True
    assert student.calls == 4


@pytest.mark.asyncio
async def test_regression_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Score drop → keep the better answer; current_fail stays put."""
    score_queue = [
        (0.5, "FAIL_A"),  # baseline
        (0.0, "FAIL_B"),  # revise — strictly worse, should be discarded
        (0.0, "FAIL_C"),  # revise again — also worse
        (0.0, "FAIL_D"),  # revise again — still worse
    ]
    monkeypatch.setattr(
        "autoconstitution.benchmark.tdd_loop._score_answer",
        _score_sequence(score_queue),
    )

    student = _ScriptedStudent(
        ["baseline", "regression_1", "regression_2", "regression_3"]
    )
    cases = [
        BenchCase(
            id="t1",
            prompt="solve",
            metadata={"hidden_tests": "def test_x(): assert True"},
        )
    ]
    report = await run_tdd_benchmark(cases, student, max_rounds=3, timeout_s=5)

    outcome = report.outcomes[0]
    assert outcome.after_score.score == 0.5
    assert outcome.after_answer == "baseline"
    assert outcome.converged is False


@pytest.mark.asyncio
async def test_lateral_with_empty_fail_not_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty revised_fail on a lateral score doesn't count as progress — that
    would poison the next prompt with a blank failure block. Only lateral moves
    with a real new failure signature get accepted.
    """
    score_queue = [
        (0.5, "FAIL_A"),
        (0.5, ""),  # same score, empty fail — should NOT be accepted
        (0.5, "FAIL_A"),  # same score, SAME fail — should NOT be accepted
    ]
    monkeypatch.setattr(
        "autoconstitution.benchmark.tdd_loop._score_answer",
        _score_sequence(score_queue),
    )

    student = _ScriptedStudent(["baseline", "junk_1", "junk_2"])
    cases = [
        BenchCase(
            id="t1",
            prompt="solve",
            metadata={"hidden_tests": "def test_x(): assert True"},
        )
    ]
    report = await run_tdd_benchmark(cases, student, max_rounds=2, timeout_s=5)

    outcome = report.outcomes[0]
    assert outcome.after_answer == "baseline"
    assert outcome.after_score.score == 0.5


@pytest.mark.asyncio
async def test_lateral_cycle_detected_and_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A→B→A cycle: the second FAIL_A must be rejected as oscillation, not
    accepted as lateral progress. Without cycle detection, the Student burns
    the rounds budget flipping between two equally-broken solutions.
    """
    score_queue = [
        (0.5, "FAIL_A"),  # baseline
        (0.5, "FAIL_B"),  # lateral, new signature — accepted
        (0.5, "FAIL_A"),  # back to A — must be rejected as cycle
        (0.5, "FAIL_B"),  # back to B — also rejected
    ]
    monkeypatch.setattr(
        "autoconstitution.benchmark.tdd_loop._score_answer",
        _score_sequence(score_queue),
    )

    student = _ScriptedStudent(["baseline", "rev_B", "rev_A_again", "rev_B_again"])
    cases = [
        BenchCase(
            id="t1",
            prompt="solve",
            metadata={"hidden_tests": "def test_x(): assert True"},
        )
    ]
    report = await run_tdd_benchmark(cases, student, max_rounds=3, timeout_s=5)

    outcome = report.outcomes[0]
    # After accepting B on round 1, round 2's FAIL_A is a known cycle — reject.
    # Round 3's FAIL_B is also a cycle. Final state should be rev_B (FAIL_B).
    assert outcome.after_answer == "rev_B"
    assert outcome.after_score.score == 0.5
