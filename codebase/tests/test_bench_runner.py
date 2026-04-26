"""Tests for the benchmark runner.

Uses an inline ``FixedScorer`` that returns predetermined scores, and
FakeProvider for the loop — so these tests exercise every branch of the
runner without depending on the real coding/judge scorers (consultant's
slice).
"""

from __future__ import annotations

import pytest

from autoconstitution.benchmark import (
    BenchCase,
    CaseOutcome,
    ScoreResult,
    run_benchmark,
)
from autoconstitution.benchmark.events import (
    BenchCaseEnd,
    BenchCaseStart,
    BenchEnd,
    BenchEvent,
    BenchPostScored,
    BenchPreScored,
    BenchStart,
)
from autoconstitution.cai import CritiqueRevisionLoop, JudgeAgent, StudentAgent
from autoconstitution.providers.fake import FakeProvider


_COMPLIANT = '{"verdict": "compliant", "critiques": []}'
_NEEDS_REVISION = (
    '{"verdict": "needs_revision", "critiques":'
    ' [{"principle":"P5","quote":"x","fix":"y","severity":"minor"}]}'
)


class FixedScorer:
    """Scores by lookup on a ``(case_id, answer_substring)`` mapping.

    If no entry matches, returns 0.0. Lets each test set up precisely the
    before/after scores it wants to exercise.
    """

    name = "fixed"

    def __init__(self, scores: dict[tuple[str, str], float]) -> None:
        self._scores = scores

    async def score(self, case: BenchCase, answer: str) -> ScoreResult:
        for (case_id, marker), value in self._scores.items():
            if case_id == case.id and marker in answer:
                return ScoreResult(score=value, detail=f"matched {marker!r}", passed=value >= 1.0)
        return ScoreResult(score=0.0, detail="no match", passed=False)

    async def close(self) -> None:
        return None


class BoomScorer:
    """Scorer that raises. Proves the runner survives it."""

    name = "boom"

    async def score(self, case: BenchCase, answer: str) -> ScoreResult:
        raise RuntimeError(f"boom on {case.id}")

    async def close(self) -> None:
        return None


def _make_loop(
    student_responses: list[str], judge_responses: list[str]
) -> CritiqueRevisionLoop:
    student = StudentAgent(provider=FakeProvider(responses=student_responses))
    judge = JudgeAgent(provider=FakeProvider(responses=judge_responses))
    return CritiqueRevisionLoop(student=student, judge=judge, max_rounds=3)


class TestRunnerHappyPath:
    @pytest.mark.asyncio
    async def test_single_case_win(self) -> None:
        # Each case triggers a baseline call (one-shot Student) PLUS the loop's
        # own Student calls (initial + one revise). So queue size = baseline (1)
        # + loop initial (1) + loop revise (1) = 3.
        case = BenchCase(id="c1", prompt="prompt")
        scorer = FixedScorer({("c1", "better"): 1.0, ("c1", "bad"): 0.0})
        loop = _make_loop(
            student_responses=["bad baseline", "bad initial", "better revision"],
            judge_responses=[_NEEDS_REVISION, _COMPLIANT],
        )

        report = await run_benchmark([case], scorer, loop)

        assert report.n == 1
        assert report.wins == 1
        assert report.ties == 0
        assert report.losses == 0
        assert report.aggregate_before == pytest.approx(0.0)
        assert report.aggregate_after == pytest.approx(1.0)
        assert report.delta == pytest.approx(1.0)
        assert report.outcomes[0].verdict == "win"

    @pytest.mark.asyncio
    async def test_single_case_loss(self) -> None:
        # Baseline scores higher than revised — CAI hurt this case.
        case = BenchCase(id="c1", prompt="prompt")
        scorer = FixedScorer({("c1", "good"): 1.0, ("c1", "worse"): 0.2})
        loop = _make_loop(
            student_responses=["good baseline", "good initial", "worse revision"],
            judge_responses=[_NEEDS_REVISION, _COMPLIANT],
        )

        report = await run_benchmark([case], scorer, loop)

        assert report.wins == 0
        assert report.losses == 1
        assert report.delta < 0
        assert report.outcomes[0].verdict == "loss"

    @pytest.mark.asyncio
    async def test_single_case_tie(self) -> None:
        case = BenchCase(id="c1", prompt="prompt")
        scorer = FixedScorer({("c1", "same"): 0.5})
        loop = _make_loop(
            student_responses=["same draft", "same draft"],
            judge_responses=[_COMPLIANT],
        )

        report = await run_benchmark([case], scorer, loop)

        assert report.wins == 0
        assert report.ties == 1
        assert report.losses == 0
        assert report.delta == pytest.approx(0.0)
        assert report.outcomes[0].verdict == "tie"


class TestRunnerEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_cases_returns_zeroed_report(self) -> None:
        scorer = FixedScorer({})
        loop = _make_loop(student_responses=[], judge_responses=[])

        report = await run_benchmark([], scorer, loop)

        assert report.n == 0
        assert report.outcomes == []
        assert report.aggregate_before == 0.0
        assert report.aggregate_after == 0.0
        assert report.delta == 0.0
        assert report.wins == 0
        assert report.ties == 0
        assert report.losses == 0
        assert report.ci95 == (0.0, 0.0)

    @pytest.mark.asyncio
    async def test_scorer_that_raises_is_treated_as_zero(self) -> None:
        case = BenchCase(id="c1", prompt="prompt")
        scorer = BoomScorer()
        loop = _make_loop(
            student_responses=["any", "any"],
            judge_responses=[_COMPLIANT],
        )

        report = await run_benchmark([case], scorer, loop)

        assert report.n == 1
        outcome = report.outcomes[0]
        assert outcome.before_score.score == 0.0
        assert outcome.after_score.score == 0.0
        assert outcome.before_score.passed is False
        assert "boom" in outcome.before_score.detail.lower()

    @pytest.mark.asyncio
    async def test_score_values_are_clamped_to_unit(self) -> None:
        class BadRangeScorer:
            name = "bad"

            async def score(self, case: BenchCase, answer: str) -> ScoreResult:
                return ScoreResult(score=1.5, detail="over", passed=True)

            async def close(self) -> None:
                return None

        case = BenchCase(id="c1", prompt="prompt")
        loop = _make_loop(
            student_responses=["any", "any"],
            judge_responses=[_COMPLIANT],
        )

        report = await run_benchmark([case], BadRangeScorer(), loop)

        assert report.aggregate_before == pytest.approx(1.0)
        assert report.aggregate_after == pytest.approx(1.0)


class TestRunnerEvents:
    @pytest.mark.asyncio
    async def test_events_emitted_in_order(self) -> None:
        case = BenchCase(id="c1", prompt="prompt")
        scorer = FixedScorer({("c1", "ok"): 1.0})
        loop = _make_loop(
            student_responses=["ok", "ok"],
            judge_responses=[_COMPLIANT],
        )

        events: list[BenchEvent] = []
        await run_benchmark([case], scorer, loop, on_event=events.append)

        types = [type(e).__name__ for e in events]
        # Full sequence: BenchStart, BenchCaseStart, BenchPreScored,
        # BenchPostScored, BenchCaseEnd, BenchEnd.
        assert types == [
            "BenchStart",
            "BenchCaseStart",
            "BenchPreScored",
            "BenchPostScored",
            "BenchCaseEnd",
            "BenchEnd",
        ]

    @pytest.mark.asyncio
    async def test_case_end_carries_verdict(self) -> None:
        case = BenchCase(id="c1", prompt="prompt")
        scorer = FixedScorer({("c1", "good"): 1.0, ("c1", "bad"): 0.0})
        loop = _make_loop(
            student_responses=["bad baseline", "bad initial", "good revision"],
            judge_responses=[_NEEDS_REVISION, _COMPLIANT],
        )

        events: list[BenchEvent] = []
        await run_benchmark([case], scorer, loop, on_event=events.append)
        end = next(e for e in events if isinstance(e, BenchCaseEnd))
        assert end.verdict == "win"
        assert end.delta == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_bench_end_matches_report(self) -> None:
        case = BenchCase(id="c1", prompt="prompt")
        scorer = FixedScorer({("c1", "ok"): 0.7})
        loop = _make_loop(
            student_responses=["ok", "ok"],
            judge_responses=[_COMPLIANT],
        )

        events: list[BenchEvent] = []
        report = await run_benchmark([case], scorer, loop, on_event=events.append)
        bench_end = next(e for e in events if isinstance(e, BenchEnd))
        assert bench_end.n_cases == report.n
        assert bench_end.wins == report.wins
        assert bench_end.losses == report.losses
        assert bench_end.delta == pytest.approx(report.delta)

    @pytest.mark.asyncio
    async def test_buggy_event_sink_does_not_break_run(self) -> None:
        case = BenchCase(id="c1", prompt="prompt")
        scorer = FixedScorer({("c1", "ok"): 1.0})
        loop = _make_loop(
            student_responses=["ok", "ok"],
            judge_responses=[_COMPLIANT],
        )

        def bad_sink(event: BenchEvent) -> None:
            raise ValueError("sink exploded")

        # Must not raise.
        report = await run_benchmark([case], scorer, loop, on_event=bad_sink)
        assert report.n == 1


class TestAggregation:
    @pytest.mark.asyncio
    async def test_multiple_cases_aggregate(self) -> None:
        cases = [
            BenchCase(id="w", prompt="p1"),
            BenchCase(id="l", prompt="p2"),
            BenchCase(id="t", prompt="p3"),
        ]
        scorer = FixedScorer(
            {
                ("w", "bad"): 0.0,
                ("w", "good"): 1.0,
                ("l", "good"): 1.0,
                ("l", "bad"): 0.0,
                ("t", "same"): 0.5,
            }
        )
        # Per case: baseline (1) + loop initial (1) + revise (0-1). Total below.
        loop = _make_loop(
            student_responses=[
                # w: baseline, initial, revise → 3 students, 2 judges
                "bad w baseline", "bad w initial", "good w revision",
                # l: baseline, initial, revise → 3 students, 2 judges
                "good l baseline", "good l initial", "bad l revision",
                # t: baseline, initial (loop compliant on first round) → 2 students, 1 judge
                "same t baseline", "same t initial",
            ],
            judge_responses=[
                _NEEDS_REVISION, _COMPLIANT,  # w
                _NEEDS_REVISION, _COMPLIANT,  # l
                _COMPLIANT,                    # t
            ],
        )

        report = await run_benchmark(cases, scorer, loop)

        assert report.n == 3
        assert report.wins == 1
        assert report.losses == 1
        assert report.ties == 1
        verdicts = [o.verdict for o in report.outcomes]
        assert verdicts == ["win", "loss", "tie"]

    @pytest.mark.asyncio
    async def test_ci95_is_finite_on_nontrivial_input(self) -> None:
        # Vary the after-score per case so deltas aren't identical; a
        # degenerate (all-equal) input collapses bootstrap to a point.
        cases = [BenchCase(id=f"c{i}", prompt="p") for i in range(10)]
        after_scores = [0.3, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0]
        scorer = FixedScorer(
            {(f"c{i}", "after"): after_scores[i] for i in range(10)}
            | {(f"c{i}", "before"): 0.3 for i in range(10)}
        )

        student_responses: list[str] = []
        for _ in range(10):
            student_responses.append("before baseline")
            student_responses.append("before initial")
            student_responses.append("after revision")
        judge_responses = [_NEEDS_REVISION, _COMPLIANT] * 10

        loop = _make_loop(student_responses, judge_responses)

        report = await run_benchmark(cases, scorer, loop)

        lo, hi = report.ci95
        assert lo <= report.delta <= hi
        assert hi > lo  # non-degenerate interval


class TestCaseOutcome:
    def test_verdict_win_on_positive_delta(self) -> None:
        outcome = CaseOutcome(
            case=BenchCase(id="c", prompt="p"),
            before_answer="x",
            after_answer="y",
            before_score=ScoreResult(score=0.1, detail="", passed=False),
            after_score=ScoreResult(score=0.9, detail="", passed=True),
            rounds_used=1,
            converged=True,
            elapsed_s=0.01,
        )
        assert outcome.verdict == "win"
        assert outcome.delta == pytest.approx(0.8)

    def test_verdict_tie_when_within_epsilon(self) -> None:
        outcome = CaseOutcome(
            case=BenchCase(id="c", prompt="p"),
            before_answer="x",
            after_answer="y",
            before_score=ScoreResult(score=0.5, detail="", passed=None),
            after_score=ScoreResult(score=0.5 + 1e-12, detail="", passed=None),
            rounds_used=1,
            converged=True,
            elapsed_s=0.01,
        )
        assert outcome.verdict == "tie"
