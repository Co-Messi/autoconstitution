"""The benchmark runner.

One function, one job: run every case through baseline + CAI loop, score
both, aggregate. Sequential for v1 — the coding scorer already spawns
subprocesses, and stacking concurrency on top has diminishing returns
with real bugs (resource limits, nondeterministic ordering). Concurrency
is a follow-up.

Renderers opt in via the ``renderer`` parameter. A missing renderer makes
the runner silent — useful for tests; obnoxious for a 10-minute real run,
which is why the CLI always wires one in.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from autoconstitution.benchmark.bootstrap import bootstrap_ci_mean
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
    Scorer,
    ScoreResult,
)
from autoconstitution.cai.critique_revision import CritiqueRevisionLoop

logger = logging.getLogger(__name__)

BenchEventSink = Callable[[BenchEvent], None]


async def run_benchmark(
    cases: list[BenchCase],
    scorer: Scorer,
    loop: CritiqueRevisionLoop,
    *,
    on_event: BenchEventSink | None = None,
) -> BenchReport:
    """Run every case through baseline + CAI loop, score both, aggregate.

    Args:
        cases: Benchmark cases to run, in input order.
        scorer: Returns a :class:`ScoreResult` for a ``(case, answer)`` pair.
        loop: Configured :class:`CritiqueRevisionLoop`. The Student's provider
            is reused to generate the one-shot baseline, so "did the critique
            cycle beat a Student alone" is the comparison reported.
        on_event: Optional callback invoked for every benchmark lifecycle
            event. Use this to drive progress bars or log lines. A buggy
            sink is isolated — if it raises, we log and continue.

    Returns:
        A :class:`BenchReport` with per-case outcomes and aggregates. An
        empty ``cases`` list returns a zeroed report — the runner never
        raises on empty input.
    """
    emit = _make_emitter(on_event)
    emit(BenchStart(n_cases=len(cases), scorer_name=scorer.name))

    outcomes: list[CaseOutcome] = []
    for index, case in enumerate(cases):
        emit(BenchCaseStart(index=index, case=case))

        started = time.monotonic()

        before_answer = await _run_baseline(loop, case)
        before_score = await _safe_score(scorer, case, before_answer)
        emit(
            BenchPreScored(
                index=index,
                case_id=case.id,
                answer=before_answer,
                score=before_score,
            )
        )

        cai_result = await loop.run(case.prompt)
        after_answer = cai_result.final_answer
        after_score = await _safe_score(scorer, case, after_answer)
        emit(
            BenchPostScored(
                index=index,
                case_id=case.id,
                answer=after_answer,
                score=after_score,
                rounds_used=cai_result.rounds_used,
                converged=cai_result.converged,
            )
        )

        elapsed = time.monotonic() - started
        outcome = CaseOutcome(
            case=case,
            before_answer=before_answer,
            after_answer=after_answer,
            before_score=before_score,
            after_score=after_score,
            rounds_used=cai_result.rounds_used,
            converged=cai_result.converged,
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

    report = _aggregate(outcomes, scorer_name=scorer.name)
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


async def _run_baseline(loop: CritiqueRevisionLoop, case: BenchCase) -> str:
    """Produce a one-shot Student answer with no critique/revise cycle.

    We reuse the loop's own Student so the baseline is the same provider
    and prompt shape the CAI run starts from — only the critique pipeline
    differs.
    """
    return await loop.student.respond(case.prompt)


async def _safe_score(
    scorer: Scorer, case: BenchCase, answer: str
) -> ScoreResult:
    """Call the scorer and turn any raised exception into a zero-score result.

    Scorers are contractually required not to raise, but we defend against
    misbehaving implementations so one bad case can't abort the whole run.
    """
    try:
        result = await scorer.score(case, answer)
    except Exception as exc:  # noqa: BLE001
        logger.exception("scorer %r raised on case %r", scorer.name, case.id)
        return ScoreResult(
            score=0.0,
            detail=f"scorer raised {type(exc).__name__}: {exc}",
            passed=False,
        )
    return ScoreResult(
        score=_clamp_unit(result.score),
        detail=result.detail,
        passed=result.passed,
    )


def _clamp_unit(x: float) -> float:
    """Clamp to ``[0.0, 1.0]`` — scorers that drift outside the contract still
    produce usable aggregates."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _aggregate(outcomes: list[CaseOutcome], *, scorer_name: str) -> BenchReport:
    """Fold per-case outcomes into the aggregate report."""
    if not outcomes:
        return BenchReport(
            outcomes=[],
            scorer_name=scorer_name,
            aggregate_before=0.0,
            aggregate_after=0.0,
            delta=0.0,
            wins=0,
            ties=0,
            losses=0,
            ci95=(0.0, 0.0),
        )

    n = len(outcomes)
    before_total = sum(o.before_score.score for o in outcomes)
    after_total = sum(o.after_score.score for o in outcomes)
    aggregate_before = before_total / n
    aggregate_after = after_total / n
    delta = aggregate_after - aggregate_before

    wins = sum(1 for o in outcomes if o.verdict == "win")
    ties = sum(1 for o in outcomes if o.verdict == "tie")
    losses = sum(1 for o in outcomes if o.verdict == "loss")

    deltas = [o.delta for o in outcomes]
    ci95 = bootstrap_ci_mean(deltas)

    return BenchReport(
        outcomes=outcomes,
        scorer_name=scorer_name,
        aggregate_before=aggregate_before,
        aggregate_after=aggregate_after,
        delta=delta,
        wins=wins,
        ties=ties,
        losses=losses,
        ci95=ci95,
    )


def _make_emitter(sink: BenchEventSink | None) -> BenchEventSink:
    """Return a safe emitter that logs sink exceptions instead of raising."""
    if sink is None:
        return _noop_emit

    def _emit(event: BenchEvent) -> None:
        try:
            sink(event)
        except Exception:  # noqa: BLE001
            logger.exception("bench event sink raised for %s", type(event).__name__)

    return _emit


def _noop_emit(event: BenchEvent) -> None:  # noqa: ARG001
    """Null emitter used when no sink is registered."""


__all__ = ["run_benchmark"]
