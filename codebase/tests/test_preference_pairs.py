"""Tests for the DPO preference-pair filter gates.

These gates exist to stop CAI noise from becoming training data:
- Traces whose final verdict was parse_error (the Student revised off garbage).
- Traces where the judge said needs_revision but provided no critique items
  (the Student revised with no actionable feedback — a random walk).
"""

from __future__ import annotations

from autoconstitution.cai.critique_revision import CritiqueResult, RevisionResult
from autoconstitution.cai.preference_pairs import (
    PreferencePairBuilder,
    _ended_in_parse_error,
)


def _make_result(
    *,
    initial: str = "short",
    final: str = "a far longer final answer that passes the edit-distance gate by a wide margin",
    critiques: list[CritiqueResult],
    rounds_used: int = 1,
    converged: bool = False,
) -> RevisionResult:
    return RevisionResult(
        prompt="p",
        initial_answer=initial,
        final_answer=final,
        critiques=critiques,
        rounds_used=rounds_used,
        converged=converged,
    )


class TestEndedInParseError:
    def test_empty_critiques_is_not_parse_error(self) -> None:
        assert not _ended_in_parse_error(_make_result(critiques=[]))

    def test_last_verdict_parse_error_flags(self) -> None:
        r = _make_result(
            critiques=[
                CritiqueResult(round=1, verdict="needs_revision", critiques=[{"fix": "x"}]),
                CritiqueResult(round=2, verdict="parse_error"),
            ],
        )
        assert _ended_in_parse_error(r)

    def test_last_verdict_compliant_does_not_flag(self) -> None:
        r = _make_result(
            critiques=[CritiqueResult(round=1, verdict="compliant")],
        )
        assert not _ended_in_parse_error(r)


class TestParseErrorGate:
    def test_parse_error_trace_dropped_by_default(self) -> None:
        builder = PreferencePairBuilder()
        r = _make_result(
            critiques=[
                CritiqueResult(round=1, verdict="needs_revision", critiques=[{"fix": "x"}]),
                CritiqueResult(round=2, verdict="parse_error"),
            ],
            rounds_used=2,
        )
        added = builder.add_results([r])
        assert added == 0
        assert len(builder) == 0

    def test_parse_error_trace_kept_when_flag_off(self) -> None:
        builder = PreferencePairBuilder(drop_parse_error_traces=False)
        r = _make_result(
            critiques=[
                CritiqueResult(round=1, verdict="needs_revision", critiques=[{"fix": "x"}]),
                CritiqueResult(round=2, verdict="parse_error"),
            ],
            rounds_used=2,
        )
        assert builder.add_results([r]) == 1


class TestEmptyCritiqueGate:
    def test_trace_with_zero_critique_items_dropped(self) -> None:
        builder = PreferencePairBuilder()
        r = _make_result(
            critiques=[CritiqueResult(round=1, verdict="needs_revision", critiques=[])],
            rounds_used=1,
        )
        assert builder.add_results([r]) == 0

    def test_trace_with_one_critique_item_kept(self) -> None:
        builder = PreferencePairBuilder()
        r = _make_result(
            critiques=[
                CritiqueResult(
                    round=1,
                    verdict="needs_revision",
                    critiques=[{"principle": "p", "fix": "f"}],
                )
            ],
            rounds_used=1,
        )
        assert builder.add_results([r]) == 1

    def test_min_critique_items_threshold_honored(self) -> None:
        builder = PreferencePairBuilder(min_critique_items=3)
        r = _make_result(
            critiques=[
                CritiqueResult(
                    round=1,
                    verdict="needs_revision",
                    critiques=[{"fix": "a"}, {"fix": "b"}],
                )
            ],
            rounds_used=1,
        )
        assert builder.add_results([r]) == 0

    def test_zero_rounds_used_bypasses_critique_gate(self) -> None:
        # Converged on round 0 (baseline was compliant) — no revision happened,
        # so the "critique items" gate doesn't apply.
        builder = PreferencePairBuilder()
        r = _make_result(
            critiques=[CritiqueResult(round=1, verdict="compliant")],
            rounds_used=0,
            converged=True,
            initial="baseline",
            final="baseline",
        )
        # Still dropped because chosen == rejected (is_trivial).
        assert builder.add_results([r]) == 0


class TestHappyPathStillWorks:
    def test_clean_trace_produces_pair(self) -> None:
        builder = PreferencePairBuilder()
        r = _make_result(
            critiques=[
                CritiqueResult(
                    round=1,
                    verdict="needs_revision",
                    critiques=[{"principle": "p", "fix": "longer fix text"}],
                ),
                CritiqueResult(round=2, verdict="compliant"),
            ],
            rounds_used=2,
            converged=True,
        )
        assert builder.add_results([r]) == 1
        assert len(builder) == 1
        pair = builder._pairs[0]
        assert pair.metadata["num_critiques"] == 1
        assert pair.metadata["converged"] is True
