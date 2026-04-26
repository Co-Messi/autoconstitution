"""Built-in benchmark scorers.

Two concrete scorers ship with autoconstitution:

* :class:`~autoconstitution.benchmark.scorers.coding.CodingScorer` — runs
  hidden unit tests against the answer in a sandboxed subprocess. Binary
  pass/fail per test; score is the pass ratio.
* :class:`~autoconstitution.benchmark.scorers.judge.JudgeScorer` — an
  LLM provider rates the answer against a rubric on a 0-10 scale
  (normalized to 0-1). Continuous, no hard pass/fail.

Both satisfy :class:`~autoconstitution.benchmark.protocol.Scorer` and
can be instantiated by name via :data:`SCORERS`.
"""

from __future__ import annotations

from autoconstitution.benchmark.protocol import Scorer
from autoconstitution.benchmark.scorers.coding import CodingScorer
from autoconstitution.benchmark.scorers.judge import JudgeScorer

SCORERS: dict[str, type[Scorer]] = {
    "coding": CodingScorer,
    "judge": JudgeScorer,
}
"""Name → class mapping for CLI lookup.

The CLI does ``SCORERS[name](...kwargs)`` to construct a scorer. Coding
needs no arguments. Judge takes ``provider`` (the :class:`LLMProvider`
that evaluates the rubric).
"""

__all__ = ["SCORERS", "CodingScorer", "JudgeScorer"]
