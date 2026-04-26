# autoconstitution Benchmark Runner

**Goal:** prove the CAI loop actually improves answers, not just that it runs. Ship a before/after benchmark — Student-alone baseline vs full critique/revise loop — with a real number in the README.

**Motivation:** autoresearch's identity is measurable improvement. The existing `benchmark/` directory at the repo root has design docs and reproducibility scaffolding but no runnable code. Users trying the repo today have no way to answer "is the CAI loop worth the extra tokens?"

## Scope — v1

- A pluggable `Scorer` protocol so the same runner works across domains.
- Two concrete scorers: `coding` (hidden unit tests, objective) and `judge` (separate LLM rates vs a rubric, universal).
- A hero dataset: 10-20 small buggy Python functions with hidden tests.
- A `autoconstitution bench` top-level CLI command that runs the loop against the dataset and prints a Rich report.
- Exit code is a CI-gate: 0 if aggregate post > pre by a configurable threshold, 1 otherwise.

Out of scope for v1: cross-model comparison, longitudinal tracking, multi-scorer ensembles, web UI. All possible follow-ups once v1 ships.

## Architecture

### Scorer protocol

```python
class Scorer(Protocol):
    name: str

    async def score(self, case: BenchCase, answer: str) -> ScoreResult:
        ...
```

- Scorers are async because `judge` hits an LLM. `coding` is sync under the hood but wears the async shape for uniformity.
- Scorers NEVER raise on bad answers; a failure to score is a `ScoreResult(score=0.0, passed=False, detail="...")` so the aggregate doesn't throw.
- Per-case metadata (hidden tests, reference answer, rubric) lives in `BenchCase.metadata`. The scorer knows which keys it needs.

### Data model

```python
@dataclass(frozen=True, slots=True)
class BenchCase:
    id: str
    prompt: str
    metadata: dict[str, Any]

@dataclass(frozen=True, slots=True)
class ScoreResult:
    score: float         # 0.0 - 1.0
    passed: bool
    detail: str

@dataclass(frozen=True, slots=True)
class CaseOutcome:
    case: BenchCase
    before_answer: str
    after_answer: str
    before_score: ScoreResult
    after_score: ScoreResult
    rounds_used: int
    converged: bool
    elapsed_s: float

@dataclass(frozen=True, slots=True)
class BenchReport:
    outcomes: list[CaseOutcome]
    scorer_name: str
    aggregate_before: float
    aggregate_after: float
    delta: float
    wins: int
    ties: int
    losses: int
    ci95: tuple[float, float]   # bootstrap 95% CI on the delta
```

### Runner

```python
async def run_benchmark(
    cases: list[BenchCase],
    scorer: Scorer,
    loop: CritiqueRevisionLoop,
    *,
    baseline_provider: LLMProvider,
) -> BenchReport
```

Sequence per case:

1. Generate `before` via `baseline_provider.complete(case.prompt)` — one-shot Student, no critique.
2. Generate `after` via `loop.run(case.prompt)` — full critique/revise cycle.
3. Score both with the same scorer.
4. Record timing, rounds used, convergence, scores.

The runner returns a `BenchReport`, not a stream of events, because benchmarking is inherently a bulk operation. Progress is shown via a Rich `Progress` bar; the role-panel live dashboard is the wrong UI here — the user wants aggregate signal, not per-round drama.

### Report rendering

Two artifacts:

1. A Rich table: one row per case, columns = id, before score, after score, Δ, verdict (win/tie/loss), time.
2. A summary panel: aggregate before → after, delta, win/tie/loss counts, 95% CI on the delta (bootstrap, 1000 resamples).

The table is capped at terminal height; --verbose prints all cases, default shows best/worst five per side and hides the middle.

### CLI shape

```
autoconstitution bench \
  --dataset autoconstitution/benchmark/datasets/coding_bugs.jsonl \
  --scorer coding \
  --rounds 3 \
  --threshold 0.05
```

- `--dataset`: path to JSONL, one `BenchCase`-shaped dict per line.
- `--scorer`: name of a registered scorer (`coding`, `judge`).
- `--rounds`: max critique/revise rounds (forwarded to `CritiqueRevisionLoop`).
- `--threshold`: minimum aggregate improvement required to exit 0. Default 0.0 (any improvement).
- `--judge-provider`: override provider for the judge scorer (default: same as main provider).

Exit 0 iff `delta >= threshold`. Exit 2 on validation errors (missing dataset, unknown scorer, etc.). Exit 1 on any other failure.

### Safety — coding scorer

Running LLM-proposed Python is dangerous. The `coding` scorer MUST:

- Execute in a subprocess, not via `exec` in-process.
- Kill after a hard timeout (default 5s per case).
- Restrict filesystem writes to a tmpdir that's deleted after.
- Block network by default (set `NO_PROXY`, unset standard HTTP env vars; best-effort on a dev machine).
- Capture stdout/stderr into the `ScoreResult.detail` so the user can debug failures.

This is the consultant's slice of the work. They built FakeProvider's chunk-delay testing and the subprocess-savvy bits; safe execution is their wheelhouse.

### Dataset format

JSONL, one object per line:

```json
{
  "id": "fizzbuzz_off_by_one",
  "prompt": "Here's a FizzBuzz implementation with a bug. Fix it and return the corrected function.\n\ndef fizzbuzz(n): ...",
  "metadata": {
    "hidden_tests": "def test_5(): assert fizzbuzz(5) == 'Buzz'\ndef test_15(): ..."
  }
}
```

Ten to twenty hand-crafted buggy Python functions: off-by-one, wrong operator, missing edge case, shadowed variable, etc. Each pairs a prompt (the buggy code + 'fix it' instruction) with hidden tests the scorer runs.

## Test strategy

- Runner tests use `FakeProvider` for both baseline and loop provider, and a `DeterministicScorer` test double that scores by substring match. Asserts:
  - Case outcomes are the same length as the input cases.
  - Aggregate scores and delta are computed correctly.
  - A zero-case input returns an empty report without crashing.
  - A scorer that always returns 0 gives a zero-delta report with 100% ties.
  - A case where `after > before` records a win; inverse records a loss.
  - 95% CI computation gives a finite interval on non-trivial data.
- Coding scorer tests use hand-crafted pass/fail Python snippets and verify timeout, safe cleanup, and score correctness. (Consultant's slice.)
- Judge scorer tests use `FakeProvider` scripted to return specific rubric scores. (Consultant's slice.)
- CLI tests via `typer.testing.CliRunner`:
  - `bench --help` exits 0.
  - `bench --dataset /nonexistent` exits 2.
  - `bench --scorer nope` exits 2.
  - `bench` on a tiny embedded dataset with an always-improving FakeProvider exits 0 and prints the summary panel.

## Quality gate

Same as slices 1-4: pytest green, `ruff check` clean, `mypy --strict` clean on every module we author. The benchmark package joins the strict-enforced list in pyproject.

## Deliverables

- Spec (this doc).
- `autoconstitution/benchmark/` package (mine): `__init__.py`, `protocol.py`, `runner.py`, `report.py`, `bootstrap.py`, `datasets/coding_bugs.jsonl`.
- `autoconstitution/benchmark/scorers/` (consultant's): `coding.py`, `judge.py`, `__init__.py` registry.
- `cli.py` — `autoconstitution bench` command (mine).
- `tests/test_bench_runner.py`, `tests/test_bench_coding_scorer.py`, `tests/test_bench_judge_scorer.py`, `tests/test_bench_cli.py`.
- README update — a real number, once we run it against llama3.2:3b.

## Risks

- **Coding scorer complexity.** Safe subprocess execution has real footguns. Consultant's pytest-in-subprocess experience makes this the right split, but it's the highest-risk slice. If it proves harder than expected, ship with only the judge scorer in v1 and defer coding to v1.1.
- **Small-model critique noise.** llama3.2:3b might produce judge verdicts that parse as `parse_error`. The runner must treat those as "no improvement" (use `before_score` for `after_score`) rather than zero, otherwise we'd under-measure.
- **Confidence interval on small N.** With 10-20 cases, a bootstrap CI is wide. That's honest; we display it anyway so readers can see the sample size caveat.

## Non-goals

- Comparing multiple models head-to-head.
- Persistent benchmark history across runs (one-shot only in v1).
- A web dashboard.
- Automated publication of benchmark numbers to the README.
