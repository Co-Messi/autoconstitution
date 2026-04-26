"""Held-out HumanEval+ scorer for the distillation experiment.

Given a JSONL of ``{task_id, prompt, entry_point, hidden_tests}`` produced
by ``ingest_humanevalplus.py`` and a callable that maps prompt → answer,
score every problem and emit a pass@1 aggregate.

Reuses the subprocess sandbox (env sanitization, tmpdir cleanup, pytest
invocation) from ``benchmark.scorers.coding`` so the training-side code
and the eval-side code use the same execution environment.

Produces:
    - per-problem JSONL log with pass/fail + extracted code
    - aggregate summary (pass@1 with bootstrap 95% CI over seeds)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "codebase"))

from autoconstitution.benchmark.scorers.coding import (  # noqa: E402
    _CODE_FENCE,
    _compose_test_module,
    _sanitized_env,
)

logger = logging.getLogger(__name__)


AnswerFn = Callable[[str], Awaitable[str]]
"""A (prompt) -> awaitable[answer] function. Callers inject the model."""


@dataclass
class ProblemResult:
    task_id: str
    passed: bool
    extracted_code: str
    raw_answer: str
    detail: str


def _extract_code(answer: str) -> str:
    m = _CODE_FENCE.search(answer)
    return m.group(1) if m else answer


def _run_pytest(
    code: str, hidden_tests: str, *, prompt_prefix: str, timeout_s: float,
) -> tuple[bool, str]:
    """Execute hidden tests against ``code`` and return ``(passed, detail)``.

    The HumanEval+ prompt defines the function signature; we prepend it as
    a docstring comment so collection errors pointing at the tmpfile have
    human-readable context. Combining code + prompt_prefix handles cases
    where the student's ```python`` block doesn't include the signature.
    """
    # Some model answers only emit the function body, relying on the prompt's
    # signature to be "concatenated" by the reader. Prepend the prompt so we
    # have a full, executable module. It's cheap insurance; duplicated signatures
    # in the code block are harmless (Python rebinds).
    full_module = _compose_test_module(
        setup=None,
        code=prompt_prefix + "\n\n" + code if prompt_prefix else code,
        hidden_tests=hidden_tests,
    )
    tmpdir = tempfile.mkdtemp(prefix="heval-")
    try:
        tmp_path = Path(tmpdir)
        test_file = tmp_path / "test_case.py"
        test_file.write_text(full_module, encoding="utf-8")
        env = _sanitized_env(tmp_path)
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_file), "-q", "--no-header",
            "-rN", "--color=no", "--tb=no",
            "-p", "no:cacheprovider",
        ]
        try:
            completed = subprocess.run(
                cmd, capture_output=True, timeout=timeout_s,
                cwd=tmpdir, env=env, check=False, text=True,
            )
        except subprocess.TimeoutExpired:
            return (False, f"timeout after {timeout_s}s")
        output = (completed.stdout or "") + (completed.stderr or "")
        # Passed iff pytest exit code is 0 AND at least one test was collected.
        if completed.returncode != 0:
            return (False, _tail(output, 300))
        if " passed" not in output:
            return (False, f"no tests collected: {_tail(output, 200)}")
        return (True, "ok")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _tail(text: str, n: int) -> str:
    return text if len(text) <= n else "…" + text[-(n - 1) :]


async def score_one(
    problem: dict, answer_fn: AnswerFn, *, timeout_s: float,
) -> ProblemResult:
    raw = await answer_fn(problem["prompt"])
    code = _extract_code(raw)
    passed, detail = await asyncio.to_thread(
        _run_pytest,
        code, problem["hidden_tests"],
        prompt_prefix=problem["prompt"], timeout_s=timeout_s,
    )
    return ProblemResult(
        task_id=problem["task_id"], passed=passed,
        extracted_code=code, raw_answer=raw, detail=detail,
    )


async def score_all(
    heval_path: Path,
    answer_fn: AnswerFn,
    *,
    log_path: Path | None = None,
    timeout_s: float = 15.0,
    concurrency: int = 4,
) -> list[ProblemResult]:
    problems: list[dict] = []
    with heval_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))

    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(p: dict) -> ProblemResult:
        async with semaphore:
            return await score_one(p, answer_fn, timeout_s=timeout_s)

    results: list[ProblemResult] = []
    tasks = [asyncio.create_task(_bounded(p)) for p in problems]
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        r = await fut
        results.append(r)
        if i % 10 == 0 or i == len(tasks):
            logger.info("progress: %d/%d scored", i, len(tasks))

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps({
                    "task_id": r.task_id,
                    "passed": r.passed,
                    "detail": r.detail,
                    "extracted_code": r.extracted_code,
                }, ensure_ascii=False))
                fh.write("\n")

    return results


def pass_at_1(results: list[ProblemResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.passed) / len(results)


def bootstrap_ci(
    results: list[ProblemResult], *, resamples: int = 1000, seed: int = 0,
) -> tuple[float, float]:
    """95% CI on pass@1 via bootstrap resampling at the problem level."""
    if not results:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(results)
    passes = [1.0 if r.passed else 0.0 for r in results]
    means: list[float] = []
    for _ in range(resamples):
        draw = [rng.choice(passes) for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()
    return (means[int(0.025 * resamples)], means[int(0.975 * resamples)])


# -----------------------------------------------------------------------------
# CLI: standalone smoke test with canonical solutions (sanity check the scorer)
# -----------------------------------------------------------------------------


async def _canonical_smoke_test(heval_path: Path, timeout_s: float) -> None:
    """Score every HumanEval+ problem using its own canonical_solution.

    Expected pass@1 ≈ 1.0. If it's meaningfully lower, the hidden-tests wrapper
    or the extraction logic has a bug and every real-model eval would be
    artificially depressed. This is the sanity check before any real run.
    """
    problems: list[dict] = []
    with heval_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                problems.append(json.loads(line))

    async def _canonical(prompt: str) -> str:
        # Find the matching row — prompt is unique per task in HumanEval+.
        for p in problems:
            if p["prompt"] == prompt:
                return f"```python\n{p['prompt']}\n{p['canonical_solution']}\n```"
        return ""

    started = time.monotonic()
    results = await score_all(
        heval_path, _canonical, timeout_s=timeout_s, concurrency=6,
    )
    p1 = pass_at_1(results)
    lo, hi = bootstrap_ci(results)
    elapsed = time.monotonic() - started
    print(f"\n=== HumanEval+ canonical smoke ===")
    print(f"  problems:     {len(results)}")
    print(f"  pass@1:       {p1:.4f}")
    print(f"  95% CI:       [{lo:.4f}, {hi:.4f}]")
    print(f"  elapsed:      {elapsed:.1f}s")
    if p1 < 0.95:
        failing = [r for r in results if not r.passed]
        print(f"\n  WARNING: canonical pass@1 = {p1:.2f} < 0.95 — scorer may be buggy.")
        print("  First 3 failing task_ids + details:")
        for r in failing[:3]:
            print(f"    {r.task_id}: {r.detail[:200]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--humanevalplus", type=Path,
        default=Path("data/humanevalplus_eval.jsonl"),
    )
    parser.add_argument(
        "--mode", choices=["smoke"], default="smoke",
        help="'smoke' runs canonical solutions to sanity-check the scorer.",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.mode == "smoke":
        asyncio.run(_canonical_smoke_test(args.humanevalplus, args.timeout))


if __name__ == "__main__":
    main()
