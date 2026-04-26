"""Best-of-N pair generation: tests are the teacher, no Judge LLM.

For each MBPP problem, sample N initial drafts from the Student at varying
temperatures, score each against hidden_tests, and emit a DPO pair
``(chosen=best_draft, rejected=worst_draft)`` when ``best.passed >
worst.passed`` strictly. Intermediate drafts are discarded.

Rationale
---------
The critique-revision pipeline depends on a Judge LLM that can produce
actionable code feedback. Small local judges don't do this well; even
a cloud Judge costs API calls and cycles. Best-of-N sidesteps the Judge
entirely: we generate diversity from the Student, let the hidden tests
rank them, and keep the (best, worst) pair as training signal.

The narrative this underwrites is cleaner product-wise: "autoconstitution
uses test oracles to self-distill the Student — no teacher API required."

Output shape is identical to ``generate_pairs.py`` (TRL DPO JSONL):

    {"prompt": ..., "chosen": <best>, "rejected": <worst>, ...}

so the downstream training pipeline doesn't care which generator produced
a given JSONL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "codebase"))

from autoconstitution.benchmark.scorers.coding import (  # noqa: E402
    _CODE_FENCE,
    _compose_test_module,
    _sanitized_env,
)

logger = logging.getLogger(__name__)


@dataclass
class BestOfNConfig:
    student_model: str = "llama3.2:1b"
    student_base_url: str = "http://localhost:11434/v1"
    samples_per_problem: int = 6
    # Temperatures to sample at, one per sample (cycled if shorter than N).
    temperatures: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9, 1.0, 1.2)
    max_tokens: int = 1024
    subprocess_timeout_s: float = 15.0
    concurrency: int = 6
    limit: int | None = None


# -----------------------------------------------------------------------------
# Student sampling via aiohttp OpenAI-compat
# -----------------------------------------------------------------------------


async def _sample_one(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    url = f"{base_url.rstrip('/')}/chat/completions"
    async with session.post(
        url,
        json=body,
        headers={"Authorization": "Bearer ollama", "Content-Type": "application/json"},
    ) as r:
        r.raise_for_status()
        data = await r.json()
    return data["choices"][0]["message"]["content"] or ""


# -----------------------------------------------------------------------------
# Test scoring
# -----------------------------------------------------------------------------


def _extract_code(answer: str) -> str:
    m = _CODE_FENCE.search(answer)
    return m.group(1) if m else answer


def _count_passed(
    code: str, hidden_tests: str, *, setup: str | None, timeout_s: float,
) -> tuple[int, int]:
    test_module = _compose_test_module(
        setup=setup if setup else None, code=code, hidden_tests=hidden_tests,
    )
    tmpdir = tempfile.mkdtemp(prefix="bofn-")
    try:
        tmp_path = Path(tmpdir)
        test_file = tmp_path / "test_case.py"
        test_file.write_text(test_module, encoding="utf-8")
        env = _sanitized_env(tmp_path)
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_file), "-q", "--no-header",
            "-rN", "--color=no", "--tb=no", "-p", "no:cacheprovider",
        ]
        try:
            completed = subprocess.run(
                cmd, capture_output=True, timeout=timeout_s,
                cwd=tmpdir, env=env, check=False, text=True,
            )
        except subprocess.TimeoutExpired:
            return (0, 0)
        import re as _re
        passed = failed = errors = 0
        for line in (completed.stdout + completed.stderr).splitlines()[::-1]:
            if "passed" in line or "failed" in line or "error" in line:
                mp = _re.search(r"(\d+)\s+passed", line)
                mf = _re.search(r"(\d+)\s+failed", line)
                me = _re.search(r"(\d+)\s+error", line)
                passed = int(mp.group(1)) if mp else 0
                failed = int(mf.group(1)) if mf else 0
                errors = int(me.group(1)) if me else 0
                if passed or failed or errors:
                    return (passed, passed + failed + errors)
        return (0, 0)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Best-of-N pair extraction
# -----------------------------------------------------------------------------


@dataclass
class _Sample:
    raw: str
    code: str
    passed: int
    total: int


async def _process_problem(
    session: aiohttp.ClientSession,
    problem: dict,
    *,
    cfg: BestOfNConfig,
) -> dict | None:
    prompt_text = problem["prompt"]
    hidden_tests = problem["hidden_tests"]
    setup = problem.get("test_imports") or None

    full_prompt = (
        f"{prompt_text}\n\n"
        "Return only a ```python``` code block defining the function. "
        "No prose before or after."
    )

    # Sample N drafts in parallel.
    temps = list(cfg.temperatures)
    if len(temps) < cfg.samples_per_problem:
        # Cycle temperatures if fewer provided than samples requested.
        temps = (temps * ((cfg.samples_per_problem // len(temps)) + 1))[
            : cfg.samples_per_problem
        ]
    else:
        temps = temps[: cfg.samples_per_problem]

    async def _one(temp: float) -> str:
        try:
            return await _sample_one(
                session,
                base_url=cfg.student_base_url,
                model=cfg.student_model,
                prompt=full_prompt,
                temperature=temp,
                max_tokens=cfg.max_tokens,
            )
        except Exception as exc:
            logger.warning("sample failed on %s: %s", problem.get("task_id"), exc)
            return ""

    raws = await asyncio.gather(*(_one(t) for t in temps))

    # Score each in a thread pool (pytest subprocess is blocking).
    async def _score(raw: str) -> _Sample:
        code = _extract_code(raw)
        passed, total = await asyncio.to_thread(
            _count_passed, code, hidden_tests,
            setup=setup, timeout_s=cfg.subprocess_timeout_s,
        )
        return _Sample(raw=raw, code=code, passed=passed, total=total)

    samples = await asyncio.gather(*(_score(r) for r in raws))
    # Keep only samples that compiled (total > 0 means tests were collected).
    scored = [s for s in samples if s.total > 0]
    if len(scored) < 2:
        return None

    total_tests = max(s.total for s in scored)
    # Pick best (most passes) and worst (fewest passes). Ties broken by
    # order of sampling (stable sort).
    best = max(scored, key=lambda s: s.passed)
    worst = min(scored, key=lambda s: s.passed)

    # Drop degenerate pairs where best == worst on pass count.
    if best.passed <= worst.passed:
        return None
    # Drop cases where worst is already perfect (no signal).
    if worst.passed >= total_tests:
        return None

    return {
        "task_id": int(problem["task_id"]),
        "prompt": prompt_text,
        "chosen": best.code,
        "rejected": worst.code,
        "best_passed": best.passed,
        "worst_passed": worst.passed,
        "total_tests": total_tests,
        "n_samples": len(scored),
    }


async def generate_pairs(
    mbpp_path: Path, output_path: Path, *, cfg: BestOfNConfig,
) -> dict[str, int]:
    problems: list[dict] = []
    with mbpp_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    if cfg.limit is not None:
        problems = problems[: cfg.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"total": 0, "kept": 0, "dropped": 0}
    started = time.monotonic()

    semaphore = asyncio.Semaphore(cfg.concurrency)
    timeout = aiohttp.ClientTimeout(total=120.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def _bounded(p: dict) -> dict | None:
            async with semaphore:
                return await _process_problem(session, p, cfg=cfg)

        tasks = [asyncio.create_task(_bounded(p)) for p in problems]
        with output_path.open("w", encoding="utf-8") as fh:
            for i, fut in enumerate(asyncio.as_completed(tasks), 1):
                record = await fut
                stats["total"] += 1
                if record is not None:
                    stats["kept"] += 1
                    fh.write(json.dumps(record, ensure_ascii=False))
                    fh.write("\n")
                    fh.flush()
                else:
                    stats["dropped"] += 1
                if i % 10 == 0 or i == len(tasks):
                    elapsed = time.monotonic() - started
                    logger.info(
                        "progress: %d/%d, kept=%d, elapsed=%.1fs",
                        i, len(tasks), stats["kept"], elapsed,
                    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mbpp", type=Path, default=Path("data/mbpp_train.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student-model", default="llama3.2:1b")
    parser.add_argument("--student-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--samples-per-problem", type=int, default=6)
    parser.add_argument(
        "--temperatures", default="0.3,0.5,0.7,0.9,1.0,1.2",
        help="Comma-separated list. Cycled if shorter than --samples-per-problem.",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    temps = tuple(float(x) for x in args.temperatures.split(","))
    cfg = BestOfNConfig(
        student_model=args.student_model,
        student_base_url=args.student_base_url,
        samples_per_problem=args.samples_per_problem,
        temperatures=temps,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        limit=args.limit,
    )
    stats = asyncio.run(generate_pairs(args.mbpp, args.output, cfg=cfg))
    print("\n=== best-of-N pair generation done ===")
    print(f"  output:   {args.output}")
    print(f"  total:    {stats['total']}")
    print(f"  kept:     {stats['kept']}")
    print(f"  dropped:  {stats['dropped']}")


if __name__ == "__main__":
    main()
