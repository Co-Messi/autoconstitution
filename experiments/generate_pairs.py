"""Pair-generation pipeline for the distillation experiment.

For each MBPP training problem:

1. Run the existing CAI critique-revision loop with a 1b Student and a
   separate Judge (either local 3b or cloud MiniMax). The Judge critiques
   against the constitution — it does **not** see the hidden tests.
2. Extract ``initial_answer`` and ``final_answer`` from the loop result.
3. Score both against MBPP ``hidden_tests`` using the existing subprocess
   sandbox from ``benchmark.scorers.coding``.
4. Keep the pair iff ``pytest(final)`` passes strictly more tests than
   ``pytest(initial)``. Drop problems the Student already solved perfectly
   on the initial draft (no room to improve), and drop problems where
   the revision didn't help.

Output is TRL DPO-ready JSONL::

    {"prompt": <mbpp_text>, "chosen": <final_code>, "rejected": <initial_code>}

Peer's training script strips ``rejected`` for SFT on iteration 1 but the
shape is preserved so iteration 2 can upgrade to true DPO without regenerating.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Make the codebase/ package importable without a full install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "codebase"))

import aiohttp  # noqa: E402

from autoconstitution.benchmark.scorers.coding import (  # noqa: E402
    _CODE_FENCE,
    _compose_test_module,
    _sanitized_env,
)
from autoconstitution.cai.critique_revision import CritiqueRevisionLoop  # noqa: E402
from autoconstitution.cai.hierarchy import JudgeAgent, StudentAgent  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class PairGenConfig:
    student_provider: str = "ollama"
    student_model: str = "llama3.2:1b"
    student_base_url: str | None = None  # defaults to http://localhost:11434/v1
    judge_provider: str = "ollama"       # "ollama" or "openai_compat"
    judge_model: str = "llama3.2:3b"
    judge_base_url: str | None = None    # for MiniMax: https://api.minimaxi.chat/v1
    judge_api_key_env: str = "MINIMAX_API_KEY"
    max_rounds: int = 3
    subprocess_timeout_s: float = 15.0
    concurrency: int = 4
    limit: int | None = None
    constitution_path: Path | None = None


# -----------------------------------------------------------------------------
# Provider construction
# -----------------------------------------------------------------------------


class _AiohttpChatProvider:
    """LLMProvider adapter backed by aiohttp against any OpenAI-compat endpoint.

    Bypasses httpx/openai-client — those libraries hit RemoteProtocolError
    against local Ollama 0.21 on some setups, while aiohttp works. Same
    path (``/v1/chat/completions``) handles Ollama and MiniMax identically;
    only base_url, api_key, and model differ.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        request_timeout_s: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._request_timeout_s = request_timeout_s

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/chat/completions"
        timeout = aiohttp.ClientTimeout(total=self._request_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=body, headers=headers) as r:
                r.raise_for_status()
                data = await r.json()
        # OpenAI-compat response shape
        return data["choices"][0]["message"]["content"] or ""


async def _build_student(cfg: PairGenConfig):
    """Student is always an Ollama local model. Uses OpenAI-compat v1."""
    base = cfg.student_base_url or "http://localhost:11434/v1"
    return _AiohttpChatProvider(
        base_url=base, api_key="ollama", model=cfg.student_model,
    )


async def _build_judge(cfg: PairGenConfig):
    """Judge can be local Ollama or a MiniMax-style OpenAI-compat endpoint."""
    if cfg.judge_provider == "ollama":
        base = cfg.judge_base_url or "http://localhost:11434/v1"
        return _AiohttpChatProvider(
            base_url=base, api_key="ollama", model=cfg.judge_model,
        )
    if cfg.judge_provider == "openai_compat":
        key = os.environ.get(cfg.judge_api_key_env)
        if not key:
            raise RuntimeError(
                f"${cfg.judge_api_key_env} is unset; "
                f"required for judge_provider=openai_compat"
            )
        if not cfg.judge_base_url:
            raise RuntimeError(
                "judge_base_url is required for judge_provider=openai_compat"
            )
        return _AiohttpChatProvider(
            base_url=cfg.judge_base_url, api_key=key, model=cfg.judge_model,
        )
    raise ValueError(f"unknown judge_provider: {cfg.judge_provider}")


# -----------------------------------------------------------------------------
# Scoring — thin wrapper around the existing subprocess sandbox
# -----------------------------------------------------------------------------


def _extract_code(answer: str) -> str:
    """Return the first ```python fence body, or the whole answer if no fence."""
    m = _CODE_FENCE.search(answer)
    return m.group(1) if m else answer


def _count_tests_passed(
    code: str,
    hidden_tests: str,
    *,
    setup: str | None,
    timeout_s: float,
) -> tuple[int, int]:
    """Run pytest against ``code`` + ``hidden_tests``; return ``(passed, total)``.

    Synchronous — callers run in a thread via ``asyncio.to_thread`` to keep
    the event loop responsive. Returns ``(0, 0)`` on collection failures
    (syntax error, import error) so those count as "zero improvement".
    """
    test_module = _compose_test_module(
        setup=setup if setup else None,
        code=code,
        hidden_tests=hidden_tests,
    )
    tmpdir = tempfile.mkdtemp(prefix="pairgen-")
    try:
        tmp_path = Path(tmpdir)
        test_file = tmp_path / "test_case.py"
        test_file.write_text(test_module, encoding="utf-8")
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
            return (0, 0)
        return _parse_counts(completed.stdout + completed.stderr)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_counts(output: str) -> tuple[int, int]:
    """Extract (passed, total) from pytest's summary line; (0,0) on failure."""
    import re as _re
    passed = failed = errors = 0
    for line in output.splitlines()[::-1]:
        if "passed" in line or "failed" in line or "error" in line:
            m_p = _re.search(r"(\d+)\s+passed", line)
            m_f = _re.search(r"(\d+)\s+failed", line)
            m_e = _re.search(r"(\d+)\s+error", line)
            passed = int(m_p.group(1)) if m_p else 0
            failed = int(m_f.group(1)) if m_f else 0
            errors = int(m_e.group(1)) if m_e else 0
            if passed or failed or errors:
                return (passed, passed + failed + errors)
    return (0, 0)


# -----------------------------------------------------------------------------
# Pair generation
# -----------------------------------------------------------------------------


@dataclass
class PairResult:
    task_id: int
    initial_passed: int
    final_passed: int
    total_tests: int
    kept: bool
    drop_reason: str
    prompt: str
    initial_answer: str
    final_answer: str


async def _generate_one_pair(
    problem: dict,
    *,
    student_provider,
    judge_provider,
    cfg: PairGenConfig,
) -> PairResult:
    """Run the CAI loop on one MBPP problem and decide whether to keep it."""
    task_id = int(problem["task_id"])
    prompt = problem["prompt"]
    hidden_tests = problem["hidden_tests"]
    setup = problem.get("test_imports") or None

    # Wrap the MBPP prompt into a full instruction for the Student.
    full_prompt = (
        f"{prompt}\n\n"
        "Return only a ```python``` code block defining the function. "
        "No prose before or after."
    )

    student = StudentAgent(provider=student_provider)
    judge = JudgeAgent(
        provider=judge_provider,
        constitution_path=cfg.constitution_path,
    )
    loop = CritiqueRevisionLoop(
        student=student, judge=judge, max_rounds=cfg.max_rounds,
        # Don't halt early on identical revisions — on coding tasks the
        # Student sometimes repeats itself once, then diverges on round 2+.
        skip_identical_revisions=False,
    )

    try:
        result = await loop.run(full_prompt)
    except Exception as exc:
        return PairResult(
            task_id=task_id, initial_passed=0, final_passed=0, total_tests=0,
            kept=False, drop_reason=f"loop_error:{type(exc).__name__}",
            prompt=prompt, initial_answer="", final_answer="",
        )

    initial_code = _extract_code(result.initial_answer)
    final_code = _extract_code(result.final_answer)

    initial_passed, total = await asyncio.to_thread(
        _count_tests_passed, initial_code, hidden_tests,
        setup=setup, timeout_s=cfg.subprocess_timeout_s,
    )
    final_passed, total_f = await asyncio.to_thread(
        _count_tests_passed, final_code, hidden_tests,
        setup=setup, timeout_s=cfg.subprocess_timeout_s,
    )
    # Test count should match between runs; if it doesn't, fall back to max.
    total_tests = max(total, total_f)

    # Drop already-solved (no room to improve).
    if initial_passed >= total_tests > 0:
        return PairResult(
            task_id=task_id, initial_passed=initial_passed, final_passed=final_passed,
            total_tests=total_tests, kept=False, drop_reason="already_solved",
            prompt=prompt, initial_answer=initial_code, final_answer=final_code,
        )
    # Drop collection failures (both 0/0 means pytest couldn't even load).
    if total_tests == 0:
        return PairResult(
            task_id=task_id, initial_passed=0, final_passed=0, total_tests=0,
            kept=False, drop_reason="collection_failure",
            prompt=prompt, initial_answer=initial_code, final_answer=final_code,
        )
    # Drop if revision didn't strictly improve.
    if final_passed <= initial_passed:
        return PairResult(
            task_id=task_id, initial_passed=initial_passed, final_passed=final_passed,
            total_tests=total_tests, kept=False, drop_reason="no_improvement",
            prompt=prompt, initial_answer=initial_code, final_answer=final_code,
        )
    # Keep.
    return PairResult(
        task_id=task_id, initial_passed=initial_passed, final_passed=final_passed,
        total_tests=total_tests, kept=True, drop_reason="",
        prompt=prompt, initial_answer=initial_code, final_answer=final_code,
    )


async def generate_pairs(
    mbpp_path: Path, output_path: Path, *, cfg: PairGenConfig,
) -> dict[str, int]:
    problems: list[dict] = []
    with mbpp_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))
    if cfg.limit is not None:
        problems = problems[: cfg.limit]

    student_provider = await _build_student(cfg)
    judge_provider = await _build_judge(cfg)

    semaphore = asyncio.Semaphore(cfg.concurrency)

    async def _bounded(problem: dict) -> PairResult:
        async with semaphore:
            return await _generate_one_pair(
                problem,
                student_provider=student_provider,
                judge_provider=judge_provider,
                cfg=cfg,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "total": 0, "kept": 0, "already_solved": 0,
        "no_improvement": 0, "collection_failure": 0, "loop_error": 0,
    }

    started = time.monotonic()
    with output_path.open("w", encoding="utf-8") as fh:
        tasks = [asyncio.create_task(_bounded(p)) for p in problems]
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            result = await fut
            stats["total"] += 1
            if result.kept:
                stats["kept"] += 1
                record = {
                    "task_id": result.task_id,
                    "prompt": result.prompt,
                    "chosen": result.final_answer,
                    "rejected": result.initial_answer,
                    "initial_passed": result.initial_passed,
                    "final_passed": result.final_passed,
                    "total_tests": result.total_tests,
                }
                fh.write(json.dumps(record, ensure_ascii=False))
                fh.write("\n")
                fh.flush()
            else:
                key = result.drop_reason.split(":")[0]
                stats[key] = stats.get(key, 0) + 1
            if i % 10 == 0 or i == len(tasks):
                elapsed = time.monotonic() - started
                logger.info(
                    "progress: %d/%d processed, %d kept, elapsed=%.1fs",
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
    parser.add_argument(
        "--judge-provider", choices=["ollama", "openai_compat"], default="ollama",
    )
    parser.add_argument("--judge-model", default="llama3.2:3b")
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-api-key-env", default="MINIMAX_API_KEY")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--constitution",
        type=Path,
        default=Path(__file__).with_name("code_review_constitution.md"),
        help="Path to the judge's constitution. Defaults to the code-review one.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = PairGenConfig(
        student_model=args.student_model,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_api_key_env=args.judge_api_key_env,
        max_rounds=args.max_rounds,
        concurrency=args.concurrency,
        limit=args.limit,
        constitution_path=args.constitution,
    )
    stats = asyncio.run(generate_pairs(args.mbpp, args.output, cfg=cfg))

    print(f"\n=== pair generation done ===")
    print(f"  output:             {args.output}")
    print(f"  total processed:    {stats['total']}")
    print(f"  pairs kept:         {stats['kept']}")
    print(f"  already solved:     {stats.get('already_solved', 0)}")
    print(f"  no improvement:     {stats.get('no_improvement', 0)}")
    print(f"  collection errors:  {stats.get('collection_failure', 0)}")
    print(f"  loop errors:        {stats.get('loop_error', 0)}")


if __name__ == "__main__":
    main()
