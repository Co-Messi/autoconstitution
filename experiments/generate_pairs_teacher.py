"""Teacher→Student distillation pair-generation.

For each MBPP problem:

1. **Teacher (MiniMax M2.7 by default)** generates ``N_t`` candidates at low
   temperature. First one whose code passes all hidden tests becomes the
   ``chosen`` example.
2. **Student (llama3.2:1b)** generates ``N_s`` candidates at mid temperature.
   First one whose code fails at least one hidden test becomes the ``rejected``
   example (guarantees real DPO signal — the student's own failure mode).
3. Emit a pair iff both sides land: teacher has a full-pass output *and* the
   student has a genuine failure on the same prompt. Drop already-solved
   problems (no signal) and drop problems the teacher can't solve (no target).

Output shape matches ``generate_pairs.py`` / ``generate_pairs_bestofn.py``::

    {"task_id", "prompt", "chosen", "rejected",
     "chosen_passed", "rejected_passed", "total_tests",
     "teacher_samples_used", "student_samples_used"}

Why teacher→student, not best-of-N?
-----------------------------------
Best-of-N self-distills the student into itself — its ceiling is capped at
the student's own capability. Measured: trained-1b dropped from 0.300 to
0.225 on HumanEval+, consistent with the "no teaching signal" hypothesis.
A teacher model with real headroom (MiniMax M2.7) supplies outputs the
student cannot currently produce, which is the actual distillation path.

MiniMax is the default because the user already has the key. Any OpenAI-compat
endpoint works; to use the local 8b as teacher, pass
``--teacher-provider ollama --teacher-model llama3.1:8b-instruct-q4_0``
(or whatever tag you have), no code change needed.
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

import aiohttp

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "codebase"))

from autoconstitution.benchmark.scorers.coding import (  # noqa: E402
    _CODE_FENCE,
    _compose_test_module,
    _sanitized_env,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class TeacherConfig:
    # Teacher (default: MiniMax M2.7 via openai-compat)
    teacher_provider: str = "openai_compat"
    teacher_model: str = "MiniMax-M2"
    teacher_base_url: str = "https://api.minimaxi.com/v1"
    teacher_api_key_env: str = "MINIMAX_API_KEY"
    teacher_samples: int = 3
    teacher_temperature: float = 0.2
    teacher_max_tokens: int = 2048

    # Student (default: local 1b via Ollama)
    student_provider: str = "ollama"
    student_model: str = "llama3.2:1b"
    student_base_url: str = "http://localhost:11434/v1"
    student_samples: int = 4
    student_temperature: float = 0.7
    student_max_tokens: int = 1024

    subprocess_timeout_s: float = 15.0
    teacher_concurrency: int = 2   # MiniMax rate-limited; peer's run died at 4
    student_concurrency: int = 6   # local is cheap
    limit: int | None = None


# -----------------------------------------------------------------------------
# HTTP sampling (OpenAI-compat; works for both Ollama and MiniMax)
# -----------------------------------------------------------------------------


_TEACHER_SYSTEM_PROMPT = (
    "You are a Python programmer. Respond with a single ```python code block "
    "containing ONLY the complete function definition. No prose before or "
    "after. No explanation. No docstring unless it's part of the function."
)


async def _sample(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> str:
    """Sample with exponential-backoff retry on transient connector errors."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            async with session.post(url, json=body, headers=headers) as r:
                r.raise_for_status()
                data = await r.json()
            return data["choices"][0]["message"]["content"] or ""
        except (aiohttp.ClientConnectorError, aiohttp.ClientOSError,
                aiohttp.ServerDisconnectedError, asyncio.TimeoutError) as exc:
            last_exc = exc
            await asyncio.sleep(1.5 ** attempt)
        except aiohttp.ClientResponseError as exc:
            if exc.status in (429, 500, 502, 503, 504):
                last_exc = exc
                await asyncio.sleep(2.0 * (2 ** attempt))
            else:
                raise
    assert last_exc is not None
    raise last_exc


def _strip_think_tags(text: str) -> str:
    """MiniMax M2.7 emits <think>...</think> before the answer. Strip it."""
    import re as _re
    return _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------


def _extract_code(answer: str) -> str:
    m = _CODE_FENCE.search(answer)
    return m.group(1) if m else answer


def _first_assert(hidden_tests: str) -> str:
    for line in hidden_tests.splitlines():
        s = line.strip()
        if s.startswith("assert "):
            return s
    return ""


def _count_passed(
    code: str, hidden_tests: str, *, setup: str | None, timeout_s: float,
) -> tuple[int, int]:
    test_module = _compose_test_module(
        setup=setup if setup else None, code=code, hidden_tests=hidden_tests,
    )
    tmpdir = tempfile.mkdtemp(prefix="teacher-")
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
# Per-problem flow
# -----------------------------------------------------------------------------


async def _find_passing_teacher_sample(
    session: aiohttp.ClientSession,
    *,
    cfg: TeacherConfig,
    teacher_api_key: str,
    prompt: str,
    hidden_tests: str,
    setup: str | None,
) -> tuple[str | None, int, int]:
    """Sample teacher until one passes all tests, or budget exhausted.

    Returns (winning_code, total_tests, samples_used). winning_code is None
    if no sample passed.
    """
    # First attempt greedy-ish, later retries at higher temperature for diversity.
    temps = [cfg.teacher_temperature] + [0.5] * (cfg.teacher_samples - 1)
    for i in range(cfg.teacher_samples):
        try:
            raw = await _sample(
                session,
                base_url=cfg.teacher_base_url,
                api_key=teacher_api_key,
                model=cfg.teacher_model,
                prompt=prompt,
                system=_TEACHER_SYSTEM_PROMPT,
                temperature=temps[i],
                max_tokens=cfg.teacher_max_tokens,
            )
        except Exception as exc:
            logger.warning("teacher sample %d failed: %s", i, exc)
            continue
        cleaned = _strip_think_tags(raw)
        code = _extract_code(cleaned)
        passed, total = await asyncio.to_thread(
            _count_passed, code, hidden_tests,
            setup=setup, timeout_s=cfg.subprocess_timeout_s,
        )
        if total > 0 and passed == total:
            return (code, total, i + 1)
    return (None, 0, cfg.teacher_samples)


async def _find_failing_student_sample(
    session: aiohttp.ClientSession,
    *,
    cfg: TeacherConfig,
    prompt: str,
    hidden_tests: str,
    setup: str | None,
    total_tests_hint: int,
) -> tuple[str | None, int, int]:
    """Sample student until one fails at least one test.

    Returns (losing_code, passed_count, samples_used). losing_code is None
    if every student sample passed everything (rare — the student is 1b).
    """
    for i in range(cfg.student_samples):
        try:
            raw = await _sample(
                session,
                base_url=cfg.student_base_url,
                api_key="ollama",
                model=cfg.student_model,
                prompt=prompt,
                temperature=cfg.student_temperature,
                max_tokens=cfg.student_max_tokens,
            )
        except Exception as exc:
            logger.warning("student sample %d failed: %s", i, exc)
            continue
        code = _extract_code(raw)
        passed, total = await asyncio.to_thread(
            _count_passed, code, hidden_tests,
            setup=setup, timeout_s=cfg.subprocess_timeout_s,
        )
        if total == 0:
            # Collection failure — treat as a real failure (0 passed).
            return (code, 0, i + 1)
        if passed < total:
            return (code, passed, i + 1)
    return (None, total_tests_hint, cfg.student_samples)


async def _process_problem(
    session_teacher: aiohttp.ClientSession,
    session_student: aiohttp.ClientSession,
    problem: dict,
    *,
    cfg: TeacherConfig,
    teacher_api_key: str,
) -> dict | None:
    prompt_text = problem["prompt"]
    hidden_tests = problem["hidden_tests"]
    setup = problem.get("test_imports") or None

    # Include the first assert line so the teacher sees the required function
    # name & signature — standard MBPP convention. Without this, the teacher
    # picks plausible-but-wrong names (e.g. longest_chain vs max_chain_length)
    # and every test fails on NameError.
    example_hint = _first_assert(hidden_tests)
    example_block = f"\nExample test:\n    {example_hint}\n" if example_hint else ""

    full_prompt = (
        f"{prompt_text}\n{example_block}\n"
        "Return only a ```python``` code block defining the function. "
        "The function name and signature must match the example test. "
        "No prose before or after."
    )

    tid = problem.get("task_id")
    teacher_code, total_tests, t_used = await _find_passing_teacher_sample(
        session_teacher,
        cfg=cfg, teacher_api_key=teacher_api_key,
        prompt=full_prompt, hidden_tests=hidden_tests, setup=setup,
    )
    if teacher_code is None:
        logger.info("drop task_id=%s: teacher failed all %d samples", tid, cfg.teacher_samples)
        return None

    student_code, s_passed, s_used = await _find_failing_student_sample(
        session_student,
        cfg=cfg, prompt=full_prompt, hidden_tests=hidden_tests, setup=setup,
        total_tests_hint=total_tests,
    )
    if student_code is None:
        logger.info("drop task_id=%s: student passed all %d samples (no signal)", tid, cfg.student_samples)
        return None

    return {
        "task_id": int(problem["task_id"]),
        "prompt": prompt_text,
        "chosen": teacher_code,
        "rejected": student_code,
        "chosen_passed": total_tests,
        "rejected_passed": s_passed,
        "total_tests": total_tests,
        "teacher_samples_used": t_used,
        "student_samples_used": s_used,
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


async def generate_pairs(
    mbpp_path: Path, output_path: Path, *, cfg: TeacherConfig,
) -> dict[str, int]:
    teacher_api_key = ""
    if cfg.teacher_provider == "openai_compat":
        teacher_api_key = os.environ.get(cfg.teacher_api_key_env, "")
        if not teacher_api_key:
            raise RuntimeError(
                f"${cfg.teacher_api_key_env} is unset (required for "
                f"teacher_provider=openai_compat)"
            )
    elif cfg.teacher_provider == "ollama":
        teacher_api_key = "ollama"
    else:
        raise ValueError(f"unknown teacher_provider: {cfg.teacher_provider}")

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

    # Separate semaphores for teacher vs student — teacher is rate-limited.
    teacher_sem = asyncio.Semaphore(cfg.teacher_concurrency)
    student_sem = asyncio.Semaphore(cfg.student_concurrency)
    # Single session for teacher (connection reuse, important for MiniMax).
    timeout = aiohttp.ClientTimeout(total=300.0)

    async with aiohttp.ClientSession(timeout=timeout) as session_teacher, \
               aiohttp.ClientSession(timeout=timeout) as session_student:

        async def _bounded(p: dict) -> dict | None:
            async with teacher_sem, student_sem:
                return await _process_problem(
                    session_teacher, session_student, p,
                    cfg=cfg, teacher_api_key=teacher_api_key,
                )

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
                if i % 5 == 0 or i == len(tasks):
                    elapsed = time.monotonic() - started
                    logger.info(
                        "progress: %d/%d, kept=%d, elapsed=%.1fs",
                        i, len(tasks), stats["kept"], elapsed,
                    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mbpp", type=Path, default=Path("data/mbpp_train.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--teacher-provider", choices=["openai_compat", "ollama"],
        default="openai_compat",
    )
    parser.add_argument("--teacher-model", default="MiniMax-M2")
    parser.add_argument("--teacher-base-url", default="https://api.minimaxi.com/v1")
    parser.add_argument("--teacher-api-key-env", default="MINIMAX_API_KEY")
    parser.add_argument("--teacher-samples", type=int, default=3)
    parser.add_argument("--teacher-temperature", type=float, default=0.2)
    parser.add_argument("--student-model", default="llama3.2:1b")
    parser.add_argument("--student-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--student-samples", type=int, default=4)
    parser.add_argument("--student-temperature", type=float, default=0.7)
    parser.add_argument("--teacher-concurrency", type=int, default=2)
    parser.add_argument("--student-concurrency", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = TeacherConfig(
        teacher_provider=args.teacher_provider,
        teacher_model=args.teacher_model,
        teacher_base_url=args.teacher_base_url,
        teacher_api_key_env=args.teacher_api_key_env,
        teacher_samples=args.teacher_samples,
        teacher_temperature=args.teacher_temperature,
        student_model=args.student_model,
        student_base_url=args.student_base_url,
        student_samples=args.student_samples,
        student_temperature=args.student_temperature,
        teacher_concurrency=args.teacher_concurrency,
        student_concurrency=args.student_concurrency,
        limit=args.limit,
    )
    stats = asyncio.run(generate_pairs(args.mbpp, args.output, cfg=cfg))
    print("\n=== teacher-distillation pair generation done ===")
    print(f"  output:   {args.output}")
    print(f"  total:    {stats['total']}")
    print(f"  kept:     {stats['kept']}")
    print(f"  dropped:  {stats['dropped']}")


if __name__ == "__main__":
    main()
