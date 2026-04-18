"""Coding scorer: run hidden pytest against an LLM-generated answer.

The scorer drops the extracted Python code + the case's ``hidden_tests``
into a tempfile, runs ``python -m pytest`` in a subprocess with a hard
timeout and a sanitized environment, then parses the output to count
PASSED / FAILED tests. Score is ``passed / total``.

Safety posture (documented in the ``autoconstitution bench`` command's
startup banner too):

* No shell — ``shell=False``, command passed as a list.
* Timeout on the subprocess, killed if exceeded.
* Sanitized environment: API-key vars stripped, ``HOME`` and ``TMPDIR``
  point into the case's tmp directory so sloppy test code can't write
  into the user's filesystem.
* ``cwd`` is the same tmp directory — always deleted on exit.
* **Not** an adversarial sandbox. Benchmarks are expected to ship
  with trusted ``hidden_tests``. If you're running a third-party
  dataset, inspect it first.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from autoconstitution.benchmark.protocol import BenchCase, ScoreResult

_CODE_FENCE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)\n```",
    flags=re.DOTALL | re.IGNORECASE,
)

# Keys we refuse to forward to the subprocess regardless of host env.
_SANITIZE_ENV_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MOONSHOT_API_KEY",
    "KIMI_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
)

class CodingScorer:
    """Runs ``hidden_tests`` against the extracted code in a subprocess."""

    name: str = "coding"

    def __init__(self, *, timeout_s: float = 15.0) -> None:
        self._timeout_s = timeout_s

    async def score(self, case: BenchCase, answer: str) -> ScoreResult:
        """Score ``answer`` by running ``case.metadata['hidden_tests']``."""
        hidden_tests = case.metadata.get("hidden_tests")
        if not isinstance(hidden_tests, str) or not hidden_tests.strip():
            return ScoreResult(
                score=0.0,
                detail=f"case {case.id!r} has no 'hidden_tests' in metadata",
                passed=False,
            )

        code, had_fence = _extract_python(answer)
        if not code.strip():
            return ScoreResult(
                score=0.0,
                detail="answer contained no Python code after extraction",
                passed=False,
            )

        setup = case.metadata.get("setup")
        test_module = _compose_test_module(
            setup=setup if isinstance(setup, str) else None,
            code=code,
            hidden_tests=hidden_tests,
        )

        try:
            return await asyncio.to_thread(
                _run_pytest_subprocess,
                test_module=test_module,
                timeout_s=self._timeout_s,
                had_fence=had_fence,
            )
        except Exception as exc:  # defensive — Protocol forbids raising
            return ScoreResult(
                score=0.0,
                detail=f"unexpected scorer error: {type(exc).__name__}: {exc}",
                passed=False,
            )

    async def close(self) -> None:
        # Nothing to release — subprocesses and tmpdirs are per-call.
        return None


def _extract_python(answer: str) -> tuple[str, bool]:
    """Return ``(code, had_fence)``.

    If the answer contains a ```python``` fence we return the first block.
    Otherwise we treat the whole answer as code and return ``had_fence=False``
    so the detail string can warn that fenceless output is unreliable.
    """
    match = _CODE_FENCE.search(answer)
    if match:
        return match.group(1), True
    return answer, False


def _compose_test_module(
    *,
    setup: str | None,
    code: str,
    hidden_tests: str,
) -> str:
    """Concatenate setup, the student's code, and hidden tests into one file."""
    parts = []
    if setup:
        parts.append("# --- setup ---\n" + setup.rstrip())
    parts.append("# --- student code ---\n" + code.rstrip())
    parts.append("# --- hidden tests ---\n" + hidden_tests.rstrip())
    return "\n\n".join(parts) + "\n"


def _run_pytest_subprocess(
    *,
    test_module: str,
    timeout_s: float,
    had_fence: bool,
) -> ScoreResult:
    """Synchronous worker invoked via ``asyncio.to_thread``.

    Isolated into its own function so the async path stays readable and
    the subprocess plumbing is unit-testable on its own.
    """
    tmpdir = tempfile.mkdtemp(prefix="autoconstitution-bench-")
    try:
        tmp_path = Path(tmpdir)
        test_file = tmp_path / "test_case.py"
        test_file.write_text(test_module, encoding="utf-8")

        env = _sanitized_env(tmp_path)
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "-q",
            "--no-header",
            "-rN",  # suppress short-summary section; we parse PASSED/FAILED directly
            "--color=no",
            "--tb=no",
            "-p",
            "no:cacheprovider",
        ]
        try:
            completed = subprocess.run(  # noqa: S603 - command is a literal list, shell=False
                cmd,
                capture_output=True,
                timeout=timeout_s,
                cwd=tmpdir,
                env=env,
                check=False,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return ScoreResult(
                score=0.0,
                detail=f"pytest timed out after {timeout_s:.1f}s",
                passed=False,
            )

        output = (completed.stdout or "") + (completed.stderr or "")
        passed_count, failed_count, error_count = _parse_pytest_output(output)
        total = passed_count + failed_count + error_count

        if total == 0:
            # Pytest didn't collect any tests — possibly a syntax error.
            return ScoreResult(
                score=0.0,
                detail=_collection_failure_detail(completed.returncode, output, had_fence),
                passed=False,
            )

        score = passed_count / total
        passed = failed_count == 0 and error_count == 0
        detail = f"{passed_count}/{total} tests passed"
        if not had_fence:
            detail += " (answer had no ```python fence — extracted whole output)"
        if failed_count or error_count:
            detail += f"; {failed_count} failed, {error_count} errors"
        return ScoreResult(score=score, detail=detail, passed=passed)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _sanitized_env(tmp_path: Path) -> dict[str, str]:
    """Strip secrets and redirect HOME/TMPDIR into the tmp sandbox."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper() not in _SANITIZE_ENV_KEYS
    }
    # Clear any lingering PYTHONPATH — subprocess should resolve imports
    # from the tmpdir + stdlib only.
    env.pop("PYTHONPATH", None)
    env["HOME"] = str(tmp_path)
    env["TMPDIR"] = str(tmp_path)
    return env


def _parse_pytest_output(output: str) -> tuple[int, int, int]:
    """Count PASSED / FAILED / ERROR markers in pytest's line-oriented output.

    We use ``-v``-less output where pytest emits one char per test (.FE),
    plus a summary line like ``2 passed, 1 failed in 0.02s``. The summary
    line is the source of truth; fall back to char-count if we can't find it.
    """
    summary = _find_summary_line(output)
    if summary:
        return summary

    # Fall back to counting the per-test marker characters.
    passed = output.count(".")  # NOT perfect — dots can appear elsewhere;
    # use char-count only when summary is missing. Subtract obvious noise.
    passed -= output.count("in ")  # summary-ish "in 0.02s"
    passed -= output.count("..")   # double-count correction
    passed = max(passed, 0)
    failed = output.count("F")
    errors = output.count("E")
    return passed, failed, errors


def _find_summary_line(output: str) -> tuple[int, int, int] | None:
    """Extract (passed, failed, errors) from pytest's summary line if present."""
    for line in output.splitlines()[::-1]:
        if "passed" in line or "failed" in line or "error" in line:
            passed = _extract_count(line, "passed")
            failed = _extract_count(line, "failed")
            errors = _extract_count(line, "error")
            if passed or failed or errors:
                return passed, failed, errors
    return None


def _extract_count(line: str, keyword: str) -> int:
    """Return the integer preceding ``keyword`` in ``line``, or 0."""
    match = re.search(rf"(\d+)\s+{keyword}", line)
    return int(match.group(1)) if match else 0


def _collection_failure_detail(returncode: int, output: str, had_fence: bool) -> str:
    """Human-readable explanation when pytest collected zero tests."""
    trimmed = output.strip().splitlines()
    tail = "\n".join(trimmed[-5:]) if trimmed else "(no output)"
    prefix = "pytest collected 0 tests"
    if returncode != 0:
        prefix += f" (exit {returncode})"
    if not had_fence:
        prefix += "; answer had no ```python fence"
    return f"{prefix}. Tail: {tail[:300]}"


__all__ = ["CodingScorer"]
