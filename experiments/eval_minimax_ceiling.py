"""Sanity-check the MiniMax-M2.7 teacher ceiling on HumanEval+.

Before pivoting the pair-gen pipeline to teacher→student distillation with
MiniMax as the teacher, we need to know MiniMax's own pass@1 on the same
40-problem HumanEval+ subset the three-way eval used. If it's ≥0.80, there's
real headroom to distill from. If ≤0.60, the thesis ceiling is set there
and we recalibrate.

Uses the same humaneval_scorer as the three-way eval for apples-to-apples.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path

import aiohttp

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(_REPO_ROOT / "codebase"))

import humaneval_scorer  # noqa: E402

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _build_minimax_answer_fn(
    *, base_url: str, api_key: str, model: str, max_tokens: int, timeout_s: float,
):
    async def _answer(prompt: str) -> str:
        body = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a Python programmer. Respond with a single "
                        "```python code block containing ONLY the complete "
                        "function definition that satisfies the user's prompt. "
                        "Do not include tests, explanations, or prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        url = f"{base_url.rstrip('/')}/chat/completions"
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=body, headers=headers) as r:
                        r.raise_for_status()
                        data = await r.json()
                raw = data["choices"][0]["message"]["content"] or ""
                return _strip_think(raw)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"MiniMax request failed after 4 attempts: {last_err}")

    return _answer


async def _main(args) -> int:  # noqa: ANN001
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("error: MINIMAX_API_KEY is unset", file=sys.stderr)
        return 2
    base_url = os.environ.get("BASE_URL") or args.base_url
    if not base_url:
        print("error: BASE_URL is unset and --base-url not given", file=sys.stderr)
        return 2

    answer_fn = _build_minimax_answer_fn(
        base_url=base_url, api_key=api_key, model=args.model,
        max_tokens=args.max_tokens, timeout_s=args.request_timeout_s,
    )

    log_path = args.output.with_name(args.output.stem + "_log.jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ceiling] problems={args.problems}  model={args.model}", flush=True)
    started = time.monotonic()
    results = await humaneval_scorer.score_all(
        args.problems, answer_fn,
        log_path=log_path, timeout_s=args.timeout_s, concurrency=args.concurrency,
    )
    elapsed = time.monotonic() - started

    p1 = humaneval_scorer.pass_at_1(results)
    lo, hi = humaneval_scorer.bootstrap_ci(results)
    passed = sum(1 for r in results if r.passed)

    summary = {
        "model": args.model,
        "n_problems": len(results),
        "pass@1": p1,
        "passed": passed,
        "ci95": [lo, hi],
        "elapsed_s": elapsed,
    }
    import json
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"MiniMax ceiling on {args.problems.name}")
    print(f"  pass@1 = {p1:.3f}  ({passed}/{len(results)})")
    print(f"  95% CI = [{lo:.3f}, {hi:.3f}]")
    print(f"  elapsed = {elapsed:.1f}s")
    print("=" * 60)
    # Headline for humans
    if p1 >= 0.80:
        print("✓ strong headroom — pivot to MiniMax-as-teacher")
    elif p1 >= 0.60:
        print("~ modest headroom — usable but thesis risk")
    else:
        print("✗ low ceiling — recalibrate before burning tokens")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problems", type=Path,
                   default=Path("artifacts/eval-rerun/eval_subset.jsonl"))
    p.add_argument("--model", default="MiniMax-M2")
    p.add_argument("--base-url", default=None,
                   help="Falls back to $BASE_URL.")
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/minimax-ceiling/ceiling.json"))
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--timeout-s", type=float, default=15.0,
                   help="Per-problem pytest timeout.")
    p.add_argument("--request-timeout-s", type=float, default=120.0,
                   help="HTTP request timeout to MiniMax.")
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()

    if not args.problems.exists():
        print(f"error: {args.problems} not found", file=sys.stderr)
        return 2

    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
