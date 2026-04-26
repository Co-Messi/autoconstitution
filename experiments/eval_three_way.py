"""Three-way HumanEval+ evaluation: base 1b vs base 8b vs SFT-trained 1b.

This is the experiment's acceptance test. We want to know whether a
llama3.2:1b fine-tuned on autoconstitution-generated pairs can beat a
llama3.1:8b untrained generalist on the same held-out coding benchmark.

All models are generated via the same mlx-lm pipeline — identical prompt
template, temperature, max-tokens, and seed set — so the only variable
is the model. Scoring delegates to the consultant's ``humaneval_scorer``.

Usage:
    python scripts/experiments/eval_three_way.py \\
        --problems data/humanevalplus_eval.jsonl \\
        --adapter checkpoints/adapter-bestofn \\
        --output eval_report.json \\
        --seeds 3 --max-problems 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MLX_1B = "mlx-community/Llama-3.2-1B-Instruct-4bit"
MLX_8B = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"


@dataclass
class _ModelResult:
    name: str
    per_seed_pass_rates: list[float] = field(default_factory=list)
    total_problems: int = 0
    total_elapsed_s: float = 0.0

    @property
    def mean_pass_rate(self) -> float:
        return statistics.fmean(self.per_seed_pass_rates) if self.per_seed_pass_rates else 0.0

    @property
    def stdev_pass_rate(self) -> float:
        return (
            statistics.stdev(self.per_seed_pass_rates)
            if len(self.per_seed_pass_rates) > 1
            else 0.0
        )


def _build_mlx_answer_fn(
    model_id: str, adapter_path: Path | None, max_tokens: int,
) -> Callable[[str], Awaitable[str]]:
    """Return an async answer_fn for consultant's score_all.

    The scorer expects ``answer_fn(prompt: str) -> Awaitable[str]``. We load
    the model once in the closure so we don't reload per problem/seed.
    """
    from mlx_lm import generate, load

    model, tokenizer = load(
        model_id,
        adapter_path=str(adapter_path) if adapter_path else None,
    )

    async def _answer(prompt: str) -> str:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        # Intentionally synchronous — mlx-lm's Metal backend is not safe to
        # call from multiple threads concurrently (completion-handler-after-
        # commit assertion). We serialize via concurrency=1 in the scorer
        # and do straight-line generate here. One GPU, one generate at a time.
        return generate(
            model, tokenizer, prompt=chat,
            max_tokens=max_tokens, verbose=False,
        )

    return _answer


async def _eval_one_seed(
    name: str,
    answer_fn: Callable,
    problems_path: Path,
    seed: int,
    log_dir: Path,
    timeout_s: float,
    concurrency: int,
    scorer,
) -> tuple[float, float]:
    """Evaluate one (model, seed) pair. Returns ``(pass_rate, elapsed_s)``."""
    import random as _r

    _r.seed(seed)
    started = time.monotonic()
    log_path = log_dir / f"{name}_seed{seed}.jsonl"
    results = await scorer.score_all(
        problems_path,
        answer_fn,
        log_path=log_path,
        timeout_s=timeout_s,
        concurrency=concurrency,
    )
    elapsed = time.monotonic() - started
    pass_rate = scorer.pass_at_1(results)
    print(
        f"[eval] {name} seed={seed} "
        f"pass@1={pass_rate:.3f} ({int(pass_rate * len(results))}/{len(results)}) "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return pass_rate, elapsed


async def _score_model(
    name: str,
    answer_fn: Callable,
    problems_path: Path,
    seeds: list[int],
    log_dir: Path,
    timeout_s: float,
    concurrency: int,
    scorer,
) -> _ModelResult:
    # Count problems for the result metadata — one quick line count.
    with problems_path.open("r", encoding="utf-8") as f:
        total_problems = sum(1 for line in f if line.strip())
    result = _ModelResult(name=name, total_problems=total_problems)
    for seed in seeds:
        pass_rate, elapsed = await _eval_one_seed(
            name, answer_fn, problems_path, seed, log_dir, timeout_s, concurrency, scorer,
        )
        result.per_seed_pass_rates.append(pass_rate)
        result.total_elapsed_s += elapsed
    return result


def _bootstrap_ci(
    samples: list[float], confidence: float = 0.95, resamples: int = 1000, seed: int = 0,
) -> tuple[float, float]:
    if not samples:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(samples)
    means = []
    for _ in range(resamples):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(resample))
    means.sort()
    alpha = (1 - confidence) / 2
    lo_idx = int(alpha * resamples)
    hi_idx = int((1 - alpha) * resamples) - 1
    return (means[lo_idx], means[hi_idx])


def _subset_problems(src: Path, dst: Path, n: int, seed: int = 0) -> None:
    """Materialize a deterministic N-problem subset so every model sees the same set."""
    with src.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    rng = random.Random(seed)
    rng.shuffle(lines)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        f.writelines(lines[:n])


def _write_report(
    base_1b: _ModelResult,
    base_8b: _ModelResult,
    trained: _ModelResult | None,
    out_path: Path,
) -> dict:
    rows = [base_1b, base_8b]
    if trained is not None:
        rows.append(trained)

    out: dict[str, Any] = {
        "n_problems": base_1b.total_problems,
        "n_seeds": len(base_1b.per_seed_pass_rates),
        "models": {},
    }
    for r in rows:
        lo, hi = _bootstrap_ci(r.per_seed_pass_rates)
        out["models"][r.name] = {
            "mean_pass@1": r.mean_pass_rate,
            "stdev_pass@1": r.stdev_pass_rate,
            "ci95": [lo, hi],
            "per_seed": r.per_seed_pass_rates,
            "elapsed_s": r.total_elapsed_s,
        }
    if trained is not None:
        out["models"][trained.name]["delta_vs_base_1b"] = (
            trained.mean_pass_rate - base_1b.mean_pass_rate
        )
        out["models"][trained.name]["delta_vs_base_8b"] = (
            trained.mean_pass_rate - base_8b.mean_pass_rate
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


async def _main_async(args) -> int:  # noqa: ANN001 - argparse Namespace
    # Import consultant's scorer.
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        scorer = __import__(args.scorer_module)
    except ImportError as e:
        print(f"error: could not import {args.scorer_module}: {e}", file=sys.stderr)
        return 2

    # Subset problems if requested — deterministic across all models.
    run_dir = args.output.parent
    if args.max_problems and args.max_problems > 0:
        subset_path = run_dir / "eval_subset.jsonl"
        _subset_problems(args.problems, subset_path, args.max_problems, seed=0)
        problems_path = subset_path
    else:
        problems_path = args.problems

    # Per-model logs go under <run_dir>/eval_logs/
    log_dir = run_dir / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seeds))

    print(f"[eval] problems_path={problems_path}, seeds={seeds}", flush=True)

    # Base 1b
    print(f"[eval] loading base 1b ({args.model_1b})…", flush=True)
    answer_1b = _build_mlx_answer_fn(args.model_1b, None, args.max_tokens)
    base_1b = await _score_model(
        "base_1b", answer_1b, problems_path, seeds, log_dir,
        args.timeout_s, args.concurrency, scorer,
    )

    # Base 8b (competitor)
    print(f"[eval] loading base 8b ({args.model_8b})…", flush=True)
    answer_8b = _build_mlx_answer_fn(args.model_8b, None, args.max_tokens)
    base_8b = await _score_model(
        "base_8b", answer_8b, problems_path, seeds, log_dir,
        args.timeout_s, args.concurrency, scorer,
    )

    # Trained 1b
    trained: _ModelResult | None = None
    if args.adapter and args.adapter.exists():
        print(f"[eval] loading trained 1b (adapter={args.adapter})…", flush=True)
        answer_trained = _build_mlx_answer_fn(args.model_1b, args.adapter, args.max_tokens)
        trained = await _score_model(
            args.trained_name, answer_trained, problems_path, seeds, log_dir,
            args.timeout_s, args.concurrency, scorer,
        )

    report = _write_report(base_1b, base_8b, trained, args.output)

    print("=" * 60)
    print("FINAL REPORT")
    for name, data in report["models"].items():
        lo, hi = data["ci95"]
        line = (
            f"  {name}: pass@1 = {data['mean_pass@1']:.3f} "
            f"[{lo:.3f}, {hi:.3f}]"
        )
        if "delta_vs_base_8b" in data:
            line += (
                f"  Δ_vs_1b = {data['delta_vs_base_1b']:+.3f}  "
                f"Δ_vs_8b = {data['delta_vs_base_8b']:+.3f}"
            )
        print(line)
    print("=" * 60)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problems", type=Path, required=True)
    p.add_argument("--adapter", type=Path, default=None,
                   help="Adapter directory for the trained 1b.")
    p.add_argument("--trained-name", default="trained_1b_bestofn",
                   help="Label for the trained model in the report.")
    p.add_argument("--model-1b", default=MLX_1B)
    p.add_argument("--model-8b", default=MLX_8B)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--max-problems", type=int, default=0,
                   help="Subset size. 0 = evaluate every problem.")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout-s", type=float, default=15.0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--scorer-module", default="humaneval_scorer")
    args = p.parse_args()

    if not args.problems.exists():
        print(f"error: {args.problems} not found", file=sys.stderr)
        return 2

    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
