"""Ingest HumanEval+ (evalplus) into a clean JSONL for the held-out eval.

Output shape (one per line):
    {
        "task_id": str,          # e.g. "HumanEval/0"
        "prompt": str,           # function signature + docstring
        "entry_point": str,      # function name the tests call
        "hidden_tests": str,     # pytest-wrapped version of the evalplus test block
        "canonical_solution": str,  # reference implementation, for debugging only
    }

HumanEval+'s raw ``test`` field defines a ``check(candidate)`` helper and
then calls ``check(<entry_point>)`` at module scope. Pytest collects
``test_*`` functions, so we wrap the check invocation in
``def test_humaneval_<id>()``. Collection errors become test failures with
a clear message, mirroring the MBPP pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


def _sanitize_task_id(task_id: str) -> str:
    """``HumanEval/0`` → ``0`` (or any non-alnum → underscore)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", task_id)


def _wrap_check(test_block: str, *, entry_point: str, task_id: str) -> str:
    """Turn evalplus's ``check(candidate)`` block into a pytest-collectable test.

    evalplus files end with a top-level ``check(<entry_point>)`` call, which
    pytest would otherwise only run via ``python -m pytest --assert=plain``
    with the right collection mode. Wrapping in a ``def test_*`` is the
    robust way — pytest collects the wrapper, import executes top-level
    assertions and defines ``check``, the wrapper invokes it.
    """
    safe_id = _sanitize_task_id(task_id)
    # Strip any trailing ``check(...)`` line so our wrapper is the only invocation.
    lines = test_block.rstrip().splitlines()
    while lines and lines[-1].strip().startswith("check("):
        lines.pop()
    trimmed = "\n".join(lines).rstrip()
    return (
        trimmed
        + "\n\n"
        + f"def test_humaneval_{safe_id}():\n"
        + f"    check({entry_point})\n"
    )


def ingest(output_path: Path, *, limit: int | None = None) -> int:
    """Load HumanEval+ test split, emit one JSONL row per task."""
    ds = load_dataset("evalplus/humanevalplus", split="test")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            task_id = row["task_id"]
            record = {
                "task_id": task_id,
                "prompt": row["prompt"],
                "entry_point": row["entry_point"],
                "hidden_tests": _wrap_check(
                    row["test"], entry_point=row["entry_point"], task_id=task_id,
                ),
                "canonical_solution": row["canonical_solution"],
            }
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/humanevalplus_eval.jsonl"),
        help="Output JSONL path (default: data/humanevalplus_eval.jsonl).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: cap to first N problems (default: all 164).",
    )
    args = parser.parse_args()
    n = ingest(args.output, limit=args.limit)
    print(f"wrote {n} HumanEval+ problems → {args.output}")


if __name__ == "__main__":
    main()
