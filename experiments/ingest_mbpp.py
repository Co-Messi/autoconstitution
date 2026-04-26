"""Ingest MBPP (full, train split) into a clean JSONL.

Output shape (one per line):
    {
        "task_id": int,
        "prompt": str,          # the task description
        "hidden_tests": str,    # joined test_list, pytest-ready
        "test_imports": str,    # joined test_setup_code (may be empty)
        "canonical_solution": str,  # reference solution, for debugging only
    }

The resulting file is used as input to the pair-generation pipeline. Hidden
tests are the filter oracle (not fed into the critique prompt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def _test_block(test_list: list[str], *, task_id: int) -> str:
    """Wrap MBPP's raw assert strings into pytest-collectable functions.

    **One function per assert.** Previously wrapped all asserts into a
    single ``def test_mbpp_<id>()`` which collapsed to 1 pytest test
    regardless of assert count — the quality filter then lost all
    partial-credit signal. One function per assert keeps pytest's
    granularity honest (3 asserts = 3 tests = 3 pass/fail slots).
    """
    parts = []
    for i, assertion in enumerate(test_list):
        parts.append(f"def test_mbpp_{task_id}_{i}():\n    {assertion}\n")
    return "\n".join(parts)


_SPLITS = ("train", "validation", "prompt")
"""MBPP full splits usable for pair generation.

``test`` is the canonical MBPP benchmark — excluded to preserve held-out
integrity. ``train`` + ``validation`` + ``prompt`` give us 374 + 90 + 10
= 474 problems, all with the same schema.
"""


def ingest(output_path: Path) -> int:
    """Load MBPP full train+val+prompt, emit one JSONL row per task.

    Task IDs are globally unique across MBPP splits, so we flatten without
    dedup concerns. ``test_list`` shape is identical across splits.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for split in _SPLITS:
            ds = load_dataset("google-research-datasets/mbpp", split=split)
            for row in ds:
                task_id = int(row["task_id"])
                record = {
                    "task_id": task_id,
                    "split": split,
                    "prompt": row["text"].strip(),
                    "hidden_tests": _test_block(row["test_list"], task_id=task_id),
                    "test_imports": (row.get("test_setup_code") or "").strip(),
                    "canonical_solution": row["code"],
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
        default=Path("data/mbpp_train.jsonl"),
        help="Output JSONL path (default: data/mbpp_train.jsonl).",
    )
    args = parser.parse_args()
    n = ingest(args.output)
    print(f"wrote {n} MBPP train problems → {args.output}")


if __name__ == "__main__":
    main()
