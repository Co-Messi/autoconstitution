"""Verify MBPP train and HumanEval+ eval are disjoint.

A single leaked problem (same task appearing in both train and eval sets)
invalidates the "held-out" claim. This script hashes normalized prompts
from each file and reports any overlap. Exits non-zero if overlap found.

Normalization: lowercase, strip, collapse whitespace, drop punctuation.
Aggressive on purpose — we'd rather false-positive and investigate than
miss a near-duplicate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9 ]")


def _normalize(text: str) -> str:
    """Lowercase, strip, drop punctuation, collapse whitespace."""
    lower = text.lower().strip()
    no_punct = _PUNCT.sub(" ", lower)
    return _WS.sub(" ", no_punct).strip()


def _hash_prompt(text: str) -> str:
    return hashlib.sha1(_normalize(text).encode("utf-8")).hexdigest()[:16]


def _load_prompts(path: Path) -> dict[str, tuple[str, str]]:
    """Return ``{hash: (task_id, original_prompt)}``."""
    out: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt", ""))
            h = _hash_prompt(prompt)
            out[h] = (str(row.get("task_id", "?")), prompt)
    return out


def check(mbpp_path: Path, heval_path: Path) -> int:
    mbpp = _load_prompts(mbpp_path)
    heval = _load_prompts(heval_path)
    overlap = set(mbpp) & set(heval)
    print(f"MBPP train:       {len(mbpp)} prompts")
    print(f"HumanEval+ eval:  {len(heval)} prompts")
    print(f"Overlap:          {len(overlap)} prompt(s)")
    if overlap:
        print("\nOVERLAPPING PROMPTS (normalized-prompt hash match):")
        for h in overlap:
            mb_id, mb_p = mbpp[h]
            he_id, he_p = heval[h]
            print(f"  hash={h}")
            print(f"    MBPP[{mb_id}]:       {mb_p[:120]!r}")
            print(f"    HumanEval+[{he_id}]: {he_p[:120]!r}")
        return 1
    print("\nDISJOINT — no overlap.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mbpp",
        type=Path,
        default=Path("data/mbpp_train.jsonl"),
    )
    parser.add_argument(
        "--humanevalplus",
        type=Path,
        default=Path("data/humanevalplus_eval.jsonl"),
    )
    args = parser.parse_args()
    sys.exit(check(args.mbpp, args.humanevalplus))


if __name__ == "__main__":
    main()
