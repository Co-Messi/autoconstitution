"""
Preference pair generation for DPO / IPO / KTO training.

Converts RevisionResult traces from the critique-revision loop into the
(prompt, chosen, rejected) triples that TRL's DPOTrainer consumes.

Also handles:
    - Filtering for pair quality (skip pairs where chosen == rejected).
    - Anti-collapse: preserving a configurable fraction of original human data.
    - Train/eval split with stratification by prompt domain.
    - Export to JSONL and HuggingFace Datasets formats.

Example:
    >>> builder = PreferencePairBuilder(min_edit_distance=10)
    >>> builder.add_results(revision_results)
    >>> builder.add_anchor(human_pairs, fraction=0.1)
    >>> builder.export_jsonl(Path("data/train.jsonl"))
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoconstitution.cai.critique_revision import RevisionResult

logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    """A single DPO training example."""

    prompt: str
    chosen: str
    rejected: str
    source: str = "cai"  # "cai" | "human_anchor" | "external"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "source": self.source,
            "metadata": self.metadata,
        }

    @property
    def is_trivial(self) -> bool:
        """True if chosen and rejected differ by only whitespace."""
        return self.chosen.strip() == self.rejected.strip()

    def edit_distance(self) -> int:
        """Cheap approximation: absolute length delta.

        Real Levenshtein is O(n*m); for filtering purposes length delta is
        sufficient. Use ``difflib.SequenceMatcher`` if you need a real score.
        """
        return abs(len(self.chosen) - len(self.rejected))


def _ended_in_parse_error(result: RevisionResult) -> bool:
    """True if the final judge verdict was parse_error.

    CAI can't produce a trustworthy preference pair when the last signal was
    garbage — the Student was told to revise off a failed-to-parse critique
    and the chosen/rejected labels are noise.
    """
    if not result.critiques:
        return False
    return result.critiques[-1].verdict == "parse_error"


class PreferencePairBuilder:
    """Collect, filter, and export preference pairs for DPO training."""

    def __init__(
        self,
        *,
        min_edit_distance: int = 5,
        drop_non_converged: bool = False,
        min_critique_items: int = 1,
        drop_parse_error_traces: bool = True,
        seed: int = 42,
    ) -> None:
        """Configure the gates.

        Args:
            min_edit_distance: Minimum length delta between chosen and rejected
                answers; pairs that differ by less are treated as trivial.
            drop_non_converged: Drop traces where the judge never said
                compliant. Off by default — non-converged traces still carry
                signal if the judge produced concrete critiques.
            min_critique_items: Minimum total critique items across all rounds
                required for a trace that performed at least one revision.
                Guards against the judge saying needs_revision with an empty
                critiques list (Student revises blind → noise pair). Set to
                ``0`` as an escape hatch for verdict-only graders that don't
                articulate items.
            drop_parse_error_traces: Drop traces whose final verdict was
                parse_error (the Student revised off an unparseable critique).
            seed: RNG seed for reproducible shuffling.
        """
        self.min_edit_distance = min_edit_distance
        self.drop_non_converged = drop_non_converged
        self.min_critique_items = min_critique_items
        self.drop_parse_error_traces = drop_parse_error_traces
        self._rng = random.Random(seed)
        self._pairs: list[PreferencePair] = []

    # -- Ingestion ---------------------------------------------------------

    def add_results(self, results: Iterable[RevisionResult]) -> int:
        """Convert CAI revision traces to preference pairs. Returns count added."""
        results_list = list(results)
        added = 0
        for r in results_list:
            if self.drop_non_converged and not r.converged:
                continue
            if self.drop_parse_error_traces and _ended_in_parse_error(r):
                continue
            total_items = sum(len(c.critiques) for c in r.critiques)
            # A trace with revisions but no concrete critique items means the
            # Student was asked to "fix" the answer with no pointer to what
            # was wrong — the result is a random walk, not a preference signal.
            if r.rounds_used > 0 and total_items < self.min_critique_items:
                continue
            pair = PreferencePair(
                prompt=r.prompt,
                chosen=r.chosen,
                rejected=r.rejected,
                source="cai",
                metadata={
                    "rounds_used": r.rounds_used,
                    "converged": r.converged,
                    "num_critiques": total_items,
                },
            )
            if self._accept(pair):
                self._pairs.append(pair)
                added += 1
        logger.info("added %d pairs from %d results", added, len(results_list))
        return added

    def add_anchor(
        self,
        human_pairs: list[PreferencePair],
        *,
        fraction: float = 0.1,
    ) -> int:
        """Inject human-written pairs as anti-collapse anchors.

        The CAI paper and the model-collapse literature both recommend keeping
        a non-trivial fraction (≥10%) of the dataset as real human labels to
        prevent the model from drifting into self-referential nonsense.
        """
        if not human_pairs:
            return 0
        if fraction <= 0 or fraction >= 1:
            raise ValueError("fraction must be in (0, 1)")

        current_size = len(self._pairs)
        # Solve: anchor_count / (current + anchor_count) = fraction
        # => anchor_count = fraction * current / (1 - fraction)
        target = int(round(fraction * current_size / (1 - fraction)))
        target = min(target, len(human_pairs))
        chosen = self._rng.sample(human_pairs, target) if target > 0 else []
        for p in chosen:
            p.source = "human_anchor"
            self._pairs.append(p)
        logger.info("added %d human anchor pairs (target fraction=%.2f)", len(chosen), fraction)
        return len(chosen)

    def _accept(self, pair: PreferencePair) -> bool:
        if pair.is_trivial:
            return False
        return pair.edit_distance() >= self.min_edit_distance

    # -- Splitting ---------------------------------------------------------

    def split(
        self,
        *,
        eval_fraction: float = 0.1,
    ) -> tuple[list[PreferencePair], list[PreferencePair]]:
        """Shuffle and split into (train, eval)."""
        pairs = list(self._pairs)
        self._rng.shuffle(pairs)
        cutoff = int(len(pairs) * (1 - eval_fraction))
        return pairs[:cutoff], pairs[cutoff:]

    # -- Export ------------------------------------------------------------

    def export_jsonl(self, path: Path) -> int:
        """Write pairs to a JSONL file. Returns record count."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for pair in self._pairs:
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
        logger.info("wrote %d pairs to %s", len(self._pairs), path)
        return len(self._pairs)

    def to_hf_dataset(self) -> Any:
        """Return a HuggingFace `datasets.Dataset`. Requires `datasets` installed."""
        try:
            from datasets import Dataset  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "HuggingFace `datasets` is required. "
                "Install with: pip install 'autoconstitution[train]'"
            ) from e
        return Dataset.from_list([p.to_dict() for p in self._pairs])

    # -- Diagnostics -------------------------------------------------------

    def __len__(self) -> int:
        return len(self._pairs)

    def stats(self) -> dict[str, Any]:
        """Summary: count by source, avg edit distance, etc."""
        by_source: dict[str, int] = {}
        total_edit = 0
        for p in self._pairs:
            by_source[p.source] = by_source.get(p.source, 0) + 1
            total_edit += p.edit_distance()
        return {
            "total": len(self._pairs),
            "by_source": by_source,
            "avg_edit_distance": total_edit / max(len(self._pairs), 1),
        }
