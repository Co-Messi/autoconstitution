"""Bootstrap confidence intervals for small benchmark samples.

With 10-20 cases the interval will be wide — that's honest. Separating
the math into its own module keeps the runner free of numerical noise
and makes the CI logic independently testable.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def bootstrap_ci_mean(
    samples: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int | None = 0,
) -> tuple[float, float]:
    """Return the bootstrap CI on the sample mean.

    Args:
        samples: observed values (e.g., per-case deltas).
        confidence: two-sided confidence level, e.g. ``0.95`` for 95%.
        resamples: number of bootstrap resamples to draw.
        seed: random seed for reproducibility. Use ``None`` for nondeterministic.

    Returns:
        ``(low, high)`` — the lower and upper percentile bounds of the mean
        across resamples. Returns ``(0.0, 0.0)`` on an empty input rather
        than raising, since a benchmark with no cases is a degenerate but
        legal report.
    """
    if not samples:
        return (0.0, 0.0)
    if len(samples) == 1:
        only = samples[0]
        return (only, only)

    rng = random.Random(seed)
    n = len(samples)
    means: list[float] = []
    for _ in range(resamples):
        draw = [rng.choice(samples) for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()

    alpha = 1.0 - confidence
    lo_idx = max(0, int(alpha / 2.0 * resamples))
    hi_idx = min(resamples - 1, int((1.0 - alpha / 2.0) * resamples))
    return (means[lo_idx], means[hi_idx])


__all__ = ["bootstrap_ci_mean"]
