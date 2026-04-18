"""Generate the README charts. Run once, commit the PNGs.

Two outputs under ``docs/images/``:
- ``delta-by-config.png`` — headline bar chart, Δ across the four configurations
  measured on ``coding_hard.jsonl``. Error bars for the two configs where we
  actually have 95% bootstrap CIs.
- ``per-case-lift.png`` — horizontal grouped bar chart of per-case baseline vs
  final score for the winning configuration (test-grounded, post-fix). Sorted
  by Δ so the wins float to the top.

Run: ``python scripts/make_readme_charts.py``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Matplotlib's default DejaVu Sans is a strong "AI-generated chart" tell.
# Helvetica/Arial read as neutral editorial across platforms.
mpl.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Data -------------------------------------------------------------------

CONFIGS = [
    ("Symmetric CAI\n(3b judge ↔ 3b student)", -0.1667, None, None, 4),
    ("Asymmetric CAI\n(3b judge > 1b student)", -0.1111, None, None, 5),
    ("Test-grounded\n(baseline)", +0.0741, +0.0000, +0.2222, 0),
    ("Test-grounded\n(after bug fixes)", +0.2513, +0.0952, +0.4193, 0),
]

PER_CASE = [
    # (name, before, after)
    ("longest_substring_no_repeat", 0.111, 0.889),
    ("lru_cache_class", 0.167, 0.833),
    ("three_sum", 0.000, 0.571),
    ("kth_largest_element", 0.429, 1.000),
    ("rotate_array_in_place", 0.571, 1.000),
    ("binary_search_edge_cases", 1.000, 1.000),
    ("merge_intervals", 0.857, 0.857),
    ("valid_parentheses_multi", 0.800, 0.800),
    ("group_anagrams_sorted", 0.667, 0.667),
    ("topological_sort_with_cycle_detection", 0.429, 0.429),
    ("word_break_dp", 0.143, 0.143),
    ("count_islands_dfs", 0.000, 0.000),
]


# -- Styling ----------------------------------------------------------------

RED = "#c0392b"
GREEN = "#27ae60"
BLUE = "#2c7fb8"
GRAY = "#7f8c8d"
BG = "#ffffff"
FG = "#1a1a1a"
GRID = "#e6e6e6"


def _setup_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color(GRAY)
    ax.tick_params(colors=FG, which="both")
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)


# -- Chart A: Δ by config ---------------------------------------------------


def render_delta_by_config() -> None:
    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    fig.patch.set_facecolor(BG)

    labels = [c[0] for c in CONFIGS]
    deltas = [c[1] for c in CONFIGS]
    ci_low = [c[2] for c in CONFIGS]
    ci_high = [c[3] for c in CONFIGS]

    # Error bars: only draw where we have CI data.
    errs_low = [
        (d - lo) if (lo is not None) else 0 for d, lo in zip(deltas, ci_low, strict=True)
    ]
    errs_high = [
        (hi - d) if (hi is not None) else 0 for d, hi in zip(deltas, ci_high, strict=True)
    ]
    yerr = [errs_low, errs_high]

    colors = [RED if d < 0 else GREEN for d in deltas]

    xs = list(range(len(labels)))
    ax.bar(
        xs,
        deltas,
        color=colors,
        width=0.65,
        zorder=3,
        edgecolor="white",
        linewidth=1.2,
    )
    ax.errorbar(
        xs,
        deltas,
        yerr=yerr,
        fmt="none",
        ecolor=FG,
        elinewidth=1.4,
        capsize=6,
        zorder=4,
    )

    # Zero baseline
    ax.axhline(0, color=FG, linewidth=1.2, zorder=2)

    # Value labels
    for i, d in enumerate(deltas):
        offset = 0.012 if d >= 0 else -0.012
        va = "bottom" if d >= 0 else "top"
        ax.text(
            i,
            d + offset,
            f"{d:+.4f}",
            ha="center",
            va=va,
            fontsize=11,
            fontweight="bold",
            color=FG,
        )

    # Loss / CI-absence annotations below each bar.
    for i, cfg in enumerate(CONFIGS):
        losses, ci_lo = cfg[4], cfg[2]
        if ci_lo is None:
            note, color = f"{losses} losses · single run", RED
        elif losses > 0:
            note, color = f"{losses} losses", RED
        else:
            note, color = "0 losses · 95% CI", GREEN
        ax.text(
            i,
            -0.42,
            note,
            ha="center",
            va="top",
            fontsize=9,
            color=color,
            style="italic",
        )

    ax.set_ylim(-0.46, 0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10.5, color=FG)
    ax.set_ylabel("Δ aggregate score  (after − before)", fontsize=11, color=FG)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_title(
        "Hypothesis testing: which critic actually improves the Student?",
        fontsize=14,
        fontweight="bold",
        color=FG,
        pad=18,
        loc="left",
    )
    fig.text(
        0.125,
        0.905,
        "coding_hard.jsonl · 12 cases · llama3.2:1b Student · 5 revision rounds max",
        fontsize=10,
        color=GRAY,
    )
    fig.text(
        0.125,
        0.03,
        "Three approaches produced net-negative or marginal outcomes; "
        "the test-grounded loop after two rounds of adversarial review clears "
        "+0.25 with 95% CI strictly positive.",
        fontsize=9,
        color=GRAY,
        style="italic",
    )

    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    out = OUT_DIR / "delta-by-config.png"
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -- Chart B: Per-case lift -------------------------------------------------


def render_per_case_lift() -> None:
    # Sort by Δ descending so wins float to top.
    data = sorted(PER_CASE, key=lambda row: row[2] - row[1], reverse=True)
    names = [d[0] for d in data]
    before = [d[1] for d in data]
    after = [d[2] for d in data]
    deltas = [a - b for a, b in zip(after, before, strict=True)]

    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    fig.patch.set_facecolor(BG)

    ys = list(range(len(names)))
    bar_height = 0.38
    ys_before = [y + bar_height / 2 for y in ys]
    ys_after = [y - bar_height / 2 for y in ys]

    ax.barh(
        ys_before,
        before,
        height=bar_height,
        color=GRAY,
        label="Before (baseline)",
        zorder=3,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.barh(
        ys_after,
        after,
        height=bar_height,
        color=BLUE,
        label="After (test-grounded loop)",
        zorder=3,
        edgecolor="white",
        linewidth=0.5,
    )

    # Δ labels on the right
    for y, b, a, d in zip(ys, before, after, deltas, strict=True):
        if d > 0:
            ax.text(
                max(b, a) + 0.02,
                y,
                f"Δ +{d:.2f}",
                va="center",
                ha="left",
                fontsize=10,
                color=GREEN,
                fontweight="bold",
            )
        else:
            ax.text(
                max(b, a) + 0.02,
                y,
                f"tie",
                va="center",
                ha="left",
                fontsize=9,
                color=GRAY,
                style="italic",
            )

    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=10, color=FG)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.22)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Hidden-test pass rate", fontsize=11, color=FG)
    _setup_ax(ax)

    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=10,
        labelcolor=FG,
    )
    ax.set_title(
        "Per-case lift on coding_hard — 5 wins, 7 ties, 0 losses",
        fontsize=14,
        fontweight="bold",
        color=FG,
        pad=14,
        loc="left",
    )
    fig.text(
        0.125,
        0.925,
        "llama3.2:1b Student · --critique-mode tests · 5 revision rounds max",
        fontsize=10,
        color=GRAY,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = OUT_DIR / "per-case-lift.png"
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    render_delta_by_config()
    render_per_case_lift()
