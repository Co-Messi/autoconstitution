#!/usr/bin/env bash
# Autonomous end-to-end experiment runner.
#
# Chains: consultant's best-of-N pair-gen → mlx-lm SFT → three-way eval.
# Logs everything under artifacts/<timestamp>/. Exits cleanly when done; the
# Claude agent harness picks up the completion notification and pings the
# user's phone via PushNotification.
#
# Pipeline pivoted from (3b-teacher + minimax-teacher critique-revise) to
# best-of-N self-distillation after smoke tests showed the 1b Student can't
# act on LLM-generated critiques (capacity-limited, not judge-quality-limited).
# So we now have ONE pair set, not two.
#
# Expects the consultant's outputs to exist before this runs:
#   data/train_pairs_bestofn.jsonl          (best-of-N pairs)
#   data/humaneval_plus_subset.jsonl        (held-out eval)
#   scripts/experiments/humaneval_scorer.py (eval scorer)
#
# Writes:
#   artifacts/<run>/training.log
#   artifacts/<run>/eval.log
#   artifacts/<run>/eval_report.json
#   artifacts/<run>/SUMMARY.md

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT="artifacts/$RUN_ID"
mkdir -p "$OUT"

PAIRS="${PAIRS:-data/train_pairs_bestofn.jsonl}"
EVAL_PROBLEMS="${EVAL_PROBLEMS:-data/humaneval_plus_subset.jsonl}"
SEEDS="${SEEDS:-3}"

ADAPTER="$OUT/adapter-bestofn"

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"
}

# ---------- preflight ------------------------------------------------------

log "run id: $RUN_ID"
log "expected inputs:"
log "  pairs:          $PAIRS"
log "  eval problems:  $EVAL_PROBLEMS"

for f in "$PAIRS" "$EVAL_PROBLEMS"; do
  if [[ ! -s "$f" ]]; then
    log "ERROR: required input missing or empty: $f"
    log "(is the consultant's pair-gen + eval-scorer pipeline finished?)"
    exit 2
  fi
done

# Right-size training iters based on pair count.
# Rule: aim for ~6 epochs with batch size 2. Small-data LoRA.
# 60 pairs → 180 iters, 100 pairs → 300 iters, 200 pairs → 600 iters.
PAIR_COUNT=$(wc -l < "$PAIRS" | tr -d ' ')
ITERS="${ITERS:-$((PAIR_COUNT * 3))}"
# Floor and ceiling guard rails.
if (( ITERS < 100 )); then ITERS=100; fi
if (( ITERS > 1200 )); then ITERS=1200; fi
log "pair count: $PAIR_COUNT  →  training iters: $ITERS"

# ---------- training -------------------------------------------------------

log "starting SFT training on best-of-N pairs…"
python scripts/experiments/train_sft_mlx.py \
  --pairs "$PAIRS" \
  --adapter-dir "$ADAPTER" \
  --iters "$ITERS" \
  2>&1 | tee "$OUT/training.log"
log "training done"

# ---------- three-way eval -------------------------------------------------

MAX_PROBLEMS="${MAX_PROBLEMS:-40}"
log "starting three-way eval (base 1b, base 8b, trained 1b) on $MAX_PROBLEMS problems…"
python scripts/experiments/eval_three_way.py \
  --problems "$EVAL_PROBLEMS" \
  --adapter "$ADAPTER" \
  --output "$OUT/eval_report.json" \
  --seeds "$SEEDS" \
  --max-problems "$MAX_PROBLEMS" \
  --concurrency 1 \
  2>&1 | tee "$OUT/eval.log"
log "eval done"

# ---------- summary --------------------------------------------------------

python - <<PY > "$OUT/SUMMARY.md"
import json, pathlib
report = json.loads(pathlib.Path("$OUT/eval_report.json").read_text())
lines = [
  f"# Experiment run $RUN_ID",
  "",
  f"Problems: {report['n_problems']}  ·  Seeds: {report['n_seeds']}",
  "",
  "| Model | pass@1 | 95% CI | Δ vs base 8b |",
  "|---|---:|:---|---:|",
]
for name, data in report["models"].items():
    lo, hi = data["ci95"]
    mean = data["mean_pass@1"]
    delta = data.get("delta_vs_base_8b")
    delta_str = f"{delta:+.3f}" if delta is not None else "—"
    lines.append(f"| {name} | {mean:.3f} | [{lo:.3f}, {hi:.3f}] | {delta_str} |")
lines.append("")
# Headline interpretation
trained_names = [n for n in report["models"] if n.startswith("trained_")]
base8 = report["models"].get("base_8b", {}).get("mean_pass@1", 0.0)
best_trained = max(
    (report["models"][n]["mean_pass@1"], n) for n in trained_names
) if trained_names else None
if best_trained:
    best_score, best_name = best_trained
    if best_score > base8:
        lines.append(
            f"**Headline:** {best_name} ({best_score:.3f}) beats base_8b "
            f"({base8:.3f}) on HumanEval+ pass@1."
        )
    else:
        gap = base8 - best_score
        pct_closed = max(0.0, 1 - gap / max(base8 - report["models"]["base_1b"]["mean_pass@1"], 1e-6))
        lines.append(
            f"**Result:** best trained 1b = {best_name} ({best_score:.3f}). "
            f"Did NOT beat base_8b ({base8:.3f}); closed "
            f"{pct_closed * 100:.0f}% of the base_1b→base_8b gap."
        )
print("\n".join(lines))
PY

log "SUMMARY written to $OUT/SUMMARY.md"
log "run complete"
