"""SFT fine-tuning of llama3.2:1b via mlx-lm's LoRA trainer.

Reads a DPO-format JSONL (prompt/chosen/rejected) produced by the consultant's
pair-generation pipeline, extracts the chosen answers only, and fine-tunes
the Student to produce them. This is supervised specialization — weaker
than true DPO because it only teaches "produce good output" not "avoid bad
output," but reliable on Apple Silicon and fast to ship.

If this attempt doesn't clear a meaningful Δ on HumanEval+, the fallback is
true DPO via TRL + PEFT (fallback #5 in the experiment ladder).

Usage:
    python scripts/experiments/train_sft_mlx.py \\
        --pairs train_pairs_3b.jsonl \\
        --adapter-dir checkpoints/llama-1b-sft-3b-teacher \\
        --iters 300

Output: an MLX adapter directory ready to use with ``mlx_lm.generate``.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

MLX_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def _dpo_to_chat(record: dict) -> dict:
    """Transform a DPO record into mlx-lm's chat message format.

    We only keep the chosen answer — that's what SFT learns from. The
    rejected answer is dropped at this stage (will be revisited if we
    upgrade to true DPO via TRL in a later iteration).
    """
    return {
        "messages": [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["chosen"]},
        ]
    }


def _prepare_dataset(
    pairs_path: Path, out_dir: Path, val_fraction: float, seed: int
) -> tuple[int, int]:
    """Read a DPO JSONL, convert to chat format, split into train/valid.

    Returns ``(n_train, n_valid)``.
    """
    with pairs_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        raise ValueError(f"No records in {pairs_path}")

    rng = random.Random(seed)
    rng.shuffle(records)
    cutoff = max(1, int(len(records) * (1 - val_fraction)))
    train_records = records[:cutoff]
    valid_records = records[cutoff:]
    # Guarantee at least one validation example — mlx-lm requires valid.jsonl.
    if not valid_records:
        valid_records = [train_records[-1]]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(_dpo_to_chat(r), ensure_ascii=False) + "\n")
    with valid_path.open("w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(_dpo_to_chat(r), ensure_ascii=False) + "\n")

    return len(train_records), len(valid_records)


def _write_lora_config(
    config_path: Path,
    *,
    rank: int,
    scale: float,
    dropout: float,
) -> None:
    """Write a minimal mlx-lm YAML config with lora_parameters.

    mlx-lm's ``--num-layers`` sets how many transformer layers get LoRA
    modules attached, which is orthogonal to the rank ``r`` of the low-rank
    decomposition. Rank and scale (== alpha / rank) only go through the
    YAML config file, passed via ``-c``.

    Note: mlx-lm's own default ``scale`` is 20.0, which is aggressive for
    small datasets. We default to 2.0 (community convention: alpha = 2*rank
    → scale = alpha/rank = 2.0), which is safer for overfit-prone runs.
    """
    lines = [
        "lora_parameters:",
        f"  rank: {rank}",
        f"  scale: {scale}",
        f"  dropout: {dropout}",
        "",
    ]
    config_path.write_text("\n".join(lines), encoding="utf-8")


def _run_mlx_lora(
    *,
    model: str,
    data_dir: Path,
    adapter_dir: Path,
    iters: int,
    batch_size: int,
    learning_rate: float,
    num_layers: int,
    lora_rank: int,
    lora_scale: float,
    lora_dropout: float,
    steps_per_report: int,
    steps_per_eval: int,
    seed: int,
) -> int:
    """Invoke mlx_lm.lora as a subprocess. Returns the exit code."""
    adapter_dir.mkdir(parents=True, exist_ok=True)

    config_path = adapter_dir / "lora_config.yaml"
    _write_lora_config(
        config_path, rank=lora_rank, scale=lora_scale, dropout=lora_dropout
    )

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "-c",
        str(config_path),
        "--train",
        "--model",
        model,
        "--data",
        str(data_dir),
        "--fine-tune-type",
        "lora",
        "--num-layers",
        str(num_layers),
        "--batch-size",
        str(batch_size),
        "--iters",
        str(iters),
        "--learning-rate",
        str(learning_rate),
        "--adapter-path",
        str(adapter_dir),
        "--mask-prompt",
        "--save-every",
        str(max(50, iters // 4)),
        "--steps-per-report",
        str(steps_per_report),
        "--steps-per-eval",
        str(steps_per_eval),
        "--seed",
        str(seed),
    ]
    print(f"[train] {' '.join(cmd)}", flush=True)

    # Tee stdout+stderr to a log file inside the adapter dir so we can
    # post-process train/val loss into a CSV and spot overfit fingerprints
    # (prior regression was train=0.001 / val=0.247).
    log_path = adapter_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_f.write(line)
        return proc.wait()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="Path to DPO-format JSONL (prompt/chosen/rejected).",
    )
    p.add_argument(
        "--adapter-dir",
        type=Path,
        required=True,
        help="Directory to save the trained LoRA adapter.",
    )
    p.add_argument(
        "--model",
        default=MLX_MODEL,
        help=f"Base MLX model (default: {MLX_MODEL}).",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=300,
        help="Training iterations. Default 300 is ~1-2 epochs on 500 pairs.",
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument(
        "--num-layers",
        type=int,
        default=16,
        help="Number of transformer layers receiving LoRA modules "
        "(mlx-lm default is 16). Orthogonal to --lora-rank.",
    )
    p.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="Rank r of the LoRA low-rank decomposition (default 8).",
    )
    p.add_argument(
        "--lora-scale",
        type=float,
        default=2.0,
        help="LoRA scale (== alpha / rank). Default 2.0 matches the "
        "community 'alpha = 2*rank' convention. mlx-lm's own default "
        "is 20.0, which is aggressive for small datasets.",
    )
    p.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout (default 0.0).",
    )
    p.add_argument(
        "--steps-per-report",
        type=int,
        default=10,
        help="Training-loss log cadence (iters).",
    )
    p.add_argument(
        "--steps-per-eval",
        type=int,
        default=20,
        help="Validation-loss eval cadence (iters).",
    )
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.pairs.exists():
        print(f"error: {args.pairs} not found", file=sys.stderr)
        return 2

    data_dir = args.adapter_dir / "dataset"
    n_train, n_valid = _prepare_dataset(
        args.pairs, data_dir, args.val_fraction, args.seed
    )
    print(f"[train] prepared {n_train} train + {n_valid} valid records")

    rc = _run_mlx_lora(
        model=args.model,
        data_dir=data_dir,
        adapter_dir=args.adapter_dir,
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_layers=args.num_layers,
        lora_rank=args.lora_rank,
        lora_scale=args.lora_scale,
        lora_dropout=args.lora_dropout,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        seed=args.seed,
    )
    if rc != 0:
        print(f"[train] mlx_lm.lora exited with code {rc}", file=sys.stderr)
        return rc

    manifest = {
        "model": args.model,
        "adapter_dir": str(args.adapter_dir),
        "pairs_source": str(args.pairs),
        "n_train": n_train,
        "n_valid": n_valid,
        "iters": args.iters,
        "learning_rate": args.learning_rate,
        "num_layers": args.num_layers,
        "lora_rank": args.lora_rank,
        "lora_scale": args.lora_scale,
        "lora_dropout": args.lora_dropout,
        "batch_size": args.batch_size,
        "steps_per_report": args.steps_per_report,
        "steps_per_eval": args.steps_per_eval,
        "seed": args.seed,
    }
    (args.adapter_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[train] done, adapter at {args.adapter_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
