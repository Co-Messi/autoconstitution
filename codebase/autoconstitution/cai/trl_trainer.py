"""
TRL integration — thin wrapper around HuggingFace TRL's DPOTrainer.

Kept intentionally minimal: TRL's API moves fast and we don't want to re-export
every knob. This module just standardises:

    - reading preference pairs (from JSONL or HF dataset)
    - loading a base model + tokenizer in a way that works on both CUDA and MPS
    - launching DPO training with sensible defaults
    - emitting a post-training eval + ratchet decision

Requires: ``pip install "autoconstitution[train]"``

Example:
    >>> from autoconstitution.cai.trl_trainer import DPOConfig, run_dpo
    >>> cfg = DPOConfig(
    ...     base_model="Qwen/Qwen2.5-1.5B",
    ...     train_file=Path("data/train.jsonl"),
    ...     output_dir=Path("checkpoints/gen-001"),
    ... )
    >>> metrics = run_dpo(cfg)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Minimal configuration for a DPO training run."""

    base_model: str
    train_file: Path
    output_dir: Path

    eval_file: Optional[Path] = None
    learning_rate: float = 5e-7
    beta: float = 0.1  # DPO temperature; lower = stay closer to ref model
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    max_prompt_length: int = 512
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)


def _lazy_imports() -> tuple[Any, ...]:
    """Import heavy deps only when actually training."""
    try:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )
        from trl import DPOConfig as TRLDPOConfig  # type: ignore[import-not-found]
        from trl import DPOTrainer  # type: ignore[import-not-found]
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "Training dependencies missing. Install with:\n"
            "    pip install 'autoconstitution[train]'\n"
            f"Original error: {e}"
        ) from e

    try:
        from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]
    except ImportError:
        LoraConfig = None  # type: ignore[assignment]
        get_peft_model = None  # type: ignore[assignment]

    return (
        torch,
        AutoModelForCausalLM,
        AutoTokenizer,
        TRLDPOConfig,
        DPOTrainer,
        load_dataset,
        LoraConfig,
        get_peft_model,
    )


def _detect_device() -> str:
    """Pick CUDA / MPS / CPU in that order."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_dpo(cfg: DPOConfig) -> dict[str, Any]:
    """Execute a single DPO training round. Returns a metrics dict."""
    (
        torch,
        AutoModelForCausalLM,
        AutoTokenizer,
        TRLDPOConfig,
        DPOTrainer,
        load_dataset,
        LoraConfig,
        get_peft_model,
    ) = _lazy_imports()

    device = _detect_device()
    logger.info("training on device=%s", device)

    # 1. Dataset ---------------------------------------------------------
    data_files: dict[str, str] = {"train": str(cfg.train_file)}
    if cfg.eval_file is not None:
        data_files["eval"] = str(cfg.eval_file)
    ds = load_dataset("json", data_files=data_files)
    train_ds = ds["train"]
    eval_ds = ds.get("eval")

    # 2. Tokenizer + model ----------------------------------------------
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
    )

    if cfg.use_peft and LoraConfig is not None:
        peft_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    # 3. Training config ------------------------------------------------
    training_args = TRLDPOConfig(
        output_dir=str(cfg.output_dir),
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        seed=cfg.seed,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_ds is not None else "no",
        **cfg.extra,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT → uses the base weights via the adapter
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
    )

    # 4. Train ----------------------------------------------------------
    train_result = trainer.train()
    trainer.save_model(str(cfg.output_dir))
    tok.save_pretrained(str(cfg.output_dir))

    # 5. Eval -----------------------------------------------------------
    eval_metrics: dict[str, Any] = {}
    if eval_ds is not None:
        eval_metrics = trainer.evaluate()

    return {
        "device": device,
        "base_model": cfg.base_model,
        "output_dir": str(cfg.output_dir),
        "train_metrics": train_result.metrics if hasattr(train_result, "metrics") else {},
        "eval_metrics": eval_metrics,
    }
