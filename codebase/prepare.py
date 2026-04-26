"""
autoconstitution Data Preparation Script.

This script downloads, tokenizes, and prepares datasets for autoconstitution experiments.
Inspired by Karpathy's prepare.py from nanoGPT/llm.c.

Usage:
    python prepare.py --dataset openwebtext
    python prepare.py --dataset fineweb --sample_size 1000000
    python prepare.py --dataset custom --data_dir /path/to/data

The script will:
    1. Download the specified dataset (if not cached)
    2. Tokenize using the specified tokenizer
    3. Save tokenized data as memory-mappable binary files
    4. Generate a metadata file with dataset statistics
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------------------------


class Tokenizer(Protocol):
    """Protocol for tokenizer interface."""
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...
    
    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        ...
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        ...
    
    @property
    def eot_token(self) -> int:
        """Return end-of-text token ID."""
        ...


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    url: str | None = None
    huggingface_name: str | None = None
    split: str = "train"
    text_column: str = "text"
    sample_size: int | None = None
    shuffle: bool = True
    seed: int = 1337


@dataclass  
class TokenizerConfig:
    """Configuration for tokenizer."""
    name: str = "gpt2"
    vocab_size: int = 50257
    custom_tokenizer_path: str | None = None


@dataclass
class PrepareConfig:
    """Configuration for data preparation."""
    dataset: DatasetConfig
    tokenizer: TokenizerConfig
    output_dir: Path
    num_proc: int = 8
    batch_size: int = 1000
    validation_split: float = 0.01
    compress: bool = False


@dataclass
class DatasetStats:
    """Statistics about the prepared dataset."""
    dataset_name: str
    tokenizer_name: str
    num_tokens: int
    num_documents: int
    vocab_size: int
    avg_tokens_per_doc: float
    split: str
    file_path: Path
    compression_ratio: float = 1.0


# ---------------------------------------------------------------------------
# Tokenizer Implementations
# ---------------------------------------------------------------------------


class GPT2Tokenizer:
    """GPT-2 tokenizer wrapper using tiktoken."""
    
    def __init__(self) -> None:
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for GPT2Tokenizer. "
                "Install with: pip install tiktoken"
            )
        self.enc = tiktoken.get_encoding("gpt2")
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})
    
    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        return self.enc.decode(tokens)
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.enc.n_vocab
    
    @property
    def eot_token(self) -> int:
        """Return end-of-text token ID."""
        return self.enc.eot_token


class CharacterTokenizer:
    """Simple character-level tokenizer for debugging/small experiments."""
    
    def __init__(self, chars: str | None = None) -> None:
        self.chars = chars or (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            " !@#$%^&*()_+-=[]{}|;':\",./<>?\n"
        )
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return [self.stoi.get(c, self.stoi.get(" ", 0)) for c in text]
    
    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        return "".join(self.itos.get(t, " ") for t in tokens)
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.chars)
    
    @property
    def eot_token(self) -> int:
        """Return end-of-text token ID."""
        return 0


class HuggingFaceTokenizer:
    """Wrapper for HuggingFace tokenizers."""
    
    def __init__(self, model_name: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for HuggingFaceTokenizer. "
                "Install with: pip install transformers"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=True)
    
    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.tokenizer)
    
    @property
    def eot_token(self) -> int:
        """Return end-of-text token ID."""
        return self.tokenizer.eos_token_id or 0


def get_tokenizer(config: TokenizerConfig) -> Tokenizer:
    """Factory function to get the appropriate tokenizer."""
    if config.name == "gpt2":
        return GPT2Tokenizer()
    elif config.name == "character":
        return CharacterTokenizer()
    elif config.name.startswith("hf:"):
        model_name = config.name[3:]
        return HuggingFaceTokenizer(model_name)
    else:
        raise ValueError(f"Unknown tokenizer: {config.name}")


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------


def load_openwebtext(
    sample_size: int | None = None,
    split: str = "train",
    seed: int = 1337
) -> list[str]:
    """Load OpenWebText dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets is required for loading datasets. "
            "Install with: pip install datasets"
        )
    
    print(f"Loading OpenWebText dataset (split={split})...")
    dataset = load_dataset(
        "openwebtext",
        split="train",
        trust_remote_code=True
    )
    
    if sample_size:
        dataset = dataset.shuffle(seed=seed).select(range(min(sample_size, len(dataset))))
    
    return [item["text"] for item in dataset]


def load_fineweb(
    sample_size: int | None = None,
    split: str = "train",
    seed: int = 1337
) -> list[str]:
    """Load FineWeb dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets is required for loading datasets. "
            "Install with: pip install datasets"
        )
    
    print(f"Loading FineWeb dataset (split={split})...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        trust_remote_code=True
    )
    
    if sample_size:
        dataset = dataset.shuffle(seed=seed).select(range(min(sample_size, len(dataset))))
    
    return [item["text"] for item in dataset]


def load_tinyshakespeare(
    sample_size: int | None = None,
    split: str = "train",
    seed: int = 1337
) -> list[str]:
    """Load Tiny Shakespeare dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets is required for loading datasets. "
            "Install with: pip install datasets"
        )
    
    print(f"Loading Tiny Shakespeare dataset...")
    dataset = load_dataset(
        "tinyshakespeare",
        trust_remote_code=True
    )
    
    # Tiny Shakespeare has a single 'train' split with all data
    text = dataset["train"][0]["text"]
    
    # Split into chunks (roughly train/val/test)
    lines = text.split("\n\n")
    n = len(lines)
    
    if split == "train":
        lines = lines[:int(0.9 * n)]
    elif split == "validation":
        lines = lines[int(0.9 * n):int(0.95 * n)]
    else:
        lines = lines[int(0.95 * n):]
    
    return ["\n\n".join(lines)]


def load_custom_dataset(
    data_dir: Path,
    text_column: str = "text",
    sample_size: int | None = None,
    split: str = "train",
    seed: int = 1337
) -> list[str]:
    """Load custom dataset from directory or files."""
    texts: list[str] = []
    
    if data_dir.is_file():
        # Single file
        files = [data_dir]
    else:
        # Directory - find all text files
        files = list(data_dir.glob("**/*.txt"))
        files.extend(data_dir.glob("**/*.json"))
        files.extend(data_dir.glob("**/*.jsonl"))
    
    print(f"Loading custom dataset from {data_dir} ({len(files)} files)...")
    
    for file_path in files:
        if file_path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        elif file_path.suffix in (".json", ".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.suffix == ".jsonl":
                    for line in f:
                        data = json.loads(line)
                        texts.append(data.get(text_column, str(data)))
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                texts.append(item.get(text_column, str(item)))
                            else:
                                texts.append(str(item))
                    else:
                        texts.append(data.get(text_column, str(data)))
    
    if sample_size and len(texts) > sample_size:
        import random
        random.seed(seed)
        texts = random.sample(texts, sample_size)
    
    return texts


def load_dataset(config: DatasetConfig) -> list[str]:
    """Factory function to load the appropriate dataset."""
    if config.name == "openwebtext":
        return load_openwebtext(config.sample_size, config.split, config.seed)
    elif config.name == "fineweb":
        return load_fineweb(config.sample_size, config.split, config.seed)
    elif config.name == "tinyshakespeare":
        return load_tinyshakespeare(config.sample_size, config.split, config.seed)
    elif config.name == "custom":
        if not config.url:
            raise ValueError("Custom dataset requires url/data_dir")
        return load_custom_dataset(
            Path(config.url),
            config.text_column,
            config.sample_size,
            config.split,
            config.seed
        )
    else:
        raise ValueError(f"Unknown dataset: {config.name}")


# ---------------------------------------------------------------------------
# Tokenization and Saving
# ---------------------------------------------------------------------------


def tokenize_documents(
    documents: list[str],
    tokenizer: Tokenizer,
    batch_size: int = 1000,
    num_proc: int = 1
) -> list[list[int]]:
    """Tokenize a list of documents."""
    tokenized: list[list[int]] = []
    
    print(f"Tokenizing {len(documents):,} documents...")
    
    if num_proc > 1:
        # Parallel tokenization
        from multiprocessing import Pool
        
        def encode_batch(batch: list[str]) -> list[list[int]]:
            return [tokenizer.encode(doc) + [tokenizer.eot_token] for doc in batch]
        
        batches = [
            documents[i:i + batch_size]
            for i in range(0, len(documents), batch_size)
        ]
        
        with Pool(num_proc) as pool:
            results = list(tqdm(
                pool.imap(encode_batch, batches),
                total=len(batches),
                desc="Tokenizing"
            ))
        
        for batch_tokens in results:
            tokenized.extend(batch_tokens)
    else:
        # Sequential tokenization
        for doc in tqdm(documents, desc="Tokenizing"):
            tokens = tokenizer.encode(doc)
            tokens.append(tokenizer.eot_token)
            tokenized.append(tokens)
    
    return tokenized


def save_tokenized_data(
    tokenized_docs: list[list[int]],
    output_path: Path,
    dtype: np.dtype = np.uint16
) -> tuple[int, float]:
    """Save tokenized data to disk as memory-mappable binary file.
    
    Returns:
        Tuple of (num_tokens, file_size_mb)
    """
    # Flatten all tokens into a single array
    all_tokens: list[int] = []
    for tokens in tokenized_docs:
        all_tokens.extend(tokens)
    
    # Convert to numpy array
    arr = np.array(all_tokens, dtype=dtype)
    
    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use numpy's memmap-compatible format
    with open(output_path, "wb") as f:
        f.write(arr.tobytes())
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    return len(all_tokens), file_size_mb


def save_metadata(
    stats: DatasetStats,
    output_path: Path
) -> None:
    """Save dataset metadata to JSON."""
    metadata = {
        "dataset_name": stats.dataset_name,
        "tokenizer_name": stats.tokenizer_name,
        "num_tokens": stats.num_tokens,
        "num_documents": stats.num_documents,
        "vocab_size": stats.vocab_size,
        "avg_tokens_per_doc": stats.avg_tokens_per_doc,
        "split": stats.split,
        "file_path": str(stats.file_path),
        "compression_ratio": stats.compression_ratio,
    }
    
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)


def prepare_dataset(config: PrepareConfig) -> tuple[DatasetStats, DatasetStats | None]:
    """Prepare a dataset by downloading, tokenizing, and saving.
    
    Returns:
        Tuple of (train_stats, val_stats) where val_stats may be None
    """
    # Get tokenizer
    tokenizer = get_tokenizer(config.tokenizer)
    print(f"Using tokenizer: {config.tokenizer.name} (vocab_size={tokenizer.vocab_size:,})")
    
    # Load dataset
    documents = load_dataset(config.dataset)
    print(f"Loaded {len(documents):,} documents")
    
    # Split into train/val if needed
    val_docs: list[str] | None = None
    if config.validation_split > 0:
        import random
        random.seed(config.dataset.seed)
        random.shuffle(documents)
        split_idx = int(len(documents) * (1 - config.validation_split))
        val_docs = documents[split_idx:]
        documents = documents[:split_idx]
        print(f"Split: {len(documents):,} train, {len(val_docs):,} validation")
    
    # Tokenize train set
    train_tokens = tokenize_documents(
        documents,
        tokenizer,
        config.batch_size,
        config.num_proc
    )
    
    # Save train set
    train_file = config.output_dir / f"{config.dataset.name}_train.bin"
    num_train_tokens, train_size_mb = save_tokenized_data(train_tokens, train_file)
    
    train_stats = DatasetStats(
        dataset_name=config.dataset.name,
        tokenizer_name=config.tokenizer.name,
        num_tokens=num_train_tokens,
        num_documents=len(documents),
        vocab_size=tokenizer.vocab_size,
        avg_tokens_per_doc=num_train_tokens / len(documents),
        split="train",
        file_path=train_file,
    )
    
    print(f"\nTrain set saved to: {train_file}")
    print(f"  Tokens: {num_train_tokens:,}")
    print(f"  Size: {train_size_mb:.2f} MB")
    print(f"  Avg tokens/doc: {train_stats.avg_tokens_per_doc:.1f}")
    
    # Tokenize and save validation set if present
    val_stats: DatasetStats | None = None
    if val_docs:
        val_tokens = tokenize_documents(
            val_docs,
            tokenizer,
            config.batch_size,
            config.num_proc
        )
        
        val_file = config.output_dir / f"{config.dataset.name}_val.bin"
        num_val_tokens, val_size_mb = save_tokenized_data(val_tokens, val_file)
        
        val_stats = DatasetStats(
            dataset_name=config.dataset.name,
            tokenizer_name=config.tokenizer.name,
            num_tokens=num_val_tokens,
            num_documents=len(val_docs),
            vocab_size=tokenizer.vocab_size,
            avg_tokens_per_doc=num_val_tokens / len(val_docs),
            split="validation",
            file_path=val_file,
        )
        
        print(f"\nValidation set saved to: {val_file}")
        print(f"  Tokens: {num_val_tokens:,}")
        print(f"  Size: {val_size_mb:.2f} MB")
        print(f"  Avg tokens/doc: {val_stats.avg_tokens_per_doc:.1f}")
    
    # Save metadata
    metadata_file = config.output_dir / f"{config.dataset.name}_metadata.json"
    save_metadata(train_stats, metadata_file)
    print(f"\nMetadata saved to: {metadata_file}")
    
    return train_stats, val_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for prepare.py."""
    parser = argparse.ArgumentParser(
        description="Prepare datasets for autoconstitution training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare OpenWebText dataset
  python prepare.py --dataset openwebtext
  
  # Prepare FineWeb with 1M samples
  python prepare.py --dataset fineweb --sample_size 1000000
  
  # Prepare custom dataset from directory
  python prepare.py --dataset custom --data_dir /path/to/text/files
  
  # Use different tokenizer
  python prepare.py --dataset openwebtext --tokenizer gpt2
  
  # Specify output directory
  python prepare.py --dataset openwebtext --output_dir ./data
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["openwebtext", "fineweb", "tinyshakespeare", "custom"],
        help="Dataset to prepare"
    )
    
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Path to custom dataset directory (required for custom dataset)"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to use (None for all)"
    )
    
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer to use (gpt2, character, or hf:model_name)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data"),
        help="Output directory for prepared data"
    )
    
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Number of processes for parallel tokenization"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for tokenization"
    )
    
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.01,
        help="Fraction of data to use for validation"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for shuffling"
    )
    
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column name containing text (for JSON/JSONL files)"
    )
    
    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if args.dataset == "custom" and not args.data_dir:
        parser.error("--data_dir is required for custom dataset")
    
    # Create config
    dataset_config = DatasetConfig(
        name=args.dataset,
        url=str(args.data_dir) if args.data_dir else None,
        sample_size=args.sample_size,
        seed=args.seed,
        text_column=args.text_column,
    )
    
    tokenizer_config = TokenizerConfig(
        name=args.tokenizer,
    )
    
    prepare_config = PrepareConfig(
        dataset=dataset_config,
        tokenizer=tokenizer_config,
        output_dir=args.output_dir,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
    )
    
    # Run preparation
    print("=" * 60)
    print("autoconstitution Data Preparation")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    try:
        train_stats, val_stats = prepare_dataset(prepare_config)
        
        print("\n" + "=" * 60)
        print("Preparation Complete!")
        print("=" * 60)
        print(f"\nTo use this dataset in training:")
        print(f"  data_dir = '{args.output_dir}'")
        print(f"  dataset = '{args.dataset}'")
        
        return 0
        
    except Exception as e:
        print(f"\nError during preparation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
