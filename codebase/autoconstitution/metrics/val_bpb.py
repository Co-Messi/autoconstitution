"""
Validation Bits Per Byte (val_bpb) metric for autoconstitution.

This module implements the vocabulary-size independent evaluation metric
from Karpathy's autoresearch framework. BPB (bits per byte) provides a
fair comparison across different architectures and tokenization schemes.

Key properties:
- Vocabulary-size independent: results comparable across different vocab sizes
- Architecture independent: fair comparison across model architectures
- Lower is better (measures compression efficiency)
- Based on cross-entropy loss normalized by UTF-8 byte length

Reference:
    Karpathy, A. (2026). autoresearch: AI agents running research on 
    single-GPU nanochat training automatically.
    https://github.com/karpathy/autoresearch
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------------------------

@runtime_checkable
class TokenizerProtocol(Protocol):
    """Protocol for tokenizer interface required by BPB evaluation."""
    
    def encode(self, text: str | list[str], prepend: int | str | None = None, num_threads: int = 8) -> list[int] | list[list[int]]:
        """Encode text to token IDs."""
        ...
    
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        ...
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        ...


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for model interface required by BPB evaluation."""
    
    def __call__(self, x: Tensor, y: Tensor | None = None, reduction: str = "mean") -> Tensor:
        """
        Forward pass with optional target and reduction mode.
        
        Args:
            x: Input token IDs of shape (batch_size, seq_len)
            y: Target token IDs of shape (batch_size, seq_len), optional
            reduction: Loss reduction mode ("mean", "sum", "none")
        
        Returns:
            Loss tensor or logits depending on inputs
        """
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BPBConfig:
    """Configuration for BPB evaluation.
    
    Attributes:
        max_seq_len: Maximum sequence length for evaluation
        eval_tokens: Total number of tokens to evaluate
        device: Device for computation ("cuda", "cpu", etc.)
    """
    max_seq_len: int = 2048
    eval_tokens: int = 40 * 524288  # ~21M tokens
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Token Bytes Lookup
# ---------------------------------------------------------------------------

class TokenBytesLookup:
    """
    Lookup table for UTF-8 byte lengths of tokens.
    
    This class builds and caches the byte length for each token in the
    vocabulary, which is essential for computing BPB. Special tokens
    (e.g., BOS, EOS, PAD) are assigned a byte length of 0 to exclude
    them from the BPB calculation.
    """
    
    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        special_token_ids: set[int] | None = None,
        device: str = "cpu"
    ) -> None:
        """
        Initialize token bytes lookup table.
        
        Args:
            tokenizer: Tokenizer with encode/decode methods
            special_token_ids: Set of special token IDs to exclude (byte length = 0)
            device: Device for the lookup tensor
        """
        self.vocab_size = tokenizer.get_vocab_size()
        self.device = device
        self.special_token_ids = special_token_ids or set()
        
        # Build byte length lookup table
        self._token_bytes = self._build_lookup(tokenizer)
    
    def _build_lookup(self, tokenizer: TokenizerProtocol) -> Tensor:
        """
        Build the token bytes lookup tensor.
        
        For each token ID, decode it to string and measure UTF-8 byte length.
        Special tokens get byte length 0.
        
        Returns:
            Tensor of shape (vocab_size,) with byte lengths
        """
        token_bytes_list: list[int] = []
        
        for token_id in range(self.vocab_size):
            if token_id in self.special_token_ids:
                token_bytes_list.append(0)
            else:
                try:
                    token_str = tokenizer.decode([token_id])
                    token_bytes_list.append(len(token_str.encode("utf-8")))
                except (UnicodeDecodeError, ValueError):
                    # Handle edge cases where token can't be decoded
                    token_bytes_list.append(0)
        
        return torch.tensor(token_bytes_list, dtype=torch.int32, device=self.device)
    
    def __getitem__(self, token_ids: Tensor) -> Tensor:
        """
        Get byte lengths for given token IDs.
        
        Args:
            token_ids: Tensor of token IDs
        
        Returns:
            Tensor of byte lengths
        """
        return self._token_bytes[token_ids]
    
    def to(self, device: str) -> TokenBytesLookup:
        """Move lookup table to specified device."""
        self._token_bytes = self._token_bytes.to(device)
        self.device = device
        return self


# ---------------------------------------------------------------------------
# DataLoader Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class DataLoaderProtocol(Protocol):
    """Protocol for data loader that yields batches for BPB evaluation."""
    
    def __iter__(self) -> DataLoaderProtocol:
        """Return self as iterator."""
        ...
    
    def __next__(self) -> tuple[Tensor, Tensor, int]:
        """
        Get next batch.
        
        Returns:
            Tuple of (inputs, targets, epoch_number)
        """
        ...


# ---------------------------------------------------------------------------
# BPB Metric Implementation
# ---------------------------------------------------------------------------

class BitsPerByteMetric:
    """
    Bits Per Byte (BPB) evaluation metric.
    
    BPB measures the average number of bits needed to encode each byte
    of text, computed as:
    
        BPB = total_nats / (log(2) * total_bytes)
    
    Where:
        - total_nats: Sum of cross-entropy losses (in nats, i.e., natural log)
        - total_bytes: Sum of UTF-8 byte lengths of target tokens
        - log(2): Conversion factor from nats to bits
    
    Special tokens (with byte length 0) are excluded from both sums.
    
    Properties:
        - Lower is better (indicates better compression/model quality)
        - Vocabulary-size independent
        - Architecture independent
        - Comparable across different tokenization schemes
    
    Example:
        >>> metric = BitsPerByteMetric(tokenizer, config)
        >>> bpb = metric.evaluate(model, dataloader)
        >>> print(f"BPB: {bpb:.4f}")
    """
    
    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        config: BPBConfig | None = None,
        special_token_ids: set[int] | None = None
    ) -> None:
        """
        Initialize BPB metric.
        
        Args:
            tokenizer: Tokenizer for encoding/decoding
            config: BPB evaluation configuration
            special_token_ids: Set of special token IDs to exclude
        """
        self.tokenizer = tokenizer
        self.config = config or BPBConfig()
        self.special_token_ids = special_token_ids or set()
        
        # Initialize token bytes lookup
        self._token_bytes = TokenBytesLookup(
            tokenizer=tokenizer,
            special_token_ids=special_token_ids,
            device=self.config.device
        )
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module | ModelProtocol,
        dataloader: DataLoaderProtocol,
        max_steps: int | None = None
    ) -> float:
        """
        Compute BPB over validation data.
        
        Args:
            model: Model with forward(x, y, reduction) interface
            dataloader: DataLoader yielding (inputs, targets, epoch) tuples
            max_steps: Maximum number of steps to evaluate (None = use config)
        
        Returns:
            Bits per byte value (lower is better)
        
        Raises:
            ValueError: If no valid tokens are evaluated
            RuntimeError: If model evaluation fails
        """
        # Determine number of evaluation steps
        if max_steps is None:
            batch_size = self._infer_batch_size(dataloader)
            max_steps = self.config.eval_tokens // (batch_size * self.config.max_seq_len)
        
        # Ensure token bytes are on correct device
        token_bytes = self._token_bytes.to(self.config.device)
        
        total_nats: float = 0.0
        total_bytes: int = 0
        steps_completed: int = 0
        
        for step in range(max_steps):
            try:
                x, y, _ = next(dataloader)
            except StopIteration:
                break
            
            # Move data to device
            x = x.to(self.config.device)
            y = y.to(self.config.device)
            
            # Compute per-token losses (no reduction)
            loss_flat = model(x, y, reduction='none').view(-1)
            y_flat = y.view(-1)
            
            # Get byte lengths for each target token
            nbytes = token_bytes[y_flat]
            
            # Mask out special tokens (byte length = 0)
            mask = nbytes > 0
            
            # Accumulate weighted losses and byte counts
            total_nats += (loss_flat * mask).sum().item()
            total_bytes += nbytes.sum().item()
            
            steps_completed += 1
        
        if total_bytes == 0:
            raise ValueError(
                "No valid tokens evaluated. Check that dataloader produces "
                "valid targets and special_token_ids are correctly set."
            )
        
        # Convert nats/byte to bits/byte
        bpb = total_nats / (math.log(2) * total_bytes)
        
        return bpb
    
    def _infer_batch_size(self, dataloader: DataLoaderProtocol) -> int:
        """
        Infer batch size from first batch.
        
        Args:
            dataloader: DataLoader to sample from
        
        Returns:
            Batch size
        """
        x, _, _ = next(dataloader)
        return x.size(0)
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.tokenizer.get_vocab_size()


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_bpb(
    model: nn.Module | ModelProtocol,
    dataloader: DataLoaderProtocol,
    token_bytes: TokenBytesLookup | Tensor,
    max_steps: int,
    device: str = "cuda"
) -> float:
    """
    Functional API to compute bits per byte.
    
    This is a lower-level function for computing BPB when you already
    have the token bytes lookup prepared.
    
    Args:
        model: Model with forward(x, y, reduction) interface
        dataloader: DataLoader yielding (inputs, targets, epoch) tuples
        token_bytes: Token bytes lookup (TokenBytesLookup or Tensor)
        max_steps: Number of steps to evaluate
        device: Device for computation
    
    Returns:
        Bits per byte value (lower is better)
    
    Example:
        >>> token_bytes = TokenBytesLookup(tokenizer, device="cuda")
        >>> bpb = compute_bpb(model, dataloader, token_bytes, max_steps=100)
    """
    if isinstance(token_bytes, TokenBytesLookup):
        token_bytes = token_bytes.to(device)._token_bytes
    else:
        token_bytes = token_bytes.to(device)
    
    total_nats: float = 0.0
    total_bytes: int = 0
    
    for _ in range(max_steps):
        try:
            x, y, _ = next(dataloader)
        except StopIteration:
            break
        
        x = x.to(device)
        y = y.to(device)
        
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    
    if total_bytes == 0:
        raise ValueError("No valid tokens evaluated")
    
    return total_nats / (math.log(2) * total_bytes)


def create_token_bytes_lookup(
    tokenizer: TokenizerProtocol,
    special_token_ids: set[int] | None = None,
    device: str = "cpu"
) -> TokenBytesLookup:
    """
    Factory function to create a TokenBytesLookup.
    
    Args:
        tokenizer: Tokenizer with encode/decode methods
        special_token_ids: Set of special token IDs to exclude
        device: Device for the lookup tensor
    
    Returns:
        TokenBytesLookup instance
    """
    return TokenBytesLookup(
        tokenizer=tokenizer,
        special_token_ids=special_token_ids,
        device=device
    )


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def nats_to_bits(nats: float) -> float:
    """Convert nats (natural log) to bits."""
    return nats / math.log(2)


def bits_to_nats(bits: float) -> float:
    """Convert bits to nats (natural log)."""
    return bits * math.log(2)


def cross_entropy_to_bpb(
    cross_entropy: float,
    avg_bytes_per_token: float
) -> float:
    """
    Convert cross-entropy loss to bits per byte.
    
    Args:
        cross_entropy: Cross-entropy loss (in nats)
        avg_bytes_per_token: Average bytes per token
    
    Returns:
        Bits per byte
    """
    return nats_to_bits(cross_entropy) / avg_bytes_per_token


# ---------------------------------------------------------------------------
# Aliases for Convenience
# ---------------------------------------------------------------------------

val_bpb = BitsPerByteMetric  # Main class alias
BPB = BitsPerByteMetric  # Short alias


# ---------------------------------------------------------------------------
# __all__ Definition
# ---------------------------------------------------------------------------

__all__ = [
    # Main classes
    "BitsPerByteMetric",
    "BPB",
    "val_bpb",
    
    # Configuration
    "BPBConfig",
    
    # Lookup
    "TokenBytesLookup",
    
    # Functional API
    "compute_bpb",
    "create_token_bytes_lookup",
    
    # Utilities
    "nats_to_bits",
    "bits_to_nats",
    "cross_entropy_to_bpb",
    
    # Protocols
    "TokenizerProtocol",
    "ModelProtocol",
    "DataLoaderProtocol",
]
