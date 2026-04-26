"""
Example usage of the val_bpb metric for autoconstitution.

This example demonstrates how to use the BitsPerByteMetric (val_bpb)
for evaluating language models in a vocabulary-size independent manner.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterator

# Import the val_bpb module
from .val_bpb import (
    BitsPerByteMetric,
    BPBConfig,
    TokenBytesLookup,
    compute_bpb,
    nats_to_bits,
)


# =============================================================================
# Example 1: Basic Usage with Custom Tokenizer and Model
# =============================================================================

class SimpleTokenizer:
    """Example tokenizer implementation."""
    
    def __init__(self, vocab_size: int = 1000) -> None:
        self._vocab_size = vocab_size
    
    def encode(self, text: str | list[str], prepend: int | str | None = None, num_threads: int = 8) -> list[int] | list[list[int]]:
        if isinstance(text, str):
            return [ord(c) % self._vocab_size for c in text]
        return [[ord(c) % self._vocab_size for c in t] for t in text]
    
    def decode(self, ids: list[int]) -> str:
        return "".join(chr(97 + (i % 26)) for i in ids)
    
    def get_vocab_size(self) -> int:
        return self._vocab_size


class SimpleLanguageModel(nn.Module):
    """Example language model with required interface."""
    
    def __init__(self, vocab_size: int, d_model: int = 128) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: Tensor, y: Tensor | None = None, reduction: str = "mean") -> Tensor:
        """
        Forward pass supporting BPB evaluation.
        
        Args:
            x: Input token IDs of shape (batch_size, seq_len)
            y: Target token IDs of shape (batch_size, seq_len), optional
            reduction: Loss reduction mode ("mean", "sum", "none")
        
        Returns:
            Loss tensor or logits depending on inputs
        """
        # Embed and process
        embedded = self.embedding(x)
        transformed = self.transformer(embedded)
        logits = self.output(transformed)
        
        if y is None:
            return logits
        
        # Compute cross-entropy loss
        loss_fn = nn.CrossEntropyLoss(reduction=reduction)
        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = y.view(-1)
        loss = loss_fn(logits_flat, targets_flat)
        
        # Reshape for per-token losses
        if reduction == "none":
            return loss.view(x.shape)
        return loss


class SimpleDataLoader:
    """Example dataloader for BPB evaluation."""
    
    def __init__(self, batch_size: int, seq_len: int, vocab_size: int, num_batches: int = 100) -> None:
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_batches = num_batches
        self._count = 0
    
    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, int]]:
        self._count = 0
        return self
    
    def __next__(self) -> tuple[Tensor, Tensor, int]:
        if self._count >= self.num_batches:
            raise StopIteration
        
        # Generate random data
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        y = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        epoch = 1
        
        self._count += 1
        return x, y, epoch


def example_basic_usage() -> None:
    """
    Example 1: Basic usage of BitsPerByteMetric.
    """
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)
    
    # Create configuration
    config = BPBConfig(
        max_seq_len=128,
        eval_tokens=10000,
        device="cpu"  # Use CPU for this example
    )
    
    # Create BPB metric
    metric = BitsPerByteMetric(tokenizer, config)
    
    # Create model and dataloader
    model = SimpleLanguageModel(vocab_size=1000)
    dataloader = SimpleDataLoader(batch_size=4, seq_len=128, vocab_size=1000)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        bpb = metric.evaluate(model, dataloader, max_steps=10)
    
    print(f"Bits Per Byte (BPB): {bpb:.4f}")
    print(f"Interpretation: Lower is better!")
    print()


def example_custom_special_tokens() -> None:
    """
    Example 2: Using custom special tokens.
    """
    print("=" * 60)
    print("Example 2: Custom Special Tokens")
    print("=" * 60)
    
    tokenizer = SimpleTokenizer(vocab_size=1000)
    
    # Define special token IDs to exclude from BPB calculation
    special_tokens = {0, 1, 2}  # BOS, EOS, PAD
    
    config = BPBConfig(device="cpu")
    metric = BitsPerByteMetric(
        tokenizer,
        config,
        special_token_ids=special_tokens
    )
    
    print(f"Special tokens excluded: {special_tokens}")
    print(f"These tokens will have byte length 0 in BPB calculation")
    print()


def example_functional_api() -> None:
    """
    Example 3: Using the functional API.
    """
    print("=" * 60)
    print("Example 3: Functional API")
    print("=" * 60)
    
    tokenizer = SimpleTokenizer(vocab_size=1000)
    
    # Create token bytes lookup
    lookup = TokenBytesLookup(tokenizer, device="cpu")
    
    # Create model and dataloader
    model = SimpleLanguageModel(vocab_size=1000)
    dataloader = SimpleDataLoader(batch_size=4, seq_len=128, vocab_size=1000)
    
    # Evaluate using functional API
    model.eval()
    with torch.no_grad():
        bpb = compute_bpb(
            model=model,
            dataloader=dataloader,
            token_bytes=lookup,
            max_steps=10,
            device="cpu"
        )
    
    print(f"Bits Per Byte (via functional API): {bpb:.4f}")
    print()


def example_conversion_utilities() -> None:
    """
    Example 4: Using conversion utilities.
    """
    print("=" * 60)
    print("Example 4: Conversion Utilities")
    print("=" * 60)
    
    import math
    
    # Convert nats to bits
    nats = 2.0
    bits = nats_to_bits(nats)
    print(f"{nats} nats = {bits:.4f} bits")
    
    # Common conversion
    cross_entropy_nats = math.log(2)  # Cross-entropy of 1 bit in nats
    cross_entropy_bits = nats_to_bits(cross_entropy_nats)
    print(f"Cross-entropy of log(2) nats = {cross_entropy_bits:.4f} bits")
    print()


def example_comparison_explanation() -> None:
    """
    Example 5: Why BPB is vocabulary-size independent.
    """
    print("=" * 60)
    print("Example 5: Why BPB is Vocabulary-Size Independent")
    print("=" * 60)
    
    print("""
BPB (Bits Per Byte) is vocabulary-size independent because:

1. It measures compression efficiency in terms of BYTES, not TOKENS.

2. The formula is:
   
   BPB = total_nats / (log(2) * total_bytes)
   
   Where:
   - total_nats: Sum of cross-entropy losses (in natural log units)
   - total_bytes: Sum of UTF-8 byte lengths of target tokens
   - log(2): Converts nats to bits

3. Example comparison:
   
   Model A: vocab_size=1000, cross_entropy=2.0, avg_bytes_per_token=1.5
   BPB_A = 2.0 / (log(2) * 1.5) = 1.92
   
   Model B: vocab_size=50000, cross_entropy=1.5, avg_bytes_per_token=2.0
   BPB_B = 1.5 / (log(2) * 2.0) = 1.08
   
   Even though Model B has a larger vocabulary, its BPB is better (lower),
   indicating better compression efficiency.

4. This allows fair comparison across:
   - Different vocabulary sizes
   - Different tokenization schemes (BPE, WordPiece, Byte-level)
   - Different model architectures
    """)


if __name__ == "__main__":
    print("\n")
    print("#" * 60)
    print("# val_bpb Metric Usage Examples")
    print("#" * 60)
    print()
    
    example_basic_usage()
    example_custom_special_tokens()
    example_functional_api()
    example_conversion_utilities()
    example_comparison_explanation()
    
    print("#" * 60)
    print("# All examples completed!")
    print("#" * 60)
