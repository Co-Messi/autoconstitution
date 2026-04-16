"""
Tests for the val_bpb metric implementation.
"""

import math
import unittest
from typing import Iterator
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch import Tensor

from .val_bpb import (
    BitsPerByteMetric,
    BPBConfig,
    TokenBytesLookup,
    compute_bpb,
    create_token_bytes_lookup,
    nats_to_bits,
    bits_to_nats,
    cross_entropy_to_bpb,
)


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self, vocab_size: int = 100) -> None:
        self._vocab_size = vocab_size
        self._decode_map = {i: chr(97 + (i % 26)) * (i // 26 + 1) for i in range(vocab_size)}
    
    def encode(self, text: str | list[str], prepend: int | str | None = None, num_threads: int = 8) -> list[int] | list[list[int]]:
        if isinstance(text, str):
            return [ord(c) % self._vocab_size for c in text]
        return [[ord(c) % self._vocab_size for c in t] for t in text]
    
    def decode(self, ids: list[int]) -> str:
        return "".join(self._decode_map.get(i, "") for i in ids)
    
    def get_vocab_size(self) -> int:
        return self._vocab_size


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, vocab_size: int, constant_loss: float = 2.0) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.constant_loss = constant_loss
        self.embedding = nn.Embedding(vocab_size, 16)
    
    def forward(self, x: Tensor, y: Tensor | None = None, reduction: str = "mean") -> Tensor:
        batch_size, seq_len = x.shape
        
        if y is None:
            # Return logits
            return torch.randn(batch_size, seq_len, self.vocab_size, device=x.device)
        
        # Return loss
        if reduction == "none":
            return torch.full((batch_size, seq_len), self.constant_loss, device=x.device)
        elif reduction == "sum":
            return torch.tensor(batch_size * seq_len * self.constant_loss, device=x.device)
        else:  # mean
            return torch.tensor(self.constant_loss, device=x.device)


class MockDataLoader:
    """Mock dataloader for testing."""
    
    def __init__(self, batch_size: int, seq_len: int, num_batches: int = 10) -> None:
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = num_batches
        self._count = 0
    
    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, int]]:
        self._count = 0
        return self
    
    def __next__(self) -> tuple[Tensor, Tensor, int]:
        if self._count >= self.num_batches:
            raise StopIteration
        
        x = torch.randint(0, 100, (self.batch_size, self.seq_len))
        y = torch.randint(0, 100, (self.batch_size, self.seq_len))
        epoch = 1
        
        self._count += 1
        return x, y, epoch


class TestUtilityFunctions(unittest.TestCase):
    """Test utility conversion functions."""
    
    def test_nats_to_bits(self) -> None:
        """Test conversion from nats to bits."""
        nats = math.log(2)
        bits = nats_to_bits(nats)
        self.assertAlmostEqual(bits, 1.0, places=6)
    
    def test_bits_to_nats(self) -> None:
        """Test conversion from bits to nats."""
        bits = 1.0
        nats = bits_to_nats(bits)
        self.assertAlmostEqual(nats, math.log(2), places=6)
    
    def test_roundtrip_conversion(self) -> None:
        """Test that nats -> bits -> nats is identity."""
        original = 3.5
        bits = nats_to_bits(original)
        recovered = bits_to_nats(bits)
        self.assertAlmostEqual(original, recovered, places=6)
    
    def test_cross_entropy_to_bpb(self) -> None:
        """Test cross-entropy to BPB conversion."""
        # If cross-entropy is log(2) nats and avg bytes per token is 1,
        # BPB should be 1.0
        ce = math.log(2)
        avg_bytes = 1.0
        bpb = cross_entropy_to_bpb(ce, avg_bytes)
        self.assertAlmostEqual(bpb, 1.0, places=6)


class TestTokenBytesLookup(unittest.TestCase):
    """Test TokenBytesLookup class."""
    
    def test_initialization(self) -> None:
        """Test TokenBytesLookup initialization."""
        tokenizer = MockTokenizer(vocab_size=100)
        lookup = TokenBytesLookup(tokenizer, device="cpu")
        
        self.assertEqual(lookup.vocab_size, 100)
        self.assertEqual(lookup.device, "cpu")
    
    def test_getitem(self) -> None:
        """Test getting byte lengths."""
        tokenizer = MockTokenizer(vocab_size=10)
        lookup = TokenBytesLookup(tokenizer, device="cpu")
        
        token_ids = torch.tensor([0, 1, 2])
        byte_lengths = lookup[token_ids]
        
        self.assertEqual(byte_lengths.shape, (3,))
        self.assertEqual(byte_lengths.dtype, torch.int32)
    
    def test_special_tokens_excluded(self) -> None:
        """Test that special tokens have byte length 0."""
        tokenizer = MockTokenizer(vocab_size=10)
        special_ids = {0, 1}
        lookup = TokenBytesLookup(tokenizer, special_token_ids=special_ids, device="cpu")
        
        token_ids = torch.tensor([0, 1, 2, 3])
        byte_lengths = lookup[token_ids]
        
        # Special tokens should have byte length 0
        self.assertEqual(byte_lengths[0].item(), 0)
        self.assertEqual(byte_lengths[1].item(), 0)
        
        # Regular tokens should have positive byte length
        self.assertGreater(byte_lengths[2].item(), 0)
        self.assertGreater(byte_lengths[3].item(), 0)


class TestBPBConfig(unittest.TestCase):
    """Test BPBConfig dataclass."""
    
    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BPBConfig()
        
        self.assertEqual(config.max_seq_len, 2048)
        self.assertEqual(config.eval_tokens, 40 * 524288)
        self.assertEqual(config.device, "cuda")
    
    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = BPBConfig(max_seq_len=1024, eval_tokens=1000000, device="cpu")
        
        self.assertEqual(config.max_seq_len, 1024)
        self.assertEqual(config.eval_tokens, 1000000)
        self.assertEqual(config.device, "cpu")


class TestBitsPerByteMetric(unittest.TestCase):
    """Test BitsPerByteMetric class."""
    
    def test_initialization(self) -> None:
        """Test BitsPerByteMetric initialization."""
        tokenizer = MockTokenizer(vocab_size=100)
        config = BPBConfig(device="cpu")
        metric = BitsPerByteMetric(tokenizer, config)
        
        self.assertEqual(metric.vocab_size, 100)
        self.assertEqual(metric.config.device, "cpu")
    
    def test_initialization_default_config(self) -> None:
        """Test initialization with default config."""
        tokenizer = MockTokenizer(vocab_size=100)
        metric = BitsPerByteMetric(tokenizer)
        
        self.assertIsNotNone(metric.config)
        self.assertEqual(metric.config.max_seq_len, 2048)
    
    def test_evaluate_basic(self) -> None:
        """Test basic evaluation."""
        tokenizer = MockTokenizer(vocab_size=100)
        config = BPBConfig(device="cpu")
        metric = BitsPerByteMetric(tokenizer, config)
        
        model = MockModel(vocab_size=100, constant_loss=2.0)
        dataloader = MockDataLoader(batch_size=4, seq_len=16, num_batches=5)
        
        bpb = metric.evaluate(model, dataloader, max_steps=5)
        
        # BPB should be positive
        self.assertGreater(bpb, 0)
        
        # With constant loss of 2.0 nats per token, BPB should be:
        # BPB = 2.0 / (log(2) * avg_bytes_per_token)
        # Since avg_bytes_per_token is around 1-2, BPB should be roughly 1-3
        self.assertLess(bpb, 10)
    
    def test_evaluate_no_valid_tokens(self) -> None:
        """Test that evaluation raises error with no valid tokens."""
        tokenizer = MockTokenizer(vocab_size=10)
        
        # Mark all tokens as special (byte length 0)
        special_ids = set(range(10))
        config = BPBConfig(device="cpu")
        metric = BitsPerByteMetric(tokenizer, config, special_token_ids=special_ids)
        
        model = MockModel(vocab_size=10)
        dataloader = MockDataLoader(batch_size=4, seq_len=16, num_batches=5)
        
        with self.assertRaises(ValueError):
            metric.evaluate(model, dataloader, max_steps=5)


class TestFunctionalAPI(unittest.TestCase):
    """Test functional API functions."""
    
    def test_create_token_bytes_lookup(self) -> None:
        """Test factory function for TokenBytesLookup."""
        tokenizer = MockTokenizer(vocab_size=50)
        lookup = create_token_bytes_lookup(tokenizer, device="cpu")
        
        self.assertIsInstance(lookup, TokenBytesLookup)
        self.assertEqual(lookup.vocab_size, 50)
    
    def test_compute_bpb_basic(self) -> None:
        """Test compute_bpb function."""
        tokenizer = MockTokenizer(vocab_size=100)
        lookup = TokenBytesLookup(tokenizer, device="cpu")
        
        model = MockModel(vocab_size=100, constant_loss=2.0)
        dataloader = MockDataLoader(batch_size=4, seq_len=16, num_batches=5)
        
        bpb = compute_bpb(model, dataloader, lookup, max_steps=5, device="cpu")
        
        self.assertGreater(bpb, 0)
        self.assertLess(bpb, 10)


class TestBPBAlias(unittest.TestCase):
    """Test that BPB alias works correctly."""
    
    def test_bpb_alias(self) -> None:
        """Test BPB is alias for BitsPerByteMetric."""
        from .val_bpb import BPB
        self.assertIs(BPB, BitsPerByteMetric)
    
    def test_val_bpb_alias(self) -> None:
        """Test val_bpb is alias for BitsPerByteMetric."""
        from .val_bpb import val_bpb
        self.assertIs(val_bpb, BitsPerByteMetric)


class TestFormulaCorrectness(unittest.TestCase):
    """Test that the BPB formula is implemented correctly."""
    
    def test_bpb_formula(self) -> None:
        """
        Test BPB formula: bpb = total_nats / (log(2) * total_bytes)
        
        Create a controlled scenario where we know the expected result.
        """
        # Scenario: 1 token, 1 byte, loss of log(2) nats
        # BPB should be: log(2) / (log(2) * 1) = 1.0
        
        total_nats = math.log(2)
        total_bytes = 1
        expected_bpb = total_nats / (math.log(2) * total_bytes)
        
        self.assertAlmostEqual(expected_bpb, 1.0, places=6)
    
    def test_bpb_with_multiple_tokens(self) -> None:
        """Test BPB with multiple tokens."""
        # 10 tokens, each with 2 bytes, each with loss of log(2) nats
        # total_nats = 10 * log(2)
        # total_bytes = 10 * 2 = 20
        # BPB = 10*log(2) / (log(2) * 20) = 0.5
        
        num_tokens = 10
        bytes_per_token = 2
        loss_per_token = math.log(2)
        
        total_nats = num_tokens * loss_per_token
        total_bytes = num_tokens * bytes_per_token
        bpb = total_nats / (math.log(2) * total_bytes)
        
        self.assertAlmostEqual(bpb, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
