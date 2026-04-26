"""
Minimal LLM Training Script for autoconstitution
Based on Karpathy's nanoGPT pattern
Optimized for M4 (CPU-only) training
"""

import math
import os
import pickle
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
from torch.nn import functional as F

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Training and model configuration."""
    # Model architecture
    block_size: int = 1024          # Maximum sequence length
    vocab_size: int = 50304         # GPT-2 vocab size (50257 rounded up for efficiency)
    n_layer: int = 6                # Number of transformer layers
    n_head: int = 6                 # Number of attention heads
    n_embd: int = 384               # Embedding dimension
    dropout: float = 0.2            # Dropout rate
    bias: bool = False              # Use bias in Linear and LayerNorm layers
    
    # Training
    batch_size: int = 12            # Micro batch size
    max_iters: int = 5000           # Total training iterations
    learning_rate: float = 6e-4     # Peak learning rate
    weight_decay: float = 0.1       # Weight decay coefficient
    beta1: float = 0.9              # Adam beta1
    beta2: float = 0.95             # Adam beta2
    grad_clip: float = 1.0          # Gradient clipping threshold
    
    # Learning rate schedule
    warmup_iters: int = 100         # Warmup iterations
    lr_decay_iters: int = 5000      # Total iterations for LR decay
    min_lr: float = 6e-5            # Minimum learning rate
    
    # Evaluation and checkpointing
    eval_interval: int = 100        # Evaluation interval
    eval_iters: int = 100           # Iterations for loss estimation
    log_interval: int = 10          # Logging interval
    
    # Data
    dataset: str = "openwebtext"    # Dataset name
    
    # System
    device: str = "cpu"             # Device (cpu/cuda/mps)
    dtype: str = "float32"          # Data type (float32/bfloat16/float16)
    compile: bool = False           # Compile model with torch.compile
    
    # Reproducibility
    seed: int = 1337


# =============================================================================
# Model Components
# =============================================================================

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    
    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Flash attention (not available on CPU, use manual implementation)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # Manual causal mask
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # Calculate query, key, values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Causal self-attention
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # Manual attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block."""
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model."""
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.transformer['wte'].weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
    
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Forward pass
        tok_emb = self.transformer['wte'](idx)
        pos_emb = self.transformer['wpe'](pos)
        x = self.transformer['drop'](tok_emb + pos_emb)
        for block in self.transformer['h']:
            x = block(x)
        x = self.transformer['ln_f'](x)
        
        if targets is not None:
            # Training: compute loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference: only compute last token
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
    
    def crop_block_size(self, block_size: int) -> None:
        """Reduce block size for efficiency."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer['wpe'].weight = nn.Parameter(self.transformer['wpe'].weight[:block_size])
        for block in self.transformer['h']:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
    
    @torch.no_grad()
    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer['wpe'].weight.numel()
        return n_params


# =============================================================================
# Data Loading
# =============================================================================

class DataLoader:
    """Simple data loader for tokenized datasets."""
    
    def __init__(self, data_dir: str, batch_size: int, block_size: int) -> None:
        self.batch_size = batch_size
        self.block_size = block_size
        
        # Load data
        train_path = os.path.join(data_dir, 'train.bin')
        val_path = os.path.join(data_dir, 'val.bin')
        
        if os.path.exists(train_path):
            self.train_data = self._load_tokens(train_path)
            self.val_data = self._load_tokens(val_path) if os.path.exists(val_path) else self.train_data
        else:
            # Create dummy data for testing
            print(f"Warning: No data found at {data_dir}, using dummy data")
            self.train_data = torch.randint(0, 50257, (1000000,), dtype=torch.long)
            self.val_data = torch.randint(0, 50257, (100000,), dtype=torch.long)
        
        self.reset()
    
    def _load_tokens(self, path: str) -> torch.Tensor:
        """Load tokenized data from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return torch.tensor(data, dtype=torch.long)
    
    def reset(self) -> None:
        """Reset to beginning of dataset."""
        self.current_position = 0
    
    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data."""
        data = self.train_data if split == 'train' else self.val_data
        
        # Sample random positions
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        
        return x, y


# =============================================================================
# Training Utilities
# =============================================================================

def get_lr(it: int, config: Config) -> float:
    """Learning rate schedule with warmup and cosine decay."""
    # 1) Linear warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    
    # 2) Minimum learning rate after decay
    if it > config.lr_decay_iters:
        return config.min_lr
    
    # 3) Cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(
    model: GPT, 
    data_loader: DataLoader, 
    config: Config
) -> Dict[str, float]:
    """Estimate loss on train/val sets."""
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = data_loader.get_batch(split)
            X, Y = X.to(config.device), Y.to(config.device)
            with torch.autocast(device_type=config.device, dtype=torch.float32):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    
    model.train()
    return out


def configure_optimizers(model: GPT, config: Config) -> torch.optim.Optimizer:
    """Configure AdamW optimizer with weight decay."""
    # Separate parameters that should/shouldn't have weight decay
    decay = set()
    no_decay = set()
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, (nn.Linear, nn.Conv1d)):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, nn.Embedding):
                no_decay.add(fpn)
    
    # Validate
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    
    assert len(inter_params) == 0, f"Parameters in both decay and no_decay: {inter_params}"
    assert len(param_dict.keys() - union_params) == 0, f"Parameters not in either set: {param_dict.keys() - union_params}"
    
    # Create optimizer groups
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}
    ]
    
    return torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(config.beta1, config.beta2))


# =============================================================================
# Main Training Loop
# =============================================================================

def main() -> None:
    """Main training function."""
    # Setup
    config = Config()
    
    # Set seed
    torch.manual_seed(config.seed)
    
    # Device setup
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config.device = 'cpu'
    
    print(f"Using device: {config.device}")
    
    # Data loader
    data_dir = os.path.join('data', config.dataset)
    data_loader = DataLoader(data_dir, config.batch_size, config.block_size)
    
    # Model
    model = GPT(config)
    model = model.to(config.device)
    
    print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")
    
    # Optimizer
    optimizer = configure_optimizers(model, config)
    
    # Compile model (optional, may not work on all platforms)
    if config.compile:
        try:
            print("Compiling model...")
            model = torch.compile(model)
        except Exception as e:
            print(f"Model compilation failed: {e}")
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for iter_num in range(config.max_iters):
        # Determine learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, data_loader, config)
            print(f"Step {iter_num:5d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f} | lr: {lr:.2e}")
        
        # Get batch
        X, Y = data_loader.get_batch('train')
        X, Y = X.to(config.device), Y.to(config.device)
        
        # Forward pass
        logits, loss = model(X, Y)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Update
        optimizer.step()
        
        # Logging
        if iter_num % config.log_interval == 0:
            dt = time.time() - start_time
            lossf = loss.item()
            print(f"Step {iter_num:5d} | loss: {lossf:.4f} | dt: {dt*1000:.2f}ms")
            start_time = time.time()
    
    # Final evaluation
    losses = estimate_loss(model, data_loader, config)
    print(f"Final | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")
    
    # Save checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
        'iter_num': config.max_iters,
    }
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(checkpoint, 'checkpoints/model.pt')
    print("Checkpoint saved to checkpoints/model.pt")
    
    # Generate sample
    print("\nGenerating sample text...")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
    print(f"Generated: {generated[0].tolist()}")


if __name__ == "__main__":
    main()
