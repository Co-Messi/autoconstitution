# Karpathy Single-Agent Autoresearch Baseline Reproduction Report

## Executive Summary

This document reproduces Andrej Karpathy's single-agent autoresearch baseline results, which serve as the comparison point that SwarmResearch must beat. The baseline demonstrates autonomous LLM training optimization through a single AI agent running experiments overnight on a single GPU.

**Key Baseline Metrics:**
- **700 experiments** conducted autonomously over **2 days**
- **~20 genuine improvements** discovered
- **11% training speedup** achieved (Time-to-GPT-2: 2.02h → 1.80h)
- **Success rate:** ~2.9% (20/700 improvements kept)
- **Throughput:** ~12 experiments/hour (single GPU)

---

## 1. Exact Experimental Setup

### 1.1 System Architecture

The autoresearch system is built on three core files with a strict separation of concerns:

```
autoresearch/
├── prepare.py      # FIXED: Data preparation, tokenizer, evaluation harness
├── train.py        # MUTABLE: Model architecture, optimizer, training loop (~630 lines)
└── program.md      # HUMAN-WRITTEN: Research strategy and agent instructions
```

#### File Responsibilities

| File | Purpose | Agent Access |
|------|---------|--------------|
| `prepare.py` | Data download, BPE tokenizer training, evaluation function `evaluate_bpb()` | **READ-ONLY** |
| `train.py` | GPT model definition, Muon + AdamW optimizer, training loop | **FULL WRITE** |
| `program.md` | Research agenda, constraints, exploration strategy | **READ-ONLY** |

### 1.2 Hardware Configuration

**Primary Test Environment:**
- **GPU:** NVIDIA H100 (80GB VRAM)
- **OS:** Linux (Ubuntu 22.04+)
- **Python:** 3.10+
- **Package Manager:** uv (for fast dependency resolution)
- **Framework:** PyTorch

**Alternative Configurations Tested:**
- RTX 6000 Ada (48GB): Required `DEVICE_BATCH_SIZE` reduction from 128 → 64
- Apple Silicon (M4 Max): Community fork using MLX framework achieved val_bpb 1.294

### 1.3 Model Configuration (Baseline)

```python
# Default nanochat architecture parameters
DEPTH = 12                    # Transformer layers
MODEL_DIM = 384              # Hidden dimension (8 * ASPECT_RATIO)
ASPECT_RATIO = 48            # Width scaling factor
NUM_HEADS = 6                # Attention heads
MAX_SEQ_LEN = 512            # Sequence length
VOCAB_SIZE = 4096            # BPE tokenizer vocabulary

# Training parameters
TOTAL_BATCH_SIZE = 2**18     # ~524K tokens per step
DEVICE_BATCH_SIZE = 128      # Per-device batch (H100)
LEARNING_RATE = 0.001        # Base learning rate
WARMUP_STEPS = 100
```

### 1.4 Experiment Loop Design

**Fixed-Time Budget Protocol:**
- Each experiment runs for exactly **5 minutes wall-clock time**
- Training is interrupted at 5-minute mark regardless of progress
- This ensures fair comparison across different architectures/configurations

**Evaluation Metric:**
- **Primary:** `val_bpb` (validation bits per byte)
- **Properties:** Vocabulary-size-independent, lower is better
- **Rationale:** Allows architectural changes (tokenizer, embeddings) without metric bias

**Decision Logic:**
```
1. Agent reads program.md + train.py + results.tsv
2. Proposes modification to train.py with reasoning
3. Runs 5-minute training experiment
4. Extracts val_bpb from output
5. IF val_bpb < baseline: git commit (keep)
   ELSE: git reset (discard)
6. Log result to results.tsv
7. REPEAT
```

### 1.5 Agent Configuration

**Agent Type:** Claude Code (Claude Opus 4.6) or GPT-4 class models

**Key program.md Instructions:**
- "NEVER STOP" - Agent must not pause for human input
- Modify only train.py
- Each experiment must run exactly 5 minutes
- Commit improvements immediately
- Handle crashes by reading last 50 lines of error log

---

## 2. Metrics Achieved

### 2.1 Primary Results

| Run | Duration | Experiments | Improvements | Start val_bpb | End val_bpb | Improvement |
|-----|----------|-------------|--------------|---------------|-------------|-------------|
| Initial overnight | ~8h | 83 | 15 | ~1.000 | 0.975 | 2.5% |
| Extended 2-day | ~48h | ~700 | ~20 | ~1.000 | ~0.970 | ~3.0% |
| Community (126 exp) | ~11h | 126 | - | 0.9979 | 0.9697 | 2.8% |
| SkyPilot (16 GPUs) | 8h | ~910 | - | 1.003 | 0.974 | 2.9% |

### 2.2 Production Impact

**Time-to-GPT-2 Benchmark:**
- **Before autoresearch:** 2.02 hours
- **After autoresearch:** 1.80 hours
- **Improvement:** 11% faster training

**Transfer Validation:**
- All 20 improvements discovered on depth=12 model transferred successfully to depth=24 model
- Improvements were additive (stacked without interference)
- Demonstrated generalization across model scales

### 2.3 Success Rate Analysis

| Metric | Value |
|--------|-------|
| Overall success rate | ~2.9% (20/700) |
| Early-stage success rate | ~18% (15/83) |
| Late-stage success rate | ~10% (as search space depletes) |
| Crash/rejection rate | ~97% |

### 2.4 Community Reproductions

**Tobi Lütke (Shopify CEO):**
- 37 experiments overnight
- 19% performance gain on internal query-expansion model
- 0.8B parameter model outperformed previous 1.6B hand-tuned baseline

**RTX 6000 Ada Reproduction:**
- 35 experiments in ~3 hours
- Baseline val_bpb: 1.234593
- Agent successfully adapted batch size for 48GB VRAM

**Apple Silicon (MLX):**
- Overnight run on M4 Max
- Achieved val_bpb 1.294 from 2.667 baseline
- Discovered hardware-specific optimizations

---

## 3. Time Taken

### 3.1 Experiment Throughput

| Configuration | Experiments/Hour | Daily Throughput | Notes |
|---------------|------------------|------------------|-------|
| Single H100 | ~12 | ~100 overnight | Default setup |
| 16 GPU cluster | ~114 | ~910 in 8h | Parallel search |
| RTX 6000 Ada | ~12 | ~100 overnight | Similar to H100 |
| M4 Max (MLX) | ~12 | ~100 overnight | Apple Silicon |

### 3.2 Timeline Breakdown (2-Day Run)

```
Hour 0-8:   Initial exploration, high success rate (~18%)
Hour 8-24:  Architecture discovery phase, testing aspect ratios
Hour 24-36: Fine-tuning phase, optimizer parameter sweeps
Hour 36-48: Diminishing returns, combinatorial searches
```

### 3.3 Phase Analysis (SkyPilot 16-GPU Run)

| Phase | Experiments | Focus Area | val_bpb Change |
|-------|-------------|------------|----------------|
| Phase 1 (0-200) | ~200 | Hyperparameters | 1.003 → 0.981 (Δ=0.022) |
| Phase 2 (200-420) | ~220 | Architecture (width) | 0.981 → 0.977 (Δ=0.004) |
| Phase 3 (420-560) | ~140 | Fine-tuning | 0.977 → 0.975 (Δ=0.002) |
| Phase 4 (560-700) | ~140 | Optimizer tuning | 0.975 → 0.974 (Δ=0.001) |
| Phase 5 (700-910) | ~210 | Combinatorial | 0.974 → ~0.974 (Δ<0.0001) |

---

## 4. Number of Experiments

### 4.1 Experiment Distribution

**Karpathy's Official Runs:**
- **Run 1:** 83 experiments (single overnight) → 15 kept
- **Run 2:** ~700 experiments (2 days) → ~20 kept
- **Production (8x H100):** 276 experiments → 29 kept

**Community Runs:**
- **161 experiments (RTX 3060):** 23 kept (14% hit rate)
  - Session 1: 5 experiments, 3 kept (60%)
  - Session 2: 42 experiments, 8 kept (19%)
  - Session 3: 114 experiments, 12 kept (10.5%)

### 4.2 Experiment Categories

Based on analysis of kept improvements:

| Category | Estimated Count | % of Improvements |
|----------|-----------------|-------------------|
| Architecture changes | ~4 | 20% |
| Optimizer tuning | ~6 | 30% |
| Hyperparameter adjustments | ~5 | 25% |
| Regularization additions | ~3 | 15% |
| Initialization fixes | ~2 | 10% |

---

## 5. Types of Improvements Found

### 5.1 Specific Improvements Discovered

#### Architecture Modifications

1. **QK-Norm Fix**
   - **Issue:** Missing scalar multiplier in QK-Norm implementation
   - **Effect:** Attention was too diffuse across heads
   - **Fix:** Added learnable scalar multiplier
   - **Impact:** Sharper attention patterns

2. **Aspect Ratio Optimization**
   - **Discovery:** AR=96 (model_dim=768) outperformed default AR=48
   - **Trade-off:** Fewer steps in 5 minutes but better per-step improvement
   - **Result:** ~1,060 steps vs ~1,450 steps, but better final val_bpb

3. **Attention Pattern Tuning**
   - **Finding:** Banded attention window patterns improved efficiency
   - **Implementation:** Alternating sliding + local attention ("SL" pattern)

#### Optimizer Improvements

4. **Muon Beta2 Adjustment**
   - **Change:** `muon_beta2` from 0.95 → 0.98
   - **Mechanism:** Smoother gradient normalization adaptation
   - **Impact:** Largest late-stage improvement (~0.001 val_bpb)

5. **AdamW Beta Tuning**
   - **Optimal:** `ADAM_BETAS = (0.70, 0.95)`
   - **Effect:** Better momentum scheduling for non-matrix parameters

6. **Newton-Schulz Steps**
   - **Optimization:** Tuned Muon optimizer's orthogonalization steps
   - **Range tested:** Various step counts for matrix orthogonalization

#### Learning Rate & Scheduling

7. **Warmdown Ratio**
   - **Optimal:** `WARMDOWN_RATIO = 0.6`
   - **Effect:** Smoother LR decay in final training phase

8. **Final LR Fraction**
   - **Optimal:** `FINAL_LR_FRAC = 0.05`
   - **Effect:** Better convergence at end of training

9. **Learning Rate Specialization**
   - **Matrix LR:** 0.05 (Muon for weight matrices)
   - **Embedding LR:** 0.6 (AdamW for token embeddings)
   - **Scalar LR:** 0.5 (AdamW for residual mixing scalars)

#### Regularization

10. **Value Embedding Regularization**
    - **Addition:** Applied regularization to value embeddings
    - **Effect:** Reduced overfitting, better generalization

11. **Weight Decay Scheduling**
    - **Optimal:** `WEIGHT_DECAY = 0.08`
    - **Effect:** Better regularization throughout training

#### Initialization

12. **Initialization Corrections**
    - **Finding:** Default initialization suboptimal for certain layers
    - **Fix:** Layer-specific initialization adjustments

### 5.2 Improvement Characteristics

| Characteristic | Observation |
|----------------|-------------|
| Additivity | All 20 improvements stacked without interference |
| Transferability | Improvements transferred from depth=12 to depth=24 |
| Hardware-specificity | Optimal configs vary by GPU (H100 vs M4 Max) |
| Discovery pattern | Early: large gains; Late: diminishing returns |

### 5.3 What the Agent Did NOT Find

- Novel attention mechanisms (e.g., no new attention variants invented)
- Alternative architectures (e.g., no Mamba, RWKV, or hybrid proposals)
- Alternative optimizers (e.g., no Lion, Adafactor, or custom optimizer design)
- Data pipeline modifications (prepare.py was read-only)

---

## 6. Baseline Comparison Framework

### 6.1 Metrics SwarmResearch Must Beat

| Metric | Karpathy Baseline | SwarmResearch Target |
|--------|-------------------|---------------------|
| Experiments/24h (1 GPU) | ~100 | >100 |
| Success rate | ~2.9% | >2.9% |
| Time-to-GPT-2 improvement | 11% | >11% |
| val_bpb reduction | ~3% | >3% |
| Improvements found (2 days) | ~20 | >20 |
| Transfer success rate | 100% (20/20) | 100% |

### 6.2 Key Differentiators to Test

1. **Multi-agent coordination** vs single-agent sequential search
2. **Parallel exploration** vs greedy hill-climbing
3. **Knowledge sharing** between agents vs isolated experiments
4. **Diverse search strategies** vs single strategy

---

## 7. Reproduction Instructions

### 7.1 Setup

```bash
# Clone repository
git clone https://github.com/karpathy/autoresearch
cd autoresearch

# Install dependencies
uv sync

# Prepare data (one-time)
uv run prepare.py

# Verify baseline
uv run train.py  # Expect val_bpb ~1.0
```

### 7.2 Running Autoresearch

```bash
# Start autonomous loop
python autorun.py

# Monitor progress
tail -f results.tsv
watch -n 60 'git log --oneline -10'
```

### 7.3 Expected Output

```
# results.tsv format
experiment_id	timestamp	val_bpb	peak_memory_mb	commit_hash	notes
1	2026-03-07T00:00:00	1.0000	45000	abc123	baseline
2	2026-03-07T00:05:00	0.9985	45200	def456	increased learning rate
...
```

---

## 8. Conclusion

Karpathy's autoresearch baseline establishes a clear benchmark for autonomous LLM training optimization:

- **Single agent** on **single GPU**
- **700 experiments** in **2 days**
- **~20 improvements** discovered
- **11% training speedup** achieved
- **100% transfer success** to larger models

This baseline represents the state-of-the-art in single-agent autonomous research. SwarmResearch must demonstrate superior performance through multi-agent coordination, parallel exploration, and knowledge sharing to establish a new benchmark.

---

## References

1. Karpathy, A. (2026). autoresearch. GitHub: https://github.com/karpathy/autoresearch
2. Karpathy, A. (2026). nanochat. GitHub: https://github.com/karpathy/nanochat
3. SkyPilot Blog (2026). Scaling Karpathy's Autoresearch
4. Community reproductions and discussions on GitHub

---

*Report generated: March 2026*
*Baseline version: autoresearch v1.0*
