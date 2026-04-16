# LLM Training Efficiency Techniques Survey (2022-2026)
## Comprehensive Analysis with Discoverability Assessment for Automated Agent Experimentation

---

## Executive Summary

This report surveys major techniques for improving LLM training efficiency across four categories: Data Efficiency, Architectural Efficiency, Optimizer Efficiency, and Compute Efficiency. For each technique, we assess discoverability via automated agent experimentation and identify the signals an agent would need to detect effectiveness.

**Key Finding**: Techniques vary dramatically in discoverability. Compute efficiency methods (Flash Attention, mixed precision) offer clear, immediate signals (memory reduction, throughput gains), while architectural changes (MoE, attention variants) require longer training runs and downstream evaluation to validate. Optimizer efficiency sits in the middle with convergence speed as a primary signal.

---

## 1. DATA EFFICIENCY

### 1.1 Curriculum Learning

**Description**: Training models on progressively harder examples rather than random sampling. Inspired by human learning, easier concepts are introduced first before complex ones.

**Key Papers/Methods**:
- Difficulty-Aware Self-Training (DAST) - 2025
- Various difficulty-scoring heuristics (perplexity-based, length-based, rule-based)

**Discoverability Assessment**: **MODERATE**

| Signal | Detectability | Time to Signal |
|--------|--------------|----------------|
| Training loss trajectory | HIGH - smoother decrease | Early (first 10% of training) |
| Convergence speed | MODERATE - needs baseline comparison | Mid-training |
| Downstream performance | LOW - requires full evaluation | End of training |

**Agent Detection Requirements**:
- Ability to compute difficulty scores for training examples
- Track loss trajectory smoothness (lower variance in loss decreases)
- Compare against random-sampling baseline
- Measure examples/epoch to convergence

**Recommendation for SwarmResearch**: 
- Implement multiple difficulty metrics (perplexity, length, linguistic complexity)
- Run A/B tests with curriculum vs. random sampling
- Monitor loss variance as early signal

---

### 1.2 Data Pruning

**Description**: Removing low-quality or redundant training examples before or during training to improve efficiency.

**Key Papers/Methods**:
- Perplexity-based pruning (Marion et al., 2023)
- DSIR (Xie et al., 2023b) - data selection with importance resampling
- LESS (Xia et al., 2024) - gradient-based selection
- AgilePruner - attention-based pruning for VLMs (2026)

**Discoverability Assessment**: **HIGH**

| Signal | Detectability | Time to Signal |
|--------|--------------|----------------|
| Training speed (tokens/sec) | HIGH - immediate | Immediate |
| Memory usage reduction | HIGH - immediate | Immediate |
| Loss per token | MODERATE - compare to baseline | Early training |
| Final model quality | MODERATE - requires evaluation | End of training |

**Agent Detection Requirements**:
- Compute data quality scores (perplexity, diversity metrics)
- Track training throughput improvements
- Monitor validation loss at fixed token budgets
- Compare quality/efficiency tradeoffs

**Recommendation for SwarmResearch**:
- Implement fast data quality estimation (perplexity from small model)
- Track tokens processed per hour as primary efficiency metric
- Validate that pruned data maintains model quality

---

### 1.3 Synthetic Data Generation

**Description**: Using LLMs to generate training data, either augmenting or replacing natural data. Includes instruction-tuning data, reasoning traces, and domain-specific content.

**Key Papers/Methods**:
- Self-Instruct (Wang et al., 2023)
- Evol-Instruct (Xu et al., 2024)
- Phi series "textbook" approach (Gunasekar et al., 2023)
- Cosmopedia (Allal et al., 2024)
- MAGPIE (Xu et al., 2025)
- OpenThoughts/DeepMath (2025)

**Discoverability Assessment**: **LOW-MODERATE**

| Signal | Detectability | Time to Signal |
|--------|--------------|----------------|
| Data diversity metrics | MODERATE - statistical measures | Pre-training |
| Training loss | LOW - similar to natural data | Throughout |
| Downstream task performance | LOW - requires evaluation | End of training |
| Cost efficiency | HIGH - clear compute savings | Immediate |

**Agent Detection Requirements**:
- Measure data diversity (n-gram diversity, embedding diversity)
- Track generation cost vs. acquisition cost
- Evaluate on held-out tasks
- Monitor for model collapse indicators

**Recommendation for SwarmResearch**:
- Start with verification-easy domains (math with answer checking)
- Track diversity metrics alongside training
- Implement contamination detection

---

### 1.4 Data Mixing

**Description**: Optimizing the proportion of different data sources/domains during training rather than using uniform sampling.

**Key Papers/Methods**:
- DoReMi (Xie et al., 2023) - Group DRO for domain weighting
- RegMix (Liu et al., 2025) - regression-based mixture prediction
- CLIMB (Diao et al., 2025) - clustering-based iterative bootstrapping
- MixMin (Thudi et al., 2025) - convex minimization approach
- DoGraph (2026) - graph-constrained reweighting

**Discoverability Assessment**: **MODERATE**

| Signal | Detectability | Time to Signal |
|--------|--------------|----------------|
| Domain-specific losses | MODERATE - requires per-domain tracking | Early training |
| Worst-case domain performance | MODERATE - needs evaluation | Mid-training |
| Overall convergence | MODERATE - compare to uniform | Throughout |

**Agent Detection Requirements**:
- Track per-domain loss separately
- Compute worst-case excess loss across domains
- Compare uniform vs. optimized mixing
- Monitor domain balance throughout training

**Recommendation for SwarmResearch**:
- Implement domain-aware logging
- Use small proxy models to test mixtures
- Track worst-domain performance as fairness metric

---

## 2. ARCHITECTURAL EFFICIENCY

### 2.1 Mixture of Experts (MoE)

**Description**: Sparse activation architecture where only a subset of "expert" networks process each token, enabling massive parameter counts with sub-linear compute.

**Key Papers/Methods**:
- Switch Transformer (Fedus et al., 2022)
- GShard (Lepikhin et al., 2020)
- GLaM (Du et al., 2022)
- Mixtral 8x7B (Jiang et al., 2024) - top-2 routing
- DeepSeek MoE (Dai et al., 2024) - fine-grained + shared experts
- Qwen1.5-MoE, DBRX (2024)
- OpenMoE (Xue et al., 2024)

**Discoverability Assessment**: **LOW**

| Signal | Detectability | Time to Signal |
|--------|--------------|----------------|
| Active parameter count | HIGH - architectural property | Immediate |
| Training throughput | MODERATE - compare FLOP-equivalent | Early |
| Expert load balance | HIGH - track routing distribution | Throughout |
| Final performance | LOW - requires full training | End |
| Routing stability | MODERATE - measure entropy | Early |

**Agent Detection Requirements**:
- Monitor expert utilization (load balancing loss)
- Track active vs. total parameters ratio
- Measure training throughput per FLOP
- Evaluate on diverse tasks to detect specialization

**Recommendation for SwarmResearch**:
- Implement load balancing auxiliary loss
- Track expert usage entropy (should be high)
- Compare against dense baseline at same active parameters
- Monitor for expert collapse (low entropy = bad)

---

### 2.2 Attention Variants

**Description**: Modifications to standard multi-head attention to reduce memory and compute, particularly for KV cache during inference.

**Key Papers/Methods**:
- **MQA (Multi-Query Attention)** (Shazeer, 2019) - single KV head
- **GQA (Grouped-Query Attention)** (Ainslie et al., 2023) - grouped KV heads
- **MLA (Multi-Head Latent Attention)** (Liu et al., 2024) - low-rank compression
- Tucker Attention (2026) - tensor decomposition generalization

**Discoverability Assessment**: **MODERATE-HIGH**

| Signal | Detectability | Time to Signal |
|--------|--------------|----------------|
| Memory reduction | HIGH - immediate | Immediate |
| KV cache size | HIGH - clear metric | Immediate |
| Training throughput | HIGH - measurable | Early |
| Quality degradation | MODERATE - needs evaluation | Mid-training |

**Agent Detection Requirements**:
- Measure KV cache memory footprint
- Track training throughput improvements
- Monitor attention pattern quality
- Compare perplexity vs. memory tradeoff

**Recommendation for SwarmResearch**:
- Start with GQA as balanced approach
- Track memory-per-token metric
- Validate with long-context evaluations
- Measure throughput gains

---

### 2.3 State Space Models (SSMs)

**Description**: Alternative sequence modeling architectures with linear (vs. quadratic) complexity in sequence length.

**Key Papers/Methods**:
- S4 (Gu et al., 2022) - structured state spaces
- **Mamba** (Gu & Dao, 2023) - input-conditioned selective SSM
- Mamba-2 (Dao & Gu, 2024) - improved parallelization
- **Hybrid Architectures**: Jamba, Samba, Zamba, Hymba, BlackMamba (2024)
- Nemotron-H family (2025)
- Cobra (2025) - multimodal SSM

**Discoverability Assessment**: **LOW**

| Signal | Detectability | Time to Signal |
|--------|--------------|----------------|
| Training throughput | HIGH - significant gains | Early |
- Memory scaling | HIGH - linear vs quadratic | Immediate |
| Long-context performance | MODERATE - needs evaluation | Mid-training |
| Reasoning quality | LOW - requires benchmarks | End |
| Stability | MODERATE - training dynamics | Early |

**Agent Detection Requirements**:
- Measure throughput at varying sequence lengths
- Track memory usage scaling
- Evaluate on long-context benchmarks
- Monitor for instability in training
- Test reasoning capabilities (known SSM weakness)

**Recommendation for SwarmResearch**:
- Consider hybrid architectures (Mamba + Attention)
- Test scaling from short to long sequences
- Monitor training stability carefully
- Evaluate on diverse task types

---

## 3. OPTIMIZER EFFICIENCY

### 3.1 Muon Optimizer

**Description**: Matrix-aware optimizer using orthogonalized momentum updates via Newton-Schulz iteration. Breaks from element-wise Adam paradigm.

**Key Papers/Methods**:
- Original Muon (Jordan et al., 2024)
- Moonlight/Moonshot scaling (Liu et al., 2025)
- MuonPower (2026) - combined with GradPower
- NuMuon (2026) - nuclear-norm constrained
- HTMuon (2026) - heavy-tailed variant
- Teon (2026) - tensor-level generalization

**Discoverability Assessment**: **HIGH**

| Signal | Detectability | Time to Signal |
|--------|--------------|----------------|
| Convergence speed | HIGH - clear loss trajectory | Early (first 20% training) |
| Final loss | HIGH - compare to AdamW | End |
| Training stability | MODERATE - monitor loss spikes | Early |
| Compute per step | MODERATE - orthogonalization cost | Immediate |

**Agent Detection Requirements**:
- Track loss vs. steps comparison
- Measure time-to-target-loss
- Monitor for training instabilities
- Compare final model quality

**Recommendation for SwarmResearch**:
- Implement as drop-in AdamW replacement
- Use for 2D/3D weight matrices (attention, MLP)
- Keep AdamW for embeddings/output head
- Monitor convergence curves closely

---

### 3.2 AdamW Variants

**Description**: Modifications to standard AdamW optimizer for improved efficiency or stability.

**Key Papers/Methods**:
- **AdamW** (Loshchilov & Hutter, 2019) - standard
- **Lion** (Chen et al., 2023) - evolved optimizer, simpler updates
- **Sophia** (Liu et al., 2024) - second-order with clipped updates
- **SOAP** (Vyas et al., 2025) - Shampoo-Adam hybrid
- **Adam-mini** (Zhang et al., 2025) - memory-reduced
- Blockwise LR (Wang et al., 2025) - per-block learning rates

**Discoverability Assessment**: **HIGH**

| Signal | Detectability | Time to Signal |
|--------|--------------|----------------|
| Convergence speed | HIGH - clear metric | Early |
| Memory usage | HIGH - for memory-efficient variants | Immediate |
| Training stability | MODERATE - loss spike detection | Early |
| Final performance | MODERATE - evaluation needed | End |

**Agent Detection Requirements**:
- Compare loss curves across optimizers
- Track memory overhead
- Monitor update norm distributions
- Measure time-to-convergence

**Recommendation for SwarmResearch**:
- Lion: Fastest convergence, good for experiments
- Sophia: Best final loss, worth compute overhead
- AdamW: Most reliable, best downstream performance
- Test on proxy models before scaling

---

### 3.3 Learning Rate Schedules

**Description**: Strategies for adjusting learning rate during training for optimal convergence.

**Key Papers/Methods**:
- **Cosine Decay** - standard, slow start/end, fast middle
- **Warmup-Stable-Decay (WSD)** - constant then short decay
- **Warmup-Stable-Only (WSO)** - no decay, better for SFT
- **Trapezoidal** - constant then linear cooldown
- Linear decay, polynomial decay variants

**Discoverability Assessment**: **MODERATE**

| Signal | Detectability | Time to Signal |
|--------|--------------|----------------|
| Loss trajectory shape | MODERATE - compare schedules | Throughout |
| Final convergence | MODERATE - end-of-training | End |
| Downstream performance | LOW - requires evaluation | End |
| Training flexibility | HIGH - WSD allows extension | N/A |

**Agent Detection Requirements**:
- Track loss vs. learning rate correlation
- Monitor for loss spikes at schedule transitions
- Compare final model quality across schedules
- Measure sensitivity to schedule parameters

**Recommendation for SwarmResearch**:
- WSD for flexible training (can extend)
- Cosine for fixed-budget training
- WSO if planning mid-training + SFT
- Track validation loss throughout

---

### 3.4 Schedule-Free Optimizers

**Description**: Optimizers that eliminate the need for learning rate schedules through iterate averaging.

**Key Papers/Methods**:
- **Schedule-Free AdamW (SFO)** (Defazio et al., 2024)
- "The Road Less Scheduled" - theoretical foundation
- Won MLCommons 2024 AlgoPerf Self-Tuning track

**Discoverability Assessment**: **MODERATE**

| Signal | Detectability | Time to Signal |
|--------|--------------|----------------|
| No schedule needed | HIGH - conceptual | Immediate |
- Momentum sensitivity | MODERATE - requires testing | Early |
| Convergence quality | MODERATE - compare to scheduled | Throughout |
| Large batch behavior | MODERATE - degradation observed | Mid-training |

**Agent Detection Requirements**:
- Test multiple momentum configurations
- Compare to well-tuned cosine schedule
- Monitor for loss increases (bad momentum)
- Track performance at different batch sizes

**Recommendation for SwarmResearch**:
- Use (β1=0.95, β2=0.99) as starting point
- Avoid (0.9, 0.95) - shown to fail
- Test on small scale before committing
- Consider cooldown phase for best results

---

## 4. COMPUTE EFFICIENCY

### 4.1 Mixed Precision Training

**Description**: Using lower-precision formats (FP16, BF16, FP8) for compute-intensive operations while maintaining stability.

**Key Papers/Methods**:
- **FP16** with loss scaling (Micikevicius et al., 2018)
- **BF16** - wider range, default for most LLMs
- **FP8** (E4M3/E5M2) - Hopper GPUs, 2x speedup
- **Transformer Engine** - NVIDIA FP8 framework
- **FP8-LM** (Peng et al., 2023) - extended FP8 quantization
- **COAT** (Xi et al., 2024) - per-group FP8
- **MOSS** (2025) - microscaling FP8

**Discoverability Assessment**: **VERY HIGH**

| Signal | Detectability | Time to Signal |
|--------|--------------|----------------|
| Memory reduction | HIGH - immediate | Immediate |
| Training throughput | HIGH - clear speedup | Immediate |
| Numerical stability | MODERATE - loss spike detection | Early |
| Model quality | MODERATE - evaluation needed | End |

**Agent Detection Requirements**:
- Measure memory savings
- Track tokens/second throughput
- Monitor for NaN/Inf (instability)
- Compare final model accuracy

**Recommendation for SwarmResearch**:
- BF16 as safe default
- FP8 on Hopper for maximum speed
- Implement dynamic loss scaling
- Monitor gradient norms for overflow

---

### 4.2 Gradient Checkpointing

**Description**: Trading compute for memory by recomputing activations during backward pass instead of storing them.

**Key Papers/Methods**:
- Standard activation checkpointing (Chen et al., 2016)
- Selective activation recomputation
- Full vs. partial layer checkpointing
- CPU offloading variants

**Discoverability Assessment**: **VERY HIGH**

| Signal | Detectability | Time to Signal |
|--------|--------------|----------------|
| Memory reduction | HIGH - immediate | Immediate |
| Compute overhead | HIGH - ~20-30% more time | Immediate |
| Enabling larger batches | HIGH - clear capacity gain | Immediate |

**Agent Detection Requirements**:
- Measure peak memory allocation
- Track step time increase
- Calculate memory/compute tradeoff
- Monitor for OOM errors

**Recommendation for SwarmResearch**:
- Enable by default for large models
- Checkpoint at transformer block boundaries
- Combine with gradient accumulation
- Profile memory usage per layer

---

### 4.3 Flash Attention

**Description**: IO-aware exact attention algorithm that reduces HBM accesses through tiling and kernel fusion.

**Key Papers/Methods**:
- **FlashAttention-1** (Dao et al., 2022) - 2-4x speedup
- **FlashAttention-2** (2023) - better parallelism, 2x faster
- **FlashAttention-3** (2024) - Hopper tensor cores, FP8
- FlashAttention-4 (preview) - 22% faster than cuDNN

**Discoverability Assessment**: **VERY HIGH**

| Signal | Detectability | Time to Signal |
|--------|--------------|----------------|
| Attention speedup | HIGH - clear throughput gain | Immediate |
| Memory reduction | HIGH - no materialized attention matrix | Immediate |
| Numerical accuracy | HIGH - exact attention (no approximation) | Immediate |

**Agent Detection Requirements**:
- Measure attention kernel time
- Track memory for attention layers
- Validate numerical equivalence
- Test at various sequence lengths

**Recommendation for SwarmResearch**:
- Use FlashAttention-2 as minimum
- FlashAttention-3 on H100 for FP8
- Automatic via PyTorch scaled_dot_product_attention
- Monitor for sequence length scaling

---

### 4.4 Kernel Fusion

**Description**: Combining multiple GPU kernels into single launches to reduce overhead and memory traffic.

**Key Papers/Methods**:
- **Liger-Kernel** (Hsu et al., 2024) - fused Triton kernels
- FusedLayerNorm, FusedRMSNorm
- Fused GeGLU/SwiGLU
- Fused CrossEntropy
- FusedLinearCrossEntropy (FLCE)
- Fused RoPE

**Discoverability Assessment**: **VERY HIGH**

| Signal | Detectability | Time to Signal |
|--------|--------------|----------------|
| Kernel launch reduction | HIGH - profiler metrics | Immediate |
| Memory traffic reduction | HIGH - measurable | Immediate |
| End-to-end speedup | HIGH - clear improvement | Immediate |
| Memory savings | HIGH - no intermediate tensors | Immediate |

**Agent Detection Requirements**:
- Profile kernel launch counts
- Measure memory bandwidth usage
- Track end-to-end training throughput
- Monitor for numerical differences

**Recommendation for SwarmResearch**:
- Integrate Liger-Kernel library
- 20-60% training speedup reported
- 40-60% memory reduction
- Drop-in replacement for standard layers

---

## Discoverability Summary Matrix

| Technique Category | Discoverability | Primary Signal | Time to Detection |
|-------------------|-----------------|----------------|-------------------|
| **Data Efficiency** |
| Curriculum Learning | MODERATE | Loss trajectory smoothness | Early |
| Data Pruning | HIGH | Training throughput | Immediate |
| Synthetic Data | LOW-MODERATE | Downstream performance | End |
| Data Mixing | MODERATE | Per-domain losses | Early |
| **Architectural Efficiency** |
| MoE | LOW | Load balancing + final quality | End |
| Attention Variants | MODERATE-HIGH | Memory reduction | Immediate |
| State Space Models | LOW | Throughput + long-context | Mid |
| **Optimizer Efficiency** |
| Muon | HIGH | Convergence speed | Early |
| AdamW Variants | HIGH | Loss curves | Early |
| LR Schedules | MODERATE | Loss trajectory | Throughout |
| Schedule-Free | MODERATE | No schedule needed + convergence | Early |
| **Compute Efficiency** |
| Mixed Precision | VERY HIGH | Memory + throughput | Immediate |
| Gradient Checkpointing | VERY HIGH | Memory vs compute tradeoff | Immediate |
| Flash Attention | VERY HIGH | Attention speedup | Immediate |
| Kernel Fusion | VERY HIGH | Training throughput | Immediate |

---

## Recommendations for SwarmResearch Experiment Design

### Tier 1: High-Discoverability, High-Impact (Implement First)

1. **Flash Attention** - Immediate gains, clear signals
2. **Mixed Precision (BF16/FP8)** - 2x speedup, measurable immediately
3. **Kernel Fusion (Liger-Kernel)** - 20-60% speedup, drop-in
4. **Gradient Checkpointing** - Enables larger models
5. **Muon Optimizer** - Faster convergence, clear loss signals
6. **Data Pruning** - Immediate throughput gains

### Tier 2: Moderate-Discoverability, High-Impact (Implement Second)

7. **Attention Variants (GQA/MLA)** - Memory savings, moderate validation
8. **Learning Rate Schedules (WSD)** - Flexibility, moderate detection
9. **Schedule-Free Optimizers** - No schedule tuning needed
10. **AdamW Variants (Lion/Sophia)** - Convergence speed gains
11. **Data Mixing (DoReMi-style)** - Requires proxy experiments

### Tier 3: Low-Discoverability, High-Risk/Reward (Implement Last)

12. **MoE Architectures** - Requires full training to validate
13. **State Space Models** - Long evaluation cycles
14. **Curriculum Learning** - Subtle signals, hard to tune
15. **Synthetic Data at Scale** - Quality validation challenges

---

## Agent Experimentation Protocol

### For High-Discoverability Techniques:
```
1. Implement technique
2. Measure immediate metrics (memory, throughput)
3. Run short training (1-5% of full budget)
4. Compare to baseline on loss convergence
5. Decision: adopt/reject based on clear signals
```

### For Moderate-Discoverability Techniques:
```
1. Implement technique
2. Run proxy experiments (small model, short training)
3. Tune hyperparameters based on signals
4. Scale to target model size
5. Validate with downstream evaluation
6. Decision: adopt/reject based on full picture
```

### For Low-Discoverability Techniques:
```
1. Implement technique
2. Run full-scale experiments (cannot short-circuit)
3. Compare end-to-end metrics
4. Extensive downstream evaluation
5. Decision: adopt based on comprehensive analysis
```

---

## Key Papers Reference List (2022-2026)

### Data Efficiency
- Xie et al. (2023) - DoReMi
- Marion et al. (2023) - Data pruning
- Xia et al. (2024) - LESS
- Liu et al. (2025) - RegMix
- Ye et al. (2025) - Predictive mixture laws

### Architectural Efficiency
- Fedus et al. (2022) - Switch Transformer
- Jiang et al. (2024) - Mixtral
- Dai et al. (2024) - DeepSeek MoE
- Gu & Dao (2023) - Mamba
- Dao & Gu (2024) - Mamba-2
- Liu et al. (2024) - MLA
- Ainslie et al. (2023) - GQA

### Optimizer Efficiency
- Loshchilov & Hutter (2019) - AdamW
- Chen et al. (2023) - Lion
- Liu et al. (2024) - Sophia
- Jordan et al. (2024) - Muon
- Liu et al. (2025) - Moonlight/Muon scaling
- Defazio et al. (2024) - Schedule-Free
- Vyas et al. (2025) - SOAP

### Compute Efficiency
- Dao et al. (2022) - FlashAttention
- Micikevicius et al. (2022) - FP8 training
- Peng et al. (2023) - FP8-LM
- Xi et al. (2024) - COAT
- Hsu et al. (2024) - Liger-Kernel

---

*Report generated: 2026*
*Survey period: 2022-2026*
