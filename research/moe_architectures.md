# Mixture of Experts (MoE) and Efficient Transformer Architectures
## Research Findings Report

**Date:** January 2025  
**Research Focus:** MoE architectures, sparse attention patterns, linear attention alternatives, and automated discovery mechanisms

---

## Executive Summary

This report surveys the landscape of efficient transformer architectures, with particular emphasis on Mixture of Experts (MoE) systems, sparse attention mechanisms, and linear attention alternatives. The research reveals that MoE architectures enable scaling to hundreds of billions of parameters while maintaining constant per-token computational cost. Sparse attention patterns and linear attention alternatives provide complementary approaches to efficiency. Critically, the report identifies key signals that automated research agents could use to discover MoE-like architectures and assess the benefits of sparsity.

---

## 1. MoE Architecture Survey

### 1.1 GShard (Google, 2020)

**Key Innovation:** Automatic sharding for scaling giant models with conditional computation

**Architecture Details:**
- Scaled to 600B parameters for multilingual NMT (100 languages to English)
- Training completed in 4 days on 2048 TPU v3 accelerators
- Used Sparsely-Gated Mixture-of-Experts layers within Transformer
- Training cost: 22 TPU v3 core years vs 29 TPU years for 100 bilingual baselines

**Technical Contributions:**
- Lightweight annotation APIs for expressing parallel computation patterns
- XLA compiler extension for automatic sharding
- Position-wise MoE with parallel execution
- Top-2 or top-1 learnable routing strategies

**Key Insight:** "Conditional computation not only improves quality but remains practical and sample-efficient for training."

### 1.2 Switch Transformer (Google, 2022)

**Key Innovation:** Simplified routing with expert capacity and top-1 selection

**Architecture Details:**
- Replaced FFN layers with MoE layers
- Each token routed to only ONE expert (top-1 routing)
- Introduced expert capacity factor to prevent overload
- Added auxiliary load-balancing loss

**Routing Mechanism:**
```
r^l = W_r^l · o^(l-1)           # router logits
p_i^l = exp(r_i^l) / Σ_j exp(r_j^l)  # softmax probabilities
y^l = p_i^l · E_i^l              # output from selected expert
```

**Load Balancing Loss:**
```
L_b = n · Σ_i s_i · P_i
```
where s_i = fraction of samples to expert i, P_i = fraction of router probability for expert i

**Key Finding:** Simplifying to top-1 routing maintains quality while dramatically reducing communication and computation costs.

### 1.3 ST-MoE: Stable Transferable Mixture of Experts (Zoph et al., 2022)

**Key Innovation:** Router Z-loss for training stability

**Architecture Details:**
- 269B parameters with only 3B active per token
- Compute-efficient framework with 5-7x less compute than dense counterparts
- Router Z-loss prevents training instability
- Expert dropout for regularization

**Router Z-Loss:**
- Addresses training instability in sparse expert models
- Encourages router confidence without collapsing to single experts
- Enables stable training at massive scale

**Key Insight:** Training stability is as important as architecture design for large-scale MoE deployment.

### 1.4 Modern MoE Variants

**DeepSeekMoE (2024):**
- Fine-grained expert segmentation (64 experts per layer)
- Shared expert isolation strategy
- Addresses knowledge mixing and redundancy

**Mixtral 8x7B (2024):**
- 8 feed-forward blocks
- Top-2 routing
- Performance comparable to 70B dense model

**X-MoE:**
- Hyperspherical routing with cosine-normalized gating
- Mitigates representation collapse

---

## 2. Sparse Attention Patterns

### 2.1 Sparse Transformer (Child et al., 2019)

**Key Innovation:** Fixed-pattern attention with strided and local patterns

**Attention Patterns:**
- **Strided Attention:** Attends to every k-th token
- **Local Attention:** Sliding window of fixed size
- **Complexity:** O(n√n) vs O(n²) for full attention

**Applications:** Density modeling on Enwik8 and ImageNet-64

### 2.2 Longformer (Beltagy et al., 2020)

**Key Innovation:** Dilated sliding windows + global attention

**Attention Components:**
1. **Sliding Window:** Local attention within fixed window size w
2. **Dilated Sliding Window:** Skip every d tokens in window (extends receptive field)
3. **Global Attention:** Selected tokens attend to all positions

**Complexity:** Linear in sequence length O(n·w) where w << n

**Use Case:** Document-level NLP tasks requiring long context

### 2.3 BigBird (Zaheer et al., 2020)

**Key Innovation:** Theoretically expressive sparse attention combining three patterns

**Attention Patterns:**
1. **Random Attention:** Random token connections
2. **Window Attention:** Local sliding window
3. **Global Attention:** Global tokens (CLS, special tokens) connect to all

**Theoretical Result:** BigBird is a universal approximator (Turing complete)

**Complexity:** O(n) with appropriate pattern selection

**Key Insight:** Combination of random + local + global patterns maintains expressivity while achieving linear complexity.

### 2.4 Atomic Sparse Attention Patterns Summary

| Pattern | Description | Complexity | Use Case |
|---------|-------------|------------|----------|
| Causal | Query only attends to previous tokens | O(n²) but masked | Autoregressive generation |
| Global | Selected tokens as central hubs | O(n·g) where g = # global | Document classification |
| Sliding Window | Local neighborhood only | O(n·w) | Local dependencies |
| Random | Random token connections | O(n·r) | Long-range dependencies |
| Block Sparse | Attention on token blocks | O(n·b) | Structured data |

---

## 3. Linear Attention Alternatives

### 3.1 Linformer (Wang et al., 2020)

**Key Innovation:** Low-rank approximation of attention matrix

**Mechanism:**
- Projects keys and values to lower dimension: K' = E^T·K, V' = F^T·V
- Attention computed as: O = softmax(Q·K'^T/√d)·V'
- Complexity: O(n·k·d) where k << n

**Theoretical Basis:**
- Attention matrices empirically exhibit rapid spectral decay
- Eckart-Young-Mirsky theorem justifies low-rank approximation

**Limitations:**
- Requires rank selection hyperparameter k
- Limited in autoregressive generation
- Expressivity ceiling for certain tasks

### 3.2 Performer / FAVOR+ (Choromanski et al., 2020)

**Key Innovation:** Kernel approximation via random features

**FAVOR+ Mechanism:**
```
Attention(Q,K,V) ≈ φ(Q)·(φ(K)^T·V)
where φ(x) = exp(W·x - ||x||²/2)/√m
```

**Key Properties:**
- Unbiased or nearly-unbiased estimation
- Uniform convergence guarantees
- Low estimation variance via orthogonal random features
- No sparsity or low-rank assumptions required

**Complexity:** O(n·d²) time, O(n·d) memory

**Advantages:**
- Compatible with pretrained Transformers
- Works for arbitrary attention kernels beyond softmax
- Strong theoretical guarantees

### 3.3 Linear Transformer (Katharopoulos et al., 2020)

**Key Innovation:** Fixed positive feature map for kernel approximation

**Mechanism:**
```
φ(x) = ELU(x) + 1
Attention(Q,K,V) = φ(Q)·(φ(K)^T·V)
```

**Property:** Differentiable everywhere, better than ReLU

### 3.4 Nyströmformer (Xiong et al., 2021)

**Key Innovation:** Nyström method for attention approximation

**Mechanism:**
- Samples "landmark" points to reconstruct full attention matrix
- Iterative Moore-Penrose pseudoinverse approximation
- Residual connections for training stability

**Complexity:** Near-linear O(n·r) where r = number of landmarks

### 3.5 Linear Attention Comparison

| Method | Approach | Complexity | Key Advantage |
|--------|----------|------------|---------------|
| Linformer | Low-rank projection | O(n·k) | Simple, effective |
| Performer | Random features | O(n·d²) | Theoretical guarantees |
| Linear Transformer | Fixed kernel | O(n·d²) | No randomness |
| Nyströmformer | Landmark sampling | O(n·r) | Iterative refinement |
| cosFormer | Cosine reweighting | O(n·d²) | Position-aware |

---

## 4. Automated Discovery of MoE-like Architectures

### 4.1 Neural Architecture Search (NAS) for MoE

**MoENAS Framework:**
- Replaces standard FFN with Switch FFN layer
- Searches expert mixing space for optimal combinations
- Prunes least-used experts while maintaining performance

**Search Strategy:**
1. Replace FFN with MoE layer (multiple experts)
2. Execute search process in expert mixing space
3. Identify optimal expert combinations for accuracy/fairness/robustness
4. Prune underutilized experts

### 4.2 Discovery Signals for Automated Research

#### Signal 1: Attention Entropy Patterns
- **Observation:** High-entropy attention heads serve as pivotal hubs for semantic integration
- **Implication:** Entropy can guide where to apply sparse attention
- **Dynamic:** Layers follow deep-to-shallow maturation trajectory
- **Action:** Apply recurrence/sparsity first to deeper layers, then propagate upward

#### Signal 2: Expert Utilization Imbalance
- **Observation:** Natural emergence of preferred experts during training
- **Metric:** Router entropy decreasing over time
- **Action:** If entropy collapse detected → MoE beneficial
- **Threshold:** Entropy < threshold indicates need for load balancing

#### Signal 3: Gradient Signal Variance
- **Observation:** Different input regions benefit from different parameter sets
- **Metric:** Gradient covariance structure across input clusters
- **Action:** High covariance between clusters → separate experts

#### Signal 4: Activation Overlap
- **Observation:** Expert representations can exhibit 99% similarity
- **Metric:** Cosine similarity between expert activations
- **Action:** High overlap indicates need for specialization mechanisms

#### Signal 5: Cross-Layer Coupling
- **Observation:** Routing decisions in adjacent layers are strongly correlated
- **Metric:** Conditional activation probabilities P(expert_i^l+1 | expert_j^l)
- **Action:** Strong coupling suggests stable expert pathways

### 4.3 Automated Discovery Algorithm Sketch

```
Discovery_Agent(model, data):
    signals = {}
    
    # Phase 1: Detect sparsity opportunities
    signals['attention_entropy'] = compute_attention_entropy(model, data)
    signals['gradient_variance'] = analyze_gradient_structure(model, data)
    signals['activation_overlap'] = measure_expert_similarity(model, data)
    
    # Phase 2: Assess MoE viability
    if signals['activation_overlap'] > threshold_overlap:
        if signals['gradient_variance'] > threshold_variance:
            return "MoE recommended"
    
    # Phase 3: Determine MoE configuration
    if MoE recommended:
        n_experts = estimate_optimal_experts(signals)
        routing_strategy = select_routing(signals)
        load_balance = detect_imbalance_signals(signals)
    
    # Phase 4: Validate
    candidate = build_MoE(config)
    return evaluate(candidate, data)
```

---

## 5. Signals Indicating Sparse Architecture Benefits

### 5.1 Training Dynamics Signals

| Signal | Indicator | Threshold | Action |
|--------|-----------|-----------|--------|
| Router Entropy Collapse | H(router) → 0 | H < 0.5 | Implement load balancing |
| Expert Gradient Variance | Var(∇θ_i) high | CV > 2.0 | Increase expert count |
| Activation Similarity | cos_sim > 0.95 | > 0.9 | Add specialization loss |
| Token Distribution | Max(s_i) > 0.5 | > 0.4 | Apply capacity constraints |
| Cross-Layer Correlation | P(layer+1 | layer) high | > 0.7 | Leverage coupling |

### 5.2 Task-Structure Signals

| Signal | Description | Implication |
|--------|-------------|-------------|
| Multi-domain Data | Clear domain boundaries | Separate experts per domain |
| Hierarchical Structure | Layered task decomposition | Layer-wise expert allocation |
| Long Context Requirements | n > 4096 tokens | Sparse or linear attention |
| Locality Patterns | Nearby tokens more relevant | Sliding window attention |
| Global Context Needs | CLS token importance | Global attention tokens |

### 5.3 Computational Efficiency Signals

| Signal | Metric | Threshold | Solution |
|--------|--------|-----------|----------|
| Memory Bottleneck | OOM errors | Any | Linear attention |
| Slow Training | Time/epoch high | >2x baseline | Sparse patterns |
| Inference Latency | Tokens/sec low | <target | MoE + top-k routing |
| Parameter Efficiency | Params/performance | Suboptimal | Conditional computation |

---

## 6. Implications for autoconstitution

### 6.1 Architecture Search Strategy

**Phase 1: Signal Detection**
- Monitor attention entropy patterns across layers
- Track gradient flow and expert utilization
- Measure activation similarity between potential expert regions

**Phase 2: Candidate Generation**
- Generate MoE variants based on detected signals
- Explore sparse attention patterns matching data structure
- Consider linear attention for long-context requirements

**Phase 3: Evaluation Protocol**
- Compare against dense baseline with equal compute
- Measure both quality and efficiency metrics
- Validate specialization through expert analysis

### 6.2 Key Design Principles

1. **Start Dense, Evolve Sparse**
   - Begin with dense architecture
   - Monitor signals during training
   - Introduce sparsity where signals indicate benefit

2. **Layer-Dependent Strategies**
   - Early layers: Dense or simple patterns (general features)
   - Middle layers: Moderate sparsity
   - Deep layers: Full MoE (specialization)

3. **Dynamic Adaptation**
   - Adjust expert count based on gradient variance
   - Modify attention patterns based on entropy
   - Balance load based on token distribution

4. **Theoretical Grounding**
   - Prefer methods with convergence guarantees (FAVOR+, BigBird)
   - Validate expressivity requirements before sparsification
   - Monitor approximation error in linear methods

### 6.3 Research Priorities

| Priority | Direction | Expected Impact |
|----------|-----------|-----------------|
| High | Automated signal detection | Reduce manual tuning |
| High | Dynamic expert allocation | Improve efficiency |
| Medium | Hybrid sparse-dense layers | Balance quality/cost |
| Medium | Cross-architecture transfer | Leverage pretrained models |
| Low | Novel routing mechanisms | Push specialization bounds |

---

## 7. Key Takeaways

1. **MoE architectures** (GShard, Switch, ST-MoE) enable scaling to 600B+ parameters with constant per-token compute through conditional computation

2. **Sparse attention patterns** (Sparse Transformer, Longformer, BigBird) reduce complexity from O(n²) to O(n) while maintaining expressivity through strategic pattern selection

3. **Linear attention alternatives** (Performer, Linformer) provide theoretical guarantees and practical efficiency for long sequences

4. **Automated discovery** should focus on detecting: attention entropy patterns, gradient variance across input clusters, activation overlap, and expert utilization imbalance

5. **Critical signals** for MoE benefit include: high activation similarity (need specialization), gradient variance across domains (need separate experts), and entropy collapse (need load balancing)

6. **For autoconstitution**: Start with dense architectures, monitor training dynamics, introduce sparsity where signals indicate benefit, prefer theoretically-grounded methods

---

## References

- Lepikhin et al. (2020): GShard: Scaling Giant Models with Conditional Computation
- Fedus et al. (2022): Switch Transformers: Scaling to Trillion Parameter Models
- Zoph et al. (2022): ST-MoE: Designing Stable and Transferable Sparse Expert Models
- Child et al. (2019): Generating Long Sequences with Sparse Transformers
- Beltagy et al. (2020): Longformer: The Long-Document Transformer
- Zaheer et al. (2020): Big Bird: Transformers for Longer Sequences
- Wang et al. (2020): Linformer: Self-Attention with Linear Complexity
- Choromanski et al. (2020): Rethinking Attention with Performers
- Xiong et al. (2021): Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention

---

*Report compiled for autoconstitution initiative on efficient transformer architectures.*
