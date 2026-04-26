# Deep Analysis: Karpathy's Ratchet Mechanism and Pluggable Metrics

## Executive Summary

This report provides an in-depth analysis of Andrej Karpathy's ratchet mechanism used in the AutoResearch framework, focusing on the `val_bpb` metric and exploring alternative ratchet metrics for different optimization targets. The findings inform the design of a pluggable metric interface for autoconstitution.

---

## Part 1: Deep Analysis of val_bpb

### 1.1 What val_bpb Measures

**Validation Bits Per Byte (val_bpb)** is a fundamental metric for evaluating language model performance that measures the average number of bits required to encode each byte of text using the model's predicted probability distribution.

#### Mathematical Definition

```
val_bpb = total_nats / (ln(2) * total_bytes)
```

Where:
- `total_nats` = sum of cross-entropy losses (in natural log units)
- `total_bytes` = sum of UTF-8 encoded byte lengths of target tokens
- `ln(2)` = conversion factor from nats to bits

The metric can also be expressed as:

```
BPB = -log_likelihood / (bytes * ln(2))
```

Or equivalently:

```
BPB = log_2(e^loss) = loss / ln(2)
```

### 1.2 Why val_bpb is Vocabulary-Size Independent

The key insight behind BPB's independence from vocabulary size lies in **information-theoretic normalization**:

#### Core Principle

The total amount of information in a dataset `I(D)` is constant regardless of tokenization:

```
I(D) = L_T bits per token = L_B bits per byte
```

Where:
- `L_T` = length of dataset in tokens
- `L_B` = length of dataset in bytes

#### Why This Matters

1. **Tokenizer Agnostic**: Different tokenizers produce different numbers of tokens for the same text, but the total information content remains identical
2. **Fair Comparison**: Models with different vocabulary sizes (e.g., 8K vs 32K vs 100K) can be directly compared
3. **Architecture Independent**: Changes to tokenization don't bias the evaluation metric

#### Example Comparison

| Model | Vocab Size | Tokens/Byte | Raw Loss | Perplexity | BPB |
|-------|------------|-------------|----------|------------|-----|
| Model A | 8,192 | 0.25 | 2.0 | 7.4 | 2.89 |
| Model B | 32,000 | 0.20 | 2.3 | 9.9 | 2.89 |

Both models achieve the same BPB despite different tokenizers, indicating equivalent compression capability.

### 1.3 The Keep/Discard Decision Mechanism

The ratchet mechanism implements a **strictly monotonic improvement guarantee**:

#### Decision Logic

```python
def ratchet_decision(current_bpb, best_bpb, epsilon=0.0):
    """
    Determine whether to keep or discard a change.
    
    Args:
        current_bpb: New validation BPB from experiment
        best_bpb: Current best (lowest) BPB
        epsilon: Minimum improvement threshold
    
    Returns:
        bool: True if change should be kept
    """
    improvement = best_bpb - current_bpb
    return improvement > epsilon
```

#### Git-Based Implementation

```
1. Agent modifies train.py
2. Run training for fixed time budget (5 minutes)
3. Evaluate val_bpb on validation set
4. IF val_bpb improved:
       git commit -m "Improvement: val_bpb X -> Y"
       best_bpb = current_bpb
   ELSE:
       git reset --hard HEAD
       Discard change
5. Repeat
```

#### Key Properties

| Property | Description | Implication |
|----------|-------------|-------------|
| **Monotonic** | Never accept regressions | Guaranteed improvement over time |
| **Stateless** | Each decision independent | No complex state management |
| **Auditable** | Git history shows all attempts | Full experiment traceability |
| **Reversible** | Failed experiments rolled back | Clean codebase always maintained |

### 1.4 The Ratchet State Machine

```
                    +------------------+
                    |     START        |
                    +--------+---------+
                             |
                             v
              +--------------+--------------+
              |  Read program.md (human     |
              |  instructions)              |
              +--------------+--------------+
                             |
                             v
              +--------------+--------------+
              |  Read train.py (current     |
              |  implementation)            |
              +--------------+--------------+
                             |
                             v
              +--------------+--------------+
              |  Read results.tsv (history) |
              +--------------+--------------+
                             |
                             v
              +--------------+--------------+
              |  Agent proposes code change |
              +--------------+--------------+
                             |
                             v
              +--------------+--------------+
              |  Execute training (5 min)   |
              +--------------+--------------+
                             |
                             v
              +--------------+--------------+
              |  Evaluate val_bpb           |
              +--------------+--------------+
                             |
                    +--------+--------+
                    |                 |
              BETTER              WORSE/EQUAL
                    |                 |
                    v                 v
         +----------+--------+  +-----+------+
         |  git commit       |  | git reset  |
         |  Update baseline  |  | Discard    |
         +----------+--------+  +-----+------+
                    |                 |
                    +--------+--------+
                             |
                             v
              +--------------+--------------+
              |  Log results to results.tsv |
              +--------------+--------------+
                             |
                             v
                         [LOOP]
```

### 1.5 Why val_bpb is Superior to Other Metrics

#### Comparison with Alternative Metrics

| Metric | Vocab Independent | Hardware Independent | Continuous | Interpretable | Compression |
|--------|-------------------|---------------------|------------|---------------|-------------|
| **val_bpb** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| Perplexity | ❌ No | ✅ Yes | ✅ Yes | ⚠️ Partial | ⚠️ Indirect |
| Accuracy | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes | ❌ No |
| Loss | ❌ No | ✅ Yes | ✅ Yes | ⚠️ Partial | ⚠️ Indirect |
| F1 Score | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes | ❌ No |
| BLEU | ✅ Yes | ✅ Yes | ⚠️ Partial | ⚠️ Partial | ❌ No |

#### Key Advantages

1. **Compression Interpretation**: BPB directly measures compression ratio. A BPB of 2.0 means the model compresses text to 2 bits per byte (4:1 compression).

2. **Smooth Optimization**: Unlike discrete metrics (accuracy, F1), BPB provides continuous gradients for optimization.

3. **Cross-Model Comparison**: Enables fair comparison across different architectures, tokenizers, and training regimes.

4. **Signal-to-Noise Ratio**: Research shows BPB has higher SNR than task-specific metrics for scaling law predictions.

5. **Information-Theoretic Foundation**: Grounded in Shannon's information theory, providing theoretical rigor.

---

## Part 2: Alternative Ratchet Metrics Survey

### 2.1 Prompt Efficiency Metrics

#### Definition
Measures the quality of output relative to the number of input/prompt tokens required.

#### Key Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Tokens per Quality (TpQ)** | `quality_score / input_tokens` | Higher is better - more quality per token |
| **Quality per Token (QpT)** | `accuracy / (input_tokens + output_tokens)` | Efficiency of entire interaction |
| **Context Compression Ratio** | `relevant_info_extracted / context_tokens` | RAG system efficiency |
| **Prompt Effectiveness** | `task_success_rate / avg_prompt_length` | Prompt engineering optimization |

#### Composite Prompt Efficiency Metric

```python
def prompt_efficiency_score(accuracy, input_tokens, output_tokens, 
                           latency_ms, alpha=1.0, beta=0.5, gamma=0.1):
    """
    Composite metric for prompt efficiency.
    
    Args:
        accuracy: Task accuracy (0-1)
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        latency_ms: End-to-end latency in milliseconds
        alpha, beta, gamma: Weighting coefficients
    
    Returns:
        float: Higher is better
    """
    quality_term = accuracy
    token_efficiency = 1.0 / (1 + alpha * (input_tokens / 1000))
    speed_term = 1.0 / (1 + beta * (latency_ms / 1000))
    output_efficiency = 1.0 / (1 + gamma * (output_tokens / 1000))
    
    return quality_term * token_efficiency * speed_term * output_efficiency
```

#### Ratchet Application

```python
# For prompt efficiency optimization, LOWER input tokens for SAME quality is better
def prompt_efficiency_ratchet(current_metric, best_metric, epsilon=0.01):
    """
    Ratchet for prompt efficiency - maximize quality per token.
    """
    return current_metric > best_metric + epsilon  # Higher is better
```

### 2.2 Inference Speed Metrics

#### Core Metrics

| Metric | Unit | Description | Optimization Target |
|--------|------|-------------|---------------------|
| **Tokens per Second (TPS)** | tok/s | Generation throughput | Maximize |
| **Time to First Token (TTFT)** | ms | Initial response latency | Minimize |
| **Time per Output Token (TPOT)** | ms | Inter-token latency | Minimize |
| **End-to-End Latency** | ms | Total response time | Minimize |
| **Throughput** | req/s | Requests processed per second | Maximize |

#### Composite Speed Metric

```python
def inference_speed_score(tps, ttft_ms, tpot_ms, batch_size=1):
    """
    Composite metric balancing multiple speed dimensions.
    
    Lower is better for latency-focused applications.
    Higher is better for throughput-focused applications.
    """
    # Normalize to comparable scales
    tps_normalized = tps / 100.0  # Assume 100 tok/s baseline
    ttft_normalized = 1000.0 / (ttft_ms + 1)  # Convert to "tokens per second equivalent"
    tpot_normalized = 1000.0 / (tpot_ms + 1)
    
    # Weighted combination (adjust weights based on use case)
    w_tps, w_ttft, w_tpot = 0.5, 0.25, 0.25
    
    return w_tps * tps_normalized + w_ttft * ttft_normalized + w_tpot * tpot_normalized
```

#### Hardware-Aware Speed Metric

```python
def hardware_efficiency_score(tokens_per_sec, gpu_flops, gpu_memory_gb):
    """
    Measures how efficiently the model uses available hardware.
    Higher is better - more tokens per FLOP per GB.
    """
    return tokens_per_sec / (gpu_flops * gpu_memory_gb)
```

### 2.3 Memory Usage Metrics

#### Key Metrics

| Metric | Unit | Description | Target |
|--------|------|-------------|--------|
| **Peak VRAM** | GB | Maximum GPU memory during inference/train | Minimize |
| **Model Size** | GB | Parameter storage footprint | Minimize |
| **Activation Memory** | GB | Intermediate computation storage | Minimize |
| **KV Cache Size** | GB | Attention key-value cache per token | Minimize |
| **Memory Bandwidth** | GB/s | Data transfer rate | Maximize |

#### Memory Efficiency Formulas

```python
def memory_efficiency_score(peak_vram_gb, model_size_gb, batch_size, 
                           sequence_length, throughput_tok_s):
    """
    Composite memory efficiency metric.
    Higher is better - more throughput per GB of memory.
    """
    # Tokens processed per GB of memory per second
    total_tokens = batch_size * sequence_length
    memory_efficiency = throughput_tok_s / peak_vram_gb
    
    # Model compression ratio (model size vs peak usage)
    utilization = model_size_gb / peak_vram_gb
    
    return memory_efficiency * utilization

def memory_per_token(peak_vram_bytes, batch_size, sequence_length):
    """
    Memory required per token position.
    Lower is better.
    """
    return peak_vram_bytes / (batch_size * sequence_length)
```

#### Ratchet for Memory Optimization

```python
def memory_ratchet(current_vram, best_vram, current_bpb, best_bpb, 
                   memory_weight=0.3, quality_weight=0.7):
    """
    Multi-objective ratchet considering both memory and quality.
    
    Accepts change if:
    - VRAM decreases (good)
    - BPB stays same or improves (good)
    - Trade-off is favorable
    """
    vram_improvement = (best_vram - current_vram) / best_vram
    quality_improvement = (best_bpb - current_bpb) / best_bpb
    
    combined_score = memory_weight * vram_improvement + quality_weight * quality_improvement
    
    return combined_score > 0  # Net positive improvement
```

### 2.4 Multimodal Capability Metrics

#### Cross-Modal Performance Metrics

| Metric | Description | Application |
|--------|-------------|-------------|
| **Cross-Modal Retrieval Accuracy** | Correct retrieval across modalities | Image-text retrieval |
| **Modality Alignment Score** | Embedding space alignment | Multimodal representation |
| **Cross-Modal Consistency** | Agreement across modalities | Multimodal QA |
| **Modality Gap** | Distance between modality embeddings | Representation quality |
| **Modality-Specific Performance** | Performance on each modality individually | Balanced capability |

#### M3IRT Framework (MultiModal Item Response Theory)

Decomposes ability into three components:
- `ability_image`: Image-only capability
- `ability_text`: Text-only capability  
- `ability_cross`: Cross-modal integration capability

```python
def cross_modal_ability_score(image_acc, text_acc, cross_modal_acc, 
                              weights=(0.25, 0.25, 0.5)):
    """
    Weighted combination of unimodal and cross-modal abilities.
    Higher weight on cross-modal emphasizes true multimodal understanding.
    """
    w_img, w_txt, w_cross = weights
    return w_img * image_acc + w_txt * text_acc + w_cross * cross_modal_acc
```

#### CLIP Score for Multimodal Alignment

```python
def clip_score(image_embeddings, text_embeddings):
    """
    Measures semantic alignment between image and text.
    Higher is better.
    """
    # Cosine similarity between normalized embeddings
    similarity = cosine_similarity(image_embeddings, text_embeddings)
    return similarity.mean()
```

### 2.5 Generalization Metrics (Out-of-Distribution)

#### OOD Performance Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **OOD Accuracy** | Performance on OOD test sets | Direct accuracy on OOD data |
| **ID-OOD Gap** | Performance drop from ID to OOD | `acc_ID - acc_OOD` |
| **Worst-Group Accuracy** | Minimum accuracy across groups | `min(acc_group_i)` |
| **Robustness Score** | Consistency across distributions | `mean(acc_i) - std(acc_i)` |
| **Generalization Ratio** | OOD vs ID performance | `acc_OOD / acc_ID` |

#### OOD Detection Metrics

```python
def ood_detection_auroc(in_distribution_scores, out_distribution_scores):
    """
    Area under ROC curve for OOD detection.
    Higher is better - better at distinguishing ID from OOD.
    """
    from sklearn.metrics import roc_auc_score
    
    # Combine scores and create labels
    scores = np.concatenate([in_distribution_scores, out_distribution_scores])
    labels = np.concatenate([np.ones(len(in_distribution_scores)), 
                            np.zeros(len(out_distribution_scores))])
    
    return roc_auc_score(labels, scores)
```

#### Domain Generalization Score

```python
def domain_generalization_score(id_acc, ood_accs, method='worst'):
    """
    Aggregate OOD generalization performance.
    
    Methods:
    - 'worst': Minimum OOD accuracy (most conservative)
    - 'average': Mean OOD accuracy
    - 'weighted': Weighted by distribution shift magnitude
    """
    if method == 'worst':
        return min(ood_accs)
    elif method == 'average':
        return np.mean(ood_accs)
    elif method == 'weighted':
        # Weight by distance from ID
        weights = [1.0 / (1 + abs(id_acc - acc)) for acc in ood_accs]
        weights = np.array(weights) / sum(weights)
        return np.average(ood_accs, weights=weights)
```

---

## Part 3: Pluggable Metric Interface Design

### 3.1 Design Principles

1. **Modularity**: Each metric is self-contained and interchangeable
2. **Composability**: Metrics can be combined into composite objectives
3. **Configurability**: Weights and thresholds are externally configurable
4. **Extensibility**: New metrics can be added without modifying core code
5. **Reproducibility**: Metric configurations are versioned and logged

### 3.2 Core Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
import numpy as np

@dataclass
class MetricResult:
    """Standardized result container for all metrics."""
    value: float
    metadata: Dict[str, Any]
    timestamp: float
    is_better: Optional[bool] = None  # Set by comparator

class RatchetMetric(ABC):
    """
    Abstract base class for all ratchet metrics.
    
    All metrics must implement:
    - compute(): Calculate metric from evaluation data
    - is_better(): Compare two metric values
    - get_direction(): Return optimization direction
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def compute(self, evaluation_data: Dict[str, Any]) -> MetricResult:
        """
        Compute the metric value from evaluation data.
        
        Args:
            evaluation_data: Dictionary containing all evaluation artifacts
            
        Returns:
            MetricResult with computed value and metadata
        """
        pass
    
    @abstractmethod
    def is_better(self, current: float, best: float, 
                  epsilon: float = 0.0) -> bool:
        """
        Determine if current value is better than best.
        
        Args:
            current: New metric value
            best: Current best metric value
            epsilon: Minimum improvement threshold
            
        Returns:
            True if current is better than best by at least epsilon
        """
        pass
    
    @abstractmethod
    def get_direction(self) -> str:
        """
        Return optimization direction.
        
        Returns:
            'minimize' or 'maximize'
        """
        pass
    
    def get_description(self) -> str:
        """Return human-readable metric description."""
        return f"{self.name}: {self.get_direction()}"
```

### 3.3 Built-in Metric Implementations

```python
class BitsPerByteMetric(RatchetMetric):
    """Karpathy's original val_bpb metric."""
    
    def compute(self, evaluation_data: Dict[str, Any]) -> MetricResult:
        total_nats = evaluation_data['total_nats']
        total_bytes = evaluation_data['total_bytes']
        
        bpb = total_nats / (np.log(2) * total_bytes)
        
        return MetricResult(
            value=bpb,
            metadata={
                'total_nats': total_nats,
                'total_bytes': total_bytes,
                'interpretation': f'{bpb:.4f} bits per byte'
            },
            timestamp=time.time()
        )
    
    def is_better(self, current: float, best: float, epsilon: float = 0.0) -> bool:
        return current < best - epsilon  # Lower is better
    
    def get_direction(self) -> str:
        return 'minimize'

class InferenceSpeedMetric(RatchetMetric):
    """Tokens per second metric for inference optimization."""
    
    def compute(self, evaluation_data: Dict[str, Any]) -> MetricResult:
        tokens_generated = evaluation_data['tokens_generated']
        generation_time_sec = evaluation_data['generation_time_sec']
        
        tps = tokens_generated / generation_time_sec
        
        return MetricResult(
            value=tps,
            metadata={
                'tokens_generated': tokens_generated,
                'generation_time_sec': generation_time_sec,
                'interpretation': f'{tps:.2f} tokens/second'
            },
            timestamp=time.time()
        )
    
    def is_better(self, current: float, best: float, epsilon: float = 0.0) -> bool:
        return current > best + epsilon  # Higher is better
    
    def get_direction(self) -> str:
        return 'maximize'

class MemoryEfficiencyMetric(RatchetMetric):
    """Memory usage metric - lower is better."""
    
    def compute(self, evaluation_data: Dict[str, Any]) -> MetricResult:
        peak_vram_gb = evaluation_data['peak_vram_gb']
        model_size_gb = evaluation_data.get('model_size_gb', 0)
        
        # Could compute efficiency ratio
        efficiency = model_size_gb / peak_vram_gb if peak_vram_gb > 0 else 0
        
        return MetricResult(
            value=peak_vram_gb,  # Primary metric is raw usage
            metadata={
                'peak_vram_gb': peak_vram_gb,
                'model_size_gb': model_size_gb,
                'utilization_ratio': efficiency,
                'interpretation': f'{peak_vram_gb:.2f} GB peak VRAM'
            },
            timestamp=time.time()
        )
    
    def is_better(self, current: float, best: float, epsilon: float = 0.0) -> bool:
        return current < best - epsilon  # Lower is better
    
    def get_direction(self) -> str:
        return 'minimize'

class PromptEfficiencyMetric(RatchetMetric):
    """Quality per input token metric."""
    
    def compute(self, evaluation_data: Dict[str, Any]) -> MetricResult:
        accuracy = evaluation_data['accuracy']
        input_tokens = evaluation_data['input_tokens']
        output_tokens = evaluation_data.get('output_tokens', 0)
        
        # Quality per 1000 tokens
        efficiency = accuracy / (1 + (input_tokens + output_tokens) / 1000)
        
        return MetricResult(
            value=efficiency,
            metadata={
                'accuracy': accuracy,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'interpretation': f'{efficiency:.4f} quality per 1K tokens'
            },
            timestamp=time.time()
        )
    
    def is_better(self, current: float, best: float, epsilon: float = 0.0) -> bool:
        return current > best + epsilon  # Higher is better
    
    def get_direction(self) -> str:
        return 'maximize'
```

### 3.4 Composite Metric Support

```python
class CompositeMetric(RatchetMetric):
    """
    Combines multiple metrics with configurable weights.
    
    Supports:
    - Weighted sum
    - Hierarchical (lexicographic)
    - Pareto frontier tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.metrics: Dict[str, RatchetMetric] = {}
        self.weights: Dict[str, float] = {}
        self.combination_method = config.get('method', 'weighted_sum')
        
        # Initialize sub-metrics
        for name, metric_config in config.get('metrics', {}).items():
            metric_class = metric_config['class']
            self.metrics[name] = metric_class(metric_config.get('config', {}))
            self.weights[name] = metric_config.get('weight', 1.0)
    
    def compute(self, evaluation_data: Dict[str, Any]) -> MetricResult:
        """Compute all sub-metrics and combine."""
        sub_results = {}
        
        for name, metric in self.metrics.items():
            sub_results[name] = metric.compute(evaluation_data)
        
        # Combine based on method
        if self.combination_method == 'weighted_sum':
            combined_value = self._weighted_sum(sub_results)
        elif self.combination_method == 'hierarchical':
            combined_value = self._hierarchical(sub_results)
        elif self.combination_method == 'pareto':
            combined_value = self._pareto_score(sub_results)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        return MetricResult(
            value=combined_value,
            metadata={
                'sub_metrics': {k: v.value for k, v in sub_results.items()},
                'weights': self.weights,
                'method': self.combination_method
            },
            timestamp=time.time()
        )
    
    def _weighted_sum(self, sub_results: Dict[str, MetricResult]) -> float:
        """Weighted sum of normalized sub-metrics."""
        total = 0.0
        total_weight = sum(self.weights.values())
        
        for name, result in sub_results.items():
            weight = self.weights[name] / total_weight
            # Normalize based on direction
            metric = self.metrics[name]
            if metric.get_direction() == 'minimize':
                # Invert so all metrics are "higher is better"
                normalized = 1.0 / (1.0 + result.value)
            else:
                normalized = result.value
            total += weight * normalized
        
        return total
    
    def _hierarchical(self, sub_results: Dict[str, MetricResult]) -> float:
        """Lexicographic ordering - prioritize by weight order."""
        # Sort by weight (descending)
        sorted_names = sorted(self.weights.keys(), 
                            key=lambda x: self.weights[x], reverse=True)
        
        # Encode as hierarchical number
        value = 0.0
        for i, name in enumerate(sorted_names):
            result = sub_results[name]
            # Each level gets decreasing significance
            value += result.value * (10 ** (-i * 2))
        
        return value
    
    def is_better(self, current: float, best: float, epsilon: float = 0.0) -> bool:
        return current > best + epsilon  # Composite is always "higher is better"
    
    def get_direction(self) -> str:
        return 'maximize'
```

### 3.5 Configuration-Driven Metric Selection

```yaml
# metrics_config.yaml

# Single metric configuration
single_metric:
  name: "val_bpb"
  class: "BitsPerByteMetric"
  config: {}

# Multi-objective configuration
multi_objective:
  name: "quality_efficiency_tradeoff"
  class: "CompositeMetric"
  config:
    method: "weighted_sum"
    metrics:
      quality:
        class: "BitsPerByteMetric"
        weight: 0.7
        config: {}
      speed:
        class: "InferenceSpeedMetric"
        weight: 0.2
        config: {}
      memory:
        class: "MemoryEfficiencyMetric"
        weight: 0.1
        config: {}

# Hierarchical configuration (quality first, then speed)
hierarchical:
  name: "quality_then_speed"
  class: "CompositeMetric"
  config:
    method: "hierarchical"
    metrics:
      quality:
        class: "BitsPerByteMetric"
        weight: 1.0  # Primary
      speed:
        class: "InferenceSpeedMetric"
        weight: 0.5  # Secondary
```

### 3.6 Metric Factory and Registration

```python
class MetricRegistry:
    """Central registry for all available metrics."""
    
    _metrics: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, metric_class: type):
        """Register a new metric class."""
        if not issubclass(metric_class, RatchetMetric):
            raise ValueError(f"Metric class must inherit from RatchetMetric")
        cls._metrics[name] = metric_class
    
    @classmethod
    def create(cls, name: str, config: Dict[str, Any] = None) -> RatchetMetric:
        """Create a metric instance by name."""
        if name not in cls._metrics:
            raise ValueError(f"Unknown metric: {name}. "
                           f"Available: {list(cls._metrics.keys())}")
        return cls._metrics[name](config)
    
    @classmethod
    def list_metrics(cls) -> Dict[str, str]:
        """List all registered metrics with descriptions."""
        return {name: metric_class.__doc__ 
                for name, metric_class in cls._metrics.items()}

# Register built-in metrics
MetricRegistry.register("BitsPerByteMetric", BitsPerByteMetric)
MetricRegistry.register("InferenceSpeedMetric", InferenceSpeedMetric)
MetricRegistry.register("MemoryEfficiencyMetric", MemoryEfficiencyMetric)
MetricRegistry.register("PromptEfficiencyMetric", PromptEfficiencyMetric)
MetricRegistry.register("CompositeMetric", CompositeMetric)
```

### 3.7 Integration with Ratchet Mechanism

```python
class PluggableRatchet:
    """
    Ratchet mechanism with pluggable metrics.
    
    Usage:
        ratchet = PluggableRatchet.from_config("metrics_config.yaml")
        
        # In experiment loop
        result = ratchet.evaluate(evaluation_data)
        if result.should_keep:
            git_commit()
        else:
            git_reset()
    """
    
    def __init__(self, metric: RatchetMetric, epsilon: float = 0.0):
        self.metric = metric
        self.epsilon = epsilon
        self.best_value: Optional[float] = None
        self.history: List[Dict[str, Any]] = []
    
    @classmethod
    def from_config(cls, config_path: str) -> 'PluggableRatchet':
        """Create ratchet from YAML configuration."""
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        metric_config = config.get('metric', config)  # Support both formats
        metric_name = metric_config['class']
        metric = MetricRegistry.create(metric_name, metric_config.get('config', {}))
        
        epsilon = config.get('epsilon', 0.0)
        
        return cls(metric, epsilon)
    
    def evaluate(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate new result and determine if it should be kept.
        
        Returns:
            Dictionary with:
            - should_keep: Boolean decision
            - metric_result: Full MetricResult
            - improvement: Amount of improvement (if any)
        """
        result = self.metric.compute(evaluation_data)
        
        if self.best_value is None:
            # First evaluation
            should_keep = True
            improvement = float('inf')
        else:
            should_keep = self.metric.is_better(result.value, self.best_value, self.epsilon)
            if self.metric.get_direction() == 'minimize':
                improvement = self.best_value - result.value
            else:
                improvement = result.value - self.best_value
        
        if should_keep:
            self.best_value = result.value
        
        # Log to history
        self.history.append({
            'timestamp': result.timestamp,
            'value': result.value,
            'should_keep': should_keep,
            'improvement': improvement,
            'metadata': result.metadata
        })
        
        return {
            'should_keep': should_keep,
            'metric_result': result,
            'improvement': improvement,
            'best_value': self.best_value
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of ratchet history."""
        if not self.history:
            return {'total_experiments': 0}
        
        kept = [h for h in self.history if h['should_keep']]
        
        return {
            'total_experiments': len(self.history),
            'kept_experiments': len(kept),
            'success_rate': len(kept) / len(self.history),
            'best_value': self.best_value,
            'metric_name': self.metric.name,
            'direction': self.metric.get_direction()
        }
```

---

## Part 4: Recommendations for autoconstitution

### 4.1 Metric Selection Guidelines

| Use Case | Primary Metric | Secondary Metrics | Composite Weight |
|----------|---------------|-------------------|------------------|
| **Research Quality** | val_bpb | - | 100% |
| **Production Deployment** | Inference TPS | val_bpb, VRAM | 50/30/20 |
| **Edge Deployment** | VRAM usage | val_bpb, TPS | 50/30/20 |
| **RAG Systems** | Prompt efficiency | val_bpb, latency | 40/40/20 |
| **Multimodal Models** | Cross-modal accuracy | Modality-specific | 60/40 |
| **Safety-Critical** | OOD accuracy | ID accuracy | 60/40 |

### 4.2 Implementation Roadmap

#### Phase 1: Core Infrastructure
- [ ] Implement `RatchetMetric` base class
- [ ] Implement `BitsPerByteMetric` (val_bpb)
- [ ] Implement `MetricRegistry`
- [ ] Implement `PluggableRatchet`

#### Phase 2: Extended Metrics
- [ ] Implement inference speed metrics
- [ ] Implement memory efficiency metrics
- [ ] Implement composite metric support

#### Phase 3: Advanced Features
- [ ] Implement Pareto frontier tracking
- [ ] Implement multi-objective optimization
- [ ] Implement metric switching at runtime

#### Phase 4: Domain-Specific Metrics
- [ ] Implement multimodal metrics
- [ ] Implement OOD generalization metrics
- [ ] Implement prompt efficiency metrics

### 4.3 Configuration Examples

```yaml
# autoconstitution default configuration
ratchet:
  metric:
    class: "BitsPerByteMetric"
  epsilon: 0.0001  # 0.01% improvement threshold
  
# Production optimization
ratchet:
  metric:
    class: "CompositeMetric"
    config:
      method: "weighted_sum"
      metrics:
        quality:
          class: "BitsPerByteMetric"
          weight: 0.5
        speed:
          class: "InferenceSpeedMetric"
          weight: 0.3
        memory:
          class: "MemoryEfficiencyMetric"
          weight: 0.2
  epsilon: 0.001
```

---

## References

1. Karpathy, A. (2026). AutoResearch: AI agents running research on single-GPU nanochat training. GitHub.
2. Gao, L., et al. (2020). The Pile: An 800GB Dataset of Diverse Text for Language Modeling.
3. Rae, J., et al. (2021). Scaling Language Models: Methods, Analysis & Insights from Training Gopher.
4. Yuan, W., et al. (2023). BOSS: Benchmark for Out-of-Distribution Generalization in NLP.
5. Zhang, Y., et al. (2023). M3IRT: MultiModal and Multidimensional Item Response Theory.

---

*Report generated: Research Phase*
*Version: 1.0*
