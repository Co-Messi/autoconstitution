# Rigorous Benchmark Design: autoconstitution vs Single-Agent Baseline

## Executive Summary

This report presents a comprehensive framework for designing rigorous benchmarks that demonstrate autoconstitution's advantages over Karpathy's single-agent autoresearch baseline. Drawing from empirical studies on multi-agent collaboration, statistical rigor in ML evaluation, and reproducibility best practices, we establish a benchmark protocol that any user can run on their hardware.

---

## 1. Benchmark Design Principles

### 1.1 Core Principles for Multi-Agent vs Single-Agent Evaluation

Based on research findings [^251^][^258^][^291^], effective benchmark design must address:

| Principle | Description | Rationale |
|-----------|-------------|-----------|
| **Controlled Time Budgets** | Fix wall-clock time, not iteration count | Multi-agent systems have different per-iteration costs; time normalization ensures fair comparison |
| **Task Parallelizability Assessment** | Measure performance on parallelizable vs sequential tasks | Multi-agent excels (+81%) on parallelizable tasks but degrades (-70%) on sequential ones [^258^] |
| **Context Window Saturation Monitoring** | Track when agent context becomes overloaded | Single-agent degrades when context saturates; multi-agent distributes cognitive load |
| **Quality-Diversity Tradeoff** | Measure both solution quality and diversity | Multi-agent systems should discover more diverse solutions in the same search space |
| **Statistical Rigor** | Report confidence intervals, multiple seeds | Single-run results are insufficient for reliable conclusions [^280^] |

### 1.2 The Fundamental Trade-off

Research reveals a critical trade-off in multi-agent architectures [^291^]:

- **Subagent/Parallel Mode**: High-throughput, resilient search engine optimal for broad, shallow optimizations under strict time constraints
- **Agent Team/Expert Mode**: Higher fragility but achieves deep theoretical alignment for complex architectural refactoring given extended budgets

**Design Implication**: Benchmarks must test both scenarios to fairly characterize when multi-agent approaches excel.

---

## 2. Key Metrics That Matter

### 2.1 Primary Performance Metrics

#### 2.1.1 Wall-Clock Time to Target (WCTT)

**Definition**: Time required to reach a predefined performance threshold.

```
WCTT_X = min{t : metric(t) >= target_X%}
```

**Targets to measure**:
- WCTT_90: Time to 90% of optimal known performance
- WCTT_95: Time to 95% of optimal known performance
- WCTT_99: Time to 99% of optimal known performance

**Why it matters**: In autoresearch, the primary advantage is speed of discovery. Wall-clock time captures the true user experience.

#### 2.1.2 Time-to-Target Improvement (TTTI)

**Definition**: Relative improvement in time-to-target compared to baseline.

```
TTTI = (WCTT_baseline - WCTT_system) / WCTT_baseline × 100%
```

**Interpretation**: Positive values indicate speedup; negative values indicate slowdown.

### 2.2 Efficiency Metrics

#### 2.2.1 Experiments Per Hour (EPH)

**Definition**: Number of complete experiment cycles completed per hour.

```
EPH = N_experiments / T_total_hours
```

**Why it matters**: Multi-agent systems can run experiments in parallel, potentially achieving higher throughput.

#### 2.2.2 Success Rate (SR)

**Definition**: Proportion of experiments that yield valid, measurable results.

```
SR = N_successful / N_total × 100%
```

**Why it matters**: Multi-agent systems may have higher failure rates due to coordination complexity.

### 2.3 Quality Metrics

#### 2.3.1 Best Found Performance (BFP)

**Definition**: Best objective value achieved within the time budget.

```
BFP = max/min{f(x_i) : i ∈ [1, N]}
```

(Use max for maximization, min for minimization)

#### 2.3.2 Final Performance Gap (FPG)

**Definition**: Gap between achieved and theoretical optimum.

```
FPG = |f_best - f_optimal| / |f_optimal| × 100%
```

### 2.4 Diversity Metrics

#### 2.4.1 Solution Space Coverage (SSC)

**Definition**: Measure of how thoroughly the solution space has been explored.

```
SSC = Volume(convex_hull(solutions)) / Volume(search_space)
```

**Implementation**: Use hypervolume indicator for multi-objective or quality-diversity scenarios [^292^].

#### 2.4.2 Configuration Diversity (CD)

**Definition**: Diversity of hyperparameter configurations tried.

```
CD = 1 - (1/N²) Σᵢ Σⱼ similarity(config_i, config_j)
```

Where similarity can be cosine similarity or normalized Euclidean distance.

#### 2.4.3 Unique Local Optima Discovered (ULOD)

**Definition**: Count of distinct local optima found during search.

**Why it matters**: Multi-agent systems should discover more diverse solutions due to parallel exploration [^292^].

---

## 3. Measuring Exploration vs Exploitation

### 3.1 The Exploration-Exploitation Trade-off

In optimization and search, exploration refers to discovering new regions of the search space, while exploitation refers to refining known good solutions [^271^][^283^][^288^].

### 3.2 Quantitative Measures

#### 3.2.1 Exploration Ratio (ER)

**Definition**: Proportion of experiments that explore new regions vs refine existing ones.

```
ER = N_explore / (N_explore + N_exploit)
```

**Classification criteria**:
- **Exploration**: Configuration similarity to all previous < threshold θ
- **Exploitation**: Configuration similarity to best known > threshold θ

Recommended θ = 0.3 (normalized distance)

#### 3.2.2 Search Space Novelty (SSN)

**Definition**: Average dissimilarity of each new configuration to all previous ones.

```
SSN(t) = (1/t) Σᵢ₌₁ᵗ minⱼ₍ᵢ₎ d(config_i, config_j)
```

Higher SSN indicates more exploration.

#### 3.2.3 Improvement Velocity (IV)

**Definition**: Rate of improvement in the best-found solution over time.

```
IV(t) = (f_best(t) - f_best(t-Δt)) / Δt
```

**Interpretation**:
- High IV: System is in exploitation phase, rapidly improving
- Low IV: System is in exploration phase, searching broadly

#### 3.2.4 Acquisition Function Analysis

For Bayesian optimization-style approaches, track the balance parameter β in Upper Confidence Bound:

```
UCB(x) = μ(x) + β·σ(x)
```

Where:
- μ(x) = predicted mean (exploitation)
- σ(x) = predicted uncertainty (exploration)
- β = exploration-exploitation trade-off parameter

**Recommendation**: Log β values over time to characterize exploration patterns.

### 3.3 Visualization Methods

1. **Search Trajectory Plots**: Visualize configuration changes over time
2. **Objective vs Diversity Scatter**: Plot solution quality vs configuration diversity
3. **Exploration Heatmaps**: Show density of explored regions in 2D parameter slices
4. **Convergence Curves**: Best found vs time, with exploration/exploitation phases marked

---

## 4. Preventing Overfitting to Benchmark Artifacts

### 4.1 Sources of Benchmark Overfitting

Based on research findings [^262^][^266^][^269^][^275^]:

| Source | Description | Mitigation Strategy |
|--------|-------------|---------------------|
| **Data Leakage** | Test set information contaminates training/optimization | Strict temporal partitioning, hold-out test sets |
| **Prompt Structure Overfitting** | Models exploit benchmark-specific phrasing | Systematic prompt variations, synonym replacement [^269^] |
| **Hyperparameter Leakage** | Tuning on test set performance | Explicit train/validation/test separation |
| **Repository Bias** | Over-optimization to specific codebases | Diverse repository sampling [^275^] |
| **Artifact Exploitation** | Models learn to detect benchmark artifacts rather than solve tasks | Adversarial filtering, diverse problem instances |

### 4.2 Mitigation Strategies

#### 4.2.1 Benchmark Mutation

Transform benchmark problems to create variants that test the same underlying capability without allowing memorization [^266^]:

- Rename variables and functions
- Reorder code blocks
- Change numeric constants while preserving problem structure
- Modify surface syntax while preserving semantics

#### 4.2.2 Diverse Problem Sampling

- Use problems from multiple domains
- Vary problem difficulty systematically
- Include problems that don't have known optimal solutions
- Sample from temporal partitions (post-training-cutoff data) [^275^]

#### 4.2.3 Anti-Overfitting Instructions

When using LLM-based evaluation, explicitly instruct generators to [^278^][^286^]:
- Create prompts with diverse framings and variants
- Include prompts that should NOT cause specific behaviors (to establish boundaries)
- Focus on qualitative understanding over quantitative scores

#### 4.2.4 Cross-Validation for Benchmarks

Apply k-fold cross-validation principles:
- Split problems into k folds
- Report performance across all folds
- Measure variance across folds as overfitting indicator

---

## 5. Statistical Rigor in Benchmark Design

### 5.1 Required Statistical Practices

Based on established ML evaluation standards [^261^][^280^][^299^]:

#### 5.1.1 Multiple Random Seeds

**Minimum**: 5 independent runs with different random seeds
**Recommended**: 10+ runs for publication-quality results
**Seeds to use**: 42, 123, 456, 789, 1024, 2024, 31415, 271828, 161803, 999999 [^299^]

#### 5.1.2 Confidence Intervals

Report 95% confidence intervals for all metrics using:

**Percentile Bootstrap** (simplest):
```
CI = [θ^(2.5%), θ^(97.5%)] from B=1000 bootstrap samples
```

**BCa Bootstrap** (more accurate for skewed distributions):
```
CI = [θ^(α₁)*, θ^(α₂)*] with bias correction
```

**Analytical Methods** (for proportions):
- Wilson score intervals for accuracy/exact match rates

#### 5.1.3 Hypothesis Testing

For comparing autoconstitution vs baseline:

**Paired t-test** (same random seeds for both systems):
```
H₀: μ_swarm = μ_baseline
H₁: μ_swarm ≠ μ_baseline (two-tailed) or μ_swarm > μ_baseline (one-tailed)
```

**Effect Size** (Cohen's d):
```
d = (μ_swarm - μ_baseline) / σ_pooled
```
- d = 0.2: small effect
- d = 0.5: medium effect
- d = 0.8: large effect

**Non-parametric alternatives**:
- Wilcoxon signed-rank test (for paired samples)
- Mann-Whitney U test (for independent samples)

#### 5.1.4 Multiple Comparison Correction

When testing multiple metrics or configurations:
- **Bonferroni correction**: α_adjusted = α / m
- **False Discovery Rate (FDR)**: Control expected proportion of false discoveries

### 5.2 Reporting Standards

Every benchmark result should report:

```
Metric: Mean ± StdDev [95% CI]
Example: WCTT_95: 45.2 ± 8.3 min [38.9, 53.1]
```

Include:
1. Point estimate (mean or median)
2. Measure of dispersion (std dev or IQR)
3. Confidence interval
4. Sample size (number of runs)
5. Random seeds used

---

## 6. Reproducibility Requirements

### 6.1 Three Levels of Reproducibility [^301^]

| Level | Description | Requirements |
|-------|-------------|--------------|
| **Bitwise Exact** | Identical binary output | Fixed seeds, deterministic algorithms, containerized environment |
| **Algorithmic Deterministic** | Same numbers within floating point | Same code, same hyperparameters, same hardware class |
| **Statistical** | Same distribution within confidence bounds | Same protocol, multiple runs, statistical equivalence testing |

**Recommendation**: Target Algorithmic Deterministic for most benchmarks; Statistical for hyperparameter optimization scenarios.

### 6.2 Reproducibility Checklist

#### 6.2.1 Code and Environment

- [ ] **Version Control**: All code in git with commit hashes logged
- [ ] **Dependency Management**: requirements.txt, environment.yml, or poetry.lock
- [ ] **Containerization**: Docker image with exact environment
- [ ] **Hardware Specification**: Document CPU, GPU, RAM, disk I/O
- [ ] **Random Seed Logging**: All seeds recorded and reported

#### 6.2.2 Data and Configuration

- [ ] **Dataset Versioning**: Exact dataset snapshots with checksums
- [ ] **Configuration Files**: YAML/JSON configs for all experiments
- [ ] **Baseline Specifications**: Exact baseline configurations documented
- [ ] **Preprocessing Pipeline**: Documented and versioned

#### 6.2.3 Execution and Logging

- [ ] **Experiment Tracking**: MLflow, Weights & Biases, or similar
- [ ] **Structured Logging**: JSON logs with all relevant metrics
- [ ] **Artifact Storage**: Model checkpoints, generated code, results
- [ ] **Execution Scripts**: One-command reproduction scripts

#### 6.2.4 Documentation

- [ ] **README**: Clear instructions for reproduction
- [ ] **Method Section**: Detailed protocol description
- [ ] **Results Section**: All metrics with confidence intervals
- [ ] **Limitations**: Known sources of variance or non-determinism

### 6.3 Reproducibility Artifacts

Provide the following artifacts with benchmark release:

```
benchmark_package/
├── code/
│   ├── swarm_research/          # autoconstitution implementation
│   ├── single_agent_baseline/   # Karpathy-style baseline
│   ├── evaluation/              # Evaluation scripts
│   └── utils/                   # Shared utilities
├── configs/
│   ├── experiment_config.yaml   # Main experiment configuration
│   ├── model_config.yaml        # Model architecture config
│   └── benchmark_tasks.yaml     # Task definitions
├── data/
│   ├── datasets/                # Dataset snapshots
│   └── baselines/               # Pre-computed baseline results
├── scripts/
│   ├── run_benchmark.sh         # Main benchmark script
│   ├── reproduce_paper.sh       # Reproduce published results
│   └── analyze_results.py       # Analysis notebook/script
├── docker/
│   ├── Dockerfile               # Container definition
│   └── docker-compose.yml       # Multi-service setup
└── results/
    ├── raw/                     # Raw experiment outputs
    ├── processed/               # Processed metrics
    └── figures/                 # Generated plots
```

---

## 7. Specific Benchmark Protocol for autoconstitution

### 7.1 Benchmark Task: NanoGPT Optimization

Based on Karpathy's autoresearch paradigm [^239^][^250^][^251^][^253^]:

#### 7.1.1 Task Definition

**Objective**: Optimize a small GPT model's validation bits-per-byte (val_bpb) through automated hyperparameter and architecture search.

**Baseline Configuration**:
- Model: GPT-2 small (124M parameters)
- Dataset: FineWeb subset (10B tokens)
- Vocabulary: 4,096 BPE tokens
- Sequence length: 512 tokens
- Hardware: Single NVIDIA GPU (H100 80GB or equivalent)

**Starting Point**: Karpathy's default configuration with val_bpb ≈ 0.991

#### 7.1.2 Time Budgets

| Budget | Duration | Use Case |
|--------|----------|----------|
| Short | 1 hour | Quick iteration, smoke testing |
| Medium | 4 hours | Standard benchmark comparison |
| Long | 12 hours | Extended optimization study |
| Overnight | 24 hours | Maximum optimization potential |

#### 7.1.3 Search Space

**Hyperparameters**:
- Learning rate: [1e-5, 1e-2]
- Batch size: [16, 128]
- Optimizer: [AdamW, Muon, Sophia, SOAP]
- Weight decay: [0.0, 0.1]
- Dropout: [0.0, 0.2]
- Warmup steps: [0, 2000]
- LR schedule: [cosine, linear, constant]

**Architecture Modifications** (if supported):
- Depth: [6, 24] layers
- Width: [384, 1024] embeddings
- Attention heads: [6, 16]
- Activation: [GELU, SwiGLU, ReLU]

### 7.2 Evaluation Protocol

#### 7.2.1 Experiment Structure

```python
# Pseudocode for benchmark execution

def run_benchmark(system, time_budget_hours, n_seeds=5):
    results = []
    
    for seed in SEEDS[:n_seeds]:
        set_seed(seed)
        baseline = initialize_baseline()
        
        start_time = time.time()
        end_time = start_time + time_budget_hours * 3600
        
        experiments = []
        while time.time() < end_time:
            config = system.propose_config(history)
            result = run_experiment(config, max_duration=300)  # 5-min experiments
            
            if result.success:
                system.update(config, result)
                experiments.append({
                    'config': config,
                    'val_bpb': result.val_bpb,
                    'timestamp': time.time() - start_time
                })
        
        results.append({
            'seed': seed,
            'experiments': experiments,
            'best_val_bpb': min(e['val_bpb'] for e in experiments),
            'n_experiments': len(experiments)
        })
    
    return aggregate_results(results)
```

#### 7.2.2 Metrics to Collect

**Per Experiment**:
- Configuration (hyperparameters, architecture)
- Validation bits-per-byte (val_bpb)
- Training time
- GPU memory usage
- Success/failure status
- Error messages (if failed)

**Per Run** (single seed, full time budget):
- Best val_bpb achieved
- Time to reach 90%, 95%, 99% of known optimum
- Number of successful experiments
- Number of failed experiments
- Final exploration ratio
- Configuration diversity metric

**Aggregate** (across seeds):
- Mean ± std dev of best val_bpb
- 95% confidence intervals
- Success rate statistics
- Statistical significance tests

### 7.3 Comparison Dimensions

#### 7.3.1 Single-Agent Baseline (Karpathy-style)

**Architecture**: Single LLM agent with propose-execute-evaluate loop

**Characteristics**:
- Sequential experiment execution
- Full context of all previous experiments
- Simple accept/reject based on metric improvement

#### 7.3.2 autoconstitution Variants

**Variant A: Parallel Subagent Mode**
- Multiple agents explore in parallel
- Post-hoc consolidation of results
- High-throughput, resilient to individual failures

**Variant B: Expert Team Mode**
- Specialized agents with distinct roles
- Pre-execution handoffs between experts
- Deep theoretical deliberation for complex changes

**Variant C: Hybrid Adaptive Mode**
- Dynamically route between parallel and team modes
- Task complexity assessment
- Adaptive collaboration topology

### 7.4 Success Criteria

autoconstitution demonstrates advantage when:

| Criterion | Threshold | Evidence |
|-----------|-----------|----------|
| **Speedup** | 20%+ faster time-to-target | WCTT_95 significantly lower (p < 0.05) |
| **Quality** | 5%+ better final performance | BFP significantly better (p < 0.05) |
| **Diversity** | 2x more unique configurations | CD and ULOD metrics |
| **Robustness** | Lower variance across seeds | Smaller std dev, tighter CIs |
| **Efficiency** | Higher success rate | SR > baseline SR |

---

## 8. Implementation Guidelines

### 8.1 Hardware Requirements

**Minimum**:
- GPU: NVIDIA GPU with 24GB+ VRAM (RTX 4090, A5000)
- RAM: 32GB system memory
- Storage: 100GB free space for datasets and checkpoints

**Recommended**:
- GPU: NVIDIA H100 80GB or A100 80GB
- RAM: 64GB+ system memory
- Storage: 500GB+ NVMe SSD

### 8.2 Software Stack

```yaml
# benchmark_environment.yaml
name: swarm-benchmark
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pytorch>=2.2.0
  - pytorch-cuda=12.1
  - numpy>=1.24
  - scipy>=1.10
  - pandas>=2.0
  - matplotlib>=3.7
  - seaborn>=0.12
  - pip
  - pip:
    - transformers>=4.35
    - datasets>=2.14
    - wandb>=0.15
    - mlflow>=2.8
    - openai>=1.0
    - anthropic>=0.8
    - pydantic>=2.0
    - pyyaml>=6.0
    - pytest>=7.0
```

### 8.3 Execution Command

```bash
# Run full benchmark
./scripts/run_benchmark.sh \
  --system swarm_research \
  --baseline single_agent \
  --time-budget 4h \
  --seeds 5 \
  --output-dir ./results/

# Reproduce published results
./scripts/reproduce_paper.sh \
  --paper-version v1.0 \
  --output-dir ./reproduction/

# Analyze results
python scripts/analyze_results.py \
  --results-dir ./results/ \
  --output-dir ./analysis/
```

---

## 9. Analysis and Reporting

### 9.1 Statistical Analysis Pipeline

```python
# analysis_pipeline.py

def analyze_benchmark_results(results_dir):
    # Load all experiment results
    results = load_results(results_dir)
    
    # Compute aggregate metrics
    metrics = compute_metrics(results)
    
    # Statistical significance tests
    significance = run_significance_tests(results)
    
    # Effect size calculations
    effect_sizes = compute_effect_sizes(results)
    
    # Generate visualizations
    plots = generate_plots(results)
    
    # Compile report
    report = compile_report(metrics, significance, effect_sizes, plots)
    
    return report
```

### 9.2 Required Visualizations

1. **Convergence Curves**: Best val_bpb vs time for each system
2. **Box Plots**: Distribution of final performance across seeds
3. **Exploration Heatmaps**: Parameter space coverage comparison
4. **Time-to-Target Bar Chart**: WCTT metrics side-by-side
5. **Diversity Scatter**: Solution quality vs configuration diversity
6. **Success Rate Comparison**: Bar chart of success rates

### 9.3 Report Template

```markdown
# Benchmark Results: autoconstitution vs Single-Agent

## Executive Summary
- **Winner**: [autoconstitution | Single-Agent | Tie]
- **Key Finding**: [One-sentence summary]
- **Statistical Significance**: p = [value] < 0.05

## Experimental Setup
- Time Budget: [X] hours
- Number of Seeds: [N]
- Hardware: [GPU model]
- Date: [YYYY-MM-DD]

## Results

### Primary Metrics
| Metric | autoconstitution | Single-Agent | p-value | Effect Size |
|--------|---------------|--------------|---------|-------------|
| WCTT_95 (min) | X ± Y [CI] | A ± B [CI] | 0.0XX | 0.XX |
| Best val_bpb | X ± Y [CI] | A ± B [CI] | 0.0XX | 0.XX |
| Success Rate | X% ± Y% | A% ± B% | 0.0XX | - |

### Diversity Metrics
| Metric | autoconstitution | Single-Agent | Ratio |
|--------|---------------|--------------|-------|
| Config Diversity | X | Y | X/Y |
| Unique Optima | X | Y | X/Y |
| Search Coverage | X% | Y% | X/Y |

### Exploration Analysis
- Exploration Ratio: autoconstitution X% vs Single-Agent Y%
- Search Space Novelty: [Comparison]

## Conclusions
[Detailed interpretation of results]

## Limitations
[Known limitations and sources of uncertainty]

## Reproducibility
- Code Version: [git commit hash]
- Docker Image: [image tag]
- Raw Results: [link to data]
```

---

## 10. References and Further Reading

### Key Papers Cited

1. **Karpathy's Autoresearch**: Original single-agent autoresearch framework [^239^][^253^]
2. **Multi-Agent Collaboration Study**: Empirical study of multi-agent topologies for automated research [^251^][^291^]
3. **Single vs Multi-Agent LLMs**: Comparative analysis showing when each excels [^240^][^258^]
4. **Statistical Rigor in LLM Evaluation**: Distributed framework with bootstrap CIs [^261^]
5. **Quality Diversity Optimization**: Measuring solution diversity in high-dimensional spaces [^292^]
6. **Benchmark Overfitting**: Analysis of contamination and overfitting in benchmarks [^266^][^269^][^275^]
7. **Exploration vs Exploitation**: Bayesian optimization and multi-armed bandit perspectives [^271^][^283^]
8. **Reproducibility in ML**: Three levels of reproducibility and best practices [^301^]

### Related Benchmarks

- **SWE-Bench**: Software engineering benchmark with mutation methodology [^266^]
- **HellaSwag**: Commonsense reasoning with adversarial filtering [^277^]
- **OGB**: Open Graph Benchmark with standardized protocols [^280^]
- **MultiAgentBench**: Collaboration and competition evaluation [^257^]

---

## Appendix A: Quick Reference Card

### Metric Formulas

| Metric | Formula | Target |
|--------|---------|--------|
| WCTT_X | min{t : metric(t) >= X%} | Lower is better |
| TTTI | (WCTT_base - WCTT_sys) / WCTT_base | Higher is better |
| EPH | N_experiments / T_hours | Context-dependent |
| BFP | max/min{f(x_i)} | Optimal direction |
| ER | N_explore / N_total | 0.3-0.7 optimal |
| SSC | Volume(hull) / Volume(space) | Higher is better |

### Statistical Tests

| Test | Use Case | Python |
|------|----------|--------|
| Paired t-test | Same seeds, different systems | `scipy.stats.ttest_rel` |
| Independent t-test | Different seeds | `scipy.stats.ttest_ind` |
| Wilcoxon | Non-parametric paired | `scipy.stats.wilcoxon` |
| Bootstrap CI | Any metric | `numpy.percentile` |
| Cohen's d | Effect size | Manual calculation |

### Random Seeds

```python
SEEDS = [42, 123, 456, 789, 1024, 2024, 31415, 271828, 161803, 999999]
```

### Confidence Level

- **Standard**: 95% confidence intervals
- **Significance**: α = 0.05
- **Effect size thresholds**: 0.2 (small), 0.5 (medium), 0.8 (large)

---

*Document Version: 1.0*
*Last Updated: 2025*
*For questions or updates, refer to the autoconstitution repository*
