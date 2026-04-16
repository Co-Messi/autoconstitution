# autoconstitution Performance Metrics Definition

## Overview

This document defines the complete set of metrics for measuring autoconstitution system performance across multiple dimensions: improvement capability, exploration efficiency, solution quality, and cost-effectiveness.

---

## 1. Primary Metrics

### 1.1 Improvement Rate (IR)

**Definition**: The rate at which the swarm improves solution quality over time.

**Formula**:
```
IR = (Q_final - Q_initial) / (t_final - t_initial)
```

Where:
- `Q_final`: Best solution quality at time t_final
- `Q_initial`: Best solution quality at time t_initial
- Time can be measured in iterations, wall-clock time, or compute hours

**Variants**:
- **Instantaneous IR**: IR measured over short time windows
- **Cumulative IR**: IR from start to current point
- **Normalized IR**: IR divided by maximum possible improvement

**Target**: Higher is better. Benchmark against random search baseline.

---

### 1.2 Time to Solution (TTS)

**Definition**: Time required to reach a target quality threshold.

**Formula**:
```
TTS(Q_target) = min{t : Q_best(t) >= Q_target}
```

**Variants**:
- **TTS-90**: Time to reach 90% of optimal quality
- **TTS-95**: Time to reach 95% of optimal quality
- **TTS-optimal**: Time to reach known optimal (if available)

**Measurement Units**:
- Wall-clock time (seconds)
- Iteration count
- Compute hours (GPU/CPU hours)
- API calls (for LLM-based systems)

**Target**: Lower is better.

---

### 1.3 Success Rate (SR)

**Definition**: Percentage of runs that achieve target quality within budget.

**Formula**:
```
SR = (Number of successful runs / Total runs) × 100%
```

Success criteria:
- Reach Q_target within budget B
- Or: Find solution within ε of optimal

**Target**: Higher is better. Report confidence intervals.

---

### 1.4 Best Solution Quality (BSQ)

**Definition**: Maximum quality achieved across all swarm iterations.

**Formula**:
```
BSQ = max{Q_1, Q_2, ..., Q_n}
```

Where Q_i is the quality of solution i.

**Target**: Higher is better. Compare against:
- Known optimal (if available)
- State-of-the-art baselines
- Human expert performance

---

## 2. Secondary Metrics

### 2.1 Diversity Metrics

#### 2.1.1 Population Diversity (PD)

**Definition**: Measure of solution variety in the swarm population.

**Formula** (using pairwise distance):
```
PD = (2 / (N × (N-1))) × Σ_{i<j} d(s_i, s_j)
```

Where:
- N: Population size
- d(s_i, s_j): Distance between solutions i and j
- Distance metric depends on problem (edit distance, embedding distance, etc.)

**Alternative** (using entropy):
```
PD = -Σ_k p_k × log(p_k)
```

Where p_k is the proportion of solutions in cluster k.

**Target**: Moderate-high diversity preferred. Too low = premature convergence; too high = inefficient search.

---

#### 2.1.2 Behavioral Diversity (BD)

**Definition**: Diversity of agent behaviors or strategies.

**Formula**:
```
BD = variance({behavior_embedding(agent_i) for all agents})
```

Behavior can be characterized by:
- Action sequences
- Communication patterns
- Search strategies used

**Target**: Maintain throughout search to avoid premature convergence.

---

### 2.2 Exploration Metrics

#### 2.2.1 Coverage Ratio (CR)

**Definition**: Fraction of search space explored.

**Formula**:
```
CR = |Explored regions| / |Total searchable regions|
```

For continuous spaces, use discretization or kernel density estimation.

**Target**: Balance exploration vs exploitation.

---

#### 2.2.2 Novelty Score (NS)

**Definition**: Average novelty of discovered solutions.

**Formula**:
```
NS = (1/N) × Σ_i min_{j≠i} d(s_i, s_j)
```

Where d is a domain-appropriate distance metric.

**Target**: Higher novelty indicates better exploration.

---

#### 2.2.3 Search Entropy (SE)

**Definition**: Entropy of the search distribution.

**Formula**:
```
SE = -Σ_x P(search at x) × log P(search at x)
```

**Target**: Monitor for collapse (entropy → 0 indicates convergence).

---

### 2.3 Communication Efficiency

#### 2.3.1 Information Transfer Rate (ITR)

**Definition**: Useful information communicated per unit cost.

**Formula**:
```
ITR = (Quality improvement from communication) / (Communication cost)
```

**Target**: Higher is better. Identify communication bottlenecks.

---

#### 2.3.2 Consensus Time (CT)

**Definition**: Time for swarm to reach agreement on best solution.

**Formula**:
```
CT = min{t : variance(Q_agents(t)) < threshold}
```

**Target**: Context-dependent. Fast consensus may indicate premature convergence.

---

## 3. Quality Metrics

### 3.1 Generalization

#### 3.1.1 Cross-Validation Score (CVS)

**Definition**: Performance on held-out validation instances.

**Formula**:
```
CVS = (1/K) × Σ_k Q_best evaluated on fold k
```

Where K is number of cross-validation folds.

**Target**: High CVS indicates good generalization.

---

#### 3.1.2 Transfer Performance (TP)

**Definition**: Performance when applying discovered solution to related problems.

**Formula**:
```
TP(P_target) = Q(solution_from_P_source, P_target)
```

**Target**: Measure on problem variants with different:
- Input distributions
- Problem sizes
- Constraint sets

---

#### 3.1.3 Robustness Gap (RG)

**Definition**: Performance difference between training and test conditions.

**Formula**:
```
RG = Q_train - Q_test
```

**Target**: Lower RG indicates better generalization.

---

### 3.2 Robustness

#### 3.2.1 Perturbation Robustness (PR)

**Definition**: Performance stability under input perturbations.

**Formula**:
```
PR = E_ε[Q(solution, input + ε)]
```

Where ε is random perturbation from distribution D.

**Target**: High PR indicates robust solutions.

---

#### 3.2.2 Adversarial Robustness (AR)

**Definition**: Worst-case performance under adversarial perturbations.

**Formula**:
```
AR = min_{||ε||<δ} Q(solution, input + ε)
```

**Target**: High AR indicates adversarially robust solutions.

---

#### 3.2.3 Solution Stability (SS)

**Definition**: Variance in solution quality across multiple runs.

**Formula**:
```
SS = std({BSQ_run1, BSQ_run2, ..., BSQ_runN})
```

**Target**: Lower SS indicates more reliable system.

---

### 3.3 Solution Quality Components

#### 3.3.1 Validity Rate (VR)

**Definition**: Percentage of solutions that satisfy all constraints.

**Formula**:
```
VR = (Valid solutions / Total solutions) × 100%
```

**Target**: Higher is better. Monitor constraint satisfaction.

---

#### 3.3.2 Optimality Gap (OG)

**Definition**: Distance from known optimal solution.

**Formula**:
```
OG = |Q_best - Q_optimal| / |Q_optimal|
```

**Target**: Lower is better. Report when optimal is known.

---

## 4. Efficiency Metrics

### 4.1 Cost Metrics

#### 4.1.1 Total Compute Cost (TCC)

**Definition**: Total computational resources consumed.

**Formula**:
```
TCC = Σ (GPU_hours × GPU_cost_per_hour) + Σ (CPU_hours × CPU_cost_per_hour)
```

**Target**: Lower is better. Track for budget management.

---

#### 4.1.2 API Cost (AC)

**Definition**: Cost of external API calls (e.g., LLM APIs).

**Formula**:
```
AC = Σ (tokens_input × cost_per_input_token) + Σ (tokens_output × cost_per_output_token)
```

**Target**: Lower is better. Critical for LLM-based swarms.

---

#### 4.1.3 Total Cost (TC)

**Definition**: Combined cost of all resources.

**Formula**:
```
TC = TCC + AC + StorageCost + NetworkCost + Overhead
```

**Target**: Primary budget constraint metric.

---

### 4.2 Efficiency Ratios

#### 4.2.1 Experiments Per Dollar (EPD)

**Definition**: Number of experiments conducted per unit cost.

**Formula**:
```
EPD = Number of experiments / Total Cost ($)
```

**Target**: Higher is better. Measures cost efficiency.

---

#### 4.2.2 Quality Per Dollar (QPD)

**Definition**: Solution quality achieved per unit cost.

**Formula**:
```
QPD = BSQ / Total Cost ($)
```

**Target**: Higher is better. Key efficiency metric.

---

#### 4.2.3 Improvement Per Dollar (IPD)

**Definition**: Quality improvement per unit cost.

**Formula**:
```
IPD = (BSQ - Q_baseline) / Total Cost ($)
```

**Target**: Higher is better. Measures value of swarm approach.

---

### 4.3 Resource Utilization

#### 4.3.1 Parallel Efficiency (PE)

**Definition**: Efficiency of parallel resource usage.

**Formula**:
```
PE = Speedup / Number of agents
```

Where Speedup = T_sequential / T_parallel

**Target**: Higher is better. Ideal PE = 1 (perfect scaling).

---

#### 4.3.2 Resource Utilization (RU)

**Definition**: Percentage of allocated resources actually used.

**Formula**:
```
RU = (Actual compute used / Allocated compute) × 100%
```

**Target**: Higher is better. Identify wasted resources.

---

## 5. Metric Computation Procedures

### 5.1 Data Collection

```python
class MetricsCollector:
    def __init__(self):
        self.iteration_data = []
        self.agent_states = []
        self.solutions = []
        self.costs = []
    
    def record_iteration(self, iteration, solutions, qualities, costs):
        self.iteration_data.append({
            'iteration': iteration,
            'solutions': solutions,
            'qualities': qualities,
            'best_quality': max(qualities),
            'mean_quality': np.mean(qualities),
            'diversity': compute_diversity(solutions),
            'cost': costs
        })
    
    def compute_metrics(self):
        return {
            'improvement_rate': self._compute_ir(),
            'time_to_solution': self._compute_tts(),
            'best_quality': self._compute_bsq(),
            'diversity': self._compute_pd(),
            'total_cost': self._compute_tc()
        }
```

---

### 5.2 Statistical Reporting

For each metric, report:
- **Mean**: Average across multiple runs
- **Std**: Standard deviation
- **CI-95**: 95% confidence interval
- **Median**: Robust central tendency
- **Min/Max**: Range

```python
def report_metric(values):
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'median': np.median(values),
        'ci_95': (np.percentile(values, 2.5), np.percentile(values, 97.5)),
        'min': np.min(values),
        'max': np.max(values)
    }
```

---

### 5.3 Baseline Comparisons

Always compare against:

1. **Random Search**: Samples solutions uniformly
2. **Greedy Search**: Myopic optimization
3. **Single Agent**: Best individual agent performance
4. **Human Baseline**: Expert human performance (if available)
5. **State-of-the-Art**: Published best results

---

## 6. Metric Dashboard

### 6.1 Real-time Monitoring

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Improvement Rate | > baseline | < 50% of baseline |
| Diversity | 0.3 - 0.7 | < 0.1 or > 0.9 |
| Success Rate | > 80% | < 50% |
| Cost per Experiment | < budget | > 120% of budget |
| Validity Rate | > 95% | < 80% |

### 6.2 Summary Scorecard

```
autoconstitution Performance Scorecard
===================================
Primary Metrics:
  - Improvement Rate:        [____] / 10
  - Time to Solution:        [____] / 10
  - Best Solution Quality:   [____] / 10

Secondary Metrics:
  - Diversity:               [____] / 10
  - Exploration:             [____] / 10

Quality Metrics:
  - Generalization:          [____] / 10
  - Robustness:              [____] / 10

Efficiency Metrics:
  - Cost Efficiency:         [____] / 10
  - Resource Utilization:    [____] / 10

OVERALL SCORE: [____] / 100
```

---

## 7. Domain-Specific Adaptations

### 7.1 For Code Generation

- **Pass@k**: Probability of passing tests with k samples
- **CodeBLEU**: Syntactic and semantic similarity
- **Execution Success**: Percentage of compilable/executable code

### 7.2 For Mathematical Reasoning

- **Accuracy**: Correct answer percentage
- **Step Validity**: Validity of intermediate reasoning steps
- **Proof Completeness**: Coverage of required proof elements

### 7.3 For Scientific Discovery

- **Novelty**: Publication-worthy discoveries
- **Reproducibility**: Independent verification rate
- **Impact Citation**: Citations by other researchers

---

## 8. Implementation Checklist

- [ ] Implement metrics collection infrastructure
- [ ] Set up logging for all metric dimensions
- [ ] Create automated metric computation pipeline
- [ ] Build visualization dashboard
- [ ] Establish baseline comparisons
- [ ] Define alert thresholds
- [ ] Create reporting templates
- [ ] Document metric interpretation guidelines

---

## References

1. Eiben, A. E., & Smith, J. E. (2015). Introduction to Evolutionary Computing.
2. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
3. Deb, K., et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II.
4. Lehman, J., & Stanley, K. O. (2011). Abandoning objectives: Evolution through the search for novelty alone.
