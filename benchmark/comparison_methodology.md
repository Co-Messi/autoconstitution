# Fair Comparison Methodology: autoconstitution vs. Single-Agent AutoResearch

## Executive Summary

This document establishes a rigorous, statistically sound methodology for comparing autoconstitution (multi-agent swarm approach) against Karpathy's single-agent autoresearch paradigm. The methodology prioritizes fairness, reproducibility, and statistical validity while controlling for confounding variables.

---

## 1. Controlled Variables

### 1.1 Fixed Parameters (Must Be Identical)

| Variable | Control Strategy | Rationale |
|----------|------------------|-----------|
| **Base LLM Model** | Use identical model (e.g., GPT-4, Claude-3-Opus) with same temperature/settings | Eliminates model capability differences |
| **Context Window** | Fixed token limit for all agents/systems | Ensures equal information access |
| **Tool Access** | Same search APIs, code execution, document retrieval | Equal external resource availability |
| **Task Dataset** | Identical research questions/problems for both systems | Comparable problem difficulty |
| **Evaluation Time Limit** | Fixed maximum duration per task | Prevents indefinite running advantage |
| **Cost Budget** | Equal token/API call budget per task | Fair resource allocation |
| **Hardware Environment** | Same compute resources (CPU/memory) | Eliminates infrastructure bias |
| **Knowledge Cutoff** | Same date for training knowledge | Prevents temporal advantage |

### 1.2 Controlled Randomness

```python
# Pseudocode for controlled randomness
import random
import numpy as np

# Fixed seed for reproducibility across all experiments
SEED_BASE = 42

def get_experiment_seed(experiment_id: int) -> int:
    """Generate deterministic seed for each experiment run."""
    return SEED_BASE + experiment_id * 1000

# Apply to all stochastic components
random.seed(get_experiment_seed(exp_id))
np.random.seed(get_experiment_seed(exp_id))
# LLM temperature should be 0 for deterministic outputs when possible
```

### 1.3 Task Standardization

**Task Categories to Control:**
1. **Literature Review Tasks** - Same paper corpus, same research question
2. **Hypothesis Generation** - Same initial observations/dataset
3. **Experimental Design** - Same research objective, constraints
4. **Code Implementation** - Same algorithmic problem specification
5. **Data Analysis** - Same dataset, same analytical question
6. **Full Research Pipeline** - End-to-end from question to conclusion

### 1.4 Metric Normalization

All metrics must be normalized per-task to account for varying difficulty:

```
Normalized_Score = (Raw_Score - Task_Min) / (Task_Max - Task_Min)
```

---

## 2. Fairness Considerations

### 2.1 Architectural Fairness

#### autoconstitution-Specific Considerations
- **Agent Count**: Test with varying swarm sizes (3, 5, 7, 10 agents) to find optimal configuration
- **Communication Overhead**: Include coordination costs in total budget
- **Parallelism**: Account for wall-clock time vs. total compute time

#### Single-Agent Considerations
- **Iteration Depth**: Allow iterative refinement comparable to swarm iterations
- **Self-Reflection**: Enable equivalent self-correction mechanisms
- **Memory Management**: Provide comparable context management

### 2.2 Fair Comparison Matrix

| Scenario | autoconstitution Config | Single-Agent Config | Rationale |
|----------|---------------------|---------------------|-----------|
| Equal Compute | 5 agents × 1000 tokens | 1 agent × 5000 tokens | Same total token budget |
| Equal Time | Parallel execution | Sequential with more iterations | Same wall-clock limit |
| Equal Cost | Fixed API budget | Same API budget | Real-world cost parity |
| Optimal Config | Best-performing swarm size | Best-performing iteration count | Each at its best |

### 2.3 Bias Mitigation

**Order Bias:**
- Randomize order of system evaluation per task
- Counterbalance: half tasks Swarm first, half Single-Agent first

**Task Selection Bias:**
- Use stratified sampling across research domains
- Include both swarm-favorable and single-agent-favorable task types
- Blind evaluation: evaluators don't know which system produced output

**Implementation Bias:**
- Both systems implemented by neutral third party, OR
- Each system implemented by its proponents, then cross-validated
- Code review for fair implementation of both approaches

### 2.4 Fairness Checklist

- [ ] Both systems have access to identical external tools/APIs
- [ ] Neither system has task-specific pre-training advantage
- [ ] Evaluation criteria defined before seeing any results
- [ ] Multiple independent evaluators for subjective metrics
- [ ] Statistical tests account for multiple comparisons
- [ ] Publication bias addressed (report all experiments, not just successes)

---

## 3. Statistical Rigor

### 3.1 Experimental Design

#### Minimum Sample Sizes

| Effect Size | Power | Alpha | Min Tasks Required |
|-------------|-------|-------|-------------------|
| Small (d=0.2) | 0.80 | 0.05 | 393 tasks |
| Medium (d=0.5) | 0.80 | 0.05 | 64 tasks |
| Large (d=0.8) | 0.80 | 0.05 | 26 tasks |

**Recommendation**: Minimum 100 diverse tasks for robust comparison

#### Blocking Design

```
For each task:
  ├── Run autoconstitution (3 independent trials with different seeds)
  ├── Run Single-Agent (3 independent trials with different seeds)
  └── Evaluate all 6 outputs

Across all tasks:
  ├── Block by task domain (CS, Biology, Physics, etc.)
  ├── Block by task difficulty (Easy, Medium, Hard)
  └── Block by task type (Review, Design, Analysis)
```

### 3.2 Statistical Tests

#### Primary Comparison Tests

| Metric Type | Test | Assumptions Check |
|-------------|------|-------------------|
| Continuous (quality scores) | Paired t-test or Wilcoxon signed-rank | Normality test (Shapiro-Wilk) |
| Binary (success/failure) | McNemar's test | N/A |
| Multiple metrics | MANOVA + post-hoc Bonferroni | Homogeneity of variance |
| Time series | Mixed-effects model | Autocorrelation check |

#### Effect Size Reporting

Always report:
- **Cohen's d** for continuous outcomes
- **Odds Ratio** for binary outcomes
- **Confidence intervals** (95%) for all effect sizes

### 3.3 Handling Multiple Comparisons

```python
# Bonferroni correction for n comparisons
alpha_corrected = 0.05 / n_comparisons

# Or use False Discovery Rate (FDR) control
from statsmodels.stats.multitest import multipletests

p_values = [p1, p2, p3, ..., pn]
rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
```

### 3.4 Power Analysis

Pre-register power analysis to determine required sample size:

```python
from statsmodels.stats.power import tt_solve_power

# For medium effect size (d=0.5), power=0.8, alpha=0.05
required_n = tt_solve_power(effect_size=0.5, power=0.8, alpha=0.05)
# Result: ~64 tasks minimum
```

### 3.5 Robustness Checks

1. **Sensitivity Analysis**: Remove outliers, re-run analysis
2. **Subgroup Analysis**: Results by task domain, difficulty
3. **Non-parametric Validation**: Use rank-based tests if assumptions violated
4. **Bootstrap Confidence Intervals**: 10,000 bootstrap samples

---

## 4. Reproducibility Requirements

### 4.1 Code Reproducibility

```yaml
# reproducibility_manifest.yaml
experiment:
  name: "Swarm vs Single-Agent Comparison"
  version: "1.0.0"
  date: "2024-01-15"
  
environment:
  python_version: "3.10.12"
  requirements_file: "requirements.txt"
  docker_image: "research-benchmark:v1.0.0"
  
dependencies:
  swarm_research:
    repository: "https://github.com/.../autoconstitution"
    commit_hash: "abc123def456"
    version_tag: "v2.1.0"
  
  single_agent:
    repository: "https://github.com/karpathy/autoresearch"
    commit_hash: "xyz789uvw012"
    version_tag: "v1.3.0"
  
  llm_api:
    provider: "openai"
    model: "gpt-4-1106-preview"
    temperature: 0.0
    
seeds:
  base_seed: 42
  num_trials: 3
  
compute:
  cpu_cores: 8
  memory_gb: 32
  timeout_seconds: 3600
```

### 4.2 Data Reproducibility

**Required Artifacts:**
1. **Task Dataset**: JSON with all research problems
2. **Expected Outputs**: Gold standard answers (if available)
3. **Evaluation Rubrics**: Detailed scoring criteria
4. **Raw Results**: All system outputs (not just scores)
5. **Intermediate States**: Agent conversations, reasoning traces

**Data Organization:**
```
experiment_data/
├── tasks/
│   ├── task_001_literature_review.json
│   ├── task_002_hypothesis_gen.json
│   └── ...
├── gold_standards/
│   ├── task_001_reference_answer.json
│   └── ...
├── results/
│   ├── swarm/
│   │   ├── trial_1/
│   │   │   ├── task_001_output.json
│   │   │   └── task_001_conversation.log
│   │   └── trial_2/
│   └── single_agent/
│       └── ...
└── evaluations/
    ├── evaluator_1_scores.csv
    └── evaluator_2_scores.csv
```

### 4.3 Documentation Requirements

**Experiment Log Must Include:**
- Exact start/end timestamps
- All API calls with timestamps
- Token usage per call
- Error logs and retry attempts
- Hardware utilization metrics
- Random seeds used

### 4.4 Reproducibility Checklist

- [ ] Docker container with frozen dependencies
- [ ] All random seeds documented and set
- [ ] API versions locked
- [ ] Complete task dataset published
- [ ] Evaluation rubrics defined a priori
- [ ] Raw outputs preserved
- [ ] Analysis code published
- [ ] Statistical analysis script automated

---

## 5. Handling Randomness

### 5.1 Sources of Randomness

| Source | Mitigation Strategy |
|--------|---------------------|
| LLM sampling (temperature > 0) | Use temperature=0 when possible; otherwise multiple samples |
| Task ordering | Randomize with fixed seed |
| Network/API latency | Timeout handling, retry with exponential backoff |
| Agent initialization | Fixed seed for any random initialization |
| Search results | Mock search API with cached results |
| Concurrent execution | Deterministic scheduling where possible |

### 5.2 Randomness Control Protocol

```python
class RandomnessController:
    """Centralized randomness management for reproducibility."""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.experiment_counter = 0
        
    def get_experiment_seeds(self, experiment_id: int) -> dict:
        """Generate all seeds needed for one experiment run."""
        return {
            'task_order': self.base_seed + experiment_id * 10000,
            'llm_sampling': self.base_seed + experiment_id * 10000 + 1,
            'agent_init': self.base_seed + experiment_id * 10000 + 2,
            'data_sampling': self.base_seed + experiment_id * 10000 + 3,
        }
    
    def set_seeds(self, experiment_id: int):
        """Apply all seeds for an experiment."""
        seeds = self.get_experiment_seeds(experiment_id)
        random.seed(seeds['task_order'])
        np.random.seed(seeds['data_sampling'])
        torch.manual_seed(seeds['agent_init'])  # if using PyTorch
        # Set LLM seed through API if supported
        
    def shuffle_tasks(self, tasks: list, experiment_id: int) -> list:
        """Deterministically shuffle task order."""
        seeds = self.get_experiment_seeds(experiment_id)
        rng = random.Random(seeds['task_order'])
        shuffled = tasks.copy()
        rng.shuffle(shuffled)
        return shuffled
```

### 5.3 Stochastic LLM Handling

**Strategy 1: Deterministic Mode (Preferred)**
```python
# Use temperature=0 for maximum reproducibility
llm_config = {
    "model": "gpt-4",
    "temperature": 0.0,  # Deterministic
    "seed": 42  # If API supports it
}
```

**Strategy 2: Multiple Samples (When Temperature > 0 Required)**
```python
def run_with_multiple_samples(task, system, n_samples=5):
    """Run multiple times and aggregate results."""
    results = []
    for i in range(n_samples):
        seed = base_seed + i
        result = system.run(task, seed=seed)
        results.append(result)
    
    # Return mean and confidence interval
    return {
        'mean_score': np.mean([r.score for r in results]),
        'std_score': np.std([r.score for r in results]),
        'ci_95': np.percentile([r.score for r in results], [2.5, 97.5])
    }
```

### 5.4 Variance Quantification

Report variance components:

```
Total Variance = σ²_task + σ²_system + σ²_random

Where:
- σ²_task: Variance across different tasks
- σ²_system: Variance between Swarm vs Single-Agent
- σ²_random: Irreducible randomness
```

### 5.5 Confidence Interval Calculation

```python
from scipy import stats

def compute_confidence_interval(data, confidence=0.95):
    """Compute confidence interval for metrics."""
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of mean
    ci = stats.t.interval(confidence, n-1, loc=mean, scale=sem)
    return mean, ci
```

---

## 6. Evaluation Metrics Framework

### 6.1 Primary Metrics

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Answer Accuracy** | Correctness of final answer | Expert grading (1-10) |
| **Completeness** | Coverage of required aspects | Rubric-based scoring |
| **Citation Quality** | Proper source attribution | Precision/recall vs gold |
| **Novelty** | Original insights generated | Expert evaluation |
| **Time Efficiency** | Wall-clock time to solution | Seconds |
| **Cost Efficiency** | Total API cost | USD |
| **Token Efficiency** | Tokens used per unit quality | Tokens / accuracy |

### 6.2 Secondary Metrics

| Metric | autoconstitution Specific | Single-Agent Specific |
|--------|----------------------|----------------------|
| **Coordination Overhead** | Inter-agent communication cost | N/A |
| **Convergence Rate** | Iterations to consensus | Iterations to satisfaction |
| **Error Recovery** | Agent disagreement resolution | Self-correction effectiveness |
| **Scalability** | Performance vs agent count | Performance vs iteration count |

### 6.3 Human Evaluation Protocol

```
Evaluator Instructions:
1. Review task requirements
2. Examine system output (blinded)
3. Score on 1-10 scale for each metric
4. Provide brief justification
5. Rate confidence in assessment

Quality Control:
- Inter-rater reliability (Cohen's κ > 0.7)
- Calibration session before evaluation
- Regular spot-checks during evaluation
```

---

## 7. Experimental Protocol

### 7.1 Pre-Experiment Phase

1. **Task Curation** (Week 1)
   - Collect diverse research problems
   - Create gold standard answers
   - Validate task difficulty distribution

2. **System Implementation** (Week 2)
   - Implement both systems with shared components
   - Code review for fairness
   - Unit testing of both systems

3. **Pilot Study** (Week 3)
   - Run 10 tasks with both systems
   - Identify and fix issues
   - Validate evaluation rubrics

4. **Power Analysis** (Week 3)
   - Calculate required sample size
   - Adjust task count if needed

### 7.2 Main Experiment Phase

```
For each task in randomized_order:
    
    # Trial 1
    Set seed = base_seed + task_id * 1000 + 1
    Run autoconstitution → Record output + metrics
    Run Single-Agent → Record output + metrics
    
    # Trial 2 (independent)
    Set seed = base_seed + task_id * 1000 + 2
    Run autoconstitution → Record output + metrics
    Run Single-Agent → Record output + metrics
    
    # Trial 3 (independent)
    Set seed = base_seed + task_id * 1000 + 3
    Run autoconstitution → Record output + metrics
    Run Single-Agent → Record output + metrics
    
    # Evaluation
    Have 3 evaluators score all 6 outputs (blinded)
    Record scores + confidence

After all tasks:
    Calculate inter-rater reliability
    Resolve significant disagreements
```

### 7.3 Post-Experiment Phase

1. **Data Analysis** (Week 5)
   - Statistical tests
   - Effect size calculations
   - Subgroup analyses

2. **Robustness Checks** (Week 6)
   - Sensitivity analysis
   - Non-parametric validation
   - Bootstrap confidence intervals

3. **Documentation** (Week 7)
   - Write results paper
   - Publish code and data
   - Create reproducibility package

---

## 8. Expected Outputs

### 8.1 Results Table Template

| Metric | autoconstitution (Mean ± SD) | Single-Agent (Mean ± SD) | Difference | p-value | Cohen's d |
|--------|---------------------------|--------------------------|------------|---------|-----------|
| Accuracy | 7.8 ± 1.2 | 7.2 ± 1.5 | +0.6 | 0.023* | 0.44 |
| Time (s) | 245 ± 89 | 312 ± 102 | -67 | 0.001** | -0.70 |
| Cost ($) | 0.45 ± 0.12 | 0.38 ± 0.10 | +0.07 | 0.156 | 0.63 |

### 8.2 Visualization Requirements

1. **Box plots** showing distribution of each metric
2. **Scatter plots** of Swarm vs Single-Agent scores per task
3. **Bland-Altman plots** for agreement analysis
4. **Forest plots** for subgroup analyses
5. **Power curves** showing sample size adequacy

---

## 9. Limitations and Caveats

### 9.1 Known Limitations

1. **Task Representativeness**: Benchmark may not cover all research scenarios
2. **Implementation Quality**: Results depend on quality of both implementations
3. **Temporal Validity**: LLM capabilities change over time
4. **Evaluation Subjectivity**: Human evaluation has inherent subjectivity

### 9.2 Mitigation Strategies

- Use diverse, representative task set
- Open-source implementations for community validation
- Document LLM versions and dates
- Multiple evaluators with inter-rater reliability checks

---

## 10. Conclusion

This methodology provides a rigorous framework for comparing autoconstitution and single-agent autoresearch approaches. By controlling variables, ensuring fairness, maintaining statistical rigor, and enabling reproducibility, we can generate credible evidence about the relative strengths and weaknesses of each approach.

**Key Principles:**
1. Control everything that can be controlled
2. Randomize what cannot be controlled
3. Replicate to quantify random variation
4. Document everything for reproducibility
5. Report all results (not just positive findings)

---

## Appendix A: Task Template

```json
{
  "task_id": "TASK_001",
  "task_type": "literature_review",
  "domain": "computer_science",
  "difficulty": "medium",
  "description": "Review recent advances in transformer architectures",
  "requirements": [
    "Identify at least 5 key papers from 2022-2024",
    "Compare architectural innovations",
    "Discuss computational efficiency improvements"
  ],
  "evaluation_criteria": {
    "coverage": "Completeness of paper identification (0-10)",
    "accuracy": "Correctness of technical claims (0-10)",
    "synthesis": "Quality of comparative analysis (0-10)"
  },
  "time_limit_seconds": 1800,
  "token_budget": 10000
}
```

## Appendix B: Evaluation Rubric

| Score | Coverage | Accuracy | Synthesis |
|-------|----------|----------|-----------|
| 10 | All major works identified | No factual errors | Novel insights, clear organization |
| 7-9 | Most major works identified | Minor errors only | Good comparison, adequate organization |
| 4-6 | Some important works missing | Several errors | Superficial comparison |
| 1-3 | Major omissions | Significant errors | Poor or no synthesis |
| 0 | No relevant works | Completely wrong | No synthesis attempted |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Methodology designed for rigorous, fair comparison of AI research systems*
