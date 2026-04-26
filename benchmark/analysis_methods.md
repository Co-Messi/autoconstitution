# Statistical Analysis Methods for autoconstitution Benchmark Results

## Executive Summary

This document provides comprehensive statistical analysis methods for evaluating autoconstitution benchmark results against single-agent baselines. All methods are designed to be reproducible, statistically rigorous, and appropriate for the specific characteristics of multi-agent autoresearch systems.

---

## Table of Contents

1. [Significance Testing](#1-significance-testing)
2. [Confidence Intervals](#2-confidence-intervals)
3. [Effect Size Calculations](#3-effect-size-calculations)
4. [Multiple Comparison Correction](#4-multiple-comparison-correction)
5. [Visualization Approaches](#5-visualization-approaches)
6. [Complete Analysis Pipeline](#6-complete-analysis-pipeline)
7. [Python Implementation](#7-python-implementation)

---

## 1. Significance Testing

### 1.1 Overview of Testing Framework

Significance testing determines whether observed differences between autoconstitution and baseline systems are statistically meaningful or due to random variation.

### 1.2 Paired vs Independent Tests

#### Paired Tests (Recommended)

Use when the same random seeds are used for both systems, creating natural pairings.

**When to use**: Same hardware, same problem instances, same random seeds

**Advantages**:
- Controls for between-seed variability
- Higher statistical power
- More sensitive to true differences

**Python Implementation**:
```python
from scipy import stats

# Paired t-test (parametric)
t_stat, p_value = stats.ttest_rel(swarm_results, baseline_results)

# Wilcoxon signed-rank test (non-parametric)
w_stat, p_value = stats.wilcoxon(swarm_results, baseline_results)
```

#### Independent Tests

Use when systems are evaluated on different problem instances or seeds.

**When to use**: Different problem sets, different experimental conditions

**Python Implementation**:
```python
# Independent t-test (parametric, equal variance assumed)
t_stat, p_value = stats.ttest_ind(swarm_results, baseline_results)

# Welch's t-test (parametric, unequal variance)
t_stat, p_value = stats.ttest_ind(swarm_results, baseline_results, equal_var=False)

# Mann-Whitney U test (non-parametric)
u_stat, p_value = stats.mannwhitneyu(swarm_results, baseline_results, alternative='two-sided')
```

### 1.3 Test Selection Decision Tree

```
Are samples paired (same seeds)?
├── YES → Is distribution approximately normal?
│   ├── YES → Paired t-test
│   └── NO → Wilcoxon signed-rank test
└── NO → Are variances approximately equal?
    ├── YES → Independent t-test
    └── NO → Welch's t-test OR Mann-Whitney U
```

### 1.4 Normality Assessment

Before using parametric tests, assess normality:

```python
from scipy import stats
import numpy as np

def assess_normality(data, alpha=0.05):
    """
    Assess normality using multiple tests.
    
    Returns:
        dict: Test results and recommendation
    """
    # Shapiro-Wilk test (best for n < 50)
    if len(data) < 50:
        stat, p_value = stats.shapiro(data)
        test_name = "Shapiro-Wilk"
    else:
        # Kolmogorov-Smirnov test (for larger samples)
        stat, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        test_name = "Kolmogorov-Smirnov"
    
    # Visual assessment: Q-Q plot
    # (Generate separately using scipy.stats.probplot)
    
    is_normal = p_value > alpha
    
    return {
        'test': test_name,
        'statistic': stat,
        'p_value': p_value,
        'is_normal': is_normal,
        'recommendation': 'Parametric' if is_normal else 'Non-parametric'
    }
```

### 1.5 One-Tailed vs Two-Tailed Tests

**Two-tailed test** (default):
- H₀: μ_swarm = μ_baseline
- H₁: μ_swarm ≠ μ_baseline
- Use when direction of difference is unknown

**One-tailed test** (use with caution):
- H₀: μ_swarm ≤ μ_baseline
- H₁: μ_swarm > μ_baseline (if expecting autoconstitution to be better)
- Use only when strong prior justifies directional hypothesis
- Halves the p-value but increases false positive risk

```python
# One-tailed paired t-test (autoconstitution expected to be better)
t_stat, p_value_two_tailed = stats.ttest_rel(swarm_results, baseline_results)
p_value_one_tailed = p_value_two_tailed / 2

# Verify direction is as expected
if np.mean(swarm_results) > np.mean(baseline_results):
    print(f"One-tailed p-value: {p_value_one_tailed}")
else:
    print("Direction opposite to hypothesis; one-tailed test inappropriate")
```

### 1.6 Sample Size Requirements

| Test Type | Minimum n | Recommended n | Notes |
|-----------|-----------|---------------|-------|
| Paired t-test | 5 | 10+ | Robust to moderate non-normality with n ≥ 10 |
| Wilcoxon | 5 | 10+ | Less powerful than t-test for normal data |
| Bootstrap | 10 | 20+ | For confidence intervals and complex metrics |

### 1.7 Power Analysis

Calculate statistical power before conducting experiments:

```python
from statsmodels.stats.power import TTestPower

power_analysis = TTestPower()

# Calculate required sample size for desired power
required_n = power_analysis.solve_power(
    effect_size=0.5,  # Expected medium effect
    power=0.8,        # 80% power
    alpha=0.05        # 5% significance level
)

print(f"Required sample size: {required_n:.0f} runs per condition")

# Calculate achieved power with current sample size
achieved_power = power_analysis.solve_power(
    effect_size=0.5,
    nobs=10,          # Current sample size
    alpha=0.05
)

print(f"Achieved power with n=10: {achieved_power:.2%}")
```

---

## 2. Confidence Intervals

### 2.1 Overview

Confidence intervals (CIs) provide a range of plausible values for population parameters and indicate the precision of estimates.

### 2.2 Bootstrap Confidence Intervals

Bootstrap methods are preferred for complex metrics and when normality assumptions are questionable.

#### Percentile Bootstrap (Simplest)

```python
import numpy as np

def percentile_bootstrap_ci(data, n_bootstrap=10000, confidence=0.95):
    """
    Calculate percentile bootstrap confidence interval.
    
    Args:
        data: Array of observations
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 95%)
    
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    bootstrap_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return (lower, upper)
```

#### BCa Bootstrap (Bias-Corrected and Accelerated)

More accurate for skewed distributions and small samples.

```python
import numpy as np
from scipy import stats

def bca_bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, confidence=0.95):
    """
    Calculate BCa bootstrap confidence interval.
    
    More accurate than percentile method for skewed distributions.
    """
    n = len(data)
    theta_hat = statistic(data)
    
    # Bootstrap replicates
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))
    
    # Bias correction
    z0 = stats.norm.ppf(np.mean(np.array(bootstrap_stats) < theta_hat))
    
    # Acceleration (jackknife)
    jackknife_stats = []
    for i in range(n):
        jack_sample = np.delete(data, i)
        jackknife_stats.append(statistic(jack_sample))
    
    jack_mean = np.mean(jackknife_stats)
    numerator = np.sum((jack_mean - np.array(jackknife_stats))**3)
    denominator = 6 * (np.sum((jack_mean - np.array(jackknife_stats))**2)**(3/2))
    a = numerator / denominator if denominator != 0 else 0
    
    # Adjusted percentiles
    alpha = 1 - confidence
    z_alpha_2 = stats.norm.ppf(alpha/2)
    z_1_alpha_2 = stats.norm.ppf(1 - alpha/2)
    
    alpha_1 = stats.norm.cdf(z0 + (z0 + z_alpha_2) / (1 - a * (z0 + z_alpha_2)))
    alpha_2 = stats.norm.cdf(z0 + (z0 + z_1_alpha_2) / (1 - a * (z0 + z_1_alpha_2)))
    
    lower = np.percentile(bootstrap_stats, alpha_1 * 100)
    upper = np.percentile(bootstrap_stats, alpha_2 * 100)
    
    return (lower, upper)
```

### 2.3 Analytical Confidence Intervals

#### For Means (Normal Distribution)

```python
from scipy import stats

def mean_ci(data, confidence=0.95):
    """Calculate confidence interval for mean."""
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error of mean
    
    # t-distribution for small samples
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    
    margin = t_crit * std_err
    return (mean - margin, mean + margin)
```

#### For Proportions (Success Rates)

Use Wilson score interval for better coverage with small samples:

```python
def wilson_ci(successes, trials, confidence=0.95):
    """
    Calculate Wilson score confidence interval for proportions.
    
    More accurate than normal approximation, especially for extreme proportions.
    """
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha/2)
    
    p = successes / trials
    n = trials
    
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denominator
    margin = z * np.sqrt((p * (1-p) + z**2 / (4*n)) / n) / denominator
    
    return (centre - margin, centre + margin)
```

### 2.4 Confidence Intervals for Differences

#### Difference of Means

```python
def difference_ci(group1, group2, confidence=0.95, paired=True):
    """
    Calculate confidence interval for difference in means.
    """
    if paired:
        # Paired difference
        differences = np.array(group1) - np.array(group2)
        return mean_ci(differences, confidence)
    else:
        # Independent groups
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Welch's approximation
        se = np.sqrt(var1/n1 + var2/n2)
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        t_crit = stats.t.ppf((1 + confidence) / 2, df)
        diff = mean1 - mean2
        margin = t_crit * se
        
        return (diff - margin, diff + margin)
```

### 2.5 Confidence Intervals for Ratios

For metrics like speedup ratios (WCTT_baseline / WCTT_swarm):

```python
def ratio_ci(group1, group2, confidence=0.95, n_bootstrap=10000):
    """
    Calculate confidence interval for ratio of means using bootstrap.
    """
    ratios = []
    n = min(len(group1), len(group2))
    
    for _ in range(n_bootstrap):
        idx1 = np.random.choice(len(group1), size=n, replace=True)
        idx2 = np.random.choice(len(group2), size=n, replace=True)
        ratio = np.mean(group1[idx1]) / np.mean(group2[idx2])
        ratios.append(ratio)
    
    alpha = 1 - confidence
    lower = np.percentile(ratios, alpha/2 * 100)
    upper = np.percentile(ratios, (1 - alpha/2) * 100)
    
    return (lower, upper)
```

### 2.6 Reporting Format

Standard format for reporting with confidence intervals:

```
Metric: Mean ± StdDev [95% CI: Lower, Upper]
Example: WCTT_95: 45.2 ± 8.3 min [38.9, 53.1]
```

---

## 3. Effect Size Calculations

### 3.1 Overview

Effect sizes quantify the magnitude of differences, independent of sample size. Essential for practical significance assessment.

### 3.2 Cohen's d

Standardized mean difference. Most common effect size for continuous metrics.

```python
def cohens_d(group1, group2, paired=True):
    """
    Calculate Cohen's d effect size.
    
    Interpretation:
    - d = 0.2: Small effect
    - d = 0.5: Medium effect
    - d = 0.8: Large effect
    """
    if paired:
        # Paired Cohen's d
        differences = np.array(group1) - np.array(group2)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        d = mean_diff / std_diff
    else:
        # Independent Cohen's d
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        
        d = (mean1 - mean2) / pooled_std
    
    return d

def cohens_d_ci(d, n1, n2, confidence=0.95):
    """Calculate confidence interval for Cohen's d."""
    # Standard error approximation
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * se
    
    return (d - margin, d + margin)
```

### 3.3 Hedges' g

Bias-corrected version of Cohen's d for small samples.

```python
def hedges_g(group1, group2, paired=True):
    """
    Calculate Hedges' g (bias-corrected Cohen's d).
    
    More accurate for small samples (n < 20).
    """
    d = cohens_d(group1, group2, paired)
    
    if paired:
        n = len(group1)
        correction = 1 - 3 / (4 * n - 1)
    else:
        n1, n2 = len(group1), len(group2)
        correction = 1 - 3 / (4 * (n1 + n2) - 9)
    
    return d * correction
```

### 3.4 Cliff's Delta

Non-parametric effect size for ordinal data or when normality is violated.

```python
def cliffs_delta(group1, group2):
    """
    Calculate Cliff's delta (non-parametric effect size).
    
    Interpretation:
    - |δ| < 0.147: Negligible
    - 0.147 ≤ |δ| < 0.33: Small
    - 0.33 ≤ |δ| < 0.474: Medium
    - |δ| ≥ 0.474: Large
    
    Returns:
        delta: Effect size (range: -1 to 1)
        interpretation: String description
    """
    n1, n2 = len(group1), len(group2)
    
    # Count pairwise comparisons
    greater = sum(x > y for x in group1 for y in group2)
    less = sum(x < y for x in group1 for y in group2)
    
    delta = (greater - less) / (n1 * n2)
    
    # Interpretation
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = "negligible"
    elif abs_delta < 0.33:
        interpretation = "small"
    elif abs_delta < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return delta, interpretation
```

### 3.5 Variance Explained (R²)

Proportion of variance in outcome explained by system type.

```python
def variance_explained(group1, group2):
    """
    Calculate eta-squared (proportion of variance explained).
    
    Interpretation:
    - η² = 0.01: Small
    - η² = 0.06: Medium
    - η² = 0.14: Large
    """
    all_data = np.concatenate([group1, group2])
    grand_mean = np.mean(all_data)
    
    # Total sum of squares
    ss_total = np.sum((all_data - grand_mean)**2)
    
    # Between-group sum of squares
    mean1, mean2 = np.mean(group1), np.mean(group2)
    n1, n2 = len(group1), len(group2)
    ss_between = n1 * (mean1 - grand_mean)**2 + n2 * (mean2 - grand_mean)**2
    
    eta_squared = ss_between / ss_total
    
    return eta_squared
```

### 3.6 Probability of Superiority

Probability that a random observation from autoconstitution exceeds one from baseline.

```python
def probability_of_superiority(group1, group2):
    """
    Calculate probability that group1 > group2 for random pairs.
    
    Also known as "common language effect size" or "probability of superiority".
    
    Interpretation:
    - 0.5: No difference
    - 0.56: Small effect (equivalent to d ≈ 0.2)
    - 0.64: Medium effect (equivalent to d ≈ 0.5)
    - 0.71: Large effect (equivalent to d ≈ 0.8)
    """
    n1, n2 = len(group1), len(group2)
    
    # Count group1 > group2
    greater = sum(x > y for x in group1 for y in group2)
    
    # Add half of ties
    ties = sum(x == y for x in group1 for y in group2)
    
    prob = (greater + 0.5 * ties) / (n1 * n2)
    
    return prob
```

### 3.7 Effect Size Summary Table

| Effect Size | Type | Best For | Interpretation |
|-------------|------|----------|----------------|
| Cohen's d | Parametric | Normal data, means | 0.2/0.5/0.8 thresholds |
| Hedges' g | Parametric | Small samples | Same as Cohen's d |
| Cliff's δ | Non-parametric | Ordinal/skewed data | 0.147/0.33/0.474 thresholds |
| η² | Variance | ANOVA context | 0.01/0.06/0.14 thresholds |
| PS | Probability | General communication | 0.56/0.64/0.71 thresholds |

---

## 4. Multiple Comparison Correction

### 4.1 Overview

When testing multiple metrics or hypotheses, the familywise error rate (FWER) increases. Correction methods control this inflation.

### 4.2 Bonferroni Correction

Most conservative method. Controls FWER strongly.

```python
def bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Bonferroni correction.
    
    Formula: α_adjusted = α / m
    where m = number of comparisons
    
    Conservative but simple. Use when few comparisons (< 10).
    """
    m = len(p_values)
    alpha_adjusted = alpha / m
    
    corrected_pvalues = [min(p * m, 1.0) for p in p_values]
    
    significant = [p < alpha_adjusted for p in p_values]
    
    return {
        'alpha_original': alpha,
        'alpha_adjusted': alpha_adjusted,
        'corrected_pvalues': corrected_pvalues,
        'significant': significant
    }
```

### 4.3 Holm-Bonferroni Method

Less conservative than Bonferroni while still controlling FWER.

```python
def holm_bonferroni(p_values, alpha=0.05):
    """
    Apply Holm-Bonferroni (sequential) correction.
    
    More powerful than Bonferroni while controlling FWER.
    """
    m = len(p_values)
    
    # Sort p-values while keeping track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_pvalues = np.array(p_values)[sorted_indices]
    
    # Sequential testing
    corrected_significant = np.zeros(m, dtype=bool)
    
    for i, p in enumerate(sorted_pvalues):
        if p <= alpha / (m - i):
            corrected_significant[sorted_indices[i]] = True
        else:
            break
    
    # Corrected p-values
    corrected_pvalues = np.zeros(m)
    for i, p in enumerate(sorted_pvalues):
        corrected_pvalues[sorted_indices[i]] = min(p * (m - i), 1.0)
    
    return {
        'corrected_pvalues': corrected_pvalues.tolist(),
        'significant': corrected_significant.tolist()
    }
```

### 4.4 Benjamini-Hochberg (FDR Control)

Controls False Discovery Rate rather than FWER. More powerful for many comparisons.

```python
def benjamini_hochberg(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg correction for FDR control.
    
    Controls expected proportion of false discoveries.
    More powerful than FWER methods when many comparisons.
    
    Use when: Many comparisons (> 10), exploratory analysis
    """
    m = len(p_values)
    
    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_pvalues = np.array(p_values)[sorted_indices]
    
    # Find largest k such that p_k <= (k/m) * alpha
    k = 0
    for i, p in enumerate(sorted_pvalues):
        if p <= (i + 1) / m * alpha:
            k = i + 1
    
    # Significant if index <= k
    significant = np.zeros(m, dtype=bool)
    for i in range(k):
        significant[sorted_indices[i]] = True
    
    # Corrected p-values
    corrected_pvalues = np.zeros(m)
    for i, p in enumerate(sorted_pvalues):
        corrected_pvalues[sorted_indices[i]] = min(p * m / (i + 1), 1.0)
    
    return {
        'corrected_pvalues': corrected_pvalues.tolist(),
        'significant': significant.tolist(),
        'n_significant': k
    }
```

### 4.5 Benjamini-Yekutieli (FDR under Dependence)

More conservative FDR method that handles dependent tests.

```python
def benjamini_yekutieli(p_values, alpha=0.05):
    """
    Apply Benjamini-Yekutieli correction.
    
    Controls FDR under arbitrary dependence structures.
    More conservative than BH but more robust.
    """
    m = len(p_values)
    
    # Harmonic sum
    c_m = sum(1 / i for i in range(1, m + 1))
    
    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_pvalues = np.array(p_values)[sorted_indices]
    
    # Find significant
    k = 0
    for i, p in enumerate(sorted_pvalues):
        if p <= (i + 1) / (m * c_m) * alpha:
            k = i + 1
    
    significant = np.zeros(m, dtype=bool)
    for i in range(k):
        significant[sorted_indices[i]] = True
    
    return {
        'significant': significant.tolist(),
        'n_significant': k,
        'c_m': c_m
    }
```

### 4.6 Comparison of Methods

| Method | Controls | Best For | Power |
|--------|----------|----------|-------|
| Bonferroni | FWER | Few comparisons (< 10), confirmatory | Low |
| Holm-Bonferroni | FWER | Few comparisons, need more power | Medium |
| Benjamini-Hochberg | FDR | Many comparisons, exploratory | High |
| Benjamini-Yekutieli | FDR | Dependent tests | Medium-High |

### 4.7 Recommendation for autoconstitution Benchmarks

```python
def select_correction_method(n_comparisons, test_independence='independent'):
    """
    Recommend correction method based on scenario.
    """
    if n_comparisons <= 5:
        return "bonferroni", "Simple, conservative, appropriate for few tests"
    elif n_comparisons <= 10:
        return "holm", "More power than Bonferroni, still controls FWER"
    elif test_independence == 'dependent':
        return "benjamini_yekutieli", "Handles dependence, controls FDR"
    else:
        return "benjamini_hochberg", "Maximum power, controls FDR"
```

---

## 5. Visualization Approaches

### 5.1 Overview

Effective visualizations communicate statistical findings clearly and accurately.

### 5.2 Convergence Curves

Show performance over time for both systems.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_convergence_curves(swarm_data, baseline_data, metric_name='val_bpb'):
    """
    Plot convergence curves with confidence bands.
    
    Args:
        swarm_data: List of (time, metric) arrays for each seed
        baseline_data: List of (time, metric) arrays for each seed
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Process swarm data
    swarm_times, swarm_means, swarm_ci_lower, swarm_ci_upper = \
        process_convergence_data(swarm_data)
    
    # Process baseline data
    baseline_times, baseline_means, baseline_ci_lower, baseline_ci_upper = \
        process_convergence_data(baseline_data)
    
    # Plot
    ax.plot(swarm_times, swarm_means, 'b-', label='autoconstitution', linewidth=2)
    ax.fill_between(swarm_times, swarm_ci_lower, swarm_ci_upper, alpha=0.3, color='blue')
    
    ax.plot(baseline_times, baseline_means, 'r-', label='Single-Agent', linewidth=2)
    ax.fill_between(baseline_times, baseline_ci_lower, baseline_ci_upper, alpha=0.3, color='red')
    
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title('Convergence Comparison', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    return fig

def process_convergence_data(data_list, time_points=100):
    """Process multiple runs into mean and confidence intervals."""
    # Interpolate to common time points
    max_time = max(max(times) for times, _ in data_list)
    common_times = np.linspace(0, max_time, time_points)
    
    interpolated_values = []
    for times, values in data_list:
        interp_values = np.interp(common_times, times, values, 
                                   left=values[0], right=values[-1])
        interpolated_values.append(interp_values)
    
    values_array = np.array(interpolated_values)
    mean = np.mean(values_array, axis=0)
    ci_lower = np.percentile(values_array, 2.5, axis=0)
    ci_upper = np.percentile(values_array, 97.5, axis=0)
    
    return common_times, mean, ci_lower, ci_upper
```

### 5.3 Box Plots with Individual Points

Show distribution of final performance across seeds.

```python
def plot_comparison_boxplot(swarm_results, baseline_results, metric_name='Best val_bpb'):
    """
    Create box plot comparing two systems with individual points.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = [swarm_results, baseline_results]
    positions = [1, 2]
    labels = ['autoconstitution', 'Single-Agent']
    colors = ['lightblue', 'lightcoral']
    
    # Box plot
    bp = ax.boxplot(data, positions=positions, widths=0.6, 
                     patch_artist=True, showfliers=False)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Individual points with jitter
    for i, (results, pos) in enumerate(zip(data, positions)):
        jitter = np.random.normal(pos, 0.04, size=len(results))
        ax.scatter(jitter, results, alpha=0.6, s=50, color='black')
    
    # Mean markers
    means = [np.mean(r) for r in data]
    ax.scatter(positions, means, marker='D', s=100, color='red', 
               zorder=5, label='Mean')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title('Performance Distribution Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig
```

### 5.4 Violin Plots

Show full distribution shape.

```python
def plot_violin_comparison(swarm_results, baseline_results, metric_name='Best val_bpb'):
    """
    Create violin plot for distribution comparison.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = [swarm_results, baseline_results]
    positions = [1, 2]
    labels = ['autoconstitution', 'Single-Agent']
    
    # Violin plot
    parts = ax.violinplot(data, positions=positions, showmeans=True, 
                          showmedians=True, widths=0.7)
    
    # Color violins
    colors = ['lightblue', 'lightcoral']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    # Add individual points
    for i, (results, pos) in enumerate(zip(data, positions)):
        jitter = np.random.normal(pos, 0.04, size=len(results))
        ax.scatter(jitter, results, alpha=0.5, s=30, color='black')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title('Distribution Shape Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig
```

### 5.5 Forest Plot for Effect Sizes

Visualize effect sizes with confidence intervals.

```python
def plot_forest_plot(metrics_data):
    """
    Create forest plot for multiple metrics.
    
    Args:
        metrics_data: List of dicts with 'name', 'effect_size', 'ci_lower', 'ci_upper'
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_positions = range(len(metrics_data))
    
    for i, metric in enumerate(metrics_data):
        # Effect size point
        ax.scatter(metric['effect_size'], i, s=100, color='blue', zorder=5)
        
        # Confidence interval line
        ax.plot([metric['ci_lower'], metric['ci_upper']], [i, i], 
                'b-', linewidth=2, alpha=0.7)
        
        # Add value labels
        ax.text(metric['effect_size'] + 0.1, i, 
                f"{metric['effect_size']:.2f}", 
                va='center', fontsize=9)
    
    # Reference lines
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels([m['name'] for m in metrics_data], fontsize=10)
    ax.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
    ax.set_title('Effect Sizes with 95% Confidence Intervals', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legend for effect size thresholds
    ax.text(0.02, len(metrics_data) - 0.5, 'Small (0.2)', fontsize=8, color='gray')
    ax.text(0.52, len(metrics_data) - 0.5, 'Medium (0.5)', fontsize=8, color='gray')
    ax.text(0.82, len(metrics_data) - 0.5, 'Large (0.8)', fontsize=8, color='gray')
    
    return fig
```

### 5.6 P-Value Heatmap

Visualize significance across multiple metrics and conditions.

```python
import seaborn as sns

def plot_significance_heatmap(pvalue_matrix, row_labels, col_labels):
    """
    Create heatmap of p-values across metrics and conditions.
    
    Args:
        pvalue_matrix: 2D array of p-values
        row_labels: Labels for rows
        col_labels: Labels for columns
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create significance markers
    sig_matrix = pvalue_matrix < 0.05
    
    # Plot heatmap
    sns.heatmap(pvalue_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                xticklabels=col_labels, yticklabels=row_labels,
                vmin=0, vmax=0.1, cbar_kws={'label': 'p-value'},
                ax=ax)
    
    # Add significance indicators
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if sig_matrix[i, j]:
                ax.text(j + 0.5, i + 0.85, '*', ha='center', 
                       fontsize=16, color='black', fontweight='bold')
    
    ax.set_title('Significance Heatmap (* = p < 0.05)', fontsize=14)
    plt.tight_layout()
    
    return fig
```

### 5.7 Time-to-Target Bar Chart

Compare WCTT metrics side-by-side.

```python
def plot_ttt_comparison(swarm_ttt, baseline_ttt, targets=['90%', '95%', '99%']):
    """
    Create grouped bar chart for time-to-target comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(targets))
    width = 0.35
    
    # Calculate means and CIs
    swarm_means = [np.mean(t) for t in swarm_ttt]
    swarm_ci = [percentile_bootstrap_ci(t) for t in swarm_ttt]
    swarm_errors = [(m - ci[0], ci[1] - m) for m, ci in zip(swarm_means, swarm_ci)]
    swarm_errors = np.array(swarm_errors).T
    
    baseline_means = [np.mean(t) for t in baseline_ttt]
    baseline_ci = [percentile_bootstrap_ci(t) for t in baseline_ttt]
    baseline_errors = [(m - ci[0], ci[1] - m) for m, ci in zip(baseline_means, baseline_ci)]
    baseline_errors = np.array(baseline_errors).T
    
    # Plot bars
    bars1 = ax.bar(x - width/2, swarm_means, width, label='autoconstitution',
                   yerr=swarm_errors, capsize=5, color='lightblue', edgecolor='blue')
    bars2 = ax.bar(x + width/2, baseline_means, width, label='Single-Agent',
                   yerr=baseline_errors, capsize=5, color='lightcoral', edgecolor='red')
    
    ax.set_xlabel('Target Performance', fontsize=12)
    ax.set_ylabel('Time to Target (minutes)', fontsize=12)
    ax.set_title('Time-to-Target Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add speedup annotations
    for i, (s_mean, b_mean) in enumerate(zip(swarm_means, baseline_means)):
        speedup = (b_mean - s_mean) / b_mean * 100
        ax.annotate(f'{speedup:+.1f}%', 
                   xy=(i, max(s_mean, b_mean) + 5),
                   ha='center', fontsize=9, fontweight='bold',
                   color='green' if speedup > 0 else 'red')
    
    return fig
```

### 5.8 Exploration-Exploitation Visualization

Show search behavior over time.

```python
def plot_exploration_dynamics(exploration_ratios, timestamps, labels):
    """
    Plot exploration ratio over time for multiple systems.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green']
    
    for ratios, times, label, color in zip(exploration_ratios, timestamps, labels, colors):
        ax.plot(times, ratios, label=label, color=color, linewidth=2)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Balanced')
    ax.fill_between([min(min(t) for t in timestamps), max(max(t) for t in timestamps)],
                    0.3, 0.7, alpha=0.1, color='green', label='Optimal range')
    
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Exploration Ratio', fontsize=12)
    ax.set_title('Exploration-Exploitation Dynamics', fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig
```

---

## 6. Complete Analysis Pipeline

### 6.1 Master Analysis Function

```python
def analyze_benchmark_results(swarm_data, baseline_data, metric_names):
    """
    Complete statistical analysis pipeline for benchmark comparison.
    
    Args:
        swarm_data: Dict of metric_name -> list of values
        baseline_data: Dict of metric_name -> list of values
        metric_names: List of metric names to analyze
    
    Returns:
        Comprehensive analysis results
    """
    results = {}
    
    for metric in metric_names:
        print(f"\n{'='*60}")
        print(f"Analyzing: {metric}")
        print('='*60)
        
        swarm_vals = swarm_data[metric]
        baseline_vals = baseline_data[metric]
        
        metric_results = {}
        
        # 1. Descriptive statistics
        metric_results['descriptive'] = {
            'swarm_mean': np.mean(swarm_vals),
            'swarm_std': np.std(swarm_vals, ddof=1),
            'swarm_ci': percentile_bootstrap_ci(swarm_vals),
            'baseline_mean': np.mean(baseline_vals),
            'baseline_std': np.std(baseline_vals, ddof=1),
            'baseline_ci': percentile_bootstrap_ci(baseline_vals)
        }
        
        print(f"autoconstitution: {metric_results['descriptive']['swarm_mean']:.3f} ± "
              f"{metric_results['descriptive']['swarm_std']:.3f} "
              f"[95% CI: {metric_results['descriptive']['swarm_ci'][0]:.3f}, "
              f"{metric_results['descriptive']['swarm_ci'][1]:.3f}]")
        
        print(f"Single-Agent:  {metric_results['descriptive']['baseline_mean']:.3f} ± "
              f"{metric_results['descriptive']['baseline_std']:.3f} "
              f"[95% CI: {metric_results['descriptive']['baseline_ci'][0]:.3f}, "
              f"{metric_results['descriptive']['baseline_ci'][1]:.3f}]")
        
        # 2. Significance testing
        # Check normality
        normality_swarm = assess_normality(swarm_vals)
        normality_baseline = assess_normality(baseline_vals)
        
        if normality_swarm['is_normal'] and normality_baseline['is_normal']:
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(swarm_vals, baseline_vals)
            test_name = 'Paired t-test'
        else:
            # Wilcoxon signed-rank test
            t_stat, p_value = stats.wilcoxon(swarm_vals, baseline_vals)
            test_name = 'Wilcoxon signed-rank'
        
        metric_results['significance'] = {
            'test': test_name,
            'statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        print(f"\n{test_name}:")
        print(f"  Statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
        
        # 3. Effect size
        d = cohens_d(swarm_vals, baseline_vals, paired=True)
        d_ci = cohens_d_ci(d, len(swarm_vals), len(baseline_vals))
        
        metric_results['effect_size'] = {
            'cohens_d': d,
            'cohens_d_ci': d_ci,
            'hedges_g': hedges_g(swarm_vals, baseline_vals, paired=True),
            'variance_explained': variance_explained(swarm_vals, baseline_vals),
            'probability_superiority': probability_of_superiority(swarm_vals, baseline_vals)
        }
        
        print(f"\nEffect Sizes:")
        print(f"  Cohen's d: {d:.3f} [95% CI: {d_ci[0]:.3f}, {d_ci[1]:.3f}]")
        print(f"  Hedges' g: {metric_results['effect_size']['hedges_g']:.3f}")
        print(f"  Variance explained (η²): {metric_results['effect_size']['variance_explained']:.3f}")
        print(f"  P(Swarm > Baseline): {metric_results['effect_size']['probability_superiority']:.3f}")
        
        # 4. Difference CI
        diff_ci = difference_ci(swarm_vals, baseline_vals, paired=True)
        metric_results['difference_ci'] = diff_ci
        
        print(f"\nDifference CI: [{diff_ci[0]:.3f}, {diff_ci[1]:.3f}]")
        
        results[metric] = metric_results
    
    # 5. Multiple comparison correction
    p_values = [results[m]['significance']['p_value'] for m in metric_names]
    correction_result = benjamini_hochberg(p_values)
    
    print(f"\n{'='*60}")
    print("Multiple Comparison Correction (Benjamini-Hochberg)")
    print('='*60)
    for i, metric in enumerate(metric_names):
        results[metric]['significance']['p_value_corrected'] = correction_result['corrected_pvalues'][i]
        results[metric]['significance']['significant_corrected'] = correction_result['significant'][i]
        print(f"{metric}:")
        print(f"  Original p: {p_values[i]:.4f}")
        print(f"  Corrected p: {correction_result['corrected_pvalues'][i]:.4f}")
        print(f"  Significant (corrected): {'Yes' if correction_result['significant'][i] else 'No'}")
    
    return results
```

---

## 7. Python Implementation

### 7.1 Complete Module

```python
"""
statistical_analysis.py

Complete statistical analysis module for autoconstitution benchmarks.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Union, Optional
import warnings

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2024, 31415, 271828, 161803, 999999]
DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_SAMPLES = 10000

# Effect size thresholds
COHENS_D_THRESHOLDS = {'small': 0.2, 'medium': 0.5, 'large': 0.8}
CLIFFS_DELTA_THRESHOLDS = {'negligible': 0.147, 'small': 0.33, 'medium': 0.474}
ETA_SQUARED_THRESHOLDS = {'small': 0.01, 'medium': 0.06, 'large': 0.14}

# ============================================================================
# SIGNIFICANCE TESTING
# ============================================================================

class SignificanceTester:
    """Class for conducting significance tests."""
    
    @staticmethod
    def paired_test(group1: np.ndarray, group2: np.ndarray, 
                    alpha: float = 0.05) -> Dict:
        """Conduct paired significance test."""
        # Check normality
        _, p_normal1 = stats.shapiro(group1) if len(group1) < 50 else (0, 1)
        _, p_normal2 = stats.shapiro(group2) if len(group2) < 50 else (0, 1)
        
        is_normal = p_normal1 > alpha and p_normal2 > alpha
        
        if is_normal:
            stat, p_value = stats.ttest_rel(group1, group2)
            test_name = 'paired_ttest'
        else:
            stat, p_value = stats.wilcoxon(group1, group2)
            test_name = 'wilcoxon'
        
        return {
            'test': test_name,
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'is_normal': is_normal
        }
    
    @staticmethod
    def independent_test(group1: np.ndarray, group2: np.ndarray,
                         alpha: float = 0.05) -> Dict:
        """Conduct independent significance test."""
        # Check equal variance
        _, p_var = stats.levene(group1, group2)
        equal_var = p_var > alpha
        
        stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
        
        return {
            'test': 'welch_ttest' if not equal_var else 'independent_ttest',
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'equal_variance': equal_var
        }

# ============================================================================
# CONFIDENCE INTERVALS
# ============================================================================

class ConfidenceIntervalCalculator:
    """Class for calculating confidence intervals."""
    
    @staticmethod
    def percentile_bootstrap(data: np.ndarray, 
                             confidence: float = DEFAULT_CONFIDENCE,
                             n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES) -> Tuple[float, float]:
        """Calculate percentile bootstrap CI."""
        bootstrap_stats = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, alpha/2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
        
        return (float(lower), float(upper))
    
    @staticmethod
    def mean_ci(data: np.ndarray, confidence: float = DEFAULT_CONFIDENCE) -> Tuple[float, float]:
        """Calculate analytical CI for mean."""
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_crit * std_err
        
        return (float(mean - margin), float(mean + margin))
    
    @staticmethod
    def wilson_proportion(successes: int, trials: int,
                          confidence: float = DEFAULT_CONFIDENCE) -> Tuple[float, float]:
        """Calculate Wilson score CI for proportions."""
        alpha = 1 - confidence
        z = stats.norm.ppf(1 - alpha/2)
        
        p = successes / trials
        n = trials
        
        denominator = 1 + z**2 / n
        centre = (p + z**2 / (2*n)) / denominator
        margin = z * np.sqrt((p * (1-p) + z**2 / (4*n)) / n) / denominator
        
        return (float(centre - margin), float(centre + margin))

# ============================================================================
# EFFECT SIZES
# ============================================================================

class EffectSizeCalculator:
    """Class for calculating effect sizes."""
    
    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray, 
                 paired: bool = True) -> Dict:
        """Calculate Cohen's d and related metrics."""
        if paired:
            differences = group1 - group2
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            d = mean_diff / std_diff if std_diff > 0 else 0
        else:
            mean1, mean2 = np.mean(group1), np.mean(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # Interpretation
        abs_d = abs(d)
        if abs_d < COHENS_D_THRESHOLDS['small']:
            interpretation = 'negligible'
        elif abs_d < COHENS_D_THRESHOLDS['medium']:
            interpretation = 'small'
        elif abs_d < COHENS_D_THRESHOLDS['large']:
            interpretation = 'medium'
        else:
            interpretation = 'large'
        
        return {
            'cohens_d': float(d),
            'absolute_d': float(abs_d),
            'interpretation': interpretation
        }
    
    @staticmethod
    def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> Dict:
        """Calculate Cliff's delta."""
        n1, n2 = len(group1), len(group2)
        
        greater = sum(x > y for x in group1 for y in group2)
        less = sum(x < y for x in group1 for y in group2)
        
        delta = (greater - less) / (n1 * n2)
        
        # Interpretation
        abs_delta = abs(delta)
        if abs_delta < CLIFFS_DELTA_THRESHOLDS['negligible']:
            interpretation = 'negligible'
        elif abs_delta < CLIFFS_DELTA_THRESHOLDS['small']:
            interpretation = 'small'
        elif abs_delta < CLIFFS_DELTA_THRESHOLDS['medium']:
            interpretation = 'medium'
        else:
            interpretation = 'large'
        
        return {
            'cliffs_delta': float(delta),
            'absolute_delta': float(abs_delta),
            'interpretation': interpretation
        }

# ============================================================================
# MULTIPLE COMPARISON CORRECTION
# ============================================================================

class MultipleComparisonCorrection:
    """Class for multiple comparison correction methods."""
    
    @staticmethod
    def bonferroni(p_values: List[float], alpha: float = 0.05) -> Dict:
        """Apply Bonferroni correction."""
        m = len(p_values)
        alpha_adjusted = alpha / m
        corrected = [min(p * m, 1.0) for p in p_values]
        
        return {
            'method': 'bonferroni',
            'alpha_original': alpha,
            'alpha_adjusted': alpha_adjusted,
            'corrected_pvalues': corrected,
            'significant': [p < alpha_adjusted for p in p_values]
        }
    
    @staticmethod
    def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> Dict:
        """Apply Benjamini-Hochberg correction."""
        m = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_pvalues = np.array(p_values)[sorted_indices]
        
        # Find significant
        k = 0
        for i, p in enumerate(sorted_pvalues):
            if p <= (i + 1) / m * alpha:
                k = i + 1
        
        significant = np.zeros(m, dtype=bool)
        for i in range(k):
            significant[sorted_indices[i]] = True
        
        # Corrected p-values
        corrected = np.zeros(m)
        for i, p in enumerate(sorted_pvalues):
            corrected[sorted_indices[i]] = min(p * m / (i + 1), 1.0)
        
        return {
            'method': 'benjamini_hochberg',
            'alpha': alpha,
            'corrected_pvalues': corrected.tolist(),
            'significant': significant.tolist(),
            'n_significant': k
        }

# ============================================================================
# MAIN ANALYSIS CLASS
# ============================================================================

class BenchmarkAnalyzer:
    """Main class for benchmark statistical analysis."""
    
    def __init__(self, confidence: float = DEFAULT_CONFIDENCE):
        self.confidence = confidence
        self.significance_tester = SignificanceTester()
        self.ci_calculator = ConfidenceIntervalCalculator()
        self.effect_calculator = EffectSizeCalculator()
        self.correction = MultipleComparisonCorrection()
    
    def analyze_metric(self, swarm_data: np.ndarray, 
                       baseline_data: np.ndarray,
                       metric_name: str = 'metric') -> Dict:
        """Analyze a single metric comprehensively."""
        result = {
            'metric_name': metric_name,
            'n_samples': len(swarm_data)
        }
        
        # Descriptive statistics
        result['descriptive'] = {
            'swarm': {
                'mean': float(np.mean(swarm_data)),
                'std': float(np.std(swarm_data, ddof=1)),
                'median': float(np.median(swarm_data)),
                'ci': self.ci_calculator.percentile_bootstrap(swarm_data, self.confidence)
            },
            'baseline': {
                'mean': float(np.mean(baseline_data)),
                'std': float(np.std(baseline_data, ddof=1)),
                'median': float(np.median(baseline_data)),
                'ci': self.ci_calculator.percentile_bootstrap(baseline_data, self.confidence)
            }
        }
        
        # Significance test
        result['significance'] = self.significance_tester.paired_test(
            swarm_data, baseline_data
        )
        
        # Effect sizes
        result['effect_size'] = {
            **self.effect_calculator.cohens_d(swarm_data, baseline_data, paired=True),
            **self.effect_calculator.cliffs_delta(swarm_data, baseline_data)
        }
        
        return result
    
    def analyze_all_metrics(self, swarm_data: Dict[str, np.ndarray],
                           baseline_data: Dict[str, np.ndarray]) -> Dict:
        """Analyze all metrics with multiple comparison correction."""
        metric_names = list(swarm_data.keys())
        
        # Analyze each metric
        results = {}
        p_values = []
        
        for metric in metric_names:
            results[metric] = self.analyze_metric(
                swarm_data[metric], baseline_data[metric], metric
            )
            p_values.append(results[metric]['significance']['p_value'])
        
        # Apply multiple comparison correction
        correction = self.correction.benjamini_hochberg(p_values)
        
        for i, metric in enumerate(metric_names):
            results[metric]['significance']['p_value_corrected'] = \
                correction['corrected_pvalues'][i]
            results[metric]['significance']['significant_corrected'] = \
                correction['significant'][i]
        
        results['_correction'] = correction
        
        return results

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_result(result: Dict, metric_name: str = None) -> str:
    """Format analysis result as readable string."""
    name = metric_name or result.get('metric_name', 'Metric')
    
    desc = result['descriptive']
    sig = result['significance']
    eff = result['effect_size']
    
    output = f"\n{'='*60}\n"
    output += f"Results for: {name}\n"
    output += '='*60 + '\n'
    
    output += f"autoconstitution: {desc['swarm']['mean']:.4f} ± {desc['swarm']['std']:.4f}\n"
    output += f"  95% CI: [{desc['swarm']['ci'][0]:.4f}, {desc['swarm']['ci'][1]:.4f}]\n"
    
    output += f"Single-Agent:  {desc['baseline']['mean']:.4f} ± {desc['baseline']['std']:.4f}\n"
    output += f"  95% CI: [{desc['baseline']['ci'][0]:.4f}, {desc['baseline']['ci'][1]:.4f}]\n"
    
    output += f"\nSignificance Test ({sig['test']}):\n"
    output += f"  p-value: {sig['p_value']:.4f}"
    if 'p_value_corrected' in sig:
        output += f" (corrected: {sig['p_value_corrected']:.4f})"
    output += '\n'
    output += f"  Significant: {'Yes' if sig.get('significant_corrected', sig['significant']) else 'No'}\n"
    
    output += f"\nEffect Size:\n"
    output += f"  Cohen's d: {eff['cohens_d']:.3f} ({eff['interpretation']})\n"
    output += f"  Cliff's delta: {eff['cliffs_delta']:.3f}\n"
    
    return output

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # Example data
    np.random.seed(42)
    
    swarm_wctt = np.array([42.5, 45.2, 41.8, 44.1, 43.6, 46.0, 42.9, 44.5, 43.2, 45.8])
    baseline_wctt = np.array([52.3, 54.1, 51.8, 53.5, 52.9, 55.2, 53.1, 54.8, 52.6, 54.9])
    
    swarm_bfp = np.array([0.985, 0.982, 0.987, 0.983, 0.986, 0.981, 0.984, 0.982, 0.985, 0.983])
    baseline_bfp = np.array([0.991, 0.993, 0.990, 0.992, 0.991, 0.994, 0.992, 0.993, 0.991, 0.992])
    
    # Create analyzer
    analyzer = BenchmarkAnalyzer(confidence=0.95)
    
    # Analyze all metrics
    swarm_data = {'WCTT_95': swarm_wctt, 'Best_BPB': swarm_bfp}
    baseline_data = {'WCTT_95': baseline_wctt, 'Best_BPB': baseline_bfp}
    
    results = analyzer.analyze_all_metrics(swarm_data, baseline_data)
    
    # Print results
    for metric, result in results.items():
        if not metric.startswith('_'):
            print(format_result(result, metric))
```

---

## Appendix A: Quick Reference

### Statistical Test Selection

| Scenario | Test | Python Function |
|----------|------|-----------------|
| Paired, normal | Paired t-test | `scipy.stats.ttest_rel` |
| Paired, non-normal | Wilcoxon signed-rank | `scipy.stats.wilcoxon` |
| Independent, equal variance | Independent t-test | `scipy.stats.ttest_ind(equal_var=True)` |
| Independent, unequal variance | Welch's t-test | `scipy.stats.ttest_ind(equal_var=False)` |
| Independent, non-normal | Mann-Whitney U | `scipy.stats.mannwhitneyu` |

### Effect Size Interpretation

| Effect Size | Negligible | Small | Medium | Large |
|-------------|------------|-------|--------|-------|
| Cohen's d | < 0.2 | 0.2-0.5 | 0.5-0.8 | ≥ 0.8 |
| Cliff's δ | < 0.147 | 0.147-0.33 | 0.33-0.474 | ≥ 0.474 |
| η² | < 0.01 | 0.01-0.06 | 0.06-0.14 | ≥ 0.14 |

### Correction Method Selection

| # Comparisons | Method | Use Case |
|---------------|--------|----------|
| 1-5 | Bonferroni | Confirmatory, few tests |
| 6-10 | Holm-Bonferroni | More power needed |
| 11+ | Benjamini-Hochberg | Exploratory, many tests |
| Any (dependent) | Benjamini-Yekutieli | Known dependence |

---

*Document Version: 1.0*
*For autoconstitution Benchmark Statistical Analysis*
