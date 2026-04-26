# autoconstitution API Economics Report
## Cost Analysis & Budget-Aware Orchestration for Multi-Agent Systems

**Report Date:** 2025  
**Scope:** Cost structure analysis for running 100-agent autoconstitution runs across major AI API providers

---

## Executive Summary

This report analyzes the cost structure of running multi-agent (swarm) research systems at scale across three major AI API providers: **Kimi (Moonshot AI)**, **Anthropic Claude**, and **OpenAI**. 

### Key Findings

| Metric | Value |
|--------|-------|
| **Cheapest 100-agent overnight run** | $99.84 (GPT-4.1 Mini Batch) |
| **Best value with native swarm support** | $307.20 (Kimi K2.5) |
| **Most expensive option** | $1,440.00 (Claude Opus 4.5 Batch) |
| **Minimum student/indie budget** | $20-50 for pilot experiments |
| **Rate limit bottleneck** | Claude Tier 1: 160 hours for 8-hour workload |

---

## 1. Provider Pricing Overview

### 1.1 Kimi K2.5 (Moonshot AI)

| Metric | Value |
|--------|-------|
| Input Cost | $0.60 / million tokens |
| Output Cost | $2.50 / million tokens |
| Context Window | 256K tokens |
| Native Agent Swarm | Yes (100 parallel agents) |
| Rate Limits (Est.) | 1,000 RPM / 500K TPM |

**Key Advantage:** Native support for 100 parallel agents with Agent Swarm mode, delivering 4.5x execution speedup on parallelizable tasks.

### 1.2 Anthropic Claude

| Model | Input | Output | Context | RPM (Tier 1) | TPM In (Tier 1) | TPM Out (Tier 1) |
|-------|-------|--------|---------|--------------|-----------------|------------------|
| Haiku 3.5 | $0.80 | $4.00 | 200K | 50 | 50,000 | 10,000 |
| Sonnet 4.5 | $3.00 | $15.00 | 1M | 50 | 30,000 | 8,000 |
| Opus 4.5 | $5.00 | $25.00 | 200K | 50 | 20,000 | 4,000 |

**Batch Discount:** 50% off for asynchronous processing

**Key Advantage:** 1M token context window (Sonnet 4.5), excellent for long-document analysis.

### 1.3 OpenAI

| Model | Input | Output | Cached Input | Context | RPM (Tier 1) | TPM (Tier 1) |
|-------|-------|--------|--------------|---------|--------------|--------------|
| GPT-5 | $1.25 | $10.00 | $0.125 (90% off) | 128K | 500 | 500K |
| GPT-5 Mini | $0.25 | $2.00 | $0.025 (90% off) | 128K | 1,000 | 1M |
| GPT-4.1 | $2.00 | $8.00 | $0.50 (75% off) | 128K | 1,000 | 1M |
| GPT-4.1 Mini | $0.40 | $1.60 | $0.10 (75% off) | 128K | 5,000 | 5M |

**Batch Discount:** 50% off for Batch API

**Key Advantage:** Aggressive caching discounts (up to 90% off cached input), high rate limits on mini models.

---

## 2. 100-Agent Overnight Cost Analysis

### 2.1 Workload Assumptions

For a typical autoconstitution experiment:

| Parameter | Value |
|-----------|-------|
| Number of Agents | 100 |
| Duration | 8 hours (overnight) |
| Requests per Agent per Minute | 2 |
| Average Input Tokens per Request | 2,000 |
| Average Output Tokens per Request | 800 |

**Total Workload:**
- Total Requests: 96,000
- Total Input Tokens: 192,000,000
- Total Output Tokens: 76,800,000
- Total Tokens: 268,800,000

### 2.2 Cost Comparison

| Provider/Model | Standard Cost | Optimized Cost | Cost per Agent |
|----------------|---------------|----------------|----------------|
| **Kimi K2.5** | $307.20 | $307.20 | $3.07 |
| Claude Haiku 3.5 (Batch) | $460.80 | **$230.40** | $2.30 |
| Claude Sonnet 4.5 (Batch) | $1,728.00 | **$864.00** | $8.64 |
| Claude Opus 4.5 (Batch) | $2,880.00 | **$1,440.00** | $14.40 |
| GPT-5 (Batch) | $1,008.00 | **$504.00** | $5.04 |
| **GPT-5 Mini (Batch)** | $201.60 | **$100.80** | **$1.01** |
| GPT-4.1 (Batch) | $998.40 | **$499.20** | $4.99 |
| **GPT-4.1 Mini (Batch)** | $199.68 | **$99.84** | **$1.00** |

### 2.3 Cost Ranking (Optimized)

1. **GPT-4.1 Mini (Batch):** $99.84 (0.3x vs Kimi)
2. **GPT-5 Mini (Batch):** $100.80 (0.3x vs Kimi)
3. **Claude Haiku 3.5 (Batch):** $230.40 (0.8x vs Kimi)
4. **Kimi K2.5:** $307.20 (baseline)
5. **GPT-5 (Batch):** $504.00 (1.6x vs Kimi)
6. **GPT-4.1 (Batch):** $499.20 (1.6x vs Kimi)
7. **Claude Sonnet 4.5 (Batch):** $864.00 (2.8x vs Kimi)
8. **Claude Opus 4.5 (Batch):** $1,440.00 (4.7x vs Kimi)

---

## 3. Rate Limit Analysis

### 3.1 Required Throughput

For 100 agents at 2 requests/minute each:

| Metric | Required |
|--------|----------|
| Requests per Minute | 200 |
| Input Tokens per Minute | 400,000 |
| Output Tokens per Minute | 160,000 |

### 3.2 Rate Limit Feasibility at Tier 1

| Provider | RPM | Input TPM | Output TPM | Status |
|----------|-----|-----------|------------|--------|
| Kimi K2.5 (Est.) | 1,000 | 500,000 | 200,000 | OK |
| Claude Sonnet 4.5 | 50 | 30,000 | 8,000 | Limited |
| Claude Haiku 3.5 | 50 | 50,000 | 10,000 | Limited |
| GPT-5 | 500 | 500,000 | 200,000 | OK |
| GPT-5 Mini | 1,000 | 1,000,000 | 400,000 | OK |
| GPT-4.1 Mini | 5,000 | 5,000,000 | 2,000,000 | OK |

### 3.3 Time Required at Tier 1 Rate Limits

| Provider | Effective RPM | Time for 8-Hour Workload | Bottleneck |
|----------|---------------|--------------------------|------------|
| Kimi K2.5 | 250 | 6.4 hours | TPM |
| Claude Sonnet 4.5 | 10 | **160 hours** | TPM |
| Claude Haiku 3.5 | 12.5 | **128 hours** | TPM |
| GPT-5 | 250 | 6.4 hours | TPM |
| GPT-5 Mini | 500 | 3.2 hours | TPM |
| GPT-4.1 Mini | 2,500 | 0.6 hours | TPM |

**Critical Finding:** Claude Tier 1 rate limits make 100-agent experiments impractical without tier upgrades. A workload that should take 8 hours would require 160 hours (nearly a week) with Claude Sonnet 4.5 at Tier 1.

---

## 4. Minimum Viable Experiment Budgets

### 4.1 Experiment Size Cost Matrix

| Experiment | Agents | Hours | Total Tokens | Kimi | GPT-Mini | Claude Haiku |
|------------|--------|-------|--------------|------|----------|--------------|
| Tiny Pilot | 5 | 2 | 1.7M | $1.92 | $0.62 | $1.44 |
| Small Test | 10 | 4 | 6.7M | $7.68 | $2.50 | $5.76 |
| Medium Study | 25 | 6 | 50.4M | $57.60 | $18.72 | $43.20 |
| Full Overnight | 100 | 8 | 268.8M | $307.20 | $99.84 | $230.40 |
| Extended Run | 100 | 24 | 806.4M | $921.60 | $299.52 | $691.20 |

### 4.2 Recommended Budget Tiers

#### Students / Indie Researchers
- **Budget:** $20-50
- **Capability:** Tiny Pilot + Small Test experiments
- **Recommended Provider:** GPT-4.1 Mini or GPT-5 Mini (Batch)
- **Use Case:** Learning, prototyping, validation

#### Serious Researchers
- **Budget:** $100-200
- **Capability:** Medium Study (25 agents, 6 hours)
- **Recommended Provider:** Kimi K2.5 or GPT-5 Mini
- **Use Case:** Parameter sweeps, comparative studies

#### Full Swarm Research
- **Budget:** $300-500
- **Capability:** Full Overnight runs (100 agents, 8 hours)
- **Recommended Provider:** Kimi K2.5 (native swarm support)
- **Use Case:** Production research, comprehensive experiments

---

## 5. Cost Scaling with Experiment Complexity

### 5.1 Complexity Scenarios (50 agents, 4 hours)

| Complexity | Input/Req | Output/Req | Total Tokens | Kimi Cost | GPT-Mini Cost |
|------------|-----------|------------|--------------|-----------|---------------|
| Simple (Chat-like) | 500 | 300 | 19.2M | $25.20 | $8.16 |
| Standard Research | 2,000 | 800 | 91.2M | $91.20 | $29.76 |
| Complex Analysis | 4,000 | 2,250 | 210M | $228.60 | $74.40 |
| Deep Research | 8,000 | 6,000 | 456M | $547.20 | $177.60 |
| Exhaustive Study | 15,000 | 15,000 | 960M | $1,260.00 | $408.00 |

### 5.2 Scaling Observations

- **50x cost increase** from Simple to Exhaustive complexity
- Input context size has **linear cost impact**
- Reasoning/thinking models add **1.5-2.5x output cost multiplier**
- Tool calls add ~500 tokens overhead each
- Most expensive factor: **Large context + reasoning combined**

---

## 6. Cost-Aware Orchestration Design

### 6.1 Adaptive Research Swarm (ARS) Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT ROLE HIERARCHY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ORCHESTRATOR AGENT (1 per swarm)                                │
│     └── Model: Kimi K2.5 or Claude Sonnet 4.5                    │
│     └── Role: Task decomposition, coordination, quality control  │
│     └── Cost: ~$5-10 per 8-hour run                              │
│                                                                   │
│  RESEARCHER AGENTS (20% of swarm)                                │
│     └── Model: Kimi K2.5 or GPT-4.1                              │
│     └── Role: Complex analysis, hypothesis generation            │
│     └── Cost: ~$50-100 per 8-hour run (20 agents)                │
│                                                                   │
│  WORKER AGENTS (60% of swarm)                                    │
│     └── Model: GPT-4.1 Mini or GPT-5 Mini (Batch)                │
│     └── Role: Data processing, pattern matching, summarization   │
│     └── Cost: ~$30-60 per 8-hour run (60 agents)                 │
│                                                                   │
│  MONITOR AGENTS (20% of swarm)                                   │
│     └── Model: GPT-4.1 Mini or Claude Haiku (Batch)              │
│     └── Role: Metrics collection, anomaly detection, reporting   │
│     └── Cost: ~$10-20 per 8-hour run (20 agents)                 │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Dynamic Routing Logic

```python
class AdaptiveRouter:
    def route_task(self, task):
        complexity = self.assess_complexity(task)
        urgency = task.urgency
        
        if complexity == "high" and urgency == "high":
            return "kimi_k25"  # Best quality, fast
        elif complexity == "high" and urgency == "low":
            return "claude_batch"  # Quality with batch discount
        elif complexity == "medium":
            return "gpt41_mini"  # Balanced cost/performance
        else:  # low complexity
            return "gpt41_mini_batch"  # Maximum cost efficiency
```

### 6.3 Cost Optimization Strategies

| Strategy | Implementation | Savings |
|----------|---------------|---------|
| **Batch Processing** | Queue non-urgent tasks, process overnight | 50% |
| **Context Caching** | Share system prompts, research context | 30-50% |
| **Model Tiering** | Use mini models for 80% of tasks | 60-70% |
| **Intelligent Chunking** | Split large contexts optimally | 20-30% |
| **Rate Limit Awareness** | Throttle when approaching limits | Avoids downtime |
| **Early Stopping** | Stop low-quality agent runs early | 10-20% |

### 6.4 Budget-Aware Execution Modes

#### Economy Mode (Budget: $50-100)
- 90% GPT-4.1 Mini Batch
- 10% Kimi K2.5 for critical decisions
- Batch all non-real-time tasks
- Expected: 50 agents, 4 hours

#### Balanced Mode (Budget: $200-400)
- 60% GPT-4.1 Mini Batch
- 30% Kimi K2.5
- 10% Claude Sonnet for complex reasoning
- Expected: 100 agents, 8 hours

#### Performance Mode (Budget: $500-1000)
- 50% Kimi K2.5 (native swarm support)
- 30% Claude Sonnet 4.5
- 20% GPT-4.1 for specific capabilities
- Expected: 100 agents, 8 hours, highest quality

### 6.5 Value Per Dollar Maximization

```
Value/Dollar = (Research Quality x Insights Generated)
               ─────────────────────────────────────
               (Token Cost x Time to Results)
```

**Optimization Strategies:**
1. Prioritize parallel execution (Kimi swarm = 4.5x speedup)
2. Use batch processing for overnight runs (50% savings)
3. Cache and reuse research context (30% token reduction)
4. Early termination of low-value agent branches
5. Dynamic model selection based on task complexity

---

## 7. Recommendations

### 7.1 For Cost-Conscious Researchers

**Best Overall Value:** GPT-4.1 Mini or GPT-5 Mini with Batch API
- Lowest cost per token
- High rate limits (5,000 RPM / 5M TPM)
- 50% batch discount

**Best with Native Swarm:** Kimi K2.5
- Built-in 100-agent parallel execution
- 4.5x speedup on parallelizable tasks
- Competitive pricing at $307 for overnight run

### 7.2 For Quality-Critical Research

**Best Reasoning Quality:** Claude Sonnet 4.5
- Superior performance on complex reasoning
- 1M token context window
- Requires tier upgrade for practical swarm operation

**Best Coding/Tool Use:** Claude Sonnet 4.5 or Kimi K2.5
- 80.9% on SWE-bench (Claude)
- 76.8% on SWE-bench (Kimi)
- Native tool use capabilities

### 7.3 Rate Limit Mitigation

1. **Start with OpenAI mini models** for development (highest rate limits)
2. **Use Kimi for production swarms** (native parallel support)
3. **Upgrade Claude tiers** before scaling (Tier 1 is too restrictive)
4. **Implement intelligent queuing** for batch processing
5. **Monitor rate limit headers** and throttle proactively

---

## 8. Conclusion

Running 100-agent autoconstitution overnight is feasible across all major providers, but costs vary dramatically:

- **Minimum cost:** $99.84 (GPT-4.1 Mini Batch)
- **Best swarm experience:** $307.20 (Kimi K2.5 with native parallel support)
- **Maximum cost:** $1,440.00 (Claude Opus 4.5 Batch)

**Key Takeaway:** For most researchers, the optimal approach is a **hybrid orchestration** using GPT-4.1 Mini/GPT-5 Mini for 80% of tasks and Kimi K2.5 for orchestration and complex reasoning. This delivers the best value per dollar while maintaining research quality.

Students and indie researchers can get started with as little as **$20-50**, while serious autoconstitution requires a budget of **$300-500** for full overnight experiments.

---

## Appendix: Quick Reference Tables

### A. Cost Per 1M Tokens

| Provider | Input | Output | Batch Discount |
|----------|-------|--------|----------------|
| Kimi K2.5 | $0.60 | $2.50 | None |
| Claude Haiku 3.5 | $0.80 | $4.00 | 50% |
| Claude Sonnet 4.5 | $3.00 | $15.00 | 50% |
| Claude Opus 4.5 | $5.00 | $25.00 | 50% |
| GPT-5 | $1.25 | $10.00 | 50% |
| GPT-5 Mini | $0.25 | $2.00 | 50% |
| GPT-4.1 | $2.00 | $8.00 | 50% |
| GPT-4.1 Mini | $0.40 | $1.60 | 50% |

### B. Rate Limits at Tier 1

| Provider | RPM | Input TPM | Output TPM |
|----------|-----|-----------|------------|
| Kimi K2.5 | 1,000 | 500,000 | 200,000 |
| Claude Sonnet 4.5 | 50 | 30,000 | 8,000 |
| Claude Haiku 3.5 | 50 | 50,000 | 10,000 |
| GPT-5 | 500 | 500,000 | 200,000 |
| GPT-5 Mini | 1,000 | 1,000,000 | 400,000 |
| GPT-4.1 Mini | 5,000 | 5,000,000 | 2,000,000 |

### C. Context Windows

| Provider | Context Window |
|----------|----------------|
| Kimi K2.5 | 256K |
| Claude Sonnet 4.5 | 1M |
| Claude Haiku 3.5 | 200K |
| Claude Opus 4.5 | 200K |
| GPT-5 / GPT-4.1 | 128K |

---

*Report generated for autoconstitution cost optimization analysis.*
