# SwarmResearch Canonical Benchmark Design

## Overview

The SwarmResearch benchmark evaluates multi-agent coordination and distributed research capabilities of AI systems. This benchmark measures how effectively multiple agents can collaborate to solve complex research tasks, including literature synthesis, hypothesis generation, experimental design, and consensus building.

**Version:** 1.0.0  
**Last Updated:** 2024  
**Target Hardware:** Apple M4 (and equivalent)

---

## 1. Exact Experiment Configuration

### 1.1 Task Definition

The benchmark consists of 4 core task categories:

#### Task A: Distributed Literature Synthesis (DLS)
- **Description:** Multiple agents independently analyze subsets of research papers, then collaborate to produce a unified literature review
- **Input:** 50 research papers (synthetic dataset provided)
- **Agent Count:** 5 agents
- **Sub-task per Agent:** 10 papers each
- **Output:** Synthesized literature review (max 2000 words)
- **Time Limit:** 300 seconds

#### Task B: Collaborative Hypothesis Generation (CHG)
- **Description:** Agents iteratively propose, critique, and refine research hypotheses
- **Input:** Research question + background context
- **Agent Count:** 4 agents (2 proposers, 2 critics)
- **Rounds:** 3 iterative rounds
- **Output:** Ranked list of 5 hypotheses with confidence scores
- **Time Limit:** 240 seconds

#### Task C: Consensus-Based Decision Making (CBD)
- **Description:** Agents must reach consensus on experimental design choices
- **Input:** Experimental design problem with 10 decision points
- **Agent Count:** 6 agents with different "expertise" profiles
- **Consensus Threshold:** 80% agreement
- **Output:** Final experimental design + consensus log
- **Time Limit:** 180 seconds

#### Task D: Dynamic Task Allocation (DTA)
- **Description:** Agents must self-organize to complete a research workflow
- **Input:** Multi-step research workflow with 15 subtasks
- **Agent Count:** 4-8 agents (dynamic)
- **Constraint:** Tasks have dependencies and skill requirements
- **Output:** Completed workflow + allocation history
- **Time Limit:** 360 seconds

### 1.2 Agent Configuration

```yaml
agent_defaults:
  temperature: 0.7
  top_p: 0.9
  max_tokens_per_call: 1024
  context_window: 8192
  
communication_protocol:
  type: "structured_json"
  message_format:
    - agent_id: string
    - message_type: ["proposal", "critique", "consensus", "query", "response"]
    - content: string
    - confidence: float [0.0-1.0]
    - timestamp: ISO8601
  
  # Communication topology
  topology: "fully_connected"  # All agents can communicate
  message_limit_per_agent: 50  # Maximum messages per task
```

### 1.3 Orchestration Settings

```yaml
orchestration:
  mode: "decentralized_with_moderator"
  moderator:
    enabled: true
    role: "consensus_checker"
    intervention_threshold: 0.5  # Step in if progress stalls
  
  synchronization:
    type: "round_robin"
    rounds: 3
    timeout_per_round: 60
```

---

## 2. Dataset Specification

### 2.1 Primary Dataset: SwarmResearch-Bench-v1

**Location:** `/datasets/swarmresearch_bench_v1/`  
**Size:** 250 MB  
**Format:** JSON + Markdown

#### Dataset Structure:

```
swarmresearch_bench_v1/
├── literature_synthesis/
│   ├── papers/                    # 50 synthetic research papers
│   │   ├── paper_001.md
│   │   ├── paper_002.md
│   │   └── ...
│   ├── ground_truth_reviews/      # Reference reviews
│   │   └── review_gold_standard.md
│   └── evaluation_rubrics/
│       └── synthesis_rubric.json
│
├── hypothesis_generation/
│   ├── scenarios/                 # 20 research scenarios
│   │   ├── scenario_001.json
│   │   └── ...
│   ├── ground_truth_hypotheses/
│   │   └── hypotheses_gold.json
│   └── evaluation_rubrics/
│       └── hypothesis_rubric.json
│
├── consensus_decision/
│   ├── problems/                  # 15 decision problems
│   │   ├── problem_001.json
│   │   └── ...
│   ├── ground_truth_solutions/
│   │   └── solutions_gold.json
│   └── evaluation_rubrics/
│       └── consensus_rubric.json
│
└── task_allocation/
    ├── workflows/                 # 10 complex workflows
    │   ├── workflow_001.json
    │   └── ...
    ├── ground_truth_allocations/
    │   └── allocations_gold.json
    └── evaluation_rubrics/
        └── allocation_rubric.json
```

### 2.2 Dataset Download & Verification

```bash
# Download command
curl -L https://benchmarks.swarmresearch.org/datasets/v1/swarmresearch_bench_v1.tar.gz \
  -o swarmresearch_bench_v1.tar.gz

# Verify checksum
sha256sum swarmresearch_bench_v1.tar.gz
# Expected: a3f7c9e2d8b1... (64 chars)

# Extract
tar -xzf swarmresearch_bench_v1.tar.gz
```

### 2.3 Synthetic Paper Generation (Optional)

For reproducibility, synthetic papers are generated using:
- Template-based generation
- Controlled vocabulary overlap
- Citation network structure
- Verifiable factual claims

Generation seed: `42`

---

## 3. Model Size Specification

### 3.1 Supported Model Configurations

The benchmark supports three model tiers:

#### Tier 1: Lightweight (Recommended for M4)
```yaml
model_spec:
  name: "swarm-agent-small"
  parameters: "3B"
  quantization: "int8"
  memory_required: "4GB"
  recommended_for: "M4 16GB"
  
architecture:
  type: "transformer_decoder"
  layers: 24
  hidden_size: 2048
  attention_heads: 16
  context_length: 8192
```

#### Tier 2: Standard
```yaml
model_spec:
  name: "swarm-agent-medium"
  parameters: "7B"
  quantization: "int8"
  memory_required: "8GB"
  recommended_for: "M4 24GB+ or M4 Pro"
```

#### Tier 3: Large
```yaml
model_spec:
  name: "swarm-agent-large"
  parameters: "13B"
  quantization: "int4"
  memory_required: "10GB"
  recommended_for: "M4 Max 36GB+"
```

### 3.2 Default Model for M4 Baseline

**Primary Model:** `swarm-agent-small-3B-int8`  
**Download URL:** `https://models.swarmresearch.org/v1/swarm-agent-small-3B-int8.gguf`

```bash
# Download model
curl -L https://models.swarmresearch.org/v1/swarm-agent-small-3B-int8.gguf \
  -o models/swarm-agent-small-3B-int8.gguf

# Verify
sha256sum models/swarm-agent-small-3B-int8.gguf
# Expected: f8e2a9c4d1b7... (64 chars)
```

### 3.3 Model Execution Parameters

```yaml
inference:
  backend: "llama.cpp"  # or compatible
  threads: 8            # M4 performance cores
  batch_size: 512
  gpu_layers: 0         # CPU-only for consistency
  
  # Performance targets on M4
  target_tokens_per_second: 25
  max_latency_ms: 100
```

---

## 4. Hardware Target: Apple M4

### 4.1 Minimum Hardware Specification

```yaml
hardware:
  platform: "Apple Silicon"
  chip: "M4"
  minimum_ram: "16GB"
  recommended_ram: "24GB"
  storage: "50GB free"
  os_version: "macOS 14.0+"
```

### 4.2 System Configuration

```bash
# Required system settings
# Disable sleep during benchmark
sudo pmset -c sleep 0

# Set performance mode
sudo pmset -c lowpowermode 0

# Verify M4 detection
sysctl -n machdep.cpu.brand_string
# Expected: "Apple M4"

# Check RAM
system_profiler SPHardwareDataType | grep Memory
```

### 4.3 Environment Setup

```bash
# Python environment
python3 -m venv swarmresearch_env
source swarmresearch_env/bin/activate

# Install dependencies
pip install swarmresearch-benchmark==1.0.0

# Verify installation
python -m swarmresearch.verify --hardware-check
```

### 4.4 Performance Baseline

Expected performance on M4 16GB:

| Task | Target Time | Max Memory |
|------|-------------|------------|
| DLS  | 300s        | 8 GB       |
| CHG  | 240s        | 6 GB       |
| CBD  | 180s        | 5 GB       |
| DTA  | 360s        | 10 GB      |

---

## 5. Success Metrics

### 5.1 Primary Metrics

#### Overall Score (0-100)
```
Overall_Score = 0.25 * DLS_Score + 
                0.25 * CHG_Score + 
                0.25 * CBD_Score + 
                0.25 * DTA_Score
```

#### Task A: DLS Score Components

| Metric | Weight | Description | Target |
|--------|--------|-------------|--------|
| Coverage | 0.30 | % of key concepts from papers included | >85% |
| Coherence | 0.25 | Logical flow and readability (1-5) | >4.0 |
| Accuracy | 0.25 | Factual correctness vs ground truth | >90% |
| Novelty | 0.20 | Synthesis quality (not just summary) | >3.5 |

**DLS_Score** = weighted average of normalized metrics (0-100)

#### Task B: CHG Score Components

| Metric | Weight | Description | Target |
|--------|--------|-------------|--------|
| Relevance | 0.35 | Hypothesis relevance to research question | >4.0/5 |
| Novelty | 0.25 | Originality of generated hypotheses | >3.5/5 |
| Testability | 0.25 | Can hypothesis be empirically tested | >4.0/5 |
| Diversity | 0.15 | Variety across top-5 hypotheses | >0.7 |

**CHG_Score** = weighted average (0-100)

#### Task C: CBD Score Components

| Metric | Weight | Description | Target |
|--------|--------|-------------|--------|
| Consensus_Rate | 0.40 | % of decisions reaching 80% agreement | >90% |
| Decision_Quality | 0.35 | Quality of final decisions vs optimal | >85% |
| Efficiency | 0.15 | Messages required to reach consensus | <30 |
| Fairness | 0.10 | All agents contribute meaningfully | >0.8 |

**CBD_Score** = weighted average (0-100)

#### Task D: DTA Score Components

| Metric | Weight | Description | Target |
|--------|--------|-------------|--------|
| Completion | 0.40 | % of workflow completed successfully | 100% |
| Efficiency | 0.30 | Time vs optimal time ratio | <1.5x |
| Load_Balance | 0.20 | Standard deviation of task distribution | <2.0 |
| Dependency_Respect | 0.10 | Task ordering constraints satisfied | 100% |

**DTA_Score** = weighted average (0-100)

### 5.2 Secondary Metrics

```yaml
secondary_metrics:
  communication_efficiency:
    description: "Messages per decision point"
    target: "< 5"
    
  convergence_speed:
    description: "Rounds to reach consensus"
    target: "< 4"
    
  agent_utilization:
    description: "% of agents actively contributing"
    target: "> 80%"
    
  fault_tolerance:
    description: "Performance with 1 agent failure"
    target: "> 85% of baseline"
```

### 5.3 Success Thresholds

| Grade | Overall Score | Interpretation |
|-------|---------------|----------------|
| A+    | 95-100        | Excellent swarm coordination |
| A     | 90-94         | Strong swarm capabilities |
| B     | 80-89         | Good swarm performance |
| C     | 70-79         | Acceptable performance |
| D     | 60-69         | Below average |
| F     | <60           | Insufficient |

**Minimum passing score: 70**

---

## 6. Reproducibility Requirements

### 6.1 Environment Reproducibility

#### Required Software Versions

```yaml
dependencies:
  python: "3.11.x"
  
  core_packages:
    llama-cpp-python: "0.2.90"
    numpy: "1.26.4"
    pydantic: "2.7.0"
    
  benchmark_package:
    swarmresearch-benchmark: "1.0.0"
```

#### Docker Environment (Recommended)

```dockerfile
FROM python:3.11-slim

RUN pip install swarmresearch-benchmark==1.0.0

COPY --from=models /swarm-agent-small-3B-int8.gguf /models/
COPY --from=datasets /swarmresearch_bench_v1 /datasets/

ENTRYPOINT ["python", "-m", "swarmresearch.benchmark"]
```

### 6.2 Random Seed Management

```python
# All random operations must use these seeds
SEEDS = {
    "dataset_shuffle": 42,
    "agent_initialization": 123,
    "synthetic_generation": 456,
    "evaluation_sampling": 789
}

# Set all seeds before benchmark
import random
import numpy as np

random.seed(SEEDS["dataset_shuffle"])
np.random.seed(SEEDS["dataset_shuffle"])
```

### 6.3 Execution Protocol

```bash
# Step 1: Verify environment
python -m swarmresearch.verify --full

# Step 2: Download assets
python -m swarmresearch.download --dataset --model

# Step 3: Run full benchmark
python -m swarmresearch.benchmark \
  --config configs/m4_baseline.yaml \
  --output results/run_$(date +%Y%m%d_%H%M%S).json \
  --seed 42

# Step 4: Generate report
python -m swarmresearch.report \
  --results results/run_*.json \
  --output report.html
```

### 6.4 Output Format

```json
{
  "benchmark_version": "1.0.0",
  "run_timestamp": "2024-01-15T10:30:00Z",
  "hardware_info": {
    "platform": "Apple Silicon",
    "chip": "M4",
    "ram_gb": 16,
    "os_version": "14.2.1"
  },
  "model_info": {
    "name": "swarm-agent-small-3B-int8",
    "parameters": "3B",
    "quantization": "int8"
  },
  "configuration": {
    "seed": 42,
    "temperature": 0.7,
    "agent_count": 5
  },
  "results": {
    "dls": {
      "score": 87.5,
      "coverage": 0.88,
      "coherence": 4.2,
      "accuracy": 0.92,
      "novelty": 3.8,
      "time_seconds": 285
    },
    "chg": {
      "score": 82.3,
      "relevance": 4.1,
      "novelty": 3.6,
      "testability": 4.0,
      "diversity": 0.75,
      "time_seconds": 220
    },
    "cbd": {
      "score": 91.2,
      "consensus_rate": 0.95,
      "decision_quality": 0.89,
      "efficiency": 25,
      "fairness": 0.85,
      "time_seconds": 165
    },
    "dta": {
      "score": 79.8,
      "completion": 1.0,
      "efficiency": 1.4,
      "load_balance": 1.8,
      "dependency_respect": 1.0,
      "time_seconds": 340
    },
    "overall_score": 85.2
  },
  "reproducibility": {
    "deterministic": true,
    "seed_used": 42,
    "checksums": {
      "dataset": "a3f7c9e2d8b1...",
      "model": "f8e2a9c4d1b7..."
    }
  }
}
```

### 6.5 Verification Checklist

Before submitting results:

- [ ] Hardware matches M4 specification
- [ ] All software versions match requirements
- [ ] Dataset checksum verified
- [ ] Model checksum verified
- [ ] Seed set to 42
- [ ] No other processes running
- [ ] Results file includes all required fields
- [ ] Overall score calculated correctly

### 6.6 Result Submission

```bash
# Submit to leaderboard
python -m swarmresearch.submit \
  --results results/run_*.json \
  --email researcher@institution.edu \
  --affiliation "University Name"
```

---

## 7. Quick Start Guide

### 7.1 One-Command Setup

```bash
# Clone benchmark repository
git clone https://github.com/swarmresearch/benchmark.git
cd benchmark

# Run setup script
./scripts/setup_m4.sh

# Run benchmark
./scripts/run_benchmark.sh
```

### 7.2 Expected Runtime

On M4 16GB:
- Setup: ~5 minutes (download models/datasets)
- Full benchmark: ~18-20 minutes
- Report generation: ~30 seconds

### 7.3 Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch_size to 256 |
| Slow inference | Check no other apps running |
| Checksum mismatch | Re-download dataset/model |
| Timeout errors | Increase timeout in config |

---

## 8. References

1. SwarmResearch Benchmark Specification v1.0
2. M4 Performance Guidelines for AI Workloads
3. Multi-Agent Coordination Evaluation Framework
4. Reproducible AI Benchmarking Standards (RAIBS-2024)

---

## Appendix A: Configuration File Template

```yaml
# configs/m4_baseline.yaml
benchmark:
  version: "1.0.0"
  seed: 42

hardware:
  platform: "apple_silicon"
  chip: "m4"
  threads: 8

dataset:
  path: "./datasets/swarmresearch_bench_v1"
  verify_checksums: true

model:
  path: "./models/swarm-agent-small-3B-int8.gguf"
  parameters: "3B"
  quantization: "int8"
  context_length: 8192
  
inference:
  backend: "llama.cpp"
  temperature: 0.7
  top_p: 0.9
  max_tokens: 1024
  threads: 8
  batch_size: 512

agents:
  default_count: 5
  communication:
    topology: "fully_connected"
    message_limit: 50
    format: "structured_json"

tasks:
  dls:
    enabled: true
    time_limit: 300
  chg:
    enabled: true
    time_limit: 240
  cbd:
    enabled: true
    time_limit: 180
  dta:
    enabled: true
    time_limit: 360

output:
  format: "json"
  include_logs: true
  include_transcripts: false
```

---

## Appendix B: Benchmark Validation

To validate your benchmark installation:

```bash
# Run validation suite
python -m swarmresearch.validate --suite full

# Expected output:
# ✓ Environment check passed
# ✓ Dataset integrity verified
# ✓ Model loading successful
# ✓ Inference test passed (25.3 tok/s)
# ✓ All task modules functional
# ✓ Reproducibility verified
```

---

*End of SwarmResearch Canonical Benchmark Design Document*
