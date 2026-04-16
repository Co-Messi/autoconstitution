# Show HN: SwarmResearch – Multi-Agent Autonomous Research Framework

**TL;DR:** A Python framework where multiple LLM agents collaborate on research tasks through a shared "cross-pollination bus." Think Karpathy's agent ideas, but with structured adversarial critique and hardware-aware orchestration.

---

## What's Actually Novel

Most "agent frameworks" are just DAGs with LLM calls. SwarmResearch differs in three specific ways:

### 1. Cross-Pollination Bus with Backpressure

Agents don't just call sub-agents—they broadcast validated findings to a shared bus with token-bucket rate limiting. When one agent discovers a hyperparameter improvement, others receive it via pub/sub, not function returns.

```python
# Agent A discovers something useful
await bus.publish(Finding(
    agent_id=agent_a_id,
    finding_type=FindingType.HYPERPARAMETER_IMPROVEMENT,
    priority=FindingPriority.HIGH,
    payload={"lr": 0.001, "accuracy_gain": 0.03}
))

# Agent B receives it automatically via subscription
finding = await agent_b_client.get_next_finding(timeout=5.0)
```

The bus includes adaptive rate limiting that throttles under load—critical for preventing cascade failures when many agents discover things simultaneously.

### 2. Constitutional Critic Agent

Inspired by Anthropic's Constitutional AI but applied to multi-agent coordination. The critic doesn't just validate—it generates structured counter-arguments, predicts failure modes, and assigns confidence scores to rejections.

```python
class ConstitutionalCriticAgent(BaseAgent):
    """Argues AGAINST proposals using constitutional principles.
    
    Principles: harmlessness, honesty, helpfulness, robustness
    Output: confidence_score (0-1), predicted_failure_modes, 
            counter_arguments, conditions_for_acceptance
    """
```

This isn't "vibe check" validation—it produces auditable critique structures with severity ratings and risk scores.

### 3. Hardware-Aware Dynamic Reallocation

The orchestrator monitors agent performance and migrates agents between branches based on actual compute constraints (GPU memory, M4 Neural Engine availability), not just queue depth.

```python
# Auto-scale based on branch load + hardware capability
report = await orchestrator.analyze_and_reallocate()
# Returns: bottlenecks, reallocations, spawn/termination decisions
```

---

## Architecture

```
SwarmOrchestrator (async Python 3.11+)
├── TaskDAG              # Cycle detection, topological execution
├── AgentPoolManager     # Lifecycle + capability-based routing  
├── PerformanceMonitor   # Bottleneck identification
├── CrossPollinationBus  # Pub/sub with frequency controls
└── ResearchBranch       # Hierarchical organization

Agent Types:
├── ResearcherAgent      # Hypothesis → Experiment → CodeChange
├── ExperimenterAgent    # A/B test execution with metrics
├── CriticAgent          # Constitutional evaluation
└── SynthesiserAgent     # Merge validated improvements
```

---

## Why This Exists

After watching Karpathy's explorations with autonomous agents [1], I kept hitting the same wall: single-agent systems don't scale to real research workflows. You need:

- **Parallel hypothesis generation** (not sequential)
- **Adversarial validation** (agents that try to break each other's proposals)
- **Knowledge persistence** (findings outlive individual agent contexts)

SwarmResearch is my attempt to formalize these patterns into something reproducible.

---

## Current Status

- [x] Core orchestrator with Task DAG
- [x] Cross-pollination bus with rate limiting
- [x] Researcher + Critic agent implementations
- [x] Hardware detection (GPU/M4)
- [ ] Full LLM provider abstraction (partial)
- [ ] Distributed execution (local only)

**Not production-ready.** The framework runs, but the "research" part still requires significant domain-specific prompting to produce useful results.

---

## Installation

```bash
pip install swarmresearch
```

Requires Python 3.11+ for proper async generics.

---

## Honest Assessment

**What works:**
- Task DAG execution with proper dependency resolution
- Cross-pollination broadcasting at ~1000 msg/sec locally
- Hardware detection and agent-to-device mapping

**What's hard:**
- LLM agents still hallucinate experiment designs
- Constitutional critique quality depends heavily on the underlying model
- No built-in sandboxing for generated code (you probably want this)

**What I'd do differently:**
- Start with the bus abstraction, not the orchestrator
- Use structured generation (JSON schema) for all agent outputs
- Build sandboxing in from day one

---

## Code

https://github.com/swarmresearch/swarmresearch

MIT licensed. Contributions welcome, especially for:
- Additional LLM provider integrations
- Better experiment sandboxing
- Real benchmark results comparing single vs. multi-agent approaches

---

[1] Karpathy's work on LLM agents, particularly his explorations with tool use and autonomous coding: https://twitter.com/karpathy/status/...

---

*Happy to answer questions about the architecture, design tradeoffs, or why I chose specific async patterns. Also happy to hear why this is a terrible idea—HN feedback is usually right.*

---

## First Comment: Technical Decisions Explained

Since HN usually asks about the "why" behind technical choices, here are the main ones:

### Why Python 3.11+ and async-first?

**Short answer:** `typing.Self` and proper async generics.

The orchestrator uses complex generic types (`BaseAgent[TContext, TResult]`) that require Python 3.11's improved type system. The async-first design isn't premature optimization—when you're running 50+ agents concurrently with dependency graphs, thread-per-agent would explode.

The `asyncio.Semaphore` for `max_concurrent_tasks` is simple but effective. I considered `trio` for structured concurrency but the ecosystem isn't there yet for ML tooling.

### Why a custom TaskDAG instead of using Airflow/Prefect?

Airflow is overkill for sub-second task latencies. The built-in DAG provides:
- Cycle detection at task-add time (not runtime)
- Priority-based topological sort
- Dynamic task injection (agents can spawn new tasks mid-execution)

The key method is `get_ready_tasks()` which returns all tasks whose dependencies are satisfied—this drives the execution wave pattern.

### Why token bucket for rate limiting?

The cross-pollination bus uses a token bucket because:
1. **Burst tolerance:** Agents often discover things in clusters (after a batch completes)
2. **Fairness:** Per-agent buckets prevent one noisy agent from drowning others
3. **Priority bypass:** Critical findings (e.g., "experiment is corrupting data") bypass limits entirely

I also implemented an adaptive rate limiter that scales with system load, but it's not the default—token bucket is more predictable.

### Why "constitutional" critique instead of simple validation?

Simple validation asks "is this correct?" Constitutional critique asks "what's wrong with this, even if it seems correct?" The difference matters for research:

- **Validation:** "The learning rate change improves accuracy"
- **Critique:** "The improvement is only on the validation set, not held-out test. Likely overfitting. Predicted failure mode: performance regression on OOD data."

The eight constitutional principles (harmlessness, honesty, helpfulness, transparency, fairness, robustness, efficiency, clarity) are configurable per-deployment.

### Why no built-in LLM provider yet?

The `LLMProvider` protocol is defined but I haven't committed to an implementation because:
1. Everyone has different rate limits/cost constraints
2. Structured generation requirements vary (some need JSON mode, others don't)
3. I wanted the core to work without any LLM calls (rule-based fallbacks everywhere)

The current `openai.py` provider is ~100 lines. Adding Anthropic/others is straightforward.

### The biggest mistake so far

Not using Pydantic for the data classes. I used `@dataclass(slots=True)` for performance, but Pydantic v2's validation + JSON schema generation would have saved hours of manual serialization code. The `to_dict()` methods everywhere are technical debt.

---

*Edit: Fixed typo in code example, added clarification on Pydantic vs dataclasses.*
