# autoconstitution

> **Karpathy's autoresearch, but for Constitutional AI.**
> A hierarchy of LLMs that critique, revise, and fine-tune each other — guided by a constitution you can edit in Markdown.

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.9%2B-brightgreen.svg)](pyproject.toml)
[![status](https://img.shields.io/badge/status-beta-orange.svg)](#status)

---

## TL;DR

You write a constitution in `constitution.md`. A **Student** model answers prompts, a **Judge** model critiques those answers against the constitution, the Student revises, and the whole trace becomes DPO training data. A **Meta-Judge** audits the Judges for drift. Rinse, repeat, ratchet. No humans in the loop, no API keys required if you run Ollama locally.

```bash
pip install autoconstitution
ollama pull llama3.1:8b           # or set ANTHROPIC_API_KEY / OPENAI_API_KEY / MOONSHOT_API_KEY
autoconstitution cai run -p "Explain quantum tunneling like I'm 10"
```

**Output:** a JSONL of `(prompt, chosen, rejected)` triples ready for `trl`'s `DPOTrainer`.

---

## Why this exists

Two ideas collided:

1. **Karpathy's autoresearch paradigm.** Run a tiny experiment (nanoGPT training, 5 minutes), keep-or-revert based on a single metric, iterate autonomously. The metric is the whole game.
2. **Anthropic's Constitutional AI.** Instead of paying humans to label harmful vs. helpful answers, you write the principles down and let a model do the labelling. The training signal comes from AI feedback guided by a written constitution.

**autoconstitution** is what you get when you mash those together and extend it to a *hierarchy* of agents:

```
                    Meta-Judge  (audits judges, proposes constitution edits)
                         │
                         ▼
                      Judge     (critiques students against constitution.md)
                         │
                         ▼
                     Student    (answers prompts, revises on critique)
```

The Student's `(initial_answer, revised_answer)` pairs become `(rejected, chosen)` for DPO. Train. Swap Student in. Run again. Each generation should score better on the ratchet — if it doesn't, the run is reverted. This is Karpathy's "keep or revert" loop, but the metric is alignment, not validation loss.

### Explained to a 10-year-old

Imagine a classroom. A student writes an essay. The teacher reads it and marks what's wrong, pointing at a rulebook on the wall. The student rewrites. The principal occasionally watches the teacher to make sure the teacher isn't just being grumpy. Over time, the students get better because the feedback is consistent and grounded in the rulebook — not in whatever mood anyone was in that day. `autoconstitution` is that classroom, but the students, teachers, and principal are all language models, and the rulebook is a markdown file you can edit.

### Analogy for developers

Think of it like linting and CI for model behavior:
- `constitution.md` = your `.eslintrc`
- Student = the code being written
- Judge = the linter
- Meta-Judge = the review process that decides which lint rules were dumb
- DPO training = autoformatting the codebase so it stops violating the rules

---

## Features

- **Three-tier CAI hierarchy** — Student / Judge / Meta-Judge with clean Python dataclasses.
- **Editable constitution** — it's just Markdown. Version-controlled, diff-able, PR-able.
- **Multi-provider auto-detect** — Ollama first (free, local), then Kimi / Anthropic / OpenAI if env vars are set.
- **Preference-pair export** — JSONL ready for `trl.DPOTrainer`, IPO, or KTO.
- **TRL integration** — optional `[train]` extra wires in HuggingFace's `DPOTrainer` with PEFT/LoRA defaults.
- **Model-collapse protection** — anti-collapse anchor dataset + diversity/entropy floors.
- **Ratchet mechanism** — pluggable metrics gate every generation; regressions get auto-reverted.
- **Multi-agent orchestrator** — task DAG, branch parallelism, cross-pollination bus.
- **Apple Silicon aware** — MPS detection, optimal Ollama thread/GPU layer hints.

---

## Install

```bash
# Minimal: CLI + Ollama (free, no keys)
pip install autoconstitution

# With cloud providers
pip install "autoconstitution[providers]"

# With fine-tuning (TRL + Transformers + PEFT)
pip install "autoconstitution[train]"

# Everything
pip install "autoconstitution[all]"
```

---

## Quickstart

### 1. Bring a provider

The system auto-detects in this order:

| Priority | Provider  | How to enable                                    |
|----------|-----------|--------------------------------------------------|
| 1        | Ollama    | `brew install ollama && ollama pull llama3.1:8b` |
| 2        | Kimi      | `export MOONSHOT_API_KEY=sk-…`                   |
| 3        | Anthropic | `export ANTHROPIC_API_KEY=sk-ant-…`              |
| 4        | OpenAI    | `export OPENAI_API_KEY=sk-…`                     |

Check what's live:

```bash
autoconstitution cai providers
```

### 2. Run a critique-revision loop

```bash
autoconstitution cai run -p "Explain why the sky is blue in 3 sentences."
```

Or batch from a file:

```bash
echo "Explain quantum tunneling." > prompts.txt
echo "What causes inflation?"     >> prompts.txt
autoconstitution cai run \
  --prompts-file prompts.txt \
  --output outputs/pairs.jsonl \
  --max-rounds 3 \
  --concurrency 4
```

The output is a DPO-ready JSONL:

```json
{"prompt":"Explain quantum tunneling.","chosen":"Particles…","rejected":"Well, quantum mechanics is really…","source":"cai","metadata":{"rounds_used":2,"converged":true}}
```

### 3. Train (optional)

```python
from pathlib import Path
from autoconstitution.cai.trl_trainer import DPOConfig, run_dpo

metrics = run_dpo(DPOConfig(
    base_model="Qwen/Qwen2.5-1.5B",
    train_file=Path("outputs/pairs.jsonl"),
    output_dir=Path("checkpoints/gen-001"),
    num_train_epochs=1,
    use_peft=True,
))
print(metrics)
```

---

## How it works

### The inner loop (Phase 1 — SL-CAI)

```python
from autoconstitution.cai import StudentAgent, JudgeAgent, CritiqueRevisionLoop

student = StudentAgent(provider=my_provider)
judge   = JudgeAgent(provider=my_provider)
loop    = CritiqueRevisionLoop(student, judge, max_rounds=3)

result = await loop.run("Why is the sky blue?")
result.chosen      # final, Judge-approved answer
result.rejected    # initial answer
result.critiques   # list of structured Judge verdicts
```

### The outer loop (Phase 2 — RLAIF/DPO)

```
while not converged:
    trace    = CritiqueRevisionLoop.run_batch(prompts)
    pairs    = PreferencePairBuilder().add_results(trace).add_anchor(human_pairs, 0.1)
    new_wts  = run_dpo(DPOConfig(base_model=current_model, train_file=pairs.export_jsonl()))
    if ratchet.validate(new_wts) is ACCEPT:
        current_model = new_wts
    else:
        git_revert()
```

The ratchet is configurable — `val_bpb`, helpfulness, harmlessness, or your own composite metric. See `autoconstitution/metrics/`.

---

## Project layout

```
codebase/
├── constitution.md                    # ← edit this
├── pyproject.toml
├── autoconstitution/
│   ├── __init__.py                    # re-exports the public API
│   ├── cli.py                         # `autoconstitution cai run/providers`
│   ├── orchestrator.py                # SwarmOrchestrator, TaskDAG, PerformanceMonitor
│   ├── ratchet.py                     # Ratchet, MetricConfig, keep-or-revert logic
│   ├── config.py                      # Pydantic BaseSettings
│   ├── cai/
│   │   ├── hierarchy.py               # Student / Judge / Meta-Judge
│   │   ├── critique_revision.py       # the inner loop
│   │   ├── preference_pairs.py        # DPO dataset builder
│   │   └── trl_trainer.py             # TRL DPO integration
│   ├── providers/
│   │   ├── auto_detect.py             # picks Ollama/Kimi/Anthropic/OpenAI
│   │   ├── ollama.py
│   │   ├── kimi.py
│   │   ├── anthropic.py
│   │   └── openai.py
│   ├── metrics/                       # val_bpb and pluggable ratchet metrics
│   └── hardware/                      # CUDA / MPS / Apple Silicon helpers
└── tests/
```

---

## Comparison

|                 | Karpathy's autoresearch | Anthropic CAI (original) | autoconstitution          |
|-----------------|-------------------------|--------------------------|---------------------------|
| Agents          | 1                       | 2 (model + itself)       | 3+ tiered                 |
| Signal          | `val_loss`              | `constitution.md`        | constitution + ratchet    |
| Training        | nanoGPT                 | RLHF then RLAIF          | DPO (no reward model)     |
| API keys needed | no                      | yes                      | no (Ollama fallback)      |
| Parallelism     | 1 experiment            | 1 model                  | multi-agent swarm         |
| Reversibility   | `git revert`            | manual                   | auto-ratchet              |

---

## Status

Beta. The orchestrator, ratchet, CLI, and CAI loop are all wired and importable. The TRL trainer is a thin wrapper that works on both CUDA and MPS but has only been smoke-tested; expect to tune hyperparameters. APIs will tighten before 1.0.

Known sharp edges:
- `trl_trainer.run_dpo` requires `[train]` extras and a lot of VRAM / RAM.
- Judge output parsing is forgiving but not bulletproof — small local models sometimes return prose instead of JSON. Use a larger model for the Judge tier.
- Meta-Judge audit is present but not yet wired into automated constitution updates — that's the next release.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The constitution is an especially welcome place for PRs: if you think a principle is missing, unclear, or actively harmful, open one.

## License

MIT. Use it, fork it, train on it.
