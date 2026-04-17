# autoconstitution

> **Autoresearch, but with multiple agents.**
> A constitutional multi-agent improvement loop where agents propose, critique, revise, judge, and preserve better strategies under rules you edit in Markdown.

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11%2B-brightgreen.svg)](pyproject.toml)
[![status](https://img.shields.io/badge/status-beta-orange.svg)](#status)

---

## 60-second tour

```bash
pip install autoconstitution
autoconstitution demo
```

That's it. `demo` probes for a provider (Ollama local, or `MOONSHOT_API_KEY` /
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`), opens a live Rich dashboard, and runs
a three-round critique/revision loop against a canned prompt so you can watch
constitutional AI happen in your terminal.

> _(asciinema recording lands once the full UX pass ships — placeholder for now.)_

When you're ready to use it on your own prompts:

```bash
autoconstitution cai providers                     # see which providers are live
autoconstitution cai run -p "your task here"       # single prompt, live dashboard
autoconstitution cai run -f prompts.txt --ui json  # batch, machine-readable stream
```

---

## TL;DR

`autoconstitution` starts from the core autoresearch pattern:

1. try something
2. evaluate it
3. keep or revert

But instead of a single agent iterating alone, it uses a small society of roles:

- **Student** proposes
- **Critic** attacks weak spots
- **Teacher / Researcher** suggests better directions
- **Judge** decides what survives
- **Synthesizer** preserves useful findings across rounds

The rules live in `constitution.md`, so the loop is not just trial-and-error. It improves under explicit principles you can inspect, diff, and change.

**Output:** a structured critique/revision trace plus chosen-vs-rejected pairs that can be exported for later ratcheting or DPO-style training.

---

## Why this exists

Karpathy's autoresearch made a powerful idea legible: autonomous improvement becomes practical when the loop is small, measurable, and reversible.

But many real tasks are not best solved by one agent alone. In practice, improvement often looks like:

- one system proposes an approach
- another argues why it fails
- another rewrites it
- another judges whether it actually improved the target

`autoconstitution` is an attempt to turn that pattern into a reusable product:

- **multi-agent instead of single-agent**
- **constitutional instead of ad hoc critique**
- **ratcheted instead of vibe-based iteration**

That makes it applicable to more than one domain. The same loop can be used to improve:

- coding agents
- financial-analysis agents
- research assistants
- prompt and workflow systems
- model behavior and training traces

### A simple mental model

Think of it as a classroom:

- the **Student** answers
- the **Critic** points out what is weak
- the **Teacher** suggests a better direction
- the **Judge** checks the answer against the rulebook
- the **Synthesizer** writes down what the class learned

The rulebook is `constitution.md`.

---

## Features

- **Role-based multi-agent loop** — Student / Critic / Teacher / Judge / Meta-Judge-style structure for iterative improvement.
- **Editable constitution** — behavior rules live in Markdown, not hidden prompts.
- **Local-first provider path** — Ollama works out of the box for small builders.
- **Cloud-model support** — Kimi / Anthropic / OpenAI integrations when stronger models are needed.
- **Preference-pair export** — traces can become chosen-vs-rejected data for DPO-style training.
- **Ratchet mechanism** — keep-or-revert gating for improvements.
- **Optional orchestration layer** — branch/task/pollination infrastructure for more complex runs.
- **Apple Silicon aware** — MPS and local hardware helpers for modest machines.

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

The system can run on local models first, then use cloud providers when available:

| Priority | Provider  | How to enable                                    |
|----------|-----------|--------------------------------------------------|
| 1        | Ollama    | `brew install ollama && ollama pull llama3.1:8b` |
| 2        | Kimi      | `export MOONSHOT_API_KEY=sk-...`                 |
| 3        | Anthropic | `export ANTHROPIC_API_KEY=sk-ant-...`            |
| 4        | OpenAI    | `export OPENAI_API_KEY=sk-...`                   |

Check what's live:

```bash
autoconstitution cai providers
```

### 2. Run a critique / revision loop

```bash
autoconstitution cai run -p "Explain why the sky is blue in 3 sentences."
```

This opens a live Rich dashboard — role panels for Student / Judge / Meta-Judge, tokens streaming into the active panel, and a ratchet scoreboard in the footer. When stdout isn't a TTY the tool auto-degrades to line logs; override with `--ui`:

| `--ui` value | When to use |
|---|---|
| `auto` | Default. Live dashboard on a TTY, plain logs when piped. |
| `live` | Force the live dashboard even if stdout isn't a TTY. |
| `plain` | One `[role round=N] text` line per event. Ideal for CI. |
| `json` | One JSON object per event on stdout. For programmatic consumers. |

Batch from a file:

```bash
echo "Design a better financial-analysis workflow." > prompts.txt
echo "Improve this research-assistant prompt stack." >> prompts.txt
autoconstitution cai run \
  --prompts-file prompts.txt \
  --output outputs/pairs.jsonl \
  --max-rounds 3 \
  --concurrency 4 \
  --ui plain
```

### 3. Export or train (optional)

The critique/revision trace can be turned into preference pairs for later tuning:

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

## Hero example: financial analyst

One concrete way to use the repo is to improve the thinking process of a
financial-analysis agent.

```bash
autoconstitution cai run \
  --prompts-file examples/financial_analyst/prompts.txt \
  --constitution examples/financial_analyst/constitution.finance.md \
  --output outputs/financial_analyst_pairs.jsonl
```

That example is meant to improve workflow quality, not just generate one-off
market commentary. See [examples/financial_analyst/README.md](./examples/financial_analyst/README.md).

---

## How it works

### The inner loop

```python
from autoconstitution.cai import StudentAgent, JudgeAgent, CritiqueRevisionLoop

student = StudentAgent(provider=my_provider)
judge = JudgeAgent(provider=my_provider)
loop = CritiqueRevisionLoop(student, judge, max_rounds=3)

result = await loop.run("Why is the sky blue?")
result.chosen
result.rejected
result.critiques
```

### The general pattern

```text
task -> propose -> critique -> revise -> judge -> keep or revert
```

That pattern is the real product. The same loop can drive many task adapters as long as you provide:

- a task
- a constitution
- an evaluator or ratchet

### The outer loop

For training-oriented runs, the trace can feed a ratcheted fine-tuning cycle:

```text
trace -> preference pairs -> train -> evaluate -> keep or revert
```

The ratchet can use `val_bpb`, helpfulness, harmlessness, or a custom composite metric.

---

## Project layout

```text
codebase/
├── constitution.md                    # edit the rulebook
├── pyproject.toml
├── autoconstitution/
│   ├── __init__.py                    # public API surface
│   ├── cli.py                         # autoconstitution cai run/providers
│   ├── cai/                           # student / judge / critique / pair building
│   ├── ratchet.py                     # keep-or-revert logic
│   ├── providers/                     # Ollama / Kimi / Anthropic / OpenAI
│   ├── metrics/                       # pluggable evaluation metrics
│   ├── hardware/                      # local hardware helpers
│   └── orchestration pieces           # advanced branch/task orchestration
└── tests/
```

---

## Comparison

|                      | Karpathy's autoresearch | Anthropic CAI | autoconstitution |
|----------------------|-------------------------|---------------|------------------|
| Core loop            | keep or revert          | critique / revise | critique / revise / judge / keep or revert |
| Main structure       | one improving agent     | constitution-guided critique | multi-agent role society |
| Rules                | metric-driven           | constitution-driven | constitution + ratchet |
| Local-first option   | yes                     | not really     | yes via Ollama |
| General applicability| medium                  | medium         | high if task adapters are good |

---

## Status

Beta. The constitutional loop, provider integrations, ratchet, and orchestration primitives are present, but the repo is still being tightened around a clearer product story.

Known sharp edges:

- some internal docs still need cleanup to match the product framing
- stronger hero examples are still needed
- the end-to-end benchmark story is not yet where it should be
- small local judges can still return messy output instead of structured critiques

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). `constitution.md` is intentionally a first-class artifact, so improvements to the rules, examples, and critique quality are especially welcome.

## License

MIT. Use it, fork it, and adapt it to your own agents or systems.
