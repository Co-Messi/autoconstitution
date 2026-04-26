# autoconstitution package guide

This package contains the product-facing CAI loop plus the lower-level engine
pieces that support it.

If you only remember one thing, remember this:

> `autoconstitution` is a multi-agent autoresearch loop.

The package exposes:

- **CAI loop primitives** for critique / revision / preference-pair building
- **public role classes** like `StudentAgent`, `JudgeAgent`, and `MetaJudgeAgent`
- **ratchet primitives** for keep-or-revert decisions
- **orchestrator primitives** for more advanced multi-agent runs

## The public API

Typical high-level imports:

```python
from autoconstitution import (
    CritiqueRevisionLoop,
    JudgeAgent,
    StudentAgent,
    PreferencePairBuilder,
    Ratchet,
)
```

Lower-level orchestration imports are still available:

```python
from autoconstitution import SwarmOrchestrator, TaskDependency, BranchPriority
```

## Core package areas

```text
autoconstitution/
├── cai/            # critique / revision / pair building
├── ratchet.py      # keep-or-revert logic
├── providers/      # local + cloud model adapters
├── metrics/        # evaluation metrics
├── hardware/       # hardware helpers
└── orchestrator.py # advanced orchestration primitives
```

## Which layer should I start with?

- Start with `autoconstitution.cai` if you want the product loop
- Start with `autoconstitution.ratchet` if you want explicit gating
- Start with `autoconstitution.orchestrator` only if you need more advanced branching and scheduling

## Example

```python
from autoconstitution import CritiqueRevisionLoop, JudgeAgent, StudentAgent
from autoconstitution.providers.ollama import OllamaProvider

provider = OllamaProvider()
student = StudentAgent(provider=provider)
judge = JudgeAgent(provider=provider)
loop = CritiqueRevisionLoop(student=student, judge=judge, max_rounds=3)
```

## License

MIT License
