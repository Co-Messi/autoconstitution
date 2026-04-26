# autoconstitution Landing Page

---

## HERO SECTION

### Headline
**Autoresearch, but with multiple agents**

### Subheadline
A constitutional multi-agent improvement loop where agents propose, critique, revise, and judge each other to improve prompts, workflows, agents, and domain-specific strategies.

### Primary CTA
[Get Started - pip install autoconstitution]

### Secondary CTA
[Read the README]

### Hero Description
`autoconstitution` takes the simple keep-or-revert logic that made autoresearch compelling and turns it into a small society of agents. Instead of one agent iterating alone, a Student, Critic, Teacher, Judge, and Synthesizer can challenge and teach each other under explicit rules in `constitution.md`.

### Hero Code Snippet
```python
from autoconstitution import CritiqueRevisionLoop, JudgeAgent, StudentAgent
from autoconstitution.providers.ollama import OllamaProvider

provider = OllamaProvider()
student = StudentAgent(provider=provider)
judge = JudgeAgent(provider=provider)
loop = CritiqueRevisionLoop(student=student, judge=judge, max_rounds=3)

result = await loop.run("Design a better financial-analysis workflow.")
print(result.chosen)
```

---

## PROBLEM / SOLUTION SECTION

### Problem Headline
**Single-agent improvement loops hit perspective limits**

### Problem Statements

**One agent sees one path**
Single-agent loops are elegant, but they only explore one line of reasoning at a time. If the model misses a flaw or locks onto a mediocre direction, the whole run inherits that blind spot.

**Critique is implicit**
Many systems revise by prompting harder, not by assigning a real adversarial role. That means weak ideas are often rephrased instead of challenged.

**Rules are hidden**
When the improvement criteria live in prompts or code, it is hard to inspect, diff, or debate what the system is actually optimizing for.

### Solution Headline
**A small society of agents with explicit rules**

### Solution Statements

**Role separation**
The Student proposes, the Critic attacks, the Teacher reframes, the Judge decides, and the Synthesizer preserves useful findings.

**Editable constitution**
The rules live in `constitution.md`, so the system's behavior is explicit and inspectable.

**Keep-or-revert discipline**
The ratchet preserves the best result and rejects regressions, so iteration remains measurable and reversible.

**Local-first, scale-up later**
Run with Ollama on modest hardware or use stronger cloud models when the task needs them.

---

## KEY FEATURES SECTION

### Features Headline
**Built for reusable AI-on-AI improvement**

### Feature 1: Multi-Agent Roles
**More perspectives, same loop**

The core product is not "a swarm framework." It is a reusable improvement loop with structured roles:

- Student
- Critic
- Teacher / Researcher
- Judge
- Synthesizer

This lets the system challenge its own work instead of only extending it.

### Feature 2: Constitution in Markdown
**Rules you can inspect and change**

The rulebook is just a file:

```md
## P1 - Truthfulness
Do not invent sources, numbers, or quotations.

## P2 - Calibration
Mark uncertainty when the evidence is weak.
```

That makes the behavior reviewable and version-controlled.

### Feature 3: Ratchet Mechanism
**Keep improvements, reject regressions**

Every iteration still follows the autoresearch spirit:

```text
propose -> critique -> revise -> judge -> keep or revert
```

The point is not just to generate traces. The point is to preserve better ones.

### Feature 4: Training-Ready Outputs
**From critique traces to preference pairs**

The loop can export chosen-vs-rejected pairs for later DPO-style training, which makes the repo useful both as an inference-time scaffold and as a dataset generator.

### Feature 5: Task Adaptability
**Same loop, different domains**

The same pattern can be used for:

- coding agents
- financial-analysis workflows
- research assistants
- prompt and workflow optimization

The loop stays the same. The task and evaluator change.

---

## HOW IT WORKS SECTION

### Headline
**The loop in one picture**

```text
task -> propose -> critique -> revise -> judge -> keep or revert
```

The task might be "improve this coding agent prompt," "design a better financial-analysis workflow," or "rewrite this research answer under the constitution." The structure remains stable even as the domain changes.

---

## SOCIAL PROOF / BELIEF SECTION

### Headline
**Why people might care**

- If you liked autoresearch, this is the multi-agent extension
- If you liked Constitutional AI, this makes the rules explicit and reusable
- If you build agents, this gives you a way to let AI improve AI without hiding the rules

---

## CTA SECTION

### Primary CTA
[Try it locally with Ollama]

### Secondary CTA
[Read the critique / revision loop]

### Closing line
**A small society of agents is often more useful than one agent thinking alone.**
