# autoconstitution Launch Thread

> **Note for posting:** This is a shorter, product-first thread. The goal is instant legibility: autoresearch + multiple agents + constitution.

---

## Tweet 1 - The hook

`autoconstitution` is autoresearch, but with multiple agents.

Instead of one agent improving alone, a student, critic, teacher, and judge challenge each other under rules you edit in `constitution.md`.

Thread ↓

---

## Tweet 2 - The core pattern

Karpathy's autoresearch made a powerful loop legible:

try something -> evaluate it -> keep or revert

We keep that pattern, but turn it into a small society of agents.

---

## Tweet 3 - Why multiple agents

Many failures in agent systems are perspective failures:

- one model misses a flaw
- one model overcommits to a weak direction
- one model rewrites instead of really critiquing

Role separation helps.

---

## Tweet 4 - The roles

In `autoconstitution`:

- Student proposes
- Critic attacks weak spots
- Teacher suggests a better direction
- Judge decides what survives
- Synthesizer preserves useful findings

Same improvement loop, more internal pressure.

---

## Tweet 5 - Why "constitution"

The rules are not hidden in prompts.

They live in `constitution.md`, so you can inspect, diff, and edit the principles the agents are using to critique and revise each other.

---

## Tweet 6 - The loop

The loop is:

task -> propose -> critique -> revise -> judge -> keep or revert

That makes it feel close to autoresearch, but more collaborative and more explicit.

---

## Tweet 7 - Local first

You can run it locally with Ollama.

That matters because the whole point is to let normal builders experiment with AI improving AI, not only teams with giant infra budgets.

---

## Tweet 8 - Why this seems useful

This pattern should work across lots of domains:

- coding agents
- financial-analysis agents
- research assistants
- prompt / workflow optimization

Same loop. Different task.

---

## Tweet 9 - Output

The system can produce:

- revised answers
- critique traces
- chosen vs rejected pairs
- ratcheted decisions

So it works both as an inference-time scaffold and as training-data generation.

---

## Tweet 10 - Honest claim

I'm not claiming this magically solves every domain.

You still need:

- a real task
- a useful constitution
- a sensible evaluator

The contribution is the reusable improvement pattern.

---

## Tweet 11 - The shortest pitch

If autoresearch is one improving agent,

`autoconstitution` is a small society of improving agents.

Repo: `autoconstitution`
