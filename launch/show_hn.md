# Show HN: autoconstitution - autoresearch with multiple agents

**TL;DR:** `autoconstitution` takes the autoresearch loop and turns it into a small society of agents. Instead of one agent iterating alone, a student, critic, teacher, judge, and synthesizer can challenge and teach each other under rules you edit in `constitution.md`.

---

## The idea

Karpathy's autoresearch made a simple pattern very legible:

1. try a change
2. run an evaluation
3. keep or revert

`autoconstitution` keeps that pattern, but replaces the single improving loop with multiple roles:

- a **Student** proposes
- a **Critic** attacks weak spots
- a **Teacher/Researcher** suggests refinements
- a **Judge** decides what survives
- a **Synthesizer** preserves useful findings for the next round

The result is still a keep-or-revert system, but one where agents can challenge and teach each other instead of acting alone.

---

## Why this seems useful

Single-agent loops are compelling because they're easy to reason about, but they also bottleneck on one perspective. In practice, a lot of improvement work looks more like:

- one model proposes a direction
- another points out why it is wrong
- another reframes it
- another checks whether the change actually improved the target

That pattern shows up everywhere:

- coding agents
- financial-analysis agents
- research assistants
- prompt and workflow optimization

So the goal of `autoconstitution` is not "build a generic swarm framework." The goal is to make **multi-agent autoresearch** feel as usable as the single-agent version.

---

## How it works

The loop is:

```text
task -> propose -> critique -> revise -> judge -> keep or revert
```

And the behavior rules live in `constitution.md`, so the system is not just searching blindly. It is improving under explicit principles that you can inspect, diff, and change.

That makes the repo sit somewhere between:

- Karpathy's autoresearch
- Claude-style Constitutional AI
- a reusable multi-agent improvement scaffold

---

## Quick example

```bash
pip install autoconstitution
ollama pull llama3.1:8b
autoconstitution cai run -p "Design a better financial signal for earnings revisions."
```

What comes out is not just one answer. The system records the critique/revision trail and can export chosen-vs-rejected pairs for later training or further ratcheted evaluation.

---

## Current status

- [x] Constitutional critique / revision loop
- [x] Ratchet mechanism for keep-or-revert decisions
- [x] Local-first provider path via Ollama
- [x] Multi-provider support for Kimi / Anthropic / OpenAI
- [x] Multi-agent orchestration primitives
- [ ] Better hero examples for specific domains
- [ ] Stronger end-to-end benchmark story

**Honest assessment:** the interesting part here is the combination of the ideas, not a claim that the system already solves every domain well. You still need a task, an evaluator, and a good constitution. But the pattern itself feels broadly reusable.

---

## What makes it different

Most agent repos are one of these:

- a workflow engine
- a prompt wrapper
- a collection of role prompts

`autoconstitution` is trying to be a reusable **improvement loop**:

- multi-agent instead of single-agent
- constitutional instead of ad hoc critique
- ratcheted instead of vibe-based iteration

If that framing is wrong, I'd love to hear it. The whole point of posting this is to see whether this pattern is actually legible and useful to other people.
