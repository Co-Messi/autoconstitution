# autoconstitution Newsletter

## Headline

# **What if autoresearch had a small society of agents?**

### *Introducing autoconstitution - a multi-agent improvement loop for prompts, agents, workflows, and domain-specific systems*

---

## What is autoconstitution?

`autoconstitution` is a multi-agent autoresearch system.

It keeps the same basic discipline that made autoresearch compelling:

- try something
- evaluate it
- keep or revert

But instead of one agent iterating alone, it uses multiple roles:

- Student
- Critic
- Teacher
- Judge
- Synthesizer

The rules live in `constitution.md`, so the system's behavior is explicit rather than buried inside prompts.

---

## Why it matters

Single-agent loops are elegant, but they only get one perspective at a time.

Many real improvement tasks benefit from internal pressure:

- one model proposes
- another attacks weaknesses
- another suggests a better direction
- another judges whether the change actually improved the result

That pattern shows up everywhere:

- coding agents
- financial-analysis workflows
- research assistants
- prompt / workflow optimization

`autoconstitution` is an attempt to package that pattern so ordinary builders can run it locally and stronger teams can scale it up later.

---

## Why now

Two ideas are colliding:

1. **autoresearch** made self-improving loops legible
2. **constitutional AI** made explicit rulebooks usable

`autoconstitution` combines them:

- multi-agent instead of single-agent
- constitutional instead of ad hoc critique
- ratcheted instead of vibe-based iteration

---

## What you can do with it

- run critique / revision loops locally with Ollama
- export chosen-vs-rejected pairs for later tuning
- define your own constitution in Markdown
- plug the same loop into different task types

Example:

```bash
pip install autoconstitution
ollama pull llama3.1:8b
autoconstitution cai run -p "Design a better financial-analysis workflow."
```

---

## What we're not claiming

`autoconstitution` is not magic.

You still need:

- a real task
- a useful constitution
- a sensible evaluator

The contribution is the reusable improvement loop, not a claim that every domain is already solved.

---

## What’s next

Near-term priorities:

- stronger public examples
- better task adapters
- cleaner benchmark story
- improved judge reliability on local models

The goal is simple:

> make multi-agent self-improvement as legible as single-agent autoresearch.
