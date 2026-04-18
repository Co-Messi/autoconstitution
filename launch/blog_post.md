# autoconstitution: autoresearch with multiple agents

`autoconstitution` starts from a simple intuition:

If autoresearch is one improving agent, maybe the next step is not a bigger single agent, but a small society of agents.

That means:

- one role proposes
- another critiques
- another reframes
- another judges
- and the system keeps only the improvements that survive

This is the core idea behind `autoconstitution`.

## Why build this?

Autoresearch made a powerful loop concrete:

1. make a change
2. run an evaluation
3. keep or revert

That loop is compelling because it is measurable and reversible.

But in practice, lots of useful improvement work is not just "try another variant." It is:

- finding flaws in a proposal
- pushing back against shallow answers
- suggesting alternative directions
- preserving useful findings across rounds

In other words, it often looks more like a team than a single worker.

## The role-based loop

`autoconstitution` structures that team explicitly.

- **Student** proposes
- **Critic** attacks weak spots
- **Teacher** suggests better directions
- **Judge** decides what survives
- **Synthesizer** preserves what was learned

The rules live in `constitution.md`, so the loop has an inspectable rulebook instead of relying entirely on hidden prompts.

## Why the constitution matters

One of the most useful ideas in Constitutional AI is not just "use AI feedback." It is "write the principles down."

That matters because it gives you:

- inspectability
- version control
- debate about the rules themselves
- a cleaner separation between mechanism and policy

In `autoconstitution`, the constitution is not a side file. It is part of the product.

## What the repo is trying to be

Not:

- a generic orchestration framework
- a vague swarm platform
- a magical solve-anything system

But:

- a reusable multi-agent improvement loop
- grounded in autoresearch
- extended with role separation and explicit rules
- applicable to different domains when paired with a task and evaluator

## The local-first angle

Another important goal is accessibility.

You should be able to run the core loop on modest hardware with Ollama. Bigger teams can use stronger provider mixes and more orchestration, but the default mental model should not assume giant infrastructure.

That matters because the repo is supposed to feel usable by ordinary builders, not only by large labs.

## Where this could go

The interesting long-term direction is not just "more agents."

It is:

- better task adapters
- better evaluators
- better judgment traces
- more reusable improvement patterns for domain-specific systems

Finance, coding, research, and workflow optimization are all examples of the same deeper pattern: AI helping AI improve under explicit rules.

## The shortest possible pitch

If autoresearch is one improving agent,

`autoconstitution` is a small society of improving agents.
