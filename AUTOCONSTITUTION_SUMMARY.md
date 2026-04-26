# autoconstitution - repository summary

## Overview

`autoconstitution` is a multi-agent autoresearch system. It keeps the familiar keep-or-revert loop from single-agent autoresearch, but extends it into a role-structured society of agents that can propose, critique, revise, judge, and preserve improvements under explicit rules.

The repository combines three ideas:

- **autoresearch** as the base improvement loop
- **constitutional critique** to keep revisions grounded in explicit rules
- **multi-agent roles** so different models can attack the problem from different angles

## Product story

The intended public framing is:

> Autoresearch, but with multiple agents.

The key user flow is:

1. write or edit `constitution.md`
2. define a task or prompt set
3. let the agent roles iterate through critique and revision
4. keep the improvements that survive judgment
5. optionally export pairs or traces for later training

## What the repo contains

- `codebase/` - the Python package, CLI, tests, and local training helpers
- `architecture/` - deep design notes for orchestration, scaling, and evaluation
- `benchmark/` - reproducibility and benchmark design material
- `launch/` - positioning, launch copy, and public-facing materials
- `research/` - background research used to shape the project

## Core concepts

- **Student / Teacher / Critic / Judge / Synthesizer roles**
- **Editable constitution in Markdown**
- **Keep-or-revert ratchet**
- **Provider abstraction across local and cloud models**
- **Optional multi-agent orchestration beneath the product surface**

## Current direction

The repo is being repositioned around the multi-agent autoresearch story. The public-facing work should prioritize:

1. clear product framing
2. coherent package/API surface
3. one or two compelling domain examples
4. credible benchmarks and proof

The architecture and scaling pieces remain useful, but they should support the product story rather than dominate it.
