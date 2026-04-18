# autoconstitution Public Roadmap

This roadmap follows one principle:

> keep the product legible as multi-agent autoresearch while expanding what it can improve.

---

## v0.2 - Product coherence

**Goal:** make `autoconstitution` feel like one product instead of a clever pile of parts.

### Focus

- clear role-based story across all docs
- stable CAI loop API
- local-first onboarding with Ollama
- better packaging honesty and install guidance
- one hero example that people can immediately understand

### Success criteria

- README, CLI, and launch docs all tell the same story
- new users can run one example in minutes
- the repo feels more like "autoresearch with multiple agents" than a generic framework

---

## v0.3 - Task adapters

**Goal:** prove that the same loop applies across different domains.

### Focus

- coding-agent adapter
- financial-analysis adapter
- research-assistant adapter
- stronger evaluator / ratchet interfaces per task type

### Success criteria

- at least two memorable domain examples
- users can see the same loop applied to different tasks without confusion

---

## v0.4 - Better judgment and memory

**Goal:** make critique and synthesis more reliable across rounds.

### Focus

- stronger Judge output structure
- better Synthesizer / memory retention
- improved preference-pair generation
- more explicit anti-collapse and diversity checks

### Success criteria

- cleaner critique traces
- fewer malformed local-model judge outputs
- better preservation of useful findings over longer runs

---

## v0.5 - Broader orchestration

**Goal:** expose advanced orchestration only when it makes the product stronger.

### Focus

- optional branching and parallel role groups
- more intentional provider routing
- better local-vs-cloud execution ergonomics
- clearer separation between public product surface and advanced engine pieces

### Success criteria

- orchestration helps the product story instead of drowning it
- advanced users can scale up without making the repo harder for newcomers

---

## v1.0 - A reusable improvement loop

**Goal:** make `autoconstitution` the default mental model for multi-agent self-improvement.

### The product should feel like this

- start from autoresearch
- add explicit roles
- add explicit rules
- keep or revert based on judgment
- apply the same loop to many domains

### Not the goal

- become a vague everything-framework
- lead with infrastructure instead of product
- require large-company scale just to get started

---

## The long-term direction

If autoresearch is one improving agent,

`autoconstitution` should become the obvious way to build a small society of improving agents.
