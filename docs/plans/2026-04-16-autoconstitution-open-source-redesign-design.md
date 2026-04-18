# autoconstitution Open Source Redesign

**Goal:** Position `autoconstitution` as "autoresearch with multiple agents" while keeping the strongest implementation pieces from the current codebase.

## Approved framing

`autoconstitution` should be understood first as an autoresearch-style system with multi-agent roles.

The repo story is:

- Start with Karpathy's autoresearch loop
- Extend it from one improving agent into a small society of agents
- Use explicit rules in `constitution.md` to keep critique and revision coherent
- Apply the same improvement pattern to coding, finance, research, and other domain-specific tasks

## Product principles

1. `autoconstitution` stays the public identity
2. `swarmresearch` ideas can be borrowed as implementation pieces, not as branding
3. Public docs lead with the improvement loop, not orchestrator architecture
4. The repo should feel usable on local hardware first, scalable second
5. The star-worthy message is "autoresearch, but with multiple agents"

## Public-facing shape

The public face of the repository should emphasize:

- Roles: student, teacher, critic, judge, synthesizer
- Loop: propose -> critique -> revise -> judge -> keep or revert
- Constitution: editable rules in Markdown
- Task adapters: the same improvement loop applied to different domains
- Quickstart: local-first, Ollama-friendly, simple CLI entrypoint

The following should be demoted beneath the fold:

- Generic swarm/orchestrator framing
- Broad "research operating system" claims
- Heavy PARL / distributed-systems language on the front page
- Architecture-first summaries that are stronger on scope than proof

## Cleanup priorities

1. Rewrite the repo and launch story around the approved framing
2. Remove inherited `swarmresearch` wording from user-facing surfaces
3. Align package exports, CLI help, and docs with the actual CAI/product surface
4. Keep engine pieces that support the story, hide the rest under docs/architecture
5. Build toward one hero example people can immediately imagine using
