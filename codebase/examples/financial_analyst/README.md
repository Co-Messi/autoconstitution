# Financial Analyst Example

This example shows how to use `autoconstitution` to improve a finance-oriented
agent workflow rather than just generate a single answer.

The intended pattern is:

- the **Student** proposes an analysis workflow
- the **Critic** attacks weak assumptions and missing checks
- the **Teacher** nudges the system toward stronger structure
- the **Judge** enforces the constitution
- the **ratchet** preserves the better version

## Use case

Imagine you want an agent that analyzes:

- earnings revisions
- management guidance changes
- balance-sheet stress
- valuation context
- possible edge vs narrative noise

The goal is not "predict the stock perfectly." The goal is to produce a better,
more disciplined workflow for how the agent should think.

## Files

- `prompts.txt` — example finance prompts for the CAI loop
- `constitution.finance.md` — a domain-specific constitution overlay

## Run it

```bash
autoconstitution cai run \
  --prompts-file examples/financial_analyst/prompts.txt \
  --constitution examples/financial_analyst/constitution.finance.md \
  --output outputs/financial_analyst_pairs.jsonl
```

## What to look for

Good revised outputs should:

- distinguish signal from speculation
- state uncertainty clearly
- avoid fake precision
- separate facts, assumptions, and conclusions
- prefer process quality over overconfident calls

## Why this example matters

It shows the real promise of the repo:

`autoconstitution` is not just for safety or alignment. It is a reusable
multi-agent improvement loop for domain-specific systems.

