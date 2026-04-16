# autoconstitution Open Source Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reposition `autoconstitution` as a multi-agent autoresearch system and align the docs, API surface, and CLI with that product story.

**Architecture:** Keep `autoconstitution` as the public product and selectively preserve engine pieces from the current codebase. The public story should center on the role-based improvement loop, while orchestration and scaling internals move below the fold.

**Tech Stack:** Python, Typer, Rich, Pydantic, provider adapters, Markdown docs, pytest

---

### Task 1: Create a clean repository baseline

**Files:**
- Create: `.gitignore`
- Create: `docs/plans/2026-04-16-autoconstitution-open-source-redesign-design.md`
- Create: `docs/plans/2026-04-16-autoconstitution-open-source-redesign.md`

**Step 1: Add repo-level ignore rules**

Ignore caches, local outputs, build artifacts, and other generated files that should not enter the new git history.

**Step 2: Save the approved design**

Capture the agreed product framing and priorities inside the repo.

**Step 3: Commit the clean baseline**

Initialize git and make the first commit as the starting point for the redesign.

### Task 2: Rewrite the public narrative

**Files:**
- Modify: `codebase/README.md`
- Modify: `AUTOCONSTITUTION_SUMMARY.md`
- Modify: `launch/show_hn.md`
- Modify: `launch/positioning_statement.md`

**Step 1: Rewrite the README opening**

Lead with "autoresearch, but with multiple agents" and the role-based loop.

**Step 2: Remove inherited swarmresearch language**

Fix stale naming and generic framework wording in the summary and launch docs.

**Step 3: Commit the docs/story rewrite**

Commit once the repo's public identity is coherent.

### Task 3: Align package and CLI surface with the product

**Files:**
- Modify: `codebase/autoconstitution/__init__.py`
- Modify: `codebase/autoconstitution/cli.py`
- Modify: `codebase/autoconstitution/README.md`
- Create: `codebase/tests/test_public_api.py`

**Step 1: Write failing tests for the public API**

Add focused tests for top-level exports and product-facing CLI/help wording.

**Step 2: Run the tests and verify failure**

Confirm the failures reflect the current mismatch between story and code.

**Step 3: Implement the minimal fixes**

Export the CAI primitives that matter, update the CLI descriptions, and replace generic package docs.

**Step 4: Run the targeted tests and relevant existing suite**

Ensure the new public API and wording are stable.

**Step 5: Commit the API/CLI alignment**

Commit after the tests pass.

### Task 4: Prepare the next product layer

**Files:**
- Modify: `codebase/constitution.md`
- Create or modify: a hero example under `codebase/examples/`
- Modify: `codebase/pyproject.toml` if packaging needs further alignment

**Step 1: Elevate the constitution artifact**

Make `constitution.md` feel like a first-class product artifact instead of just a support file.

**Step 2: Add one hero example**

Choose a memorable domain example that makes the product feel real.

**Step 3: Commit the product polish**

Commit once the next visible slice of the product is in place.

