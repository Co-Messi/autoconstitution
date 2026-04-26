# Constitutional AI Structural Audit
**Subject:** Diagnostics on LLM-as-a-Judge Performance Degradation
**Focus:** Why training on local LLM critiques mathematically degrades performance instead of improving it.

After a line-by-line architectural audit of the CAI framework (`hierarchy.py`, `critique_revision.py`, and `preference_pairs.py`), I have found the root cause of why your symmetric/asymmetric local models drop in performance (`-0.1667` and `-0.1111`) while the `pytest` test-grounded mode yields `+0.2513`.

You are experiencing a textbook case of **Adversarial Reward Hacking & Data Poisoning**. The codebase is mathematically structured to train the Student model to strictly prefer broken, degraded code over correct code.

Here is the exact breakdown of the architectural failures.

### 1. The Local LLM Quality Gap (Hallucinated Critiques)
Constitutional AI was originally designed for extremely large, highly capable models (like Claude or GPT-4). When translating this to small local LLMs (e.g., Llama 3b / 8b), the assumption that "the Judge's critique is fundamentally correct" breaks down.
- Weak Judges fail to properly comprehend complex spatial or algorithmic code.
- To fulfill the `JUDGE_SYSTEM_TEMPLATE`, the Judge frequently hallucinates non-existent violations against the Constitution, forcing the Student to "fix" perfectly valid, functional code.

### 2. The Unconditional Ratchet Failure
In `cai/critique_revision.py`, the `CritiqueRevisionLoop` forces the Student to revise based on the Judge's critique.
```python
            # Ask Student to revise given critiques.
            revised = await _run_role(  ... )
            current_answer = revised
```
Unlike your `tdd_loop.py` which rigorously tests if `revised_score.score > current_score.score` before accepting a revision, the CAI loop unconditionally overwrites the code. If the Judge hallucinates a fake flaw, the Student breaks the code, and this broken code structurally becomes the new `final_answer`.

### 3. The Root Cause: Inverted Preference Pairs (Data Poisoning)
The fatal execution happens in `cai/preference_pairs.py` when it translates the loop outputs into DPO pairs:
```python
    @property
    def rejected(self) -> str:
        return self.initial_answer

    @property
    def chosen(self) -> str:
        return self.final_answer
```
This is the nail in the coffin. Because the local Judge frequently degrades the code, `final_answer` is worse/buggier than the `initial_answer`.
When you run DPO (Direct Preference Optimization), the training algorithm interprets `chosen` as the ground-truth standard. You are mathematically fine-tuning the model to:
1. **Maximize the generation of the corrupted, hallucinated `final_answer`.**
2. **Minimize and avoid the functional, correct `initial_answer`.**

Your CAI loop is acting as an **Adversarial Poisoning Engine**. It actively un-trains coding capability in favor of sycophantic, compliant-sounding but logically broken code that placated a hallucinating 3B Judge.

### Architectural Recommendations
If you wish to fix the local LLM-Critique mode, you must introduce objective grounding to the inner loop.

1. **Verify Before Trusting:** Never unconditionally accept `current_answer = revised`. Insert a hidden evaluator (like a sandbox test runner) that scores the revision. If the `revised` text degrades the original code's compile state, reject the critique entirely.
2. **Filter DPO Pairs:** Update `PreferencePairBuilder._accept()` to run a lightweight evaluation (or heuristics) on the `chosen` candidate. If the `final_answer` is objectively lower quality than the `initial_answer`, discard the trace entirely. Never train on degraded pairs.
3. **Judge Confidence Posturing:** Soften the Judge prompt so it is allowed to say "Imperfect but compliant" rather than being heavily pushed to extract critiques out of thin air just to satisfy JSON formatting requirements.
