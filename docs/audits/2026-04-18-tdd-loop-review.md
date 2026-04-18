# TDD Infrastructure Audit

While `--critique-mode tests` replaces unreliable LLM critiques with ground-truth `pytest` logs, the local models are still hitting a wall and yielding zero improvements. 

After digging into the entire infrastructure of the benchmark and runner logic (`benchmark/tdd_loop.py`), I've found exactly why local LLMs are failing. The test-harness has two fatal logic bugs that trap models in infinite loops and strip away exception data.

### 1. The "Groundhog Day" Bug (Stateless Revisions)

**Location:** `benchmark/tdd_loop.py` inside `run_tdd_benchmark` loop.

**Vulnerability:**
If the Student model generates a revision that fixes `Bug A` but introduces `Bug B`, its score might remain `0.0` (because total passing tests didn't go up). The loop evaluates `if revised_score.score > current_score.score:` and evaluates to `False`. The code then just executes `pass` and discards the new code.
On the next iteration, the system regenerates the exact same prompt using `previous_answer=current_answer` (which is still the baseline) and `failure_text=current_fail` (which is still the baseline failure). The model has no memory of its failed revision! It is repeatedly prompted to solve the exact same equation with zero context of what it just tried. Local LLMs naturally output the same bad guess repeatedly, wasting all rounds.

**Recommendation:**
Accept "lateral moves." If the score doesn't decrease, but the failure traceback changes, the model has successfully moved past the old bug. 
```python
if revised_score.score > current_score.score or (
    revised_score.score == current_score.score and revised_fail != current_fail
):
    current_answer = revised
    ...
```

### 2. The Truncated Traceback Bug (Head vs Tail slice)

**Location:** `benchmark/tdd_loop.py` inside `_extract_failure_text` and `_truncate`.

**Vulnerability:**
If `pytest` hits a SyntaxError, an ImportError, or a Collection error, it outputs an `ERRORS` section rather than a `FAILURES` section. The parser `_extract_failure_text` fails to find the block and falls back:
```python
    if start is None:
        # No structured failures — fall back to a tail of stdout.
        return _truncate(output, 800)
```
However, the `_truncate` function is implemented completely backwards:
```python
def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
```
This slice (`text[: limit]`) takes the **first 800 characters** of the output instead of the tail! Pytest output always starts with 15-20 lines of headers, platform details, plugin versions, and collection status. The actual exception and traceback are completely chopped off. 
The Student model is commanded to "Fix the implementation" based on "FAILING TESTS", but the text it receives contains zero errors—just normal pytest startup logs. It is hopelessly blinded and logically cannot improve.

**Recommendation:**
Update `_truncate` to genuinely extract the tail (where the traceback lives), or better yet, extract the tail of the output directly in the fallback handler.
```python
def _truncate_tail(text: str, limit: int) -> str:
    return text if len(text) <= limit else "…" + text[-(limit - 1):]
```
