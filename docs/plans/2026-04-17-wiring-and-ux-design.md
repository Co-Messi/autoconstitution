# autoconstitution Wiring Correctness and Live UX

**Goal:** make the CAI loop wire end-to-end without bugs, raise the code quality bar to match the project's own `pyproject.toml` config, and give the terminal experience the theatrical weight the product deserves — "you can watch constitutional AI happen."

**Scope ordering:** wiring correctness first, UX polish after. Local-model end-to-end testing is explicitly deferred — this spec is about making the product *correct* and *beautiful* first.

## Approved framing

- **Correctness before performance.** Every bug surfaced by the current test suite must either fail loudly or be fixed — no silent hangs.
- **The existing quality config is the bar.** `ruff` and `mypy --strict` are already configured in `pyproject.toml`. The public API surface must satisfy both.
- **Live-first UX.** When a human is watching, the CAI loop must feel like a performance — role panels, streaming tokens, a visible ratchet. When output is piped, the same content reduces cleanly to line-based logs.

## Work plan — vertical slices

The work is decomposed into four slices. Each slice fixes the bugs in one subsystem and polishes the UX around it. Each slice is independently reviewable and ends with tests green + `ruff` clean + `mypy --strict` clean on the code it touched.

### Slice 1 — Ratchet correctness + scoreboard

**Bugs to fix:**
- `Ratchet.commit_experiment` deadlocks: it holds `self._lock` (a non-reentrant `asyncio.Lock`) and then calls `self.validate_experiment`, which re-acquires the same lock. Fix by extracting a private `_validate_locked(...)` helper that performs the comparison without touching the lock; both `validate_experiment` and `commit_experiment` call it after acquiring the lock themselves. Delete the work-around comment in `tests/test_ratchet.py`.
- Audit `MultiMetricRatchet.validate_experiment` / `commit_experiment` for the same re-acquire pattern (lines ~1165 and ~1194 both take `_lock`, then call into per-metric ratchets that take their own locks — verify no cycle).
- Add `pytest-timeout` to the `dev` extras and set a 30s per-test timeout in `tests/pytest.ini`. A hang must fail the suite, not stall it.

**UX to add:**
- A compact ratchet scoreboard (Rich `Table`) that renders after every round: `best`, `last`, `Δ`, `decision`. Pulses green on KEEP, dim red on DISCARD, amber on TIE. Available as a standalone call so the CAI loop and any future use site share rendering.

**Done when:** full `tests/test_ratchet.py` runs to completion, all tests pass, the work-around comment is gone, `ruff` + `mypy --strict` clean on `ratchet.py`.

### Slice 2 — CAI loop correctness + role panels

**Bugs to fix:**
- Run `tests/test_orchestrator.py`, `tests/test_pollination.py`, `tests/test_public_api.py` to completion with the pytest-timeout in place. Triage every failure: categorize as real bug, stale test, or fixture issue, and fix each.
- Audit async error paths in `autoconstitution/cai/` for: silent `except Exception: pass`, provider exceptions not reaching the caller, `asyncio.gather` without `return_exceptions` handling, and cancellation leaks when a round is aborted.
- Audit `cli.py` and the `cai` subcommand for exit codes. Non-zero on provider failure, validation failure, or user `Ctrl+C`. Currently `typer` defaults may swallow these.
- Audit config validation in `config.py`: provider keys that don't exist should fail on startup with a helpful message, not on first use.

**UX to add:**
- **Live dashboard** driven by a Rich `Live` with a `Layout`:
  - Header: task prompt, round X of N, elapsed timer
  - Five role panels in a grid: Student / Critic / Teacher / Judge / Synthesizer. Each panel shows the role's current output, streamed token-by-token when the provider supports streaming, wrapped and scrolled within the panel.
  - The panel of the currently-speaking role pulses (border style cycles on a slow interval). All other panels dim slightly.
  - Footer: ratchet scoreboard from Slice 1, plus a subtle progress bar for the current round.
- Panel content uses role-specific accent colors (one per role) to make glances readable.
- A revision diff view: when the Student revises after a critique, show a compact inline diff (Rich `Syntax` + a custom differ) so the viewer can literally see the model criticizing and improving itself.

**Done when:** the CAI loop runs end-to-end against a `FakeProvider` (see Slice 4), all tests pass, live dashboard renders without layout thrashing, `ruff` + `mypy --strict` clean on `autoconstitution/cai/` and `cli.py`.

### Slice 3 — Providers + startup probe

**Bugs to fix:**
- Audit each provider (`ollama`, `kimi`, `anthropic`, `openai`) for: missing API-key handling, HTTP error → clear user-facing message, timeout defaults, retry behavior, streaming path correctness, and type errors that `mypy --strict` will surface.
- Verify the provider priority fallback described in the README actually works: the system should try providers in declared priority and skip unavailable ones with a visible notice, not silently fall through or crash.
- `autoconstitution cai providers` should never lie. If a provider says "live," sending a one-token probe to it should succeed.

**UX to add:**
- **Startup provider probe** with a Rich progress display: a line per provider (Ollama, Kimi, Anthropic, OpenAI), spinner while probing, ✓/✗ with latency on completion. Runs automatically at the start of `cai run` (skippable with `--no-probe`), and is the body of `cai providers`.
- First-run detection: if no provider is available, show a friendly Rich panel with the exact command to install Ollama or set an API key, instead of a stack trace.

**Done when:** `autoconstitution cai providers` renders the probe, every provider's error paths produce readable output, `ruff` + `mypy --strict` clean on `autoconstitution/providers/`.

### Slice 4 — Legacy CLI cleanup, TTY fallback, FakeProvider

**Bugs to fix:**
- The legacy CLI commands (`run`, `resume`, `status`, `benchmark`, `config`, `clean`) are labeled "legacy" in their help text but sit at the same visibility as `cai`. Decide per command:
  - **Keep & promote:** `config`, `clean` — still useful.
  - **Hide behind `legacy` subgroup:** `run`, `resume`, `status`, `benchmark` — `autoconstitution legacy run ...`. Print a one-line deprecation notice on use.
  - **Delete:** only if the code is truly dead — verify first.
- Audit TODO placeholders in `autoconstitution/agents/researcher.py` (5 occurrences of `# TODO: Generate ... patch`). Either implement or gate behind `NotImplementedError` with a clear message; shipping a string literal `"# TODO: ..."` as a generated patch is a silent bug.

**UX to add:**
- **TTY auto-detection.** `sys.stdout.isatty()` chooses between live dashboard and line-based logger. Line mode emits one structured line per role event with role-colored tags: `[student] ...`, `[critic] ...`, so piped output is still readable. Same content, different shell.
- **`--ui={live,plain,json}` override** for explicit control (CI uses `plain`, programmatic consumers use `json`).
- **FakeProvider** in `autoconstitution/providers/fake.py`: deterministic, scripted responses, zero network. End-to-end CAI tests run against it. This is the lynchpin that lets us assert behavior without hitting a real model.

**Done when:** `--help` output is tight, `--ui` works in all three modes, a full CAI round runs against `FakeProvider` in under a second in tests, `ruff` + `mypy --strict` clean on everything touched.

## Architecture notes

### Rendering layer split

UI rendering lives in a new module `autoconstitution/ui/` with three submodules:

- `ui/live.py` — Rich `Live`/`Layout` dashboard. One `Renderer` class with `on_role_start`, `on_token`, `on_role_end`, `on_round_end`, `on_ratchet_decision` methods.
- `ui/plain.py` — same interface, emits `[role] message` lines.
- `ui/json.py` — same interface, emits one JSON object per event on stdout.

The CAI loop emits events through a `Renderer` protocol, so the loop itself never imports Rich. Tests construct the loop with a capturing renderer.

### Event model

The loop emits a small set of typed events:

```python
class Event(Protocol): ...
@dataclass class RoleStart(Event): role: Role; round: int
@dataclass class Token(Event): role: Role; text: str
@dataclass class RoleEnd(Event): role: Role; output: str
@dataclass class Critique(Event): target_output: str; critique: str
@dataclass class Revision(Event): before: str; after: str
@dataclass class RatchetDecision(Event): kept: bool; delta: float; score: float
@dataclass class RoundEnd(Event): round: int; kept: bool
```

The renderers dispatch on event type. Adding a new UI mode means implementing the protocol, not modifying the loop.

### Testing strategy

- **Every bug gets a regression test.** The ratchet deadlock gets a `test_commit_experiment_lower_is_better` that currently would hang — it must pass after the fix.
- **`FakeProvider`** scripts exact responses, so CAI behavior is testable without a network.
- **pytest-timeout 30s per test.** Guards against future hang regressions.
- **Renderers are tested via a capturing renderer** — no terminal snapshot tests; Rich output differs by width.
- **Smoke test in CI:** `python -m autoconstitution.cli cai run --provider fake --prompt "test" --max-rounds 1 --ui plain` exits 0.

### Quality gates

Merge gate per slice:
- `pytest tests/` — zero failures, zero hangs (enforced by timeout)
- `ruff check autoconstitution/` — zero findings
- `mypy --strict autoconstitution/` — zero errors on the public API surface

These are already configured in `pyproject.toml`; this spec commits to actually using them.

## Explicit non-goals

- Real-model end-to-end tests. Deferred; local-model validation is a follow-up.
- New product features. Nothing added to the surface that doesn't already exist conceptually; this is correctness + UX.
- Breaking API changes. Public imports in `autoconstitution/__init__.py` stay stable.
- Rewriting the orchestrator. Orchestrator pieces are touched only where they intersect the CAI loop or tests.

## Risks

- **Streaming tokens across providers is uneven.** Ollama and OpenAI stream natively; Anthropic streams differently; Kimi via OpenAI-compatible endpoints usually does. The dashboard degrades to whole-message updates when streaming isn't available — no crash, just less animation on that provider.
- **Rich layout on narrow terminals.** At <100 columns the five-panel grid will be cramped. Below a threshold (say 120 cols) we collapse to a single active-role panel + a tape of prior-role outputs. Decision made at render time.
- **`mypy --strict` on legacy code may cascade.** If strict mode uncovers more than ~20 errors outside the public API surface, we narrow strict enforcement to the public API and track the rest as follow-up. This is a pragmatic concession — strict-on-everything is the ideal but not worth blocking the slice.

## Deliverables

- Code: fixes + new `autoconstitution/ui/` module + `FakeProvider`.
- Tests: regression test per fixed bug, FakeProvider-backed end-to-end tests, pytest-timeout.
- Docs: short `README.md` update showing the live dashboard (screenshot or asciicast — nice to have, not required).
- CI: nothing new required; existing gates must pass.
