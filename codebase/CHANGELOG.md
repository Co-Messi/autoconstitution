# Changelog

All notable changes to `autoconstitution` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-04-16

### Project renamed

- `SwarmResearch` → `autoconstitution` across the entire codebase, package, config
  files, and documentation. The entry-point binary is now `autoconstitution`
  (with `autoconst` as a shorter alias).

### Added — Constitutional AI primitives

- `constitution.md` at project root — the editable rulebook that drives all
  Judge critiques. Eight default principles (P1–P8) plus meta-principles and
  operational rules.
- `autoconstitution.cai.hierarchy` — `StudentAgent`, `JudgeAgent`,
  `MetaJudgeAgent` role abstractions over any provider.
- `autoconstitution.cai.critique_revision` — the inner Student ↔ Judge loop
  with `CritiqueResult`, `RevisionResult`, `CritiqueRevisionLoop`.
- `autoconstitution.cai.preference_pairs` — `PreferencePair` and
  `PreferencePairBuilder` to convert CAI traces into DPO-ready JSONL, with
  anti-collapse anchor-dataset injection.
- `autoconstitution.cai.trl_trainer` — thin wrapper over HuggingFace TRL's
  `DPOTrainer`, with CUDA/MPS auto-detection and optional PEFT/LoRA.

### Added — Provider auto-detection

- `autoconstitution.providers.auto_detect.pick_provider()` tries Ollama → Kimi
  → Anthropic → OpenAI in order and returns a unified adapter.
- Each adapter exposes the same
  `async complete(prompt, system, temperature, max_tokens)` signature so CAI
  agents never care which backend is wired in.

### Added — CLI

- New `autoconstitution cai run` command: real critique-revision loop, not a
  simulation. Streams progress and writes a DPO-ready JSONL.
- New `autoconstitution cai providers` command: diagnostic table showing which
  providers are currently reachable.

### Added — Packaging

- Optional dependency groups in `pyproject.toml`: `[kimi]`, `[anthropic]`,
  `[openai]`, `[ollama]`, `[providers]`, `[train]`, `[cuda]`, `[mps]`, `[dev]`,
  `[all]`.
- `py.typed` marker file for PEP 561 type-checking.
- Pytest `asyncio_mode = "auto"` so async tests run without per-test
  decorators.

### Fixed — Crashes and bugs

- `asyncio.RLock()` (which does not exist in Python) replaced with
  `asyncio.Lock()` in five locations in `orchestrator.py`. Previously crashed
  at import time on any Python version.
- `self.history_window` → `self._history_window` in
  `orchestrator.py:632` — the attribute lookup was for a non-existent name.
- Removed hard-coded `/mnt/okcomputer/output/codebase` `sys.path` injection in
  `tests/test_orchestrator.py` — tests now rely on `pip install -e .`.

### Fixed — Cleanup

- Deleted `ratchet2.py` (byte-for-byte duplicate of `ratchet.py`).
- Normalised internal imports from `swarmresearch.*` to `autoconstitution.*`
  across all modules, tests, and examples.

### Changed — Public API

- `__init__.py` now re-exports the full orchestrator, ratchet, and config
  surface area (`SwarmOrchestrator`, `TaskDAG`, `Ratchet`, `MetricConfig`,
  `SwarmConfig`, `@task`, `@retryable`, plus all enums and exceptions).
- `__version__` bumped to `0.2.0`.

### Known limitations

- `run_dpo` requires the `[train]` extra and non-trivial GPU memory.
- Judge JSON parsing is forgiving but small local models sometimes return
  prose. Workaround: use a larger model (≥13B) for the Judge tier.
- Meta-Judge audit is implemented but not yet wired into automated
  constitution updates — planned for 0.3.

---

## [0.1.0] — initial release (historical)

- Initial SwarmResearch CLI, orchestrator, ratchet, metrics, provider adapters
  for Kimi/Anthropic/OpenAI/Ollama, hardware detection module.
