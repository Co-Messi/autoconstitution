# autoconstitution.substrate

MANIFOLD shared-state substrate for the autoconstitution critique/revise loop.

This subpackage is a concrete realization of the
[MANIFOLD unified architecture](../../../../Teno/Ideal\ AI\ Architectures/MANIFOLD-unified-architecture.md),
specialized for autoconstitution.  All seven MANIFOLD subsystems and all
eight features are mapped to Python modules; see the tables below.

---

## Quick start

No network or Ollama required.

```bash
# Run the built-in demo (mock provider, 3 tasks, full pipeline):
python -m autoconstitution substrate demo

# Run a single prompt (falls back to mock if no real provider):
python -m autoconstitution substrate run --prompt "Explain memoization."

# Inspect the persisted state:
python -m autoconstitution substrate status
python -m autoconstitution substrate capabilities

# Curriculum management:
python -m autoconstitution substrate curriculum list
python -m autoconstitution substrate curriculum generate --n 5
python -m autoconstitution substrate curriculum next

# Revoke a packet:
python -m autoconstitution substrate forget --id <uuid>
```

---

## Module map

### Seven MANIFOLD subsystems

| MANIFOLD subsystem | Module(s) | What it does |
|---|---|---|
| Event Compiler | `packet.py` | 9 typed latent-state packets (CLAIM, CRITIQUE, REVISION, VERDICT, GOAL, FACT, EPISODE, SKILL, SHADOW); 6 edge types; confidence decay with belief half-life. |
| Streaming Backbone | *(deferred)* | Would be the async I/O bus; currently packets flow synchronously inside `loop.py`. |
| Persistent Causal State Graph | `manifold.py` + `state_graph.py` | SQLite-backed packet store with causal edges; BFS justification graph; contradiction detection; stale-node queries; revision lineage. |
| Multi-Graph Memory (MAGMA) | `manifold.py` | Single SQLite DB hosts all sub-graphs (episodic, semantic, procedural, provenance) via the unified `packets`/`edges` schema. |
| Latent Deliberation Engine | `loop.py` + `counterfactual.py` | 11-step critique/revise cycle; async counterfactual shadow branches (`asyncio.gather`); winner selection by rank. |
| Neuro-Symbolic Substrate | `proof_artifact.py` | Subprocess-isolated `pytest` proofs for code tasks; justification-graph proofs for free-form claims; `ProofArtifact` carries verdict + payload. |
| Metacognitive Controller | `capability_self_map.py` + `curriculum.py` | EWMA competence model per `TaskSignature`; auto-generates practice GOAL packets for weak spots. |
| Inter-agent Protocol | *(deferred)* | Needs cryptographic signing; SKILL packets are the logical unit for future agent-to-agent transfer. |

### Eight MANIFOLD features

| Feature | Where implemented |
|---|---|
| 1. Belief half-life | `Packet.current_confidence(now)` — `conf * 0.5^(elapsed / half_life_seconds)` |
| 2. Counterfactual shadow execution | `counterfactual.run_counterfactuals()` — N async branches, winner + ranked shadows |
| 3. Capability self-map | `CapabilitySelfMap` — EWMA score, calibration, weak-spot detection |
| 4. Proof-carrying outputs | `ProofArtifact`, `run_pytest_proof`, `justification_proof` |
| 5. Skill deprecation / quarantine | `SkillCompiler.update_stats()` — rolling window of 5 uses, quarantine at < 30 % success |
| 6. Two-phase adaptation with rollback | `ShadowValidator.validate()` / `commit()` / `rollback()` |
| 7. Inter-agent protocol | *(deferred — SKILL packets are the logical transfer unit)* |
| 8. Curriculum self-generation | `CurriculumGenerator.generate()` — synthesizes practice problems for weak TaskSignatures |

---

## Architecture

```
SubstrateLoop.run(task)
│
├─ Step 1  Derive TaskSignature (domain × difficulty × kind)
├─ Step 2  Write CLAIM packet → Manifold
├─ Step 3  Query CapabilitySelfMap → set max_rounds (1 if strong, 3 if weak/blank)
├─ Step 4  Retrieve applicable Skills → augment prompt
├─ Step 5  Critique/revise loop → CRITIQUE + REVISION packets → Manifold
├─ Step 6  (code tasks) run_pytest_proof → VERDICT packet → Manifold
├─ Step 7  run_counterfactuals → winner REVISION + SHADOW packets → Manifold
├─ Step 8  compile_from_trace → SKILL packet
│          └─ ShadowValidator.validate → commit or rollback
├─ Step 9  CapabilitySelfMap.record(Outcome)
├─ Step 10 (weak spot) asyncio.ensure_future(CurriculumGenerator.generate)
└─ Step 11 justification_proof → return SubstrateRunResult
```

### Persistence

All packets and edges go into a single SQLite database (default:
`~/.autoconstitution/substrate.db`, or a temp file for the demo).
Schema is initialized idempotently on first `Manifold()` construction.

```sql
-- packets: id, type, content, confidence, half_life_seconds,
--          valid_from, revoked_by, metadata_json, embedding_blob
-- edges:   src_id, dst_id, edge_type, created_at
-- schema_version: (single row)
```

Revoked packets carry `revoked_by = "__revoked__"` so the
`WHERE revoked_by IS NULL` filter consistently excludes them whether or
not a superseding packet exists.

### Confidence decay

```python
conf * 0.5 ** (elapsed_seconds / half_life_seconds)
```

Default half-lives:

| Packet type | Half-life |
|---|---|
| CLAIM | 86 400 s (1 day) |
| CRITIQUE | 43 200 s (12 h) |
| SHADOW | 300 s (5 min) |
| VERDICT, FACT, SKILL, EPISODE, GOAL, REVISION | None (no decay) |

### Pseudo-embedding

Every packet gets a deterministic 256-dimensional embedding derived from
SHA-256 of its content (no model call required):

```python
digest = hashlib.sha256(text.encode()).digest()
floats = struct.unpack(f"{256}f", digest * (256 * 4 // 32 + 1))[:256]
vec = np.array(floats, dtype=np.float32)
vec /= np.linalg.norm(vec) + 1e-9
```

`Manifold.neighbors(vector, k)` uses cosine similarity over all stored
embeddings (O(n), fine for thousands of packets; index later if needed).

---

## File list

| File | Purpose |
|---|---|
| `__init__.py` | Public API re-exports |
| `packet.py` | `Packet`, `PacketType`, `EdgeType`, `Provenance`; 9 factory helpers |
| `manifold.py` | `Manifold` — SQLite-backed packet/edge store |
| `state_graph.py` | `StateGraph` — BFS queries over the causal graph |
| `capability_self_map.py` | `CapabilitySelfMap`, `TaskSignature`, `Outcome`, `TrackRecord` |
| `curriculum.py` | `CurriculumGenerator` — auto-generates GOAL packets |
| `proof_artifact.py` | `ProofArtifact`, `run_pytest_proof`, `justification_proof` |
| `counterfactual.py` | `run_counterfactuals` — async shadow branch execution |
| `skill_compiler.py` | `Skill`, `SkillCompiler` — trace-to-skill distillation + quarantine |
| `shadow_validator.py` | `ShadowValidator` — two-phase skill validation |
| `loop.py` | `SubstrateLoop`, `SubstrateRunResult` — 11-step orchestrator |
| `cli.py` | `substrate_app` Typer subgroup wired into `autoconstitution` CLI |
| `README.md` | This file |

Tests live in `tests/substrate/` (124 tests, all passing).

---

## MVP vs deferred

**Implemented (MVP):**
- All 9 packet types, 6 edge types, confidence decay with half-life
- SQLite persistence, idempotent schema, atomic writes
- Justification graph, contradiction detection, stale nodes, lineage chains
- EWMA competence model, weak-spot detection, blank-region detection
- Curriculum synthesis (prompts weak spots with a synthesize call)
- Subprocess-isolated pytest proofs + justification-graph proofs
- Async counterfactual shadow execution (N branches, winner + ranks)
- Skill distillation from critique/revise traces, quarantine on drift
- Two-phase shadow validation (validate → commit or rollback)
- Full 11-step SubstrateLoop orchestrator
- Typer CLI subgroup: `run`, `status`, `capabilities`, `curriculum`, `forget`, `demo`
- Deterministic pseudo-embedding (no model call for storage/retrieval)

**Deferred:**
- Streaming backbone (async I/O bus between subsystems)
- Cryptographic inter-agent signing for SKILL packet transfer
- Vector index (FAISS/hnswlib) for large-scale neighbor search
- Real provider integration in the CLI `run` command (falls back to mock)
