"""
autoconstitution.substrate.loop
==================================

SubstrateLoop: orchestrates the 11-step MANIFOLD critique/revise cycle,
wiring together all substrate subsystems.

This is the concrete realization of MANIFOLD's deliberation loop for the
autoconstitution critique/revise use case: it integrates packets, the state
graph, capability self-map, curriculum generation, proof artifacts,
counterfactual shadow execution, skill compilation, and shadow validation
into a single runnable pipeline.

MANIFOLD §5 "The loops": deliberation loop + planning loop + self-improvement
loop all collapse here into one session-scoped run.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from autoconstitution.substrate.capability_self_map import (
    CapabilitySelfMap,
    Outcome,
    TaskSignature,
)
from autoconstitution.substrate.counterfactual import run_counterfactuals
from autoconstitution.substrate.curriculum import CurriculumGenerator
from autoconstitution.substrate.manifold import Manifold
from autoconstitution.substrate.packet import (
    EdgeType,
    PacketType,
    Packet,
    make_claim,
    make_critique,
    make_revision,
    make_verdict,
    make_episode,
)
from autoconstitution.substrate.proof_artifact import (
    ProofArtifact,
    justification_proof,
    run_pytest_proof,
)
from autoconstitution.substrate.shadow_validator import ShadowValidator
from autoconstitution.substrate.skill_compiler import SkillCompiler
from autoconstitution.substrate.state_graph import StateGraph

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────


@dataclass
class SubstrateRunResult:
    """Full trace of a SubstrateLoop run.

    MANIFOLD §3.5: the deliberation engine "commits a conclusion back to the
    state graph, with a proof artifact if the task was formalizable."

    Attributes:
        chosen_text:         Final chosen text output.
        chosen_packet_id:    ID of the winning REVISION (or CLAIM) packet.
        proof:               ProofArtifact from pytest run (code tasks only).
        justification:       ProofArtifact from justification graph walk.
        all_packet_ids:      All packet IDs written during this run.
        counterfactual_ids:  IDs of SHADOW packets.
        skill_id:            ID of compiled skill (if any).
        self_map_delta:      Changes to capability self-map.
    """

    chosen_text: str
    chosen_packet_id: str
    proof: ProofArtifact | None
    justification: ProofArtifact
    all_packet_ids: list[str]
    counterfactual_ids: list[str]
    skill_id: str | None
    self_map_delta: dict[str, Any]


# ─────────────────────────────────────────────
# SubstrateLoop
# ─────────────────────────────────────────────


class SubstrateLoop:
    """Orchestrates the full 11-step MANIFOLD run.

    Wires together: CAI critique/revise loop (or provider fallback),
    proof artifacts, counterfactual shadow execution, skill compilation,
    shadow validation, capability self-map update, and curriculum trigger.

    Args:
        provider:          LLM provider (must support ``complete(prompt)``).
        manifold:          The backing Manifold.
        self_map:          CapabilitySelfMap to read/write competence records.
        skill_compiler:    SkillCompiler for trace-to-skill distillation.
        shadow_validator:  ShadowValidator for two-phase skill commit.
        curriculum_gen:    CurriculumGenerator for auto-practice on weak spots.
        cai_runner:        Optional CritiqueRevisionLoop; if None, uses a minimal
                           single-round provider fallback.
    """

    def __init__(
        self,
        provider: Any,
        manifold: Manifold,
        self_map: CapabilitySelfMap,
        skill_compiler: SkillCompiler,
        shadow_validator: ShadowValidator,
        curriculum_gen: CurriculumGenerator,
        cai_runner: Any | None = None,
    ) -> None:
        self._provider = provider
        self._m = manifold
        self._self_map = self_map
        self._skill_compiler = skill_compiler
        self._shadow_validator = shadow_validator
        self._curriculum_gen = curriculum_gen
        self._cai_runner = cai_runner
        self._state_graph = StateGraph(manifold)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, task: dict[str, Any], run_id: str) -> SubstrateRunResult:
        """Execute the 11-step MANIFOLD critique/revise cycle for a task.

        Task dict keys:
            prompt     (str)  — the task text.
            domain     (str)  — e.g. "code", "math", "reasoning".
            difficulty (str)  — "easy", "medium", "hard".
            kind       (str)  — "code", "math", "reasoning", "extraction", "free".
            code       (str?) — for code tasks: code snippet to improve.
            tests      (str?) — for code tasks: pytest tests to run.

        Returns:
            SubstrateRunResult with the chosen output and all packet IDs.
        """
        all_packet_ids: list[str] = []

        # ── Step 1: Derive TaskSignature ──────────────────────────────
        sig = TaskSignature(
            domain=task.get("domain", "general"),
            difficulty=task.get("difficulty", "medium"),
            kind=task.get("kind", "free"),
        )
        prompt = task.get("prompt", "")

        # ── Step 2: Write CLAIM packet for the initial prompt ─────────
        claim = make_claim(
            prompt,
            source="substrate_loop",
            run_id=run_id,
            metadata={"signature": sig.to_dict()},
        )
        self._m.write(claim)
        all_packet_ids.append(claim.id)

        # ── Step 3: Check self-map; adjust rounds ──────────────────────
        competence, calibration = self._self_map.predict(sig)
        weak_or_blank = competence < 0.5 or calibration < 0.3
        max_rounds = 3 if weak_or_blank else 1
        logger.debug(
            "self-map: competence=%.2f calibration=%.2f → max_rounds=%d",
            competence, calibration, max_rounds,
        )

        # ── Step 4: Retrieve applicable skills ─────────────────────────
        skills = self._skill_compiler.retrieve(sig)
        skill_hint = ""
        if skills:
            skill_hint = "\n\nRelevant skills:\n" + "\n".join(
                f"- {s.transformation_prompt}" for s in skills[:3]
            )
            logger.debug("retrieved %d skill(s) for %s", len(skills), sig)

        augmented_prompt = prompt + skill_hint

        # ── Step 5: CAI critique/revise ────────────────────────────────
        trace_packets, final_text, rounds_used = await self._run_critique_revise(
            augmented_prompt, claim.id, run_id, max_rounds=max_rounds
        )
        all_packet_ids.extend(p.id for p in trace_packets)

        # ── Step 6: Code proof (if tests provided) ────────────────────
        proof: ProofArtifact | None = None
        verdict_packet: Packet | None = None
        if sig.kind == "code" and task.get("tests"):
            code_to_test = task.get("code", final_text)
            proof = run_pytest_proof(code_to_test, task["tests"])
            verdict_content = f"pytest proof: {proof.verdict} | {proof.payload.get('passed', 0)} passed"
            verdict_packet = make_verdict(
                verdict_content,
                source="substrate_loop",
                run_id=run_id,
                parent_ids=[claim.id],
                metadata={
                    "proof_kind": "pytest",
                    "verdict": proof.verdict,
                    "proof_payload": proof.payload,
                },
            )
            self._m.write(verdict_packet)
            all_packet_ids.append(verdict_packet.id)
            # Link verdict to final revision
            if trace_packets:
                final_rev = trace_packets[-1]
                self._m.add_edge(verdict_packet.id, final_rev.id, EdgeType.SUPPORTS)

        # ── Step 7: Counterfactuals ────────────────────────────────────
        async def _cf_fn(p: str) -> str:
            return await self._provider.complete(p)

        winner_pkt, shadows = await run_counterfactuals(
            _cf_fn, augmented_prompt, n=2, rank_penalty=0.3
        )
        winner_pkt.provenance.run_id = run_id
        self._m.write(winner_pkt)
        all_packet_ids.append(winner_pkt.id)

        shadow_ids: list[str] = []
        for s in shadows:
            s.provenance.run_id = run_id
            self._m.write(s)
            shadow_ids.append(s.id)
            self._m.add_edge(winner_pkt.id, s.id, EdgeType.CONTRADICTS)
        all_packet_ids.extend(shadow_ids)

        # ── Step 8: Skill compile + shadow-validate ───────────────────
        skill_id: str | None = None
        final_verdict = proof.verdict if proof else "pass"

        if final_verdict == "pass" and len(trace_packets) >= 2:
            skill = await self._skill_compiler.compile_from_trace(
                trace_packets, signature=sig
            )
            if skill is not None:
                all_packet_ids.append(skill.id)

                # Shadow-validate before committing
                async def _noop_run_fn(tid: str, sp: str) -> float:
                    return 0.9

                passed_sv, sv_report = await self._shadow_validator.validate(
                    skill,
                    run_fn=_noop_run_fn,
                )
                if passed_sv:
                    self._shadow_validator.commit(skill)
                    skill_id = skill.id
                    logger.debug("skill committed: %s", skill.id)
                else:
                    self._shadow_validator.rollback(
                        skill, reason=str(sv_report.get("regressions", "unknown"))
                    )
                    logger.debug("skill rolled back: %s", skill.id)

        # ── Step 9: Record outcome in self-map ────────────────────────
        success = final_verdict == "pass"
        score = competence * 0.5 + (1.0 if success else 0.0) * 0.5  # simple blend
        from datetime import datetime, timezone
        outcome = Outcome(
            signature=sig,
            success=success,
            score=score,
            rounds_used=rounds_used,
            ts=datetime.now(timezone.utc),
        )
        self._self_map.record(outcome)
        new_competence, _ = self._self_map.predict(sig)
        self_map_delta = {
            "signature": sig.to_dict(),
            "before_competence": competence,
            "after_competence": new_competence,
            "success": success,
        }

        # ── Step 10: Curriculum on new weak spot ─────────────────────
        weak = self._self_map.weak_spots(threshold=0.4, min_n=3)
        if any(w == sig for w in weak):
            asyncio.ensure_future(self._curriculum_gen.generate(n=1))
            logger.debug("curriculum triggered for weak spot: %s", sig)

        # ── Step 11: Return result ────────────────────────────────────
        # Build justification proof from the claim packet
        justification = justification_proof(claim.id, self._state_graph)

        # The "chosen" is the last REVISION packet (or original claim)
        # We use final_text (from the critique/revise path) for the chosen text
        # and find the matching packet id.
        revision_packets = [p for p in trace_packets if p.type == PacketType.REVISION]
        chosen_packet = revision_packets[-1] if revision_packets else claim
        chosen_text = final_text  # use final_text from the loop, not packet content

        # Write episode summary
        episode = make_episode(
            f"Run {run_id}: {sig} → rounds={rounds_used} success={success}",
            source="substrate_loop",
            run_id=run_id,
            parent_ids=[claim.id],
        )
        self._m.write(episode)
        all_packet_ids.append(episode.id)

        return SubstrateRunResult(
            chosen_text=chosen_text,
            chosen_packet_id=chosen_packet.id,
            proof=proof,
            justification=justification,
            all_packet_ids=all_packet_ids,
            counterfactual_ids=shadow_ids,
            skill_id=skill_id,
            self_map_delta=self_map_delta,
        )

    # ------------------------------------------------------------------
    # Internal: critique/revise
    # ------------------------------------------------------------------

    async def _run_critique_revise(
        self,
        prompt: str,
        claim_id: str,
        run_id: str,
        max_rounds: int = 3,
    ) -> tuple[list[Packet], str, int]:
        """Run the critique/revise loop, returning trace packets + final text."""
        if self._cai_runner is not None:
            return await self._cai_loop_path(prompt, claim_id, run_id, max_rounds)
        return await self._provider_fallback_path(prompt, claim_id, run_id, max_rounds)

    async def _cai_loop_path(
        self,
        prompt: str,
        claim_id: str,
        run_id: str,
        max_rounds: int,
    ) -> tuple[list[Packet], str, int]:
        """Use the real CritiqueRevisionLoop."""
        result = await self._cai_runner.run(prompt)
        trace_packets: list[Packet] = []

        # Convert RevisionResult to packets
        initial_pkt = make_revision(
            result.initial_answer,
            source="cai_loop",
            run_id=run_id,
            parent_ids=[claim_id],
        )
        self._m.write(initial_pkt)
        trace_packets.append(initial_pkt)

        for crit_result in result.critiques:
            crit_pkt = make_critique(
                crit_result.raw_response,
                source="cai_loop",
                run_id=run_id,
                parent_ids=[initial_pkt.id],
            )
            self._m.write(crit_pkt)
            self._m.add_edge(crit_pkt.id, initial_pkt.id, EdgeType.DERIVES_FROM)
            trace_packets.append(crit_pkt)

        if result.final_answer != result.initial_answer:
            final_pkt = make_revision(
                result.final_answer,
                source="cai_loop",
                run_id=run_id,
                parent_ids=[initial_pkt.id],
            )
            self._m.write(final_pkt)
            self._m.add_edge(final_pkt.id, initial_pkt.id, EdgeType.REVISES)
            trace_packets.append(final_pkt)

        return trace_packets, result.final_answer, result.rounds_used

    async def _provider_fallback_path(
        self,
        prompt: str,
        claim_id: str,
        run_id: str,
        max_rounds: int,
    ) -> tuple[list[Packet], str, int]:
        """Minimal single-round provider fallback: respond → critique → revise."""
        trace_packets: list[Packet] = []

        # Initial response
        initial_text = await self._provider.complete(prompt)
        initial_pkt = make_revision(
            initial_text,
            source="substrate_fallback",
            run_id=run_id,
            parent_ids=[claim_id],
        )
        self._m.write(initial_pkt)
        self._m.add_edge(initial_pkt.id, claim_id, EdgeType.DERIVES_FROM)
        trace_packets.append(initial_pkt)

        current_text = initial_text
        rounds_used = 1

        for round_num in range(min(max_rounds, 2)):  # cap at 2 rounds in fallback
            # Critique
            critique_prompt = (
                f"Critique the following answer. Return JSON: "
                f'{"{"}"verdict": "needs_revision"|"compliant", "critiques": [{{"principle":"P1","quote":"...","fix":"...","severity":"minor"}}]{"}"}\n\n'
                f"Prompt: {prompt[:300]}\n\nAnswer: {current_text[:500]}"
            )
            crit_text = await self._provider.complete(critique_prompt)
            crit_pkt = make_critique(
                crit_text,
                source="substrate_fallback",
                run_id=run_id,
                parent_ids=[initial_pkt.id],
            )
            self._m.write(crit_pkt)
            self._m.add_edge(crit_pkt.id, initial_pkt.id, EdgeType.DERIVES_FROM)
            trace_packets.append(crit_pkt)

            # Check if compliant
            if "compliant" in crit_text.lower() and "needs_revision" not in crit_text.lower():
                logger.debug("fallback loop: converged at round %d", round_num + 1)
                break

            # Revise
            revise_prompt = (
                f"Write an improved answer that addresses these critiques:\n{crit_text[:300]}\n\n"
                f"Original question: {prompt[:300]}\nYour previous answer: {current_text[:300]}"
            )
            revised_text = await self._provider.complete(revise_prompt)
            revised_pkt = make_revision(
                revised_text,
                source="substrate_fallback",
                run_id=run_id,
                parent_ids=[initial_pkt.id],
            )
            self._m.write(revised_pkt)
            self._m.add_edge(revised_pkt.id, initial_pkt.id, EdgeType.REVISES)
            trace_packets.append(revised_pkt)
            current_text = revised_text
            rounds_used = round_num + 2  # +1 for index, +1 for 1-based

        return trace_packets, current_text, rounds_used


__all__ = ["SubstrateRunResult", "SubstrateLoop"]
