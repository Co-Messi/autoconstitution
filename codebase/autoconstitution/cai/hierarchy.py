"""
Hierarchy of Constitutional AI agents: Student, Judge, Meta-Judge.

The three-tier structure is the core abstraction of autoconstitution:

    Student     — produces answers to user prompts.
    Judge       — critiques Student outputs against the constitution.
    Meta-Judge  — audits Judges for drift, collapse, and principle-invention.

Each tier wraps an LLM provider (Ollama/OpenAI/Claude/Kimi) and a role-specific
system prompt derived from `constitution.md`.

Example:
    >>> from autoconstitution.cai import StudentAgent, JudgeAgent
    >>> from autoconstitution.providers.ollama import OllamaProvider
    >>> student = StudentAgent(provider=OllamaProvider(model="llama3.1:8b"))
    >>> judge   = JudgeAgent(provider=OllamaProvider(model="llama3.1:70b"))
    >>> answer  = await student.respond("Why is the sky blue?")
    >>> critique = await judge.critique(prompt="Why is the sky blue?", answer=answer)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol


class CAIRole(str, enum.Enum):
    """The three tiers in a CAI hierarchy."""

    STUDENT = "student"
    JUDGE = "judge"
    META_JUDGE = "meta_judge"


class LLMProvider(Protocol):
    """Minimal protocol a provider must satisfy to plug into CAI agents."""

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Return the model's completion for ``prompt`` given an optional system prompt."""
        ...


# ---------- System prompts ------------------------------------------------

STUDENT_SYSTEM = """You are the Student.

Your job: produce clear, correct, useful answers to user prompts. You are being
trained by a Judge that will critique your answers against a written
constitution. You should:

- Answer the actual question.
- Be concrete, not hedging.
- Admit uncertainty when you're uncertain.
- Do not begin with throat-clearing ("Certainly!", "Great question!").
- Do not end with throat-clearing ("I hope this helps!").

You do not need to know the constitution — the Judge will tell you if you
violated it. Just focus on being genuinely helpful and honest.
"""

# NOTE: these templates use a `{CONSTITUTION}` marker that we substitute via
# str.replace — *not* str.format — because the constitution body may contain
# JSON braces that format() would treat as field names and crash on.
JUDGE_SYSTEM_TEMPLATE = """You are the Judge.

Your job: critique the Student's answer against the constitution below. For
each critique:

1. Cite the specific principle (e.g., "P5 — Concision").
2. Quote the offending span from the Student's answer.
3. Say *exactly* what should change.
4. Rate severity: minor / moderate / major.

If the Student's answer is constitution-compliant, return:
    {"verdict": "compliant", "critiques": []}

Otherwise return a JSON object:
    {
      "verdict": "needs_revision",
      "critiques": [
        {"principle": "P5", "quote": "...", "fix": "...", "severity": "moderate"}
      ]
    }

Do not invent principles. Do not moralize. Cite only what is in the constitution.

--- CONSTITUTION ---
__CONSTITUTION__
"""

META_JUDGE_SYSTEM_TEMPLATE = """You are the Meta-Judge.

Your job: audit the Judge for drift, collapse, and principle-invention. Given
a batch of (prompt, student_answer, judge_critique) triples, you:

- Flag critiques that invent principles not in the constitution.
- Flag critiques that are disproportional (severity doesn't match issue).
- Flag collapse (the Judge giving similar verdicts across diverse prompts).
- Propose updates to the constitution only when you see a real gap — never
  to "fix" the Judge's individual mistakes.

Return JSON:
    {
      "judge_health": "ok" | "drifting" | "collapsed",
      "flagged_critiques": [indices],
      "proposed_principles": []
    }

--- CONSTITUTION ---
__CONSTITUTION__
"""


# ---------- Agents --------------------------------------------------------


@dataclass
class _AgentBase:
    """Shared machinery: holds a provider, a role, and a system prompt."""

    provider: LLMProvider
    role: CAIRole
    system_prompt: str
    temperature: float = 0.7
    max_tokens: int = 2048
    metadata: dict[str, Any] = field(default_factory=dict)

    async def _ask(self, user_prompt: str, **overrides: Any) -> str:
        """Dispatch to the provider with role-appropriate defaults."""
        return await self.provider.complete(
            prompt=user_prompt,
            system=self.system_prompt,
            temperature=overrides.get("temperature", self.temperature),
            max_tokens=overrides.get("max_tokens", self.max_tokens),
        )


def _load_constitution(constitution_path: Optional[Path] = None) -> str:
    """Load the constitution markdown. Falls back to a minimal embedded default.

    This avoids a hard dependency on an installed `constitution.md`, which matters
    for `pip install autoconstitution` without the source tree.
    """
    if constitution_path and constitution_path.exists():
        return constitution_path.read_text(encoding="utf-8")

    # Try the packaged copy next to this file.
    default_path = Path(__file__).parent.parent / "constitution.md"
    if default_path.exists():
        return default_path.read_text(encoding="utf-8")

    # Try project root (development install).
    project_root = Path(__file__).resolve().parent.parent.parent
    root_path = project_root / "constitution.md"
    if root_path.exists():
        return root_path.read_text(encoding="utf-8")

    # Minimal fallback so the system still boots with no constitution file.
    return (
        "# Minimal Default Constitution\n\n"
        "P1 Truthfulness. P2 Harm avoidance. P3 Autonomy. P4 Calibration.\n"
        "P5 Concision. P6 Format fidelity. P7 Non-sycophancy. P8 Consistency.\n"
    )


class StudentAgent(_AgentBase):
    """Produces first-draft answers. Lowest tier."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_override: Optional[str] = None,
    ) -> None:
        super().__init__(
            provider=provider,
            role=CAIRole.STUDENT,
            system_prompt=system_override or STUDENT_SYSTEM,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def respond(self, prompt: str, **kwargs: Any) -> str:
        """Generate an answer to ``prompt``."""
        return await self._ask(prompt, **kwargs)

    async def revise(self, prompt: str, previous_answer: str, critique: str, **kwargs: Any) -> str:
        """Revise an answer given a Judge's critique."""
        revision_prompt = (
            f"Original question:\n{prompt}\n\n"
            f"Your previous answer:\n{previous_answer}\n\n"
            f"The Judge's critique:\n{critique}\n\n"
            f"Write an improved answer that addresses the critique. Do not "
            f"acknowledge the critique or the revision process — just produce the "
            f"better answer."
        )
        return await self._ask(revision_prompt, **kwargs)


class JudgeAgent(_AgentBase):
    """Critiques Student outputs against the constitution."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        constitution_path: Optional[Path] = None,
        temperature: float = 0.2,  # judges should be low-variance
        max_tokens: int = 2048,
    ) -> None:
        constitution = _load_constitution(constitution_path)
        super().__init__(
            provider=provider,
            role=CAIRole.JUDGE,
            system_prompt=JUDGE_SYSTEM_TEMPLATE.replace("__CONSTITUTION__", constitution),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._constitution = constitution

    async def critique(self, prompt: str, answer: str, **kwargs: Any) -> str:
        """Return a critique of the Student's ``answer`` to ``prompt``."""
        critique_prompt = (
            f"--- PROMPT ---\n{prompt}\n\n"
            f"--- STUDENT ANSWER ---\n{answer}\n\n"
            f"Critique the Student answer. Return JSON as specified."
        )
        return await self._ask(critique_prompt, **kwargs)


class MetaJudgeAgent(_AgentBase):
    """Audits Judges for drift, collapse, and principle-invention."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        constitution_path: Optional[Path] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        constitution = _load_constitution(constitution_path)
        super().__init__(
            provider=provider,
            role=CAIRole.META_JUDGE,
            system_prompt=META_JUDGE_SYSTEM_TEMPLATE.replace("__CONSTITUTION__", constitution),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._constitution = constitution

    async def audit(
        self,
        triples: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Audit a batch of (prompt, student_answer, judge_critique) triples."""
        audit_prompt = (
            "--- JUDGE OUTPUTS TO AUDIT ---\n"
            + "\n\n".join(
                f"[{i}] PROMPT: {t['prompt']}\n"
                f"    STUDENT: {t['student_answer']}\n"
                f"    JUDGE:   {t['judge_critique']}"
                for i, t in enumerate(triples)
            )
            + "\n\nReturn your audit JSON."
        )
        return await self._ask(audit_prompt, **kwargs)
