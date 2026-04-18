"""Tests for the _pick_student_and_judge helper and its CLI wiring.

Ensures:
- Symmetric path (no judge flags) reuses one ProviderChoice — zero regression.
- Asymmetric flags route to two separate pick_provider calls with the right args.
- --judge-model without --judge-provider inherits Student's provider name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from autoconstitution import cli


@dataclass
class _FakeChoice:
    name: str
    model: str
    provider: object
    reason: str = "fake"


@pytest.mark.asyncio
async def test_no_judge_flags_reuses_student_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def _fake_pick(
        *, prefer: list[str] | None = None, ollama_model: str | None = None,
        **_: Any,
    ) -> _FakeChoice:
        calls.append({"prefer": prefer, "ollama_model": ollama_model})
        return _FakeChoice(
            name=prefer[0] if prefer else "ollama",
            model=ollama_model or "llama3.2:auto",
            provider=object(),
        )

    monkeypatch.setattr("autoconstitution.providers.pick_provider", _fake_pick)

    student, judge = await cli._pick_student_and_judge(
        student_provider="ollama",
        student_model="llama3.2:3b",
        judge_provider=None,
        judge_model=None,
        quiet=True,
    )
    assert student is judge
    assert len(calls) == 1
    assert calls[0] == {"prefer": ["ollama"], "ollama_model": "llama3.2:3b"}


@pytest.mark.asyncio
async def test_judge_provider_triggers_second_pick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def _fake_pick(
        *, prefer: list[str] | None = None, ollama_model: str | None = None,
        **_: Any,
    ) -> _FakeChoice:
        calls.append({"prefer": prefer, "ollama_model": ollama_model})
        return _FakeChoice(
            name=prefer[0] if prefer else "ollama",
            model=ollama_model or "default",
            provider=object(),
        )

    monkeypatch.setattr("autoconstitution.providers.pick_provider", _fake_pick)

    student, judge = await cli._pick_student_and_judge(
        student_provider="ollama",
        student_model="llama3.2:1b",
        judge_provider="anthropic",
        judge_model=None,
        quiet=True,
    )
    assert student is not judge
    assert student.name == "ollama"
    assert judge.name == "anthropic"
    assert len(calls) == 2
    assert calls[0] == {"prefer": ["ollama"], "ollama_model": "llama3.2:1b"}
    assert calls[1] == {"prefer": ["anthropic"], "ollama_model": None}


@pytest.mark.asyncio
async def test_judge_model_alone_inherits_student_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--judge-model llama3.2:3b` without `--judge-provider` means same
    provider as Student, different model."""
    calls: list[dict[str, Any]] = []

    async def _fake_pick(
        *, prefer: list[str] | None = None, ollama_model: str | None = None,
        **_: Any,
    ) -> _FakeChoice:
        calls.append({"prefer": prefer, "ollama_model": ollama_model})
        return _FakeChoice(
            name=prefer[0] if prefer else "ollama",
            model=ollama_model or "default",
            provider=object(),
        )

    monkeypatch.setattr("autoconstitution.providers.pick_provider", _fake_pick)

    student, judge = await cli._pick_student_and_judge(
        student_provider="ollama",
        student_model="llama3.2:1b",
        judge_provider=None,
        judge_model="llama3.2:3b",
        quiet=True,
    )
    assert student.name == judge.name == "ollama"
    assert student.model == "llama3.2:1b"
    assert judge.model == "llama3.2:3b"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_autodetect_student_then_judge_inherits_autodetect_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Student is auto-detected (no --provider) and Judge has a model
    override, Judge should pick the same provider the Student landed on."""
    calls: list[dict[str, Any]] = []

    async def _fake_pick(
        *, prefer: list[str] | None = None, ollama_model: str | None = None,
        **_: Any,
    ) -> _FakeChoice:
        calls.append({"prefer": prefer, "ollama_model": ollama_model})
        # Student call has no prefer — simulate auto-detect picking Ollama.
        name = prefer[0] if prefer else "ollama"
        return _FakeChoice(
            name=name, model=ollama_model or "auto", provider=object(),
        )

    monkeypatch.setattr("autoconstitution.providers.pick_provider", _fake_pick)

    _student, _judge = await cli._pick_student_and_judge(
        student_provider=None,
        student_model=None,
        judge_provider=None,
        judge_model="llama3.2:3b",
        quiet=True,
    )
    # Second call should inherit Student's auto-detected name ("ollama").
    assert calls[1]["prefer"] == ["ollama"]
    assert calls[1]["ollama_model"] == "llama3.2:3b"
