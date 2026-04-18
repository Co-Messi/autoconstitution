"""Tests for providers/probe.py and ui/probe_view.py.

These tests are self-contained: we monkeypatch the per-provider probe
coroutines so no real network traffic ever happens.
"""

from __future__ import annotations

from typing import Any

import pytest

from autoconstitution.providers import probe as probe_module
from autoconstitution.providers.probe import (
    ProbeResult,
    any_ready,
    probe_all,
    ready_names,
)
from autoconstitution.ui.probe_view import (
    render_no_provider_panel,
    render_probe_table,
)


def _result(
    name: str,
    status: str = "ready",
    detail: str = "ok",
    latency_ms: float | None = 42.0,
) -> ProbeResult:
    return ProbeResult(name=name, status=status, detail=detail, latency_ms=latency_ms)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_probe_all_runs_four_probes_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_ollama(**kw: Any) -> ProbeResult:
        return _result("ollama")

    async def _fake_kimi(**kw: Any) -> ProbeResult:
        return _result("kimi", status="no_key", detail="MOONSHOT_API_KEY not set", latency_ms=None)

    async def _fake_anthropic(**kw: Any) -> ProbeResult:
        return _result("anthropic", status="error", detail="401 unauthorized")

    async def _fake_openai(**kw: Any) -> ProbeResult:
        return _result("openai", status="ready", detail="model=gpt-4o-mini")

    monkeypatch.setattr(probe_module, "_probe_ollama", _fake_ollama)
    monkeypatch.setattr(probe_module, "_probe_kimi", _fake_kimi)
    monkeypatch.setattr(probe_module, "_probe_anthropic", _fake_anthropic)
    monkeypatch.setattr(probe_module, "_probe_openai", _fake_openai)

    results = await probe_all(timeout_s=1.0)
    assert [r.name for r in results] == ["ollama", "kimi", "anthropic", "openai"]
    assert [r.status for r in results] == ["ready", "no_key", "error", "ready"]


def test_any_ready_and_ready_names() -> None:
    results = [
        _result("ollama", status="ready"),
        _result("kimi", status="no_key", latency_ms=None),
        _result("openai", status="ready"),
    ]
    assert any_ready(results) is True
    assert ready_names(results) == ["ollama", "openai"]


def test_any_ready_false_when_nothing_ready() -> None:
    results = [
        _result("ollama", status="unreachable", latency_ms=None),
        _result("kimi", status="no_key", latency_ms=None),
        _result("anthropic", status="error"),
        _result("openai", status="no_key", latency_ms=None),
    ]
    assert any_ready(results) is False
    assert ready_names(results) == []


def test_render_probe_table_produces_rich_renderable() -> None:
    results = [
        _result("ollama", status="ready", detail="model=llama3.2"),
        _result("kimi", status="no_key", latency_ms=None),
        _result("anthropic", status="error", detail="timeout"),
        _result("openai", status="ready"),
    ]
    table = render_probe_table(results)
    # Row count = number of results
    assert table.row_count == 4


def test_render_no_provider_panel_returns_panel_with_install_commands() -> None:
    panel = render_no_provider_panel()
    rendered = panel.renderable
    # The Text object holds the body; check it mentions every install path.
    text = rendered.plain if hasattr(rendered, "plain") else str(rendered)
    assert "ollama" in text.lower()
    assert "MOONSHOT_API_KEY" in text
    assert "ANTHROPIC_API_KEY" in text
    assert "OPENAI_API_KEY" in text


@pytest.mark.asyncio
async def test_probe_ollama_returns_unreachable_when_daemon_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integration-ish: with a bogus host, the probe should report unreachable."""
    result = await probe_module._probe_ollama(
        host="http://127.0.0.1:1",  # nobody listens here
        timeout_s=0.5,
    )
    assert result.name == "ollama"
    assert result.status == "unreachable"


@pytest.mark.asyncio
async def test_probe_kimi_returns_no_key_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    result = await probe_module._probe_kimi(timeout_s=0.5)
    assert result.status == "no_key"
    assert result.latency_ms is None


@pytest.mark.asyncio
async def test_probe_openai_returns_no_key_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = await probe_module._probe_openai(timeout_s=0.5)
    assert result.status == "no_key"


@pytest.mark.asyncio
async def test_probe_anthropic_returns_no_key_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = await probe_module._probe_anthropic(timeout_s=0.5)
    assert result.status == "no_key"
