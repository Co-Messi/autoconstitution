"""Parallel startup probe for provider liveness.

``cai providers`` (and the ``cai run`` startup panel) uses this module to
answer one question per provider: *can I actually talk to you right now?*

For each provider we:

1. Check the minimum prerequisite (env var, reachable host, installed SDK).
2. If prereq passes, issue a real one-token completion with a short timeout.
3. Measure latency, capture any error, and report a compact status.

All four providers are probed in parallel via :class:`asyncio.TaskGroup`, so
the whole startup probe takes ~max(per_provider_latency), not sum.

Nothing here mutates global state; the probe is safe to call at CLI startup,
in a health check, or as the body of ``autoconstitution cai providers``.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Literal

import httpx

Status = Literal["ready", "no_key", "unreachable", "no_model", "sdk_missing", "error"]
"""The outcome of a single provider probe.

* ``ready`` — a one-token probe completed successfully
* ``no_key`` — API key env var not set
* ``unreachable`` — network or host failure (Ollama daemon, connection refused)
* ``no_model`` — Ollama daemon is up but has no models installed
* ``sdk_missing`` — prerequisite Python package not installed
* ``error`` — a real call failed (authentication rejected, 5xx, timeout, etc.)
"""


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """One row of the provider probe report."""

    name: Literal["ollama", "kimi", "anthropic", "openai"]
    status: Status
    detail: str
    latency_ms: float | None
    """Wall time of the probe call, or ``None`` when the probe didn't run
    (e.g. missing env var short-circuits before a network call)."""


# -----------------------------------------------------------------------------
# Per-provider probes — every one returns a ProbeResult, never raises.
# -----------------------------------------------------------------------------


async def _probe_ollama(
    *, host: str = "http://localhost:11434", timeout_s: float = 3.0
) -> ProbeResult:
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.get(f"{host}/api/tags")
    except httpx.ConnectError:
        return ProbeResult(
            name="ollama",
            status="unreachable",
            detail=f"daemon not reachable at {host}",
            latency_ms=None,
        )
    except (httpx.TimeoutException, httpx.HTTPError) as e:
        return ProbeResult(
            name="ollama",
            status="unreachable",
            detail=f"{type(e).__name__}: {e}",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
    if r.status_code != 200:
        return ProbeResult(
            name="ollama",
            status="unreachable",
            detail=f"HTTP {r.status_code} from {host}/api/tags",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
    try:
        models = r.json().get("models", [])
    except (ValueError, KeyError):
        models = []
    if not models:
        return ProbeResult(
            name="ollama",
            status="no_model",
            detail="daemon up, no models — try `ollama pull llama3.2`",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
    model_name = models[0].get("name", "?")
    return ProbeResult(
        name="ollama",
        status="ready",
        detail=f"model={model_name}",
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )


async def _probe_openai_compatible(
    *,
    name: Literal["openai", "kimi"],
    api_key: str | None,
    key_hint: str,
    base_url: str | None,
    model: str,
    timeout_s: float,
) -> ProbeResult:
    """Shared probe body for OpenAI and Kimi (both speak the OpenAI protocol)."""
    if not api_key:
        return ProbeResult(
            name=name,
            status="no_key",
            detail=f"{key_hint} not set",
            latency_ms=None,
        )
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return ProbeResult(
            name=name,
            status="sdk_missing",
            detail="`pip install openai` required",
            latency_ms=None,
        )
    start = time.perf_counter()
    try:
        client = (
            AsyncOpenAI(api_key=api_key, base_url=base_url)
            if base_url is not None
            else AsyncOpenAI(api_key=api_key)
        )
        await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
            ),
            timeout=timeout_s,
        )
    except TimeoutError:
        return ProbeResult(
            name=name,
            status="error",
            detail=f"timeout after {timeout_s}s",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
    except Exception as e:  # SDK raises many error classes; we treat all as opaque.
        return ProbeResult(
            name=name,
            status="error",
            detail=f"{type(e).__name__}: {str(e)[:120]}",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
    return ProbeResult(
        name=name,
        status="ready",
        detail=f"model={model}",
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )


async def _probe_kimi(*, timeout_s: float = 5.0) -> ProbeResult:
    key = os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY")
    return await _probe_openai_compatible(
        name="kimi",
        api_key=key,
        key_hint="MOONSHOT_API_KEY / KIMI_API_KEY",
        base_url="https://api.moonshot.cn/v1",
        model=os.environ.get("KIMI_MODEL", "moonshot-v1-8k"),
        timeout_s=timeout_s,
    )


async def _probe_openai(*, timeout_s: float = 5.0) -> ProbeResult:
    key = os.environ.get("OPENAI_API_KEY")
    return await _probe_openai_compatible(
        name="openai",
        api_key=key,
        key_hint="OPENAI_API_KEY",
        base_url=None,
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        timeout_s=timeout_s,
    )


async def _probe_anthropic(*, timeout_s: float = 5.0) -> ProbeResult:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return ProbeResult(
            name="anthropic",
            status="no_key",
            detail="ANTHROPIC_API_KEY not set",
            latency_ms=None,
        )
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        return ProbeResult(
            name="anthropic",
            status="sdk_missing",
            detail="`pip install anthropic` required",
            latency_ms=None,
        )
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    start = time.perf_counter()
    try:
        client = AsyncAnthropic(api_key=key)
        await asyncio.wait_for(
            client.messages.create(
                model=model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            ),
            timeout=timeout_s,
        )
    except TimeoutError:
        return ProbeResult(
            name="anthropic",
            status="error",
            detail=f"timeout after {timeout_s}s",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
    except Exception as e:
        return ProbeResult(
            name="anthropic",
            status="error",
            detail=f"{type(e).__name__}: {str(e)[:120]}",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
    return ProbeResult(
        name="anthropic",
        status="ready",
        detail=f"model={model}",
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )


# -----------------------------------------------------------------------------
# Public entry point — the whole probe, in parallel.
# -----------------------------------------------------------------------------


async def probe_all(*, timeout_s: float = 5.0) -> list[ProbeResult]:
    """Run every provider probe in parallel and return results in a stable order.

    Order: ollama, kimi, anthropic, openai. Each probe is self-contained and
    catches its own exceptions, so one failure never affects another.
    """
    ollama_task: asyncio.Task[ProbeResult]
    kimi_task: asyncio.Task[ProbeResult]
    anthropic_task: asyncio.Task[ProbeResult]
    openai_task: asyncio.Task[ProbeResult]

    async with asyncio.TaskGroup() as tg:
        ollama_task = tg.create_task(_probe_ollama(timeout_s=timeout_s))
        kimi_task = tg.create_task(_probe_kimi(timeout_s=timeout_s))
        anthropic_task = tg.create_task(_probe_anthropic(timeout_s=timeout_s))
        openai_task = tg.create_task(_probe_openai(timeout_s=timeout_s))

    return [ollama_task.result(), kimi_task.result(), anthropic_task.result(), openai_task.result()]


# -----------------------------------------------------------------------------
# Convenience predicates for downstream code.
# -----------------------------------------------------------------------------


def any_ready(results: list[ProbeResult]) -> bool:
    """True if at least one provider is ``ready``."""
    return any(r.status == "ready" for r in results)


def ready_names(results: list[ProbeResult]) -> list[str]:
    """Names of the providers that passed probe, in probe order."""
    return [r.name for r in results if r.status == "ready"]


__all__ = [
    "ProbeResult",
    "Status",
    "any_ready",
    "probe_all",
    "ready_names",
]
