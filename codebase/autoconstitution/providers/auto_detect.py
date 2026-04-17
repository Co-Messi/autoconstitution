"""
Provider auto-detection.

Picks a usable LLM backend for the user *without requiring any configuration*.
Order of preference:

    1. Ollama — if the local daemon responds on http://localhost:11434.
       Free, offline, no key. Best default for the CAI loop.
    2. Kimi — if ``MOONSHOT_API_KEY`` or ``KIMI_API_KEY`` is set.
    3. Anthropic — if ``ANTHROPIC_API_KEY`` is set.
    4. OpenAI — if ``OPENAI_API_KEY`` is set.

Example:
    >>> from autoconstitution.providers import auto_detect
    >>> provider = await auto_detect.pick_provider()
    >>> # provider now satisfies the LLMProvider protocol used by CAI agents

The returned object is wrapped in a thin adapter so that every provider
exposes the same ``async complete(prompt, system, temperature, max_tokens)``
interface.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ProviderChoice:
    """What auto_detect returns: the provider instance + diagnostic info."""

    provider: Any  # satisfies LLMProvider protocol
    name: str  # "ollama" | "kimi" | "anthropic" | "openai"
    model: str
    reason: str


# ---------- Adapters: unify `.complete()` signature ----------------------


class _OllamaAdapter:
    """Wraps OllamaProvider to expose `.complete(prompt, system, ...)`."""

    def __init__(self, provider: Any, model: str) -> None:
        self._provider = provider
        self._model = model

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        from autoconstitution.providers.ollama import (
            CompletionRequest,
            Message,
            Role,
        )

        messages = []
        if system:
            messages.append(Message(role=Role.SYSTEM, content=system))
        messages.append(Message(role=Role.USER, content=prompt))

        req = CompletionRequest(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        resp = await self._provider.complete(req)
        return resp.content  # type: ignore[no-any-return]


class _OpenAIStyleAdapter:
    """Works for both OpenAI and Kimi (same API shape)."""

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content  # type: ignore[no-any-return]


class _AnthropicAdapter:
    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = await self._client.messages.create(**kwargs)
        # Anthropic response: resp.content is a list of blocks
        return "".join(block.text for block in resp.content if block.type == "text")


# ---------- Detection ----------------------------------------------------


async def _ollama_available(host: str = "http://localhost:11434", timeout: float = 2.0) -> bool:
    """Ping the local Ollama daemon; return True if it answers."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{host}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


async def _ollama_pick_model(host: str = "http://localhost:11434") -> Optional[str]:
    """Return the first installed Ollama model, or None."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{host}/api/tags")
            data = r.json()
            models = data.get("models", [])
            if models:
                return models[0]["name"]  # type: ignore[no-any-return]
    except Exception:
        pass
    return None


async def pick_provider(
    *,
    prefer: Optional[list[str]] = None,
    ollama_host: str = "http://localhost:11434",
    ollama_model: Optional[str] = None,
) -> ProviderChoice:
    """Pick the best available provider. Raises RuntimeError if none work.

    Args:
        prefer: Optional ordered list of provider names to try first.
                e.g. ``["anthropic", "ollama"]``.
        ollama_host: Override the default Ollama endpoint.
        ollama_model: Pin a specific Ollama model; auto-detect if None.
    """
    default_order = ["ollama", "kimi", "anthropic", "openai"]
    if prefer:
        unknown = [p for p in prefer if p not in default_order]
        if unknown:
            raise ValueError(
                f"Unknown provider(s) in prefer list: {unknown}. "
                f"Valid providers: {default_order}"
            )
        order = prefer + [p for p in default_order if p not in prefer]
    else:
        order = default_order

    errors: list[str] = []

    for name in order:
        try:
            if name == "ollama":
                if not await _ollama_available(ollama_host):
                    errors.append("ollama: daemon not reachable at " + ollama_host)
                    continue
                model = ollama_model or await _ollama_pick_model(ollama_host)
                if model is None:
                    errors.append("ollama: no models installed (try `ollama pull llama3.1`)")
                    continue
                from autoconstitution.providers.ollama import OllamaConfig, OllamaProvider

                provider = OllamaProvider(OllamaConfig(base_url=ollama_host))
                return ProviderChoice(
                    provider=_OllamaAdapter(provider, model),
                    name="ollama",
                    model=model,
                    reason="local daemon reachable",
                )

            if name == "kimi":
                key = os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY")
                if not key:
                    errors.append("kimi: MOONSHOT_API_KEY / KIMI_API_KEY not set")
                    continue
                try:
                    from openai import AsyncOpenAI  # type: ignore[import-not-found]
                except ImportError:
                    errors.append("kimi: `pip install openai` required")
                    continue
                client = AsyncOpenAI(
                    api_key=key,
                    base_url="https://api.moonshot.cn/v1",
                )
                return ProviderChoice(
                    provider=_OpenAIStyleAdapter(client, "moonshot-v1-8k"),
                    name="kimi",
                    model="moonshot-v1-8k",
                    reason="MOONSHOT_API_KEY set",
                )

            if name == "anthropic":
                key = os.environ.get("ANTHROPIC_API_KEY")
                if not key:
                    errors.append("anthropic: ANTHROPIC_API_KEY not set")
                    continue
                try:
                    from anthropic import AsyncAnthropic  # type: ignore[import-not-found]
                except ImportError:
                    errors.append("anthropic: `pip install anthropic` required")
                    continue
                client = AsyncAnthropic(api_key=key)
                model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
                return ProviderChoice(
                    provider=_AnthropicAdapter(client, model),
                    name="anthropic",
                    model=model,
                    reason="ANTHROPIC_API_KEY set",
                )

            if name == "openai":
                key = os.environ.get("OPENAI_API_KEY")
                if not key:
                    errors.append("openai: OPENAI_API_KEY not set")
                    continue
                try:
                    from openai import AsyncOpenAI  # type: ignore[import-not-found]
                except ImportError:
                    errors.append("openai: `pip install openai` required")
                    continue
                client = AsyncOpenAI(api_key=key)
                model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
                return ProviderChoice(
                    provider=_OpenAIStyleAdapter(client, model),
                    name="openai",
                    model=model,
                    reason="OPENAI_API_KEY set",
                )

        except Exception as e:  # pragma: no cover  — defensive
            errors.append(f"{name}: {e}")
            logger.debug("provider %s failed: %s", name, e)

    raise RuntimeError(
        "No LLM provider available. Tried in order:\n  "
        + "\n  ".join(errors)
        + "\n\nSet one of: MOONSHOT_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "or install Ollama and `ollama pull llama3.1`."
    )
