"""Deterministic, network-free FakeProvider for testing the CAI loop.

This is the lynchpin that lets end-to-end tests assert CAI behavior without
ever hitting a real model. It satisfies two interfaces simultaneously:

* The simple ``LLMProvider`` Protocol used by ``cai.hierarchy`` — the primary
  consumer — exposing ``complete(prompt, system, temperature, max_tokens) -> str``.
* The fuller request/response interface used by the CLI and by the startup
  provider probe, via ``full_complete(request) -> CompletionResponse`` and
  ``full_stream(request) -> AsyncIterator[CompletionResponse]``.

Three scripting modes are supported:

* **Queue** — ``FakeProvider(responses=["a", "b", "c"])``. Each call pops the
  next response. Raises :class:`FakeProviderExhausted` when empty.
* **Keyed** — ``FakeProvider(responses={"retirement": "...", "code": "..."})``.
  Picks the first key that appears as a substring of the prompt. Falls back
  to ``default`` if set, else raises.
* **Callable** — ``FakeProvider(responses=lambda prompt: f"echo: {prompt}")``.
  Pure function of prompt. Useful for dynamic scripted behavior.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from autoconstitution.providers import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    ProviderType,
    Role,
    TokenUsage,
    register_provider,
)


class FakeProviderExhaustedError(RuntimeError):
    """Raised when a scripted FakeProvider has no response to return."""


# Back-compat alias for the shorter name used in early conversations.
FakeProviderExhausted = FakeProviderExhaustedError

ScriptQueue = list[str]
ScriptMap = dict[str, str]
ScriptCallable = Callable[[str], str]
Responses = ScriptQueue | ScriptMap | ScriptCallable


@register_provider(ProviderType.FAKE)
class FakeProvider(BaseProvider):
    """Deterministic scripted provider. Zero network, zero flakiness."""

    def __init__(
        self,
        responses: Responses,
        *,
        default: str | None = None,
        chunk_delay_ms: float = 0.0,
        chunks_per_response: int = 8,
        model: str = "fake-model",
    ) -> None:
        super().__init__()
        self._responses: Responses = responses
        self._default: str | None = default
        self._chunk_delay_ms: float = chunk_delay_ms
        self._chunks_per_response: int = max(1, chunks_per_response)
        self._model: str = model
        self._queue_index: int = 0

    async def initialize(self) -> None:
        self._initialized = True

    async def close(self) -> None:
        self._initialized = False

    async def health_check(self) -> bool:
        return True

    def reset(self) -> None:
        """Reset the queue cursor so the same instance can be reused in tests."""
        self._queue_index = 0

    # ------------------------------------------------------------------
    # LLMProvider Protocol (consumed by cai.hierarchy._AgentBase._ask)
    # ------------------------------------------------------------------
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        del system, temperature, max_tokens  # signature padding for LLMProvider Protocol
        return self._resolve(prompt)

    async def stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """Yield text chunks for the resolved response.

        When the CAI loop's Protocol is extended to include streaming, this is
        the method it will call.
        """
        del system, temperature, max_tokens
        text = self._resolve(prompt)
        for chunk in _split_into_chunks(text, self._chunks_per_response):
            if self._chunk_delay_ms > 0:
                await asyncio.sleep(self._chunk_delay_ms / 1000.0)
            yield chunk

    # ------------------------------------------------------------------
    # Full request/response interface (CLI, startup probe)
    # ------------------------------------------------------------------
    async def full_complete(self, request: CompletionRequest) -> CompletionResponse:
        prompt = _extract_prompt(request)
        content = self._resolve(prompt)
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(content.split()))
        return CompletionResponse(
            content=content,
            model=request.model or self._model,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason="stop",
        )

    async def full_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionResponse]:
        prompt = _extract_prompt(request)
        content = self._resolve(prompt)
        chunks = _split_into_chunks(content, self._chunks_per_response)
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(content.split()))
        model = request.model or self._model
        last_index = len(chunks) - 1
        for i, chunk in enumerate(chunks):
            if self._chunk_delay_ms > 0:
                await asyncio.sleep(self._chunk_delay_ms / 1000.0)
            is_last = i == last_index
            yield CompletionResponse(
                content=chunk,
                model=model,
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens if is_last else 0,
                    completion_tokens=completion_tokens if is_last else 0,
                    total_tokens=(prompt_tokens + completion_tokens) if is_last else 0,
                ),
                finish_reason="stop" if is_last else None,
            )

    async def embed(self, *args: object, **kwargs: object) -> CompletionResponse:
        del args, kwargs
        raise NotImplementedError("FakeProvider does not support embeddings")

    # ------------------------------------------------------------------
    # Script resolution
    # ------------------------------------------------------------------
    def _resolve(self, prompt: str) -> str:
        responses = self._responses
        if isinstance(responses, list):
            if self._queue_index >= len(responses):
                raise FakeProviderExhaustedError(
                    f"FakeProvider queue exhausted after {self._queue_index} call(s); "
                    f"queue had {len(responses)} response(s). "
                    f"Append more or use keyed/callable mode."
                )
            out = responses[self._queue_index]
            self._queue_index += 1
            return out
        if isinstance(responses, dict):
            for key, value in responses.items():
                if key in prompt:
                    return value
            if self._default is not None:
                return self._default
            raise FakeProviderExhaustedError(
                f"FakeProvider keyed-mode: prompt matched none of {list(responses.keys())} "
                f"and no default was configured."
            )
        # Callable
        return responses(prompt)


def _extract_prompt(request: CompletionRequest) -> str:
    """Return the last user message content, or the last message's content."""
    for msg in reversed(request.messages):
        if msg.role == Role.USER:
            return msg.content
    return request.messages[-1].content if request.messages else ""


def _split_into_chunks(text: str, target_count: int) -> list[str]:
    """Split ``text`` into roughly ``target_count`` whitespace-preserving chunks.

    Always returns at least one chunk. Chunks joined with ``""`` reconstruct
    the original text exactly.
    """
    if not text:
        return [""]
    words = text.split(" ")
    if len(words) <= target_count:
        # One word per chunk, preserve trailing spaces on all but the last.
        return [w + " " if i < len(words) - 1 else w for i, w in enumerate(words)]
    chunk_size = max(1, len(words) // target_count)
    chunks: list[str] = []
    for i in range(0, len(words), chunk_size):
        segment_words = words[i : i + chunk_size]
        segment = " ".join(segment_words)
        if i + chunk_size < len(words):
            segment = segment + " "
        chunks.append(segment)
    return chunks


__all__ = ["FakeProvider", "FakeProviderExhausted", "FakeProviderExhaustedError"]
