"""Tests for FakeProvider. See codebase/autoconstitution/providers/fake.py."""

from __future__ import annotations

import time

import pytest

from autoconstitution.providers import (
    CompletionRequest,
    CompletionResponse,
    Message,
    ProviderType,
)
from autoconstitution.providers.fake import FakeProvider, FakeProviderExhausted


def _req(prompt: str) -> CompletionRequest:
    return CompletionRequest(messages=[Message.user(prompt)])


@pytest.mark.asyncio
async def test_queue_mode_pops_in_order() -> None:
    provider = FakeProvider(responses=["first", "second", "third"])
    assert await provider.complete("anything") == "first"
    assert await provider.complete("anything") == "second"
    assert await provider.complete("anything") == "third"


@pytest.mark.asyncio
async def test_queue_mode_exhaustion_raises() -> None:
    provider = FakeProvider(responses=["only"])
    await provider.complete("x")
    with pytest.raises(FakeProviderExhausted, match="exhausted after 1"):
        await provider.complete("x")


@pytest.mark.asyncio
async def test_queue_mode_reset_rewinds_cursor() -> None:
    provider = FakeProvider(responses=["a", "b"])
    assert await provider.complete("x") == "a"
    assert await provider.complete("x") == "b"
    provider.reset()
    assert await provider.complete("x") == "a"


@pytest.mark.asyncio
async def test_keyed_mode_substring_match() -> None:
    provider = FakeProvider(
        responses={"retirement": "plan response", "code": "code response"}
    )
    assert await provider.complete("help with my retirement plans") == "plan response"
    assert await provider.complete("fix this code bug") == "code response"


@pytest.mark.asyncio
async def test_keyed_mode_default_fallback() -> None:
    provider = FakeProvider(responses={"foo": "bar"}, default="fallback")
    assert await provider.complete("unrelated prompt") == "fallback"


@pytest.mark.asyncio
async def test_keyed_mode_no_match_no_default_raises() -> None:
    provider = FakeProvider(responses={"foo": "bar"})
    with pytest.raises(FakeProviderExhausted, match="matched none"):
        await provider.complete("unrelated")


@pytest.mark.asyncio
async def test_callable_mode_sees_prompt() -> None:
    provider = FakeProvider(responses=lambda p: f"echo:{p[:5]}")
    assert await provider.complete("helloworld") == "echo:hello"


@pytest.mark.asyncio
async def test_stream_tokens_reconstruct_full_response() -> None:
    provider = FakeProvider(
        responses=["the quick brown fox jumps over the lazy dog"],
        chunks_per_response=4,
    )
    chunks: list[str] = []
    async for chunk in provider.stream("prompt"):
        chunks.append(chunk)
    assert "".join(chunks) == "the quick brown fox jumps over the lazy dog"
    assert len(chunks) >= 2


@pytest.mark.asyncio
async def test_full_complete_returns_proper_completion_response() -> None:
    provider = FakeProvider(responses=["hello world"])
    resp = await provider.full_complete(_req("ask me"))
    assert isinstance(resp, CompletionResponse)
    assert resp.content == "hello world"
    assert resp.finish_reason == "stop"
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.completion_tokens > 0
    assert resp.usage.total_tokens == resp.usage.prompt_tokens + resp.usage.completion_tokens


@pytest.mark.asyncio
async def test_full_stream_final_chunk_has_stop_reason_and_usage() -> None:
    provider = FakeProvider(
        responses=["token one two three four five six seven"],
        chunks_per_response=4,
    )
    chunks: list[CompletionResponse] = []
    async for resp in provider.full_stream(_req("go")):
        chunks.append(resp)
    assert len(chunks) >= 2
    for intermediate in chunks[:-1]:
        assert intermediate.finish_reason is None
        assert intermediate.usage.total_tokens == 0
    last = chunks[-1]
    assert last.finish_reason == "stop"
    assert last.usage.total_tokens > 0
    assert "".join(c.content for c in chunks) == "token one two three four five six seven"


@pytest.mark.asyncio
async def test_dual_interface_single_instance() -> None:
    """Same instance must serve both the LLMProvider Protocol and the full interface."""
    provider = FakeProvider(
        responses=["simple-form", "request-form"],
    )
    simple_result = await provider.complete("prompt a")
    full_result = await provider.full_complete(_req("prompt b"))
    assert simple_result == "simple-form"
    assert full_result.content == "request-form"


@pytest.mark.asyncio
async def test_health_check_returns_true() -> None:
    provider = FakeProvider(responses=["x"])
    assert await provider.health_check() is True


@pytest.mark.asyncio
async def test_initialize_and_close_are_idempotent() -> None:
    provider = FakeProvider(responses=["x"])
    await provider.initialize()
    await provider.initialize()
    assert provider.is_initialized is True
    await provider.close()
    await provider.close()
    assert provider.is_initialized is False


@pytest.mark.asyncio
async def test_embed_raises_not_implemented() -> None:
    provider = FakeProvider(responses=["x"])
    with pytest.raises(NotImplementedError):
        await provider.embed()


@pytest.mark.asyncio
async def test_zero_delay_stream_is_fast() -> None:
    provider = FakeProvider(
        responses=["a" * 100],
        chunks_per_response=10,
        chunk_delay_ms=0.0,
    )
    start = time.perf_counter()
    collected: list[str] = []
    async for chunk in provider.stream("go"):
        collected.append(chunk)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 50
    assert "".join(collected) == "a" * 100


@pytest.mark.asyncio
async def test_nonzero_delay_stream_respects_delay() -> None:
    provider = FakeProvider(
        responses=["one two three four"],
        chunks_per_response=4,
        chunk_delay_ms=5.0,
    )
    start = time.perf_counter()
    async for _ in provider.stream("go"):
        pass
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    # 4 chunks * 5ms = ~20ms minimum; allow scheduler slop but require > 10ms.
    assert elapsed_ms > 10


@pytest.mark.asyncio
async def test_callable_mode_is_deterministic() -> None:
    counter = {"calls": 0}

    def script(prompt: str) -> str:
        counter["calls"] += 1
        return f"r{counter['calls']}:{prompt[:3]}"

    provider = FakeProvider(responses=script)
    assert await provider.complete("abcdef") == "r1:abc"
    assert await provider.complete("xyzwvu") == "r2:xyz"


def test_registered_under_fake_provider_type() -> None:
    from autoconstitution.providers import _PROVIDER_REGISTRY

    assert ProviderType.FAKE in _PROVIDER_REGISTRY
    assert _PROVIDER_REGISTRY[ProviderType.FAKE] is FakeProvider


@pytest.mark.asyncio
async def test_fakeprovider_is_drop_in_for_student_agent() -> None:
    """Prove FakeProvider works as a CAI-loop provider by wiring StudentAgent."""
    from autoconstitution.cai.hierarchy import StudentAgent

    provider = FakeProvider(responses=["student reply", "revised reply"])
    student = StudentAgent(provider=provider)
    first = await student.respond("why is the sky blue?")
    assert first == "student reply"
    revised = await student.revise(
        prompt="why is the sky blue?",
        previous_answer=first,
        critique="too vague",
    )
    assert revised == "revised reply"


@pytest.mark.asyncio
async def test_extract_prompt_uses_last_user_message() -> None:
    """Multi-turn requests: FakeProvider keys off the last user message."""
    provider = FakeProvider(responses={"second": "matched-second"}, default="nope")
    request = CompletionRequest(
        messages=[
            Message.user("first"),
            Message.assistant("...ack..."),
            Message.user("second"),
        ]
    )
    resp = await provider.full_complete(request)
    assert resp.content == "matched-second"
