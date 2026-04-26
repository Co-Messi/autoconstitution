"""
Provider Tests for autoconstitution

Comprehensive test suite for all LLM providers (Kimi, Anthropic, OpenAI, Ollama)
with mocked API responses. Tests cover:
- Common interface compliance
- Error handling
- Streaming responses
- Tool calling
- Rate limiting

Usage:
    pytest tests/test_providers.py -v
    pytest tests/test_providers.py -v -k "kimi"  # Run only Kimi tests
    pytest tests/test_providers.py -v -k "error"  # Run only error handling tests
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_aiohttp_session():
    """Fixture for mocking aiohttp ClientSession."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.closed = False
        yield mock_session


@pytest.fixture
def mock_response():
    """Fixture for creating mock HTTP responses."""
    def _create_mock_response(
        status: int = 200,
        json_data: Optional[Dict[str, Any]] = None,
        text: str = "",
        headers: Optional[Dict[str, str]] = None,
    ) -> AsyncMock:
        mock_resp = AsyncMock()
        mock_resp.status = status
        mock_resp.json = AsyncMock(return_value=json_data or {})
        mock_resp.text = AsyncMock(return_value=text)
        mock_resp.headers = headers or {}
        return mock_resp
    
    return _create_mock_response


# =============================================================================
# Kimi Provider Tests
# =============================================================================


class TestKimiProvider:
    """Test suite for Kimi provider."""
    
    @pytest.fixture
    def kimi_config(self) -> Dict[str, Any]:
        """Default Kimi configuration for tests."""
        return {
            "api_key": "test-kimi-api-key",
            "base_url": "https://api.moonshot.cn/v1",
            "timeout": 30.0,
            "max_retries": 2,
        }
    
    @pytest.fixture
    def kimi_provider(self, kimi_config: Dict[str, Any]):
        """Create a Kimi provider instance."""
        from autoconstitution.providers.kimi import KimiProvider, KimiConfig
        
        config = KimiConfig(**kimi_config)
        return KimiProvider(config)
    
    @pytest.fixture
    def mock_kimi_completion_response(self) -> Dict[str, Any]:
        """Mock Kimi API completion response."""
        return {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "kimi-k2-5",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! I'm Kimi, your AI assistant. How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25,
                "cached_tokens": 0,
            },
        }
    
    @pytest.fixture
    def mock_kimi_streaming_chunks(self) -> List[Dict[str, Any]]:
        """Mock Kimi streaming response chunks."""
        return [
            {
                "id": "chatcmpl-test123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "kimi-k2-5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-test123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "kimi-k2-5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " I'm Kimi"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-test123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "kimi-k2-5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": ", your AI assistant."},
                        "finish_reason": "stop",
                    }
                ],
            },
        ]
    
    @pytest.mark.asyncio
    async def test_kimi_provider_initialization(
        self, kimi_config: Dict[str, Any]
    ) -> None:
        """Test Kimi provider initialization."""
        from autoconstitution.providers.kimi import KimiProvider, KimiConfig
        
        config = KimiConfig(**kimi_config)
        provider = KimiProvider(config)
        
        assert provider.config.api_key == "test-kimi-api-key"
        assert provider.config.base_url == "https://api.moonshot.cn/v1"
        assert provider.api_key == "test-kimi-api-key"
    
    @pytest.mark.asyncio
    async def test_kimi_provider_initialization_missing_api_key(self) -> None:
        """Test Kimi provider fails without API key."""
        from autoconstitution.providers.kimi import (
            KimiProvider,
            KimiConfig,
            KimiAuthenticationError,
        )
        
        with patch.dict("os.environ", {}, clear=True):
            config = KimiConfig(api_key=None)
            with pytest.raises(KimiAuthenticationError):
                KimiProvider(config)
    
    @pytest.mark.asyncio
    async def test_kimi_complete_success(
        self,
        kimi_provider: Any,
        mock_kimi_completion_response: Dict[str, Any],
        mock_aiohttp_session: AsyncMock,
        mock_response: Callable,
    ) -> None:
        """Test successful Kimi completion."""
        from autoconstitution.providers.kimi import Message, CompletionRequest
        
        # Setup mock response
        mock_resp = mock_response(
            status=200, json_data=mock_kimi_completion_response
        )
        mock_aiohttp_session.request = AsyncMock(return_value=mock_resp)
        mock_aiohttp_session.post = AsyncMock(return_value=mock_resp)
        
        # Mock the session context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_aiohttp_session.post.return_value = mock_context
        
        # Initialize provider with mocked session
        await kimi_provider.initialize()
        kimi_provider._session = mock_aiohttp_session
        
        # Create request
        request = CompletionRequest(
            messages=[Message.user("Hello, who are you?")],
            model="kimi-k2-5",
            temperature=0.7,
        )
        
        # Mock the _request_with_retry to return parsed response
        expected_response = mock_kimi_completion_response
        choice = expected_response["choices"][0]
        message = choice["message"]
        usage_data = expected_response["usage"]
        
        from autoconstitution.providers.kimi import CompletionResponse, TokenUsage
        
        parsed_response = CompletionResponse(
            content=message.get("content", ""),
            model=expected_response.get("model", ""),
            usage=TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                cached_tokens=usage_data.get("cached_tokens"),
            ),
            finish_reason=choice.get("finish_reason"),
            tool_calls=message.get("tool_calls"),
            raw_response=expected_response,
            id=expected_response.get("id"),
        )
        
        with patch.object(
            kimi_provider, "_request_with_retry", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = parsed_response
            response = await kimi_provider.complete(request)
        
        assert response.content == "Hello! I'm Kimi, your AI assistant. How can I help you today?"
        assert response.model == "kimi-k2-5"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 15
        assert response.usage.total_tokens == 25
        assert response.finish_reason == "stop"
    
    @pytest.mark.asyncio
    async def test_kimi_complete_with_tools(
        self,
        kimi_provider: Any,
        mock_aiohttp_session: AsyncMock,
    ) -> None:
        """Test Kimi completion with tool calls."""
        from autoconstitution.providers.kimi import (
            Message,
            CompletionRequest,
            Tool,
            CompletionResponse,
            TokenUsage,
        )
        
        # Mock response with tool calls
        tool_response = {
            "id": "chatcmpl-test456",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "kimi-k2-5",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 25,
                "total_tokens": 45,
            },
        }
        
        choice = tool_response["choices"][0]
        message = choice["message"]
        usage_data = tool_response["usage"]
        
        parsed_response = CompletionResponse(
            content=message.get("content", ""),
            model=tool_response.get("model", ""),
            usage=TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason"),
            tool_calls=message.get("tool_calls"),
            raw_response=tool_response,
            id=tool_response.get("id"),
        )
        
        await kimi_provider.initialize()
        
        request = CompletionRequest(
            messages=[Message.user("What's the weather in San Francisco?")],
            tools=[
                Tool(
                    name="get_weather",
                    description="Get weather for a location",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                )
            ],
        )
        
        with patch.object(
            kimi_provider, "_request_with_retry", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = parsed_response
            response = await kimi_provider.complete(request)
        
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "get_weather"
        assert response.finish_reason == "tool_calls"
    
    @pytest.mark.asyncio
    async def test_kimi_rate_limit_error(
        self, kimi_provider: Any, mock_aiohttp_session: AsyncMock
    ) -> None:
        """Test Kimi rate limit error handling."""
        from autoconstitution.providers.kimi import (
            Message,
            CompletionRequest,
            KimiRateLimitError,
        )
        
        await kimi_provider.initialize()
        
        with patch.object(
            kimi_provider, "_request_with_retry", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = KimiRateLimitError(
                "Rate limit exceeded", retry_after=5.0
            )
            
            request = CompletionRequest(messages=[Message.user("Test")])
            
            with pytest.raises(KimiRateLimitError) as exc_info:
                await kimi_provider.complete(request)
            
            assert "Rate limit exceeded" in str(exc_info.value)
            assert exc_info.value.retry_after == 5.0
    
    @pytest.mark.asyncio
    async def test_kimi_authentication_error(
        self, kimi_provider: Any
    ) -> None:
        """Test Kimi authentication error handling."""
        from autoconstitution.providers.kimi import (
            KimiAuthenticationError,
        )
        
        await kimi_provider.initialize()
        
        with patch.object(
            kimi_provider, "_make_request", new_callable=AsyncMock
        ) as mock_make_request:
            mock_make_request.side_effect = KimiAuthenticationError(
                "Invalid API key", status_code=401
            )
            
            with pytest.raises(KimiAuthenticationError) as exc_info:
                await kimi_provider._make_request("POST", "/chat/completions")
            
            assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_kimi_streaming(
        self,
        kimi_provider: Any,
        mock_kimi_streaming_chunks: List[Dict[str, Any]],
    ) -> None:
        """Test Kimi streaming completion."""
        from autoconstitution.providers.kimi import Message, CompletionRequest, StreamChunk
        
        await kimi_provider.initialize()
        
        # Create async iterator for streaming chunks
        async def mock_stream():
            for chunk in mock_kimi_streaming_chunks:
                choice = chunk["choices"][0]
                delta = choice["delta"]
                yield StreamChunk(
                    content=delta.get("content", ""),
                    is_finished=choice.get("finish_reason") is not None,
                    finish_reason=choice.get("finish_reason"),
                    tool_calls=delta.get("tool_calls"),
                )
        
        with patch.object(
            kimi_provider, "_request_with_retry", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_stream()
            
            request = CompletionRequest(
                messages=[Message.user("Hello")],
                stream=True,
            )
            
            chunks = []
            async for chunk in await kimi_provider.complete_stream(request):
                chunks.append(chunk)
            
            assert len(chunks) == 3
            assert chunks[0].content == "Hello!"
            assert chunks[1].content == " I'm Kimi"
            assert chunks[2].content == ", your AI assistant."
            assert chunks[2].is_finished is True
    
    @pytest.mark.asyncio
    async def test_kimi_tool_executor(self, kimi_provider: Any) -> None:
        """Test Kimi tool executor."""
        from autoconstitution.providers.kimi import ToolExecutor
        
        executor = ToolExecutor()
        
        # Register a test function
        def get_weather(location: str) -> str:
            return f"Weather in {location}: Sunny, 72°F"
        
        executor.register("get_weather", get_weather)
        
        # Test executing a tool call
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}',
            },
        }
        
        result = await executor.execute(tool_call)
        
        assert "result" in result
        assert "San Francisco" in result["result"]
        assert "Sunny" in result["result"]
    
    @pytest.mark.asyncio
    async def test_kimi_rate_limiter(self) -> None:
        """Test Kimi rate limiter."""
        from autoconstitution.providers.kimi import RateLimiter
        
        limiter = RateLimiter(
            requests_per_minute=60.0,
            tokens_per_minute=100000.0,
        )
        
        # Should not raise
        await limiter.acquire(estimated_tokens=100)
        
        # Test token replenishment
        import time
        
        initial_tokens = limiter._token_tokens
        time.sleep(0.01)  # Small delay
        await limiter.acquire(estimated_tokens=1)
        # Tokens should have been replenished slightly
    
    @pytest.mark.asyncio
    async def test_kimi_context_manager(self, kimi_provider: Any) -> None:
        """Test Kimi provider async context manager."""
        from autoconstitution.providers.kimi import KimiProvider, KimiConfig
        
        config = KimiConfig(api_key="test-key")
        
        async with KimiProvider(config) as provider:
            assert provider.is_initialized is True
        
        # After exiting context, session should be closed
        assert provider._session is None or provider._session.closed


# =============================================================================
# Anthropic Provider Tests
# =============================================================================


class TestAnthropicProvider:
    """Test suite for Anthropic provider."""
    
    @pytest.fixture
    def anthropic_config(self) -> Dict[str, Any]:
        """Default Anthropic configuration for tests."""
        return {
            "api_key": "test-anthropic-api-key",
            "base_url": "https://api.anthropic.com",
            "timeout": 30.0,
            "max_retries": 2,
            "default_model": "claude-3-5-sonnet-20241022",
        }
    
    @pytest.fixture
    def anthropic_provider(self, anthropic_config: Dict[str, Any]):
        """Create an Anthropic provider instance."""
        from autoconstitution.providers.anthropic import AnthropicProvider, AnthropicConfig
        
        config = AnthropicConfig(**anthropic_config)
        return AnthropicProvider(config)
    
    @pytest.fixture
    def mock_anthropic_response(self) -> Dict[str, Any]:
        """Mock Anthropic API response."""
        return {
            "id": "msg_01Test123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {
                    "type": "text",
                    "text": "Hello! I'm Claude, an AI assistant made by Anthropic.",
                }
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 12,
                "output_tokens": 18,
            },
        }
    
    @pytest.mark.asyncio
    async def test_anthropic_provider_initialization(
        self, anthropic_config: Dict[str, Any]
    ) -> None:
        """Test Anthropic provider initialization."""
        from autoconstitution.providers.anthropic import AnthropicProvider, AnthropicConfig
        
        config = AnthropicConfig(**anthropic_config)
        provider = AnthropicProvider(config)
        
        assert provider.config.api_key == "test-anthropic-api-key"
        assert provider.config.default_model == "claude-3-5-sonnet-20241022"
        assert provider.default_model == "claude-3-5-sonnet-20241022"
        assert provider.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_anthropic_provider_missing_api_key(self) -> None:
        """Test Anthropic provider fails without API key."""
        from autoconstitution.providers.anthropic import (
            AnthropicProvider,
            AnthropicConfig,
            AnthropicAuthenticationError,
        )
        
        with patch.dict("os.environ", {}, clear=True):
            config = AnthropicConfig(api_key=None)
            provider = AnthropicProvider(config)
            
            with pytest.raises(AnthropicAuthenticationError):
                await provider.initialize()
    
    @pytest.mark.asyncio
    async def test_anthropic_complete_success(
        self,
        anthropic_provider: Any,
        mock_anthropic_response: Dict[str, Any],
    ) -> None:
        """Test successful Anthropic completion."""
        from autoconstitution.providers.anthropic import (
            Message,
            CompletionRequest,
            CompletionResponse,
            TokenUsage,
            Role,
        )
        
        # Mock the anthropic client
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.id = "msg_01Test123"
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.stop_reason = "end_turn"
        mock_response.stop_sequence = None
        
        # Mock content blocks
        content_block = MagicMock()
        content_block.type = "text"
        content_block.text = "Hello! I'm Claude, an AI assistant made by Anthropic."
        mock_response.content = [content_block]
        
        # Mock usage
        mock_usage = MagicMock()
        mock_usage.input_tokens = 12
        mock_usage.output_tokens = 18
        mock_response.usage = mock_usage
        
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        
        with patch(
            "autoconstitution.providers.anthropic.anthropic.AsyncAnthropic",
            return_value=mock_client,
        ):
            await anthropic_provider.initialize()
            
            request = CompletionRequest(
                messages=[Message.user("Hello, who are you?")],
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
            )
            
            # Mock the retry handler to directly return the response
            with patch.object(
                anthropic_provider._retry_handler,
                "execute",
                new_callable=AsyncMock,
            ) as mock_execute:
                mock_execute.return_value = CompletionResponse(
                    content="Hello! I'm Claude, an AI assistant made by Anthropic.",
                    model="claude-3-5-sonnet-20241022",
                    usage=TokenUsage(
                        prompt_tokens=12,
                        completion_tokens=18,
                        total_tokens=30,
                    ),
                    finish_reason="stop",
                    id="msg_01Test123",
                )
                
                response = await anthropic_provider.complete(request)
        
        assert response.content == "Hello! I'm Claude, an AI assistant made by Anthropic."
        assert response.model == "claude-3-5-sonnet-20241022"
        assert response.usage.prompt_tokens == 12
        assert response.usage.completion_tokens == 18
        assert response.finish_reason == "stop"
    
    @pytest.mark.asyncio
    async def test_anthropic_complete_with_tools(
        self,
        anthropic_provider: Any,
    ) -> None:
        """Test Anthropic completion with tool calls."""
        from autoconstitution.providers.anthropic import (
            Message,
            CompletionRequest,
            Tool,
            CompletionResponse,
            TokenUsage,
        )
        
        await anthropic_provider.initialize()
        
        # Mock response with tool use
        mock_response = CompletionResponse(
            content="",
            model="claude-3-5-sonnet-20241022",
            usage=TokenUsage(
                prompt_tokens=25,
                completion_tokens=35,
                total_tokens=60,
            ),
            finish_reason="tool_use",
            tool_calls=[
                {
                    "id": "toolu_01Test",
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": '{"expression": "2 + 2"}',
                    },
                }
            ],
            id="msg_01ToolTest",
        )
        
        with patch.object(
            anthropic_provider._retry_handler,
            "execute",
            new_callable=AsyncMock,
        ) as mock_execute:
            mock_execute.return_value = mock_response
            
            request = CompletionRequest(
                messages=[Message.user("Calculate 2 + 2")],
                tools=[
                    Tool(
                        name="calculate",
                        description="Perform calculations",
                        parameters={
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string"},
                            },
                            "required": ["expression"],
                        },
                    )
                ],
            )
            
            response = await anthropic_provider.complete(request)
        
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "calculate"
        assert response.finish_reason == "tool_use"
    
    @pytest.mark.asyncio
    async def test_anthropic_rate_limit_error(
        self, anthropic_provider: Any
    ) -> None:
        """Test Anthropic rate limit error handling."""
        from autoconstitution.providers.anthropic import (
            Message,
            CompletionRequest,
            AnthropicRateLimitError,
        )
        
        await anthropic_provider.initialize()
        
        with patch.object(
            anthropic_provider._retry_handler,
            "execute",
            new_callable=AsyncMock,
        ) as mock_execute:
            mock_execute.side_effect = AnthropicRateLimitError(
                "Rate limit exceeded",
                retry_after=10.0,
                status_code=429,
            )
            
            request = CompletionRequest(messages=[Message.user("Test")])
            
            with pytest.raises(AnthropicRateLimitError) as exc_info:
                await anthropic_provider.complete(request)
            
            assert exc_info.value.retry_after == 10.0
            assert exc_info.value.status_code == 429
    
    @pytest.mark.asyncio
    async def test_anthropic_validation_error(
        self, anthropic_provider: Any
    ) -> None:
        """Test Anthropic validation error handling."""
        from autoconstitution.providers.anthropic import (
            AnthropicValidationError,
        )
        
        await anthropic_provider.initialize()
        
        with patch.object(
            anthropic_provider,
            "_handle_error",
        ) as mock_handle_error:
            mock_handle_error.side_effect = AnthropicValidationError(
                "Invalid request",
                status_code=400,
            )
            
            with pytest.raises(AnthropicValidationError) as exc_info:
                mock_handle_error(Exception("test"))
            
            assert exc_info.value.status_code == 400
    
    @pytest.mark.asyncio
    async def test_anthropic_message_conversion(self, anthropic_provider: Any) -> None:
        """Test Anthropic message format conversion."""
        from autoconstitution.providers.anthropic import Message, Role
        
        await anthropic_provider.initialize()
        
        # Test various message types
        system_msg = Message.system("You are a helpful assistant.")
        user_msg = Message.user("Hello")
        assistant_msg = Message.assistant("Hi there!")
        tool_msg = Message.tool("Tool result", tool_call_id="call_123")
        
        messages = [system_msg, user_msg, assistant_msg, tool_msg]
        
        converted_system, converted_messages = anthropic_provider._convert_messages(
            messages
        )
        
        assert converted_system == "You are a helpful assistant."
        assert len(converted_messages) == 3  # system is separate
        assert converted_messages[0]["role"] == "user"
        assert converted_messages[1]["role"] == "assistant"
    
    @pytest.mark.asyncio
    async def test_anthropic_model_info(self, anthropic_provider: Any) -> None:
        """Test Anthropic model information."""
        from autoconstitution.providers.anthropic import AnthropicProvider
        
        # Check available models
        models = anthropic_provider.available_models
        assert "claude-3-5-sonnet-20241022" in models
        assert "claude-3-opus-20240229" in models
        
        # Check context window
        context_window = anthropic_provider.get_model_context_window()
        assert context_window == 200000
        
        # Check max output
        max_output = anthropic_provider.get_model_max_output()
        assert max_output == 8192
    
    @pytest.mark.asyncio
    async def test_anthropic_embed_not_implemented(
        self, anthropic_provider: Any
    ) -> None:
        """Test that Anthropic embeddings raise NotImplementedError."""
        from autoconstitution.providers.anthropic import EmbeddingRequest
        
        await anthropic_provider.initialize()
        
        with pytest.raises(NotImplementedError):
            await anthropic_provider.embed(EmbeddingRequest(texts=["test"]))
    
    @pytest.mark.asyncio
    async def test_anthropic_context_manager(self, anthropic_config: Dict[str, Any]) -> None:
        """Test Anthropic provider async context manager."""
        from autoconstitution.providers.anthropic import AnthropicProvider, AnthropicConfig
        
        config = AnthropicConfig(**anthropic_config)
        
        with patch(
            "autoconstitution.providers.anthropic.anthropic.AsyncAnthropic"
        ) as mock_anthropic:
            async with AnthropicProvider(config) as provider:
                assert provider.is_initialized is True


# =============================================================================
# OpenAI Provider Tests
# =============================================================================


class TestOpenAIProvider:
    """Test suite for OpenAI provider."""
    
    @pytest.fixture
    def openai_config(self) -> Dict[str, Any]:
        """Default OpenAI configuration for tests."""
        return {
            "api_key": "test-openai-api-key",
            "base_url": None,
            "timeout": 30.0,
            "max_retries": 2,
            "default_model": "gpt-4o",
        }
    
    @pytest.fixture
    def openai_provider(self, openai_config: Dict[str, Any]):
        """Create an OpenAI provider instance."""
        from autoconstitution.providers.openai import OpenAIProvider, ProviderConfig
        
        config = ProviderConfig(**openai_config)
        return OpenAIProvider(config)
    
    @pytest.fixture
    def mock_openai_completion_response(self) -> Dict[str, Any]:
        """Mock OpenAI API completion response."""
        return {
            "id": "chatcmpl-test789",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! I'm ChatGPT, an AI assistant made by OpenAI.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 20,
                "total_tokens": 35,
            },
        }
    
    @pytest.mark.asyncio
    async def test_openai_provider_initialization(
        self, openai_config: Dict[str, Any]
    ) -> None:
        """Test OpenAI provider initialization."""
        from autoconstitution.providers.openai import OpenAIProvider, ProviderConfig
        
        config = ProviderConfig(**openai_config)
        provider = OpenAIProvider(config)
        
        assert provider._api_key == "test-openai-api-key"
        assert provider.default_model == "gpt-4o"
    
    @pytest.mark.asyncio
    async def test_openai_provider_missing_api_key(self) -> None:
        """Test OpenAI provider fails without API key."""
        from autoconstitution.providers.openai import OpenAIProvider, ProviderConfig
        
        with patch.dict("os.environ", {}, clear=True):
            config = ProviderConfig(api_key=None)
            provider = OpenAIProvider(config)
            
            with pytest.raises(ValueError) as exc_info:
                await provider.initialize()
            
            assert "API key required" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_openai_complete_success(
        self,
        openai_provider: Any,
        mock_openai_completion_response: Dict[str, Any],
    ) -> None:
        """Test successful OpenAI completion."""
        from autoconstitution.providers.openai import (
            Message,
            CompletionRequest,
            CompletionResponse,
            TokenUsage,
        )
        
        # Mock the OpenAI client
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.model = "gpt-4o"
        
        # Mock choice
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_message = MagicMock()
        mock_message.content = "Hello! I'm ChatGPT, an AI assistant made by OpenAI."
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        # Mock usage
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 15
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 35
        mock_response.usage = mock_usage
        
        mock_response.model_dump = MagicMock(return_value=mock_openai_completion_response)
        
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        with patch(
            "autoconstitution.providers.openai.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            await openai_provider.initialize()
            
            # Mock rate limiter and retry handler
            with patch.object(
                openai_provider._rate_limiter, "acquire", new_callable=AsyncMock
            ), patch.object(
                openai_provider._retry_handler,
                "execute",
                new_callable=AsyncMock,
            ) as mock_execute:
                mock_execute.return_value = CompletionResponse(
                    content="Hello! I'm ChatGPT, an AI assistant made by OpenAI.",
                    model="gpt-4o",
                    usage=TokenUsage(
                        prompt_tokens=15,
                        completion_tokens=20,
                        total_tokens=35,
                    ),
                    finish_reason="stop",
                    raw_response=mock_openai_completion_response,
                )
                
                request = CompletionRequest(
                    messages=[Message.user("Hello, who are you?")],
                    model="gpt-4o",
                )
                
                response = await openai_provider.complete(request)
        
        assert response.content == "Hello! I'm ChatGPT, an AI assistant made by OpenAI."
        assert response.model == "gpt-4o"
        assert response.usage.prompt_tokens == 15
        assert response.usage.completion_tokens == 20
        assert response.finish_reason == "stop"
    
    @pytest.mark.asyncio
    async def test_openai_complete_with_tools(
        self,
        openai_provider: Any,
    ) -> None:
        """Test OpenAI completion with tool calls."""
        from autoconstitution.providers.openai import (
            Message,
            CompletionRequest,
            Tool,
            CompletionResponse,
            TokenUsage,
        )
        
        await openai_provider.initialize()
        
        mock_response = CompletionResponse(
            content="",
            model="gpt-4o",
            usage=TokenUsage(
                prompt_tokens=30,
                completion_tokens=40,
                total_tokens=70,
            ),
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "arguments": '{"query": "latest AI news"}',
                    },
                }
            ],
            raw_response={},
        )
        
        with patch.object(
            openai_provider._rate_limiter, "acquire", new_callable=AsyncMock
        ), patch.object(
            openai_provider._retry_handler,
            "execute",
            new_callable=AsyncMock,
        ) as mock_execute:
            mock_execute.return_value = mock_response
            
            request = CompletionRequest(
                messages=[Message.user("Search for latest AI news")],
                tools=[
                    Tool(
                        name="search_web",
                        description="Search the web",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                            },
                            "required": ["query"],
                        },
                    )
                ],
            )
            
            response = await openai_provider.complete(request)
        
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "search_web"
        assert response.finish_reason == "tool_calls"
    
    @pytest.mark.asyncio
    async def test_openai_rate_limit_error(self, openai_provider: Any) -> None:
        """Test OpenAI rate limit error handling."""
        from autoconstitution.providers.openai import (
            Message,
            CompletionRequest,
            OpenAIRateLimitError,
        )
        
        await openai_provider.initialize()
        
        with patch.object(
            openai_provider._rate_limiter, "acquire", new_callable=AsyncMock
        ), patch.object(
            openai_provider._retry_handler,
            "execute",
            new_callable=AsyncMock,
        ) as mock_execute:
            mock_execute.side_effect = OpenAIRateLimitError(
                "Rate limit exceeded",
                retry_after=15.0,
                status_code=429,
            )
            
            request = CompletionRequest(messages=[Message.user("Test")])
            
            with pytest.raises(OpenAIRateLimitError) as exc_info:
                await openai_provider.complete(request)
            
            assert exc_info.value.retry_after == 15.0
            assert exc_info.value.status_code == 429
    
    @pytest.mark.asyncio
    async def test_openai_authentication_error(self, openai_provider: Any) -> None:
        """Test OpenAI authentication error handling."""
        from autoconstitution.providers.openai import OpenAIAuthenticationError
        
        await openai_provider.initialize()
        
        error = OpenAIAuthenticationError("Invalid API key")
        assert error.status_code == 401
        assert "Invalid API key" in str(error)
    
    @pytest.mark.asyncio
    async def test_openai_server_error(self, openai_provider: Any) -> None:
        """Test OpenAI server error handling."""
        from autoconstitution.providers.openai import OpenAIServerError
        
        await openai_provider.initialize()
        
        error = OpenAIServerError("Internal server error", status_code=500)
        assert error.status_code == 500
        assert "Internal server error" in str(error)
    
    @pytest.mark.asyncio
    async def test_openai_embedding(self, openai_provider: Any) -> None:
        """Test OpenAI embedding generation."""
        from autoconstitution.providers.openai import (
            EmbeddingRequest,
            EmbeddingResponse,
            TokenUsage,
        )
        
        await openai_provider.initialize()
        
        # Mock embedding response
        mock_response = EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5]],
            model="text-embedding-3-small",
            usage=TokenUsage(prompt_tokens=5, total_tokens=5),
        )
        
        with patch.object(
            openai_provider._rate_limiter, "acquire", new_callable=AsyncMock
        ), patch.object(
            openai_provider._retry_handler,
            "execute",
            new_callable=AsyncMock,
        ) as mock_execute:
            mock_execute.return_value = mock_response
            
            request = EmbeddingRequest(
                texts=["Hello world"],
                model="text-embedding-3-small",
            )
            
            response = await openai_provider.embed(request)
        
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == 5
        assert response.model == "text-embedding-3-small"
    
    @pytest.mark.asyncio
    async def test_openai_rate_limiter(self) -> None:
        """Test OpenAI rate limiter."""
        from autoconstitution.providers.openai import (
            TokenBucketRateLimiter,
            RateLimitConfig,
        )
        
        config = RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=150000,
            burst_size=10,
        )
        limiter = TokenBucketRateLimiter(config)
        
        # Should not raise
        await limiter.acquire(tokens=100)
    
    @pytest.mark.asyncio
    async def test_openai_retry_handler(self) -> None:
        """Test OpenAI retry handler."""
        from autoconstitution.providers.openai import (
            ExponentialBackoffRetry,
            RetryConfig,
        )
        
        config = RetryConfig(
            max_retries=3,
            base_delay=0.1,
            max_delay=1.0,
        )
        retry_handler = ExponentialBackoffRetry(config)
        
        # Test successful operation
        async def success_op():
            return "success"
        
        result = await retry_handler.execute(success_op, "test")
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_openai_available_models(self, openai_provider: Any) -> None:
        """Test OpenAI available models."""
        models = openai_provider.available_models
        
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert "gpt-4-turbo" in models
        assert "gpt-3.5-turbo" in models
    
    @pytest.mark.asyncio
    async def test_openai_model_token_limits(self, openai_provider: Any) -> None:
        """Test OpenAI model token limits."""
        assert openai_provider.get_model_token_limit("gpt-4o") == 128000
        assert openai_provider.get_model_token_limit("gpt-4") == 8192
        assert openai_provider.get_model_token_limit("gpt-3.5-turbo") == 16385
    
    @pytest.mark.asyncio
    async def test_openai_context_manager(self, openai_config: Dict[str, Any]) -> None:
        """Test OpenAI provider async context manager."""
        from autoconstitution.providers.openai import OpenAIProvider, ProviderConfig
        
        config = ProviderConfig(**openai_config)
        
        with patch(
            "autoconstitution.providers.openai.openai.AsyncOpenAI"
        ) as mock_openai:
            async with OpenAIProvider(config) as provider:
                assert provider._initialized is True


# =============================================================================
# Ollama Provider Tests
# =============================================================================


class TestOllamaProvider:
    """Test suite for Ollama provider."""
    
    @pytest.fixture
    def ollama_config(self) -> Dict[str, Any]:
        """Default Ollama configuration for tests."""
        return {
            "base_url": "http://localhost:11434",
            "timeout": 30.0,
            "max_retries": 2,
            "default_model": "llama3.2",
        }
    
    @pytest.fixture
    def ollama_provider(self, ollama_config: Dict[str, Any]):
        """Create an Ollama provider instance."""
        from autoconstitution.providers.ollama import OllamaProvider, OllamaConfig
        
        config = OllamaConfig(**ollama_config)
        return OllamaProvider(config)
    
    @pytest.fixture
    def mock_ollama_chat_response(self) -> Dict[str, Any]:
        """Mock Ollama chat response."""
        return {
            "model": "llama3.2",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "Hello! I'm a local AI running on Ollama.",
            },
            "done": True,
            "total_duration": 1234567890,
            "load_duration": 123456789,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 12345678,
            "eval_count": 15,
            "eval_duration": 987654321,
        }
    
    @pytest.mark.asyncio
    async def test_ollama_provider_initialization(
        self, ollama_config: Dict[str, Any]
    ) -> None:
        """Test Ollama provider initialization."""
        from autoconstitution.providers.ollama import OllamaProvider, OllamaConfig
        
        config = OllamaConfig(**ollama_config)
        provider = OllamaProvider(config)
        
        assert provider.config.base_url == "http://localhost:11434"
        assert provider.default_model == "llama3.2"
        assert provider.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_ollama_provider_apple_silicon_detection(self) -> None:
        """Test Apple Silicon detection."""
        from autoconstitution.providers.ollama import (
            detect_apple_silicon,
            get_optimal_thread_count,
            get_optimal_gpu_layers,
        )
        
        # These should return values without error
        is_apple = detect_apple_silicon()
        assert isinstance(is_apple, bool)
        
        threads = get_optimal_thread_count()
        assert isinstance(threads, int)
        assert threads > 0
        
        gpu_layers = get_optimal_gpu_layers()
        assert isinstance(gpu_layers, int)
    
    @pytest.mark.asyncio
    async def test_ollama_complete_success(
        self,
        ollama_provider: Any,
        mock_ollama_chat_response: Dict[str, Any],
        mock_aiohttp_session: AsyncMock,
        mock_response: Callable,
    ) -> None:
        """Test successful Ollama completion."""
        from autoconstitution.providers.ollama import (
            Message,
            CompletionRequest,
            CompletionResponse,
            TokenUsage,
        )
        
        # Setup mock
        mock_resp = mock_response(status=200, json_data=mock_ollama_chat_response)
        
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_aiohttp_session.post.return_value = mock_context
        
        # Mock connection check
        mock_get_context = AsyncMock()
        mock_get_resp = mock_response(status=200, json_data={"models": []})
        mock_get_context.__aenter__ = AsyncMock(return_value=mock_get_resp)
        mock_get_context.__aexit__ = AsyncMock(return_value=None)
        mock_aiohttp_session.get.return_value = mock_get_context
        
        await ollama_provider.initialize()
        ollama_provider._session = mock_aiohttp_session
        
        # Create expected response
        response = CompletionResponse(
            content="Hello! I'm a local AI running on Ollama.",
            model="llama3.2",
            usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=15,
                total_tokens=25,
            ),
            finish_reason="stop",
            raw_response=mock_ollama_chat_response,
            created_at="2024-01-01T00:00:00Z",
        )
        
        with patch.object(
            ollama_provider._retry_handler,
            "execute",
            new_callable=AsyncMock,
        ) as mock_execute:
            mock_execute.return_value = response
            
            request = CompletionRequest(
                messages=[Message.user("Hello, who are you?")],
                model="llama3.2",
            )
            
            result = await ollama_provider.complete(request)
        
        assert result.content == "Hello! I'm a local AI running on Ollama."
        assert result.model == "llama3.2"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 15
    
    @pytest.mark.asyncio
    async def test_ollama_connection_error(
        self, ollama_provider: Any, mock_aiohttp_session: AsyncMock
    ) -> None:
        """Test Ollama connection error handling."""
        from autoconstitution.providers.ollama import OllamaConnectionError
        
        # Mock connection failure
        mock_aiohttp_session.get.side_effect = Exception("Connection refused")
        
        with pytest.raises(OllamaConnectionError):
            await ollama_provider.initialize()
    
    @pytest.mark.asyncio
    async def test_ollama_model_not_found_error(
        self, ollama_provider: Any
    ) -> None:
        """Test Ollama model not found error handling."""
        from autoconstitution.providers.ollama import (
            OllamaModelNotFoundError,
        )
        
        await ollama_provider.initialize()
        
        error = OllamaModelNotFoundError("llama3.2", status_code=404)
        assert error.model == "llama3.2"
        assert error.status_code == 404
        assert "llama3.2" in str(error)
    
    @pytest.mark.asyncio
    async def test_ollama_list_models(
        self,
        ollama_provider: Any,
        mock_aiohttp_session: AsyncMock,
        mock_response: Callable,
    ) -> None:
        """Test Ollama list models."""
        from autoconstitution.providers.ollama import ModelInfo
        
        mock_models_response = {
            "models": [
                {
                    "name": "llama3.2",
                    "model": "llama3.2:latest",
                    "modified_at": "2024-01-01T00:00:00Z",
                    "size": 2010000000,
                    "digest": "abc123",
                    "details": {"format": "gguf"},
                }
            ]
        }
        
        mock_resp = mock_response(status=200, json_data=mock_models_response)
        
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_aiohttp_session.get.return_value = mock_context
        
        # Mock connection check for initialize
        mock_get_context = AsyncMock()
        mock_get_resp = mock_response(status=200, json_data={"models": []})
        mock_get_context.__aenter__ = AsyncMock(return_value=mock_get_resp)
        mock_get_context.__aexit__ = AsyncMock(return_value=None)
        
        async def mock_get(*args, **kwargs):
            if "/api/tags" in args[0]:
                return mock_context
            return mock_get_context
        
        mock_aiohttp_session.get.side_effect = mock_get
        
        await ollama_provider.initialize()
        ollama_provider._session = mock_aiohttp_session
        
        models = await ollama_provider.list_models()
        
        assert len(models) == 1
        assert models[0].name == "llama3.2"
        assert models[0].size == 2010000000
    
    @pytest.mark.asyncio
    async def test_ollama_model_capabilities(self, ollama_provider: Any) -> None:
        """Test Ollama model capabilities."""
        # Test tool support
        assert ollama_provider.supports_tools("llama3.2") is True
        assert ollama_provider.supports_tools("mistral") is True
        assert ollama_provider.supports_tools("phi3") is False
        
        # Test vision support
        assert ollama_provider.supports_vision("llava") is True
        assert ollama_provider.supports_vision("llama3.2") is False
        
        # Test context window
        assert ollama_provider.get_model_context_window("llama3.2") == 128000
        assert ollama_provider.get_model_context_window("mistral") == 32768
    
    @pytest.mark.asyncio
    async def test_ollama_embedding(
        self,
        ollama_provider: Any,
        mock_aiohttp_session: AsyncMock,
        mock_response: Callable,
    ) -> None:
        """Test Ollama embedding generation."""
        from autoconstitution.providers.ollama import EmbeddingRequest
        
        mock_embed_response = {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
        
        mock_resp = mock_response(status=200, json_data=mock_embed_response)
        
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_aiohttp_session.post.return_value = mock_context
        
        await ollama_provider.initialize()
        ollama_provider._session = mock_aiohttp_session
        
        request = EmbeddingRequest(
            texts=["Hello world"],
            model="nomic-embed-text",
        )
        
        response = await ollama_provider.embed(request)
        
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == 5
    
    @pytest.mark.asyncio
    async def test_ollama_health_check(
        self,
        ollama_provider: Any,
        mock_aiohttp_session: AsyncMock,
        mock_response: Callable,
    ) -> None:
        """Test Ollama health check."""
        mock_resp = mock_response(status=200, json_data={"models": []})
        
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_aiohttp_session.get.return_value = mock_context
        
        await ollama_provider.initialize()
        ollama_provider._session = mock_aiohttp_session
        
        is_healthy = await ollama_provider.health_check()
        
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_ollama_tool_executor(self) -> None:
        """Test Ollama tool executor."""
        from autoconstitution.providers.ollama import ToolExecutor
        
        executor = ToolExecutor()
        
        # Register async function
        async def async_search(query: str) -> str:
            return f"Results for: {query}"
        
        executor.register("search", async_search)
        
        # Execute tool call
        tool_call = {
            "function": {
                "name": "search",
                "arguments": '{"query": "AI news"}',
            }
        }
        
        result = await executor.execute(tool_call)
        
        assert '"result"' in result
        assert "AI news" in result
    
    @pytest.mark.asyncio
    async def test_ollama_build_options(self, ollama_provider: Any) -> None:
        """Test Ollama options building."""
        from autoconstitution.providers.ollama import CompletionRequest
        
        request = CompletionRequest(
            messages=[],
            temperature=0.8,
            top_p=0.9,
            max_tokens=100,
            num_ctx=4096,
            seed=42,
        )
        
        options = ollama_provider._build_options(request)
        
        assert options["temperature"] == 0.8
        assert options["top_p"] == 0.9
        assert options["num_predict"] == 100
        assert options["num_ctx"] == 4096
        assert options["seed"] == 42
    
    @pytest.mark.asyncio
    async def test_ollama_context_manager(self, ollama_config: Dict[str, Any]) -> None:
        """Test Ollama provider async context manager."""
        from autoconstitution.providers.ollama import OllamaProvider, OllamaConfig
        
        config = OllamaConfig(**ollama_config)
        
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False
            
            # Mock connection check
            mock_context = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_context.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_session.get.return_value = mock_context
            
            async with OllamaProvider(config) as provider:
                assert provider.is_initialized is True


# =============================================================================
# Common Interface Compliance Tests
# =============================================================================


class TestCommonInterfaceCompliance:
    """Test suite for common interface compliance across all providers."""
    
    @pytest.mark.asyncio
    async def test_all_providers_have_initialize(self) -> None:
        """Test that all providers have initialize method."""
        from autoconstitution.providers.kimi import KimiProvider, KimiConfig
        from autoconstitution.providers.anthropic import AnthropicProvider, AnthropicConfig
        from autoconstitution.providers.openai import OpenAIProvider, ProviderConfig
        from autoconstitution.providers.ollama import OllamaProvider, OllamaConfig
        
        providers = [
            (KimiProvider, KimiConfig(api_key="test")),
            (AnthropicProvider, AnthropicConfig(api_key="test")),
            (OpenAIProvider, ProviderConfig(api_key="test")),
            (OllamaProvider, OllamaConfig()),
        ]
        
        for provider_class, config in providers:
            provider = provider_class(config)
            assert hasattr(provider, "initialize")
            assert asyncio.iscoroutinefunction(provider.initialize)
    
    @pytest.mark.asyncio
    async def test_all_providers_have_complete(self) -> None:
        """Test that all providers have complete method."""
        from autoconstitution.providers.kimi import KimiProvider, KimiConfig
        from autoconstitution.providers.anthropic import AnthropicProvider, AnthropicConfig
        from autoconstitution.providers.openai import OpenAIProvider, ProviderConfig
        from autoconstitution.providers.ollama import OllamaProvider, OllamaConfig
        
        providers = [
            (KimiProvider, KimiConfig(api_key="test")),
            (AnthropicProvider, AnthropicConfig(api_key="test")),
            (OpenAIProvider, ProviderConfig(api_key="test")),
            (OllamaProvider, OllamaConfig()),
        ]
        
        for provider_class, config in providers:
            provider = provider_class(config)
            assert hasattr(provider, "complete")
            assert asyncio.iscoroutinefunction(provider.complete)
    
    @pytest.mark.asyncio
    async def test_all_providers_have_close(self) -> None:
        """Test that all providers have close method."""
        from autoconstitution.providers.kimi import KimiProvider, KimiConfig
        from autoconstitution.providers.anthropic import AnthropicProvider, AnthropicConfig
        from autoconstitution.providers.openai import OpenAIProvider, ProviderConfig
        from autoconstitution.providers.ollama import OllamaProvider, OllamaConfig
        
        providers = [
            (KimiProvider, KimiConfig(api_key="test")),
            (AnthropicProvider, AnthropicConfig(api_key="test")),
            (OpenAIProvider, ProviderConfig(api_key="test")),
            (OllamaProvider, OllamaConfig()),
        ]
        
        for provider_class, config in providers:
            provider = provider_class(config)
            assert hasattr(provider, "close")
            assert asyncio.iscoroutinefunction(provider.close)
    
    @pytest.mark.asyncio
    async def test_all_providers_support_context_manager(self) -> None:
        """Test that all providers support async context manager."""
        from autoconstitution.providers.kimi import KimiProvider, KimiConfig
        from autoconstitution.providers.anthropic import AnthropicProvider, AnthropicConfig
        from autoconstitution.providers.openai import OpenAIProvider, ProviderConfig
        from autoconstitution.providers.ollama import OllamaProvider, OllamaConfig
        
        providers = [
            (KimiProvider, KimiConfig(api_key="test")),
            (AnthropicProvider, AnthropicConfig(api_key="test")),
            (OpenAIProvider, ProviderConfig(api_key="test")),
            (OllamaProvider, OllamaConfig()),
        ]
        
        for provider_class, config in providers:
            provider = provider_class(config)
            assert hasattr(provider, "__aenter__")
            assert hasattr(provider, "__aexit__")
    
    @pytest.mark.asyncio
    async def test_all_providers_have_default_model(self) -> None:
        """Test that all providers have default_model property."""
        from autoconstitution.providers.kimi import KimiProvider, KimiConfig
        from autoconstitution.providers.anthropic import AnthropicProvider, AnthropicConfig
        from autoconstitution.providers.openai import OpenAIProvider, ProviderConfig
        from autoconstitution.providers.ollama import OllamaProvider, OllamaConfig
        
        providers = [
            (KimiProvider, KimiConfig(api_key="test"), "kimi-k2-5"),
            (AnthropicProvider, AnthropicConfig(api_key="test"), "claude-3-5-sonnet-20241022"),
            (OpenAIProvider, ProviderConfig(api_key="test"), "gpt-4o"),
            (OllamaProvider, OllamaConfig(), "llama3.2"),
        ]
        
        for provider_class, config, expected_default in providers:
            provider = provider_class(config)
            assert hasattr(provider, "default_model")
            assert provider.default_model == expected_default
    
    @pytest.mark.asyncio
    async def test_all_providers_have_available_models(self) -> None:
        """Test that all providers have available_models property."""
        from autoconstitution.providers.kimi import KimiProvider, KimiConfig
        from autoconstitution.providers.anthropic import AnthropicProvider, AnthropicConfig
        from autoconstitution.providers.openai import OpenAIProvider, ProviderConfig
        from autoconstitution.providers.ollama import OllamaProvider, OllamaConfig
        
        providers = [
            (KimiProvider, KimiConfig(api_key="test")),
            (AnthropicProvider, AnthropicConfig(api_key="test")),
            (OpenAIProvider, ProviderConfig(api_key="test")),
            (OllamaProvider, OllamaConfig()),
        ]
        
        for provider_class, config in providers:
            provider = provider_class(config)
            assert hasattr(provider, "available_models")
            assert isinstance(provider.available_models, list)
            assert len(provider.available_models) > 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test suite for error handling across all providers."""
    
    @pytest.mark.asyncio
    async def test_kimi_errors_have_status_code(self) -> None:
        """Test Kimi errors include status codes."""
        from autoconstitution.providers.kimi import (
            KimiError,
            KimiRateLimitError,
            KimiAuthenticationError,
            KimiValidationError,
            KimiServerError,
        )
        
        errors = [
            KimiRateLimitError("Rate limit", status_code=429),
            KimiAuthenticationError("Auth failed", status_code=401),
            KimiValidationError("Invalid", status_code=400),
            KimiServerError("Server error", status_code=500),
        ]
        
        for error in errors:
            assert error.status_code is not None
            assert isinstance(error.status_code, int)
    
    @pytest.mark.asyncio
    async def test_anthropic_errors_have_status_code(self) -> None:
        """Test Anthropic errors include status codes."""
        from autoconstitution.providers.anthropic import (
            AnthropicError,
            AnthropicRateLimitError,
            AnthropicAuthenticationError,
            AnthropicValidationError,
        )
        
        errors = [
            AnthropicRateLimitError("Rate limit", status_code=429),
            AnthropicAuthenticationError("Auth failed", status_code=401),
            AnthropicValidationError("Invalid", status_code=400),
        ]
        
        for error in errors:
            assert error.status_code is not None
            assert isinstance(error.status_code, int)
    
    @pytest.mark.asyncio
    async def test_openai_errors_have_status_code(self) -> None:
        """Test OpenAI errors include status codes."""
        from autoconstitution.providers.openai import (
            OpenAIError,
            OpenAIRateLimitError,
            OpenAIAuthenticationError,
            OpenAIInvalidRequestError,
            OpenAIServerError,
        )
        
        errors = [
            OpenAIRateLimitError("Rate limit", status_code=429),
            OpenAIAuthenticationError("Auth failed"),
            OpenAIInvalidRequestError("Invalid"),
            OpenAIServerError("Server error"),
        ]
        
        for error in errors:
            assert error.status_code is not None
            assert isinstance(error.status_code, int)
    
    @pytest.mark.asyncio
    async def test_ollama_errors_have_status_code(self) -> None:
        """Test Ollama errors include status codes."""
        from autoconstitution.providers.ollama import (
            OllamaError,
            OllamaConnectionError,
            OllamaModelNotFoundError,
            OllamaValidationError,
            OllamaServerError,
        )
        
        errors = [
            OllamaConnectionError("Connection failed"),
            OllamaModelNotFoundError("model", status_code=404),
            OllamaValidationError("Invalid", status_code=400),
            OllamaServerError("Server error", status_code=500),
        ]
        
        for error in errors:
            assert hasattr(error, "status_code")
    
    @pytest.mark.asyncio
    async def test_rate_limit_errors_have_retry_after(self) -> None:
        """Test that rate limit errors include retry_after."""
        from autoconstitution.providers.kimi import KimiRateLimitError
        from autoconstitution.providers.anthropic import AnthropicRateLimitError
        from autoconstitution.providers.openai import OpenAIRateLimitError
        
        errors = [
            KimiRateLimitError("Rate limit", retry_after=5.0),
            AnthropicRateLimitError("Rate limit", retry_after=10.0),
            OpenAIRateLimitError("Rate limit", retry_after=15.0),
        ]
        
        for error in errors:
            assert hasattr(error, "retry_after")
            assert isinstance(error.retry_after, float)
    
    @pytest.mark.asyncio
    async def test_error_inheritance(self) -> None:
        """Test that all provider errors inherit from base exception."""
        from autoconstitution.providers.kimi import KimiError
        from autoconstitution.providers.anthropic import AnthropicError
        from autoconstitution.providers.openai import OpenAIError
        from autoconstitution.providers.ollama import OllamaError
        
        # All should inherit from Exception
        assert issubclass(KimiError, Exception)
        assert issubclass(AnthropicError, Exception)
        assert issubclass(OpenAIError, Exception)
        assert issubclass(OllamaError, Exception)


# =============================================================================
# Data Class Tests
# =============================================================================


class TestDataClasses:
    """Test suite for provider data classes."""
    
    @pytest.mark.asyncio
    async def test_message_creation(self) -> None:
        """Test Message creation across providers."""
        from autoconstitution.providers.kimi import Message as KimiMessage
        from autoconstitution.providers.anthropic import Message as AnthropicMessage
        from autoconstitution.providers.openai import Message as OpenAIMessage
        from autoconstitution.providers.ollama import Message as OllamaMessage
        
        # Test system message
        kimi_sys = KimiMessage.system("You are helpful")
        anthropic_sys = AnthropicMessage.system("You are helpful")
        openai_sys = OpenAIMessage.system("You are helpful")
        ollama_sys = OllamaMessage.system("You are helpful")
        
        assert kimi_sys.content == "You are helpful"
        assert anthropic_sys.content == "You are helpful"
        assert openai_sys.content == "You are helpful"
        assert ollama_sys.content == "You are helpful"
        
        # Test user message
        kimi_user = KimiMessage.user("Hello")
        anthropic_user = AnthropicMessage.user("Hello")
        openai_user = OpenAIMessage.user("Hello")
        ollama_user = OllamaMessage.user("Hello")
        
        assert kimi_user.content == "Hello"
        assert anthropic_user.content == "Hello"
        assert openai_user.content == "Hello"
        assert ollama_user.content == "Hello"
    
    @pytest.mark.asyncio
    async def test_token_usage_cost_estimate(self) -> None:
        """Test TokenUsage cost estimation."""
        from autoconstitution.providers.kimi import TokenUsage as KimiUsage
        from autoconstitution.providers.anthropic import TokenUsage as AnthropicUsage
        from autoconstitution.providers.openai import TokenUsage as OpenAIUsage
        from autoconstitution.providers.ollama import TokenUsage as OllamaUsage
        
        # Kimi usage
        kimi_usage = KimiUsage(
            prompt_tokens=1000, completion_tokens=500, total_tokens=1500
        )
        assert kimi_usage.cost_estimate > 0
        
        # Anthropic usage
        anthropic_usage = AnthropicUsage(
            prompt_tokens=1000, completion_tokens=500, total_tokens=1500
        )
        assert anthropic_usage.cost_estimate > 0
        
        # OpenAI usage
        openai_usage = OpenAIUsage(
            prompt_tokens=1000, completion_tokens=500, total_tokens=1500
        )
        assert hasattr(openai_usage, "cost_estimate")
        
        # Ollama usage (should be free)
        ollama_usage = OllamaUsage(
            prompt_tokens=1000, completion_tokens=500, total_tokens=1500
        )
        assert ollama_usage.cost_estimate == 0.0
    
    @pytest.mark.asyncio
    async def test_tool_creation(self) -> None:
        """Test Tool creation across providers."""
        from autoconstitution.providers.kimi import Tool as KimiTool
        from autoconstitution.providers.anthropic import Tool as AnthropicTool
        from autoconstitution.providers.openai import Tool as OpenAITool
        from autoconstitution.providers.ollama import Tool as OllamaTool
        
        params = {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }
        
        kimi_tool = KimiTool(
            name="get_weather", description="Get weather", parameters=params
        )
        anthropic_tool = AnthropicTool(
            name="get_weather", description="Get weather", parameters=params
        )
        openai_tool = OpenAITool(
            name="get_weather", description="Get weather", parameters=params
        )
        ollama_tool = OllamaTool(
            name="get_weather", description="Get weather", parameters=params
        )
        
        assert kimi_tool.name == "get_weather"
        assert anthropic_tool.name == "get_weather"
        assert openai_tool.name == "get_weather"
        assert ollama_tool.name == "get_weather"


# =============================================================================
# Retry Handler Tests
# =============================================================================


class TestRetryHandlers:
    """Test suite for retry handlers."""
    
    @pytest.mark.asyncio
    async def test_kimi_retry_handler(self) -> None:
        """Test Kimi retry handler."""
        from autoconstitution.providers.kimi import (
            ExponentialBackoffRetry,
            RetryConfig,
            KimiRateLimitError,
        )
        
        config = RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.1)
        handler = ExponentialBackoffRetry(config)
        
        # Test successful operation
        async def success():
            return "success"
        
        result = await handler.execute(success, "test")
        assert result == "success"
        
        # Test retry on rate limit
        call_count = 0
        
        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise KimiRateLimitError("Rate limit", retry_after=0.01)
            return "success"
        
        result = await handler.execute(fail_then_succeed, "test")
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_anthropic_retry_handler(self) -> None:
        """Test Anthropic retry handler."""
        from autoconstitution.providers.anthropic import (
            RetryHandler,
            AnthropicRateLimitError,
        )
        
        handler = RetryHandler(max_retries=2, min_delay=0.01, max_delay=0.1)
        
        # Test successful operation
        async def success():
            return "success"
        
        result = await handler.execute(success)
        assert result == "success"
        
        # Test is_retryable_error
        assert handler.is_retryable_error(AnthropicRateLimitError("test")) is True
    
    @pytest.mark.asyncio
    async def test_openai_retry_handler(self) -> None:
        """Test OpenAI retry handler."""
        from autoconstitution.providers.openai import (
            ExponentialBackoffRetry,
            RetryConfig,
            OpenAIRateLimitError,
        )
        
        config = RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.1)
        handler = ExponentialBackoffRetry(config)
        
        # Test successful operation
        async def success():
            return "success"
        
        result = await handler.execute(success, "test")
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_ollama_retry_handler(self) -> None:
        """Test Ollama retry handler."""
        from autoconstitution.providers.ollama import (
            RetryHandler,
            OllamaConnectionError,
            OllamaTimeoutError,
        )
        
        handler = RetryHandler(max_retries=2, base_delay=0.01, max_delay=0.1)
        
        # Test successful operation
        async def success():
            return "success"
        
        result = await handler.execute(success)
        assert result == "success"
        
        # Test is_retryable_error
        assert handler.is_retryable_error(OllamaConnectionError("test")) is True
        assert handler.is_retryable_error(OllamaTimeoutError("test")) is True


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    @pytest.mark.asyncio
    async def test_kimi_convenience_functions_exist(self) -> None:
        """Test Kimi convenience functions exist."""
        from autoconstitution.providers.kimi import (
            create_completion,
            create_streaming_completion,
        )
        
        assert asyncio.iscoroutinefunction(create_completion)
        assert hasattr(create_streaming_completion, "__call__")
    
    @pytest.mark.asyncio
    async def test_anthropic_convenience_functions_exist(self) -> None:
        """Test Anthropic convenience functions exist."""
        from autoconstitution.providers.anthropic import (
            create_anthropic_provider,
            anthropic_complete,
            anthropic_complete_text,
        )
        
        assert asyncio.iscoroutinefunction(create_anthropic_provider)
        assert asyncio.iscoroutinefunction(anthropic_complete)
        assert asyncio.iscoroutinefunction(anthropic_complete_text)
    
    @pytest.mark.asyncio
    async def test_openai_convenience_functions_exist(self) -> None:
        """Test OpenAI convenience functions exist."""
        from autoconstitution.providers.openai import (
            openai_complete,
            openai_complete_text,
        )
        
        assert asyncio.iscoroutinefunction(openai_complete)
        assert asyncio.iscoroutinefunction(openai_complete_text)
    
    @pytest.mark.asyncio
    async def test_ollama_convenience_functions_exist(self) -> None:
        """Test Ollama convenience functions exist."""
        from autoconstitution.providers.ollama import (
            create_ollama_provider,
            ollama_complete,
            ollama_complete_text,
            ollama_generate,
        )
        
        assert asyncio.iscoroutinefunction(create_ollama_provider)
        assert asyncio.iscoroutinefunction(ollama_complete)
        assert asyncio.iscoroutinefunction(ollama_complete_text)
        assert asyncio.iscoroutinefunction(ollama_generate)


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Test suite for module exports."""
    
    @pytest.mark.asyncio
    async def test_kimi_exports(self) -> None:
        """Test Kimi module exports."""
        from autoconstitution.providers import kimi
        
        expected_exports = [
            "KimiProvider",
            "KimiClient",
            "KimiConfig",
            "Message",
            "Tool",
            "CompletionRequest",
            "CompletionResponse",
            "KimiError",
            "KimiRateLimitError",
            "KimiAuthenticationError",
        ]
        
        for export in expected_exports:
            assert hasattr(kimi, export), f"Kimi module missing export: {export}"
    
    @pytest.mark.asyncio
    async def test_anthropic_exports(self) -> None:
        """Test Anthropic module exports."""
        from autoconstitution.providers import anthropic
        
        expected_exports = [
            "AnthropicProvider",
            "AnthropicConfig",
            "Message",
            "Tool",
            "CompletionRequest",
            "CompletionResponse",
            "AnthropicError",
            "AnthropicRateLimitError",
            "AnthropicAuthenticationError",
        ]
        
        for export in expected_exports:
            assert hasattr(anthropic, export), f"Anthropic module missing export: {export}"
    
    @pytest.mark.asyncio
    async def test_openai_exports(self) -> None:
        """Test OpenAI module exports."""
        from autoconstitution.providers import openai
        
        expected_exports = [
            "OpenAIProvider",
            "Message",
            "Tool",
            "CompletionRequest",
            "CompletionResponse",
            "OpenAIError",
            "OpenAIRateLimitError",
            "OpenAIAuthenticationError",
        ]
        
        for export in expected_exports:
            assert hasattr(openai, export), f"OpenAI module missing export: {export}"
    
    @pytest.mark.asyncio
    async def test_ollama_exports(self) -> None:
        """Test Ollama module exports."""
        from autoconstitution.providers import ollama
        
        expected_exports = [
            "OllamaProvider",
            "OllamaConfig",
            "Message",
            "Tool",
            "CompletionRequest",
            "CompletionResponse",
            "OllamaError",
            "OllamaConnectionError",
            "OllamaModelNotFoundError",
        ]
        
        for export in expected_exports:
            assert hasattr(ollama, export), f"Ollama module missing export: {export}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
