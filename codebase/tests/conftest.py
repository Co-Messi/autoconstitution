"""
Pytest configuration and shared fixtures for autoconstitution tests.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


# =============================================================================
# Event Loop Configuration
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Shared Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_response() -> Callable:
    """Factory fixture for creating mock HTTP responses."""
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


@pytest.fixture
def mock_aiohttp_session():
    """Fixture for mocking aiohttp ClientSession."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.closed = False
        yield mock_session


@pytest.fixture
def mock_anthropic_client():
    """Fixture for mocking Anthropic client."""
    mock_client = AsyncMock()
    
    # Mock messages.create response
    mock_response = MagicMock()
    mock_response.id = "msg_test123"
    mock_response.model = "claude-3-5-sonnet-20241022"
    mock_response.stop_reason = "end_turn"
    mock_response.stop_sequence = None
    
    # Mock content blocks
    content_block = MagicMock()
    content_block.type = "text"
    content_block.text = "Test response from Claude."
    mock_response.content = [content_block]
    
    # Mock usage
    mock_usage = MagicMock()
    mock_usage.input_tokens = 10
    mock_usage.output_tokens = 15
    mock_response.usage = mock_usage
    
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    
    return mock_client


@pytest.fixture
def mock_openai_client():
    """Fixture for mocking OpenAI client."""
    mock_client = AsyncMock()
    
    # Mock chat.completions.create response
    mock_response = MagicMock()
    mock_response.model = "gpt-4o"
    
    # Mock choice
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_message = MagicMock()
    mock_message.content = "Test response from GPT."
    mock_message.tool_calls = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    # Mock usage
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 15
    mock_usage.total_tokens = 25
    mock_response.usage = mock_usage
    
    mock_response.model_dump = MagicMock(return_value={
        "id": "chatcmpl_test123",
        "model": "gpt-4o",
        "choices": [{"message": {"content": "Test response"}}],
    })
    
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    return mock_client


# =============================================================================
# Provider Configuration Fixtures
# =============================================================================


@pytest.fixture
def kimi_config() -> Dict[str, Any]:
    """Default Kimi configuration for tests."""
    return {
        "api_key": "test-kimi-api-key",
        "base_url": "https://api.moonshot.cn/v1",
        "timeout": 30.0,
        "max_retries": 2,
    }


@pytest.fixture
def anthropic_config() -> Dict[str, Any]:
    """Default Anthropic configuration for tests."""
    return {
        "api_key": "test-anthropic-api-key",
        "base_url": "https://api.anthropic.com",
        "timeout": 30.0,
        "max_retries": 2,
        "default_model": "claude-3-5-sonnet-20241022",
    }


@pytest.fixture
def openai_config() -> Dict[str, Any]:
    """Default OpenAI configuration for tests."""
    return {
        "api_key": "test-openai-api-key",
        "base_url": None,
        "timeout": 30.0,
        "max_retries": 2,
        "default_model": "gpt-4o",
    }


@pytest.fixture
def ollama_config() -> Dict[str, Any]:
    """Default Ollama configuration for tests."""
    return {
        "base_url": "http://localhost:11434",
        "timeout": 30.0,
        "max_retries": 2,
        "default_model": "llama3.2",
    }


# =============================================================================
# Mock Response Data Fixtures
# =============================================================================


@pytest.fixture
def mock_kimi_completion_response() -> Dict[str, Any]:
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
def mock_anthropic_completion_response() -> Dict[str, Any]:
    """Mock Anthropic API completion response."""
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


@pytest.fixture
def mock_openai_completion_response() -> Dict[str, Any]:
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


@pytest.fixture
def mock_ollama_chat_response() -> Dict[str, Any]:
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


# =============================================================================
# Streaming Response Fixtures
# =============================================================================


@pytest.fixture
def mock_kimi_streaming_chunks() -> list:
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


# =============================================================================
# Tool Call Fixtures
# =============================================================================


@pytest.fixture
def sample_tool_definition() -> Dict[str, Any]:
    """Sample tool definition for testing."""
    return {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                },
            },
            "required": ["location"],
        },
    }


@pytest.fixture
def sample_tool_call() -> Dict[str, Any]:
    """Sample tool call for testing."""
    return {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "San Francisco", "unit": "celsius"}',
        },
    }


# =============================================================================
# Error Fixtures
# =============================================================================


@pytest.fixture
def rate_limit_response_headers() -> Dict[str, str]:
    """Rate limit response headers."""
    return {
        "retry-after": "5",
        "x-ratelimit-limit-requests": "60",
        "x-ratelimit-remaining-requests": "0",
        "x-ratelimit-reset-requests": "5s",
    }


# Import patch for fixtures
from unittest.mock import patch
