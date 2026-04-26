# Kimi K2.5 Provider for autoconstitution

A comprehensive async client for Moonshot AI's Kimi API with support for chat completions, tool calling, streaming responses, and rate limit management.

## Features

- **Full Async Support**: Built on `aiohttp` for high-performance async operations
- **Tool Calling**: Native support for function calling with automatic tool execution
- **Streaming**: Real-time streaming responses for low-latency applications
- **Error Handling**: Comprehensive exception hierarchy with automatic retries
- **Rate Limiting**: Client-side rate limiter with token bucket algorithm
- **Type Hints**: Complete type annotations for IDE support
- **autoconstitution Compatible**: Identical interface to Anthropic and OpenAI providers

## Installation

```bash
pip install aiohttp
```

Set your API key:
```bash
export KIMI_API_KEY="your-api-key"
```

## Quick Start

### Basic Usage with KimiProvider

```python
import asyncio
from autoconstitution.providers.kimi import KimiProvider, KimiConfig, CompletionRequest, Message

async def main():
    config = KimiConfig(api_key="your-api-key")
    provider = KimiProvider(config)
    
    await provider.initialize()
    
    response = await provider.complete(CompletionRequest(
        messages=[Message.user("Hello, Kimi!")],
        model="kimi-k2-5",
    ))
    
    print(response.content)
    await provider.close()

# Or use async context manager
async def main():
    config = KimiConfig()
    async with KimiProvider(config) as provider:
        response = await provider.complete(CompletionRequest(
            messages=[Message.user("Hello, Kimi!")],
        ))
        print(response.content)

asyncio.run(main())
```

### Using Environment Variables

```python
import os
from autoconstitution.providers.kimi import KimiProvider

# Set KIMI_API_KEY environment variable
async with KimiProvider() as provider:  # API key from env
    response = await provider.complete(...)
```

### Streaming Responses

```python
async def stream_example():
    async with KimiProvider() as provider:
        request = CompletionRequest(
            messages=[Message.user("Tell me a story")],
            stream=True,
        )
        
        async for chunk in provider.complete_stream(request):
            print(chunk.content, end="", flush=True)
            if chunk.is_finished:
                print(f"\nFinished: {chunk.finish_reason}")
```

### Tool Calling

```python
from autoconstitution.providers.kimi import Tool, CompletionRequest, Message

# Define tools
tools = [
    Tool(
        name="get_weather",
        description="Get current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    )
]

async def tool_example():
    async with KimiProvider() as provider:
        response = await provider.complete(CompletionRequest(
            messages=[Message.user("What's the weather in Beijing?")],
            tools=tools,
        ))
        
        # Check for tool calls
        if response.tool_calls:
            for tool_call in response.tool_calls:
                print(f"Function: {tool_call['function']['name']}")
                print(f"Arguments: {tool_call['function']['arguments']}")
```

### Tool Execution

```python
async def get_weather(location: str, unit: str = "celsius") -> dict:
    # Your implementation
    return {"temperature": 22, "unit": unit}

async def main():
    async with KimiProvider() as provider:
        # Register tool
        provider.register_tool("get_weather", get_weather)
        
        # Get completion with tool calls
        response = await provider.complete(CompletionRequest(
            messages=[Message.user("What's the weather?")],
            tools=[...],
        ))
        
        # Execute tool calls
        if response.tool_calls:
            results = await provider.execute_tools(response.tool_calls)
            print(results)
```

## Low-level KimiClient

For direct API access, use `KimiClient`:

```python
from autoconstitution.providers.kimi import KimiClient

async with KimiClient(api_key="your-api-key") as client:
    response = await client.chat.completions.create(
        model="kimi-k2-5",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response["choices"][0]["message"]["content"])
```

## Configuration Options

### KimiConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `None` | API key (or `KIMI_API_KEY` env var) |
| `base_url` | `str` | `"https://api.moonshot.cn/v1"` | API base URL |
| `timeout` | `float` | `120.0` | Request timeout in seconds |
| `max_retries` | `int` | `3` | Maximum retry attempts |
| `default_model` | `str` | `"kimi-k2-5"` | Default model ID |
| `extra_headers` | `dict` | `None` | Additional HTTP headers |
| `enable_rate_limiting` | `bool` | `True` | Enable client-side rate limiting |
| `requests_per_minute` | `float` | `60.0` | Rate limit for requests |
| `tokens_per_minute` | `float` | `100000.0` | Rate limit for tokens |
| `min_retry_delay` | `float` | `1.0` | Minimum retry delay |
| `max_retry_delay` | `float` | `60.0` | Maximum retry delay |
| `exponential_base` | `float` | `2.0` | Exponential backoff base |

### CompletionRequest Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `List[Message]` | Conversation messages |
| `model` | `str` | Model ID (uses default if not specified) |
| `temperature` | `float` | Sampling temperature (0-2) |
| `max_tokens` | `int` | Maximum tokens to generate |
| `top_p` | `float` | Nucleus sampling |
| `frequency_penalty` | `float` | Frequency penalty (-2 to 2) |
| `presence_penalty` | `float` | Presence penalty (-2 to 2) |
| `stop` | `List[str]` | Stop sequences |
| `tools` | `List[Tool]` | Available tools |
| `tool_choice` | `str/dict` | Tool selection mode |
| `stream` | `bool` | Enable streaming |
| `response_format` | `dict` | Response format (e.g., `{"type": "json_object"}`) |
| `extra_params` | `dict` | Additional parameters |

## Available Models

```python
from autoconstitution.providers.kimi import KimiModel

KimiModel.KIMI_K2_5    # "kimi-k2-5" - Latest K2.5 model
KimiModel.KIMI_K2      # "kimi-k2"
KimiModel.KIMI_K1_5    # "kimi-k1.5"
KimiModel.KIMI_K1      # "kimi-k1"
KimiModel.KIMI_LATEST  # "kimi-latest" - Always points to latest
```

## Error Handling

```python
from autoconstitution.providers.kimi import (
    KimiError,
    KimiAuthenticationError,
    KimiRateLimitError,
    KimiServerError,
    KimiValidationError,
    KimiTimeoutError,
    KimiConnectionError,
)

try:
    response = await provider.complete(request)
except KimiAuthenticationError:
    print("Invalid API key")
except KimiRateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}s")
except KimiServerError as e:
    print(f"Server error: {e.status_code}")
except KimiValidationError as e:
    print(f"Invalid request: {e.message}")
except KimiTimeoutError:
    print("Request timed out")
except KimiConnectionError:
    print("Connection failed")
except KimiError as e:
    print(f"API error: {e.message}")
```

## Rate Limit Information

```python
async with KimiProvider() as provider:
    response = await provider.complete(request)
    
    # Get rate limit info from last request
    rate_limit = provider.get_last_rate_limit_info()
    if rate_limit:
        print(f"Remaining requests: {rate_limit.remaining_requests}")
        print(f"Remaining tokens: {rate_limit.remaining_tokens}")
        print(f"Is rate limited: {rate_limit.is_exceeded}")
```

## Advanced Usage

### Custom Retry Configuration

```python
from autoconstitution.providers.kimi import KimiConfig, RetryConfig

config = KimiConfig(
    max_retries=5,
    min_retry_delay=2.0,
    max_retry_delay=120.0,
    exponential_base=2.0,
)
```

### Disable Rate Limiter

```python
config = KimiConfig(enable_rate_limiting=False)
provider = KimiProvider(config)
```

### Custom Timeouts

```python
config = KimiConfig(timeout=300.0)  # 5 minutes
provider = KimiProvider(config)
```

### Extra Headers

```python
config = KimiConfig(
    extra_headers={
        "X-Custom-Header": "value",
    }
)
provider = KimiProvider(config)
```

## Convenience Functions

```python
from autoconstitution.providers.kimi import create_completion, create_streaming_completion

# Simple completion
response = await create_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    model="kimi-k2-5",
)

# Streaming completion
async for chunk in create_streaming_completion(
    messages=[{"role": "user", "content": "Hello!"}],
):
    print(chunk.content, end="")
```

## API Reference

### KimiProvider

Main provider class for autoconstitution integration.

**Methods:**
- `initialize()` - Initialize the provider
- `close()` - Close and release resources
- `complete(request)` - Generate a completion
- `complete_stream(request)` - Generate a streaming completion
- `register_tool(name, func)` - Register a tool function
- `execute_tools(tool_calls)` - Execute tool calls
- `get_tool_definition(name, description, parameters)` - Create a tool definition
- `get_last_rate_limit_info()` - Get rate limit info

### KimiClient

Low-level client for direct API access.

**Methods:**
- `chat.completions.create(...)` - Create chat completion

### Data Classes

- `Message` - Chat message with factory methods (`system()`, `user()`, `assistant()`, `tool()`)
- `Tool` - Tool definition
- `CompletionRequest` - Request parameters
- `CompletionResponse` - Response data
- `StreamChunk` - Streaming chunk
- `TokenUsage` - Token usage statistics
- `RateLimitInfo` - Rate limit information

### Exceptions

- `KimiError` - Base exception
- `KimiAPIError` - General API error
- `KimiAuthenticationError` - 401 Unauthorized
- `KimiRateLimitError` - 429 Rate Limited
- `KimiServerError` - 5xx Server Error
- `KimiValidationError` - 400 Bad Request
- `KimiTimeoutError` - Request timeout
- `KimiConnectionError` - Connection failure

## License

MIT License
