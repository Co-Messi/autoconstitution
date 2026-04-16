"""
Kimi K2.5 Provider for autoconstitution

A comprehensive async client for Moonshot AI's Kimi API with support for:
- Chat completions with Kimi K2.5, K2, K1.5 models
- Tool/function calling
- Streaming responses
- Rate limit management
- Comprehensive error handling
- Full type hints

Usage:
    from autoconstitution.providers.kimi import KimiProvider, KimiConfig
    
    config = KimiConfig(api_key="your-api-key")
    provider = KimiProvider(config)
    await provider.initialize()
    
    response = await provider.complete(
        CompletionRequest(messages=[Message.user("Hello!")])
    )
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Union

import aiohttp
from aiohttp import ClientTimeout

# =============================================================================
# Exceptions
# =============================================================================


class KimiError(Exception):
    """Base exception for Kimi provider errors."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body or {}


class KimiAPIError(KimiError):
    """Raised when the Kimi API returns an error."""
    pass


class KimiRateLimitError(KimiError):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        status_code: Optional[int] = 429,
        response_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, status_code, response_body)
        self.retry_after = retry_after


class KimiAuthenticationError(KimiError):
    """Raised when authentication fails."""
    pass


class KimiTimeoutError(KimiError):
    """Raised when a request times out."""
    pass


class KimiValidationError(KimiError):
    """Raised when request validation fails."""
    pass


class KimiServerError(KimiError):
    """Raised when Kimi server returns an error."""
    pass


class KimiConnectionError(KimiError):
    """Raised when connection fails."""
    pass


# =============================================================================
# Data Types
# =============================================================================


class Role(Enum):
    """Message roles for chat completions."""
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    """A single message in a conversation."""
    
    role: Role
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        result: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result
    
    @classmethod
    def system(cls, content: str) -> Message:
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)
    
    @classmethod
    def user(cls, content: str, name: Optional[str] = None) -> Message:
        """Create a user message."""
        return cls(role=Role.USER, content=content, name=name)
    
    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> Message:
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)
    
    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> Message:
        """Create a tool response message."""
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id)


@dataclass
class Tool:
    """Definition of a tool/function available to the model."""
    
    name: str
    description: str
    parameters: Dict[str, Any]
    
    def to_kimi_format(self) -> Dict[str, Any]:
        """Convert to Kimi tool format (OpenAI-compatible)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        return self.to_kimi_format()


@dataclass
class TokenUsage:
    """Token usage statistics."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: Optional[int] = None
    
    @property
    def cost_estimate(self) -> float:
        """Rough cost estimate for Kimi models."""
        # Kimi K2.5: ~$2/M input, $8/M output tokens (estimated)
        input_cost = self.prompt_tokens * 0.000002
        output_cost = self.completion_tokens * 0.000008
        return input_cost + output_cost


@dataclass
class CompletionRequest:
    """Request parameters for completion/chat completion."""
    
    messages: List[Message]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    stream: bool = False
    response_format: Optional[Dict[str, Any]] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResponse:
    """Response from a completion request."""
    
    content: str
    model: str
    usage: TokenUsage
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    raw_response: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


@dataclass
class EmbeddingRequest:
    """Request for text embedding."""
    
    texts: List[str]
    model: Optional[str] = None


@dataclass
class EmbeddingResponse:
    """Response from embedding request."""
    
    embeddings: List[List[float]]
    model: str
    usage: TokenUsage


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""
    
    content: str
    is_finished: bool = False
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class RateLimitInfo:
    """Rate limit information from API response headers."""
    
    limit_requests: Optional[int] = None
    remaining_requests: Optional[int] = None
    reset_requests: Optional[str] = None
    limit_tokens: Optional[int] = None
    remaining_tokens: Optional[int] = None
    reset_tokens: Optional[str] = None
    retry_after: Optional[float] = None
    
    @property
    def is_exceeded(self) -> bool:
        """Check if rate limit is approaching."""
        if self.remaining_requests is not None and self.remaining_requests < 5:
            return True
        if self.remaining_tokens is not None and self.remaining_tokens < 1000:
            return True
        return False


# =============================================================================
# Model Enums
# =============================================================================


class KimiModel(str, Enum):
    """Available Kimi models."""
    
    KIMI_K2_5 = "kimi-k2-5"
    KIMI_K2 = "kimi-k2"
    KIMI_K1_5 = "kimi-k1.5"
    KIMI_K1 = "kimi-k1"
    KIMI_LATEST = "kimi-latest"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class KimiConfig:
    """Configuration for Kimi provider."""
    
    api_key: Optional[str] = None
    base_url: str = "https://api.moonshot.cn/v1"
    timeout: float = 120.0
    max_retries: int = 3
    default_model: str = "kimi-k2-5"
    extra_headers: Optional[Dict[str, str]] = None
    
    # Rate limiting
    enable_rate_limiting: bool = True
    requests_per_minute: float = 60.0
    tokens_per_minute: float = 100000.0
    
    # Retry configuration
    min_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    exponential_base: float = 2.0
    
    # Streaming
    stream_buffer_size: int = 1024


# =============================================================================
# Rate Limiter
# =============================================================================


class RateLimiter:
    """Token bucket rate limiter for API requests."""
    
    def __init__(
        self,
        requests_per_minute: float = 60.0,
        tokens_per_minute: float = 100000.0,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        
        self._request_tokens = 1.0
        self._token_tokens = tokens_per_minute
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, estimated_tokens: int = 1000) -> None:
        """Acquire permission to make a request."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._last_update = now
            
            # Replenish tokens
            self._request_tokens = min(
                self.requests_per_minute,
                self._request_tokens + elapsed * (self.requests_per_minute / 60.0)
            )
            self._token_tokens = min(
                self.tokens_per_minute,
                self._token_tokens + elapsed * (self.tokens_per_minute / 60.0)
            )
            
            # Check if we have enough tokens
            while self._request_tokens < 1 or self._token_tokens < estimated_tokens:
                wait_time = max(
                    (1 - self._request_tokens) / (self.requests_per_minute / 60.0),
                    (estimated_tokens - self._token_tokens) / (self.tokens_per_minute / 60.0)
                )
                await asyncio.sleep(max(0.1, wait_time))
                
                now = time.monotonic()
                elapsed = now - self._last_update
                self._last_update = now
                
                self._request_tokens = min(
                    self.requests_per_minute,
                    self._request_tokens + elapsed * (self.requests_per_minute / 60.0)
                )
                self._token_tokens = min(
                    self.tokens_per_minute,
                    self._token_tokens + elapsed * (self.tokens_per_minute / 60.0)
                )
            
            self._request_tokens -= 1
            self._token_tokens -= estimated_tokens


# =============================================================================
# Retry Logic
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retryable_status_codes: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])


class ExponentialBackoffRetry:
    """Exponential backoff retry handler."""
    
    def __init__(self, config: RetryConfig) -> None:
        self.config = config
    
    async def execute(
        self,
        operation: Callable[..., Any],
        operation_name: str = "operation",
    ) -> Any:
        """Execute an operation with retry logic."""
        last_exception: Optional[Exception] = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await operation()
            except Exception as e:
                last_exception = e
                
                # Check if we should retry
                should_retry = False
                delay = self.config.base_delay * (self.config.exponential_base ** attempt)
                delay = min(delay, self.config.max_delay)
                
                if isinstance(e, KimiRateLimitError):
                    should_retry = True
                    if e.retry_after:
                        delay = e.retry_after
                elif isinstance(e, KimiError) and e.status_code in self.config.retryable_status_codes:
                    should_retry = True
                elif isinstance(e, (asyncio.TimeoutError, KimiTimeoutError)):
                    should_retry = True
                
                if not should_retry or attempt >= self.config.max_retries:
                    raise
                
                # Add jitter to avoid thundering herd
                jitter = (asyncio.get_event_loop().time() % 1) * 0.1 * delay
                await asyncio.sleep(delay + jitter)
        
        if last_exception:
            raise last_exception
        
        raise RuntimeError("Unexpected state in retry logic")


# =============================================================================
# Tool Executor
# =============================================================================


class ToolExecutor:
    """Helper class for executing tool calls."""
    
    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}
    
    def register(self, name: str, func: Callable[..., Any]) -> None:
        """Register a tool function."""
        self._tools[name] = func
    
    def register_many(self, tools: Dict[str, Callable[..., Any]]) -> None:
        """Register multiple tool functions."""
        self._tools.update(tools)
    
    async def execute(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call."""
        function = tool_call.get("function", {})
        func_name = function.get("name")
        func = self._tools.get(func_name)
        
        if not func:
            return {"error": f"Unknown function: {func_name}"}
        
        try:
            arguments = json.loads(function.get("arguments", "{}"))
            
            if asyncio.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)
            
            return {"result": result}
        except Exception as e:
            return {"error": f"Error executing {func_name}: {str(e)}"}
    
    async def execute_all(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple tool calls concurrently."""
        return await asyncio.gather(*[self.execute(tc) for tc in tool_calls])


# =============================================================================
# Main Provider
# =============================================================================


class KimiProvider:
    """
    Complete async Kimi API provider implementation.
    
    Features:
    - Full async/await support
    - Tool calling with function definitions
    - Streaming responses
    - Comprehensive error handling
    - Rate limit management with token bucket
    - Exponential backoff retries
    - Complete type hints
    
    Example:
        >>> provider = KimiProvider(api_key="your-api-key")
        >>> await provider.initialize()
        >>> 
        >>> # Simple completion
        >>> response = await provider.complete(CompletionRequest(
        ...     messages=[Message.user("Hello!")],
        ... ))
        >>> print(response.content)
        
        >>> # With tools
        >>> response = await provider.complete(CompletionRequest(
        ...     messages=[Message.user("What's the weather?")],
        ...     tools=[Tool(
        ...         name="get_weather",
        ...         description="Get weather for a location",
        ...         parameters={
        ...             "type": "object",
        ...             "properties": {
        ...                 "location": {"type": "string"},
        ...             },
        ...             "required": ["location"],
        ...         },
        ...     )],
        ... ))
        
        >>> # Streaming
        >>> async for chunk in provider.complete_stream(request):
        ...     print(chunk.content, end="", flush=True)
    """
    
    # Model constants
    DEFAULT_MODEL = "kimi-k2-5"
    
    def __init__(self, config: Optional[KimiConfig] = None) -> None:
        """
        Initialize the Kimi provider.
        
        Args:
            config: Provider configuration. If None, uses default config
                   with API key from KIMI_API_KEY environment variable.
        """
        self.config = config or KimiConfig()
        
        # Get API key from config or environment
        self.api_key = self.config.api_key or os.environ.get("KIMI_API_KEY")
        if not self.api_key:
            raise KimiAuthenticationError(
                "API key is required. Set KIMI_API_KEY environment variable or pass in config."
            )
        
        # Initialize components
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter: Optional[RateLimiter] = None
        self._retry_handler: Optional[ExponentialBackoffRetry] = None
        self._tool_executor = ToolExecutor()
        
        # Rate limit tracking
        self._last_rate_limit_info: Optional[RateLimitInfo] = None
    
    async def initialize(self) -> None:
        """Initialize the provider."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        
        if self.config.enable_rate_limiting and self._rate_limiter is None:
            self._rate_limiter = RateLimiter(
                requests_per_minute=self.config.requests_per_minute,
                tokens_per_minute=self.config.tokens_per_minute,
            )
        
        if self._retry_handler is None:
            self._retry_handler = ExponentialBackoffRetry(
                RetryConfig(
                    max_retries=self.config.max_retries,
                    base_delay=self.config.min_retry_delay,
                    max_delay=self.config.max_retry_delay,
                    exponential_base=self.config.exponential_base,
                )
            )
    
    async def close(self) -> None:
        """Close the provider and release resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def __aenter__(self) -> KimiProvider:
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers
    
    def _parse_rate_limit_headers(self, headers: Dict[str, str]) -> RateLimitInfo:
        """Parse rate limit information from response headers."""
        def _int_or_none(value: Optional[str]) -> Optional[int]:
            if value is None:
                return None
            try:
                return int(value)
            except (ValueError, TypeError):
                return None
        
        retry_after = None
        if "retry-after" in headers:
            try:
                retry_after = float(headers["retry-after"])
            except (ValueError, TypeError):
                pass
        
        return RateLimitInfo(
            limit_requests=_int_or_none(headers.get("x-ratelimit-limit-requests")),
            remaining_requests=_int_or_none(headers.get("x-ratelimit-remaining-requests")),
            reset_requests=headers.get("x-ratelimit-reset-requests"),
            limit_tokens=_int_or_none(headers.get("x-ratelimit-limit-tokens")),
            remaining_tokens=_int_or_none(headers.get("x-ratelimit-remaining-tokens")),
            reset_tokens=headers.get("x-ratelimit-reset-tokens"),
            retry_after=retry_after,
        )
    
    def _raise_for_status(
        self,
        status_code: int,
        response_body: Optional[Dict[str, Any]],
        headers: Dict[str, str],
    ) -> None:
        """Raise appropriate exception for HTTP status code."""
        if status_code < 400:
            return
        
        error_message = "Unknown error"
        if response_body:
            if "error" in response_body:
                error = response_body["error"]
                if isinstance(error, dict):
                    error_message = error.get("message", str(error))
                else:
                    error_message = str(error)
            elif "message" in response_body:
                error_message = response_body["message"]
        
        retry_after = None
        if "retry-after" in headers:
            try:
                retry_after = float(headers["retry-after"])
            except (ValueError, TypeError):
                pass
        
        if status_code == 400:
            raise KimiValidationError(error_message, status_code, response_body)
        elif status_code == 401:
            raise KimiAuthenticationError(error_message, status_code, response_body)
        elif status_code == 429:
            raise KimiRateLimitError(error_message, retry_after, status_code, response_body)
        elif status_code >= 500:
            raise KimiServerError(error_message, status_code, response_body)
        else:
            raise KimiAPIError(error_message, status_code, response_body)
    
    async def _make_request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """Make an HTTP request with error handling."""
        if self._session is None or self._session.closed:
            raise KimiError("Provider not initialized. Call initialize() first.")
        
        url = f"{self.config.base_url.rstrip('/')}{path}"
        headers = self._get_headers()
        
        try:
            async with self._session.request(
                method=method,
                url=url,
                headers=headers,
                json=json_data,
            ) as response:
                response_headers = dict(response.headers)
                
                if stream:
                    return self._handle_streaming_response(response, response_headers)
                
                body = await response.json()
                self._raise_for_status(response.status, body, response_headers)
                
                # Store rate limit info
                self._last_rate_limit_info = self._parse_rate_limit_headers(response_headers)
                
                return body
                
        except aiohttp.ClientConnectorError as e:
            raise KimiConnectionError(f"Connection error: {e}")
        except asyncio.TimeoutError:
            raise KimiTimeoutError(f"Request timeout after {self.config.timeout}s")
    
    async def _handle_streaming_response(
        self,
        response: aiohttp.ClientResponse,
        headers: Dict[str, str],
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle streaming response."""
        self._last_rate_limit_info = self._parse_rate_limit_headers(headers)
        
        async for line in response.content:
            line = line.decode("utf-8").strip()
            
            if not line or line.startswith(":"):
                continue
            
            if line.startswith("data: "):
                data = line[6:]
                
                if data == "[DONE]":
                    break
                
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue
    
    async def _request_with_retry(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """Make a request with retry logic."""
        if self._retry_handler is None:
            raise KimiError("Provider not initialized. Call initialize() first.")
        
        async def _do_request() -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
            # Apply rate limiting
            if self._rate_limiter:
                await self._rate_limiter.acquire()
            
            return await self._make_request(method, path, json_data, stream)
        
        return await self._retry_handler.execute(_do_request, "API request")
    
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion response.
        
        Args:
            request: The completion request.
        
        Returns:
            CompletionResponse with the generated content.
        """
        # Build request body
        body: Dict[str, Any] = {
            "model": request.model or self.config.default_model,
            "messages": [m.to_dict() for m in request.messages],
            "temperature": request.temperature,
            "stream": False,
        }
        
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            body["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            body["presence_penalty"] = request.presence_penalty
        if request.stop is not None:
            body["stop"] = request.stop
        if request.tools is not None:
            body["tools"] = [t.to_kimi_format() for t in request.tools]
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
        if request.response_format is not None:
            body["response_format"] = request.response_format
        
        body.update(request.extra_params)
        
        response_data = await self._request_with_retry(
            method="POST",
            path="/chat/completions",
            json_data=body,
        )
        
        # Handle streaming case (shouldn't happen here)
        if isinstance(response_data, AsyncIterator):
            raise KimiError("Unexpected streaming response")
        
        # Parse response
        choice = response_data["choices"][0]
        message = choice["message"]
        
        usage_data = response_data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            cached_tokens=usage_data.get("cached_tokens"),
        )
        
        return CompletionResponse(
            content=message.get("content", ""),
            model=response_data.get("model", ""),
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            tool_calls=message.get("tool_calls"),
            raw_response=response_data,
            id=response_data.get("id"),
        )
    
    async def complete_stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """
        Generate a streaming completion response.
        
        Args:
            request: The completion request.
        
        Yields:
            StreamChunk objects with partial content.
        """
        # Build request body
        body: Dict[str, Any] = {
            "model": request.model or self.config.default_model,
            "messages": [m.to_dict() for m in request.messages],
            "temperature": request.temperature,
            "stream": True,
        }
        
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            body["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            body["presence_penalty"] = request.presence_penalty
        if request.stop is not None:
            body["stop"] = request.stop
        if request.tools is not None:
            body["tools"] = [t.to_kimi_format() for t in request.tools]
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
        
        body.update(request.extra_params)
        
        response_stream = await self._request_with_retry(
            method="POST",
            path="/chat/completions",
            json_data=body,
            stream=True,
        )
        
        # Handle non-streaming case (shouldn't happen here)
        if not isinstance(response_stream, AsyncIterator):
            raise KimiError("Expected streaming response")
        
        async for chunk_data in response_stream:
            choice = chunk_data["choices"][0] if chunk_data.get("choices") else None
            if not choice:
                continue
            
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")
            
            yield StreamChunk(
                content=delta.get("content", "") or "",
                is_finished=finish_reason is not None,
                finish_reason=finish_reason,
                tool_calls=delta.get("tool_calls"),
            )
    
    def register_tool(self, name: str, func: Callable[..., Any]) -> None:
        """
        Register a tool for function calling.
        
        Args:
            name: Tool name.
            func: Function to execute.
        """
        self._tool_executor.register(name, func)
    
    async def execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute tool calls and return results.
        
        Args:
            tool_calls: List of tool call dictionaries.
        
        Returns:
            List of tool results.
        """
        results = await self._tool_executor.execute_all(tool_calls)
        
        return [
            {
                "tool_call_id": tc.get("id"),
                "role": "tool",
                "content": json.dumps(r),
            }
            for tc, r in zip(tool_calls, results)
        ]
    
    def get_tool_definition(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
    ) -> Tool:
        """
        Get a tool definition for function calling.
        
        Args:
            name: Function name.
            description: Function description.
            parameters: JSON Schema for parameters.
        
        Returns:
            Tool definition.
        """
        return Tool(
            name=name,
            description=description,
            parameters=parameters,
        )
    
    def get_last_rate_limit_info(self) -> Optional[RateLimitInfo]:
        """Get the rate limit info from the last request."""
        return self._last_rate_limit_info
    
    @property
    def is_initialized(self) -> bool:
        """Check if the provider is initialized."""
        return self._session is not None and not self._session.closed


# =============================================================================
# Low-level Client (for direct API access)
# =============================================================================


class KimiClient:
    """
    Low-level async client for Moonshot AI's Kimi API.
    
    This client provides direct access to the Kimi API with minimal abstraction.
    For most use cases, use KimiProvider instead.
    
    Example:
        >>> async with KimiClient(api_key="your-api-key") as client:
        ...     response = await client.chat.completions.create(
        ...         model="kimi-k2-5",
        ...         messages=[{"role": "user", "content": "Hello!"}],
        ...     )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.moonshot.cn/v1",
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the Kimi client.
        
        Args:
            api_key: API key. If not provided, uses KIMI_API_KEY env var.
            base_url: API base URL.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries.
        """
        self.api_key = api_key or os.environ.get("KIMI_API_KEY")
        if not self.api_key:
            raise KimiAuthenticationError("API key is required")
        
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        
        self._session: Optional[aiohttp.ClientSession] = None
        self.chat = ChatClient(self)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def __aenter__(self) -> KimiClient:
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """Make an HTTP request."""
        session = await self._get_session()
        
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        last_exception: Optional[Exception] = None
        retry_delay = 1.0
        
        for attempt in range(self.max_retries + 1):
            try:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_data,
                ) as response:
                    if stream:
                        return self._handle_stream(response)
                    
                    body = await response.json()
                    
                    if response.status >= 400:
                        error_msg = "Unknown error"
                        if body and "error" in body:
                            error = body["error"]
                            error_msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                        
                        if response.status == 401:
                            raise KimiAuthenticationError(error_msg)
                        elif response.status == 429:
                            raise KimiRateLimitError(error_msg)
                        elif response.status >= 500:
                            raise KimiServerError(error_msg, response.status)
                        else:
                            raise KimiAPIError(error_msg, response.status)
                    
                    return body
                    
            except (KimiAuthenticationError, KimiValidationError):
                raise
            except KimiRateLimitError as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                raise
            except (KimiServerError, KimiConnectionError, KimiTimeoutError) as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                raise
            except Exception as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                raise KimiError(f"Request failed: {e}")
        
        if last_exception:
            raise last_exception
        
        raise KimiError("All retries exhausted")
    
    async def _handle_stream(
        self,
        response: aiohttp.ClientResponse,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle streaming response."""
        async for line in response.content:
            line = line.decode("utf-8").strip()
            
            if not line or line.startswith(":"):
                continue
            
            if line.startswith("data: "):
                data = line[6:]
                
                if data == "[DONE]":
                    break
                
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue


class ChatClient:
    """Client for chat completion endpoints."""
    
    def __init__(self, client: KimiClient) -> None:
        self._client = client
        self.completions = ChatCompletionsClient(client)


class ChatCompletionsClient:
    """Client for chat completions endpoint."""
    
    def __init__(self, client: KimiClient) -> None:
        self._client = client
    
    async def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Create a chat completion.
        
        Args:
            model: Model ID to use.
            messages: List of messages.
            tools: List of tools.
            tool_choice: Tool choice mode.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_tokens: Maximum tokens to generate.
            stop: Stop sequences.
            presence_penalty: Presence penalty.
            frequency_penalty: Frequency penalty.
            stream: Whether to stream the response.
            **kwargs: Additional parameters.
        
        Returns:
            Response dict for non-streaming, async iterator for streaming.
        """
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if stop is not None:
            body["stop"] = stop
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        
        body.update(kwargs)
        
        return await self._client._request(
            method="POST",
            path="/chat/completions",
            json_data=body,
            stream=stream,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


async def create_completion(
    messages: List[Union[Message, Dict[str, Any]]],
    api_key: Optional[str] = None,
    model: str = "kimi-k2-5",
    **kwargs: Any,
) -> CompletionResponse:
    """
    Convenience function to create a chat completion.
    
    Args:
        messages: Messages for the conversation.
        api_key: API key (or set KIMI_API_KEY env var).
        model: Model to use.
        **kwargs: Additional arguments.
    
    Returns:
        CompletionResponse.
    """
    config = KimiConfig(api_key=api_key)
    provider = KimiProvider(config)
    
    async with provider:
        parsed_messages = [
            m if isinstance(m, Message) else Message(
                role=Role(m.get("role", "user")),
                content=m.get("content", ""),
            )
            for m in messages
        ]
        
        request = CompletionRequest(messages=parsed_messages, model=model, **kwargs)
        return await provider.complete(request)


async def create_streaming_completion(
    messages: List[Union[Message, Dict[str, Any]]],
    api_key: Optional[str] = None,
    model: str = "kimi-k2-5",
    **kwargs: Any,
) -> AsyncIterator[StreamChunk]:
    """
    Convenience function to create a streaming chat completion.
    
    Args:
        messages: Messages for the conversation.
        api_key: API key (or set KIMI_API_KEY env var).
        model: Model to use.
        **kwargs: Additional arguments.
    
    Yields:
        StreamChunk objects.
    """
    config = KimiConfig(api_key=api_key)
    provider = KimiProvider(config)
    
    async with provider:
        parsed_messages = [
            m if isinstance(m, Message) else Message(
                role=Role(m.get("role", "user")),
                content=m.get("content", ""),
            )
            for m in messages
        ]
        
        request = CompletionRequest(messages=parsed_messages, model=model, stream=True, **kwargs)
        async for chunk in provider.complete_stream(request):
            yield chunk


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Exceptions
    "KimiError",
    "KimiAPIError",
    "KimiRateLimitError",
    "KimiAuthenticationError",
    "KimiTimeoutError",
    "KimiValidationError",
    "KimiServerError",
    "KimiConnectionError",
    
    # Data Types
    "Role",
    "Message",
    "Tool",
    "TokenUsage",
    "CompletionRequest",
    "CompletionResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "StreamChunk",
    "RateLimitInfo",
    
    # Models
    "KimiModel",
    
    # Configuration
    "KimiConfig",
    
    # Components
    "RateLimiter",
    "RetryConfig",
    "ExponentialBackoffRetry",
    "ToolExecutor",
    
    # Providers
    "KimiProvider",
    "KimiClient",
    "ChatClient",
    "ChatCompletionsClient",
    
    # Convenience Functions
    "create_completion",
    "create_streaming_completion",
]
