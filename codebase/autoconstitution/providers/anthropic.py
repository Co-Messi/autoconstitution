"""
Anthropic Claude Provider for autoconstitution

A comprehensive async client for Anthropic's Claude API with support for:
- Chat completions with Claude 3.5, Claude 3 models
- Tool/function calling
- Streaming responses
- Rate limit management
- Comprehensive error handling
- Full type hints

Usage:
    from autoconstitution.providers.anthropic import AnthropicProvider, AnthropicConfig
    
    config = AnthropicConfig(api_key="your-api-key")
    provider = AnthropicProvider(config)
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
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

# =============================================================================
# Exceptions
# =============================================================================


class AnthropicError(Exception):
    """Base exception for Anthropic provider errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[Dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body or {}


class AnthropicAPIError(AnthropicError):
    """Raised when the Anthropic API returns an error."""
    pass


class AnthropicRateLimitError(AnthropicError):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        status_code: Optional[int] = 429,
        response_body: Optional[Dict] = None,
    ) -> None:
        super().__init__(message, status_code, response_body)
        self.retry_after = retry_after


class AnthropicAuthenticationError(AnthropicError):
    """Raised when authentication fails."""
    pass


class AnthropicTimeoutError(AnthropicError):
    """Raised when a request times out."""
    pass


class AnthropicValidationError(AnthropicError):
    """Raised when request validation fails."""
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
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class TokenUsage:
    """Token usage statistics."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    @property
    def cost_estimate(self) -> float:
        """Rough cost estimate (varies by provider and model)."""
        # Claude 3.5 Sonnet: ~$3/M input, $15/M output tokens
        input_cost = self.prompt_tokens * 0.000003
        output_cost = self.completion_tokens * 0.000015
        return input_cost + output_cost


@dataclass
class CompletionRequest:
    """Request parameters for completion/chat completion."""
    
    messages: List[Message]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    stream: bool = False
    system: Optional[str] = None  # Anthropic-specific: separate system prompt
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
    stop_sequence: Optional[str] = None


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
    
    limit: int
    remaining: int
    reset_timestamp: float
    retry_after: Optional[float] = None
    
    @property
    def is_exceeded(self) -> bool:
        """Check if rate limit is exceeded."""
        return self.remaining <= 0
    
    @property
    def reset_in(self) -> float:
        """Seconds until rate limit resets."""
        return max(0.0, self.reset_timestamp - time.time())


@dataclass
class AnthropicConfig:
    """Configuration for Anthropic provider."""
    
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 120.0
    max_retries: int = 3
    default_model: str = "claude-3-5-sonnet-20241022"
    extra_headers: Optional[Dict[str, str]] = None
    
    # Rate limiting
    enable_rate_limiting: bool = True
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
        requests_per_minute: float = 50.0,
        tokens_per_minute: float = 40000.0,
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
# Retry Handler
# =============================================================================


T = TypeVar("T")


class RetryHandler:
    """Handles retries with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = 3,
        min_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ) -> None:
        self.max_retries = max_retries
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    async def execute(
        self,
        operation: Callable[[], Any],
        is_retryable: Optional[Callable[[Exception], bool]] = None,
    ) -> Any:
        """Execute an operation with retry logic."""
        last_exception: Optional[Exception] = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await operation()
            except Exception as e:
                last_exception = e
                
                if attempt >= self.max_retries:
                    break
                
                if is_retryable and not is_retryable(e):
                    raise
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.max_delay,
                    self.min_delay * (self.exponential_base ** attempt)
                )
                
                # Add jitter
                delay *= 0.5 + (asyncio.get_event_loop().time() % 1.0)
                
                await asyncio.sleep(delay)
        
        raise last_exception or RuntimeError("Operation failed after retries")
    
    @staticmethod
    def is_retryable_error(error: Exception) -> bool:
        """Check if an error is retryable."""
        if isinstance(error, AnthropicRateLimitError):
            return True
        if isinstance(error, AnthropicAPIError):
            # Retry on server errors (5xx)
            if error.status_code and error.status_code >= 500:
                return True
        if isinstance(error, AnthropicTimeoutError):
            return True
        return False


# =============================================================================
# Anthropic Provider
# =============================================================================


class AnthropicProvider:
    """
    Async client for Anthropic's Claude API.
    
    Features:
    - Full async/await support
    - Tool calling with function definitions
    - Streaming responses
    - Automatic retries with exponential backoff
    - Rate limit management
    - Comprehensive error handling
    
    Example:
        provider = AnthropicProvider(AnthropicConfig(api_key="..."))
        await provider.initialize()
        
        response = await provider.complete(
            CompletionRequest(
                messages=[Message.user("Hello, Claude!")],
                model="claude-3-5-sonnet-20241022"
            )
        )
        print(response.content)
    """
    
    # Model constants
    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    
    AVAILABLE_MODELS = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]
    
    # Context window sizes
    MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-haiku-20241022": 200000,
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
    }
    
    # Max output tokens
    MODEL_MAX_OUTPUT: Dict[str, int] = {
        "claude-3-5-sonnet-20241022": 8192,
        "claude-3-5-haiku-20241022": 4096,
        "claude-3-opus-20240229": 4096,
        "claude-3-sonnet-20240229": 4096,
        "claude-3-haiku-20240307": 4096,
    }
    
    def __init__(self, config: Optional[AnthropicConfig] = None, **kwargs: Any) -> None:
        """
        Initialize the Anthropic provider.
        
        Args:
            config: Configuration object
            **kwargs: Override config values
        """
        self.config = config or AnthropicConfig()
        
        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(self.config, key) and value is not None:
                setattr(self.config, key, value)
        
        # Get API key from environment if not provided
        self._api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        self._base_url = self.config.base_url or os.getenv(
            "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
        )
        
        # Client state
        self._client: Optional[Any] = None
        self._initialized = False
        
        # Rate limiting
        self._rate_limiter = RateLimiter() if self.config.enable_rate_limiting else None
        self._last_rate_limit_info: Optional[RateLimitInfo] = None
        
        # Retry handler
        self._retry_handler = RetryHandler(
            max_retries=self.config.max_retries,
            min_delay=self.config.min_retry_delay,
            max_delay=self.config.max_retry_delay,
            exponential_base=self.config.exponential_base,
        )
    
    @property
    def is_initialized(self) -> bool:
        """Check if the provider is initialized."""
        return self._initialized
    
    @property
    def default_model(self) -> str:
        """Get the default model."""
        return self.config.default_model
    
    @property
    def available_models(self) -> List[str]:
        """Get list of available models."""
        return self.AVAILABLE_MODELS.copy()
    
    def get_model_context_window(self, model: Optional[str] = None) -> int:
        """Get the context window size for a model."""
        model = model or self.default_model
        return self.MODEL_CONTEXT_WINDOWS.get(model, 200000)
    
    def get_model_max_output(self, model: Optional[str] = None) -> int:
        """Get the max output tokens for a model."""
        model = model or self.default_model
        return self.MODEL_MAX_OUTPUT.get(model, 4096)
    
    async def initialize(self) -> None:
        """
        Initialize the Anthropic client.
        
        Raises:
            ImportError: If anthropic package is not installed
            AnthropicAuthenticationError: If API key is missing
        """
        if self._initialized:
            return
        
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "Anthropic provider requires 'anthropic' package. "
                "Install with: pip install anthropic"
            ) from e
        
        if not self._api_key:
            raise AnthropicAuthenticationError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key to config."
            )
        
        # Build client kwargs
        client_kwargs: Dict[str, Any] = {"api_key": self._api_key}
        
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        if self.config.timeout:
            client_kwargs["timeout"] = self.config.timeout
        if self.config.max_retries:
            client_kwargs["max_retries"] = 0  # We handle retries ourselves
        
        # Create client
        self._client = anthropic.AsyncAnthropic(**client_kwargs)
        self._initialized = True
    
    async def close(self) -> None:
        """Close the client and release resources."""
        if self._client and hasattr(self._client, 'close'):
            await self._client.close()
        self._initialized = False
    
    async def __aenter__(self) -> AnthropicProvider:
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
    
    def _ensure_initialized(self) -> None:
        """Ensure the provider is initialized."""
        if not self._initialized:
            raise RuntimeError(
                "Anthropic provider not initialized. Call initialize() first."
            )
    
    def _get_model(self, request: CompletionRequest) -> str:
        """Get the model to use for a request."""
        return request.model or self.config.default_model
    
    def _handle_error(self, error: Exception) -> None:
        """Convert anthropic errors to provider errors."""
        import anthropic
        
        if isinstance(error, anthropic.AuthenticationError):
            raise AnthropicAuthenticationError(
                f"Authentication failed: {error}",
                status_code=401,
            ) from error
        
        if isinstance(error, anthropic.RateLimitError):
            retry_after = None
            if hasattr(error, 'response') and error.response:
                headers = getattr(error.response, 'headers', {})
                retry_after_str = headers.get('retry-after')
                if retry_after_str:
                    retry_after = float(retry_after_str)
            
            raise AnthropicRateLimitError(
                f"Rate limit exceeded: {error}",
                retry_after=retry_after,
                status_code=429,
            ) from error
        
        if isinstance(error, anthropic.BadRequestError):
            raise AnthropicValidationError(
                f"Bad request: {error}",
                status_code=400,
            ) from error
        
        if isinstance(error, anthropic.APIStatusError):
            raise AnthropicAPIError(
                f"API error: {error}",
                status_code=getattr(error, 'status_code', None),
            ) from error
        
        if isinstance(error, anthropic.APITimeoutError):
            raise AnthropicTimeoutError(f"Request timed out: {error}") from error
        
        if isinstance(error, anthropic.APIConnectionError):
            raise AnthropicError(f"Connection error: {error}") from error
        
        # Re-raise unknown errors
        raise AnthropicError(f"Unexpected error: {error}") from error
    
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Execute a non-streaming completion request.
        
        Args:
            request: The completion request
            
        Returns:
            CompletionResponse with the generated content
            
        Raises:
            AnthropicError: On API errors
        """
        self._ensure_initialized()
        
        if self._rate_limiter:
            await self._rate_limiter.acquire(
                estimated_tokens=request.max_tokens or 1000
            )
        
        async def _do_complete() -> CompletionResponse:
            try:
                system_msg, messages = self._convert_messages(request.messages)
                params = self._build_params(request, system_msg, messages)
                
                response = await self._client.messages.create(**params)
                
                return self._parse_response(response)
                
            except Exception as e:
                self._handle_error(e)
                raise  # Unreachable, but satisfies type checker
        
        return await self._retry_handler.execute(
            _do_complete,
            is_retryable=RetryHandler.is_retryable_error,
        )
    
    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionResponse]:
        """
        Execute a streaming completion request.
        
        Args:
            request: The completion request
            
        Yields:
            CompletionResponse chunks as they arrive
            
        Raises:
            AnthropicError: On API errors
        """
        self._ensure_initialized()
        
        if self._rate_limiter:
            await self._rate_limiter.acquire(
                estimated_tokens=request.max_tokens or 1000
            )
        
        system_msg, messages = self._convert_messages(request.messages)
        params = self._build_params(request, system_msg, messages)
        params["stream"] = True
        
        try:
            stream = await self._client.messages.create(**params)
            
            current_content = ""
            current_tool_calls: List[Dict[str, Any]] = []
            current_tool_call: Optional[Dict[str, Any]] = None
            
            async for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool_call = {
                            "id": event.content_block.id,
                            "type": "function",
                            "function": {
                                "name": event.content_block.name,
                                "arguments": "",
                            },
                        }
                
                elif event.type == "content_block_delta":
                    if hasattr(event.delta, "text") and event.delta.text:
                        current_content += event.delta.text
                        yield CompletionResponse(
                            content=event.delta.text,
                            model=params["model"],
                            usage=TokenUsage(),
                            raw_response={"type": "text_delta", "text": event.delta.text},
                        )
                    
                    elif hasattr(event.delta, "partial_json"):
                        if current_tool_call:
                            current_tool_call["function"]["arguments"] += event.delta.partial_json
                
                elif event.type == "content_block_stop":
                    if current_tool_call:
                        current_tool_calls.append(current_tool_call)
                        current_tool_call = None
                
                elif event.type == "message_stop":
                    # Final chunk with complete info
                    usage = TokenUsage()
                    if hasattr(event, 'message') and event.message:
                        if hasattr(event.message, 'usage'):
                            usage = TokenUsage(
                                prompt_tokens=event.message.usage.input_tokens,
                                completion_tokens=event.message.usage.output_tokens,
                                total_tokens=(
                                    event.message.usage.input_tokens +
                                    event.message.usage.output_tokens
                                ),
                            )
                    
                    yield CompletionResponse(
                        content="",
                        model=params["model"],
                        usage=usage,
                        finish_reason="stop",
                        tool_calls=current_tool_calls if current_tool_calls else None,
                        raw_response={"type": "message_stop"},
                        is_finished=True,  # type: ignore
                    )
        
        except Exception as e:
            self._handle_error(e)
    
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings for texts.
        
        Note: Anthropic does not currently offer embeddings API.
        This method raises NotImplementedError.
        
        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError(
            "Anthropic provider does not support embeddings. "
            "Use OpenAI, Cohere, or another provider for embeddings."
        )
    
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Try a simple completion
            test_request = CompletionRequest(
                messages=[Message.user("Hi")],
                max_tokens=5,
            )
            await self.complete(test_request)
            return True
        except Exception:
            return False
    
    def _convert_messages(
        self, messages: List[Message]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert messages to Anthropic format.
        
        Anthropic uses a separate system parameter and different message format.
        
        Args:
            messages: List of messages
            
        Returns:
            Tuple of (system_message, anthropic_messages)
        """
        system_msg: Optional[str] = None
        anthropic_messages: List[Dict[str, Any]] = []
        
        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Anthropic supports only one system message
                system_msg = msg.content
            
            elif msg.role == Role.USER:
                anthropic_messages.append({
                    "role": "user",
                    "content": msg.content,
                })
            
            elif msg.role == Role.ASSISTANT:
                content = msg.content
                # Handle assistant messages with tool calls
                if msg.tool_calls:
                    content_blocks: List[Dict[str, Any]] = []
                    if content:
                        content_blocks.append({"type": "text", "text": content})
                    for tc in msg.tool_calls:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "input": json.loads(tc.get("function", {}).get("arguments", "{}")),
                        })
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content_blocks,
                    })
                else:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content,
                    })
            
            elif msg.role == Role.TOOL:
                # Tool results are sent as user messages with tool_result blocks
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id or "",
                        "content": msg.content,
                    }],
                })
        
        return system_msg, anthropic_messages
    
    def _build_params(
        self,
        request: CompletionRequest,
        system_msg: Optional[str],
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build Anthropic API parameters from request.
        
        Args:
            request: The completion request
            system_msg: System message (if any)
            messages: Converted messages
            
        Returns:
            Dictionary of API parameters
        """
        model = self._get_model(request)
        max_tokens = request.max_tokens or self.get_model_max_output(model)
        
        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        
        # Add system message if present
        if system_msg:
            params["system"] = system_msg
        
        # Add optional parameters
        if request.temperature is not None:
            params["temperature"] = request.temperature
        
        if request.top_p is not None:
            params["top_p"] = request.top_p
        
        if request.top_k is not None:
            params["top_k"] = request.top_k
        
        if request.stop:
            params["stop_sequences"] = request.stop
        
        # Add tools if present
        if request.tools:
            params["tools"] = [t.to_anthropic_format() for t in request.tools]
        
        # Add tool choice if present
        if request.tool_choice:
            if request.tool_choice == "auto":
                params["tool_choice"] = {"type": "auto"}
            elif request.tool_choice == "any":
                params["tool_choice"] = {"type": "any"}
            elif request.tool_choice == "none":
                params["tool_choice"] = {"type": "none"}
            elif isinstance(request.tool_choice, dict):
                params["tool_choice"] = request.tool_choice
        
        # Add extra headers
        if self.config.extra_headers:
            params["extra_headers"] = self.config.extra_headers
        
        # Add any extra params
        params.update(request.extra_params)
        
        return params
    
    def _parse_response(self, response: Any) -> CompletionResponse:
        """
        Parse Anthropic API response to CompletionResponse.
        
        Args:
            response: Raw API response
            
        Returns:
            Parsed CompletionResponse
        """
        content = ""
        tool_calls: Optional[List[Dict[str, Any]]] = None
        
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })
        
        return CompletionResponse(
            content=content,
            model=response.model,
            usage=TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
            finish_reason=response.stop_reason,
            tool_calls=tool_calls,
            id=response.id,
            stop_sequence=response.stop_sequence,
            raw_response={
                "id": response.id,
                "type": response.type,
                "role": response.role,
                "content": [self._serialize_block(block) for block in response.content],
                "model": response.model,
                "stop_reason": response.stop_reason,
                "stop_sequence": response.stop_sequence,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            },
        )
    
    def _serialize_block(self, block: Any) -> Dict[str, Any]:
        """Serialize a content block to dictionary."""
        if hasattr(block, 'model_dump'):
            return block.model_dump()
        elif hasattr(block, '__dict__'):
            return {
                "type": getattr(block, 'type', 'unknown'),
                **block.__dict__,
            }
        else:
            return {"type": "unknown", "value": str(block)}


# =============================================================================
# Convenience Functions
# =============================================================================


async def create_anthropic_provider(
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> AnthropicProvider:
    """
    Create and initialize an Anthropic provider.
    
    Args:
        api_key: Anthropic API key (optional, uses env var if not provided)
        **kwargs: Additional config options
        
    Returns:
        Initialized AnthropicProvider
        
    Example:
        provider = await create_anthropic_provider()
        response = await provider.complete(
            CompletionRequest(messages=[Message.user("Hello!")])
        )
    """
    config = AnthropicConfig(api_key=api_key, **kwargs)
    provider = AnthropicProvider(config)
    await provider.initialize()
    return provider


async def anthropic_complete(
    messages: List[Message],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Tool]] = None,
    **kwargs: Any,
) -> CompletionResponse:
    """
    Simple convenience function for Anthropic completions.
    
    Args:
        messages: List of messages
        api_key: API key (optional)
        model: Model name (optional)
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        tools: Available tools
        **kwargs: Additional parameters
        
    Returns:
        CompletionResponse
        
    Example:
        response = await anthropic_complete(
            messages=[Message.user("What is the weather?")],
            model="claude-3-5-sonnet-20241022"
        )
        print(response.content)
    """
    async with AnthropicProvider(AnthropicConfig(api_key=api_key)) as provider:
        request = CompletionRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        )
        return await provider.complete(request)


async def anthropic_complete_text(
    prompt: str,
    system: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    Simple text completion with Anthropic.
    
    Args:
        prompt: User prompt
        system: Optional system message
        api_key: API key (optional)
        **kwargs: Additional parameters
        
    Returns:
        Generated text
        
    Example:
        text = await anthropic_complete_text(
            "What is the capital of France?",
            system="You are a helpful assistant."
        )
    """
    messages: List[Message] = []
    if system:
        messages.append(Message.system(system))
    messages.append(Message.user(prompt))
    
    response = await anthropic_complete(messages, api_key=api_key, **kwargs)
    return response.content


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Exceptions
    "AnthropicError",
    "AnthropicAPIError",
    "AnthropicRateLimitError",
    "AnthropicAuthenticationError",
    "AnthropicTimeoutError",
    "AnthropicValidationError",
    
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
    "AnthropicConfig",
    
    # Core Classes
    "RateLimiter",
    "RetryHandler",
    "AnthropicProvider",
    
    # Convenience Functions
    "create_anthropic_provider",
    "anthropic_complete",
    "anthropic_complete_text",
]
