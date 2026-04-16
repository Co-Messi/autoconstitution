"""
OpenAI Provider for autoconstitution

A complete async client for OpenAI's API with:
- Full async support
- Tool calling
- Streaming responses
- Error handling
- Rate limit management
- Complete type hints

Usage:
    from autoconstitution.providers.openai import OpenAIProvider
    
    provider = OpenAIProvider(api_key="your-api-key")
    await provider.initialize()
    
    response = await provider.complete(CompletionRequest(
        messages=[Message.user("Hello!")],
        model="gpt-4o",
    ))
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
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

try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    openai = None  # type: ignore

# =============================================================================
# Exceptions
# =============================================================================


class OpenAIError(Exception):
    """Base exception for OpenAI provider errors."""
    
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

    def __str__(self) -> str:
        if self.status_code:
            return f"[OpenAI Error {self.status_code}] {self.message}"
        return f"[OpenAI Error] {self.message}"


class OpenAIRateLimitError(OpenAIError):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class OpenAIAuthenticationError(OpenAIError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs: Any) -> None:
        super().__init__(message, status_code=401, **kwargs)


class OpenAIInvalidRequestError(OpenAIError):
    """Raised when the request is invalid."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, status_code=400, **kwargs)


class OpenAIServerError(OpenAIError):
    """Raised when OpenAI server returns an error."""
    
    def __init__(self, message: str = "Server error", status_code: int = 500, **kwargs: Any) -> None:
        super().__init__(message, status_code=status_code, **kwargs)


class OpenAITimeoutError(OpenAIError):
    """Raised when request times out."""
    
    def __init__(self, message: str = "Request timed out", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)


# =============================================================================
# Rate Limiter
# =============================================================================


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    requests_per_minute: int = 60
    tokens_per_minute: int = 150000
    burst_size: int = 10


class TokenBucketRateLimiter:
    """Token bucket rate limiter for API requests."""
    
    def __init__(self, config: RateLimitConfig) -> None:
        self.config = config
        self._request_tokens = float(config.burst_size)
        self._token_tokens = float(config.tokens_per_minute)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> None:
        """Acquire permission to make a request."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            
            # Replenish tokens
            self._request_tokens = min(
                self.config.burst_size,
                self._request_tokens + elapsed * (self.config.requests_per_minute / 60.0)
            )
            self._token_tokens = min(
                self.config.tokens_per_minute,
                self._token_tokens + elapsed * (self.config.tokens_per_minute / 60.0)
            )
            
            self._last_update = now
            
            # Check if we can proceed
            if self._request_tokens < 1:
                wait_time = (1 - self._request_tokens) / (self.config.requests_per_minute / 60.0)
                await asyncio.sleep(wait_time)
                self._request_tokens = 0
            else:
                self._request_tokens -= 1
            
            if tokens > 0 and self._token_tokens < tokens:
                wait_time = (tokens - self._token_tokens) / (self.config.tokens_per_minute / 60.0)
                await asyncio.sleep(wait_time)
                self._token_tokens = 0
            else:
                self._token_tokens -= tokens


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
        operation: Callable[[], Any],
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
                
                if isinstance(e, OpenAIRateLimitError):
                    should_retry = True
                    if e.retry_after:
                        delay = e.retry_after
                elif isinstance(e, OpenAIError) and e.status_code in self.config.retryable_status_codes:
                    should_retry = True
                elif isinstance(e, (asyncio.TimeoutError, OpenAITimeoutError)):
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
# Type Definitions (re-exported from main module)
# =============================================================================

from autoconstitution.providers import (
    ProviderType,
    Role,
    Message,
    Tool,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    TokenUsage,
    ProviderConfig,
    BaseProvider,
    register_provider,
)


# =============================================================================
# OpenAI Provider Implementation
# =============================================================================


@register_provider(ProviderType.OPENAI)
class OpenAIProvider(BaseProvider):
    """
    Complete async OpenAI API provider implementation.
    
    Features:
    - Full async/await support
    - Tool calling with function definitions
    - Streaming responses
    - Comprehensive error handling
    - Rate limit management with token bucket
    - Exponential backoff retries
    - Complete type hints
    
    Example:
        >>> provider = OpenAIProvider(api_key="sk-...")
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
    DEFAULT_MODEL = "gpt-4o"
    AVAILABLE_MODELS = [
        # GPT-4o models
        "gpt-4o",
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        # GPT-4 Turbo models
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-turbo-preview",
        # GPT-4 models
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-0314",
        "gpt-4-32k",
        # GPT-3.5 models
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-16k",
        # Embedding models
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
        # o1 models
        "o1-preview",
        "o1-mini",
    ]
    
    # Token limits per model
    MODEL_TOKEN_LIMITS: Dict[str, int] = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-16k": 16385,
        "o1-preview": 128000,
        "o1-mini": 128000,
    }
    
    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the OpenAI provider.
        
        Args:
            config: Provider configuration
            rate_limit_config: Rate limiting configuration
            retry_config: Retry behavior configuration
            **kwargs: Additional configuration overrides
        """
        super().__init__(config, **kwargs)
        
        # API credentials
        self._api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = self.config.base_url or os.getenv("OPENAI_BASE_URL")
        self._organization = self.config.organization or os.getenv("OPENAI_ORG_ID")
        
        # Rate limiting and retry
        self._rate_limiter = TokenBucketRateLimiter(
            rate_limit_config or RateLimitConfig()
        )
        self._retry_handler = ExponentialBackoffRetry(
            retry_config or RetryConfig(max_retries=self.config.max_retries)
        )
        
        # Client instance
        self._client: Optional[Any] = None
    
    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type identifier."""
        return ProviderType.OPENAI
    
    @property
    def default_model(self) -> str:
        """Return the default model for this provider."""
        return self.config.default_model or self.DEFAULT_MODEL
    
    @property
    def available_models(self) -> List[str]:
        """Return list of available models."""
        return self.AVAILABLE_MODELS.copy()
    
    def get_model_token_limit(self, model: Optional[str] = None) -> int:
        """Get the token limit for a model."""
        model = model or self.default_model
        return self.MODEL_TOKEN_LIMITS.get(model, 4096)

    def _get_model(self, request: CompletionRequest) -> str:
        """Get the model for a request."""
        return request.model or self.default_model
    
    async def initialize(self) -> None:
        """
        Initialize the OpenAI async client.
        
        Raises:
            ImportError: If openai package is not installed
            ValueError: If API key is not provided
        """
        if self._initialized:
            return
        
        if openai is None:
            raise ImportError(
                "OpenAI provider requires 'openai' package. "
                "Install with: pip install openai>=1.0.0"
            )
        
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key to config."
            )
        
        # Build client kwargs
        client_kwargs: Dict[str, Any] = {
            "api_key": self._api_key,
        }
        
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        if self.config.timeout:
            client_kwargs["timeout"] = self.config.timeout
        if self.config.max_retries:
            client_kwargs["max_retries"] = 0  # We handle retries ourselves
        if self._organization:
            client_kwargs["organization"] = self._organization
        if self.config.extra_headers:
            client_kwargs["default_headers"] = self.config.extra_headers
        
        self._client = openai.AsyncOpenAI(**client_kwargs)
        self._initialized = True
    
    def _handle_error(self, error: Exception) -> OpenAIError:
        """Convert OpenAI errors to our error types."""
        if openai is not None and isinstance(error, openai.RateLimitError):
            retry_after = None
            if hasattr(error, 'headers') and error.headers:
                retry_after_str = error.headers.get('retry-after')
                if retry_after_str:
                    retry_after = float(retry_after_str)
            return OpenAIRateLimitError(
                message=str(error),
                retry_after=retry_after,
                status_code=429,
            )
        elif openai is not None and isinstance(error, openai.AuthenticationError):
            return OpenAIAuthenticationError(str(error))
        elif openai is not None and isinstance(error, openai.BadRequestError):
            return OpenAIInvalidRequestError(str(error), status_code=400)
        elif openai is not None and isinstance(error, openai.InternalServerError):
            return OpenAIServerError(str(error), status_code=500)
        elif isinstance(error, asyncio.TimeoutError):
            return OpenAITimeoutError(str(error))
        elif isinstance(error, OpenAIError):
            return error
        else:
            return OpenAIError(str(error))
    
    async def _make_request(
        self,
        operation: Callable[[], Any],
        operation_name: str = "request",
        estimated_tokens: int = 1000,
    ) -> Any:
        """Make a request with rate limiting and retry logic."""
        await self._rate_limiter.acquire(tokens=estimated_tokens)
        
        async def wrapped_operation() -> Any:
            try:
                return await operation()
            except Exception as e:
                raise self._handle_error(e)
        
        return await self._retry_handler.execute(wrapped_operation, operation_name)
    
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Execute a non-streaming chat completion request.
        
        Args:
            request: The completion request
            
        Returns:
            CompletionResponse with the generated content
            
        Raises:
            OpenAIError: On API errors
            RuntimeError: If provider not initialized
        """
        self._ensure_initialized()
        
        params = self._build_params(request)
        model = self._get_model(request)
        
        # Estimate tokens for rate limiting
        estimated_tokens = sum(
            len(msg.content.split()) * 2 for msg in request.messages
        )
        estimated_tokens += request.max_tokens or 1000
        
        async def operation() -> CompletionResponse:
            response = await self._client.chat.completions.create(**params)
            
            choice = response.choices[0]
            message = choice.message
            
            # Extract tool calls if present
            tool_calls: Optional[List[Dict[str, Any]]] = None
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            
            return CompletionResponse(
                content=message.content or "",
                model=response.model,
                usage=TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                ),
                finish_reason=choice.finish_reason,
                tool_calls=tool_calls,
                raw_response=response.model_dump(),
            )
        
        return await self._make_request(
            operation,
            operation_name="chat.completion",
            estimated_tokens=estimated_tokens,
        )
    
    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionResponse]:
        """
        Execute a streaming chat completion request.
        
        Args:
            request: The completion request
            
        Yields:
            CompletionResponse chunks with partial content
            
        Raises:
            OpenAIError: On API errors
            RuntimeError: If provider not initialized
        """
        self._ensure_initialized()
        
        params = self._build_params(request)
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
        
        model = self._get_model(request)
        
        # Estimate tokens for rate limiting
        estimated_tokens = sum(
            len(msg.content.split()) * 2 for msg in request.messages
        )
        estimated_tokens += request.max_tokens or 1000
        
        await self._rate_limiter.acquire(tokens=estimated_tokens)
        
        try:
            stream = await self._client.chat.completions.create(**params)
            
            accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}
            
            async for chunk in stream:
                if not chunk.choices:
                    # Usage information at the end
                    if chunk.usage:
                        yield CompletionResponse(
                            content="",
                            model=chunk.model,
                            usage=TokenUsage(
                                prompt_tokens=chunk.usage.prompt_tokens,
                                completion_tokens=chunk.usage.completion_tokens,
                                total_tokens=chunk.usage.total_tokens,
                            ),
                            finish_reason="stop",
                            raw_response=chunk.model_dump(),
                        )
                    continue
                
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason
                
                # Handle tool call deltas
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": tc.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc.id:
                            accumulated_tool_calls[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                accumulated_tool_calls[idx]["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                accumulated_tool_calls[idx]["function"]["arguments"] += tc.function.arguments
                
                content = delta.content or ""
                
                # Build tool calls list if we have accumulated any
                tool_calls = None
                if accumulated_tool_calls and finish_reason:
                    tool_calls = list(accumulated_tool_calls.values())
                
                if content or tool_calls or finish_reason:
                    yield CompletionResponse(
                        content=content,
                        model=chunk.model,
                        usage=TokenUsage(),  # Per-chunk usage not available in streaming
                        finish_reason=finish_reason,
                        tool_calls=tool_calls if finish_reason else None,
                        raw_response=chunk.model_dump(),
                    )
        
        except Exception as e:
            raise self._handle_error(e)
    
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings for texts.
        
        Args:
            request: The embedding request with texts
            
        Returns:
            EmbeddingResponse with embeddings
            
        Raises:
            OpenAIError: On API errors
            RuntimeError: If provider not initialized
        """
        self._ensure_initialized()
        
        model = request.model or "text-embedding-3-small"
        
        # Estimate tokens for rate limiting
        estimated_tokens = sum(len(text.split()) for text in request.texts)
        
        async def operation() -> EmbeddingResponse:
            response = await self._client.embeddings.create(
                model=model,
                input=request.texts,
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return EmbeddingResponse(
                embeddings=embeddings,
                model=response.model,
                usage=TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    total_tokens=response.usage.total_tokens,
                ),
            )
        
        return await self._make_request(
            operation,
            operation_name="embeddings.create",
            estimated_tokens=estimated_tokens,
        )
    
    async def embed_single(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            model: Embedding model to use
            
        Returns:
            Embedding vector
        """
        response = await self.embed(EmbeddingRequest(texts=[text], model=model))
        return response.embeddings[0]
    
    async def create_image(
        self,
        prompt: str,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
    ) -> List[str]:
        """
        Generate images using DALL-E.
        
        Args:
            prompt: Image description
            model: DALL-E model (dall-e-2 or dall-e-3)
            size: Image size
            quality: Image quality (standard or hd for dall-e-3)
            n: Number of images
            
        Returns:
            List of image URLs
        """
        self._ensure_initialized()
        
        async def operation() -> List[str]:
            response = await self._client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=n,
            )
            return [img.url for img in response.data]
        
        return await self._make_request(operation, operation_name="images.generate")
    
    async def transcribe_audio(
        self,
        audio_file: Union[str, bytes],
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_file: Path to audio file or bytes
            model: Whisper model
            language: Optional language code
            prompt: Optional prompt
            
        Returns:
            Transcribed text
        """
        self._ensure_initialized()
        
        import io
        
        async def operation() -> str:
            if isinstance(audio_file, str):
                with open(audio_file, "rb") as f:
                    file_obj = io.BytesIO(f.read())
                    file_obj.name = audio_file
            else:
                file_obj = io.BytesIO(audio_file)
                file_obj.name = "audio.mp3"
            
            kwargs: Dict[str, Any] = {
                "model": model,
                "file": file_obj,
            }
            if language:
                kwargs["language"] = language
            if prompt:
                kwargs["prompt"] = prompt
            
            response = await self._client.audio.transcriptions.create(**kwargs)
            return response.text
        
        return await self._make_request(operation, operation_name="audio.transcribe")
    
    async def moderate(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Check content against moderation policies.
        
        Args:
            text: Text or list of texts to moderate
            
        Returns:
            Moderation results
        """
        self._ensure_initialized()
        
        async def operation() -> Dict[str, Any]:
            response = await self._client.moderations.create(
                model="omni-moderation-latest",
                input=text,
            )
            return response.model_dump()
        
        return await self._make_request(operation, operation_name="moderations.create")
    
    def _build_params(self, request: CompletionRequest) -> Dict[str, Any]:
        """
        Build OpenAI API parameters from request.
        
        Args:
            request: Completion request
            
        Returns:
            Dictionary of API parameters
        """
        params: Dict[str, Any] = {
            "model": self._get_model(request),
            "messages": [msg.to_dict() for msg in request.messages],
            "temperature": request.temperature,
        }
        
        # Optional parameters
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            params["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            params["presence_penalty"] = request.presence_penalty
        if request.stop:
            params["stop"] = request.stop
        if request.tools:
            params["tools"] = [t.to_openai_format() for t in request.tools]
        if request.tool_choice:
            params["tool_choice"] = request.tool_choice
        
        # Add any extra parameters
        params.update(request.extra_params)
        
        return params
    
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Try a simple, cheap completion
            test_request = CompletionRequest(
                messages=[Message.user("Hi")],
                model="gpt-4o-mini",
                max_tokens=5,
            )
            await self.complete(test_request)
            return True
        except Exception:
            return False
    
    async def close(self) -> None:
        """Close the provider and release resources."""
        if self._client:
            maybe_close = self._client.close()
            if asyncio.iscoroutine(maybe_close):
                await maybe_close
        self._initialized = False
    
    async def __aenter__(self) -> OpenAIProvider:
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


# =============================================================================
# Convenience Functions
# =============================================================================

async def openai_complete(
    messages: List[Message],
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> CompletionResponse:
    """
    Quick completion with OpenAI.
    
    Args:
        messages: List of messages
        api_key: OpenAI API key (or from env)
        model: Model to use
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        **kwargs: Additional parameters
        
    Returns:
        Completion response
    """
    provider = OpenAIProvider(api_key=api_key)
    await provider.initialize()
    
    try:
        request = CompletionRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return await provider.complete(request)
    finally:
        await provider.close()


async def openai_complete_text(
    prompt: str,
    system: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    Quick text completion with OpenAI.
    
    Args:
        prompt: User prompt
        system: Optional system message
        api_key: OpenAI API key
        **kwargs: Additional parameters
        
    Returns:
        Generated text
    """
    messages: List[Message] = []
    if system:
        messages.append(Message.system(system))
    messages.append(Message.user(prompt))
    
    response = await openai_complete(messages, api_key=api_key, **kwargs)
    return response.content


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Exceptions
    "OpenAIError",
    "OpenAIRateLimitError",
    "OpenAIAuthenticationError",
    "OpenAIInvalidRequestError",
    "OpenAIServerError",
    "OpenAITimeoutError",
    # Rate Limiting
    "RateLimitConfig",
    "TokenBucketRateLimiter",
    # Retry
    "RetryConfig",
    "ExponentialBackoffRetry",
    # Provider
    "OpenAIProvider",
    # Convenience Functions
    "openai_complete",
    "openai_complete_text",
]
