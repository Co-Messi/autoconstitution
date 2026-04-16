"""
autoconstitution Providers Package.

This package contains provider implementations for various LLM APIs.
"""

from __future__ import annotations

# Initialize __all__
__all__: list[str] = []

# Auto-detect helper is always importable (only depends on httpx, which is a
# base dep). Individual providers load lazily inside pick_provider().
from autoconstitution.providers import auto_detect
from autoconstitution.providers.auto_detect import ProviderChoice, pick_provider

__all__.extend(["auto_detect", "ProviderChoice", "pick_provider"])

# Import Kimi provider when available
try:
    from .kimi import (
        # Client
        KimiClient,
        ChatClient,
        ChatCompletionsClient,
        
        # Provider Interface
        KimiProvider,
        ProviderConfig,
        
        # Models
        KimiModel,
        
        # Data Classes
        Message,
        Tool,
        ToolCall,
        FunctionCall,
        FunctionDefinition,
        Usage,
        Choice,
        StreamChoice,
        DeltaMessage,
        ChatCompletion,
        ChatCompletionChunk,
        RateLimitInfo,
        
        # Exceptions
        KimiError,
        KimiAuthenticationError,
        KimiPermissionError,
        KimiNotFoundError,
        KimiRateLimitError,
        KimiServerError,
        KimiValidationError,
        KimiTimeoutError,
        KimiConnectionError,
        
        # Helpers
        ToolExecutor,
        RateLimiter,
        
        # Convenience Functions
        create_completion,
        create_streaming_completion,
        
        # Constants
        DEFAULT_BASE_URL,
        DEFAULT_TIMEOUT,
        DEFAULT_MAX_RETRIES,
        DEFAULT_RETRY_DELAY,
    )
    
    __all__.extend([
        # Client
        "KimiClient",
        "ChatClient",
        "ChatCompletionsClient",
        
        # Provider Interface
        "KimiProvider",
        "ProviderConfig",
        
        # Models
        "KimiModel",
        
        # Data Classes
        "Message",
        "Tool",
        "ToolCall",
        "FunctionCall",
        "FunctionDefinition",
        "Usage",
        "Choice",
        "StreamChoice",
        "DeltaMessage",
        "ChatCompletion",
        "ChatCompletionChunk",
        "RateLimitInfo",
        
        # Exceptions
        "KimiError",
        "KimiAuthenticationError",
        "KimiPermissionError",
        "KimiNotFoundError",
        "KimiRateLimitError",
        "KimiServerError",
        "KimiValidationError",
        "KimiTimeoutError",
        "KimiConnectionError",
        
        # Helpers
        "ToolExecutor",
        "RateLimiter",
        
        # Convenience Functions
        "create_completion",
        "create_streaming_completion",
        
        # Constants
        "DEFAULT_BASE_URL",
        "DEFAULT_TIMEOUT",
        "DEFAULT_MAX_RETRIES",
        "DEFAULT_RETRY_DELAY",
    ])
    
except ImportError:
    pass  # aiohttp not installed


# Import Ollama provider when available
try:
    from .ollama import (
        # Exceptions
        OllamaError,
        OllamaConnectionError,
        OllamaModelNotFoundError,
        OllamaTimeoutError,
        OllamaValidationError,
        OllamaServerError,
        
        # Data Types
        Role,
        Message,
        Tool,
        TokenUsage,
        CompletionRequest,
        CompletionResponse,
        EmbeddingRequest,
        EmbeddingResponse,
        StreamChunk,
        ModelInfo,
        OllamaConfig,
        
        # Core Classes
        RetryHandler,
        ToolExecutor,
        OllamaProvider,
        
        # Apple Silicon Helpers
        detect_apple_silicon,
        get_optimal_thread_count,
        get_optimal_gpu_layers,
        
        # Convenience Functions
        create_ollama_provider,
        ollama_complete,
        ollama_complete_text,
        ollama_generate,
    )
    
    __all__.extend([
        # Exceptions
        "OllamaError",
        "OllamaConnectionError",
        "OllamaModelNotFoundError",
        "OllamaTimeoutError",
        "OllamaValidationError",
        "OllamaServerError",
        
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
        "ModelInfo",
        "OllamaConfig",
        
        # Core Classes
        "RetryHandler",
        "ToolExecutor",
        "OllamaProvider",
        
        # Apple Silicon Helpers
        "detect_apple_silicon",
        "get_optimal_thread_count",
        "get_optimal_gpu_layers",
        
        # Convenience Functions
        "create_ollama_provider",
        "ollama_complete",
        "ollama_complete_text",
        "ollama_generate",
    ])
    
except ImportError:
    pass  # aiohttp not installed
