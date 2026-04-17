"""
autoconstitution providers.

This module provides:

- shared provider-facing data structures used by the OpenAI adapter
- a tiny provider registry for optional provider registration
- lazy imports for the concrete provider modules
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ProviderType(str, Enum):
    KIMI = "kimi"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    FAKE = "fake"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str, name: str | None = None) -> Message:
        return cls(role=Role.USER, content=content, name=name)

    @classmethod
    def assistant(
        cls, content: str, tool_calls: list[dict[str, Any]] | None = None
    ) -> Message:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> Message:
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_format(self) -> dict[str, Any]:
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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def cost_estimate(self) -> float:
        input_cost = self.prompt_tokens * 0.000005
        output_cost = self.completion_tokens * 0.000015
        return input_cost + output_cost


@dataclass
class CompletionRequest:
    messages: list[Message]
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | None = None
    tools: list[Tool] | None = None
    tool_choice: Any | None = None
    stream: bool = False
    response_format: dict[str, Any] | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResponse:
    content: str
    model: str
    usage: TokenUsage
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    raw_response: dict[str, Any] | None = None
    id: str | None = None


@dataclass
class EmbeddingRequest:
    texts: list[str]
    model: str | None = None


@dataclass
class EmbeddingResponse:
    embeddings: list[list[float]]
    model: str
    usage: TokenUsage


@dataclass
class ProviderConfig:
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 30.0
    max_retries: int = 3
    default_model: str | None = None
    organization: str | None = None
    extra_headers: dict[str, str] | None = None


class BaseProvider(ABC):
    def __init__(self, config: ProviderConfig | None = None, **kwargs: Any) -> None:
        self.config = config or ProviderConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key) and value is not None:
                setattr(self.config, key, value)
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")


_PROVIDER_REGISTRY: dict[ProviderType, type[Any]] = {}


def register_provider(provider_type: ProviderType) -> Callable[[type[Any]], type[Any]]:
    def decorator(cls: type[Any]) -> type[Any]:
        _PROVIDER_REGISTRY[provider_type] = cls
        return cls

    return decorator


__all__ = [
    "ProviderType",
    "Role",
    "Message",
    "Tool",
    "TokenUsage",
    "CompletionRequest",
    "CompletionResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ProviderConfig",
    "BaseProvider",
    "register_provider",
]

# Lazy convenience imports
from autoconstitution.providers import auto_detect
from autoconstitution.providers.auto_detect import ProviderChoice, pick_provider

__all__.extend(["auto_detect", "ProviderChoice", "pick_provider"])

