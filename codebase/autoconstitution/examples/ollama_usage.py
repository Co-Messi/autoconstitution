"""
Ollama Provider Usage Examples for autoconstitution

This module demonstrates how to use the Ollama provider for local LLM inference
on Apple Silicon (M1/M2/M3) Macs.

Prerequisites:
    1. Install Ollama: https://ollama.com or `brew install ollama`
    2. Start Ollama server: `ollama serve`
    3. Pull a model: `ollama pull llama3.2`
    4. Install Python dependencies: `pip install aiohttp`

Usage:
    python -m autoconstitution.examples.ollama_usage
"""

from __future__ import annotations

import asyncio
from typing import Any

from autoconstitution.providers.ollama import (
    OllamaProvider,
    OllamaConfig,
    Message,
    Tool,
    CompletionRequest,
    create_ollama_provider,
    ollama_complete,
    ollama_complete_text,
    detect_apple_silicon,
)


async def example_basic_completion() -> None:
    """Example: Basic text completion."""
    print("=" * 60)
    print("Example 1: Basic Completion")
    print("=" * 60)
    
    async with OllamaProvider() as provider:
        response = await provider.complete(CompletionRequest(
            messages=[Message.user("What is the capital of France?")],
            model="llama3.2",
            temperature=0.7,
            max_tokens=100,
        ))
        
        print(f"Model: {response.model}")
        print(f"Response: {response.content}")
        print(f"Tokens used: {response.usage.total_tokens}")
        print()


async def example_conversation() -> None:
    """Example: Multi-turn conversation."""
    print("=" * 60)
    print("Example 2: Multi-turn Conversation")
    print("=" * 60)
    
    messages = [
        Message.system("You are a helpful assistant specialized in geography."),
        Message.user("What is the largest country in the world?"),
    ]
    
    async with OllamaProvider() as provider:
        # First turn
        response = await provider.complete(CompletionRequest(
            messages=messages,
            model="llama3.2",
        ))
        print(f"Assistant: {response.content}")
        
        # Add assistant response and ask follow-up
        messages.append(Message.assistant(response.content))
        messages.append(Message.user("What is its population?"))
        
        response = await provider.complete(CompletionRequest(
            messages=messages,
            model="llama3.2",
        ))
        print(f"Assistant: {response.content}")
        print()


async def example_streaming() -> None:
    """Example: Streaming completion."""
    print("=" * 60)
    print("Example 3: Streaming Completion")
    print("=" * 60)
    
    async with OllamaProvider() as provider:
        request = CompletionRequest(
            messages=[Message.user("Write a short poem about AI.")],
            model="llama3.2",
            max_tokens=200,
        )
        
        print("Response: ", end="", flush=True)
        async for chunk in provider.complete_stream(request):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print("\n")


async def example_with_tools() -> None:
    """Example: Tool calling."""
    print("=" * 60)
    print("Example 4: Tool Calling")
    print("=" * 60)
    
    # Define a simple calculator tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            result = eval(expression)  # Safe in this controlled context
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    
    tools = [
        Tool(
            name="calculator",
            description="Evaluate a mathematical expression",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    },
                },
                "required": ["expression"],
            },
        ),
    ]
    
    async with OllamaProvider() as provider:
        # Check if model supports tools
        if provider.supports_tools("llama3.2"):
            response = await provider.complete(CompletionRequest(
                messages=[Message.user("What is 123 * 456?")],
                model="llama3.2",
                tools=tools,
            ))
            
            print(f"Response: {response.content}")
            if response.tool_calls:
                print(f"Tool calls: {response.tool_calls}")
        else:
            print("Model does not support tool calling")
        print()


async def example_embeddings() -> None:
    """Example: Generate embeddings."""
    print("=" * 60)
    print("Example 5: Embeddings")
    print("=" * 60)
    
    async with OllamaProvider() as provider:
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
        ]
        
        response = await provider.embed(
            type("EmbeddingRequest", (), {"texts": texts, "model": "nomic-embed-text"})()
        )
        
        print(f"Model: {response.model}")
        print(f"Number of embeddings: {len(response.embeddings)}")
        print(f"Embedding dimension: {len(response.embeddings[0])}")
        print()


async def example_model_management() -> None:
    """Example: Model management."""
    print("=" * 60)
    print("Example 6: Model Management")
    print("=" * 60)
    
    async with OllamaProvider() as provider:
        # List available models
        models = await provider.list_models()
        print(f"Available models ({len(models)}):")
        for model in models[:5]:  # Show first 5
            print(f"  - {model.name} ({model.size_gb:.2f} GB)")
        
        # Get Ollama version
        version = await provider.get_version()
        print(f"\nOllama version: {version}")
        
        # Get running models
        running = await provider.get_ps()
        print(f"Running models: {len(running)}")
        print()


async def example_apple_silicon_optimizations() -> None:
    """Example: Apple Silicon optimizations."""
    print("=" * 60)
    print("Example 7: Apple Silicon Optimizations")
    print("=" * 60)
    
    # Check if running on Apple Silicon
    is_apple_silicon = detect_apple_silicon()
    print(f"Running on Apple Silicon: {is_apple_silicon}")
    
    # Create provider with Apple Silicon optimizations
    config = OllamaConfig(
        default_model="llama3.2",
        # These are auto-detected on Apple Silicon, but can be overridden
        num_ctx=8192,  # Larger context window
    )
    
    async with OllamaProvider(config) as provider:
        print(f"Is Apple Silicon: {provider.is_apple_silicon}")
        print(f"Default model: {provider.default_model}")
        print(f"Context window: {provider.get_model_context_window()}")
        print()


async def example_convenience_functions() -> None:
    """Example: Convenience functions."""
    print("=" * 60)
    print("Example 8: Convenience Functions")
    print("=" * 60)
    
    # Quick completion
    response = await ollama_complete_text(
        prompt="What is machine learning?",
        system="You are a helpful AI tutor.",
        model="llama3.2",
        max_tokens=150,
    )
    print(f"Response: {response}")
    print()


async def example_custom_options() -> None:
    """Example: Custom generation options."""
    print("=" * 60)
    print("Example 9: Custom Generation Options")
    print("=" * 60)
    
    async with OllamaProvider() as provider:
        response = await provider.complete(CompletionRequest(
            messages=[Message.user("Write a creative story about space exploration.")],
            model="llama3.2",
            temperature=0.9,  # More creative
            top_p=0.95,
            top_k=50,
            max_tokens=300,
            # Ollama-specific options
            repeat_penalty=1.1,
            seed=42,  # For reproducibility
        ))
        
        print(f"Response: {response.content}")
        print(f"Load duration: {response.load_duration} ns")
        print(f"Eval duration: {response.eval_duration} ns")
        print()


async def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Ollama Provider Examples for autoconstitution")
    print("=" * 60 + "\n")
    
    try:
        # Run examples
        await example_basic_completion()
        await example_conversation()
        await example_streaming()
        await example_with_tools()
        await example_embeddings()
        await example_model_management()
        await example_apple_silicon_optimizations()
        await example_convenience_functions()
        await example_custom_options()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("  1. Ollama is installed: https://ollama.com")
        print("  2. Ollama server is running: ollama serve")
        print("  3. Required models are pulled: ollama pull llama3.2")


if __name__ == "__main__":
    asyncio.run(main())
