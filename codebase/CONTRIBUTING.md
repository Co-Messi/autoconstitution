# Contributing to autoconstitution

Welcome! We're thrilled that you're interested in contributing to autoconstitution. This document will guide you through the process of contributing to our project, whether you're adding a new provider, implementing a new metric, or improving the codebase.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Add a New Provider](#how-to-add-a-new-provider)
- [How to Add a New Metric](#how-to-add-a-new-metric)
- [How to Run the Benchmark Suite](#how-to-run-the-benchmark-suite)
- [Code Style Guide](#code-style-guide)
- [Pull Request Process](#pull-request-process)
- [Getting Help](#getting-help)

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- A GitHub account

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/autoconstitution.git
cd autoconstitution
```

3. Add the upstream remote:

```bash
git remote add upstream https://github.com/autoconstitution/autoconstitution.git
```

---

## Development Setup

### 1. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode along with all development dependencies:
- `pytest` and `pytest-cov` for testing
- `mypy` for type checking
- `ruff` for linting and formatting
- `black` for code formatting

### 3. Verify Your Setup

```bash
# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy autoconstitution
```

---

## How to Add a New Provider

Providers in autoconstitution are async-first implementations of LLM API clients. They follow a consistent pattern for easy integration.

### Step 1: Create the Provider File

Create a new file in `autoconstitution/providers/` named after your provider (e.g., `myprovider.py`).

### Step 2: Implement the Provider Class

Your provider should inherit from `BaseProvider` and implement the required interface:

```python
"""
MyProvider for autoconstitution

A complete async client for MyProvider's API.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

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
# Exceptions
# =============================================================================

class MyProviderError(Exception):
    """Base exception for MyProvider errors."""
    
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


class MyProviderRateLimitError(MyProviderError):
    """Raised when rate limit is exceeded."""
    pass


class MyProviderAuthenticationError(MyProviderError):
    """Raised when authentication fails."""
    pass


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MyProviderConfig(ProviderConfig):
    """Configuration for MyProvider."""
    
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_model: str = "default-model"
    timeout: float = 60.0
    max_retries: int = 3


# =============================================================================
# Provider Implementation
# =============================================================================

@register_provider(ProviderType.MYPROVIDER)
class MyProvider(BaseProvider):
    """
    Async MyProvider API implementation.
    
    Features:
    - Full async/await support
    - Tool calling
    - Streaming responses
    - Comprehensive error handling
    - Rate limit management
    
    Example:
        >>> provider = MyProvider(api_key="your-api-key")
        >>> await provider.initialize()
        >>> response = await provider.complete(CompletionRequest(
        ...     messages=[Message.user("Hello!")],
        ... ))
        >>> print(response.content)
    """
    
    # Model constants
    DEFAULT_MODEL = "default-model"
    AVAILABLE_MODELS = [
        "model-1",
        "model-2",
        "model-3",
    ]
    
    # Token limits per model
    MODEL_TOKEN_LIMITS: Dict[str, int] = {
        "model-1": 8192,
        "model-2": 32768,
        "model-3": 128000,
    }
    
    def __init__(
        self,
        config: Optional[MyProviderConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the MyProvider provider."""
        super().__init__(config, **kwargs)
        
        # API credentials
        self._api_key = self.config.api_key or os.getenv("MYPROVIDER_API_KEY")
        self._base_url = self.config.base_url or os.getenv("MYPROVIDER_BASE_URL")
        
        # Client instance
        self._client: Optional[Any] = None
    
    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type identifier."""
        return ProviderType.MYPROVIDER
    
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
    
    async def initialize(self) -> None:
        """
        Initialize the async client.
        
        Raises:
            ImportError: If required package is not installed
            ValueError: If API key is not provided
        """
        if self._initialized:
            return
        
        try:
            import myprovider_package
        except ImportError:
            raise ImportError(
                "MyProvider requires 'myprovider_package'. "
                "Install with: pip install myprovider_package"
            )
        
        if not self._api_key:
            raise ValueError(
                "MyProvider API key required. Set MYPROVIDER_API_KEY environment variable "
                "or pass api_key to config."
            )
        
        # Initialize your client here
        self._client = myprovider_package.AsyncClient(api_key=self._api_key)
        self._initialized = True
    
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Execute a non-streaming chat completion request.
        
        Args:
            request: The completion request
            
        Returns:
            CompletionResponse with the generated content
        """
        self._ensure_initialized()
        
        # Build request parameters
        params = self._build_params(request)
        
        try:
            response = await self._client.chat.completions.create(**params)
            
            return CompletionResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage=TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                ),
                finish_reason=response.choices[0].finish_reason,
                raw_response=response.model_dump(),
            )
        except Exception as e:
            raise self._handle_error(e)
    
    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionResponse]:
        """
        Execute a streaming chat completion request.
        
        Args:
            request: The completion request
            
        Yields:
            CompletionResponse chunks with partial content
        """
        self._ensure_initialized()
        
        params = self._build_params(request)
        params["stream"] = True
        
        try:
            stream = await self._client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    yield CompletionResponse(
                        content=delta.content or "",
                        model=chunk.model,
                        usage=TokenUsage(),
                        finish_reason=chunk.choices[0].finish_reason,
                        raw_response=chunk.model_dump(),
                    )
        except Exception as e:
            raise self._handle_error(e)
    
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for texts."""
        self._ensure_initialized()
        
        response = await self._client.embeddings.create(
            model=request.model or "embedding-model",
            input=request.texts,
        )
        
        return EmbeddingResponse(
            embeddings=[item.embedding for item in response.data],
            model=response.model,
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )
    
    def _build_params(self, request: CompletionRequest) -> Dict[str, Any]:
        """Build API parameters from request."""
        params: Dict[str, Any] = {
            "model": self._get_model(request),
            "messages": [msg.to_dict() for msg in request.messages],
            "temperature": request.temperature,
        }
        
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.tools:
            params["tools"] = [t.to_openai_format() for t in request.tools]
        
        params.update(request.extra_params)
        return params
    
    def _handle_error(self, error: Exception) -> MyProviderError:
        """Convert provider errors to our error types."""
        # Map provider-specific errors to our error types
        if "rate limit" in str(error).lower():
            return MyProviderRateLimitError(str(error))
        elif "authentication" in str(error).lower():
            return MyProviderAuthenticationError(str(error))
        else:
            return MyProviderError(str(error))
    
    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible."""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Try a simple, cheap completion
            test_request = CompletionRequest(
                messages=[Message.user("Hi")],
                max_tokens=5,
            )
            await self.complete(test_request)
            return True
        except Exception:
            return False
    
    async def close(self) -> None:
        """Close the provider and release resources."""
        if self._client:
            await self._client.close()
        self._initialized = False
```

### Step 3: Update the Provider Package

Add your provider to `autoconstitution/providers/__init__.py`:

```python
# Import MyProvider provider when available
try:
    from .myprovider import (
        MyProviderError,
        MyProviderRateLimitError,
        MyProviderAuthenticationError,
        MyProviderConfig,
        MyProvider,
    )
    
    __all__.extend([
        "MyProviderError",
        "MyProviderRateLimitError",
        "MyProviderAuthenticationError",
        "MyProviderConfig",
        "MyProvider",
    ])
    
except ImportError:
    pass  # myprovider_package not installed
```

### Step 4: Add Tests

Create tests in `tests/test_providers.py`:

```python
def test_myprovider_initialization():
    """Test MyProvider initialization."""
    provider = MyProvider(api_key="test-key")
    assert provider.provider_type == ProviderType.MYPROVIDER
    assert provider.default_model == "default-model"


@pytest.mark.asyncio
async def test_myprovider_complete():
    """Test MyProvider completion."""
    provider = MyProvider(api_key="test-key")
    # Add your test logic here
```

### Step 5: Update Documentation

Add your provider to the README.md provider list and include any special configuration options.

---

## How to Add a New Metric

Metrics in autoconstitution are captured through the `Experiment` class and stored in `MetricsSnapshot` objects.

### Step 1: Define Your Metric

Decide on:
- **Name**: A unique identifier (e.g., `custom_accuracy`)
- **Type**: One of `SCALAR`, `TIMESERIES`, `HISTOGRAM`, `COUNTER`, `GAUGE`
- **Unit**: What the metric represents (e.g., `percent`, `seconds`, `count`)

### Step 2: Add Metric Collection to Your Experiment

```python
from autoconstitution.experiment import Experiment, ExperimentConfig

async def my_training_loop(experiment: Experiment) -> dict:
    """Training loop with custom metrics."""
    
    for epoch in range(num_epochs):
        # Your training logic here
        custom_metric_value = calculate_custom_metric()
        
        # Log the metric
        experiment.log_scalar("custom_accuracy", custom_metric_value, step=epoch)
        
        # Log multiple metrics at once
        experiment.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "custom_accuracy": custom_metric_value,
        }, step=epoch)
        
        # Increment step (optional)
        experiment.increment_step()
    
    return {"final_accuracy": custom_metric_value}

# Run the experiment
config = ExperimentConfig(
    name="my_experiment",
    timeout_seconds=3600,
)
experiment = Experiment(config)
result = await experiment.run(my_training_loop)
```

### Step 3: Create a Custom Metrics Collector (Optional)

For more complex metrics, implement a custom `MetricsCollector`:

```python
from autoconstitution.experiment import MetricsCollector, MetricsSnapshot, MetricValue, MetricType

class CustomMetricsCollector:
    """Custom metrics collector for specialized measurements."""
    
    def __init__(self) -> None:
        self._metrics: dict[str, list[float]] = {}
    
    async def collect(self) -> MetricsSnapshot:
        """Collect current metrics snapshot."""
        snapshot = MetricsSnapshot()
        
        # Add your custom metrics
        snapshot.add_scalar(
            "custom_metric",
            self._calculate_metric(),
            tags={"source": "custom_collector"}
        )
        
        return snapshot
    
    async def record(self, metric: MetricValue) -> None:
        """Record a single metric value."""
        if metric.name not in self._metrics:
            self._metrics[metric.name] = []
        self._metrics[metric.name].append(float(metric.value))
    
    def _calculate_metric(self) -> float:
        """Calculate your custom metric."""
        # Your calculation logic
        return 0.0
```

### Step 4: Access Metrics in Results

```python
# Get metric from experiment result
result = await experiment.run(my_training_loop)

# Access final metrics
accuracy = result.final_metrics.get_scalar("custom_accuracy")

# Get full metric metadata
metric = result.get_metric("custom_accuracy")
if metric:
    print(f"Value: {metric.value}")
    print(f"Timestamp: {metric.timestamp}")
    print(f"Tags: {metric.tags}")

# Export to JSON
print(result.to_json())
```

---

## How to Run the Benchmark Suite

autoconstitution includes a comprehensive benchmark suite for comparing configurations and measuring performance.

### Basic Benchmark Run

```bash
# Run default benchmarks
autoconstitution benchmark

# Custom agent counts
autoconstitution benchmark --agents 10 50 100 500

# Custom iterations
autoconstitution benchmark --iterations 200

# Multiple runs for statistical significance
autoconstitution benchmark --runs 5

# Save results to file
autoconstitution benchmark --output benchmark_results.json
```

### Using a Benchmark Configuration File

Create `benchmark_config.yaml`:

```yaml
# Benchmark configuration
name: "my_benchmark"
description: "Comparing agent configurations"

# Agent configurations to test
agent_counts: [10, 50, 100, 200]
iterations: 100
runs_per_config: 3

# Metrics to collect
metrics:
  - convergence_rate
  - communication_overhead
  - execution_time
  - memory_usage

# Output settings
output_dir: "./benchmark_results"
save_visualizations: true
```

Run with configuration:

```bash
autoconstitution benchmark --config benchmark_config.yaml
```

### Programmatic Benchmark Usage

```python
from autoconstitution.cli import benchmark
from pathlib import Path

# Run benchmarks programmatically
benchmark(
    agents=[10, 50, 100],
    iterations=100,
    runs=3,
    output=Path("results.json"),
)
```

### Interpreting Results

The benchmark suite produces:

1. **Console Output**: Real-time progress and summary tables
2. **JSON Results**: Detailed results saved to file
3. **Visualizations**: Charts comparing configurations (if enabled)

Example output:

```
Benchmark Results
┏━━━━━━━━┳━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Agents ┃ Run ┃ Iterations ┃ Time (s) ┃ Throughput (it/s) ┃
┡━━━━━━━━╇━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│     10 │   1 │        100 │    1.234 │             81.04 │
│     10 │   2 │        100 │    1.198 │             83.47 │
│     50 │   1 │        100 │    2.456 │             40.72 │
└────────┴─────┴────────────┴──────────┴───────────────────┘

Summary Statistics:
┏━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Agents ┃ Avg Time (s) ┃ Avg Throughput (s)  ┃ Speedup ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│     10 │        1.216 │               82.26 │   1.00x │
│     50 │        2.412 │               41.46 │   0.50x │
└────────┴──────────────┴─────────────────────┴─────────┘
```

---

## Code Style Guide

We use automated tools to enforce code style. Please ensure your code passes all checks before submitting.

### Formatting with Black

```bash
# Format all files
black autoconstitution tests

# Check formatting without making changes
black --check autoconstitution tests
```

Configuration (in `pyproject.toml`):
- Line length: 100 characters
- Target Python versions: 3.9, 3.10, 3.11, 3.12

### Linting with Ruff

```bash
# Check for linting errors
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Format code
ruff format .
```

Enabled rules:
- `E`, `W`: pycodestyle errors and warnings
- `F`: Pyflakes
- `I`: isort (import sorting)
- `N`: pep8-naming
- `UP`: pyupgrade
- `B`: flake8-bugbear
- `C4`: flake8-comprehensions
- `SIM`: flake8-simplify

### Type Checking with MyPy

```bash
# Run type checker
mypy autoconstitution
```

Configuration:
- Strict mode enabled
- All functions must have type annotations
- No untyped definitions allowed

### Import Style

```python
# Standard library imports
import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

# Third-party imports
import typer
import yaml
from rich.console import Console

# Local imports
from autoconstitution.providers import BaseProvider
from autoconstitution.experiment import Experiment
```

### Documentation Strings

All public functions, classes, and modules must have docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When this error occurs
        
    Example:
        >>> result = my_function("test", 42)
        >>> print(result)
        True
    """
    return True
```

### Naming Conventions

- **Modules**: `lowercase.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`
- **Type variables**: `T`, `TConfig`, `TMetrics` (PascalCase, short)

---

## Pull Request Process

### Before You Start

1. **Check existing issues**: Look for related issues or discussions
2. **Open an issue**: For significant changes, discuss first
3. **Claim an issue**: Comment on the issue you'd like to work on

### Development Workflow

1. **Create a branch**:
   ```bash
   git checkout -b feature/my-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes**:
   - Write clear, focused commits
   - Follow the code style guide
   - Add tests for new functionality
   - Update documentation as needed

3. **Run quality checks**:
   ```bash
   # Run all checks
   ruff check .
   black --check .
   mypy autoconstitution
   pytest
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new provider for X API"
   ```

   Commit message format:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test additions/changes
   - `refactor:` Code refactoring
   - `style:` Code style changes (formatting)
   - `chore:` Maintenance tasks

5. **Push to your fork**:
   ```bash
   git push origin feature/my-feature-name
   ```

6. **Create a Pull Request**:
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill out the PR template

### PR Requirements

- [ ] All CI checks pass
- [ ] Code is formatted with Black
- [ ] No linting errors with Ruff
- [ ] Type checking passes with MyPy
- [ ] Tests pass with pytest
- [ ] New code has test coverage
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (if applicable)

### PR Review Process

1. **Automated checks**: CI will run linting, type checking, and tests
2. **Code review**: Maintainers will review your code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, a maintainer will merge

### After Merge

Your contribution will be:
- Merged into the main branch
- Included in the next release
- Credited in the release notes

---

## Getting Help

### Resources

- **Documentation**: [https://autoconstitution.readthedocs.io](https://autoconstitution.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/autoconstitution/autoconstitution/issues)
- **Discussions**: [GitHub Discussions](https://github.com/autoconstitution/autoconstitution/discussions)

### Communication Guidelines

- Be respectful and constructive
- Provide context when asking questions
- Search before asking
- Help others when you can

### Reporting Bugs

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Reproduction**: Steps to reproduce
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: Python version, OS, package versions
6. **Logs**: Relevant error messages or logs

Example bug report:

```markdown
**Description**
The OpenAI provider fails to initialize when using Azure endpoints.

**Reproduction**
1. Set `OPENAI_BASE_URL` to Azure endpoint
2. Call `OpenAIProvider().initialize()`
3. Error occurs

**Expected behavior**
Provider should initialize successfully.

**Actual behavior**
Raises `OpenAIAuthenticationError: Invalid API key format`.

**Environment**
- Python: 3.11.4
- OS: Ubuntu 22.04
- autoconstitution: 0.1.0
- openai: 1.12.0

**Logs**
```
Traceback (most recent call last):
  File "...", line 42, in <module>
    await provider.initialize()
OpenAIAuthenticationError: Invalid API key format
```
```

---

## Recognition

Contributors will be:
- Listed in the CONTRIBUTORS.md file
- Mentioned in release notes
- Credited in documentation where appropriate

Thank you for contributing to autoconstitution! Your efforts help make this project better for everyone.

---

## License

By contributing to autoconstitution, you agree that your contributions will be licensed under the MIT License.
