---
icon: custom/cpu
---
# LLM Providers

Axion provides a unified interface for working with multiple LLM providers through the `LLMRegistry`. This guide covers using built-in providers and registering custom ones.

## Quick Start

```python
from axion.llm_registry import LLMRegistry

# Create a registry locked to a provider
registry = LLMRegistry(provider='openai')
llm = registry.get_llm('gpt-4o')

# Or use flexible mode to switch providers
registry = LLMRegistry()
llm_openai = registry.get_llm('gpt-4o', provider='openai')
llm_claude = registry.get_llm('claude-3-5-sonnet-20241022', provider='anthropic')
```

## Built-in Providers

<div class="rule-grid" markdown="0">
<div class="rule-card">
<span class="rule-card__number">&#x2713;</span>
<p class="rule-card__title">OpenAI</p>
<p class="rule-card__desc">GPT-4o, GPT-4, o1, o3, etc. Embeddings supported. Auth: <code>OPENAI_API_KEY</code></p>
</div>
<div class="rule-card">
<span class="rule-card__number">&#x2713;</span>
<p class="rule-card__title">Anthropic</p>
<p class="rule-card__desc">Claude 3.5, Claude 3, etc. No embeddings. Auth: <code>ANTHROPIC_API_KEY</code></p>
</div>
<div class="rule-card">
<span class="rule-card__number">&#x2713;</span>
<p class="rule-card__title">Gemini</p>
<p class="rule-card__desc">Gemini 1.5 Pro/Flash, etc. Embeddings supported. Auth: <code>GOOGLE_API_KEY</code></p>
</div>
<div class="rule-card">
<span class="rule-card__number">&#x2713;</span>
<p class="rule-card__title">Vertex AI / HuggingFace</p>
<p class="rule-card__desc">Gemini &amp; Claude on Vertex (GCP auth) + any HF Hub model (local inference).</p>
</div>
</div>

### OpenAI

```python
registry = LLMRegistry(provider='openai')

# LLM
llm = registry.get_llm('gpt-4o')
llm = registry.get_llm('gpt-4o-mini', temperature=0.7)
llm = registry.get_llm('o1-preview')

# Embeddings
embedding = registry.get_embedding_model('text-embedding-3-small')
```

### Anthropic

```python
registry = LLMRegistry(provider='anthropic')

# LLM - automatically prefixed with 'anthropic/' for LiteLLM
llm = registry.get_llm('claude-3-5-sonnet-20241022')
llm = registry.get_llm('claude-3-opus-20240229')

# Embeddings - NOT supported (use OpenAI or HuggingFace)
# registry.get_embedding_model()  # Raises NotImplementedError
```

### Gemini

```python
registry = LLMRegistry(provider='gemini')

# LLM - automatically prefixed with 'gemini/' for LiteLLM
llm = registry.get_llm('gemini-1.5-pro')
llm = registry.get_llm('gemini-1.5-flash')

# Embeddings
embedding = registry.get_embedding_model('models/embedding-001')
```

### Vertex AI

```python
# Requires GCP service account authentication
registry = LLMRegistry(
    provider='vertex_ai',
    vertex_project='my-gcp-project',
    vertex_location='us-central1',
    vertex_credentials='/path/to/service-account.json'
)

# Or use environment variables:
# VERTEXAI_PROJECT, VERTEXAI_LOCATION, GOOGLE_APPLICATION_CREDENTIALS

# LLM - automatically prefixed with 'vertex_ai/' for LiteLLM
llm = registry.get_llm('gemini-1.5-pro')

# Embeddings
embedding = registry.get_embedding_model('text-embedding-004')
```

### HuggingFace

```python
registry = LLMRegistry(provider='huggingface')

# LLM - uses LlamaIndex for local inference (not LiteLLM)
llm = registry.get_llm('meta-llama/Llama-2-7b-chat-hf')

# Embeddings
embedding = registry.get_embedding_model('BAAI/bge-small-en-v1.5')
```

!!! warning "HuggingFace Requirements"
    HuggingFace provider requires `torch` and `transformers` to be installed.
    Models run locally, not via API.

## Registering a Custom Provider

To add a new provider, create a class that inherits from `BaseProvider` and use the `@LLMRegistry.register()` decorator.

### Basic Structure

```python
from typing import Any, Optional
from axion.llm_registry import BaseProvider, LLMRegistry, LiteLLMWrapper

@LLMRegistry.register('my_provider')
class MyProvider(BaseProvider):
    """Custom provider for My LLM Service."""

    # LiteLLM prefix for routing (e.g., 'anthropic/', 'gemini/')
    LITELLM_PREFIX = 'my_provider/'

    # Whether this provider supports embedding models
    SUPPORTS_EMBEDDINGS = True

    def __init__(self, api_key: Optional[str] = None, **credentials):
        super().__init__(api_key=api_key, **credentials)

    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        """Create an embedding model client."""
        # Return your embedding model instance
        pass
```

### Example: Custom OpenAI-Compatible Endpoint

```python
from typing import Any, Optional
from axion.llm_registry import BaseProvider, LLMRegistry, LiteLLMWrapper

@LLMRegistry.register('custom_openai')
class CustomOpenAIProvider(BaseProvider):
    """Provider for OpenAI-compatible endpoints (e.g., vLLM, LocalAI)."""

    LITELLM_PREFIX = 'openai/'  # Use OpenAI routing
    SUPPORTS_EMBEDDINGS = False

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **credentials
    ):
        super().__init__(api_key=api_key, **credentials)
        self.api_base = api_base

    def _create_litellm_wrapper(self, model_name: str, **kwargs) -> LiteLLMWrapper:
        """Override to add custom base URL."""
        if self.LITELLM_PREFIX and not model_name.startswith(self.LITELLM_PREFIX):
            model_name = f'{self.LITELLM_PREFIX}{model_name}'

        temperature = kwargs.pop('temperature', 0.0)

        return LiteLLMWrapper(
            model=model_name,
            provider='custom_openai',
            temperature=temperature,
            api_key=self.api_key,
            api_base=self.api_base,
            **kwargs
        )

    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        raise NotImplementedError('This provider does not support embeddings.')


# Usage
registry = LLMRegistry(
    provider='custom_openai',
    api_key='your-api-key',
    api_base='http://localhost:8000/v1'
)
llm = registry.get_llm('my-local-model')
```

### Example: Non-LiteLLM Backend

For providers that don't use LiteLLM (like HuggingFace local inference), override `create_llm()`:

```python
from typing import Any, Optional
from axion.llm_registry import BaseProvider, LLMRegistry

@LLMRegistry.register('ollama')
class OllamaProvider(BaseProvider):
    """Provider for Ollama local models."""

    LITELLM_PREFIX = ''
    SUPPORTS_EMBEDDINGS = True

    def __init__(self, base_url: str = 'http://localhost:11434', **credentials):
        super().__init__(api_key=None, **credentials)
        self.base_url = base_url

    def create_llm(self, model_name: str, **kwargs) -> Any:
        """Override to use Ollama-specific client instead of LiteLLM."""
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=model_name,
            base_url=self.base_url,
            **kwargs
        )

    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        from llama_index.embeddings.ollama import OllamaEmbedding

        return OllamaEmbedding(
            model_name=model_name,
            base_url=self.base_url,
            **kwargs
        )


# Usage
registry = LLMRegistry(provider='ollama')
llm = registry.get_llm('llama2')
```

## Provider Class Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `LITELLM_PREFIX` | `str` | Prefix added to model names for LiteLLM routing (e.g., `'anthropic/'`) |
| `SUPPORTS_EMBEDDINGS` | `bool` | Whether the provider offers embedding models (default: `True`) |

## Key Methods to Implement

| Method | Required | Description |
|--------|----------|-------------|
| `__init__()` | Yes | Initialize with credentials |
| `create_embedding_model()` | Yes | Return embedding model instance (or raise `NotImplementedError`) |
| `create_llm()` | No | Override only for non-LiteLLM backends |
| `_create_litellm_wrapper()` | No | Override to customize LiteLLM wrapper creation |

## Cost Estimation

Add custom model pricing to the cost estimator:

```python
from axion.llm_registry import LLMCostEstimator

# Add custom model pricing (price per token)
LLMCostEstimator.add_custom_model(
    model='my-custom-model',
    input_price_per_token=0.001 / 1000,   # $0.001 per 1K tokens
    output_price_per_token=0.002 / 1000   # $0.002 per 1K tokens
)

# Estimate cost
cost = LLMCostEstimator.estimate(
    model_name='my-custom-model',
    prompt_tokens=1000,
    completion_tokens=500
)
```

## View Registered Providers

```python
from axion.llm_registry import LLMRegistry

# Display all registered providers with details
LLMRegistry.display()
```

---

[Environment Configuration :octicons-arrow-right-24:](../deep-dives/internals/environment.md){ .md-button .md-button--primary }
[Tracing :octicons-arrow-right-24:](../deep-dives/internals/tracing.md){ .md-button }
[LLM Registry Reference :octicons-arrow-right-24:](../reference/llm-registry.md){ .md-button }
