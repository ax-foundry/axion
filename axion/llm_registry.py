import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from axion._core.environment import settings
from axion._core.logging import get_logger

logger = get_logger(__name__)


class LLMCostEstimator:
    """
    DEPRECATED: Use LiteLLM's native cost tracking instead.

    Prefer using `response._hidden_params["response_cost"]` or `litellm.completion_cost()`
    for accurate real-time pricing from api.litellm.ai.

    This class now delegates to LiteLLM's cost calculation and is maintained only
    for backward compatibility.

    Example (deprecated):
    ```
    # Old way (deprecated):
    cost = LLMCostEstimator.estimate("gpt-4o", prompt_tokens=1500, completion_tokens=500)

    # New way (recommended):
    response = litellm.completion(...)
    cost = response._hidden_params.get("response_cost", 0.0)
    # or
    cost = litellm.completion_cost(completion_response=response)
    ```
    """

    @classmethod
    def estimate(
        cls,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        round_level: int = 8,
    ) -> float:
        """
        DEPRECATED: Estimate the cost of an LLM call using LiteLLM's pricing data.

        Prefer using `response._hidden_params["response_cost"]` directly from LiteLLM
        responses for real-time pricing.

        Args:
            model_name (str): The name of the LLM model.
            prompt_tokens (int): The number of tokens sent in the prompt.
            completion_tokens (int): The number of tokens generated in the response.
            round_level (int): Round level for precision (default 8)

        Returns:
            float: The estimated total cost of the call, in USD.
        """
        warnings.warn(
            "LLMCostEstimator.estimate() is deprecated. "
            "Use litellm.completion_cost() or response._hidden_params['response_cost'] instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        try:
            import litellm

            # Strip LiteLLM provider prefix if present
            model_key = model_name.lower()
            if '/' in model_key:
                model_key = model_key.split('/', 1)[1]

            # Use LiteLLM's cost_per_token function
            cost = litellm.cost_per_token(
                model=model_key,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            # litellm.cost_per_token returns a tuple (prompt_cost, completion_cost)
            if isinstance(cost, tuple):
                return round(cost[0] + cost[1], round_level)
            return round(cost, round_level)
        except Exception as e:
            logger.warning(f'Could not estimate cost for model {model_name}: {e}')
            return 0.0

    @classmethod
    def calculate_cost(
        cls, model_name: str, input_tokens: int, output_tokens: int
    ) -> float:
        """
        DEPRECATED: Alternative method name for backward compatibility.

        Args:
            model_name (str): The name of the LLM model
            input_tokens (int): The number of input/prompt tokens
            output_tokens (int): The number of output/completion tokens

        Returns:
            float: The estimated total cost of the call, in USD.
        """
        return cls.estimate(model_name, input_tokens, output_tokens)

    @classmethod
    def get_model_pricing(cls, model_name: str) -> Dict[str, float]:
        """
        DEPRECATED: Get the pricing information for a specific model from LiteLLM.

        Args:
            model_name (str): The name of the LLM model

        Returns:
            Dict[str, float]: Dictionary with 'input' and 'output' pricing per token
        """
        warnings.warn(
            "LLMCostEstimator.get_model_pricing() is deprecated. "
            "Use litellm.model_cost or litellm.get_model_cost_map() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        try:
            import litellm

            model_key = model_name.lower()
            if '/' in model_key:
                model_key = model_key.split('/', 1)[1]

            # Get pricing from LiteLLM's model_cost dictionary
            if hasattr(litellm, 'model_cost') and model_key in litellm.model_cost:
                model_info = litellm.model_cost[model_key]
                return {
                    'input': model_info.get('input_cost_per_token', 0.0),
                    'output': model_info.get('output_cost_per_token', 0.0),
                }
        except Exception as e:
            logger.warning(f'Could not get pricing for model {model_name}: {e}')

        # Return zero pricing if not found
        return {'input': 0.0, 'output': 0.0}

    @classmethod
    def list_supported_models(cls) -> List[str]:
        """
        DEPRECATED: List all supported models from LiteLLM's pricing database.

        Returns:
            List[str]: List of supported model names
        """
        warnings.warn(
            "LLMCostEstimator.list_supported_models() is deprecated. "
            "Use litellm.model_cost.keys() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        try:
            import litellm

            if hasattr(litellm, 'model_cost'):
                return list(litellm.model_cost.keys())
        except Exception as e:
            logger.warning(f'Could not list models: {e}')

        return []

    @classmethod
    def add_custom_model(
        cls, model: str, input_price_per_token: float, output_price_per_token: float
    ) -> None:
        """
        DEPRECATED: Add custom model pricing to LiteLLM's model_cost dictionary.

        Args:
            model (str): The name of the custom model
            input_price_per_token (float): Price per input token in USD
            output_price_per_token (float): Price per output token in USD
        """
        warnings.warn(
            "LLMCostEstimator.add_custom_model() is deprecated. "
            "Use litellm.register_model() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        try:
            import litellm

            if hasattr(litellm, 'model_cost'):
                litellm.model_cost[model.lower()] = {
                    'input_cost_per_token': input_price_per_token,
                    'output_cost_per_token': output_price_per_token,
                }
        except Exception as e:
            logger.warning(f'Could not add custom model {model}: {e}')


# MockLLM to bypass checks if needed
class MockLLM:
    def __init__(self, **kwargs):
        pass

    def complete(self, **kwargs):
        pass

    def acomplete(self, **kwargs):
        pass


class LiteLLMWrapper:
    """
    Lightweight LLM wrapper for LiteLLM-based execution.

    This class stores model configuration and is used by LLMHandler to route
    API calls through LiteLLM. It provides a consistent interface across
    different providers (OpenAI, Anthropic, Gemini, etc.).

    The wrapper stores:
    - model: The LiteLLM-compatible model name (e.g., 'gpt-4o', 'anthropic/claude-3-5-sonnet')
    - temperature: Generation temperature (default 0.0 for deterministic outputs)
    - provider: The provider name for metadata/logging purposes
    - Additional kwargs for provider-specific configurations

    Example:
        >>> from axion.llm_registry import LLMRegistry
        >>> registry = LLMRegistry(provider='openai')
        >>> llm = registry.get_llm('gpt-4o')
        >>> response = llm.complete('What is 2+2?')
        >>> print(response.text)
    """

    def __init__(
        self,
        model: str,
        provider: str = 'openai',
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the LiteLLM wrapper.

        Args:
            model: The LiteLLM-compatible model name
            provider: Provider identifier for metadata
            temperature: Generation temperature (0.0-1.0)
            api_key: Optional API key for the provider
            **kwargs: Additional model configuration
        """
        self.model = model
        self.temperature = temperature
        self._provider = provider
        self._api_key = api_key
        self._kwargs = kwargs

        # Create a metadata-like object for compatibility with handler code
        class Metadata:
            def __init__(self, provider: str):
                self.provider = provider

        self.metadata = Metadata(provider)

    def __repr__(self) -> str:
        return f"LiteLLMWrapper(model='{self.model}', provider='{self._provider}')"

    def complete(self, prompt: str, **kwargs) -> 'CompletionResponse':
        """
        Generate a completion for the given prompt (synchronous).

        Args:
            prompt: The input prompt to complete
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            CompletionResponse with the generated text

        Example:
            >>> llm = LLMRegistry(provider='openai', api_key='sk-...').get_llm('gpt-4o')
            >>> response = llm.complete('What is the capital of France?')
            >>> print(response.text)
        """
        import litellm

        messages = [{'role': 'user', 'content': prompt}]
        temperature = kwargs.pop('temperature', self.temperature)

        # Build call kwargs, including api_key if provided
        call_kwargs = {**self._kwargs, **kwargs}
        if self._api_key:
            call_kwargs['api_key'] = self._api_key

        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=temperature,
            **call_kwargs
        )

        # Handle case where content is None (e.g., tool calls, function calls)
        content = response.choices[0].message.content or ''

        return CompletionResponse(
            text=content,
            raw=response
        )

    async def acomplete(self, prompt: str, **kwargs) -> 'CompletionResponse':
        """
        Generate a completion for the given prompt (asynchronous).

        Args:
            prompt: The input prompt to complete
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            CompletionResponse with the generated text

        Example:
            >>> import asyncio
            >>> llm = LLMRegistry(provider='openai', api_key='sk-...').get_llm('gpt-4o')
            >>> response = asyncio.run(llm.acomplete('What is 2+2?'))
            >>> print(response.text)
        """
        import litellm

        messages = [{'role': 'user', 'content': prompt}]
        temperature = kwargs.pop('temperature', self.temperature)

        # Build call kwargs, including api_key if provided
        call_kwargs = {**self._kwargs, **kwargs}
        if self._api_key:
            call_kwargs['api_key'] = self._api_key

        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            temperature=temperature,
            **call_kwargs
        )

        # Handle case where content is None (e.g., tool calls, function calls)
        content = response.choices[0].message.content or ''

        return CompletionResponse(
            text=content,
            raw=response
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> 'CompletionResponse':
        """
        Generate a chat completion from a list of messages (synchronous).

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            CompletionResponse with the generated text

        Example:
            >>> llm = LLMRegistry(provider='openai', api_key='sk-...').get_llm('gpt-4o')
            >>> messages = [
            ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
            ...     {'role': 'user', 'content': 'Hello!'}
            ... ]
            >>> response = llm.chat(messages)
            >>> print(response.text)
        """
        import litellm

        temperature = kwargs.pop('temperature', self.temperature)

        # Build call kwargs, including api_key if provided
        call_kwargs = {**self._kwargs, **kwargs}
        if self._api_key:
            call_kwargs['api_key'] = self._api_key

        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=temperature,
            **call_kwargs
        )

        # Handle case where content is None (e.g., tool calls, function calls)
        content = response.choices[0].message.content or ''

        return CompletionResponse(
            text=content,
            raw=response
        )

    async def achat(self, messages: List[Dict[str, str]], **kwargs) -> 'CompletionResponse':
        """
        Generate a chat completion from a list of messages (asynchronous).

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            CompletionResponse with the generated text
        """
        import litellm

        temperature = kwargs.pop('temperature', self.temperature)

        # Build call kwargs, including api_key if provided
        call_kwargs = {**self._kwargs, **kwargs}
        if self._api_key:
            call_kwargs['api_key'] = self._api_key

        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            temperature=temperature,
            **call_kwargs
        )

        # Handle case where content is None (e.g., tool calls, function calls)
        content = response.choices[0].message.content or ''

        return CompletionResponse(
            text=content,
            raw=response
        )


class CompletionResponse:
    """
    Response object from LLM completion calls.

    Attributes:
        text: The generated text content
        raw: The raw response object from LiteLLM
    """

    def __init__(self, text: str, raw: Any = None):
        self.text = text
        self.raw = raw

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"CompletionResponse(text='{self.text[:50]}...')" if len(self.text) > 50 else f"CompletionResponse(text='{self.text}')"


class BaseProvider(ABC):
    """
    Abstract base class for all providers. It defines the contract that
    every provider must follow.

    Subclasses should:
    - Set LITELLM_PREFIX for LiteLLM routing (e.g., 'anthropic/', 'gemini/')
    - Implement create_embedding_model() for provider-specific embeddings
    - Optionally override create_llm() for non-LiteLLM backends (e.g., HuggingFace)

    Note: When using LiteLLM in the handler, API keys are read from environment
    variables automatically (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY).
    """

    # Provider-specific prefix for LiteLLM routing (override in subclass)
    LITELLM_PREFIX: str = ''

    # Whether this provider supports embedding models (override in subclass if False)
    SUPPORTS_EMBEDDINGS: bool = True

    def __init__(self, api_key: Optional[str] = None, **credentials):
        """Initializes the provider, storing the API key if provided."""
        self.api_key = api_key
        self._credentials = credentials
        # API key is optional - LiteLLM reads from env vars automatically
        if not self.api_key:
            logger.debug(
                'No API key provided to provider. LiteLLM will use environment variables.'
            )

    def _create_litellm_wrapper(self, model_name: str, **kwargs) -> LiteLLMWrapper:
        """
        Centralized LiteLLM wrapper creation.

        Handles model name prefixing and common configuration.
        Subclasses can override create_llm() for non-LiteLLM backends.
        """
        # Add provider prefix if not already present
        if self.LITELLM_PREFIX and not model_name.startswith(self.LITELLM_PREFIX):
            model_name = f'{self.LITELLM_PREFIX}{model_name}'

        temperature = kwargs.pop('temperature', 0.0)
        return LiteLLMWrapper(
            model=model_name,
            provider=self.LITELLM_PREFIX.rstrip('/') or 'openai',
            temperature=temperature,
            api_key=self.api_key,
            **kwargs
        )

    def create_llm(self, model_name: str, **kwargs) -> Any:
        """
        Create a language model client.

        Default implementation delegates to _create_litellm_wrapper().
        Override for non-LiteLLM backends (e.g., HuggingFace local inference).
        """
        return self._create_litellm_wrapper(model_name, **kwargs)

    @abstractmethod
    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        """
        Create an embedding model client.

        Must be implemented by subclasses - embeddings are provider-specific.
        """
        pass

    @staticmethod
    def estimate_cost(
        model_name: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """
        Estimate the cost of a call using the central estimator.
        This method is inherited by all subclasses.
        """
        return LLMCostEstimator.estimate(model_name, prompt_tokens, completion_tokens)


class LLMRegistry:
    """
    A factory and registry for LLM and embedding model providers.

    This class provides a unified interface for creating LLM and embedding clients
    from different providers using simple string identifiers or global settings.
    """

    _registry: Dict[str, Type[BaseProvider]] = {}

    def __init__(self, provider: Optional[str] = None, **credentials):
        """
        Initializes the registry. If a provider is specified, it configures for that provider.
        Otherwise, it remains flexible to serve requests based on global settings.

        Args:
            provider (Optional[str]): The name of the provider to lock this registry instance to.
            **credentials: Credentials required by the provider, such as 'api_key'.
        """
        self._provider_instance = None
        if provider:
            if provider not in self._registry:
                raise ValueError(
                    f"Provider '{provider}' is not registered. Available providers: {list(self._registry.keys())}"
                )
            provider_class = self._registry[provider]
            self._provider_instance = provider_class(**credentials)
            logger.debug(
                f"LLMRegistry initialized and locked to provider: '{provider}'"
            )
        else:
            logger.debug(
                'LLMRegistry initialized in flexible mode (will use global settings).'
            )

    def _get_provider_instance(self, provider_name: str, **credentials) -> BaseProvider:
        """Helper to get a provider instance on-the-fly."""
        if provider_name not in self._registry:
            raise ValueError(f"Provider '{provider_name}' is not registered.")
        return self._registry[provider_name](**credentials)

    @classmethod
    def register(cls, name: str):
        """A class method decorator to register a new provider."""

        def decorator(provider_class: Type[BaseProvider]):
            cls._registry[name] = provider_class
            logger.debug(f"Provider '{name}' registered.")
            return provider_class

        return decorator

    @classmethod
    def display(cls):
        """Display the LLM provider registry."""
        from axion.docs.display_registry import (
            create_llm_registry_display,
            prepare_llm_registry,
        )

        providers = cls._registry
        prepared_providers = prepare_llm_registry(providers)
        display = create_llm_registry_display()
        display.display(prepared_providers)

    def get_llm(
        self, model_name: Optional[str] = None, provider: Optional[str] = None, **kwargs
    ) -> Any:
        """
        Gets a language model instance from a provider.

        Uses the instance's locked provider if set, otherwise falls back to the
        globally configured `settings.llm_provider`.

        Args:
            model_name (Optional[str]): The model name to use. Defaults to settings.model_name.
            provider (Optional[str]): Override the provider for this specific call.
            **kwargs: Additional parameters for the LLM's constructor.

        Returns:
            An instance of a language model client.
        """
        if self._provider_instance and not provider:
            # Registry is locked to a provider - use provider's default first
            model_name = (
                model_name
                or getattr(self._provider_instance, 'DEFAULT_LLM_MODEL', None)
                or settings.llm_model_name
            )
            logger.debug(
                f"Using locked provider to create LLM client for model '{model_name}'"
            )
            return self._provider_instance.create_llm(model_name=model_name, **kwargs)

        # Flexible mode or provider override
        provider_name = provider or settings.llm_provider
        provider_instance = self._get_provider_instance(provider_name, **kwargs)
        model_name = (
            model_name
            or getattr(provider_instance, 'DEFAULT_LLM_MODEL', None)
            or settings.llm_model_name
        )
        logger.debug(
            f"Using '{provider_name}' provider to create LLM client for model '{model_name}'"
        )
        return provider_instance.create_llm(model_name=model_name, **kwargs)

    def get_embedding_model(
        self, model_name: Optional[str] = None, provider: Optional[str] = None, **kwargs
    ) -> Any:
        """
        Gets an embedding model instance from a provider.

        Uses the instance's locked provider if set, otherwise falls back to the
        globally configured `settings.embedding_provider`.

        Args:
            model_name (Optional[str]): The model name to use. Defaults to settings.embedding_model.
            provider (Optional[str]): Override the provider for this specific call.
            **kwargs: Additional parameters for the embedding model's constructor.

        Returns:
            An instance of an embedding model client.
        """
        if self._provider_instance and not provider:
            # Registry is locked to a provider - use provider's default first
            model_name = (
                model_name
                or getattr(self._provider_instance, 'DEFAULT_EMBEDDING_MODEL', None)
                or settings.embedding_model_name
            )
            logger.debug(
                f"Using locked provider to create embedding client for model '{model_name}'"
            )
            return self._provider_instance.create_embedding_model(
                model_name=model_name, **kwargs
            )

        # Flexible mode or provider override
        provider_name = provider or settings.embedding_provider
        provider_instance = self._get_provider_instance(provider_name, **kwargs)
        model_name = (
            model_name
            or getattr(provider_instance, 'DEFAULT_EMBEDDING_MODEL', None)
            or settings.embedding_model_name
        )
        logger.debug(
            f"Using '{provider_name}' provider to create embedding client for model '{model_name}'"
        )
        return provider_instance.create_embedding_model(model_name=model_name, **kwargs)

    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> float:
        """
        Estimates the cost of an LLM call based on token usage.

        The method determines the correct provider and model based on the same precedence
        rules as get_llm (direct arguments > locked instance > global settings).

        Args:
            prompt_tokens (int): The number of tokens in the prompt.
            completion_tokens (int): The number of tokens in the completion.
            model_name (Optional[str]): The model name to use for pricing. Defaults to settings.model_name.
            provider (Optional[str]): Override the provider for this specific call.

        Returns:
            float: The estimated cost in USD.
        """
        model_name = model_name or settings.llm_model_name

        if self._provider_instance and not provider:
            # Registry is locked to a provider
            logger.debug(
                f"Using locked provider to estimate cost for model '{model_name}'"
            )
            return self._provider_instance.estimate_cost(
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        # Flexible mode or provider override
        provider_name = provider or settings.llm_provider
        provider_instance = self._get_provider_instance(provider_name)
        logger.debug(
            f"Using '{provider_name}' provider to estimate cost for model '{model_name}'"
        )
        return provider_instance.estimate_cost(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @property
    def default_llm_model_name(self) -> str:
        """Default LLM model name"""
        return settings.llm_model_name

    @property
    def default_embedding_model_name(self) -> str:
        """Default Embedding model name"""
        return settings.embedding_model_name


@LLMRegistry.register('openai')
class OpenAIProvider(BaseProvider):
    """
    Provider for OpenAI models via LiteLLM.

    Supports all OpenAI models including GPT-4, GPT-4o, GPT-3.5, o1, o3, etc.
    Model names are passed directly to LiteLLM without prefixing.

    Example:
        >>> registry = LLMRegistry(provider='openai')
        >>> llm = registry.get_llm('gpt-4o')
        >>> llm = registry.get_llm('gpt-4o-mini')
        >>> llm = registry.get_llm('o1-preview')
    """

    LITELLM_PREFIX = ''  # OpenAI is LiteLLM's default, no prefix needed
    DEFAULT_LLM_MODEL = 'gpt-4o'
    DEFAULT_EMBEDDING_MODEL = 'text-embedding-ada-002'

    def __init__(self, api_key: Optional[str] = None, **credentials):
        """Initialize OpenAI provider with optional API key."""
        super().__init__(api_key=api_key or settings.openai_api_key, **credentials)

    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        """
        Create an embedding model for OpenAI.

        Uses LlamaIndex's OpenAI embedding for compatibility with existing code.
        """
        from llama_index.embeddings.openai import (
            OpenAIEmbedding as LlamaIndexOpenAIEmbedding,
        )

        return LlamaIndexOpenAIEmbedding(
            api_key=self.api_key, model=model_name, **kwargs
        )


@LLMRegistry.register('anthropic')
class AnthropicProvider(BaseProvider):
    """
    Provider for Anthropic Claude models via LiteLLM.

    Supports all Claude models. Model names are automatically prefixed with
    'anthropic/' for LiteLLM routing.

    Note: Anthropic does not provide embedding models. Use OpenAI or HuggingFace
    for embeddings when using Anthropic for LLM.

    Example:
        >>> registry = LLMRegistry(provider='anthropic')
        >>> llm = registry.get_llm('claude-sonnet-4-5-20250929')
        >>> llm = registry.get_llm('claude-opus-4-5-20251101')
    """

    LITELLM_PREFIX = 'anthropic/'
    SUPPORTS_EMBEDDINGS = False  # Anthropic does not provide embedding models
    DEFAULT_LLM_MODEL = 'claude-sonnet-4-5-20250929'
    DEFAULT_EMBEDDING_MODEL = None

    def __init__(self, api_key: Optional[str] = None, **credentials):
        """Initialize Anthropic provider with optional API key."""
        super().__init__(api_key=api_key or settings.anthropic_api_key, **credentials)

    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        """
        Anthropic does not provide embedding models.

        Raises:
            NotImplementedError: Always raised - use OpenAI or HuggingFace for embeddings.
        """
        raise NotImplementedError(
            'Anthropic does not provide embedding models. '
            'Use OpenAI or HuggingFace provider for embeddings.'
        )


@LLMRegistry.register('gemini')
class GeminiProvider(BaseProvider):
    """
    Provider for Google Gemini models via LiteLLM.

    Supports all Gemini models. Model names are automatically prefixed with
    'gemini/' for LiteLLM routing.

    Example:
        >>> registry = LLMRegistry(provider='gemini')
        >>> llm = registry.get_llm('gemini-1.5-pro')
        >>> llm = registry.get_llm('gemini-1.5-flash')
    """

    LITELLM_PREFIX = 'gemini/'
    DEFAULT_LLM_MODEL = 'gemini-2.0-flash'
    DEFAULT_EMBEDDING_MODEL = 'text-embedding-004'

    def __init__(self, api_key: Optional[str] = None, **credentials):
        """Initialize Gemini provider with optional API key."""
        super().__init__(api_key=api_key or settings.google_api_key, **credentials)

    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        """
        Create an embedding model for Google.

        Uses LlamaIndex's Gemini embedding for compatibility.
        """
        from llama_index.embeddings.gemini import GeminiEmbedding

        return GeminiEmbedding(
            api_key=self.api_key, model_name=model_name, **kwargs
        )


@LLMRegistry.register('vertex_ai')
class VertexAIProvider(BaseProvider):
    """
    Provider for Google Vertex AI models via LiteLLM.

    Supports Gemini models, Claude on Vertex, and legacy models (chat-bison, etc.).
    Requires GCP service account authentication.

    Authentication options:
    1. Environment variables: GOOGLE_APPLICATION_CREDENTIALS, VERTEXAI_PROJECT, VERTEXAI_LOCATION
    2. Direct parameters: vertex_credentials, vertex_project, vertex_location

    Example:
        >>> registry = LLMRegistry(
        ...     provider='vertex_ai',
        ...     vertex_project='my-gcp-project',
        ...     vertex_location='us-central1',
        ...     vertex_credentials='/path/to/service-account.json'
        ... )
        >>> llm = registry.get_llm('gemini-1.5-pro')
    """

    LITELLM_PREFIX = 'vertex_ai/'
    SUPPORTS_EMBEDDINGS = True
    DEFAULT_LLM_MODEL = 'gemini-2.0-flash'
    DEFAULT_EMBEDDING_MODEL = 'text-embedding-004'

    def __init__(
        self,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_credentials: Optional[str] = None,
        **credentials
    ):
        """Initialize Vertex AI provider with GCP configuration."""
        # Vertex AI uses service account auth, not API keys - remove if passed
        credentials.pop('api_key', None)
        super().__init__(api_key=None, **credentials)
        self.vertex_project = vertex_project or settings.vertex_project
        self.vertex_location = vertex_location or settings.vertex_location
        self.vertex_credentials = vertex_credentials or settings.vertex_credentials

    def _create_litellm_wrapper(self, model_name: str, **kwargs) -> LiteLLMWrapper:
        """Override to add Vertex AI specific parameters."""
        if self.LITELLM_PREFIX and not model_name.startswith(self.LITELLM_PREFIX):
            model_name = f'{self.LITELLM_PREFIX}{model_name}'

        temperature = kwargs.pop('temperature', 0.0)

        # Add Vertex AI specific kwargs
        vertex_kwargs = {}
        if self.vertex_project:
            vertex_kwargs['vertex_project'] = self.vertex_project
        if self.vertex_location:
            vertex_kwargs['vertex_location'] = self.vertex_location
        if self.vertex_credentials:
            vertex_kwargs['vertex_credentials'] = self.vertex_credentials

        return LiteLLMWrapper(
            model=model_name,
            provider='vertex_ai',
            temperature=temperature,
            api_key=None,  # Vertex AI doesn't use API keys
            **{**vertex_kwargs, **kwargs}
        )

    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        """
        Create an embedding model for Vertex AI.

        Supported models: text-embedding-004, textembedding-gecko, etc.
        """
        from llama_index.embeddings.vertex import VertexTextEmbedding

        return VertexTextEmbedding(
            model_name=model_name,
            project=self.vertex_project,
            location=self.vertex_location,
            credentials_path=self.vertex_credentials,
            **kwargs
        )


@LLMRegistry.register('huggingface')
class HuggingFaceProvider(BaseProvider):
    """
    Provider for HuggingFace Hub models.

    Uses LlamaIndex for local inference (not LiteLLM).
    Requires torch and transformers to be installed.

    Example:
        >>> registry = LLMRegistry(provider='huggingface')
        >>> llm = registry.get_llm('meta-llama/Llama-2-7b-chat-hf')
    """

    DEFAULT_LLM_MODEL = None  # HuggingFace requires explicit model specification
    DEFAULT_EMBEDDING_MODEL = None

    def __init__(self, api_key: Optional[str] = None, **credentials):
        """Initialize HuggingFace provider."""
        # HuggingFace may use HF_TOKEN for gated models
        super().__init__(api_key=api_key, **credentials)

    def create_llm(self, model_name: str, **kwargs) -> Any:
        """
        Create an LLM client for HuggingFace models.

        Overrides base implementation to use LlamaIndex for local inference
        instead of LiteLLM.

        Args:
            model_name: HuggingFace model identifier (e.g., 'meta-llama/Llama-2-7b-chat-hf')
            **kwargs: Additional configuration (device_map, torch_dtype, etc.)

        Returns:
            LlamaIndex HuggingFaceLLM instance
        """
        from llama_index.llms.huggingface import HuggingFaceLLM

        return HuggingFaceLLM(model_name=model_name, **kwargs)

    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        """
        Create an embedding model for HuggingFace.

        Args:
            model_name: HuggingFace embedding model identifier
            **kwargs: Additional configuration

        Returns:
            LlamaIndex HuggingFaceEmbedding instance
        """
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=model_name, **kwargs)
