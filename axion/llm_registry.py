from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from axion._core.environment import settings
from axion._core.logging import get_logger

logger = get_logger(__name__)


class LLMCostEstimator:
    """
    This class provides model-specific pricing for popular LLMs including OpenAI, Anthropic, and Google models.
    It calculates the estimated cost of a given request based on the number of prompt and completion tokens.

    Note: When using LiteLLM in the handler, `litellm.completion_cost()` is preferred for accurate
    real-time pricing. This class serves as a fallback for custom/internal models.

    Example:
    ```
     cost = LLMCostEstimator.estimate("gpt-4o", prompt_tokens=1500, completion_tokens=500)
     print(f"Estimated cost: ${cost}")
    ```
    """

    MODEL_PRICING: Dict[str, Dict[str, float]] = {
        # OpenAI Models
        'gpt-4o-mini': {'input': 0.150 / 1e6, 'output': 0.600 / 1e6},
        'gpt-4o': {'input': 2.50 / 1e6, 'output': 10.00 / 1e6},
        'gpt-4-turbo': {'input': 10.00 / 1e6, 'output': 30.00 / 1e6},
        'gpt-4-turbo-preview': {'input': 10.00 / 1e6, 'output': 30.00 / 1e6},
        'gpt-4-0125-preview': {'input': 10.00 / 1e6, 'output': 30.00 / 1e6},
        'gpt-4-1106-preview': {'input': 10.00 / 1e6, 'output': 30.00 / 1e6},
        'gpt-4': {'input': 30.00 / 1e6, 'output': 60.00 / 1e6},
        'gpt-4-32k': {'input': 60.00 / 1e6, 'output': 120.00 / 1e6},
        'gpt-3.5-turbo-1106': {'input': 1.00 / 1e6, 'output': 2.00 / 1e6},
        'gpt-3.5-turbo': {'input': 0.50 / 1e6, 'output': 1.50 / 1e6},
        'gpt-3.5-turbo-16k': {'input': 3.00 / 1e6, 'output': 4.00 / 1e6},
        'gpt-3.5-turbo-0125': {'input': 0.50 / 1e6, 'output': 1.50 / 1e6},
        'gpt-3.5-turbo-instruct': {'input': 1.50 / 1e6, 'output': 2.00 / 1e6},
        'o1': {'input': 15.00 / 1e6, 'output': 60.00 / 1e6},
        'o1-preview': {'input': 15.00 / 1e6, 'output': 60.00 / 1e6},
        'o1-2024-12-17': {'input': 15.00 / 1e6, 'output': 60.00 / 1e6},
        'o3-mini': {'input': 1.10 / 1e6, 'output': 4.40 / 1e6},
        'o3-mini-2025-01-31': {'input': 1.10 / 1e6, 'output': 4.40 / 1e6},
        'o4-mini': {'input': 1.10 / 1e6, 'output': 4.40 / 1e6},
        'gpt-4.1': {'input': 2.00 / 1e6, 'output': 8.00 / 1e6},
        'gpt-4.1-mini': {'input': 0.4 / 1e6, 'output': 1.60 / 1e6},
        'gpt-4.1-nano': {'input': 0.1 / 1e6, 'output': 0.4 / 1e6},
        'gpt-4.5-preview': {'input': 75.00 / 1e6, 'output': 150.00 / 1e6},
        'gpt-5': {'input': 1.25 / 1e6, 'output': 10.00 / 1e6},
        'gpt-5-2025-08-07': {'input': 1.25 / 1e6, 'output': 10.00 / 1e6},
        'gpt-5-mini': {'input': 0.25 / 1e6, 'output': 2.00 / 1e6},
        'gpt-5-mini-2025-08-07': {'input': 0.25 / 1e6, 'output': 2.00 / 1e6},
        'gpt-5-nano': {'input': 0.05 / 1e6, 'output': 0.40 / 1e6},
        'gpt-5-nano-2025-08-07': {'input': 0.05 / 1e6, 'output': 0.40 / 1e6},
        'gpt-5-chat-latest': {'input': 1.25 / 1e6, 'output': 10.00 / 1e6},
        # Anthropic Claude Models (use with "anthropic/" prefix in LiteLLM)
        'claude-3-5-sonnet-20241022': {'input': 3.00 / 1e6, 'output': 15.00 / 1e6},
        'claude-3-5-sonnet-latest': {'input': 3.00 / 1e6, 'output': 15.00 / 1e6},
        'claude-3-5-haiku-20241022': {'input': 0.80 / 1e6, 'output': 4.00 / 1e6},
        'claude-3-opus-20240229': {'input': 15.00 / 1e6, 'output': 75.00 / 1e6},
        'claude-3-sonnet-20240229': {'input': 3.00 / 1e6, 'output': 15.00 / 1e6},
        'claude-3-haiku-20240307': {'input': 0.25 / 1e6, 'output': 1.25 / 1e6},
        # Google Gemini Models (use with "gemini/" prefix in LiteLLM)
        'gemini-1.5-pro': {'input': 1.25 / 1e6, 'output': 5.00 / 1e6},
        'gemini-1.5-pro-latest': {'input': 1.25 / 1e6, 'output': 5.00 / 1e6},
        'gemini-1.5-flash': {'input': 0.075 / 1e6, 'output': 0.30 / 1e6},
        'gemini-1.5-flash-latest': {'input': 0.075 / 1e6, 'output': 0.30 / 1e6},
        'gemini-2.0-flash-exp': {'input': 0.10 / 1e6, 'output': 0.40 / 1e6},
        'gemini-pro': {'input': 0.50 / 1e6, 'output': 1.50 / 1e6},
    }

    DEFAULT_PRICING: Dict[str, float] = {
        'input': 2.50 / 1e6,
        'output': 10.00 / 1e6,
    }  # Default to gpt-4o pricing

    @classmethod
    def estimate(
        cls,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        round_level: int = 8,
    ) -> float:
        """
        Estimate the cost of an LLM call for a given model based on token usage.

        Args:
            model_name (str): The name of the LLM model. Supports both plain names
                (e.g., 'gpt-4o') and LiteLLM prefixed names (e.g., 'anthropic/claude-3-5-sonnet').
            prompt_tokens (int): The number of tokens sent in the prompt.
            completion_tokens (int): The number of tokens generated in the response.
            round_level (int): Round level for precision (default 8)

        Returns:
            float: The estimated total cost of the call, in USD.

        Notes:
            - Prices are per token (already calculated per million in MODEL_PRICING).
            - If the model name is unrecognized, default pricing (gpt-4o) is applied.
            - LiteLLM prefixed model names (e.g., 'anthropic/claude-3-5-sonnet') are
              automatically stripped to match pricing keys.
        """
        model_key = model_name.lower()

        # Strip LiteLLM provider prefix if present (e.g., "anthropic/claude-3" -> "claude-3")
        if '/' in model_key:
            model_key = model_key.split('/', 1)[1]

        pricing = cls.MODEL_PRICING.get(model_key, cls.DEFAULT_PRICING)

        prompt_cost = prompt_tokens * pricing['input']
        completion_cost = completion_tokens * pricing['output']

        return round(prompt_cost + completion_cost, round_level)

    @classmethod
    def calculate_cost(
        cls, model_name: str, input_tokens: int, output_tokens: int
    ) -> float:
        """
        Alternative method name for backward compatibility.

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
        Get the pricing information for a specific model.

        Args:
            model_name (str): The name of the LLM model (supports LiteLLM prefixed names)

        Returns:
            Dict[str, float]: Dictionary with 'input' and 'output' pricing per token
        """
        model_key = model_name.lower()
        # Strip LiteLLM provider prefix if present
        if '/' in model_key:
            model_key = model_key.split('/', 1)[1]
        return cls.MODEL_PRICING.get(model_key, cls.DEFAULT_PRICING).copy()

    @classmethod
    def list_supported_models(cls) -> List[str]:
        """
        List all supported models with pricing information.

        Returns:
            List[str]: List of supported model names
        """
        return list(cls.MODEL_PRICING.keys())

    @classmethod
    def add_custom_model(
        cls, model: str, input_price_per_token: float, output_price_per_token: float
    ) -> None:
        """
        Add custom model pricing.

        Args:
            model (str): The name of the custom model
            input_price_per_token (float): Price per input token in USD
            output_price_per_token (float): Price per output token in USD
        """
        cls.MODEL_PRICING[model.lower()] = {
            'input': input_price_per_token,
            'output': output_price_per_token,
        }


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

        return CompletionResponse(
            text=response.choices[0].message.content,
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

        return CompletionResponse(
            text=response.choices[0].message.content,
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

        return CompletionResponse(
            text=response.choices[0].message.content,
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

        return CompletionResponse(
            text=response.choices[0].message.content,
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

    Note: When using LiteLLM in the handler, API keys are read from environment
    variables automatically (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY).
    """

    def __init__(self, api_key: Optional[str] = None, **credentials):
        """Initializes the provider, storing the API key if provided."""
        self.api_key = api_key or settings.openai_api_key
        # API key is optional - LiteLLM reads from env vars automatically
        if not self.api_key:
            logger.debug(
                'No API key provided to provider. LiteLLM will use environment variables.'
            )

    @abstractmethod
    def create_llm(self, model: str, **kwargs) -> Any:
        """Abstract method to create a language model client."""
        pass

    @abstractmethod
    def create_embedding_model(self, model: str, **kwargs) -> Any:
        """Abstract method to create an embedding model client."""
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
        model_name = model_name or settings.llm_model_name

        if self._provider_instance and not provider:
            # Registry is locked to a provider
            logger.debug(
                f"Using locked provider to create LLM client for model '{model_name}'"
            )
            return self._provider_instance.create_llm(model_name=model_name, **kwargs)

        # Flexible mode or provider override
        provider_name = provider or settings.llm_provider
        provider_instance = self._get_provider_instance(provider_name, **kwargs)
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
        model_name = model_name or settings.embedding_model_name

        if self._provider_instance and not provider:
            # Registry is locked to a provider
            logger.debug(
                f"Using locked provider to create embedding client for model '{model_name}'"
            )
            return self._provider_instance.create_embedding_model(
                model_name=model_name, **kwargs
            )

        # Flexible mode or provider override
        provider_name = provider or settings.embedding_provider
        provider_instance = self._get_provider_instance(provider_name, **kwargs)
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

    def __init__(self, api_key: Optional[str] = None, **credentials):
        """Initialize OpenAI provider with optional API key."""
        # Use OpenAI-specific API key from settings
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            logger.debug(
                'No OpenAI API key provided. LiteLLM will use OPENAI_API_KEY env var.'
            )

    def create_llm(self, model_name: str, **kwargs) -> LiteLLMWrapper:
        """
        Create an LLM wrapper for OpenAI models.

        Args:
            model_name: OpenAI model name (e.g., 'gpt-4o', 'gpt-4o-mini', 'o1')
            **kwargs: Additional configuration (temperature, etc.)

        Returns:
            LiteLLMWrapper configured for the specified OpenAI model
        """
        temperature = kwargs.pop('temperature', 0.0)
        return LiteLLMWrapper(
            model=model_name,
            provider='openai',
            temperature=temperature,
            api_key=self.api_key,
            **kwargs
        )

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

    Example:
        >>> registry = LLMRegistry(provider='anthropic')
        >>> llm = registry.get_llm('claude-3-5-sonnet-20241022')
        >>> llm = registry.get_llm('claude-3-opus-20240229')
    """

    def __init__(self, api_key: Optional[str] = None, **credentials):
        """Initialize Anthropic provider with optional API key."""
        # Use Anthropic-specific API key from settings
        self.api_key = api_key or settings.anthropic_api_key
        if not self.api_key:
            logger.debug(
                'No Anthropic API key provided. LiteLLM will use ANTHROPIC_API_KEY env var.'
            )

    def create_llm(self, model_name: str, **kwargs) -> LiteLLMWrapper:
        """
        Create an LLM wrapper for Anthropic Claude models.

        Args:
            model_name: Claude model name (e.g., 'claude-3-5-sonnet-20241022')
                        Will be prefixed with 'anthropic/' for LiteLLM
            **kwargs: Additional configuration (temperature, etc.)

        Returns:
            LiteLLMWrapper configured for the specified Claude model
        """
        # Add anthropic/ prefix if not already present
        if not model_name.startswith('anthropic/'):
            model_name = f'anthropic/{model_name}'

        temperature = kwargs.pop('temperature', 0.0)
        return LiteLLMWrapper(
            model=model_name,
            provider='anthropic',
            temperature=temperature,
            api_key=self.api_key,
            **kwargs
        )

    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        """
        Create an embedding model for Anthropic.

        Note: Anthropic does not provide embedding models directly.
        Falls back to OpenAI embeddings or raises an error.
        """
        logger.warning(
            'Anthropic does not provide embedding models. '
            'Consider using OpenAI or HuggingFace embeddings instead.'
        )
        # Fall back to OpenAI embeddings as a reasonable default
        from llama_index.embeddings.openai import (
            OpenAIEmbedding as LlamaIndexOpenAIEmbedding,
        )

        return LlamaIndexOpenAIEmbedding(model=model_name, **kwargs)


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

    def __init__(self, api_key: Optional[str] = None, **credentials):
        """Initialize Gemini provider with optional API key."""
        # Use Google-specific API key from settings
        self.api_key = api_key or settings.google_api_key
        if not self.api_key:
            logger.debug(
                'No Google API key provided. LiteLLM will use GOOGLE_API_KEY env var.'
            )

    def create_llm(self, model_name: str, **kwargs) -> LiteLLMWrapper:
        """
        Create an LLM wrapper for Google Gemini models.

        Args:
            model_name: Gemini model name (e.g., 'gemini-1.5-pro', 'gemini-1.5-flash')
                        Will be prefixed with 'gemini/' for LiteLLM
            **kwargs: Additional configuration (temperature, etc.)

        Returns:
            LiteLLMWrapper configured for the specified Gemini model
        """
        # Add gemini/ prefix if not already present
        if not model_name.startswith('gemini/'):
            model_name = f'gemini/{model_name}'

        temperature = kwargs.pop('temperature', 0.0)
        return LiteLLMWrapper(
            model=model_name,
            provider='gemini',
            temperature=temperature,
            api_key=self.api_key,
            **kwargs
        )

    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        """
        Create an embedding model for Google.

        Uses LlamaIndex's Gemini embedding for compatibility.
        """
        from llama_index.embeddings.gemini import GeminiEmbedding

        return GeminiEmbedding(
            api_key=self.api_key, model_name=model_name, **kwargs
        )


@LLMRegistry.register('huggingface')
class HuggingFaceProvider(BaseProvider):
    """
    Provider for HuggingFace Hub models.

    Supports local HuggingFace models via LlamaIndex integration.
    Requires torch and transformers to be installed.

    Example:
        >>> registry = LLMRegistry(provider='huggingface')
        >>> llm = registry.get_llm('meta-llama/Llama-2-7b-chat-hf')
    """

    def __init__(self, api_key: Optional[str] = None, **credentials):
        """Initialize HuggingFace provider."""
        # HuggingFace may use HF_TOKEN for gated models
        self.api_key = api_key
        if not self.api_key:
            logger.debug(
                'No HuggingFace token provided. Set HF_TOKEN env var for gated models.'
            )

    def create_llm(self, model_name: str, **kwargs) -> Any:
        """
        Create an LLM client for HuggingFace models.

        Args:
            model_name: HuggingFace model identifier (e.g., 'meta-llama/Llama-2-7b-chat-hf')
            **kwargs: Additional configuration (device_map, torch_dtype, etc.)

        Returns:
            LlamaIndex HuggingFaceLLM instance
        """
        from llama_index.llms.huggingface import HuggingFaceLLM

        # Note: May require torch, transformers, etc. to be installed.
        # Additional kwargs like device_map="auto" can be passed.
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
