from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from axion._core.environment import settings
from axion._core.logging import get_logger

logger = get_logger(__name__)


class LLMCostEstimator:
    """
    This class provides model-specific pricing for popular LLMs such as GPT-4o, GPT-3.5-Turbo, and others.
    It calculates the estimated cost of a given request based on the number of prompt and completion tokens,
    using pricing tiers published by OpenAI (or defined manually for internal models).

    Example:
    ```
     cost = LLMCostEstimator.estimate("gpt-4o", prompt_tokens=1500, completion_tokens=500)
     print(f"Estimated cost: ${cost}")
    ```
    """

    MODEL_PRICING: Dict[str, Dict[str, float]] = {
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
            model_name (str): The name of the LLM model (e.g., 'gpt-4o', 'gpt-3.5-turbo').
            prompt_tokens (int): The number of tokens sent in the prompt.
            completion_tokens (int): The number of tokens generated in the response.
            round_level (int): Round level for precision (default 8)

        Returns:
            float: The estimated total cost of the call, in USD.

        Notes:
            - Prices are per token (already calculated per million in MODEL_PRICING).
            - If the model name is unrecognized, default pricing (gpt-4o) is applied.
        """
        model_key = model_name.lower()
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
            model_name (str): The name of the LLM model

        Returns:
            Dict[str, float]: Dictionary with 'input' and 'output' pricing per token
        """
        model_key = model_name.lower()
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


class BaseProvider(ABC):
    """
    Abstract base class for all providers. It defines the contract that
    every provider must follow.
    """

    def __init__(self, api_key: Optional[str] = None, **credentials):
        """Initializes the provider, primarily by storing the API key."""
        self.api_key = api_key or settings.gateway_api_key
        if not self.api_key:
            raise ValueError(
                'API key must be provided either via settings.gateway_api_key, '
                'as an argument, or the `GATEWAY_API_KEY` environment variable.'
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


@LLMRegistry.register('llm_gateway')
class LLMGatewayProvider(BaseProvider):
    """Provider for connecting to a local proxy/gateway server via LlamaIndex."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **credentials
    ):
        super().__init__(api_key=api_key, **credentials)
        # LlamaIndex uses 'api_base' for the proxy URL.
        self.api_base = base_url or settings.api_base_url

    def create_llm(self, model_name: str, **kwargs) -> Any:
        from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

        # Pass api_base to the LlamaIndex client to route through the proxy.
        return LlamaIndexOpenAI(
            api_key=self.api_key,
            model=model_name,
            api_base=self.api_base,
            max_retries=0,  # We are custom handling retries
            timeout=120.0,
            **kwargs,
        )

    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        from llama_index.embeddings.openai import (
            OpenAIEmbedding as LlamaIndexOpenAIEmbedding,
        )

        # Pass api_base to the LlamaIndex client to route through the proxy.
        return LlamaIndexOpenAIEmbedding(
            api_key=self.api_key,
            model=model_name,
            api_base=self.api_base,
            max_retries=0,  # We are custom handling retries
            **kwargs,
        )


@LLMRegistry.register('llama_index')
class LlamaIndexProvider(BaseProvider):
    """Provider for creating LlamaIndex-specific OpenAI clients."""

    def create_llm(self, model_name: str, **kwargs) -> Any:
        from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

        return LlamaIndexOpenAI(api_key=self.api_key, model=model_name, **kwargs)

    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        from llama_index.embeddings.openai import (
            OpenAIEmbedding as LlamaIndexOpenAIEmbedding,
        )

        return LlamaIndexOpenAIEmbedding(
            api_key=self.api_key, model=model_name, **kwargs
        )


@LLMRegistry.register('huggingface')
class HuggingFaceProvider(BaseProvider):
    """Provider for HuggingFace Hub models."""

    def create_llm(self, model_name: str, **kwargs) -> Any:
        from llama_index.llms.huggingface import HuggingFaceLLM

        # Note: May require torch, transformers, etc. to be installed.
        # Additional kwargs like device_map="auto" can be passed.
        return HuggingFaceLLM(model_name=model_name, **kwargs)

    def create_embedding_model(self, model_name: str, **kwargs) -> Any:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=model_name, **kwargs)
