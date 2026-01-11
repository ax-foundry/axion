import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
from axion._core.environment import settings
from axion._core.logging import get_logger
from axion._core.metadata.schema import ToolMetadata
from axion._core.networking import RateLimitInfo
from axion._core.schema import Callbacks, LLMRunnable, PromptValue
from axion._core.tracing import init_tracer, trace
from axion._core.tracing.handlers import BaseTraceHandler
from axion._handlers.validation import Validation
from axion.llm_registry import LLMCostEstimator, LLMGatewayProvider
from llama_index.core.llms import CompletionResponse
from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

try:
    from ragas.llms import BaseRagasLLM
except ImportError:
    BaseRagasLLM = object  # Fallback if ragas is not installed
    import warnings

    warnings.warn('ragas is not installed. RagasLLM will not be fully functional.')

logger = get_logger(__name__)


@dataclass
class RunConfig:
    timeout: int = 180
    max_retries: int = 10
    max_wait: int = 60
    max_workers: int = 16
    exception_types: Union[
        Type[BaseException],
        Tuple[Type[BaseException], ...],
    ] = (Exception,)
    log_tenacity: bool = False
    seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(seed=self.seed)


class CacheInterface(ABC):
    """Abstract base class defining the interface for cache implementations."""

    @abstractmethod
    def get(self, key: str) -> Any:
        pass

    @abstractmethod
    def set(self, key: str, value) -> None:
        pass

    @abstractmethod
    def has_key(self, key: str) -> bool:
        pass

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.is_instance_schema(cls),  # The validator function
        )


class Generation(BaseModel):
    """A single text generation output"""

    text: str
    generation_info: Optional[Dict[str, Any]] = None
    type: Literal['Generation'] = 'Generation'


class RagasCompletionResponse(BaseModel):
    """
    Represents a result from an LLM generation.
    Follows the structure used by Ragas for consistency.
    """

    generations: List[List[Generation]]
    model_config = ConfigDict(extra='allow')


class RagasLLM(BaseRagasLLM):
    _cost_estimate: float = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        _openai_api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        llm: Optional[LLMRunnable] = None,
        temperature: float = 0,
        run_config: Optional[RunConfig] = None,
        multiple_completion_supported: bool = False,
        cache: Optional[CacheInterface] = None,
        tracer: Optional[BaseTraceHandler] = None,
        rate_limit_buffer: float = 1.0,
        max_retries: int = 5,
        **kwargs,
    ):
        """
        Initialize RagasLLM.

        Args:
            llm: Pre-configured LLMGatewayLLM instance
            run_config: Ragas run configuration
            multiple_completion_supported: Whether model supports multiple completions
            cache: Optional cache interface
            rate_limit_buffer: Buffer time to add to rate limit resets (seconds)
            max_retries: Max Retries
            **kwargs: Additional parameters
        """
        if run_config is None:
            run_config = RunConfig()

        self.kwargs = kwargs

        super().__init__(
            run_config=run_config,
            multiple_completion_supported=multiple_completion_supported,
            cache=cache,
        )

        self.model_name = llm.model if llm else model_name or settings.llm_model_name

        self.base_url = base_url
        self._openai_api_key = _openai_api_key
        self.temperature = temperature
        self.rate_limit_buffer = rate_limit_buffer
        self.max_retries = max_retries
        self.llm = llm or self.get_model()
        Validation.validate_llm_model(self.llm)
        self.temperature = getattr(llm, 'temperature', 1e-8)
        self.model = getattr(llm, 'model', 'unknown')

        tool_metadata = ToolMetadata(
            name=f'{self.__class__.__name__.lower()}',
            description=f'Ragas LLMGateway Evaluation ({self.__class__.__name__}) Model',
            owner='AXION',
            version='1.0.0',
        )

        self.tracer = init_tracer('llm', tool_metadata, tracer)

    def get_model(self):
        provider = LLMGatewayProvider(self._openai_api_key, self.base_url)
        return provider.create_llm(self.model_name, temperature=self.temperature)

    @trace(name='prompt_to_string', capture_args=True, capture_result=True)
    def _prompt_to_string(self, prompt: PromptValue) -> str:
        """
        Convert a PromptValue to a string.

        Args:
            prompt: The prompt value

        Returns:
            String representation of the prompt
        """
        if isinstance(prompt, str):
            return prompt
        elif hasattr(prompt, 'to_string'):
            return prompt.to_string()
        else:
            return str(prompt)

    @trace(name='extract_response')
    def _extract_response(
        self,
        response: Union[CompletionResponse, Dict[str, Any]],
    ) -> RagasCompletionResponse:
        """
        Extract the generated text from a CompletionResponse or dictionary-based response.
        """
        if isinstance(response, CompletionResponse):
            # LlamaIndex CompletionResponse output (Proxy Supported)
            generations = [[Generation(text=response.text)]]
        elif isinstance(response, dict) and response.get('generations'):
            # Salesforce API output
            generations = [response['generations']]
        else:
            # Fallback
            generations = [[]]

        return RagasCompletionResponse(generations=generations)

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """
        Check if the error is a rate limit (429) error.

        Handles both direct 429 errors and wrapped errors like:
        - Error code: 500 - {'detail': '500: 429: {...}'}
        """
        error_str = str(error).lower()
        # Check for direct 429 or rate limit mentions
        if '429' in error_str or 'rate limit' in error_str:
            return True
        # Check for wrapped 500 errors containing 429
        if '500' in error_str and (
            '429' in error_str
            or 'rate_limit' in error_str
            or 'llm_gatekeeper' in error_str
        ):
            return True
        return False

    def _handle_rate_limit_with_retry(
        self, func, *args, max_retries: int = None, **kwargs
    ):
        """
        Execute a function with rate-limit aware retry logic.

        This provides the same intelligent retry behavior as LLMHandler and DeepEvalLLM,
        but for synchronous Ragas calls.
        """

        if max_retries is None:
            max_retries = getattr(self.run_config, 'max_retries', self.max_retries)

        last_exception = None

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt == max_retries - 1:
                    # Final attempt failed
                    logger.error(
                        f'❌ All {max_retries} attempts failed. Final error: {str(e)}'
                    )
                    raise

                # Check if this is a rate limit error
                if self._is_rate_limit_error(e):
                    rate_info = RateLimitInfo.from_error(str(e))

                    if rate_info:
                        wait_time = rate_info.get_wait_time(self.rate_limit_buffer)
                        max_wait = 120.0  # Cap at 2 minutes
                        actual_wait = min(wait_time, max_wait)

                        logger.warning(
                            f'⏱️  Rate limit hit: {rate_info}. '
                            f'Waiting {actual_wait:.1f}s for reset (attempt {attempt + 1}/{max_retries})'
                        )

                        time.sleep(actual_wait)
                    else:
                        # Rate limit error but couldn't parse - use exponential backoff
                        wait_time = 1.0 * (2**attempt)
                        wait_time = min(wait_time, 10.0)

                        logger.warning(
                            f'⏱️  Rate limit error (could not parse details). '
                            f'Waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})'
                        )

                        time.sleep(wait_time)
                else:
                    # Non-rate-limit error - use exponential backoff
                    wait_time = 1.0 * (2**attempt)
                    wait_time = min(wait_time, 10.0)

                    logger.warning(
                        f'⏱️  Retrying with exponential backoff: {wait_time:.1f}s '
                        f'(attempt {attempt + 1}/{max_retries})'
                    )

                    time.sleep(wait_time)

        # Should never reach here, but just in case
        raise last_exception

    async def _ahandle_rate_limit_with_retry(
        self, async_func, *args, max_retries: int = None, **kwargs
    ):
        """
        Execute an async function with rate-limit aware retry logic.

        This provides the same intelligent retry behavior as LLMHandler and DeepEvalLLM,
        but for asynchronous Ragas calls.
        """
        import asyncio

        if max_retries is None:
            max_retries = getattr(self.run_config, 'max_retries', self.max_retries)

        last_exception = None

        for attempt in range(max_retries):
            try:
                return await async_func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt == max_retries - 1:
                    # Final attempt failed
                    logger.error(
                        f'❌ All {max_retries} attempts failed. Final error: {str(e)}'
                    )
                    raise

                # Check if this is a rate limit error
                if self._is_rate_limit_error(e):
                    rate_info = RateLimitInfo.from_error(str(e))

                    if rate_info:
                        wait_time = rate_info.get_wait_time(self.rate_limit_buffer)
                        max_wait = 120.0  # Cap at 2 minutes
                        actual_wait = min(wait_time, max_wait)

                        logger.warning(
                            f'⏱️  Rate limit hit: {rate_info}. '
                            f'Waiting {actual_wait:.1f}s for reset (attempt {attempt + 1}/{max_retries})'
                        )

                        await asyncio.sleep(actual_wait)
                    else:
                        # Rate limit error but couldn't parse - use exponential backoff
                        wait_time = 1.0 * (2**attempt)
                        wait_time = min(wait_time, 10.0)

                        logger.warning(
                            f'⏱️  Rate limit error (could not parse details). '
                            f'Waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})'
                        )

                        await asyncio.sleep(wait_time)
                else:
                    # Non-rate-limit error - use exponential backoff
                    wait_time = 1.0 * (2**attempt)
                    wait_time = min(wait_time, 10.0)

                    logger.warning(
                        f'⏱️  Retrying with exponential backoff: {wait_time:.1f}s '
                        f'(attempt {attempt + 1}/{max_retries})'
                    )

                    await asyncio.sleep(wait_time)

        # Should never reach here, but just in case
        raise last_exception

    @staticmethod
    def check_args(
        n: int,
        temperature: float,
        stop: Optional[List[str]],
    ) -> Dict[str, Any]:
        return {
            'n': n,
            'temperature': temperature,
            'stop': stop,
        }

    @trace(name='generate_text', capture_args=True, capture_result=True)
    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> CompletionResponse:
        """
        Generate text using the model with rate-limit aware retry logic.

        Args:
            prompt: The prompt to send to the model
            n: Number of completions to generate
            temperature: Sampling temperature
            stop: List of stop sequences
            callbacks: Optional callbacks

        Returns:
            CompletionResponse object
        """
        if temperature is not None:
            self.temperature = temperature
            self.llm._temperature = temperature
        kwargs = self.check_args(n, temperature, stop)
        prompt_str = self._prompt_to_string(prompt)

        # Wrap the LLM call with rate-limit aware retry
        def _call_llm():
            return self.llm.complete(prompt_str, **kwargs)

        response = self._handle_rate_limit_with_retry(_call_llm)

        self._cost_estimate = LLMCostEstimator.estimate(
            model_name=self.model_name,
            prompt_tokens=response.raw.usage.prompt_tokens,
            completion_tokens=response.raw.usage.completion_tokens,
        )

        return self._extract_response(response)

    @trace(name='generate', capture_args=True, capture_result=True)
    async def generate(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> RagasCompletionResponse:
        """
        Generate text using the model.

        Args:
            prompt: The prompt to send to the model
            n: Number of completions to generate
            temperature: Sampling temperature
            stop: List of stop sequences
            callbacks: Optional callbacks

        Returns:
            CompletionResponse object
        """
        return self.generate_text(
            prompt=prompt,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )

    @trace(name='generate_text', capture_args=True, capture_result=True)
    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs,
    ) -> RagasCompletionResponse:
        """
        Asynchronously generate text using the model with rate-limit aware retry logic.

        Args:
            prompt: The prompt to send to the model
            n: Number of completions to generate
            temperature: Sampling temperature
            stop: List of stop sequences
            callbacks: Optional callbacks

        Returns:
            CompletionResponse object
        """
        # Use default temperature if not provided
        if temperature is None:
            temperature = self.get_temperature(n)

        self.temperature = temperature
        self.llm._temperature = temperature

        prompt_str = self._prompt_to_string(prompt)
        kwargs = self.check_args(n, temperature, stop)

        # Wrap the async LLM call with rate-limit aware retry
        async def _call_llm():
            return await self.llm.acomplete(prompt_str, **kwargs)

        response = await self._ahandle_rate_limit_with_retry(_call_llm)

        self._cost_estimate = LLMCostEstimator.estimate(
            model_name=self.model_name,
            prompt_tokens=response.raw.usage.prompt_tokens,
            completion_tokens=response.raw.usage.completion_tokens,
        )

        return self._extract_response(response)

    def is_finished(self, response: RagasCompletionResponse) -> bool:
        """
        Check if the response is complete.
        """
        return True

    def get_temperature(self, n: int) -> float:
        """
        Get the temperature to use for n completions.
        """
        return self.temperature

    @property
    def cost_estimate(self):
        """
        Cost Estimate Per Run. Cost is calculated per prompt
        tokens and completion tokens on a model basis.
        """
        return self._cost_estimate
