"""Lightweight RAGAS LLM wrapper using LiteLLM with cost tracking."""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import litellm
import numpy as np
from pydantic import BaseModel, ConfigDict

from axion._core.logging import get_logger
from axion._core.tracing import trace

logger = get_logger(__name__)

try:
    from ragas.llms import BaseRagasLLM
except ImportError:
    BaseRagasLLM = object
    import warnings

    warnings.warn('ragas is not installed. LiteLLMRagas will not be fully functional.')


@dataclass
class RunConfig:
    """Configuration for Ragas run behavior."""

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


class Generation(BaseModel):
    """A single text generation output."""

    text: str
    generation_info: Optional[Dict[str, Any]] = None


class LLMResult(BaseModel):
    """Result from LLM generation."""

    generations: List[List[Generation]]
    model_config = ConfigDict(extra='allow')


class LiteLLMRagas(BaseRagasLLM):
    """
    RAGAS-compatible LLM wrapper using LiteLLM with cost tracking.

    Accumulates costs across multiple LLM calls during metric evaluation.
    Uses LiteLLM's native cost tracking via response._hidden_params["response_cost"].

    Example:
        >>> from axion.integrations.models.litellm_ragas import LiteLLMRagas
        >>> from ragas.metrics.collections import Faithfulness
        >>>
        >>> llm = LiteLLMRagas(model='gpt-4o')
        >>> metric = Faithfulness(llm=llm)
        >>> # After evaluation:
        >>> print(metric.llm.cost_estimate)
    """

    def __init__(
        self,
        model: str = 'gpt-4o',
        temperature: float = 0.0,
        run_config: Optional[RunConfig] = None,
        multiple_completion_supported: bool = False,
        cache: Optional[Any] = None,
        rate_limit_buffer: float = 1.0,
        max_retries: int = 5,
    ):
        """
        Initialize the LiteLLM RAGAS wrapper.

        Args:
            model: LiteLLM-compatible model name (e.g., 'gpt-4o', 'anthropic/claude-3-5-sonnet')
            temperature: Generation temperature (default 0.0 for deterministic outputs)
            run_config: Optional RAGAS run configuration
            multiple_completion_supported: Whether model supports multiple completions
            cache: Optional cache interface
            rate_limit_buffer: Buffer time to add to rate limit resets (seconds)
            max_retries: Maximum number of retries for failed requests
        """
        if run_config is None:
            run_config = RunConfig()

        super().__init__(
            run_config=run_config,
            multiple_completion_supported=multiple_completion_supported,
            cache=cache,
        )

        self.model = model
        self._temperature = temperature
        self.rate_limit_buffer = rate_limit_buffer
        self.max_retries = max_retries

        # Accumulated cost tracking
        self._accumulated_cost = 0.0
        self._accumulated_prompt_tokens = 0
        self._accumulated_completion_tokens = 0

    def _extract_cost(self, response) -> float:
        """
        Extract cost from LiteLLM response.

        Priority:
        1. response._hidden_params["response_cost"] (real-time pricing)
        2. litellm.completion_cost(response)
        3. litellm.cost_per_token() with token counts
        4. Return 0.0 if all methods fail
        """
        # Primary: Use response_cost from _hidden_params
        try:
            cost = response._hidden_params.get('response_cost', 0.0)
            if cost:
                logger.debug(f'Cost from _hidden_params: {cost}')
                return float(cost)
        except (AttributeError, KeyError):
            pass

        # Fallback: Use completion_cost function
        try:
            cost = litellm.completion_cost(completion_response=response)
            if cost:
                logger.debug(f'Cost from completion_cost: {cost}')
                return cost
        except Exception:
            pass

        # Final fallback: Use token counts
        try:
            usage = response.usage
            if usage:
                cost = litellm.cost_per_token(
                    model=self.model,
                    prompt_tokens=usage.prompt_tokens or 0,
                    completion_tokens=usage.completion_tokens or 0,
                )
                if isinstance(cost, tuple):
                    total = cost[0] + cost[1]
                    logger.debug(f'Cost from cost_per_token: {total}')
                    return total
                logger.debug(f'Cost from cost_per_token: {cost}')
                return cost
        except Exception as e:
            logger.debug(f'Cost extraction failed: {e}')

        return 0.0

    def _track_usage(self, response) -> None:
        """Track usage and cost from response."""
        cost = self._extract_cost(response)
        self._accumulated_cost += cost

        usage = getattr(response, 'usage', None)
        if usage:
            self._accumulated_prompt_tokens += usage.prompt_tokens or 0
            self._accumulated_completion_tokens += usage.completion_tokens or 0

        logger.debug(
            f'LLM call cost: {cost}, accumulated: {self._accumulated_cost}, '
            f'tokens: {self._accumulated_prompt_tokens}+{self._accumulated_completion_tokens}'
        )

    def reset_cost(self) -> None:
        """Reset accumulated cost. Call before starting a new evaluation."""
        self._accumulated_cost = 0.0
        self._accumulated_prompt_tokens = 0
        self._accumulated_completion_tokens = 0

    @property
    def cost_estimate(self) -> float:
        """Total accumulated cost from all LLM calls."""
        return self._accumulated_cost

    @property
    def prompt_tokens(self) -> int:
        """Total accumulated prompt tokens."""
        return self._accumulated_prompt_tokens

    @property
    def completion_tokens(self) -> int:
        """Total accumulated completion tokens."""
        return self._accumulated_completion_tokens

    def _prompt_to_string(self, prompt) -> str:
        """Convert a prompt to string."""
        if isinstance(prompt, str):
            return prompt
        elif hasattr(prompt, 'to_string'):
            return prompt.to_string()
        else:
            return str(prompt)

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if the error is a rate limit (429) error."""
        error_str = str(error).lower()
        if '429' in error_str or 'rate limit' in error_str:
            return True
        if '500' in error_str and (
            '429' in error_str
            or 'rate_limit' in error_str
            or 'llm_gatekeeper' in error_str
        ):
            return True
        return False

    def _handle_rate_limit_with_retry(self, func, *args, **kwargs):
        """Execute a function with rate-limit aware retry logic."""
        max_retries = getattr(self.run_config, 'max_retries', self.max_retries)
        last_exception = None

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt == max_retries - 1:
                    logger.error(
                        f'All {max_retries} attempts failed. Final error: {str(e)}'
                    )
                    raise

                if self._is_rate_limit_error(e):
                    wait_time = min(1.0 * (2**attempt), 120.0)
                    logger.warning(
                        f'Rate limit hit. Waiting {wait_time:.1f}s '
                        f'(attempt {attempt + 1}/{max_retries})'
                    )
                    time.sleep(wait_time)
                else:
                    wait_time = min(1.0 * (2**attempt), 10.0)
                    logger.warning(
                        f'Retrying with exponential backoff: {wait_time:.1f}s '
                        f'(attempt {attempt + 1}/{max_retries})'
                    )
                    time.sleep(wait_time)

        raise last_exception

    async def _ahandle_rate_limit_with_retry(self, async_func, *args, **kwargs):
        """Execute an async function with rate-limit aware retry logic."""
        import asyncio

        max_retries = getattr(self.run_config, 'max_retries', self.max_retries)
        last_exception = None

        for attempt in range(max_retries):
            try:
                return await async_func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt == max_retries - 1:
                    logger.error(
                        f'All {max_retries} attempts failed. Final error: {str(e)}'
                    )
                    raise

                if self._is_rate_limit_error(e):
                    wait_time = min(1.0 * (2**attempt), 120.0)
                    logger.warning(
                        f'Rate limit hit. Waiting {wait_time:.1f}s '
                        f'(attempt {attempt + 1}/{max_retries})'
                    )
                    await asyncio.sleep(wait_time)
                else:
                    wait_time = min(1.0 * (2**attempt), 10.0)
                    logger.warning(
                        f'Retrying with exponential backoff: {wait_time:.1f}s '
                        f'(attempt {attempt + 1}/{max_retries})'
                    )
                    await asyncio.sleep(wait_time)

        raise last_exception

    @trace(name='generate_text', capture_args=True, capture_result=True)
    def generate_text(
        self,
        prompt,
        n: int = 1,
        temperature: float = 1e-8,
        stop: Optional[List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        """Synchronous text generation."""
        prompt_str = self._prompt_to_string(prompt)

        def _call_llm():
            return litellm.completion(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt_str}],
                temperature=temperature or self._temperature,
                n=n,
                stop=stop,
            )

        response = self._handle_rate_limit_with_retry(_call_llm)
        self._track_usage(response)

        generations = [
            [Generation(text=c.message.content or '') for c in response.choices]
        ]
        return LLMResult(generations=generations)

    @trace(name='generate', capture_args=True, capture_result=True)
    async def generate(
        self,
        prompt,
        n: int = 1,
        temperature: float = 1e-8,
        stop: Optional[List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        """Async generate - calls generate_text."""
        return self.generate_text(
            prompt=prompt,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )

    @trace(name='agenerate_text', capture_args=True, capture_result=True)
    async def agenerate_text(
        self,
        prompt,
        n: int = 1,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        """Asynchronous text generation."""
        if temperature is None:
            temperature = self.get_temperature(n)

        prompt_str = self._prompt_to_string(prompt)

        async def _call_llm():
            return await litellm.acompletion(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt_str}],
                temperature=temperature or self._temperature,
                n=n,
                stop=stop,
            )

        response = await self._ahandle_rate_limit_with_retry(_call_llm)
        self._track_usage(response)

        generations = [
            [Generation(text=c.message.content or '') for c in response.choices]
        ]
        return LLMResult(generations=generations)

    def is_finished(self, response: LLMResult) -> bool:
        """Check if the response is complete."""
        return True

    def get_temperature(self, n: int) -> float:
        """Get the temperature to use for n completions."""
        return self._temperature
