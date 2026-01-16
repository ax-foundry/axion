"""Lightweight DeepEval LLM wrapper using LiteLLM with cost tracking."""

import json
from typing import Optional, Union

import litellm
from pydantic import BaseModel

from axion._core.logging import get_logger
from axion._core.tracing import trace

logger = get_logger(__name__)

try:
    from deepeval.models.base_model import DeepEvalBaseLLM
except ImportError:
    DeepEvalBaseLLM = object
    import warnings

    warnings.warn(
        'deepeval is not installed. LiteLLMDeepEval will not be fully functional.'
    )


class LiteLLMDeepEval(DeepEvalBaseLLM):
    """
    DeepEval-compatible LLM wrapper using LiteLLM with cost tracking.

    Accumulates costs across multiple LLM calls during metric evaluation.
    Uses LiteLLM's native cost tracking via response._hidden_params["response_cost"].

    Example:
        >>> from axion.integrations.models.litellm_deepeval import LiteLLMDeepEval
        >>> from deepeval.metrics import AnswerRelevancyMetric
        >>>
        >>> llm = LiteLLMDeepEval(model='gpt-4o')
        >>> metric = AnswerRelevancyMetric(model=llm)
        >>> # After evaluation:
        >>> print(metric.model.cost_estimate)
    """

    def __init__(
        self,
        model: str = 'gpt-4o',
        temperature: float = 0.0,
    ):
        """
        Initialize the LiteLLM DeepEval wrapper.

        Args:
            model: LiteLLM-compatible model name (e.g., 'gpt-4o', 'anthropic/claude-3-5-sonnet')
            temperature: Generation temperature (default 0.0 for deterministic outputs)
        """
        self.model_name = model
        self._temperature = temperature
        self._accumulated_cost = 0.0
        self._accumulated_prompt_tokens = 0
        self._accumulated_completion_tokens = 0
        super().__init__(model_name=model)

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
                    model=self.model_name,
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

    def load_model(self):
        """Required by base class but not used."""
        return None

    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model_name

    @trace(name='generate', capture_args=True, capture_result=True)
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Union[str, BaseModel]:
        """
        Synchronous text generation.

        Args:
            prompt: The prompt to send to the model
            schema: Optional Pydantic model for structured output parsing

        Returns:
            Generated text or parsed Pydantic model if schema is provided
        """
        response = litellm.completion(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=self._temperature,
        )

        self._track_usage(response)
        text = response.choices[0].message.content or ''

        if schema:
            try:
                data = json.loads(text)
                return schema.model_validate(data)
            except Exception:
                return text
        return text

    @trace(name='a_generate', capture_args=True, capture_result=True)
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Union[str, BaseModel]:
        """
        Asynchronous text generation.

        Args:
            prompt: The prompt to send to the model
            schema: Optional Pydantic model for structured output parsing

        Returns:
            Generated text or parsed Pydantic model if schema is provided
        """
        response = await litellm.acompletion(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=self._temperature,
        )

        self._track_usage(response)
        text = response.choices[0].message.content or ''

        if schema:
            try:
                data = json.loads(text)
                return schema.model_validate(data)
            except Exception:
                return text
        return text
