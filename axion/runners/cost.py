"""
Cost extraction utilities for metric runners.

This module provides a scalable way to extract cost estimates from different
metric sources (Ragas, DeepEval, Axion) using a unified interface.

Based on Ragas cost tracking pattern:
https://docs.ragas.io/en/stable/howtos/applications/_cost/?h=cost
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from axion._core.logging import get_logger
from axion.llm_registry import LLMCostEstimator

logger = get_logger(__name__)


@dataclass
class TokenUsage:
    """
    Token usage information from an LLM call.

    Follows the Ragas TokenUsage pattern for consistency.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ''

    @property
    def total_tokens(self) -> int:
        """Total tokens used in the call."""
        return self.input_tokens + self.output_tokens

    def compute_cost(
        self,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
    ) -> float:
        """
        Compute cost based on token usage.

        Args:
            cost_per_input_token: Cost per input token (if None, uses LLMCostEstimator)
            cost_per_output_token: Cost per output token (if None, uses LLMCostEstimator)

        Returns:
            Estimated cost in USD
        """
        if cost_per_input_token is not None and cost_per_output_token is not None:
            return (
                self.input_tokens * cost_per_input_token
                + self.output_tokens * cost_per_output_token
            )

        # Use LLMCostEstimator for model-specific pricing
        if self.model:
            return LLMCostEstimator.estimate(
                model_name=self.model,
                prompt_tokens=self.input_tokens,
                completion_tokens=self.output_tokens,
            )

        return 0.0

    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        """Combine two TokenUsage instances."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            model=self.model or other.model,
        )


@dataclass
class CostResult:
    """
    Result of cost extraction containing both cost estimate and token usage.
    """

    cost_estimate: float = 0.0
    token_usage: Optional[TokenUsage] = None
    source: str = 'unknown'
    details: Dict[str, Any] = field(default_factory=dict)

    def __add__(self, other: 'CostResult') -> 'CostResult':
        """Combine two CostResult instances."""
        combined_tokens = None
        if self.token_usage and other.token_usage:
            combined_tokens = self.token_usage + other.token_usage
        elif self.token_usage:
            combined_tokens = self.token_usage
        elif other.token_usage:
            combined_tokens = other.token_usage

        return CostResult(
            cost_estimate=self.cost_estimate + other.cost_estimate,
            token_usage=combined_tokens,
            source=f'{self.source}+{other.source}',
            details={**self.details, **other.details},
        )


@runtime_checkable
class HasCostEstimate(Protocol):
    """Protocol for objects that have a cost_estimate attribute."""

    @property
    def cost_estimate(self) -> float: ...


@runtime_checkable
class HasLLM(Protocol):
    """Protocol for objects that have an llm attribute with cost tracking."""

    @property
    def llm(self) -> Any: ...


@runtime_checkable
class HasModel(Protocol):
    """Protocol for objects that have a model attribute with cost tracking."""

    @property
    def model(self) -> Any: ...


class CostExtractor:
    """
    Unified cost extractor for different metric sources.

    Provides a scalable way to extract cost estimates from Ragas, DeepEval,
    and Axion metrics using multiple extraction strategies.

    Example:
        >>> extractor = CostExtractor()
        >>> cost_result = extractor.extract(metric)
        >>> print(f"Cost: ${cost_result.cost_estimate:.6f}")
    """

    # Extraction strategies in priority order
    EXTRACTION_STRATEGIES = [
        '_extract_direct_cost',
        '_extract_evaluation_cost',  # DeepEval metric.evaluation_cost
        '_extract_from_llm',
        '_extract_from_model',
        '_extract_from_evaluation_model',
        '_extract_from_usage_tracker',
    ]

    def extract(self, metric: Any, default: float = 0.0) -> CostResult:
        """
        Extract cost information from a metric using multiple strategies.

        This method tries different extraction strategies in priority order
        until one succeeds. This makes it robust to different metric implementations.

        Args:
            metric: The metric object to extract cost from
            default: Default cost value if extraction fails

        Returns:
            CostResult with extracted cost and token usage information
        """
        if metric is None:
            return CostResult(cost_estimate=default, source='none')

        for strategy_name in self.EXTRACTION_STRATEGIES:
            strategy = getattr(self, strategy_name, None)
            if strategy:
                try:
                    result = strategy(metric)
                    if result and result.cost_estimate > 0:
                        logger.debug(
                            f'Cost extracted via {strategy_name}: ${result.cost_estimate:.6f}'
                        )
                        return result
                except Exception as e:
                    logger.debug(f'Strategy {strategy_name} failed: {e}')
                    continue

        return CostResult(cost_estimate=default, source='default')

    def _extract_direct_cost(self, metric: Any) -> Optional[CostResult]:
        """Extract cost directly from metric.cost_estimate attribute."""
        cost = getattr(metric, 'cost_estimate', None)
        if cost is not None and isinstance(cost, (int, float)) and cost > 0:
            return CostResult(
                cost_estimate=float(cost),
                source='direct',
                details={'attribute': 'cost_estimate'},
            )
        return None

    def _extract_evaluation_cost(self, metric: Any) -> Optional[CostResult]:
        """
        Extract cost from metric.evaluation_cost (DeepEval pattern).

        DeepEval metrics expose evaluation_cost as a direct attribute
        after execution.
        """
        cost = getattr(metric, 'evaluation_cost', None)
        if cost is not None and isinstance(cost, (int, float)) and cost > 0:
            return CostResult(
                cost_estimate=float(cost),
                source='deepeval_evaluation_cost',
                details={'attribute': 'evaluation_cost'},
            )
        return None

    def _extract_from_llm(self, metric: Any) -> Optional[CostResult]:
        """
        Extract cost from metric.llm.cost_estimate (Ragas pattern).

        This follows the Ragas approach where the LLM wrapper tracks costs.
        """
        llm = getattr(metric, 'llm', None)
        if llm is None:
            return None

        # Try direct cost_estimate on LLM
        cost = getattr(llm, 'cost_estimate', None)
        if cost is not None and isinstance(cost, (int, float)) and cost > 0:
            return CostResult(
                cost_estimate=float(cost),
                source='ragas_llm',
                token_usage=self._extract_token_usage(llm),
                details={'llm_type': type(llm).__name__},
            )

        # Try _cost_estimate (internal attribute)
        cost = getattr(llm, '_cost_estimate', None)
        if cost is not None and isinstance(cost, (int, float)) and cost > 0:
            return CostResult(
                cost_estimate=float(cost),
                source='ragas_llm_internal',
                token_usage=self._extract_token_usage(llm),
                details={'llm_type': type(llm).__name__},
            )

        return None

    def _extract_from_model(self, metric: Any) -> Optional[CostResult]:
        """
        Extract cost from metric.model.cost_estimate (DeepEval pattern).

        This follows the DeepEval approach where the model wrapper tracks costs.
        """
        model = getattr(metric, 'model', None)
        if model is None:
            return None

        # Try direct cost_estimate on model
        cost = getattr(model, 'cost_estimate', None)
        if cost is not None and isinstance(cost, (int, float)) and cost > 0:
            return CostResult(
                cost_estimate=float(cost),
                source='deepeval_model',
                token_usage=self._extract_token_usage(model),
                details={'model_type': type(model).__name__},
            )

        # Try _cost_estimate (internal attribute)
        cost = getattr(model, '_cost_estimate', None)
        if cost is not None and isinstance(cost, (int, float)) and cost > 0:
            return CostResult(
                cost_estimate=float(cost),
                source='deepeval_model_internal',
                token_usage=self._extract_token_usage(model),
                details={'model_type': type(model).__name__},
            )

        return None

    def _extract_from_evaluation_model(self, metric: Any) -> Optional[CostResult]:
        """
        Extract cost from metric.evaluation_model (alternative DeepEval pattern).
        """
        eval_model = getattr(metric, 'evaluation_model', None)
        if eval_model is None:
            return None

        cost = getattr(eval_model, 'cost_estimate', None)
        if cost is not None and isinstance(cost, (int, float)) and cost > 0:
            return CostResult(
                cost_estimate=float(cost),
                source='evaluation_model',
                token_usage=self._extract_token_usage(eval_model),
                details={'model_type': type(eval_model).__name__},
            )

        return None

    def _extract_from_usage_tracker(self, metric: Any) -> Optional[CostResult]:
        """
        Extract cost from a token usage tracker if available.

        Some metrics may track token usage separately and compute cost from that.
        """
        # Check for token_usage or usage attribute
        usage = getattr(metric, 'token_usage', None) or getattr(metric, 'usage', None)
        if usage is None:
            return None

        token_usage = self._parse_token_usage(usage, metric)
        if token_usage and token_usage.total_tokens > 0:
            cost = token_usage.compute_cost()
            return CostResult(
                cost_estimate=cost,
                source='usage_tracker',
                token_usage=token_usage,
                details={'tracked_tokens': token_usage.total_tokens},
            )

        return None

    def _extract_token_usage(self, obj: Any) -> Optional[TokenUsage]:
        """
        Extract token usage information from an LLM/model object.
        """
        if obj is None:
            return None

        # Try various attribute names for token counts
        input_tokens = (
            getattr(obj, 'input_tokens', 0)
            or getattr(obj, 'prompt_tokens', 0)
            or getattr(obj, '_input_tokens', 0)
        )

        output_tokens = (
            getattr(obj, 'output_tokens', 0)
            or getattr(obj, 'completion_tokens', 0)
            or getattr(obj, '_output_tokens', 0)
        )

        model_name = (
            getattr(obj, 'model', '')
            or getattr(obj, 'model_name', '')
            or getattr(obj, '_model', '')
        )

        if input_tokens or output_tokens:
            return TokenUsage(
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                model=str(model_name) if model_name else '',
            )

        return None

    def _parse_token_usage(self, usage: Any, metric: Any) -> Optional[TokenUsage]:
        """
        Parse token usage from various formats.
        """
        if isinstance(usage, TokenUsage):
            return usage

        if isinstance(usage, dict):
            return TokenUsage(
                input_tokens=usage.get('input_tokens', 0)
                or usage.get('prompt_tokens', 0),
                output_tokens=usage.get('output_tokens', 0)
                or usage.get('completion_tokens', 0),
                model=usage.get('model', getattr(metric, 'model_name', '')),
            )

        # Try to get tokens from object attributes
        return self._extract_token_usage(usage)


class CostAggregator:
    """
    Aggregates costs across multiple metrics and evaluations.

    Example:
        >>> aggregator = CostAggregator()
        >>> for metric in metrics:
        ...     aggregator.add(metric)
        >>> print(f"Total cost: ${aggregator.total_cost:.4f}")
    """

    def __init__(self):
        self._extractor = CostExtractor()
        self._results: List[CostResult] = []
        self._total_usage = TokenUsage()

    def add(self, metric: Any) -> CostResult:
        """
        Add a metric's cost to the aggregator.

        Args:
            metric: The metric to extract and add cost from

        Returns:
            The extracted CostResult
        """
        result = self._extractor.extract(metric)
        self._results.append(result)

        if result.token_usage:
            self._total_usage = self._total_usage + result.token_usage

        return result

    def add_result(self, result: CostResult) -> None:
        """Add a pre-computed CostResult to the aggregator."""
        self._results.append(result)
        if result.token_usage:
            self._total_usage = self._total_usage + result.token_usage

    @property
    def total_cost(self) -> float:
        """Total cost across all added metrics."""
        return sum(r.cost_estimate for r in self._results)

    @property
    def total_tokens(self) -> TokenUsage:
        """Total token usage across all added metrics."""
        return self._total_usage

    @property
    def results(self) -> List[CostResult]:
        """All collected cost results."""
        return self._results.copy()

    def compute_total_cost(
        self,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
    ) -> float:
        """
        Compute total cost with custom token pricing.

        Follows the Ragas pattern of allowing custom cost_per_token values.

        Args:
            cost_per_input_token: Custom cost per input token
            cost_per_output_token: Custom cost per output token

        Returns:
            Total cost in USD
        """
        if cost_per_input_token is not None and cost_per_output_token is not None:
            return self._total_usage.compute_cost(
                cost_per_input_token, cost_per_output_token
            )
        return self.total_cost

    def summary(self) -> Dict[str, Any]:
        """Get a summary of all costs."""
        return {
            'total_cost': self.total_cost,
            'total_input_tokens': self._total_usage.input_tokens,
            'total_output_tokens': self._total_usage.output_tokens,
            'total_tokens': self._total_usage.total_tokens,
            'num_metrics': len(self._results),
            'costs_by_source': self._costs_by_source(),
        }

    def _costs_by_source(self) -> Dict[str, float]:
        """Group costs by their extraction source."""
        by_source: Dict[str, float] = {}
        for result in self._results:
            source = result.source
            by_source[source] = by_source.get(source, 0.0) + result.cost_estimate
        return by_source


class CostProviderMixin:
    """
    Mixin for custom metrics to track costs from LiteLLM responses.

    This mixin provides methods to accumulate costs across multiple LLM calls
    and exposes a cost_estimate property that works with the CostExtractor.

    Example:
        >>> class MyMetric(BaseMetric, CostProviderMixin):
        ...     async def execute(self, item, **kwargs):
        ...         self.reset_cost()
        ...         response = litellm.completion(...)
        ...         self.add_cost_from_response(response)
        ...         return result
    """

    _accumulated_cost: float = 0.0
    _accumulated_input_tokens: int = 0
    _accumulated_output_tokens: int = 0

    def add_cost_from_response(self, response: Any) -> None:
        """
        Extract and accumulate cost from a LiteLLM response.

        Prioritizes response._hidden_params['response_cost'] (real-time pricing),
        falling back to litellm.completion_cost() if not available.

        Args:
            response: A LiteLLM completion response object
        """
        import litellm

        # Primary: Use response_cost from LiteLLM (real-time pricing)
        cost = 0.0
        try:
            cost = response._hidden_params.get('response_cost', 0.0)
            if not cost:
                # Fallback: Use completion_cost function
                cost = litellm.completion_cost(completion_response=response)
        except Exception:
            pass

        self._accumulated_cost += cost

        # Track token usage if available
        usage = getattr(response, 'usage', None)
        if usage:
            self._accumulated_input_tokens += getattr(usage, 'prompt_tokens', 0) or 0
            self._accumulated_output_tokens += (
                getattr(usage, 'completion_tokens', 0) or 0
            )

    def add_cost(
        self,
        cost_usd: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """
        Manually add cost and token usage.

        Args:
            cost_usd: Cost in USD to add
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
        """
        self._accumulated_cost += cost_usd
        self._accumulated_input_tokens += input_tokens
        self._accumulated_output_tokens += output_tokens

    def reset_cost(self) -> None:
        """Reset accumulated costs. Call this before each metric execution."""
        self._accumulated_cost = 0.0
        self._accumulated_input_tokens = 0
        self._accumulated_output_tokens = 0

    @property
    def cost_estimate(self) -> float:
        """Return the accumulated cost. Works with CostExtractor."""
        return self._accumulated_cost

    @property
    def accumulated_tokens(self) -> TokenUsage:
        """Return the accumulated token usage."""
        return TokenUsage(
            input_tokens=self._accumulated_input_tokens,
            output_tokens=self._accumulated_output_tokens,
        )


# Singleton instance for convenience
_default_extractor = CostExtractor()


def extract_cost(metric: Any, default: float = 0.0) -> float:
    """
    Convenience function to extract cost from a metric.

    Args:
        metric: The metric object to extract cost from
        default: Default value if cost cannot be extracted

    Returns:
        Extracted cost estimate
    """
    result = _default_extractor.extract(metric, default)
    return result.cost_estimate


def extract_cost_result(metric: Any, default: float = 0.0) -> CostResult:
    """
    Convenience function to extract full cost result from a metric.

    Args:
        metric: The metric object to extract cost from
        default: Default value if cost cannot be extracted

    Returns:
        CostResult with cost and token usage information
    """
    return _default_extractor.extract(metric, default)
