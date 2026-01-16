import numpy as np
from axion.dataset import DatasetItem
from axion._core.tracing import trace
from axion.metrics.base import (
    BaseMetric,
    MetricEvaluationResult,
    metric,
)


@metric(
    name='Latency',
    description='Execution time for the task. Normalization options anchor on threshold.',
    required_fields=['latency'],
    optional_fields=['query'],
    default_threshold=5,
    score_range=(0, np.inf),
    tags=['heuristic'],
)
class Latency(BaseMetric):
    inverse_scoring_metric: bool = True

    def __init__(
        self,
        normalize: bool = False,
        normalization_method: str = 'exponential',
        **kwargs,
    ):
        """
        Initialize the Latency metric.

        Args:
            normalize (bool): If True, normalize latency scores to [0, 1] range.
            normalization_method (str): Method for normalization. Options:
                - 'exponential': Uses exp(-latency/threshold) for smooth decay
                - 'sigmoid': Uses 1/(1 + exp((latency-threshold)/scale)) for S-curve
                - 'reciprocal': Uses threshold/(threshold + latency) for hyperbolic decay
                - 'linear': Uses max(0, 1 - latency/threshold) for linear decay
            **kwargs: Additional arguments passed to the base metric.
        """
        self.normalize = normalize
        self.normalization_method = normalization_method
        super().__init__(**kwargs)

    def _normalize_score(self, latency: float) -> float:
        """
        Normalize latency to [0, 1] range using the specified method.

        Args:
            latency (float): Raw latency value in seconds.

        Returns:
            float: Normalized score in [0, 1] where higher is better.
        """
        if self.normalization_method == 'exponential':
            # Exponential decay: at threshold
            return np.exp(-latency / self.threshold)

        elif self.normalization_method == 'sigmoid':
            # Sigmoid function: smooth S-curve centered at threshold
            scale = self.threshold / 4  # Controls steepness
            return 1 / (1 + np.exp((latency - self.threshold) / scale))

        elif self.normalization_method == 'reciprocal':
            # Hyperbolic decay: at threshold
            return self.threshold / (self.threshold + latency)

        elif self.normalization_method == 'linear':
            # Linear decay: at threshold
            return max(0.0, 1.0 - (latency / self.threshold))

        else:
            raise ValueError(
                f'Unknown normalization method: {self.normalization_method}. '
                f"Choose from: 'exponential', 'sigmoid', 'reciprocal', 'linear'"
            )

    def _generate_explanation(self, latency: float, score: float) -> str:
        """
        Generate a human-readable explanation of the latency score.

        Args:
            latency (float): Raw latency value in seconds.
            score (float): Computed score (normalized or raw).

        Returns:
            str: Explanation of the score.
        """
        if self.normalize:
            # Determine performance relative to threshold
            if latency < self.threshold * 0.5:
                performance = 'excellent'
            elif latency < self.threshold:
                performance = 'good'
            elif latency < self.threshold * 1.5:
                performance = 'acceptable'
            else:
                performance = 'poor'

            explanation = (
                f'Latency: {latency:.3f}s. Normalized score: {score:.3f} '
                f'(threshold: {self.threshold}s, method: {self.normalization_method}). '
                f'Performance: {performance}.'
            )
        else:
            # Raw latency explanation
            if latency < self.threshold:
                comparison = f'below threshold ({self.threshold}s)'
            elif latency == self.threshold:
                comparison = f'at threshold ({self.threshold}s)'
            else:
                comparison = f'above threshold ({self.threshold}s)'

            explanation = f'Raw latency: {latency:.3f}s, {comparison}.'

        return explanation

    @trace(name='Latency', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """
        Returns the latency recorded for a given test case as the metric score.

        This metric assumes the `latency` field is already populated on the DatasetItem
        and returns it as-is or normalized based on the initialization parameters.

        Args:
            item (DatasetItem): The evaluation data point containing latency information.

        Returns:
            MetricEvaluationResult: The result object containing the latency as the score.
        """
        latency = item.latency

        if self.normalize:
            score = self._normalize_score(latency)
        else:
            score = latency

        explanation = self._generate_explanation(latency, score)

        return MetricEvaluationResult(
            score=score,
            explanation=explanation,
        )
