from abc import ABC, abstractmethod

import numpy as np


class AggregationStrategy(ABC):
    """Abstract base class for different score aggregation strategies."""

    @abstractmethod
    def aggregate(self, scores: np.ndarray, weights: np.ndarray) -> float:
        """
        Aggregates scores, handling potential np.nan values.

        Args:
            scores (np.ndarray): An array of scores, which may contain np.nan for failed metrics.
            weights (np.ndarray): An array of corresponding weights.

        Returns:
            float: The aggregated score, or np.nan if aggregation is not possible.
        """
        pass


class WeightedAverage(AggregationStrategy):
    """
    Calculates the weighted average of scores, ignoring any NaN values.
    If all scores are NaN, the result will be NaN.
    """

    def aggregate(self, scores: np.ndarray, weights: np.ndarray) -> float:
        """
        Calculates the weighted average, ignoring NaNs, using a more direct NumPy approach.
        """
        valid_weights = weights[~np.isnan(scores)]
        total_valid_weight = np.sum(valid_weights)
        if total_valid_weight == 0:
            return np.nan
        weighted_sum = np.nansum(scores * weights)

        return weighted_sum / total_valid_weight


class MinAggregation(AggregationStrategy):
    """Returns the minimum score, ignoring NaN values."""

    def aggregate(self, scores: np.ndarray, weights: np.ndarray) -> float:
        result = np.nanmin(scores)
        return float(result) if not np.isnan(result) else np.nan


class MaxAggregation(AggregationStrategy):
    """Returns the maximum score, ignoring NaN values."""

    def aggregate(self, scores: np.ndarray, weights: np.ndarray) -> float:
        result = np.nanmax(scores)
        return float(result) if not np.isnan(result) else np.nan


AGGREGATION_STRATEGIES = {
    'weighted_average': WeightedAverage,
    'min': MinAggregation,
    'max': MaxAggregation,
}
