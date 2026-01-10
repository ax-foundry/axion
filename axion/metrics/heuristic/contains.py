from axion.dataset import DatasetItem
from axion.metrics.base import (
    BaseMetric,
    MetricEvaluationResult,
    metric,
)


@metric(
    name='Contains Match',
    key='contains_match',
    description='Returns 1.0 if the actual output contains the expected output (after stripping).',
    required_fields=['actual_output', 'expected_output'],
    optional_fields=[],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['heuristic', 'binary'],
)
class ContainsMatch(BaseMetric):
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """
        Returns 1.0 if the actual output contains the expected output (after stripping).
        Returns 0.0 otherwise.
        """
        if item.actual_output is None or item.expected_output is None:
            return MetricEvaluationResult(score=0.0)

        expected = item.expected_output.strip()
        is_contained = expected in item.actual_output
        return MetricEvaluationResult(score=1.0 if is_contained else 0.0)
