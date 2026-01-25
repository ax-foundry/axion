from axion._core.tracing import trace
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
    @trace(name='ContainsMatch', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """
        Returns 1.0 if the actual output contains the expected output (after stripping).
        Returns 0.0 otherwise.
        """
        actual_output = self.get_field(item, 'actual_output')
        expected_output = self.get_field(item, 'expected_output')
        if actual_output is None or expected_output is None:
            return MetricEvaluationResult(score=0.0)

        expected = expected_output.strip()
        is_contained = expected in actual_output
        return MetricEvaluationResult(score=1.0 if is_contained else 0.0)
