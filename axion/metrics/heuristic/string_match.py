from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import (
    BaseMetric,
    MetricEvaluationResult,
    metric,
)


@metric(
    name='Exact String Match',
    key='exact_string_match',
    description='Checks whether the actual output matches the expected output exactly (after stripping whitespace).',
    required_fields=['actual_output', 'expected_output'],
    optional_fields=[],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['heuristic', 'binary'],
)
class ExactStringMatch(BaseMetric):
    @trace(name='ExactStringMatch', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """
        Returns 1.0 if the actual output exactly matches the expected output (after stripping).
        Returns 0.0 otherwise.
        """
        actual_output = self.get_field(item, 'actual_output')
        expected_output = self.get_field(item, 'expected_output')
        if actual_output is None or expected_output is None:
            return MetricEvaluationResult(score=0.0)

        is_match = actual_output == expected_output.strip()
        return MetricEvaluationResult(score=1.0 if is_match else 0.0)
