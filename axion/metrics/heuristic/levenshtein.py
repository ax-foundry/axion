from difflib import SequenceMatcher

from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import (
    BaseMetric,
    MetricEvaluationResult,
    metric,
)


@metric(
    name='Levenshtein Ratio',
    description=(
        'Calculates the Levenshtein ratio (string similarity) between actual and expected outputs. '
        'Returns a score between 0.0 and 1.0, where 1.0 means identical strings.'
    ),
    required_fields=['actual_output', 'expected_output'],
    optional_fields=[],
    default_threshold=0.2,
    score_range=(0, 1),
    tags=['heuristic'],
)
class LevenshteinRatio(BaseMetric):
    def __init__(self, case_sensitive: bool = False, **kwargs):
        """
        Initializes the LevenshteinRatio metric.

        Args:
            case_sensitive (bool): If False (default), the comparison is done in lowercase.
        """
        super().__init__(**kwargs)
        self.case_sensitive = case_sensitive

    @trace(name='LevenshteinRatio', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """
        Computes the Levenshtein ratio between the actual and expected outputs.

        Args:
            item (DatasetItem): The evaluation input containing expected and actual strings.

        Returns:
            EvaluationResult: Score between 0.0 and 1.0 representing string similarity.
        """
        actual = item.actual_output or ''
        expected = item.expected_output or ''

        if not self.case_sensitive:
            actual = actual.lower()
            expected = expected.lower()

        ratio = SequenceMatcher(None, actual, expected).ratio()
        return MetricEvaluationResult(score=round(ratio, 4))
