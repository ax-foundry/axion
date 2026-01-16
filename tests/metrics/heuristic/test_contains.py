import pytest

from axion.dataset import DatasetItem
from axion.metrics.heuristic.contains import ContainsMatch


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'actual, expected, expected_score',
    [
        ('The sky is blue', 'sky', 1.0),
        ('Look at the stars', 'moon', 0.0),
        ('Deep learning is cool', 'learning', 1.0),
    ],
)
async def test_contains_match(actual, expected, expected_score):
    item = DatasetItem(actual_output=actual, expected_output=expected)
    metric = ContainsMatch()
    result = await metric.execute(item)
    assert result.score == expected_score
