import pytest
from axion.dataset import DatasetItem
from axion.metrics.heuristic.string_match import ExactStringMatch


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'actual, expected, expected_score',
    [
        ('Hello', 'Hello', 1.0),
        ('Hello', 'hello', 0.0),  # Case-sensitive by default
        ('Goodbye', 'Good Bye', 0.0),
    ],
)
async def test_exact_string_match(actual, expected, expected_score):
    item = DatasetItem(actual_output=actual, expected_output=expected)
    metric = ExactStringMatch()
    result = await metric.execute(item)
    assert result.score == expected_score
