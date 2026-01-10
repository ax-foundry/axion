import pytest
from axion.dataset import DatasetItem
from axion.metrics.heuristic.levenshtein import LevenshteinRatio


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'actual, expected, expected_score',
    [
        ('hello world', 'hello world', 1.0),
        ('hello world', 'hello wrld', pytest.approx(0.95, rel=1e-1)),
        ('abcdef', 'uvwxyz', pytest.approx(0.0, abs=1e-1)),
    ],
)
async def test_levenshtein_ratio(actual, expected, expected_score):
    item = DatasetItem(actual_output=actual, expected_output=expected)
    metric = LevenshteinRatio()
    result = await metric.execute(item)
    assert isinstance(result.score, float)
    assert result.score == expected_score
