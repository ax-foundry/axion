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


@pytest.mark.asyncio
async def test_contains_match_with_field_mapping():
    """Test that ContainsMatch works with field_mapping for custom field sources."""
    item = DatasetItem(
        query='test',
        additional_output={'summary': 'The sky is blue'},
        additional_input={'target': 'sky'},
    )
    metric = ContainsMatch(
        field_mapping={
            'actual_output': 'additional_output.summary',
            'expected_output': 'additional_input.target',
        }
    )
    result = await metric.execute(item)
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_contains_match_backward_compatible():
    """Verify metric works without field_mapping (backward compatibility)."""
    item = DatasetItem(actual_output='The sky is blue', expected_output='sky')
    metric = ContainsMatch()  # No field_mapping
    result = await metric.execute(item)
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_contains_match_with_none_values():
    """Test that ContainsMatch returns 0.0 when fields are None."""
    item = DatasetItem(actual_output=None, expected_output='sky')
    metric = ContainsMatch()
    result = await metric.execute(item)
    assert result.score == 0.0

    item2 = DatasetItem(actual_output='The sky is blue', expected_output=None)
    result2 = await metric.execute(item2)
    assert result2.score == 0.0
