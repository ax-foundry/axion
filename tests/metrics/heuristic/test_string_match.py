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


@pytest.mark.asyncio
async def test_exact_string_match_with_field_mapping():
    """Test that ExactStringMatch works with field_mapping for custom field sources."""
    item = DatasetItem(
        query='test',
        additional_output={'result': 'Hello'},
        additional_input={'expected': 'Hello'},
    )
    metric = ExactStringMatch(
        field_mapping={
            'actual_output': 'additional_output.result',
            'expected_output': 'additional_input.expected',
        }
    )
    result = await metric.execute(item)
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_exact_string_match_backward_compatible():
    """Verify metric works without field_mapping (backward compatibility)."""
    item = DatasetItem(actual_output='Hello', expected_output='Hello')
    metric = ExactStringMatch()  # No field_mapping
    result = await metric.execute(item)
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_exact_string_match_with_none_values():
    """Test that ExactStringMatch returns 0.0 when fields are None."""
    item = DatasetItem(actual_output=None, expected_output='Hello')
    metric = ExactStringMatch()
    result = await metric.execute(item)
    assert result.score == 0.0

    item2 = DatasetItem(actual_output='Hello', expected_output=None)
    result2 = await metric.execute(item2)
    assert result2.score == 0.0
