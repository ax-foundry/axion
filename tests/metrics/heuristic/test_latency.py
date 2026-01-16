import pytest

from axion._core.error import CustomValidationError
from axion.dataset import DatasetItem
from axion.metrics.heuristic.latency import Latency


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'latency, expected_score', [(2.12, 2.12), (0.0, 0.0), ('100', 100)]
)
async def test_latency(latency, expected_score):
    item = DatasetItem(latency=latency)
    metric = Latency()
    result = await metric.execute(item)
    assert result.score == expected_score


@pytest.mark.asyncio
@pytest.mark.parametrize('latency, expected_score', [('one hundred', 100)])
async def test_latency_fail(latency, expected_score):
    with pytest.raises(CustomValidationError):
        item = DatasetItem(latency=latency)
        metric = Latency()
        _ = await metric.execute(item)
