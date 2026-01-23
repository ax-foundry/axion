from typing import List

import pytest

from axion.align.web_eval import WebAlignEval
from axion.dataset import Dataset, DatasetItem
from axion.llm_registry import MockLLM
from axion.metrics.base import BaseMetric
from axion.schema import MetricScore, TestResult


class DummyMetric(BaseMetric):
    pass


def _make_dataset() -> Dataset:
    items: List[DatasetItem] = [
        DatasetItem(
            id='item-1',
            query='Q1',
            expected_output='E1',
            actual_output='A1',
            judgment=1,
        ),
        DatasetItem(
            id='item-2',
            query='Q2',
            expected_output='E2',
            actual_output='A2',
            judgment=0,
        ),
    ]
    return Dataset(items=items)


@pytest.fixture()
def dummy_metric() -> BaseMetric:
    return DummyMetric(llm=MockLLM())


def test_to_dict_returns_serializable_structures(monkeypatch, dummy_metric):
    dataset = _make_dataset()
    evaluator = WebAlignEval(dataset, dummy_metric)

    async def fake_execute_batch(self, items):
        return [
            TestResult(
                test_case=item,
                score_results=[
                    MetricScore(
                        id=item.id,
                        name='dummy',
                        score=float(item.judgment),
                        explanation='ok',
                    )
                ],
            )
            for item in items
        ]

    monkeypatch.setattr(
        'axion.align.base.MetricRunner.execute_batch', fake_execute_batch
    )

    result = evaluator.execute()

    assert set(result.keys()) == {'results', 'metrics', 'confusion_matrix'}
    assert isinstance(result['results'], list)
    assert result['results'][0]['id'] == 'item-1'
    assert isinstance(result['metrics'], dict)
    assert isinstance(result['confusion_matrix'], dict)


def test_progress_callback_invoked(monkeypatch, dummy_metric):
    dataset = _make_dataset()
    evaluator = WebAlignEval(dataset, dummy_metric)
    calls = []

    async def fake_execute_batch(self, items):
        return [
            TestResult(
                test_case=item,
                score_results=[
                    MetricScore(
                        id=item.id,
                        name='dummy',
                        score=float(item.judgment),
                        explanation='ok',
                    )
                ],
            )
            for item in items
        ]

    monkeypatch.setattr(
        'axion.align.base.MetricRunner.execute_batch', fake_execute_batch
    )

    evaluator.execute(on_progress=lambda current, total: calls.append((current, total)))

    assert calls == [(1, 2), (2, 2)]
