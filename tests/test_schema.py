from datetime import datetime, timezone

import numpy as np
import pandas as pd
from axion.dataset import DatasetItem
from axion.schema import EvaluationResult, MetricScore, TestResult


def test_metric_score_defaults():
    metric = MetricScore(name='accuracy')
    assert metric.name == 'accuracy'
    assert np.isnan(metric.score)
    assert metric.timestamp is not None
    assert isinstance(metric.timestamp, str)


def test_metric_score_with_values():
    metric = MetricScore(
        name='coherence',
        score=0.85,
        threshold=0.8,
        passed=True,
        explanation='Above threshold.',
        metadata={'tokens': 50},
        version='1.0.0',
        source='test-suite',
    )
    assert metric.score == 0.85
    assert metric.passed is True
    assert metric.metadata['tokens'] == 50


def test_test_result_structure():
    test_case = DatasetItem(
        id='test1', query='What is AI?', expected_output='Artificial Intelligence.'
    )
    metric = MetricScore(name='faithfulness', score=1.0)
    result = TestResult(test_case=test_case, score_results=[metric])

    assert result.test_case.id == 'test1'
    assert result.score_results[0].name == 'faithfulness'
    assert isinstance(result.metadata, dict)


def test_evaluation_result_to_dataframe():
    test_case = DatasetItem(id='123', query='Q?', expected_output='A.')
    metric = MetricScore(name='clarity', score=0.9)
    result = TestResult(test_case=test_case, score_results=[metric])

    eval_result = EvaluationResult(
        run_id='run-001',
        evaluation_name='test-exp',
        timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        results=[result],
        metadata={'model': 'gpt-4'},
    )

    df = eval_result.to_dataframe(id_as_index=True, rename_columns=False)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.index.name == 'id'
    assert 'score' in df.columns


def test_evaluation_result_empty_test_case():
    metric = MetricScore(name='robustness', score=0.75)
    result = TestResult(test_case=None, score_results=[metric])

    eval_result = EvaluationResult(
        run_id='run-002',
        evaluation_name=None,
        timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        results=[result],
    )

    df = eval_result.to_dataframe()
    assert 'metric_name' in df.columns
    assert df['metric_name'].iloc[0] == 'robustness'


def test_evaluation_result_rename_columns():
    metric = MetricScore(name='robustness', score=0.75)
    result = TestResult(test_case=None, score_results=[metric])

    eval_result = EvaluationResult(
        run_id='run-002',
        evaluation_name=None,
        timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        results=[result],
    )

    df = eval_result.to_dataframe(rename_columns=False)
    assert 'name' in df.columns
    assert df['name'].iloc[0] == 'robustness'
