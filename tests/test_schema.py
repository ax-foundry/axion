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


def test_to_normalized_dataframes_basic():
    """Test basic normalized dataframes with single item and single metric."""
    test_case = DatasetItem(
        id='test-1', query='What is AI?', expected_output='Artificial Intelligence'
    )
    metric = MetricScore(name='faithfulness', score=0.95)
    result = TestResult(test_case=test_case, score_results=[metric])

    eval_result = EvaluationResult(
        run_id='run-001',
        evaluation_name='test-exp',
        timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        results=[result],
    )

    dataset_df, metrics_df = eval_result.to_normalized_dataframes()

    # Verify dataset items table (by_alias=True uses dataset_id)
    assert isinstance(dataset_df, pd.DataFrame)
    assert len(dataset_df) == 1
    assert dataset_df['dataset_id'].iloc[0] == 'test-1'
    assert dataset_df['query'].iloc[0] == 'What is AI?'

    # Verify metric results table
    assert isinstance(metrics_df, pd.DataFrame)
    assert len(metrics_df) == 1
    assert metrics_df['metric_name'].iloc[0] == 'faithfulness'
    assert metrics_df['metric_score'].iloc[0] == 0.95
    assert metrics_df['dataset_id'].iloc[0] == 'test-1'  # FK to dataset item


def test_to_normalized_dataframes_multiple_metrics():
    """Test normalized dataframes with one item and multiple metrics (FK validation)."""
    test_case = DatasetItem(
        id='test-2', query='Explain ML', expected_output='Machine Learning'
    )
    metrics = [
        MetricScore(name='faithfulness', score=0.9),
        MetricScore(name='coherence', score=0.85),
        MetricScore(name='relevancy', score=0.95),
    ]
    result = TestResult(test_case=test_case, score_results=metrics)

    eval_result = EvaluationResult(
        run_id='run-002',
        evaluation_name='multi-metric-test',
        timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        results=[result],
    )

    dataset_df, metrics_df = eval_result.to_normalized_dataframes()

    # One dataset item row (by_alias=True uses dataset_id)
    assert len(dataset_df) == 1
    assert dataset_df['dataset_id'].iloc[0] == 'test-2'

    # Three metric rows, all with same FK
    assert len(metrics_df) == 3
    assert all(metrics_df['dataset_id'] == 'test-2')

    # Verify FK relationship
    assert set(metrics_df['dataset_id'].dropna()).issubset(
        set(dataset_df['dataset_id'])
    )


def test_to_normalized_dataframes_multiple_items():
    """Test normalized dataframes with multiple items, each with metrics."""
    test_case_1 = DatasetItem(id='item-1', query='Q1', expected_output='A1')
    test_case_2 = DatasetItem(id='item-2', query='Q2', expected_output='A2')

    result_1 = TestResult(
        test_case=test_case_1, score_results=[MetricScore(name='m1', score=0.8)]
    )
    result_2 = TestResult(
        test_case=test_case_2, score_results=[MetricScore(name='m1', score=0.9)]
    )

    eval_result = EvaluationResult(
        run_id='run-003',
        evaluation_name='multi-item-test',
        timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        results=[result_1, result_2],
    )

    dataset_df, metrics_df = eval_result.to_normalized_dataframes()

    # Two dataset item rows (no duplication, by_alias=True uses dataset_id)
    assert len(dataset_df) == 2
    assert len(dataset_df) == dataset_df['dataset_id'].nunique()

    # Two metric rows
    assert len(metrics_df) == 2

    # Verify FK relationship
    assert set(metrics_df['dataset_id'].dropna()).issubset(
        set(dataset_df['dataset_id'])
    )


def test_to_normalized_dataframes_empty_test_case():
    """Test handling of None test_case."""
    metric = MetricScore(name='robustness', score=0.75)
    result = TestResult(test_case=None, score_results=[metric])

    eval_result = EvaluationResult(
        run_id='run-004',
        evaluation_name='null-test',
        timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        results=[result],
    )

    dataset_df, metrics_df = eval_result.to_normalized_dataframes()

    # Empty dataset items table (no test case to extract)
    assert len(dataset_df) == 0

    # Metric row exists with None FK (by_alias=True uses dataset_id)
    assert len(metrics_df) == 1
    assert metrics_df['dataset_id'].iloc[0] is None


def test_to_normalized_dataframes_column_ordering():
    """Verify column order in output DataFrames."""
    test_case = DatasetItem(id='test-5', query='Q?', expected_output='A.')
    metric = MetricScore(name='test_metric', score=0.5)
    result = TestResult(test_case=test_case, score_results=[metric])

    eval_result = EvaluationResult(
        run_id='run-005',
        evaluation_name='order-test',
        timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        results=[result],
    )

    dataset_df, metrics_df = eval_result.to_normalized_dataframes()

    # Verify 'dataset_id' is first column in dataset table (by_alias=True)
    assert dataset_df.columns[0] == 'dataset_id'

    # Verify expected columns exist in metrics table
    assert 'run_id' in metrics_df.columns
    assert 'metric_name' in metrics_df.columns
    assert 'metric_score' in metrics_df.columns
    assert 'dataset_id' in metrics_df.columns


def test_to_normalized_dataframes_rename_columns():
    """Verify column renaming works correctly."""
    test_case = DatasetItem(id='test-6', query='Q', expected_output='A')
    metric = MetricScore(name='test', score=0.5, type='metric')
    result = TestResult(test_case=test_case, score_results=[metric])

    eval_result = EvaluationResult(
        run_id='run-006',
        evaluation_name='rename-test',
        timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        results=[result],
    )

    # With renaming (default)
    _, metrics_df_renamed = eval_result.to_normalized_dataframes(rename_columns=True)
    assert 'metric_name' in metrics_df_renamed.columns
    assert 'metric_score' in metrics_df_renamed.columns
    assert 'metric_type' in metrics_df_renamed.columns
    assert 'name' not in metrics_df_renamed.columns

    # Without renaming (by_alias=True is still default)
    _, metrics_df_original = eval_result.to_normalized_dataframes(rename_columns=False)
    assert 'name' in metrics_df_original.columns
    assert 'score' in metrics_df_original.columns
    assert 'type' in metrics_df_original.columns
    assert 'dataset_id' in metrics_df_original.columns  # FK to DatasetItem


def test_to_normalized_dataframes_by_alias():
    """Verify by_alias controls metric_id column in output."""
    test_case = DatasetItem(id='test-7', query='Q', expected_output='A')
    metric = MetricScore(id='metric-uuid-123', name='test', score=0.5)
    result = TestResult(test_case=test_case, score_results=[metric])

    eval_result = EvaluationResult(
        run_id='run-007',
        evaluation_name='metric-id-test',
        timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        results=[result],
    )

    # With by_alias=True - metric_id and dataset_id columns are used
    _, metrics_df_with_alias = eval_result.to_normalized_dataframes(by_alias=True)
    assert 'metric_id' in metrics_df_with_alias.columns
    assert metrics_df_with_alias['metric_id'].iloc[0] == 'metric-uuid-123'
    assert 'dataset_id' in metrics_df_with_alias.columns
    assert metrics_df_with_alias['dataset_id'].iloc[0] == 'test-7'  # FK to DatasetItem

    # With by_alias=False - no metric_id column, uses 'id' for FK
    _, metrics_df_no_alias = eval_result.to_normalized_dataframes(by_alias=False)
    assert 'metric_id' not in metrics_df_no_alias.columns
    assert 'dataset_id' not in metrics_df_no_alias.columns
    assert metrics_df_no_alias['id'].iloc[0] == 'test-7'  # FK to DatasetItem
