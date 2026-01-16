import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from axion.runners.summary import (
    BaseSummary,
    MetricSummary,
    SimpleSummary,
)


@dataclass
class MockDatasetItem:
    """Mock dataset item for testing"""

    query: str = 'test query'
    expected_output: str = 'test output'


@dataclass
class MockMetricScore:
    """Mock MetricScore for testing"""

    name: str
    score: Optional[float] = None
    threshold: Optional[float] = None
    passed: Optional[bool] = None
    explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MockTestResult:
    """Mock TestResult for testing"""

    test_case: Optional[MockDatasetItem] = None
    score_results: List[MockMetricScore] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class TestBaseSummary:
    """Test the abstract base class"""

    def test_base_summary_not_implemented(self):
        """Test that BaseSummary raises NotImplementedError"""
        base = BaseSummary()
        with pytest.raises(NotImplementedError):
            base.execute([], 0.0)


class TestSimpleSummary:
    """Test SimpleSummary class"""

    @pytest.fixture
    def simple_summary(self):
        return SimpleSummary()

    @pytest.fixture
    def sample_results(self):
        """Create sample test results for testing"""
        return [
            MockTestResult(
                test_case=MockDatasetItem(),
                score_results=[
                    MockMetricScore(name='accuracy', score=0.85, passed=True),
                    MockMetricScore(name='precision', score=0.90, passed=True),
                ],
            ),
            MockTestResult(
                test_case=MockDatasetItem(),
                score_results=[
                    MockMetricScore(name='accuracy', score=0.75, passed=False),
                    MockMetricScore(name='precision', score=0.80, passed=True),
                ],
            ),
        ]

    def test_is_valid_score(self, simple_summary):
        """Test score validation logic"""
        assert simple_summary._is_valid_score(0.5) is True
        assert simple_summary._is_valid_score(0.0) is True
        assert simple_summary._is_valid_score(1.0) is True
        assert simple_summary._is_valid_score(None) is False
        assert simple_summary._is_valid_score(float('nan')) is False

    def test_execute_empty_results(self, simple_summary, capsys):
        """Test execute with empty results"""
        result = simple_summary.execute([], 10.0)
        captured = capsys.readouterr()

        assert result == {}
        assert 'No evaluation data available' in captured.out

    def test_execute_with_results(self, simple_summary, sample_results, capsys):
        """Test execute with sample results"""
        result = simple_summary.execute(sample_results, 25.5)
        captured = capsys.readouterr()

        assert result == {}
        assert 'EVALUATION REPORT' in captured.out
        assert 'Performance Score' in captured.out
        assert 'Consistency Index' in captured.out
        assert 'Samples Analyzed: 2' in captured.out
        assert '00:25' in captured.out  # 25.5 seconds formatted

    def test_execute_with_task_runs(self, simple_summary, sample_results, capsys):
        """Test execute with total_task_runs parameter"""
        _ = simple_summary.execute(sample_results, 10.0, total_task_runs=100)
        captured = capsys.readouterr()

        assert 'Task Runs: 100' in captured.out

    def test_execute_with_invalid_scores(self, simple_summary, capsys):
        """Test execute with invalid scores (None, NaN)"""
        results_with_invalid = [
            MockTestResult(
                score_results=[
                    MockMetricScore(name='metric1', score=None),
                    MockMetricScore(name='metric2', score=float('nan')),
                    MockMetricScore(name='metric3', score=0.8),
                ]
            )
        ]

        result = simple_summary.execute(results_with_invalid, 5.0)
        captured = capsys.readouterr()

        # Should handle invalid scores gracefully and only use valid ones
        assert 'EVALUATION REPORT' in captured.out
        assert result == {}


class TestMetricSummary:
    """Test MetricSummary class"""

    @pytest.fixture
    def metric_summary(self):
        return MetricSummary(show_distribution=True)

    @pytest.fixture
    def metric_summary_no_dist(self):
        return MetricSummary(show_distribution=False)

    @pytest.fixture
    def sample_results(self):
        """Create sample test results for testing"""
        return [
            MockTestResult(
                score_results=[
                    MockMetricScore(name='accuracy', score=0.85, passed=True),
                    MockMetricScore(name='f1_score', score=0.78, passed=False),
                ]
            ),
            MockTestResult(
                score_results=[
                    MockMetricScore(name='accuracy', score=0.92, passed=True),
                    MockMetricScore(name='f1_score', score=0.81, passed=True),
                ]
            ),
        ]

    def test_initialization(self):
        """Test MetricSummary initialization"""
        summary = MetricSummary(show_distribution=False)
        assert summary.show_distribution is False
        assert isinstance(summary.terminal_width, int)
        assert summary.terminal_width > 0

    def test_get_terminal_width(self, metric_summary):
        """Test terminal width detection"""
        width = metric_summary._get_terminal_width()
        assert isinstance(width, int)
        assert width > 0
        assert width <= 120  # Should be capped at 120

    def test_is_valid_score(self, metric_summary):
        """Test score validation logic"""
        assert metric_summary._is_valid_score(0.5) is True
        assert metric_summary._is_valid_score(0.0) is True
        assert metric_summary._is_valid_score(None) is False
        assert metric_summary._is_valid_score(float('nan')) is False

    def test_execute_empty_results(self, metric_summary, capsys):
        """Test execute with empty results"""
        result = metric_summary.execute([], 10.0)
        captured = capsys.readouterr()

        assert result == {}
        assert 'No results to summarize' in captured.out

    def test_execute_with_results(self, metric_summary, sample_results, capsys):
        """Test execute with sample results"""
        result = metric_summary.execute(sample_results, 15.0)
        captured = capsys.readouterr()

        assert isinstance(result, dict)
        assert 'DETAILED METRIC ANALYSIS' in captured.out
        assert 'ACCURACY' in captured.out
        assert 'F1 SCORE' in captured.out

        # Verify that metrics were processed
        assert len(result) == 2  # Should have accuracy and f1_score
        assert 'accuracy' in result
        assert 'f1_score' in result

    def test_execute_with_task_runs(self, metric_summary, sample_results, capsys):
        """Test execute with total_task_runs parameter"""
        result = metric_summary.execute(sample_results, 10.0, total_task_runs=50)
        _ = capsys.readouterr()

        assert isinstance(result, dict)

    def test_execute_with_mixed_valid_invalid_scores(self, metric_summary, capsys):
        """Test execute with mix of valid and invalid scores"""
        mixed_results = [
            MockTestResult(
                score_results=[
                    MockMetricScore(name='metric1', score=0.8, passed=True),
                    MockMetricScore(name='metric2', score=None),  # Invalid
                    MockMetricScore(name='metric3', score=float('nan')),  # Invalid
                    MockMetricScore(
                        name='metric1', score=0.7, passed=False
                    ),  # Same metric, different result
                ]
            )
        ]

        result = metric_summary.execute(mixed_results, 5.0)
        _ = capsys.readouterr()

        assert isinstance(result, dict)
        assert 'metric1' in result
        assert result['metric1']['valid_count'] == 2  # Should count only valid scores
        assert result['metric1']['total_count'] == 2  # Total for metric1


class TestIntegration:
    """Integration tests for both summary classes"""

    def test_both_summaries_with_same_data(self, capsys):
        """Test that both summary classes can process the same data"""
        results = [
            MockTestResult(
                score_results=[
                    MockMetricScore(name='accuracy', score=0.85, passed=True),
                    MockMetricScore(name='precision', score=0.90, passed=True),
                ]
            )
        ]

        simple = SimpleSummary()
        metric = MetricSummary()

        # Test SimpleSummary
        simple_result = simple.execute(results, 10.0)
        simple_output = capsys.readouterr().out

        # Test MetricSummary
        metric_result = metric.execute(results, 10.0)
        metric_output = capsys.readouterr().out

        # Both should handle the same data without errors
        assert isinstance(simple_result, dict)
        assert isinstance(metric_result, dict)
        assert 'EVALUATION REPORT' in simple_output
        assert 'DETAILED METRIC ANALYSIS' in metric_output

    def test_performance_with_large_dataset(self):
        """Test performance with larger dataset"""
        # Create a larger dataset
        large_results = []
        for i in range(100):
            large_results.append(
                MockTestResult(
                    score_results=[
                        MockMetricScore(
                            name=f'metric_{j}',
                            score=0.5 + (i * j * 0.001) % 0.5,
                            passed=True,
                        )
                        for j in range(5)
                    ]
                )
            )

        simple = SimpleSummary()
        metric = MetricSummary()

        # Both should handle large datasets without errors
        start_time = time.time()
        simple_result = simple.execute(large_results, 60.0)
        simple_time = time.time() - start_time

        start_time = time.time()
        metric_result = metric.execute(large_results, 60.0)
        metric_time = time.time() - start_time

        # Performance should be reasonable (less than 5 seconds each)
        assert simple_time < 5.0
        assert metric_time < 5.0
        assert isinstance(simple_result, dict)
        assert isinstance(metric_result, dict)
