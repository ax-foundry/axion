import asyncio
from dataclasses import dataclass
from typing import List
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from axion._core.tracing import configure_tracing
from axion.dataset import DatasetItem
from axion.runners.metric import (
    AxionRunner,
    BaseMetricRunner,
    DeepEvalRunner,
    MetricRunner,
    MetricRunnerFactory,
    RagasRunner,
)
from axion.runners.summary import MetricSummary
from axion.schema import ErrorConfig, MetricScore, TestResult

configure_tracing('noop', force=True)


# Mock metric implementations for testing
class MockAXIONMetric:
    """Mock AXION metric for testing."""

    def __init__(
        self,
        score_value: float = 0.8,
        should_fail: bool = False,
        cost_estimate: float = 0.01,
    ):
        self.score_value = score_value
        self.should_fail = should_fail
        self.threshold = 0.5
        self.cost_estimate = cost_estimate
        self.name = 'MockAXIONMetric'

    async def execute(self, input_data: DatasetItem, **kwargs):
        if self.should_fail:
            raise RuntimeError('Mock metric execution failed')

        return MockMetricResult(score=self.score_value, explanation='Mock explanation')


class MockAXIONMetricTest:
    """Mock AXION metric for testing."""

    __module__ = 'axion.metrics.something'

    def __init__(self, score_value: float = 0.8):
        self.score_value = score_value
        self.threshold = 0.5
        self.name = 'MockAXIONMetric'

    async def execute(self, input_data: DatasetItem, **kwargs):
        return MockMetricResult(score=self.score_value, explanation='Mock explanation')


class MockRagasMetric:
    """Mock Ragas metric for testing."""

    def __init__(
        self,
        score_value: float = 0.75,
        should_fail: bool = False,
        cost_estimate: float = 0.01,
    ):
        self.score_value = score_value
        self.should_fail = should_fail
        self.threshold = 0.6
        self.cost_estimate = cost_estimate
        self.name = 'MockRagasMetric'

    async def single_turn_ascore(self, sample):
        if self.should_fail:
            raise RuntimeError('Mock Ragas metric execution failed')
        return self.score_value


class MockRagasMetricTest:
    """Mock Ragas metric for testing."""

    __module__ = 'ragas.something'

    def __init__(self, score_value: float = 0.75, cost_estimate: float = 0.01):
        self.score_value = score_value
        self.threshold = 0.6
        self.cost_estimate = cost_estimate
        self.name = 'MockRagasMetricTest'

    async def single_turn_ascore(self, sample):
        return self.score_value


class MockDeepEvalMetric:
    """Mock DeepEval metric for testing."""

    def __init__(self, score_value: float = 0.9, should_fail: bool = False):
        self.score_value = score_value
        self.should_fail = should_fail
        self.threshold = 0.7
        self.score = None
        self.reason = 'Mock DeepEval reason'

    async def a_measure(self, test_case, _show_indicator=False):
        if self.should_fail:
            raise RuntimeError('Mock DeepEval metric execution failed')
        self.score = self.score_value


class MockDeepEvalMetricTest:
    """Mock DeepEval metric for testing."""

    __module__ = 'deepeval.something'

    def __init__(self, score_value: float = 0.9):
        self.score_value = score_value
        self.threshold = 0.7
        self.score = None
        self.reason = 'Mock DeepEval reason'

    async def a_measure(self, test_case, _show_indicator=False):
        self.score = self.score_value


@dataclass
class MockMetricResult:
    """Mock metric result for testing."""

    score: float
    explanation: str = 'Mock explanation'


class MockCacheManager:
    """Mock cache manager for testing."""

    def __init__(self):
        self.cache = {}

    def get(self, key: str):
        return self.cache.get(key)

    def set(self, key: str, value):
        self.cache[key] = value


class MockSummaryGenerator:
    """Mock summary generator for testing."""

    def __init__(self):
        self.displayed_results = None
        self.displayed_time = None

    def display(self, results: List[TestResult], execution_time: float):
        self.displayed_results = results
        self.displayed_time = execution_time


# Custom test metric runners for testing
@MetricRunnerFactory.register('test_metric')
class MockTestMetricRunner(BaseMetricRunner):
    """Mock test metric runner for testing."""

    _name = 'test_metric'

    async def execute(self, input_data: DatasetItem, **kwargs) -> MetricScore:
        return MetricScore(
            id=input_data.id,
            name=self.metric_name,
            score=0.8,
            threshold=self.threshold,
            passed=self._has_passed(0.8),
            explanation='Test metric result',
            source=self.source,
        )


@MetricRunnerFactory.register('slow_metric')
class MockSlowMetricRunner(BaseMetricRunner):
    """Mock slow metric runner for testing concurrency."""

    _name = 'slow_metric'

    async def execute(self, input_data: DatasetItem, **kwargs) -> MetricScore:
        await asyncio.sleep(0.1)  # Simulate slow execution
        return MetricScore(
            id=input_data.id,
            name=self.metric_name,
            score=0.7,
            threshold=self.threshold,
            passed=self._has_passed(0.7),
            explanation='Slow metric result',
            source=self.source,
        )


@MetricRunnerFactory.register('failing_metric')
class MockFailingMetricRunner(BaseMetricRunner):
    """Mock failing metric runner for error testing."""

    _name = 'failing_metric'

    async def execute(self, input_data: DatasetItem, **kwargs) -> MetricScore:
        raise RuntimeError(f'Metric failed for input: {input_data.id}')


class TestMetricScore:
    """Test suite for MetricScore model."""

    def test_init_with_defaults(self):
        """Test MetricScore initialization with minimal fields."""
        score = MetricScore(
            id='test_id', name='test_metric', score=0.8, source='test_source'
        )

        assert score.id == 'test_id'
        assert score.name == 'test_metric'
        assert score.score == 0.8
        assert score.source == 'test_source'
        assert score.threshold is None
        assert score.passed is None
        assert score.explanation is None

    def test_init_with_all_fields(self):
        """Test MetricScore initialization with all fields."""
        score = MetricScore(
            id='test_id',
            name='test_metric',
            score=0.85,
            threshold=0.7,
            passed=True,
            explanation='Test explanation',
            source='test_source',
        )

        assert score.id == 'test_id'
        assert score.name == 'test_metric'
        assert score.score == 0.85
        assert score.threshold == 0.7
        assert score.passed is True
        assert score.explanation == 'Test explanation'
        assert score.source == 'test_source'


class TestErrorConfig:
    """Test suite for ErrorConfig."""

    def test_default_error_config(self):
        """Test default ErrorConfig values."""
        config = ErrorConfig()
        assert config.ignore_errors is True
        assert config.skip_on_missing_params is False

    def test_custom_error_config(self):
        """Test custom ErrorConfig values."""
        config = ErrorConfig(ignore_errors=False, skip_on_missing_params=True)
        assert config.ignore_errors is False
        assert config.skip_on_missing_params is True


class TestMetricRunner:
    """Test suite for MetricRunner class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear registry before each test
        MetricRunnerFactory._registry.clear()

        # Re-register test runners
        MetricRunnerFactory.register('test_metric')(MockTestMetricRunner)
        MetricRunnerFactory.register('slow_metric')(MockSlowMetricRunner)
        MetricRunnerFactory.register('failing_metric')(MockFailingMetricRunner)

    def test_init_with_single_metric(self):
        """Test MetricRunner initialization with single metric."""
        mock_metric = MockAXIONMetric()
        runner = MetricRunner(metrics=[mock_metric])

        assert runner.name == 'MetricRunner'
        assert runner.max_concurrent == 5
        assert len(runner.executors) == 1
        assert runner.thresholds is None
        assert isinstance(runner.summary_generator, MetricSummary)

    def test_init_with_multiple_metrics(self):
        """Test MetricRunner initialization with multiple metrics."""
        metrics = [MockAXIONMetric(), MockRagasMetric(), MockDeepEvalMetric()]
        thresholds = {'MockAXIONMetric': 0.6, 'MockRagasMetric': 0.7}

        runner = MetricRunner(metrics=metrics, thresholds=thresholds, max_concurrent=10)

        assert len(runner.executors) == 3
        assert runner.max_concurrent == 10
        assert runner.thresholds == thresholds

    def test_init_with_cache_manager(self):
        """Test MetricRunner initialization with cache manager."""
        mock_metric = MockAXIONMetric()
        cache_manager = MockCacheManager()

        runner = MetricRunner(metrics=[mock_metric], cache_manager=cache_manager)

        assert runner.cache_manager is cache_manager

    def test_init_with_custom_summary_generator(self):
        """Test MetricRunner initialization with custom summary generator."""
        mock_metric = MockAXIONMetric()
        summary_gen = MockSummaryGenerator()

        runner = MetricRunner(metrics=[mock_metric], summary_generator=summary_gen)

        assert runner.summary_generator is summary_gen

    def test_register_decorator(self):
        """Test the register decorator."""

        @MetricRunnerFactory.register('custom_metric')
        class CustomMetricRunner(BaseMetricRunner):
            _name = 'custom_metric'

            async def execute(self, input_data: DatasetItem, **kwargs) -> MetricScore:
                return MetricScore(
                    id=input_data.id,
                    name='CustomMetric',
                    score=0.5,
                    source='custom_metric',
                )

        assert 'custom_metric' in MetricRunnerFactory._registry
        assert MetricRunnerFactory._registry['custom_metric'] == CustomMetricRunner

    def test_get_metric_type(self):
        """Test metric type detection."""

        class UnknownAxionMetric:
            __module__ = 'unknown.evaluation.metrics.something'

        _ = MetricRunner(metrics=[])
        metric = UnknownAxionMetric()
        assert MetricRunnerFactory._get_metric_type(metric) == 'axion'

    def test_available_types_property(self):
        """Test available_types property."""
        runner = MetricRunner(metrics=[])

        # Should include registered types
        available = runner.available_types
        assert 'test_metric' in available
        assert 'slow_metric' in available
        assert 'failing_metric' in available

    @pytest.mark.asyncio
    async def test_execute_batch_with_single_metric(self):
        """Test batch execution with single metric."""
        mock_metric = MockAXIONMetric()
        runner = MetricRunner(metrics=[mock_metric])

        dataset_items = [
            DatasetItem(id='item1', query='test query 1', actual_output='response 1'),
            DatasetItem(id='item2', query='test query 2', actual_output='response 2'),
        ]

        results = await runner.execute_batch(dataset_items, show_progress=False)
        print(results)

        assert len(results) == 2
        assert all(isinstance(r, TestResult) for r in results)
        assert len(results[0].score_results) == 1
        assert results[0].test_case.id == 'item1'
        assert results[1].test_case.id == 'item2'

    @pytest.mark.asyncio
    async def test_execute_batch_with_multiple_metrics(self):
        """Test batch execution with multiple metrics."""
        metrics = [MockAXIONMetric(), MockRagasMetric()]
        runner = MetricRunner(metrics=metrics)

        dataset_items = [
            DatasetItem(id='item1', query='test query', actual_output='response')
        ]

        results = await runner.execute_batch(dataset_items, show_progress=False)

        assert len(results) == 1
        assert len(results[0].score_results) == 2  # Two metrics

    @pytest.mark.asyncio
    async def test_execute_batch_with_dataframe_use_default_name(self):
        """Test batch execution with DataFrame input."""
        mock_metric = MockAXIONMetric()
        runner = MetricRunner(metrics=[mock_metric])

        df = pd.DataFrame(
            {
                'id': ['df1', 'df2'],
                'query': ['query 1', 'query 2'],
                'actual_output': ['response 1', 'response 2'],
            }
        )

        await runner.execute_batch(df, show_progress=False)

    @pytest.mark.asyncio
    async def test_execute_batch_with_dataframe(self):
        """Test batch execution with DataFrame input."""
        mock_metric = MockAXIONMetric()
        runner = MetricRunner(metrics=[mock_metric], dataset_name='test')

        df = pd.DataFrame(
            {
                'id': ['df1', 'df2'],
                'query': ['query 1', 'query 2'],
                'actual_output': ['response 1', 'response 2'],
            }
        )

        results = await runner.execute_batch(df, show_progress=False)

        assert len(results) == 2
        assert results[0].test_case.id == 'df1'
        assert results[1].test_case.id == 'df2'

    @pytest.mark.asyncio
    async def test_execute_batch_with_empty_input(self):
        """Test batch execution with empty input."""
        mock_metric = MockAXIONMetric()
        runner = MetricRunner(metrics=[mock_metric])

        results = await runner.execute_batch([], show_progress=False)
        assert results == []

    @pytest.mark.asyncio
    async def test_execute_batch_with_no_executors(self):
        """Test batch execution with no executors."""
        runner = MetricRunner(metrics=[])

        dataset_items = [
            DatasetItem(id='item1', query='test query', actual_output='response')
        ]

        results = await runner.execute_batch(dataset_items, show_progress=False)
        assert results == []

    def test_display_method(self):
        """Test display method."""
        mock_metric = MockAXIONMetric()
        _ = MetricRunner(metrics=[mock_metric])

        # Should not raise any exceptions
        # Note: This would need proper mocking of the display system in real tests
        try:
            MetricRunner.display()
        except ImportError:
            # Expected if display modules aren't available
            pass

    @pytest.mark.asyncio
    async def test_caching_functionality(self):
        """Test caching functionality."""
        mock_metric = MockAXIONMetric()
        cache_manager = MockCacheManager()
        runner = MetricRunner(metrics=[mock_metric], cache_manager=cache_manager)

        dataset_item = DatasetItem(
            id='cached_item', query='test', actual_output='response'
        )

        # First execution - should hit the metric
        results1 = await runner.execute_batch([dataset_item], show_progress=False)

        # Second execution - should use cache
        results2 = await runner.execute_batch([dataset_item], show_progress=False)

        assert len(results1) == 1
        assert len(results2) == 1
        assert len(cache_manager.cache) > 0  # Cache should have entries

    @pytest.mark.asyncio
    async def test_skip_on_missing_params(self):
        """Test skipping metrics when parameters are missing."""

        @MetricRunnerFactory.register('param_dependent')
        class ParamDependentRunner(BaseMetricRunner):
            _name = 'param_dependent'
            required_params = {'required_field'}

            async def execute(self, input_data: DatasetItem, **kwargs) -> MetricScore:
                return MetricScore(
                    id=input_data.id,
                    name='ParamDependent',
                    score=0.8,
                    source='param_dependent',
                )

        mock_metric = MockAXIONMetric()
        error_config = ErrorConfig(skip_on_missing_params=True)
        runner = MetricRunner(metrics=[mock_metric], error_config=error_config)

        # Add executor manually for this test
        param_executor = ParamDependentRunner(mock_metric)
        runner.executors.append(param_executor)

        # Dataset item without required field
        dataset_item = DatasetItem(
            id='test_item', query='test', actual_output='response'
        )
        results = await runner.execute_batch([dataset_item], show_progress=False)

        assert len(results) == 1
        # Should have fewer score results due to skipped metric
        assert len(results[0].score_results) < len(runner.executors)


class TestBaseMetricRunner:
    """Test suite for BaseMetricRunner abstract class."""

    def test_init(self):
        """Test BaseMetricRunner initialization."""
        mock_metric = MockAXIONMetric()
        runner = MockTestMetricRunner(mock_metric, threshold=0.7)

        assert runner.metric is mock_metric
        assert runner.threshold == 0.7

    def test_init_with_metric_threshold(self):
        """Test initialization using metric's default threshold."""
        mock_metric = MockAXIONMetric()
        mock_metric.threshold = 0.6
        runner = MockTestMetricRunner(mock_metric)

        assert runner.threshold == 0.6

    def test_abstract_execute_method(self):
        """Test that BaseMetricRunner is abstract."""
        mock_metric = MockAXIONMetric()

        # Should not be able to instantiate BaseMetricRunner directly
        with pytest.raises(TypeError):
            BaseMetricRunner(mock_metric)

    def test_prepare_retrieved_content_with_string(self):
        """Test _prepare_retrieved_content with string input."""
        content = 'single document'
        result = BaseMetricRunner._prepare_retrieved_content(content)

        assert result == ['single document']

    def test_prepare_retrieved_content_with_list(self):
        """Test _prepare_retrieved_content with list input."""
        content = ['doc1', 'doc2']
        result = BaseMetricRunner._prepare_retrieved_content(content)

        assert result == ['doc1', 'doc2']

    def test_prepare_retrieved_content_with_none(self):
        """Test _prepare_retrieved_content with None input."""
        result = BaseMetricRunner._prepare_retrieved_content(None)

        assert result == []

    def test_has_passed_with_valid_score(self):
        """Test _has_passed with valid score above threshold."""
        mock_metric = MockAXIONMetric()
        runner = MockTestMetricRunner(mock_metric, threshold=0.5)

        assert runner._has_passed(0.7) is True
        assert runner._has_passed(0.3) is False
        assert runner._has_passed(0.5) is True  # Equal to threshold

    def test_has_passed_with_invalid_inputs(self):
        """Test _has_passed with invalid inputs."""
        mock_metric = MockAXIONMetric()
        runner = MockTestMetricRunner(mock_metric, threshold=0.5)

        assert runner._has_passed(None) is None
        assert runner._has_passed(np.nan) is None

        # Test with no threshold
        runner.threshold = None
        assert runner._has_passed(0.8) is None

    def test_create_error_score(self):
        """Test _create_error_score method."""
        mock_metric = MockAXIONMetric()
        runner = MockTestMetricRunner(mock_metric)
        error = RuntimeError('Test error')

        error_score = runner._create_error_score('test_id', error)

        assert isinstance(error_score, MetricScore)
        assert error_score.id == 'test_id'
        assert error_score.name == runner.metric_name
        assert np.isnan(error_score.score)
        assert 'Error executing metric: Test error' in error_score.explanation
        assert error_score.source == runner.source

    def test_metric_name_property(self):
        """Test metric_name property."""
        mock_metric = MockAXIONMetric()
        runner = MockTestMetricRunner(mock_metric)

        assert runner.metric_name == 'MockAXIONMetric'

    def test_source_property(self):
        """Test source property."""
        mock_metric = MockAXIONMetric()
        runner = MockTestMetricRunner(mock_metric)

        assert runner.source == 'test_metric'


class TestAxionRunner:
    """Test suite for AxionRunner."""

    def setup_method(self):
        """Set up test fixtures."""
        MetricRunnerFactory._registry.clear()
        MetricRunnerFactory.register('axion')(AxionRunner)

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful AXION metric execution."""
        mock_metric = MockAXIONMetric(score_value=0.85)
        runner = AxionRunner(mock_metric, threshold=0.7)

        input_data = DatasetItem(
            id='test_id', query='test query', actual_output='response'
        )
        result = await runner.execute(input_data)

        assert isinstance(result, MetricScore)
        assert result.id == 'test_id'
        assert result.name == 'MockAXIONMetric'
        assert result.score == 0.85
        assert result.threshold == 0.7
        assert result.passed is True
        assert result.explanation == 'Mock explanation'
        assert result.source == 'axion'

    @pytest.mark.asyncio
    async def test_failed_execution(self):
        """Test failed AXION metric execution."""
        mock_metric = MockAXIONMetric(should_fail=True)
        runner = AxionRunner(mock_metric)

        input_data = DatasetItem(
            id='test_id', query='test query', actual_output='response'
        )
        result = await runner.execute(input_data)

        assert isinstance(result, MetricScore)
        assert result.id == 'test_id'
        assert np.isnan(result.score)
        assert 'Error executing metric' in result.explanation

    @pytest.mark.asyncio
    async def test_execution_with_dict_input(self):
        """Test execution with dictionary input."""
        mock_metric = MockAXIONMetric(score_value=0.75)
        runner = AxionRunner(mock_metric)

        input_dict = {
            'id': 'test_id',
            'query': 'test query',
            'actual_output': 'response',
        }
        result = await runner.execute(input_dict)

        assert isinstance(result, MetricScore)
        assert result.id == 'test_id'
        assert result.score == 0.75


class TestRagasRunner:
    """Test suite for RagasRunner."""

    def setup_method(self):
        """Set up test fixtures."""
        MetricRunnerFactory._registry.clear()
        MetricRunnerFactory.register('ragas')(RagasRunner)

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful Ragas metric execution."""
        with patch('ragas.SingleTurnSample') as _:
            mock_metric = MockRagasMetric(score_value=0.8)
            runner = RagasRunner(mock_metric, threshold=0.6)

            input_data = DatasetItem(
                id='test_id',
                query='test query',
                actual_output='response',
                expected_output='expected',
                retrieved_content=['doc1', 'doc2'],
            )

            result = await runner.execute(input_data)

            assert isinstance(result, MetricScore)
            assert result.id == 'test_id'
            assert result.name == 'MockRagasMetric'
            assert result.score == 0.8
            assert result.threshold == 0.6
            assert result.passed is True
            assert result.source == 'ragas'

    @pytest.mark.asyncio
    async def test_missing_actual_output(self):
        """Test execution with missing actual_output."""
        mock_metric = MockRagasMetric()
        runner = RagasRunner(mock_metric)

        input_data = DatasetItem(id='test_id', query='test query')

        # Should raise validation error for missing actual_output
        with pytest.raises(Exception):  # EvaluationValidation should raise
            await runner.execute(input_data)

    @pytest.mark.asyncio
    async def test_failed_execution(self):
        """Test failed Ragas metric execution."""
        with patch('ragas.SingleTurnSample'):
            mock_metric = MockRagasMetric(should_fail=True)
            runner = RagasRunner(mock_metric)

            input_data = DatasetItem(
                id='test_id', query='test', actual_output='response'
            )
            result = await runner.execute(input_data)

            assert isinstance(result, MetricScore)
            assert np.isnan(result.score)
            assert 'Error executing metric' in result.explanation


class TestDeepEvalRunner:
    """Test suite for DeepEvalRunner."""

    def setup_method(self):
        """Set up test fixtures."""
        MetricRunnerFactory._registry.clear()
        MetricRunnerFactory.register('deepeval')(DeepEvalRunner)

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful DeepEval metric execution."""
        with patch('deepeval.test_case.LLMTestCase') as _:
            mock_metric = MockDeepEvalMetric(score_value=0.9)
            runner = DeepEvalRunner(mock_metric, threshold=0.8)

            input_data = DatasetItem(
                id='test_id',
                query='test query',
                actual_output='response',
                expected_output='expected',
                retrieved_content=['doc1', 'doc2'],
            )

            result = await runner.execute(input_data)

            assert isinstance(result, MetricScore)
            assert result.id == 'test_id'
            assert result.name == 'MockDeepEvalMetric'
            assert result.score == 0.9
            assert result.threshold == 0.8
            assert result.passed is True
            assert result.explanation == 'Mock DeepEval reason'
            assert result.source == 'deepeval'

    @pytest.mark.asyncio
    async def test_failed_execution(self):
        """Test failed DeepEval metric execution."""
        with patch('deepeval.test_case.LLMTestCase'):
            mock_metric = MockDeepEvalMetric(should_fail=True)
            runner = DeepEvalRunner(mock_metric)

            input_data = DatasetItem(
                id='test_id', query='test', actual_output='response'
            )
            result = await runner.execute(input_data)

            assert isinstance(result, MetricScore)
            assert np.isnan(result.score)
            assert 'Error executing metric' in result.explanation


class TestMetricRunnerIntegration:
    """Integration tests for MetricRunner with multiple metric types."""

    def setup_method(self):
        """Set up test fixtures."""
        MetricRunnerFactory._registry.clear()
        MetricRunnerFactory.register('axion')(AxionRunner)
        MetricRunnerFactory.register('ragas')(RagasRunner)
        MetricRunnerFactory.register('deepeval')(DeepEvalRunner)

    @pytest.mark.asyncio
    async def test_mixed_metric_types_execution(self):
        """Test execution with multiple metric types."""
        with (
            patch('ragas.SingleTurnSample'),
            patch('deepeval.test_case.LLMTestCase'),
        ):
            metrics = [
                MockAXIONMetricTest(score_value=0.8),
                MockRagasMetricTest(score_value=0.7),
                MockDeepEvalMetricTest(score_value=0.9),
            ]

            runner = MetricRunner(metrics=metrics)

            dataset_item = DatasetItem(
                id='test_id',
                query='test query',
                actual_output='response',
                expected_output='expected',
            )

            results = await runner.execute_batch([dataset_item], show_progress=False)

            assert len(results) == 1
            assert len(results[0].score_results) == 3

            # Check that all metric types were executed
            score_sources = [score.source for score in results[0].score_results]
            assert 'axion' in score_sources
            assert 'ragas' in score_sources
            assert 'deepeval' in score_sources


class TestMetricRunnerErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        MetricRunnerFactory._registry.clear()
        MetricRunnerFactory.register('test_metric')(MockTestMetricRunner)
        MetricRunnerFactory.register('failing_metric')(MockFailingMetricRunner)

    @pytest.mark.asyncio
    async def test_invalid_dataset_input(self):
        """Test handling of invalid dataset input."""
        mock_metric = MockAXIONMetric()
        runner = MetricRunner(metrics=[mock_metric])

        # Test with invalid input type
        with pytest.raises(Exception):
            await runner.execute_batch('invalid_input', show_progress=False)

    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        """Test cache key generation."""
        mock_metric = MockAXIONMetric()
        runner = MetricRunner(metrics=[mock_metric])

        # Create a mock executor to test cache key generation
        executor = AxionRunner(mock_metric)
        dataset_item = DatasetItem(id='test_id', query='test')

        cache_key = runner._get_cache_key(executor, dataset_item)

        assert isinstance(cache_key, str)
        assert len(cache_key) == 64  # SHA256 hash length

        # Same input should generate same key
        cache_key2 = runner._get_cache_key(executor, dataset_item)
        assert cache_key == cache_key2

        # Different input should generate different key
        different_item = DatasetItem(id='different_id', query='test')
        different_key = runner._get_cache_key(executor, different_item)
        assert cache_key != different_key


class TestMetricRunnerPerformance:
    """Performance-related tests."""

    def setup_method(self):
        """Set up test fixtures."""
        MetricRunnerFactory._registry.clear()
        MetricRunnerFactory.register('test_metric')(MockTestMetricRunner)
        MetricRunnerFactory.register('slow_metric')(MockSlowMetricRunner)

    @pytest.mark.asyncio
    async def test_large_batch_execution(self):
        """Test execution with large batch of dataset items."""
        mock_metric = MockAXIONMetric()
        runner = MetricRunner(metrics=[mock_metric], max_concurrent=10)

        dataset_items = [
            DatasetItem(
                id=f'item_{i}', query=f'query {i}', actual_output=f'response {i}'
            )
            for i in range(50)
        ]

        results = await runner.execute_batch(dataset_items, show_progress=False)

        assert len(results) == 50
        assert all(isinstance(r, TestResult) for r in results)
        assert all(len(r.score_results) == 1 for r in results)


# Fixtures for common test data
@pytest.fixture
def sample_dataset_items():
    """Fixture providing sample dataset items."""
    return [
        DatasetItem(
            id='item_1',
            query='What is artificial intelligence?',
            actual_output='AI is a field of computer science.',
            expected_output='AI is the simulation of human intelligence.',
            retrieved_content=['AI definition doc', 'Machine learning overview'],
        ),
        DatasetItem(
            id='item_2',
            query='How does machine learning work?',
            actual_output='ML uses algorithms to learn from data.',
            expected_output='ML is a subset of AI that learns patterns.',
            retrieved_content=['ML algorithms doc', 'Neural networks guide'],
        ),
        DatasetItem(
            id='item_3',
            query='What is deep learning?',
            actual_output='Deep learning uses neural networks.',
            expected_output='Deep learning is a subset of machine learning.',
            retrieved_content=['Deep learning basics', 'CNN architectures'],
        ),
    ]


@pytest.fixture
def sample_metrics():
    """Fixture providing sample metrics."""
    return [
        MockAXIONMetric(score_value=0.8),
        MockRagasMetric(score_value=0.75),
        MockDeepEvalMetric(score_value=0.9),
    ]


@pytest.fixture
def configured_metric_runner(sample_metrics):
    """Fixture providing a pre-configured MetricRunner."""
    MetricRunnerFactory._registry.clear()
    MetricRunnerFactory.register('axion')(AxionRunner)
    MetricRunnerFactory.register('ragas')(RagasRunner)
    MetricRunnerFactory.register('deepeval')(DeepEvalRunner)

    thresholds = {
        'MockAXIONMetric': 0.7,
        'MockRagasMetric': 0.6,
        'MockDeepEvalMetric': 0.8,
    }

    return MetricRunner(
        metrics=sample_metrics,
        thresholds=thresholds,
        max_concurrent=3,
        cache_manager=MockCacheManager(),
        dataset_name='test',
    )


@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame."""
    return pd.DataFrame(
        {
            'id': ['df_1', 'df_2', 'df_3'],
            'query': [
                'What is NLP?',
                'How does reinforcement learning work?',
                'What are transformers?',
            ],
            'actual_output': [
                'NLP processes human language.',
                'RL learns through rewards and penalties.',
                'Transformers are attention-based models.',
            ],
            'expected_output': [
                'NLP is natural language processing.',
                'RL is learning through trial and error.',
                'Transformers revolutionized NLP.',
            ],
        }
    )


class TestWithFixtures:
    """Tests using pytest fixtures."""

    def test_threshold_application(self, sample_metrics):
        """Test that thresholds are correctly applied."""
        thresholds = {
            'MockAXIONMetric': 0.9,  # High threshold
            'MockRagasMetric': 0.5,  # Low threshold
            'MockDeepEvalMetric': 0.95,  # Very high threshold
        }

        runner = MetricRunner(metrics=sample_metrics, thresholds=thresholds)

        # Check that thresholds were applied to executors
        axion_executor = next(
            (e for e in runner.executors if e.source == 'axion'), None
        )
        ragas_executor = next(
            (e for e in runner.executors if e.source == 'ragas'), None
        )
        deepeval_executor = next(
            (e for e in runner.executors if e.source == 'deepeval'), None
        )

        if axion_executor:
            assert axion_executor.threshold == 0.9
        if ragas_executor:
            assert ragas_executor.threshold == 0.5
        if deepeval_executor:
            assert deepeval_executor.threshold == 0.95

    def test_available_types_with_fixtures(self, configured_metric_runner):
        """Test available types property with fixtures."""
        available_types = configured_metric_runner.available_types

        assert 'axion' in available_types
        assert 'ragas' in available_types
        assert 'deepeval' in available_types
