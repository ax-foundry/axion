from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from axion._core.cache.schema import CacheConfig
from axion._core.tracing import clear_tracing_config, configure_tracing
from axion.dataset import DatasetItem
from axion.runners import (
    EvaluationConfig,
    EvaluationRunner,
    evaluation_runner,
)
from axion.runners.summary import MetricSummary
from axion.schema import ErrorConfig, EvaluationResult

clear_tracing_config()
configure_tracing('noop')


class MockTaskOutput(BaseModel):
    """Mock task output with model_dump method"""

    actual_output: str

    def dict(self):
        return {'actual_output': self.actual_output}


class TestExperimentConfig:
    """Test ExperimentConfig dataclass"""

    def test_config_creation_minimal(self):
        """Test creating config with minimal required fields"""

        class Metric:
            pass

        mock_metric = Metric()
        config = EvaluationConfig(
            evaluation_name='test_experiment',
            evaluation_inputs=[],
            scoring_metrics=[mock_metric],
        )

        assert config.evaluation_name == 'test_experiment'
        assert config.evaluation_inputs == []
        assert config.scoring_metrics == [mock_metric]
        assert config.max_concurrent == 5
        assert config.show_progress is True
        assert isinstance(config.summary_generator, MetricSummary)
        assert isinstance(config.cache_config, CacheConfig)
        assert isinstance(config.error_config, ErrorConfig)


class TestExperimentRunner:
    """Test ExperimentRunner class"""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""

        # Create a proper async mock function instead of AsyncMock()
        async def mock_task_func(item):
            return {'actual_output': 'test'}

        return EvaluationConfig(
            evaluation_name='test_experiment',
            evaluation_inputs=[],
            scoring_metrics=[Mock()],
            task=mock_task_func,
            max_concurrent=2,
        )

    @pytest.fixture
    def sample_dataset_items(self):
        """Create sample dataset items for testing"""
        return [
            DatasetItem(id='item1', query='test1'),
            DatasetItem(id='item2', query='test2'),
        ]

    def test_runner_initialization(self, mock_config):
        """Test ExperimentRunner initialization"""
        with (
            patch('axion._core.tracing.init_tracer') as _,
            patch('axion._core.cache.manager.CacheManager'),
        ):
            runner = EvaluationRunner(mock_config)

            assert runner.config == mock_config
            assert runner.run_id.startswith('eval')
            assert runner.task_semaphore.max_concurrent == 2

    def test_process_task_output_with_model_dump(self, mock_config):
        """Test _process_task_output with object that has model_dump"""
        with (
            patch('axion._core.tracing.init_tracer'),
            patch('axion._core.cache.manager.CacheManager'),
        ):
            runner = EvaluationRunner(mock_config)
            task_output = MockTaskOutput(actual_output='test')
            result = runner._process_task_output(task_output, 'item1')

            assert result == {'actual_output': 'test'}

    def test_process_task_output_with_dict(self, mock_config):
        """Test _process_task_output with dictionary input"""
        with (
            patch('axion._core.tracing.init_tracer'),
            patch('axion._core.cache.manager.CacheManager'),
        ):
            runner = EvaluationRunner(mock_config)
            task_output = {'actual_output': 'test'}

            result = runner._process_task_output(task_output, 'item1')

            assert result == {'actual_output': 'test'}

    def test_process_task_output_with_key_mapping(self, mock_config):
        """Test _process_task_output with key mapping"""
        mock_config.scoring_key_mapping = {'actual_output': 'answer'}

        with (
            patch('axion._core.tracing.init_tracer'),
            patch('axion._core.cache.manager.CacheManager'),
        ):
            runner = EvaluationRunner(mock_config)
            # Use the original key in the task output, mapping should transform it
            task_output = {'answer': 'test'}

            result = runner._process_task_output(task_output, 'item1')

            # The result should have the mapped key
            assert result == {'actual_output': 'test'}

    @pytest.mark.asyncio
    async def test_run_single_task_no_cache(self, mock_config, sample_dataset_items):
        """Test _run_single_task without caching"""

        # Create a proper async mock that returns the expected value
        async def mock_task_func(item):
            return {'actual_output': 'test_response'}

        mock_config.task = mock_task_func
        mock_config.cache_config.cache_task = False

        with (
            patch('axion._core.tracing.init_tracer'),
            patch(
                'axion._core.cache.manager.CacheManager'
            ) as mock_cache_manager,
        ):
            mock_cache_manager.return_value = None
            runner = EvaluationRunner(mock_config)
            item = sample_dataset_items[0]

            result = await runner._run_single_task(item)

            assert result.actual_output == 'test_response'

    @pytest.mark.asyncio
    async def test_run_generation_stage(self, mock_config, sample_dataset_items):
        """Test _run_generation_stage"""

        # Create a proper async mock that returns the expected value
        async def mock_task_func(item):
            return {'actual_output': 'test'}

        mock_config.task = mock_task_func
        mock_config.show_progress = False

        with (
            patch('axion._core.tracing.init_tracer'),
            patch('axion._core.cache.manager.CacheManager'),
        ):
            runner = EvaluationRunner(mock_config)

            with patch.object(runner, '_run_single_task') as mock_run_single:
                mock_run_single.side_effect = lambda item: item
                results = await runner._run_generation_stage(
                    sample_dataset_items, return_result=True
                )
                assert len(results) == 2
                assert mock_run_single.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_full_workflow(self, mock_config):
        """Test the complete execute workflow"""
        mock_config.evaluation_inputs = [DatasetItem(id='test', query='test')]

        # Create a proper async mock that returns the expected value
        async def mock_task_func(item):
            return {'actual_output': 'test'}

        mock_config.task = mock_task_func
        runner = EvaluationRunner(mock_config)

        result = await runner.execute()
        assert isinstance(result, EvaluationResult)
        assert result.evaluation_name == 'test_experiment'
        assert result.run_id.startswith('eval')


class TestExperimentRunnerFunction:
    """Test the experiment_runner function"""

    @patch('axion._core.asyncio.run_async_function')
    @patch('axion._core.tracing.init_tracer')
    def test_experiment_runner_with_all_params(self, mock_init_tracer, mock_run_async):
        """Test experiment_runner function with all parameters"""

        # Create a proper async mock function instead of AsyncMock()
        async def mock_task_func(item):
            return {'actual_output': 'test'}

        mock_summary = Mock()
        mock_cache_config = CacheConfig()
        mock_error_config = ErrorConfig()
        mock_evaluation_result = Mock(spec=EvaluationResult)
        mock_run_async.return_value = mock_evaluation_result

        result = evaluation_runner(
            evaluation_inputs=[DatasetItem(id='test', query='test')],
            scoring_metrics=[Mock()],
            evaluation_name='full_test',
            task=mock_task_func,
            scoring_key_mapping={'new': 'old'},
            evaluation_description='Test description',
            evaluation_metadata={'key': 'value'},
            max_concurrent=10,
            summary_generator=mock_summary,
            cache_config=mock_cache_config,
            error_config=mock_error_config,
            thresholds={'metric1': 0.8},
            show_progress=False,
            dataset_name='test_dataset',
        )

        # Verify the result is returned
        assert isinstance(result, EvaluationResult)

    @patch('axion._core.asyncio.run_async_function')
    @patch('axion._core.tracing.init_tracer')
    def test_experiment_runner_sync_execution(self, mock_init_tracer, mock_run_async):
        """Test that experiment_runner is truly synchronous"""
        mock_evaluation_result = Mock(spec=EvaluationResult)
        mock_run_async.return_value = mock_evaluation_result

        from dataclasses import dataclass

        @dataclass
        class MockMetricResult:
            """Mock metric result for testing."""

            score: float
            explanation: str = 'Mock explanation'

        class MockAxionMetricTest:
            """Mock AXION metric for testing."""

            __module__ = 'axion.metrics.something'

            def __init__(self, score_value: float = 0.8):
                self.score_value = score_value
                self.threshold = 0.5

            async def execute(self, input_data: DatasetItem):
                return MockMetricResult(
                    score=self.score_value, explanation='Mock explanation'
                )

        # This should not require await and should return immediately
        result = evaluation_runner(
            evaluation_inputs=[DatasetItem(id='sync_test', actual_output='test')],
            scoring_metrics=[MockAxionMetricTest()],
            evaluation_name='sync_test',
        )

        assert isinstance(result, EvaluationResult)
