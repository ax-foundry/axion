import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import pytest
from axion._core.tracing import configure_tracing
from axion.dataset import DatasetItem
from axion.runners.api import (
    APIResponseData,
    APIRunner,
    BaseAPIRunner,
    RetryConfig,
)

configure_tracing('noop', force=True)


@APIRunner.register('test_api')
class MockTestAPIRunner(BaseAPIRunner):
    """Mock test implementation of API runner."""

    def __init__(
        self,
        config: Dict[str, Any],
        max_concurrent: int = 5,
        show_progress: bool = True,
        retry_config: Optional[Union[RetryConfig, Dict[str, Any]]] = None,
        tracer: Optional[Any] = None,
    ):
        super().__init__(config, max_concurrent, show_progress, retry_config, tracer)

    def execute(self, query: str, **kwargs) -> APIResponseData:
        return APIResponseData(
            query=query,
            actual_output=f'Response to: {query}',
            latency=0.1,
            status='success',
        )


@APIRunner.register('slow_api')
class MockSlowAPIRunner(BaseAPIRunner):
    """Mock slow test implementation for testing concurrency."""

    def __init__(
        self,
        config: Dict[str, Any],
        max_concurrent: int = 5,
        show_progress: bool = True,
        retry_config: Optional[Union[RetryConfig, Dict[str, Any]]] = None,
        tracer: Optional[Any] = None,
    ):
        super().__init__(config, max_concurrent, show_progress, retry_config, tracer)

    def execute(self, query: str, **kwargs) -> APIResponseData:
        import time

        time.sleep(0.1)  # Simulate slow API
        return APIResponseData(
            query=query,
            actual_output=f'Slow response to: {query}',
            latency=0.1,
            status='success',
        )


@APIRunner.register('failing_api')
class MockFailingAPIRunner(BaseAPIRunner):
    """Mock failing test implementation for error testing."""

    def __init__(
        self,
        config: Dict[str, Any],
        max_concurrent: int = 5,
        show_progress: bool = True,
        retry_config: Optional[Union[RetryConfig, Dict[str, Any]]] = None,
        tracer: Optional[Any] = None,
    ):
        super().__init__(config, max_concurrent, show_progress, retry_config, tracer)

    def execute(self, query: str, **kwargs) -> APIResponseData:
        raise RuntimeError(f'API failed for query: {query}')


# Mock implementations of real API runners for testing
class MockMIAWRunner(BaseAPIRunner):
    """Mock MIAW API runner for testing."""

    def __init__(
        self,
        config: Dict[str, Any],
        ignore_responses: List[str] = None,
        max_concurrent: int = 5,
        show_progress: bool = True,
        retry_config: Optional[Union[RetryConfig, Dict[str, Any]]] = None,
        tracer: Optional[Any] = None,
    ):
        super().__init__(config, max_concurrent, show_progress, retry_config, tracer)
        self.ignore_responses = (
            config.pop('ignore_responses', None) or ignore_responses or []
        )

    def execute(self, query: str, **kwargs) -> APIResponseData:
        # Mock MIAW API behavior
        if any(ignore in query for ignore in self.ignore_responses):
            return APIResponseData(actual_output='', status='ignored')

        return APIResponseData(
            query=query,
            actual_output=f'MIAW response to: {query}',
            latency=0.2,
            status='success',
        )


class MockAgentAPIRunner(BaseAPIRunner):
    """Mock Agent API runner for testing."""

    def __init__(
        self,
        config: Dict[str, Any],
        max_concurrent: int = 5,
        show_progress: bool = True,
        retry_config: Optional[Union[RetryConfig, Dict[str, Any]]] = None,
        tracer: Optional[Any] = None,
    ):
        super().__init__(config, max_concurrent, show_progress, retry_config, tracer)
        # Mock API initialization
        self.api_key = config.get('api_key', 'mock_key')

    def execute(self, query: str, **kwargs) -> APIResponseData:
        return APIResponseData(
            query=query,
            actual_output=f'Agent response to: {query}',
            latency=0.15,
            status='success',
        )


class MockPromptTemplateAPIRunner(BaseAPIRunner):
    """Mock Prompt Template API runner for testing."""

    def __init__(
        self,
        config: Dict[str, Any],
        max_concurrent: int = 5,
        parser: Callable = None,
        show_progress: bool = True,
        retry_config: Optional[Union[RetryConfig, Dict[str, Any]]] = None,
        tracer: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(config, max_concurrent, show_progress, retry_config, tracer)
        self.parser = config.pop('parser', None) or parser

    def execute(self, query: str, **kwargs) -> APIResponseData:
        # Mock retrieval and response
        mock_retrieved = ['Retrieved doc 1', 'Retrieved doc 2']
        mock_response = f'Template response to: {query}'

        return APIResponseData(
            query=query,
            actual_output=mock_response,
            retrieved_content=mock_retrieved,
            latency=0.3,
            status='success',
        )


class TestAPIResponseData:
    """Test suite for APIResponseData model."""

    def test_init_with_all_fields(self):
        """Test APIResponseData initialization with all fields."""
        response = APIResponseData(
            query='test query',
            actual_output='test response',
            retrieved_content=['doc1', 'doc2'],
            latency=1.5,
            trace={'step': 1},
            additional_output={'extra': 'data'},
            status='completed',
            timestamp='2024-01-01T12:00:00Z',
        )

        assert response.query == 'test query'
        assert response.actual_output == 'test response'
        assert response.retrieved_content == ['doc1', 'doc2']
        assert response.latency == 1.5
        assert response.trace == {'step': 1}
        assert response.additional_output == {'extra': 'data'}
        assert response.status == 'completed'
        assert response.timestamp == '2024-01-01T12:00:00Z'

    def test_init_with_partial_fields(self):
        """Test APIResponseData initialization with partial fields."""
        response = APIResponseData(
            query='test query', actual_output='partial response', latency=0.5
        )

        assert response.actual_output == 'partial response'
        assert response.latency == 0.5
        assert response.status == 'success'  # Default
        assert response.additional_output == {}  # Default


class TestAPIRunner:
    """Test suite for APIRunner class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear registry before each test
        APIRunner._registry.clear()

        # Re-register test runners
        APIRunner.register('test_api')(MockTestAPIRunner)
        APIRunner.register('slow_api')(MockSlowAPIRunner)
        APIRunner.register('failing_api')(MockFailingAPIRunner)

    def test_init_with_dict_config(self):
        """Test APIRunner initialization with dictionary config."""
        config = {
            'test_api': {'api_key': 'test_key'},
            'slow_api': {'api_key': 'slow_key'},
        }

        runner = APIRunner(config=config)

        assert runner.name == 'APIRunner'
        assert runner.max_concurrent == 5
        assert len(runner.executors) == 0

        _ = runner.get_executor('test_api')
        _ = runner.get_executor('slow_api')
        assert len(runner.executors) == 2

        assert 'test_api' in runner.executors
        assert 'slow_api' in runner.executors

    def test_init_with_no_matching_apis(self):
        """Test APIRunner initialization with no matching APIs."""
        config = {'unknown_api': {'api_key': 'key'}}

        runner = APIRunner(config=config)

        assert len(runner.executors) == 0

    def test_register_decorator(self):
        """Test the register decorator."""

        @APIRunner.register('custom_api')
        class CustomAPIRunner(BaseAPIRunner):
            def __init__(self, config: Dict[str, Any], max_concurrent: int = 5):
                super().__init__(config, max_concurrent)

            def execute(self, query: str, **kwargs) -> APIResponseData:
                return APIResponseData(
                    query='test query', actual_output='custom response'
                )

        assert 'custom_api' in APIRunner._registry
        assert APIRunner._registry['custom_api'] == CustomAPIRunner

    def test_get_executor_existing(self):
        """Test getting an existing executor."""
        config = {'test_api': {'api_key': 'test_key'}}
        runner = APIRunner(config=config)

        executor = runner.get_executor('test_api')
        assert isinstance(executor, MockTestAPIRunner)

    def test_get_executor_non_existing(self):
        """Test getting a non-existing executor raises error."""
        config = {'test_api': {'api_key': 'test_key'}}
        runner = APIRunner(config=config)

        with pytest.raises(ValueError):
            runner.get_executor('non_existing')

    def test_getitem_method(self):
        """Test dictionary-style access to executors."""
        config = {'test_api': {'api_key': 'test_key'}}
        runner = APIRunner(config=config)

        executor = runner['test_api']
        assert isinstance(executor, MockTestAPIRunner)

        with pytest.raises(ValueError):
            _ = runner['non_existing']

    def test_list_available_apis(self):
        """Test listing available APIs."""
        config = {
            'test_api': {'api_key': 'test_key'},
            'slow_api': {'api_key': 'slow_key'},
        }
        runner = APIRunner(config=config)

        apis = runner.list_available_apis()
        assert isinstance(apis, list)
        assert 'test_api' in apis
        assert 'slow_api' in apis
        assert len(apis) == 2

    def test_execute_single(self):
        """Test executing a single query."""
        config = {'test_api': {'api_key': 'test_key'}}
        runner = APIRunner(config=config)

        response = runner.execute('test_api', 'test query')

        assert isinstance(response, APIResponseData)
        assert response.query == 'test query'
        assert response.actual_output == 'Response to: test query'
        assert response.latency == 0.1
        assert response.status == 'success'

    @pytest.mark.asyncio
    async def test_execute_batch(self):
        """Test executing a batch of queries."""
        config = {'test_api': {'api_key': 'test_key'}}
        runner = APIRunner(config=config)

        queries = ['query1', 'query2', 'query3']
        responses = await runner.execute_batch('test_api', queries)

        assert len(responses) == 3
        assert all(isinstance(r, APIResponseData) for r in responses)
        assert responses[0].actual_output == 'Response to: query1'
        assert responses[1].actual_output == 'Response to: query2'
        assert responses[2].actual_output == 'Response to: query3'

    def test_display(self):
        """Test display method."""
        config = {'test_api': {'api_key': 'test_key'}}
        runner = APIRunner(config=config)

        # Should not raise any exceptions
        runner.display()


class TestBaseAPIRunner:
    """Test suite for BaseAPIRunner abstract class."""

    def test_init(self):
        """Test BaseAPIRunner initialization."""
        config = {'api_key': 'test_key'}
        runner = MockTestAPIRunner(config, max_concurrent=3)

        assert runner.config == config
        assert runner.semaphore.max_concurrent == 3

    def test_abstract_execute_method(self):
        """Test that BaseAPIRunner is abstract."""
        config = {'api_key': 'test_key'}

        # Should not be able to instantiate BaseAPIRunner directly
        with pytest.raises(TypeError):
            BaseAPIRunner(config, max_concurrent=5)

    @pytest.mark.asyncio
    async def test_execute_batch_with_strings(self):
        """Test batch execution with string queries."""
        config = {'api_key': 'test_key'}
        runner = MockTestAPIRunner(config, max_concurrent=3)

        queries = ['query1', 'query2']
        responses = await runner.execute_batch(queries)

        assert len(responses) == 2
        assert responses[0].actual_output == 'Response to: query1'
        assert responses[1].actual_output == 'Response to: query2'

    @pytest.mark.asyncio
    async def test_execute_batch_with_dataset_items(self):
        """Test batch execution with DatasetItem objects."""
        config = {'api_key': 'test_key'}
        runner = MockTestAPIRunner(config, max_concurrent=3)

        items = [DatasetItem(query='item query 1'), DatasetItem(query='item query 2')]
        responses = await runner.execute_batch(items)

        assert len(responses) == 2
        assert responses[0].actual_output == 'Response to: item query 1'
        assert responses[1].actual_output == 'Response to: item query 2'

    @pytest.mark.asyncio
    async def test_execute_batch_with_dataframe(self):
        """Test batch execution with DataFrame."""
        config = {'api_key': 'test_key'}
        runner = MockTestAPIRunner(config, max_concurrent=3)

        df = pd.DataFrame({'query': ['df query 1', 'df query 2']})
        responses = await runner.execute_batch(df)

        assert len(responses) == 2
        assert responses[0].actual_output == 'Response to: df query 1'
        assert responses[1].actual_output == 'Response to: df query 2'


class TestConcreteAPIRunners:
    """Test suite for concrete API runner implementations."""

    def setup_method(self):
        """Set up test fixtures."""
        APIRunner._registry.clear()
        APIRunner.register('test_api')(MockTestAPIRunner)
        APIRunner.register('slow_api')(MockSlowAPIRunner)
        APIRunner.register('failing_api')(MockFailingAPIRunner)

    def test_mock_test_api_runner(self):
        """Test MockTestAPIRunner implementation."""
        config = {'api_key': 'test_key'}
        runner = MockTestAPIRunner(config, max_concurrent=5)

        response = runner.execute('test query')

        assert response.actual_output == 'Response to: test query'
        assert response.latency == 0.1
        assert response.status == 'success'

    def test_mock_slow_api_runner(self):
        """Test MockSlowAPIRunner implementation."""
        config = {'api_key': 'slow_key'}
        runner = MockSlowAPIRunner(config, max_concurrent=5)

        import time

        start_time = time.time()
        response = runner.execute('slow query')
        end_time = time.time()

        assert response.actual_output == 'Slow response to: slow query'
        assert end_time - start_time >= 0.1  # Should take at least 0.1 seconds

    def test_mock_failing_api_runner(self):
        """Test MockFailingAPIRunner implementation."""
        config = {'api_key': 'fail_key'}
        runner = MockFailingAPIRunner(config, max_concurrent=5)

        with pytest.raises(RuntimeError, match='API failed for query: fail query'):
            runner.execute('fail query')


class TestAPIRunnerIntegration:
    """Integration tests for APIRunner with multiple executors."""

    def setup_method(self):
        """Set up test fixtures."""
        APIRunner._registry.clear()
        APIRunner.register('test_api')(MockTestAPIRunner)
        APIRunner.register('slow_api')(MockSlowAPIRunner)
        APIRunner.register('failing_api')(MockFailingAPIRunner)

    def test_multiple_api_execution(self):
        """Test executing queries on multiple APIs."""
        config = {
            'test_api': {'api_key': 'test_key'},
            'slow_api': {'api_key': 'slow_key'},
        }
        runner = APIRunner(config=config)

        # Test both APIs
        test_response = runner.execute('test_api', 'test query')
        slow_response = runner.execute('slow_api', 'slow query')

        assert test_response.actual_output == 'Response to: test query'
        assert slow_response.actual_output == 'Slow response to: slow query'

    @pytest.mark.asyncio
    async def test_concurrent_execution_across_apis(self):
        """Test concurrent execution across different APIs."""
        config = {
            'test_api': {'api_key': 'test_key'},
            'slow_api': {'api_key': 'slow_key'},
        }
        runner = APIRunner(config=config)

        # Execute batches on both APIs concurrently
        test_task = runner.execute_batch('test_api', ['query1', 'query2'])
        slow_task = runner.execute_batch('slow_api', ['slow1', 'slow2'])

        test_responses, slow_responses = await asyncio.gather(test_task, slow_task)

        assert len(test_responses) == 2
        assert len(slow_responses) == 2
        assert test_responses[0].actual_output == 'Response to: query1'
        assert slow_responses[0].actual_output == 'Slow response to: slow1'

    def test_error_handling_with_failing_api(self):
        """Test error handling when one API fails."""
        config = {
            'test_api': {'api_key': 'test_key'},
            'failing_api': {'api_key': 'fail_key'},
        }
        runner = APIRunner(config=config)

        # Working API should work
        response = runner.execute('test_api', 'test query')
        assert response.actual_output == 'Response to: test query'

        # Failing API should raise exception
        with pytest.raises(RuntimeError):
            runner.execute('failing_api', 'fail query')


class TestAPIRunnerErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        APIRunner._registry.clear()
        APIRunner.register('test_api')(MockTestAPIRunner)

    def test_empty_config(self):
        """Test APIRunner with empty config."""
        runner = APIRunner(config={})

        assert len(runner.executors) == 0
        assert runner.list_available_apis() == []

    def test_invalid_config_type(self):
        """Test APIRunner with invalid config type."""
        # Should handle gracefully
        with pytest.raises(ValueError):
            _ = APIRunner(config=None)

    def test_api_name_case_sensitivity(self):
        """Test that API names are case-sensitive."""
        config = {'test_api': {'api_key': 'test_key'}}
        runner = APIRunner(config=config)

        # Should work
        response = runner.execute('test_api', 'query')
        assert response is not None

        # Should fail - case-sensitive
        with pytest.raises(ValueError):
            runner.execute('TEST_API', 'query')

    @pytest.mark.asyncio
    async def test_empty_batch_execution(self):
        """Test batch execution with empty query list."""
        config = {'test_api': {'api_key': 'test_key'}}
        runner = APIRunner(config=config)
        with pytest.raises(ValueError):
            _ = await runner.execute_batch('test_api', [])

    @pytest.mark.asyncio
    async def test_batch_execution_with_mixed_success_failure(self):
        """Test batch execution with mixed success and failure."""

        @APIRunner.register('mixed_api')
        class MixedAPIRunner(BaseAPIRunner):
            def __init__(
                self,
                config: Dict[str, Any],
                max_concurrent: int = 5,
                show_progress: bool = True,
                retry_config: Optional[Union[RetryConfig, Dict[str, Any]]] = None,
                tracer: Optional[Any] = None,
            ):
                super().__init__(
                    config, max_concurrent, show_progress, retry_config, tracer
                )

            def execute(self, query: str, **kwargs) -> APIResponseData:
                if 'success' in query:
                    return APIResponseData(
                        query=query, actual_output=f'Response to: {query}'
                    )
                else:
                    raise RuntimeError(f'Failed: {query}')

        config = {'mixed_api': {'api_key': 'mixed_key'}}
        runner = APIRunner(config=config)

        queries = ['success query', 'failure query']

        # Should raise exception for the failed query
        with pytest.raises(RuntimeError):
            await runner.execute_batch('mixed_api', queries)


class TestAPIRunnerPerformance:
    """Performance-related tests."""

    def setup_method(self):
        """Set up test fixtures."""
        APIRunner._registry.clear()
        APIRunner.register('test_api')(MockTestAPIRunner)
        APIRunner.register('slow_api')(MockSlowAPIRunner)

    @pytest.mark.asyncio
    async def test_concurrency_performance(self):
        """Test that concurrent execution is faster than sequential."""
        config = {'slow_api': {'api_key': 'slow_key'}}
        runner = APIRunner(config=config, max_concurrent=3)

        queries = ['query1', 'query2', 'query3']

        import time

        start_time = time.time()
        responses = await runner.execute_batch('slow_api', queries)
        end_time = time.time()

        # Should take roughly 0.1 seconds (concurrent) rather than 0.3 (sequential)
        assert len(responses) == 3
        # Allow some overhead, but should be much faster than sequential
        assert end_time - start_time < 0.25

    @pytest.mark.asyncio
    async def test_large_batch_execution(self):
        """Test execution with large batch of queries."""
        config = {'test_api': {'api_key': 'test_key'}}
        runner = APIRunner(config=config, max_concurrent=10)

        queries = [f'query_{i}' for i in range(100)]
        responses = await runner.execute_batch('test_api', queries)

        assert len(responses) == 100
        assert all(isinstance(r, APIResponseData) for r in responses)
        assert all(r.status == 'success' for r in responses)


# Fixtures for common test data
@pytest.fixture
def sample_config():
    """Fixture providing sample config."""
    return {
        'test_api': {'api_key': 'test_key', 'base_url': 'https://test.api'},
        'slow_api': {'api_key': 'slow_key', 'timeout': 30},
    }


@pytest.fixture
def configured_runner(sample_config):
    """Fixture providing a pre-configured APIRunner."""
    APIRunner._registry.clear()
    APIRunner.register('test_api')(MockTestAPIRunner)
    APIRunner.register('slow_api')(MockSlowAPIRunner)

    return APIRunner(config=sample_config)


@pytest.fixture
def sample_queries():
    """Fixture providing sample queries."""
    return ['What is AI?', 'How does ML work?', 'Explain deep learning']


@pytest.fixture
def sample_dataset_items():
    """Fixture providing sample dataset items."""
    return [
        DatasetItem(query='Dataset query 1'),
        DatasetItem(query='Dataset query 2'),
        DatasetItem(query='Dataset query 3'),
    ]


class TestWithFixtures:
    """Tests using pytest fixtures."""

    def test_configured_runner_execution(self, configured_runner, sample_queries):
        """Test execution with pre-configured runner."""
        response = configured_runner.execute('test_api', sample_queries[0])

        assert isinstance(response, APIResponseData)
        assert response.actual_output == f'Response to: {sample_queries[0]}'
        assert response.status == 'success'

    @pytest.mark.asyncio
    async def test_configured_runner_batch_execution(
        self, configured_runner, sample_queries
    ):
        """Test batch execution with pre-configured runner."""
        responses = await configured_runner.execute_batch('test_api', sample_queries)

        assert len(responses) == len(sample_queries)
        assert all(isinstance(r, APIResponseData) for r in responses)
        assert responses[0].actual_output == f'Response to: {sample_queries[0]}'

    def test_multiple_api_availability(self, configured_runner):
        """Test multiple API availability."""
        apis = configured_runner.list_available_apis()

        assert 'test_api' in apis
        assert 'slow_api' in apis
        assert len(apis) == 2

    @pytest.mark.asyncio
    async def test_dataset_items_execution(
        self, configured_runner, sample_dataset_items
    ):
        """Test execution with dataset items."""
        test_executor = configured_runner.get_executor('test_api')
        responses = await test_executor.execute_batch(sample_dataset_items)

        assert len(responses) == len(sample_dataset_items)
        assert responses[0].actual_output == 'Response to: Dataset query 1'
