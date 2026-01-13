from unittest.mock import patch

import pytest


class TestOpikTracer:
    """Tests for OpikTracer implementation."""

    @patch('axion._core.tracing.opik.tracer.OPIK_AVAILABLE', False)
    def test_tracer_without_opik_installed(self):
        """Test graceful degradation when opik is not installed."""
        from axion._core.tracing.opik.tracer import OpikTracer

        tracer = OpikTracer(metadata_type='llm')
        assert tracer._client is None

        # Should not raise
        with tracer.span('test-operation'):
            pass

    @patch('axion._core.tracing.opik.tracer.OPIK_AVAILABLE', True)
    @patch('axion._core.tracing.opik.tracer.Opik')
    def test_tracer_initialization_with_credentials(self, mock_opik_class):
        """Test tracer initializes with correct settings."""
        from axion._core.tracing.opik.tracer import OpikTracer

        tracer = OpikTracer(
            metadata_type='llm',
            api_key='test-key',
            workspace='test-workspace',
            project_name='test-project',
        )

        mock_opik_class.assert_called_once_with(
            project_name='test-project',
            workspace='test-workspace',
        )
        assert tracer._api_key == 'test-key'
        assert tracer._workspace == 'test-workspace'
        assert tracer._project_name == 'test-project'

    @patch('axion._core.tracing.opik.tracer.OPIK_AVAILABLE', True)
    def test_tracer_without_api_key(self):
        """Test tracer warns when API key not configured."""
        from axion._core.tracing.opik.tracer import OpikTracer

        # When no API key is provided, client should be None
        tracer = OpikTracer(metadata_type='llm', api_key=None)
        assert tracer._client is None

    def test_tracer_factory_method(self):
        """Test create() factory method."""
        from axion._core.tracing.opik.tracer import OpikTracer

        tracer = OpikTracer.create(metadata_type='evaluation')
        assert tracer.metadata_type == 'evaluation'

    def test_tracer_registry_registration(self):
        """Test that OpikTracer is registered in TracerRegistry."""
        # Import to trigger registration
        from axion._core.tracing import opik  # noqa: F401
        from axion._core.tracing.registry import TracerRegistry

        assert TracerRegistry.is_registered('opik')
        assert TracerRegistry.get('opik').__name__ == 'OpikTracer'

    @patch('axion._core.tracing.opik.tracer.OPIK_AVAILABLE', False)
    def test_span_stack_management(self):
        """Test span stack is properly managed."""
        from axion._core.tracing.opik.tracer import OpikTracer

        tracer = OpikTracer(metadata_type='llm')

        assert len(tracer._span_stack) == 0
        assert tracer._current_span is None

        with tracer.span('outer') as outer_span:
            assert len(tracer._span_stack) == 1
            assert tracer._current_span == outer_span

            with tracer.span('inner') as inner_span:
                assert len(tracer._span_stack) == 2
                assert tracer._current_span == inner_span

            assert len(tracer._span_stack) == 1
            assert tracer._current_span == outer_span

        assert len(tracer._span_stack) == 0
        assert tracer._current_span is None

    @patch('axion._core.tracing.opik.tracer.OPIK_AVAILABLE', False)
    def test_tracer_start_complete(self):
        """Test tracer start and complete methods."""
        from axion._core.tracing.opik.tracer import OpikTracer

        tracer = OpikTracer(metadata_type='llm')
        tracer.start()
        assert tracer._metadata.status.value == 'running'

        tracer.complete(output_data={'result': 'test'})
        assert tracer._metadata.status.value == 'completed'
        assert tracer._metadata.output_data == {'result': 'test'}

    @patch('axion._core.tracing.opik.tracer.OPIK_AVAILABLE', False)
    def test_tracer_fail(self):
        """Test tracer fail method."""
        from axion._core.tracing.opik.tracer import OpikTracer

        tracer = OpikTracer(metadata_type='llm')
        tracer.start()
        tracer.fail('Test error')

        assert tracer._metadata.status.value == 'failed'
        assert tracer._metadata.error == 'Test error'

    @patch('axion._core.tracing.opik.tracer.OPIK_AVAILABLE', False)
    def test_tracer_getattr_fallback(self):
        """Test __getattr__ provides noop fallback for undefined methods."""
        from axion._core.tracing.opik.tracer import OpikTracer

        tracer = OpikTracer(metadata_type='llm')

        # Should return a noop function
        result = tracer.undefined_method()
        assert result is None

        # get_ methods should return appropriate defaults
        result = tracer.get_undefined_statistics()
        assert result == {}

    @patch('axion._core.tracing.opik.tracer.OPIK_AVAILABLE', False)
    def test_tracer_properties(self):
        """Test tracer property accessors."""
        from axion._core.tracing.opik.tracer import OpikTracer

        tracer = OpikTracer(metadata_type='llm')

        # Test metadata property
        assert tracer.metadata is not None
        assert tracer.metadata.name == 'opik_tracer'

        # Test trace_id property
        assert tracer.trace_id is not None
        assert len(tracer.trace_id) > 0

        # Test handler property
        assert tracer.handler is None

        # Test current_span property
        assert tracer.current_span is None

    @patch('axion._core.tracing.opik.tracer.OPIK_AVAILABLE', False)
    def test_add_trace(self):
        """Test add_trace method."""
        from axion._core.tracing.opik.tracer import OpikTracer

        tracer = OpikTracer(metadata_type='llm')
        tracer.add_trace('test_event', 'Test message', {'key': 'value'})

        assert len(tracer._metadata.traces) == 1
        assert tracer._metadata.traces[0].event_type == 'test_event'
        assert tracer._metadata.traces[0].message == 'Test message'


class TestOpikTracerAsync:
    """Async tests for OpikTracer."""

    @pytest.mark.asyncio
    @patch('axion._core.tracing.opik.tracer.OPIK_AVAILABLE', False)
    async def test_async_span(self):
        """Test async_span context manager."""
        from axion._core.tracing.opik.tracer import OpikTracer

        tracer = OpikTracer(metadata_type='llm')

        async with tracer.async_span('async-operation') as span:
            assert tracer._current_span == span
            assert len(tracer._span_stack) == 1

        assert tracer._current_span is None
        assert len(tracer._span_stack) == 0

    @pytest.mark.asyncio
    @patch('axion._core.tracing.opik.tracer.OPIK_AVAILABLE', False)
    async def test_async_context(self):
        """Test acontext async context manager."""
        from axion._core.tracing.opik.tracer import OpikTracer

        tracer = OpikTracer(metadata_type='llm')

        async with tracer.acontext():
            # Context should be set
            pass
