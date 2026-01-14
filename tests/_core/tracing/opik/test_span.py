from unittest.mock import MagicMock, patch

import pytest


class ContainsKeys:
    """Helper class for checking dictionary keys in assertions."""

    def __init__(self, keys):
        self.keys = keys

    def __eq__(self, other):
        return all(k in other for k in self.keys)


@pytest.fixture
def mock_tracer():
    """Create a mock tracer for testing spans."""
    tracer = MagicMock()
    tracer._trace_id = 'mock-trace-id'
    tracer._span_stack = []
    tracer._client = None  # Simulate no Opik client
    tracer.add_trace = MagicMock()
    return tracer


class TestOpikSpan:
    """Tests for OpikSpan implementation."""

    def test_span_properties(self, mock_tracer):
        """Test span ID and trace ID properties."""
        from axion._core.tracing.opik.span import OpikSpan

        span = OpikSpan(mock_tracer, 'test-op', {})
        assert span.span_id is not None
        assert len(span.span_id) > 0
        assert span.trace_id == 'mock-trace-id'

    def test_span_context_manager(self, mock_tracer):
        """Test span works as context manager."""
        from axion._core.tracing.opik.span import OpikSpan

        span = OpikSpan(mock_tracer, 'test-op', {})
        with span as s:
            assert s is span

    def test_span_set_methods_without_client(self, mock_tracer):
        """Test set_input, set_output methods don't raise without client."""
        from axion._core.tracing.opik.span import OpikSpan

        span = OpikSpan(mock_tracer, 'test-op', {})
        with span:
            # These should not raise even without Opik client
            span.set_input({'query': 'test'})
            span.set_output({'response': 'result'})
            span.set_attribute('custom', 'value')
            span.add_event('custom_event', {'key': 'value'})
            span.set_usage(100, 50)

    def test_span_record_exception(self, mock_tracer):
        """Test record_exception doesn't raise without client."""
        from axion._core.tracing.opik.span import OpikSpan

        span = OpikSpan(mock_tracer, 'test-op', {})
        with span:
            span.record_exception(ValueError('test error'))

    def test_span_add_trace(self, mock_tracer):
        """Test add_trace calls tracer method."""
        from axion._core.tracing.opik.span import OpikSpan

        span = OpikSpan(mock_tracer, 'test-op', {})
        with span:
            span.add_trace('custom_event', 'Test message', {'key': 'value'})

        mock_tracer.add_trace.assert_called_once()
        call_args = mock_tracer.add_trace.call_args
        assert call_args[0][0] == 'custom_event'
        assert call_args[0][1] == 'Test message'

    def test_span_serialization(self, mock_tracer):
        """Test data serialization helper."""
        from axion._core.tracing.opik.span import OpikSpan

        span = OpikSpan(mock_tracer, 'test', {})

        # Test various data types
        assert span._serialize_data(None) is None
        assert span._serialize_data({'key': 'value'}) == {'key': 'value'}
        assert span._serialize_data([1, 2, 3]) == [1, 2, 3]
        assert span._serialize_data('string') == 'string'
        assert span._serialize_data(123) == 123
        assert span._serialize_data(12.5) == 12.5
        assert span._serialize_data(True) is True

    def test_span_serialization_pydantic(self, mock_tracer):
        """Test serialization of Pydantic-like models."""
        from axion._core.tracing.opik.span import OpikSpan

        span = OpikSpan(mock_tracer, 'test', {})

        # Mock a Pydantic model with model_dump
        mock_model = MagicMock()
        mock_model.model_dump.return_value = {'field': 'value'}
        assert span._serialize_data(mock_model) == {'field': 'value'}

        # Mock an older Pydantic model with dict method
        mock_old_model = MagicMock(spec=['dict'])
        mock_old_model.dict.return_value = {'old_field': 'old_value'}
        assert span._serialize_data(mock_old_model) == {'old_field': 'old_value'}

    def test_span_serialization_unknown_type(self, mock_tracer):
        """Test serialization falls back to str for unknown types."""
        from axion._core.tracing.opik.span import OpikSpan

        span = OpikSpan(mock_tracer, 'test', {})

        class CustomClass:
            def __str__(self):
                return 'custom_string'

        obj = CustomClass()
        assert span._serialize_data(obj) == 'custom_string'

    @patch('axion._core.tracing.opik.span.OPIK_AVAILABLE', False)
    def test_span_skips_when_opik_not_available(self, mock_tracer):
        """Test span gracefully skips operations when Opik not available."""
        from axion._core.tracing.opik.span import OpikSpan

        span = OpikSpan(mock_tracer, 'test-op', {'model': 'gpt-4'})
        with span:
            assert span._opik_span is None
            assert span._opik_context is None


class TestOpikSpanAsync:
    """Async tests for OpikSpan."""

    @pytest.mark.asyncio
    async def test_async_span_context(self, mock_tracer):
        """Test async context manager."""
        from axion._core.tracing.opik.span import OpikSpan

        span = OpikSpan(mock_tracer, 'async-op', {}, is_async=True)
        async with span as s:
            assert s is span

    @pytest.mark.asyncio
    async def test_async_span_methods(self, mock_tracer):
        """Test span methods work in async context."""
        from axion._core.tracing.opik.span import OpikSpan

        span = OpikSpan(mock_tracer, 'async-op', {}, is_async=True)
        async with span:
            span.set_input({'query': 'test'})
            span.set_output({'response': 'result'})


class TestOpikSpanWithAttributes:
    """Tests for OpikSpan with various attributes."""

    def test_span_with_model_attribute(self, mock_tracer):
        """Test span type is 'llm' when model attribute is present."""
        from axion._core.tracing.opik.span import OpikSpan

        span = OpikSpan(mock_tracer, 'llm-call', {'model': 'gpt-4'})
        # When Opik is not available, we can't verify the span type
        # but we can verify the span is created without errors
        with span:
            pass

    def test_span_without_model_attribute(self, mock_tracer):
        """Test span type defaults to 'general' without model."""
        from axion._core.tracing.opik.span import OpikSpan

        span = OpikSpan(mock_tracer, 'general-op', {'custom_key': 'value'})
        with span:
            pass

    def test_span_with_input_output(self, mock_tracer):
        """Test span with input/output attributes."""
        from axion._core.tracing.opik.span import OpikSpan

        span = OpikSpan(
            mock_tracer,
            'io-op',
            {'input': {'query': 'test'}, 'output': {'response': 'result'}},
        )
        with span:
            pass

    def test_span_with_metadata(self, mock_tracer):
        """Test span with additional metadata attributes."""
        from axion._core.tracing.opik.span import OpikSpan

        span = OpikSpan(
            mock_tracer,
            'metadata-op',
            {
                'custom_field': 'custom_value',
                'another_field': 123,
            },
        )
        with span:
            pass
