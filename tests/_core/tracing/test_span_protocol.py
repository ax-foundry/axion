"""Tests for BaseSpan protocol and span helper standardization."""
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest

from axion._core.tracing.registry import BaseSpan


class TestNoopSpan:
    def test_set_input_output(self):
        from axion._core.tracing.noop.span import Span

        mock_tracer = MagicMock()
        mock_tracer._trace_id = 'test-trace-id'
        span = Span('test_op', mock_tracer, {})

        # Should not raise
        span.set_input({'query': 'hello'})
        span.set_output({'answer': 'world'})


class TestLogfireSpan:
    def test_set_input_output(self):
        from axion._core.tracing.logfire.span import Span

        mock_tracer = MagicMock()
        mock_tracer._trace_id = 'test-trace-id'
        mock_tracer._add_trace_internal = MagicMock()
        span = Span('test_op', mock_tracer, {})

        span.set_input({'query': 'hello'})
        span.set_output({'answer': 'world'})
        assert span.attributes.get('input') is not None
        assert span.attributes.get('output') is not None


class TestLangfuseSpan:
    def test_set_input_output(self):
        from axion._core.tracing.langfuse.span import LangfuseSpan

        mock_tracer = MagicMock()
        mock_tracer._trace_id = 'test-trace-id'
        mock_tracer._client = None
        mock_tracer._langfuse_trace = None
        span = LangfuseSpan('test_op', mock_tracer, {})

        span.set_input({'query': 'hello'})
        span.set_output({'answer': 'world'})


class TestOpikSpan:
    @patch('axion._core.tracing.opik.tracer.OPIK_AVAILABLE', False)
    def test_set_input_output(self):
        from axion._core.tracing.opik.span import OpikSpan

        mock_tracer = MagicMock()
        mock_tracer._trace_id = 'test-trace-id'
        mock_tracer._client = None
        span = OpikSpan('test_op', mock_tracer, {})

        span.set_input({'query': 'hello'})
        span.set_output({'answer': 'world'})


class TestBaseSpanProtocol:
    def _make_spans(self):
        from axion._core.tracing.langfuse.span import LangfuseSpan
        from axion._core.tracing.logfire.span import Span as LogfireSpan
        from axion._core.tracing.noop.span import Span as NoopSpan
        from axion._core.tracing.opik.span import OpikSpan

        mock_tracer = MagicMock()
        mock_tracer._trace_id = 'test-trace-id'
        mock_tracer._add_trace_internal = MagicMock()
        mock_tracer._client = None
        mock_tracer._langfuse_trace = None

        return [
            NoopSpan('op', mock_tracer, {}),
            LogfireSpan('op', mock_tracer, {}),
            LangfuseSpan('op', mock_tracer, {}),
            OpikSpan('op', mock_tracer, {}),
        ]

    def test_all_spans_satisfy_basespan_protocol(self):
        for span in self._make_spans():
            assert isinstance(span, BaseSpan), (
                f'{type(span).__name__} does not satisfy BaseSpan protocol'
            )

    def test_decorator_capture_result_calls_set_output(self):
        """@trace(capture_result=True) calls set_output without hasattr guard."""
        from axion._core.tracing.decorators import trace
        from axion._core.tracing.noop.span import Span
        from axion._core.tracing.noop.tracer import NoOpTracer

        tracer = NoOpTracer.create(metadata_type='llm')

        # Track set_output calls
        set_output_calls = []
        original_span = tracer.span

        class TrackingSpan:
            def __init__(self, inner):
                self._inner = inner

            def __enter__(self):
                self._inner.__enter__()
                return self

            def __exit__(self, *args):
                return self._inner.__exit__(*args)

            def set_attribute(self, key, value):
                self._inner.set_attribute(key, value)

            def set_input(self, data):
                self._inner.set_input(data)

            def set_output(self, data):
                set_output_calls.append(data)
                self._inner.set_output(data)

        from contextlib import contextmanager

        @contextmanager
        def tracking_span(name, **attrs):
            with original_span(name, **attrs) as inner:
                yield TrackingSpan(inner)

        tracer.span = tracking_span

        class Service:
            def __init__(self):
                self.tracer = tracer

            @trace(capture_result=True)
            def run(self):
                return 'hello'

        svc = Service()
        result = svc.run()
        assert result == 'hello'
        assert len(set_output_calls) == 1
        assert set_output_calls[0] == 'hello'

    @pytest.mark.asyncio
    async def test_async_decorator_capture_result_calls_set_output(self):
        """@trace(capture_result=True) calls set_output on the async wrapper too."""
        from axion._core.tracing.decorators import trace
        from axion._core.tracing.noop.tracer import NoOpTracer

        tracer = NoOpTracer.create(metadata_type='llm')

        set_output_calls = []
        original_async_span = tracer.async_span

        class TrackingSpan:
            def __init__(self, inner):
                self._inner = inner

            async def __aenter__(self):
                await self._inner.__aenter__()
                return self

            async def __aexit__(self, *args):
                return await self._inner.__aexit__(*args)

            def set_attribute(self, key, value):
                self._inner.set_attribute(key, value)

            def set_input(self, data):
                self._inner.set_input(data)

            def set_output(self, data):
                set_output_calls.append(data)
                self._inner.set_output(data)

        @asynccontextmanager
        async def tracking_async_span(name, **attrs):
            async with original_async_span(name, **attrs) as inner:
                yield TrackingSpan(inner)

        tracer.async_span = tracking_async_span

        class Service:
            def __init__(self):
                self.tracer = tracer

            @trace(capture_result=True)
            async def run(self):
                return 'async-hello'

        svc = Service()
        result = await svc.run()
        assert result == 'async-hello'
        assert len(set_output_calls) == 1
        assert set_output_calls[0] == 'async-hello'
