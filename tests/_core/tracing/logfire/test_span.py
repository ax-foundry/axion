from unittest.mock import MagicMock

import pytest
from axion._core.tracing.logfire.span import Span


class ContainsKeys:
    def __init__(self, keys):
        self.keys = keys

    def __eq__(self, other):
        return all(k in other for k in self.keys)


@pytest.fixture
def mock_tracer():
    tracer = MagicMock()
    tracer._trace_id = 'mock-trace-id'
    tracer._span_stack = []
    tracer._add_trace_internal = MagicMock()
    tracer.tool_metadata.name = 'mock-tool'
    tracer.enable_logfire = False
    return tracer


def test_sync_span_lifecycle(mock_tracer):
    span = Span('test-op', tracer=mock_tracer)
    with span:
        assert span._is_entered
        assert span in mock_tracer._span_stack
        assert mock_tracer._current_span == span

    assert not span._is_entered
    assert span not in mock_tracer._span_stack
    mock_tracer._add_trace_internal.assert_any_call(
        'test-op_start', 'Started test-op', ContainsKeys(['span_id', 'trace_id'])
    )
    mock_tracer._add_trace_internal.assert_any_call(
        'test-op_complete', 'Completed test-op', ContainsKeys(['latency'])
    )


def test_span_error_handling(mock_tracer):
    class DummyError(Exception):
        pass

    span = Span('error-op', tracer=mock_tracer)
    with pytest.raises(DummyError):
        with span:
            raise DummyError('fail trace')

    calls = [c[0][0] for c in mock_tracer._add_trace_internal.call_args_list]
    assert 'error-op_failed' in calls[-1]


def test_span_manual_tracing(mock_tracer):
    span = Span('manual-op', tracer=mock_tracer, auto_trace=False)
    with span:
        span.add_trace('custom_event', 'Something', {'foo': 'bar'})
        span.set_attribute('key', 'value')

    mock_tracer._add_trace_internal.assert_any_call(
        'custom_event', 'Something', ContainsKeys(['foo', 'span_id', 'trace_id'])
    )


def test_double_enter_is_safe(mock_tracer):
    span = Span('double-op', tracer=mock_tracer)
    span._enter()
    span._enter()  # should not re-enter or restart
    assert span._is_entered
    span._exit(None, None, None)
    assert not span._is_entered


def test_context_stack_tracking(mock_tracer):
    span1 = Span('outer', tracer=mock_tracer)
    span2 = Span('inner', tracer=mock_tracer)

    with span1:
        assert mock_tracer._current_span == span1
        with span2:
            assert mock_tracer._current_span == span2
        assert mock_tracer._current_span == span1
    assert mock_tracer._current_span is None
