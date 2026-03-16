import sys
import types
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

_langfuse_stub = types.ModuleType('langfuse')


class _FakeLangfuse:
    def __init__(self, **kwargs):
        self._update_calls: List[Dict[str, Any]] = []

    def start_as_current_observation(self, **kwargs):
        obs = MagicMock()
        obs.__enter__ = MagicMock(return_value=obs)
        obs.__exit__ = MagicMock(return_value=False)
        obs.id = 'obs-id'
        obs.trace_id = 'trace-id'
        return obs

    def update_current_trace(self, **kwargs):
        self._update_calls.append(kwargs)

    def flush(self):
        pass

    def shutdown(self):
        pass


_langfuse_stub.Langfuse = _FakeLangfuse
sys.modules.setdefault('langfuse', _langfuse_stub)

# Now import the modules under test (after stub is in place)
from axion._core.tracing.langfuse.span import LangfuseSpan  # noqa: E402
from axion._core.tracing.langfuse.tracer import LangfuseTracer  # noqa: E402


def _make_tracer(session_id=None, tags=None, **kwargs) -> LangfuseTracer:
    """Build a LangfuseTracer with a fake client."""
    tracer = LangfuseTracer.__new__(LangfuseTracer)
    tracer.metadata_type = 'default'
    tracer.tool_metadata = LangfuseTracer._create_default_tool_meta()
    tracer.auto_flush = False
    tracer.kwargs = {}
    tracer.logger = MagicMock()
    from axion._core.utils import Timer

    tracer.timer = Timer()
    tracer._current_span = None
    tracer._span_stack = []
    from axion._core.uuid import uuid7

    tracer._trace_id = str(uuid7())
    tracer._metadata = tracer._create_metadata()
    tracer.tags = tags or []
    tracer.environment = None
    tracer.session_id = LangfuseTracer._validate_session_id(session_id)
    tracer._client = _FakeLangfuse()
    return tracer


class TestValidateSessionId:
    def test_none(self):
        assert LangfuseTracer._validate_session_id(None) is None

    def test_valid(self):
        assert LangfuseTracer._validate_session_id('chat-123') == 'chat-123'

    def test_stripped(self):
        assert LangfuseTracer._validate_session_id('  abc  ') == 'abc'

    def test_too_long(self):
        assert LangfuseTracer._validate_session_id('x' * 201) is None

    def test_exactly_200_chars(self):
        s = 'a' * 200
        assert LangfuseTracer._validate_session_id(s) == s

    def test_non_ascii(self):
        assert LangfuseTracer._validate_session_id('caf\u00e9') is None

    def test_control_chars_tab(self):
        assert LangfuseTracer._validate_session_id('hello\tworld') is None

    def test_control_chars_newline(self):
        assert LangfuseTracer._validate_session_id('hello\nworld') is None

    def test_whitespace_only(self):
        assert LangfuseTracer._validate_session_id('   ') is None

    def test_non_string(self):
        assert LangfuseTracer._validate_session_id(42) is None  # type: ignore[arg-type]


class TestTracerSessionIdWiring:
    def test_stores_valid_session_id(self):
        tracer = _make_tracer(session_id='my-session')
        assert tracer.session_id == 'my-session'

    def test_create_forwards_session_id(self):
        with patch.object(LangfuseTracer, '_initialize_client'):
            tracer = LangfuseTracer.create(session_id='s123')
        assert tracer.session_id == 's123'

    def test_invalid_session_id_stored_as_none(self):
        tracer = _make_tracer(session_id='caf\u00e9')
        assert tracer.session_id is None

    def test_default_session_id_is_none(self):
        tracer = _make_tracer()
        assert tracer.session_id is None


class TestSpanSessionIdBehaviour:
    def _enter_root_span(self, tracer: LangfuseTracer) -> LangfuseSpan:
        span = LangfuseSpan(tracer, 'root-op', {}, is_async=False)
        tracer._span_stack.append(span)
        span.__enter__()
        return span

    def _enter_child_span(self, tracer: LangfuseTracer) -> LangfuseSpan:
        span = LangfuseSpan(tracer, 'child-op', {}, is_async=False)
        tracer._span_stack.append(span)
        span.__enter__()
        return span

    def test_root_span_update_called_with_session_id_only(self):
        tracer = _make_tracer(session_id='s1')
        self._enter_root_span(tracer)
        calls = tracer._client._update_calls
        assert len(calls) == 1
        assert calls[0] == {'session_id': 's1'}

    def test_root_span_update_called_with_tags_and_session_id(self):
        tracer = _make_tracer(session_id='s2', tags=['t1', 't2'])
        self._enter_root_span(tracer)
        calls = tracer._client._update_calls
        assert len(calls) == 1
        assert calls[0].get('session_id') == 's2'
        assert calls[0].get('tags') == ['t1', 't2']
        assert set(calls[0].keys()) == {'session_id', 'tags'}

    def test_root_span_no_update_when_no_session_no_tags(self):
        tracer = _make_tracer()
        self._enter_root_span(tracer)
        assert tracer._client._update_calls == []

    def test_root_span_invalid_session_id_no_update(self):
        tracer = _make_tracer(session_id='caf\u00e9')
        assert tracer.session_id is None
        self._enter_root_span(tracer)
        assert tracer._client._update_calls == []

    def test_child_span_does_not_call_update_current_trace(self):
        tracer = _make_tracer(session_id='s3')
        self._enter_root_span(tracer)
        assert len(tracer._client._update_calls) == 1
        self._enter_child_span(tracer)
        # Still exactly one call (root only)
        assert len(tracer._client._update_calls) == 1

    def test_session_id_span_attribute_dropped_from_metadata(self):
        tracer = _make_tracer()
        span = LangfuseSpan(
            tracer, 'op', {'session_id': 'should-be-dropped'}, is_async=False
        )
        tracer._span_stack.append(span)
        span.__enter__()
        # The fake client's start_as_current_observation is a method; inspect that
        # session_id never leaked into update_current_trace
        assert tracer._client._update_calls == []

    @pytest.mark.asyncio
    async def test_async_root_span_applies_session_id(self):
        tracer = _make_tracer(session_id='async-session')
        span = LangfuseSpan(tracer, 'async-root', {}, is_async=True)
        tracer._span_stack.append(span)
        await span.__aenter__()
        calls = tracer._client._update_calls
        assert len(calls) == 1
        assert calls[0].get('session_id') == 'async-session'

    @pytest.mark.asyncio
    async def test_async_child_span_no_update(self):
        tracer = _make_tracer(session_id='async-session-2')
        root = LangfuseSpan(tracer, 'async-root', {}, is_async=True)
        tracer._span_stack.append(root)
        await root.__aenter__()
        assert len(tracer._client._update_calls) == 1

        child = LangfuseSpan(tracer, 'async-child', {}, is_async=True)
        tracer._span_stack.append(child)
        await child.__aenter__()
        assert len(tracer._client._update_calls) == 1  # no additional call


class TestInitTracerSessionIdEndToEnd:
    def test_init_tracer_session_id_end_to_end(self):
        import axion._core.tracing.factory as factory_mod
        from axion._core.tracing import init_tracer

        with (
            patch.object(LangfuseTracer, '_initialize_client'),
            patch.object(factory_mod, 'get_tracer', return_value=LangfuseTracer),
        ):
            tracer = init_tracer('llm', force_new=True, session_id='chat-abc')

        assert hasattr(tracer, 'session_id')
        assert tracer.session_id == 'chat-abc'
