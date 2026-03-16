from unittest.mock import patch

import pytest

from axion._core.tracing.langfuse.span import LangfuseSpan
from axion._core.tracing.langfuse.tracer import LangfuseTracer

# conftest.py stubs langfuse and provides make_tracer / _FakeLangfuse
from tests._core.tracing.langfuse.conftest import make_tracer

# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidateUserId:
    def test_validate_user_id_none(self):
        assert LangfuseTracer._validate_user_id(None) is None

    def test_validate_user_id_valid(self):
        assert LangfuseTracer._validate_user_id('user-42') == 'user-42'

    def test_validate_user_id_stripped(self):
        assert LangfuseTracer._validate_user_id('  u1  ') == 'u1'

    def test_validate_user_id_too_long(self):
        assert LangfuseTracer._validate_user_id('x' * 201) is None

    def test_validate_user_id_exactly_200_chars(self):
        s = 'a' * 200
        assert LangfuseTracer._validate_user_id(s) == s

    def test_validate_user_id_non_ascii(self):
        assert LangfuseTracer._validate_user_id('caf\u00e9') is None

    def test_validate_user_id_control_chars(self):
        assert LangfuseTracer._validate_user_id('user\nid') is None

    def test_validate_user_id_whitespace_only(self):
        assert LangfuseTracer._validate_user_id('   ') is None

    def test_validate_user_id_non_string(self):
        assert LangfuseTracer._validate_user_id(123) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Log privacy: raw value must not appear in log output
# ---------------------------------------------------------------------------


class TestValidateUserIdLogPrivacy:
    def test_validate_user_id_invalid_log_omits_raw_value(self):
        bad_value = 'caf\u00e9-secret'
        log_calls = []

        # Patch logger.debug on the tracer module to capture calls
        import axion._core.tracing.langfuse.tracer as tracer_mod

        original = tracer_mod.logger.debug
        try:
            tracer_mod.logger.debug = lambda msg, *args, **kw: log_calls.append(
                (msg, args)
            )
            LangfuseTracer._validate_identity_field('user_id', bad_value)
        finally:
            tracer_mod.logger.debug = original

        # The raw value must not appear in any captured log call
        for msg, args in log_calls:
            full = (msg % args) if args else msg
            assert bad_value not in full, f'Raw value leaked in log: {full!r}'


# ---------------------------------------------------------------------------
# Tracer wiring tests
# ---------------------------------------------------------------------------


class TestTracerUserIdWiring:
    def test_tracer_stores_valid_user_id(self):
        tracer = make_tracer(user_id='u-abc')
        assert tracer.user_id == 'u-abc'

    def test_tracer_create_forwards_user_id(self):
        with patch.object(LangfuseTracer, '_initialize_client'):
            tracer = LangfuseTracer.create(user_id='u123')
        assert tracer.user_id == 'u123'

    def test_tracer_invalid_user_id_stored_as_none(self):
        tracer = make_tracer(user_id='caf\u00e9')
        assert tracer.user_id is None

    def test_default_user_id_is_none(self):
        tracer = make_tracer()
        assert tracer.user_id is None


# ---------------------------------------------------------------------------
# Span behaviour tests
# ---------------------------------------------------------------------------


class TestSpanUserIdBehaviour:
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

    def test_root_span_update_called_with_user_id_only(self):
        tracer = make_tracer(user_id='u1')
        self._enter_root_span(tracer)
        calls = tracer._client._update_calls
        assert len(calls) == 1
        assert calls[0] == {'user_id': 'u1'}

    def test_root_span_update_called_with_all_three(self):
        tracer = make_tracer(user_id='u2', session_id='s2', tags=['t1'])
        self._enter_root_span(tracer)
        calls = tracer._client._update_calls
        assert len(calls) == 1
        assert calls[0].get('user_id') == 'u2'
        assert calls[0].get('session_id') == 's2'
        assert calls[0].get('tags') == ['t1']
        assert set(calls[0].keys()) == {'user_id', 'session_id', 'tags'}

    def test_root_span_user_id_none_does_not_alter_session_tags_call(self):
        tracer = make_tracer(user_id=None, session_id='s3', tags=['t'])
        self._enter_root_span(tracer)
        calls = tracer._client._update_calls
        assert len(calls) == 1
        assert calls[0] == {'session_id': 's3', 'tags': ['t']}
        assert 'user_id' not in calls[0]

    def test_root_span_no_update_when_no_user_no_session_no_tags(self):
        tracer = make_tracer()
        self._enter_root_span(tracer)
        assert tracer._client._update_calls == []

    def test_root_span_invalid_user_id_no_update_no_session_no_tags(self):
        tracer = make_tracer(user_id='caf\u00e9')
        assert tracer.user_id is None
        self._enter_root_span(tracer)
        assert tracer._client._update_calls == []

    def test_child_span_does_not_call_update_current_trace(self):
        tracer = make_tracer(user_id='u3')
        self._enter_root_span(tracer)
        assert len(tracer._client._update_calls) == 1
        self._enter_child_span(tracer)
        assert len(tracer._client._update_calls) == 1  # root only

    def test_user_id_span_attribute_dropped_from_metadata(self):
        tracer = make_tracer()
        span = LangfuseSpan(
            tracer, 'op', {'user_id': 'should-be-dropped'}, is_async=False
        )
        tracer._span_stack.append(span)
        span.__enter__()
        assert tracer._client._update_calls == []

    @pytest.mark.asyncio
    async def test_async_root_span_applies_user_id(self):
        tracer = make_tracer(user_id='async-user')
        span = LangfuseSpan(tracer, 'async-root', {}, is_async=True)
        tracer._span_stack.append(span)
        await span.__aenter__()
        calls = tracer._client._update_calls
        assert len(calls) == 1
        assert calls[0].get('user_id') == 'async-user'

    @pytest.mark.asyncio
    async def test_async_child_span_no_update(self):
        tracer = make_tracer(user_id='async-user-2')
        root = LangfuseSpan(tracer, 'async-root', {}, is_async=True)
        tracer._span_stack.append(root)
        await root.__aenter__()
        assert len(tracer._client._update_calls) == 1

        child = LangfuseSpan(tracer, 'async-child', {}, is_async=True)
        tracer._span_stack.append(child)
        await child.__aenter__()
        assert len(tracer._client._update_calls) == 1  # no additional call

    def test_init_tracer_user_id_end_to_end(self):
        import axion._core.tracing.factory as factory_mod
        from axion._core.tracing import init_tracer

        with (
            patch.object(LangfuseTracer, '_initialize_client'),
            patch.object(factory_mod, 'get_tracer', return_value=LangfuseTracer),
        ):
            tracer = init_tracer('llm', force_new=True, user_id='u42')

        assert hasattr(tracer, 'user_id')
        assert tracer.user_id == 'u42'
