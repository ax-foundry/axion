from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from axion._core.tracing.collection.scores import TraceScore
from axion._core.tracing.loaders.langfuse import LangfuseTraceLoader


def _raw_score(name, value, trace_id='trace-1', data_type='NUMERIC', string_value=None):
    return SimpleNamespace(
        name=name,
        value=value,
        data_type=data_type,
        trace_id=trace_id,
        observation_id=None,
        string_value=string_value,
        comment=None,
        source='API',
        timestamp=None,
    )


def _scores_response(scores, total_pages=1, page=1):
    return SimpleNamespace(
        data=scores,
        meta=SimpleNamespace(page=page, total_pages=total_pages),
    )


def _loader(scores_api) -> LangfuseTraceLoader:
    loader = LangfuseTraceLoader.__new__(LangfuseTraceLoader)
    loader.client = SimpleNamespace(api=SimpleNamespace(scores=scores_api))
    loader._client_initialized = True
    loader.request_pacing = 0.0
    loader._execute_with_retry = lambda fn, description: fn()
    return loader


# ---------------------------------------------------------------------------
# fetch_scores_for_trace
# ---------------------------------------------------------------------------


def test_fetch_scores_for_trace_single_page():
    api = MagicMock()
    api.get_many.return_value = _scores_response(
        [_raw_score('halluc', 1.0), _raw_score('scope', 0.8)]
    )
    loader = _loader(api)
    scores = loader.fetch_scores_for_trace('trace-1')
    assert len(scores) == 2
    assert all(isinstance(s, TraceScore) for s in scores)
    assert scores[0].name == 'halluc'
    assert scores[1].name == 'scope'
    api.get_many.assert_called_once_with(trace_id='trace-1', page=1)


def test_fetch_scores_for_trace_paginates():
    api = MagicMock()
    api.get_many.side_effect = [
        _scores_response([_raw_score('a', 0.9)], total_pages=2, page=1),
        _scores_response([_raw_score('b', 0.7)], total_pages=2, page=2),
    ]
    loader = _loader(api)
    scores = loader.fetch_scores_for_trace('trace-1')
    assert len(scores) == 2
    assert [s.name for s in scores] == ['a', 'b']
    assert api.get_many.call_count == 2


def test_fetch_scores_for_trace_empty():
    api = MagicMock()
    api.get_many.return_value = _scores_response([])
    loader = _loader(api)
    scores = loader.fetch_scores_for_trace('trace-1')
    assert scores == []


def test_fetch_scores_for_trace_error_returns_empty(caplog):
    api = MagicMock()
    api.get_many.side_effect = RuntimeError('network error')
    loader = _loader(api)
    # _execute_with_retry re-raises; the method must catch and warn
    loader._execute_with_retry = lambda fn, description: fn()
    scores = loader.fetch_scores_for_trace('trace-1')
    assert scores == []


def test_fetch_scores_for_trace_empty_id():
    loader = LangfuseTraceLoader.__new__(LangfuseTraceLoader)
    loader._client_initialized = True
    loader.request_pacing = 0.0
    scores = loader.fetch_scores_for_trace('')
    assert scores == []


# ---------------------------------------------------------------------------
# fetch_scores_for_session
# ---------------------------------------------------------------------------


def test_fetch_scores_for_session_groups_by_trace_id():
    api = MagicMock()
    api.get_many.return_value = _scores_response(
        [
            _raw_score('halluc', 1.0, trace_id='trace-a'),
            _raw_score('scope', 0.5, trace_id='trace-b'),
            _raw_score('under', 0.0, trace_id='trace-a'),
        ]
    )
    loader = _loader(api)
    result = loader.fetch_scores_for_session('sess-1')
    assert set(result.keys()) == {'trace-a', 'trace-b'}
    assert len(result['trace-a']) == 2
    assert len(result['trace-b']) == 1
    api.get_many.assert_called_once_with(session_id='sess-1', page=1)


def test_fetch_scores_for_session_paginates():
    api = MagicMock()
    api.get_many.side_effect = [
        _scores_response([_raw_score('a', 1.0, trace_id='t1')], total_pages=2, page=1),
        _scores_response([_raw_score('b', 0.5, trace_id='t2')], total_pages=2, page=2),
    ]
    loader = _loader(api)
    result = loader.fetch_scores_for_session('sess-1')
    assert 't1' in result and 't2' in result
    assert api.get_many.call_count == 2


def test_fetch_scores_for_session_error_returns_empty():
    api = MagicMock()
    api.get_many.side_effect = RuntimeError('timeout')
    loader = _loader(api)
    loader._execute_with_retry = lambda fn, description: fn()
    result = loader.fetch_scores_for_session('sess-1')
    assert result == {}


# ---------------------------------------------------------------------------
# Session.from_langfuse fetch_scores=True integration
# ---------------------------------------------------------------------------


def test_session_from_langfuse_fetch_scores_attaches():
    from axion._core.tracing.collection.session import Session

    raw_trace = SimpleNamespace(id='trace-1', name='abby-web-chat', observations=[])
    fake_session = SimpleNamespace(id='sess-1')

    loader = MagicMock()
    loader.get_session_with_traces.return_value = (fake_session, [raw_trace])
    loader.fetch_scores_for_session.return_value = {
        'trace-1': [TraceScore(name='halluc', value=1.0, data_type='NUMERIC')]
    }

    session = Session.from_langfuse(
        'sess-1', loader=loader, fetch_scores=True, turns_only=False
    )

    loader.fetch_scores_for_session.assert_called_once_with('sess-1')
    assert len(session) == 1
    assert session[0].scores[0].name == 'halluc'


def test_session_from_langfuse_fetch_scores_false_leaves_empty():
    from axion._core.tracing.collection.session import Session

    raw_trace = SimpleNamespace(id='trace-1', name='abby-web-chat', observations=[])
    fake_session = SimpleNamespace(id='sess-1')

    loader = MagicMock()
    loader.get_session_with_traces.return_value = (fake_session, [raw_trace])

    session = Session.from_langfuse(
        'sess-1', loader=loader, fetch_scores=False, turns_only=False
    )

    loader.fetch_scores_for_session.assert_not_called()
    assert session[0].scores == []


# ---------------------------------------------------------------------------
# TraceCollection._from_traces — score preservation through filter()
# ---------------------------------------------------------------------------


def test_filter_preserves_scores():
    """filter() must not drop scores attached via fetch_scores=True."""
    from axion._core.tracing.collection.trace_collection import TraceCollection

    raw_a = SimpleNamespace(id='trace-a', name='keep', observations=[])
    raw_b = SimpleNamespace(id='trace-b', name='drop', observations=[])

    collection = TraceCollection([raw_a, raw_b])
    collection[0]._scores = [TraceScore(name='halluc', value=1.0, data_type='NUMERIC')]
    collection[1]._scores = [TraceScore(name='scope', value=0.5, data_type='NUMERIC')]

    filtered = collection.filter(lambda t: getattr(t.raw, 'name', '') == 'keep')

    assert len(filtered) == 1
    assert filtered[0].scores[0].name == 'halluc'


def test_filter_by_preserves_scores():
    """filter_by() (wraps filter) must also preserve scores."""
    from axion._core.tracing.collection.trace_collection import TraceCollection

    raw_a = SimpleNamespace(id='trace-a', name='keep', observations=[])
    collection = TraceCollection([raw_a])
    collection[0]._scores = [TraceScore(name='x', value=0.9, data_type='NUMERIC')]

    filtered = collection.filter_by(name='keep')
    assert filtered[0].scores[0].name == 'x'


# ---------------------------------------------------------------------------
# TraceCollection.from_session — fetch_scores=True
# ---------------------------------------------------------------------------


def test_trace_collection_from_session_fetch_scores():
    from axion._core.tracing.collection.trace_collection import TraceCollection

    raw_trace = SimpleNamespace(id='trace-1', observations=[])

    loader = MagicMock()
    loader.get_session_traces.return_value = [raw_trace]
    loader.fetch_scores_for_session.return_value = {
        'trace-1': [TraceScore(name='under', value=0.0, data_type='NUMERIC')]
    }

    collection = TraceCollection.from_session(
        'sess-1', loader=loader, fetch_scores=True
    )

    loader.fetch_scores_for_session.assert_called_once_with('sess-1')
    assert collection[0].scores[0].name == 'under'


def test_trace_collection_from_session_fetch_scores_false():
    from axion._core.tracing.collection.trace_collection import TraceCollection

    raw_trace = SimpleNamespace(id='trace-1', observations=[])
    loader = MagicMock()
    loader.get_session_traces.return_value = [raw_trace]

    collection = TraceCollection.from_session(
        'sess-1', loader=loader, fetch_scores=False
    )

    loader.fetch_scores_for_session.assert_not_called()
    assert collection[0].scores == []
