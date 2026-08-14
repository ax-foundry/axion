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

    raw_trace = SimpleNamespace(id='trace-1', name='chat-turn', observations=[])
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

    raw_trace = SimpleNamespace(id='trace-1', name='chat-turn', observations=[])
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


# ---------------------------------------------------------------------------
# SessionCollection.from_langfuse — fetch_scores=True
# ---------------------------------------------------------------------------


def _session_loader(scores_by_session):
    """A loader returning one single-trace session per id, with canned scores."""
    loader = MagicMock()

    def _get(session_id, **kwargs):
        raw = SimpleNamespace(
            id=f'trace-{session_id}', name='chat-turn', observations=[]
        )
        return SimpleNamespace(id=session_id), [raw]

    loader.get_session_with_traces.side_effect = _get
    loader.fetch_scores_for_session.side_effect = lambda sid: scores_by_session[sid]
    return loader


def test_session_collection_from_langfuse_fetch_scores_attaches_per_session():
    from axion._core.tracing.collection.session_collection import SessionCollection

    loader = _session_loader(
        {
            'sess-1': {
                'trace-sess-1': [
                    TraceScore(name='accuracy', value=0.6, data_type='NUMERIC')
                ]
            },
            'sess-2': {
                'trace-sess-2': [
                    TraceScore(name='reliability', value=1.0, data_type='NUMERIC')
                ]
            },
        }
    )

    collection = SessionCollection.from_langfuse(
        ['sess-1', 'sess-2'], loader=loader, fetch_scores=True, turns_only=False
    )

    # One paginated call per session, not per trace.
    assert loader.fetch_scores_for_session.call_count == 2
    assert collection[0][0].scores[0].name == 'accuracy'
    assert collection[1][0].scores[0].value == 1.0


def test_session_collection_from_langfuse_defaults_to_no_scores():
    """The default has to stay off, matching the three sibling constructors.

    This is also the regression: a session-grain metric rolling up per-turn
    scores saw them empty here while the same traces carried scores everywhere
    else, so it scored None rather than failing.
    """
    from axion._core.tracing.collection.session_collection import SessionCollection

    loader = _session_loader({'sess-1': {}})

    collection = SessionCollection.from_langfuse(
        ['sess-1'], loader=loader, turns_only=False
    )

    loader.fetch_scores_for_session.assert_not_called()
    assert collection[0][0].scores == []


def test_session_collection_fetch_scores_reaches_only_retained_traces():
    """turns_only prunes before the back-fill, so pruned traces keep no scores.

    Documented rather than fixed: the traces have to exist to be matched by id,
    and widening the back-fill to pruned traces would resurrect what the caller
    asked to drop.
    """
    from axion._core.tracing.collection.session_collection import SessionCollection

    turn = SimpleNamespace(id='trace-turn', name='chat-turn', observations=[])
    pipeline = SimpleNamespace(
        id='trace-pipeline', name='pipeline-run', observations=[]
    )

    loader = MagicMock()
    loader.get_session_with_traces.return_value = (
        SimpleNamespace(id='sess-1'),
        [turn, pipeline],
    )
    loader.fetch_scores_for_session.return_value = {
        'trace-turn': [TraceScore(name='kept', value=1.0, data_type='NUMERIC')],
        'trace-pipeline': [TraceScore(name='pruned', value=0.0, data_type='NUMERIC')],
    }

    collection = SessionCollection.from_langfuse(
        ['sess-1'],
        loader=loader,
        fetch_scores=True,
        turn_name='chat-turn',
        turns_only=True,
    )

    session = collection[0]
    assert len(session) == 1
    assert session[0].scores[0].name == 'kept'


def test_session_collection_fetch_scores_noop_on_loader_without_support():
    """A loader with no score API must not raise — the helper guards on it."""
    from axion._core.tracing.collection.session_collection import SessionCollection

    raw = SimpleNamespace(id='trace-1', name='chat-turn', observations=[])
    loader = MagicMock(spec=['get_session_with_traces'])
    loader.get_session_with_traces.return_value = (SimpleNamespace(id='sess-1'), [raw])

    collection = SessionCollection.from_langfuse(
        ['sess-1'], loader=loader, fetch_scores=True, turns_only=False
    )

    assert collection[0][0].scores == []


def test_fetch_scores_falls_back_to_scores_on_the_trace_payload():
    """Langfuse's session score query only returns scores carrying a sessionId.

    A score written against a trace has none, so it is recovered from the trace
    payload, which already carries it.
    """
    from axion._core.tracing.collection.session_collection import SessionCollection

    raw = SimpleNamespace(
        id='trace-1',
        name='chat-turn',
        observations=[],
        scores=[_raw_score('accuracy', 0.75)],
    )
    loader = MagicMock()
    loader.get_session_with_traces.return_value = (SimpleNamespace(id='sess-1'), [raw])
    loader.fetch_scores_for_session.return_value = {}

    collection = SessionCollection.from_langfuse(
        ['sess-1'], loader=loader, fetch_scores=True, turns_only=False
    )

    scores = collection[0][0].scores
    assert [(s.name, s.value) for s in scores] == [('accuracy', 0.75)]


def test_session_query_scores_take_precedence_over_the_trace_payload():
    from axion._core.tracing.collection.session_collection import SessionCollection

    raw = SimpleNamespace(
        id='trace-1',
        name='chat-turn',
        observations=[],
        scores=[_raw_score('accuracy', 0.75)],
    )
    loader = MagicMock()
    loader.get_session_with_traces.return_value = (SimpleNamespace(id='sess-1'), [raw])
    loader.fetch_scores_for_session.return_value = {
        'trace-1': [TraceScore(name='accuracy', value=1.0, data_type='NUMERIC')],
    }

    collection = SessionCollection.from_langfuse(
        ['sess-1'], loader=loader, fetch_scores=True, turns_only=False
    )

    assert [s.value for s in collection[0][0].scores] == [1.0]


def test_unusable_trace_payload_scores_are_skipped_not_raised():
    """Some Langfuse payloads carry bare score ids rather than score objects."""
    from axion._core.tracing.collection.session_collection import SessionCollection

    raw = SimpleNamespace(
        id='trace-1',
        name='chat-turn',
        observations=[],
        scores=['score-id-1', _raw_score('accuracy', 0.5)],
    )
    loader = MagicMock()
    loader.get_session_with_traces.return_value = (SimpleNamespace(id='sess-1'), [raw])
    loader.fetch_scores_for_session.return_value = {}

    collection = SessionCollection.from_langfuse(
        ['sess-1'], loader=loader, fetch_scores=True, turns_only=False
    )

    assert [s.name for s in collection[0][0].scores] == ['accuracy']
