from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from axion._core.tracing.collection.scores import TraceScore
from axion._core.tracing.loaders.langfuse import LangfuseTraceLoader
from tests._core.tracing.langfuse.fakes import make_loader, row, score

TRACE_START = '2026-09-01T12:00:00Z'
SCORED_AT = '2026-09-01T13:00:00Z'


def _scored_loader() -> LangfuseTraceLoader:
    rows = [
        row('t1', 'root', TRACE_START, session_id='s1'),
        row('t1', 'gen', '2026-09-01T12:00:01Z', parent='root', session_id='s1'),
    ]
    on_t1 = {'trace_id': 't1', 'session_id': 's1'}
    scores = [
        score('n', 'accuracy', 'NUMERIC', 0.75, SCORED_AT, **on_t1),
        score('b', 'resolved', 'BOOLEAN', True, SCORED_AT, **on_t1),
        score('c', 'flag', 'CATEGORICAL', 'tone', SCORED_AT, comment='why', **on_t1),
        score('o', 'step', 'NUMERIC', 1, SCORED_AT, observation_id='gen', **on_t1),
        score('early', 'early', 'NUMERIC', 1, '2026-09-01T10:00:00Z', **on_t1),
        score('late', 'late', 'NUMERIC', 1, '2026-10-01T12:00:00Z', **on_t1),
        score('too-early', 'too_early', 'NUMERIC', 1, '2026-08-30T12:00:00Z', **on_t1),
        score('sess', 'csat', 'NUMERIC', 4, '2026-09-02T12:00:00Z', session_id='s1'),
    ]
    return make_loader(rows, scores, score_page_size=2)


# (data_type, value, string_value) as TraceScore holds them.
EXPECTED = {
    'accuracy': ('NUMERIC', 0.75, None),
    'resolved': ('BOOLEAN', 1.0, 'True'),
    'flag': ('CATEGORICAL', 0.0, 'tone'),
    'step': ('NUMERIC', 1.0, None),
    'early': ('NUMERIC', 1.0, None),
    'late': ('NUMERIC', 1.0, None),
}


def _raw_score(name, value, trace_id='trace-1', data_type='NUMERIC', string_value=None):
    """A score as a trace payload's ``scores`` carries it."""
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


# ---------------------------------------------------------------------------
# fetch_scores_for_trace
# ---------------------------------------------------------------------------


def test_fetch_scores_for_trace_pages_and_keeps_the_value_shape():
    loader = _scored_loader()
    scores = loader.fetch_scores_for_trace('t1')
    assert all(isinstance(s, TraceScore) for s in scores)
    assert {s.name: (s.data_type, s.value, s.string_value) for s in scores} == (
        EXPECTED
    )
    assert len(loader.client.api.scores_v3.calls) > 1
    step = next(s for s in scores if s.name == 'step')
    assert (step.trace_id, step.observation_id) == ('t1', 'gen')
    assert next(s for s in scores if s.name == 'flag').comment == 'why'


def test_score_window_spans_a_day_before_the_trace_to_now():
    loader = _scored_loader()
    names = {s.name for s in loader.fetch_scores_for_trace('t1')}
    assert {'early', 'late'} <= names  # 2 h before the trace, 30 days after
    assert 'too_early' not in names  # 2 days before: outside the window
    window = loader.client.api.scores_v3.calls[0]
    assert window['from_timestamp'] == datetime(2026, 8, 31, 12, tzinfo=UTC)
    assert window['to_timestamp'] > datetime.now(UTC)


def test_fetched_trace_carries_its_scores():
    trace = _scored_loader().fetch_trace('t1')
    on_payload = {
        s.name: (s.data_type, s.value, getattr(s, 'string_value', None))
        for s in trace.scores
    }
    assert on_payload == EXPECTED


def test_fetch_scores_for_unknown_trace_is_empty():
    assert _scored_loader().fetch_scores_for_trace('missing') == []


def test_fetch_scores_for_trace_error_returns_empty():
    loader = _scored_loader()

    def boom(**_: Any) -> Any:
        raise RuntimeError('network error')

    loader.client.api.scores_v3.get_many_v3 = boom
    loader.max_retries = 0
    assert loader.fetch_scores_for_trace('t1') == []


def test_fetch_scores_for_trace_empty_id():
    loader = LangfuseTraceLoader.__new__(LangfuseTraceLoader)
    loader._client_initialized = True
    loader.request_pacing = 0.0
    assert loader.fetch_scores_for_trace('') == []


# ---------------------------------------------------------------------------
# fetch_scores_for_session / list_score_rows
# ---------------------------------------------------------------------------


def test_fetch_scores_for_session_keys_by_trace_and_session():
    by_trace = _scored_loader().fetch_scores_for_session('s1')
    assert {s.name for s in by_trace['t1']} == set(EXPECTED)
    assert [s.name for s in by_trace['']] == ['csat']


def test_fetch_scores_for_session_error_returns_empty():
    loader = _scored_loader()

    def boom(**_: Any) -> Any:
        raise RuntimeError('timeout')

    loader.client.api.scores_v3.get_many_v3 = boom
    loader.max_retries = 0
    assert loader.fetch_scores_for_session('s1') == {}


def test_list_score_rows_filters_by_name_and_raises_on_failure():
    loader = _scored_loader()
    rows = loader.list_score_rows(session_id='s1', name='csat')
    assert [(r.name, r.value) for r in rows] == [('csat', 4.0)]
    assert loader.list_score_rows(session_id='nope') == []

    def boom(**_: Any) -> Any:
        raise RuntimeError('500')

    loader.client.api.scores_v3.get_many_v3 = boom
    loader.max_retries = 0
    with pytest.raises(RuntimeError):
        loader.list_score_rows(session_id='s1')


def test_list_score_rows_needs_exactly_one_subject():
    loader = _scored_loader()
    with pytest.raises(ValueError):
        loader.list_score_rows()
    with pytest.raises(ValueError):
        loader.list_score_rows(session_id='s1', trace_id='t1')


def test_session_scores_attach_through_session_from_langfuse():
    from axion._core.tracing.collection.session import Session

    session = Session.from_langfuse(
        's1',
        loader=_scored_loader(),
        show_progress=False,
        fetch_scores=True,
        turns_only=False,
    )
    [trace] = list(session)
    assert {s.name for s in trace.scores} == set(EXPECTED)


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
