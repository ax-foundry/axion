"""Tests for the SessionCollection class."""

from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import MagicMock

from axion._core.tracing.collection.session import Session
from axion._core.tracing.collection.session_collection import SessionCollection

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeObservation:
    name: str = 'step1'
    type: str = 'GENERATION'
    input: Any = ''
    output: Any = ''


@dataclass
class FakeRawTrace:
    id: str = 'trace-1'
    name: str = 'athena-chat'
    input: Any = None
    output: Any = None
    observations: List[Any] = field(default_factory=list)
    timestamp: Any = None
    tags: List[str] = field(default_factory=list)


def _chat_trace(i, observations=None):
    return FakeRawTrace(
        id=f'chat-{i}',
        name='athena-chat',
        input=f'question {i}',
        output=f'answer {i}',
        observations=observations or [],
    )


def _session_dict(sid, n_turns=2, environment='production'):
    return {
        'id': sid,
        'environment': environment,
        'traces': [_chat_trace(i) for i in range(n_turns)],
    }


# ---------------------------------------------------------------------------
# Construction / protocol
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_from_dicts(self):
        sc = SessionCollection([_session_dict('s1'), _session_dict('s2')])
        assert len(sc) == 2
        assert isinstance(sc[0], Session)
        assert sc[0].id == 's1'

    def test_accepts_session_instances(self):
        s = Session(_session_dict('s1'))
        sc = SessionCollection([s])
        assert len(sc) == 1
        assert sc[0] is s

    def test_iter(self):
        sc = SessionCollection([_session_dict('s1'), _session_dict('s2')])
        ids = [s.id for s in sc]
        assert ids == ['s1', 's2']

    def test_repr(self):
        sc = SessionCollection([_session_dict('s1')])
        assert 'count=1' in repr(sc)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_filter(self):
        sc = SessionCollection(
            [_session_dict('s1', n_turns=1), _session_dict('s2', n_turns=3)]
        )
        filtered = sc.filter(lambda s: len(s) > 1)
        assert len(filtered) == 1
        assert filtered[0].id == 's2'
        assert isinstance(filtered, SessionCollection)

    def test_filter_by(self):
        sc = SessionCollection(
            [
                _session_dict('s1', environment='production'),
                _session_dict('s2', environment='staging'),
            ]
        )
        filtered = sc.filter_by(environment='staging')
        assert len(filtered) == 1
        assert filtered[0].id == 's2'

    def test_filter_by_missing_attr(self):
        sc = SessionCollection([_session_dict('s1')])
        assert len(sc.filter_by(nonexistent='x')) == 0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_by_type_across_sessions(self):
        s1 = {
            'id': 's1',
            'traces': [_chat_trace(0, [FakeObservation(name='a', type='TOOL')])],
        }
        s2 = {
            'id': 's2',
            'traces': [
                _chat_trace(0, [FakeObservation(name='b', type='TOOL')]),
                _chat_trace(1, [FakeObservation(name='c', type='TOOL')]),
            ],
        }
        sc = SessionCollection([s1, s2])
        assert len(sc.by_type('TOOL')) == 3
        assert len(sc.tools()) == 3


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_list(self):
        sc = SessionCollection([_session_dict('s1'), _session_dict('s2')])
        data = sc.to_list()
        assert len(data) == 2
        assert data[0]['id'] == 's1'

    def test_save_and_load_json(self, tmp_path):
        sc = SessionCollection([_session_dict('s1'), _session_dict('s2')])
        path = tmp_path / 'sessions.json'
        sc.save_json(path)
        loaded = SessionCollection.load_json(path)
        assert len(loaded) == 2
        assert {s.id for s in loaded} == {'s1', 's2'}

    def test_load_json_rejects_non_list(self, tmp_path):
        path = tmp_path / 'bad.json'
        path.write_text('{"id": "s1"}')
        import pytest

        with pytest.raises(ValueError):
            SessionCollection.load_json(path)


# ---------------------------------------------------------------------------
# Dataset conversion
# ---------------------------------------------------------------------------


class TestToDataset:
    def test_one_item_per_session(self):
        sc = SessionCollection([_session_dict('s1'), _session_dict('s2')])
        ds = sc.to_dataset()
        assert len(ds.items) == 2
        assert all(i.multi_turn_conversation is not None for i in ds.items)

    def test_skips_sessions_without_turns(self):
        pipeline = {
            'id': 's1',
            'traces': [
                FakeRawTrace(
                    id='p0',
                    name='athena',
                    input={'case_id': 'C0'},
                    output={'outcome': 'done'},
                )
            ],
        }
        sc = SessionCollection([pipeline, _session_dict('s2')])
        ds = sc.to_dataset()
        assert len(ds.items) == 1

    def test_list_returning_transform_flattens(self):
        sc = SessionCollection([_session_dict('s1'), _session_dict('s2')])

        def transform(session):
            return [
                {'query': f'{session.id}-q1', 'actual_output': 'a'},
                {'query': f'{session.id}-q2', 'actual_output': 'b'},
            ]

        ds = sc.to_dataset(transform=transform)
        assert len(ds.items) == 4


# ---------------------------------------------------------------------------
# from_langfuse (mocked loader)
# ---------------------------------------------------------------------------


class TestFromLangfuse:
    def test_loops_session_ids(self):
        loader = MagicMock()

        def fake_get(session_id, show_progress=True, enrich=True):
            return ({'id': session_id, 'environment': 'production'}, [_chat_trace(0)])

        loader.get_session_with_traces.side_effect = fake_get
        sc = SessionCollection.from_langfuse(['s1', 's2'], loader=loader)
        assert len(sc) == 2
        assert {s.id for s in sc} == {'s1', 's2'}
        assert loader.get_session_with_traces.call_count == 2

    def test_turn_name_threaded_to_sessions(self):
        loader = MagicMock()

        def fake_get(session_id, show_progress=True, enrich=True):
            traces = [
                _chat_trace(0),  # name 'athena-chat'
                FakeRawTrace(
                    id='other',
                    name='other-chat',
                    input='hey',
                    output='yo',
                ),
            ]
            return ({'id': session_id}, traces)

        loader.get_session_with_traces.side_effect = fake_get
        sc = SessionCollection.from_langfuse(
            ['s1'], loader=loader, turn_name='athena-chat'
        )
        conv = sc[0].conversation()
        assert len(conv.messages) == 2  # only the athena-chat turn
        # Default survives filtering (re-wrap preserves config).
        assert sc.filter(lambda s: True)[0]._turn_name == 'athena-chat'

    def test_enrich_false_threaded_to_loader(self):
        loader = MagicMock()
        loader.get_session_with_traces.return_value = (
            {'id': 's1'},
            [_chat_trace(0)],
        )
        SessionCollection.from_langfuse(['s1'], loader=loader, enrich=False)
        _, kwargs = loader.get_session_with_traces.call_args
        assert kwargs['enrich'] is False

    def test_skips_missing_sessions(self):
        loader = MagicMock()

        def fake_get(session_id, show_progress=True, enrich=True):
            if session_id == 'missing':
                return (None, [])
            return ({'id': session_id}, [_chat_trace(0)])

        loader.get_session_with_traces.side_effect = fake_get
        sc = SessionCollection.from_langfuse(['s1', 'missing'], loader=loader)
        assert len(sc) == 1
        assert sc[0].id == 's1'
