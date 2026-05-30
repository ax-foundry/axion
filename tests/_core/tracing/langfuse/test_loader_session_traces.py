"""Tests for LangfuseTraceLoader.get_session_with_traces enrichment behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from axion._core.tracing.loaders.langfuse import LangfuseTraceLoader


def _loader() -> LangfuseTraceLoader:
    # Skip __init__ — we patch the network-touching methods directly.
    return LangfuseTraceLoader.__new__(LangfuseTraceLoader)


def _session_with_stubs(*trace_ids):
    stubs = [
        SimpleNamespace(id=tid, input=f'in-{tid}', output=f'out-{tid}')
        for tid in trace_ids
    ]
    return SimpleNamespace(id='sess-1', traces=stubs)


def _session_with_named_stubs(*named):
    """*named* is a sequence of ``(trace_id, name)`` pairs."""
    stubs = [
        SimpleNamespace(id=tid, name=name, input=f'in-{tid}', output=f'out-{tid}')
        for tid, name in named
    ]
    return SimpleNamespace(id='sess-1', traces=stubs)


class TestEnrich:
    def test_enrich_true_fetches_full_traces(self):
        loader = _loader()
        loader.fetch_session = MagicMock(return_value=_session_with_stubs('a', 'b'))
        loader.fetch_trace = MagicMock(
            side_effect=lambda tid: SimpleNamespace(id=tid, full=True)
        )

        session, traces = loader.get_session_with_traces(
            'sess-1', show_progress=False, enrich=True
        )
        assert session.id == 'sess-1'
        assert loader.fetch_trace.call_count == 2
        assert all(getattr(t, 'full', False) for t in traces)

    def test_enrich_false_skips_fetch_trace_and_returns_stubs(self):
        loader = _loader()
        session_obj = _session_with_stubs('a', 'b')
        loader.fetch_session = MagicMock(return_value=session_obj)
        loader.fetch_trace = MagicMock()

        session, traces = loader.get_session_with_traces(
            'sess-1', show_progress=False, enrich=False
        )
        assert session is session_obj
        loader.fetch_trace.assert_not_called()
        assert [t.id for t in traces] == ['a', 'b']
        # Stub-level I/O is present so a conversation can still be built.
        assert traces[0].input == 'in-a'

    def test_missing_session_returns_none(self):
        loader = _loader()
        loader.fetch_session = MagicMock(return_value=None)
        loader.fetch_trace = MagicMock()

        session, traces = loader.get_session_with_traces(
            'nope', show_progress=False, enrich=False
        )
        assert session is None
        assert traces == []
        loader.fetch_trace.assert_not_called()

    def test_empty_session_returns_no_traces(self):
        loader = _loader()
        loader.fetch_session = MagicMock(
            return_value=SimpleNamespace(id='sess-1', traces=[])
        )
        loader.fetch_trace = MagicMock()

        session, traces = loader.get_session_with_traces(
            'sess-1', show_progress=False, enrich=False
        )
        assert session.id == 'sess-1'
        assert traces == []


class TestTraceFilter:
    def test_trace_name_only_fetches_matching(self):
        loader = _loader()
        loader.fetch_session = MagicMock(
            return_value=_session_with_named_stubs(
                ('a', 'chat-turn'), ('b', 'pipeline'), ('c', 'chat-turn')
            )
        )
        loader.fetch_trace = MagicMock(
            side_effect=lambda tid: SimpleNamespace(id=tid, full=True)
        )

        _, traces = loader.get_session_with_traces(
            'sess-1', show_progress=False, trace_name='chat-turn'
        )
        # Only the two 'chat-turn' stubs are enriched; 'pipeline' is never fetched.
        assert loader.fetch_trace.call_count == 2
        assert sorted(c.args[0] for c in loader.fetch_trace.call_args_list) == [
            'a',
            'c',
        ]
        assert {t.id for t in traces} == {'a', 'c'}

    def test_trace_name_with_enrich_false_filters_stubs(self):
        loader = _loader()
        loader.fetch_session = MagicMock(
            return_value=_session_with_named_stubs(
                ('a', 'chat-turn'), ('b', 'pipeline')
            )
        )
        loader.fetch_trace = MagicMock()

        _, traces = loader.get_session_with_traces(
            'sess-1', show_progress=False, enrich=False, trace_name='chat-turn'
        )
        loader.fetch_trace.assert_not_called()
        assert [t.id for t in traces] == ['a']

    def test_trace_predicate_filters(self):
        loader = _loader()
        loader.fetch_session = MagicMock(
            return_value=_session_with_named_stubs(
                ('a', 'chat-turn'), ('b', 'pipeline'), ('c', 'chat-turn-v2')
            )
        )
        loader.fetch_trace = MagicMock(
            side_effect=lambda tid: SimpleNamespace(id=tid, full=True)
        )

        _, traces = loader.get_session_with_traces(
            'sess-1',
            show_progress=False,
            trace_predicate=lambda s: s.name.startswith('chat'),
        )
        assert {t.id for t in traces} == {'a', 'c'}

    def test_no_match_returns_empty(self):
        loader = _loader()
        loader.fetch_session = MagicMock(
            return_value=_session_with_named_stubs(('a', 'pipeline'))
        )
        loader.fetch_trace = MagicMock()

        session, traces = loader.get_session_with_traces(
            'sess-1', show_progress=False, trace_name='chat-turn'
        )
        assert session.id == 'sess-1'
        assert traces == []
        loader.fetch_trace.assert_not_called()
