"""Tests for LangfuseTraceLoader session reads."""

from __future__ import annotations

import random
from datetime import UTC, datetime
from typing import Any

import pytest

from axion._core.tracing.collection.session import Session
from tests._core.tracing.langfuse.fakes import make_loader, row


def _session_rows() -> list[Any]:
    rows = []
    for minute in (30, 10, 20):
        tid = f'turn{minute}'
        rows += [
            row(
                tid,
                f'{tid}-root',
                f'2026-11-01T10:{minute}:00Z',
                session_id='s1',
                name='chat-turn',
                input=f'"in-{tid}"',
            ),
            row(
                tid,
                f'{tid}-gen',
                f'2026-11-01T10:{minute}:01Z',
                parent=f'{tid}-root',
                session_id='s1',
            ),
        ]
    rows.append(
        row(
            'other',
            'other-root',
            '2026-11-01T10:05:00Z',
            session_id='s1',
            name='pipeline',
        )
    )
    random.Random(7).shuffle(rows)
    return rows


def _flaky(loader: Any, bad_id: str) -> None:
    build = loader._build_trace

    def flaky(trace_id: str, rows: Any, scores: Any) -> Any:
        if trace_id == bad_id:
            raise ValueError('bad row')
        return build(trace_id, rows, scores)

    loader._build_trace = flaky


class TestEnrich:
    def test_enrich_true_rebuilds_full_traces_in_start_order(self):
        loader = make_loader(_session_rows())
        session, traces = loader.get_session_with_traces('s1', show_progress=False)
        order = ['other', 'turn10', 'turn20', 'turn30']
        assert session.id == 's1'
        assert [t.id for t in traces] == order
        assert [len(t.observations) for t in traces] == [1, 2, 2, 2]
        assert [t.id for t in session.traces] == order

    def test_enrich_false_returns_stubs_and_reads_no_scores(self):
        loader = make_loader(_session_rows())
        session, traces = loader.get_session_with_traces(
            's1', show_progress=False, enrich=False
        )
        assert session.id == 's1'
        assert [t.id for t in traces] == ['other', 'turn10', 'turn20', 'turn30']
        assert not hasattr(traces[0], 'observations')
        # Stub-level I/O is present so a conversation can still be built.
        assert traces[1].input == 'in-turn10'
        assert loader.client.api.scores_v3.calls == []

    def test_unknown_session_returns_none(self):
        loader = make_loader(_session_rows())
        result = loader.get_session_with_traces('nope', show_progress=False)
        assert result == (None, [])

    def test_failed_read_returns_none(self):
        loader = make_loader(_session_rows())

        def boom(**_: Any) -> Any:
            raise RuntimeError('400 bad request')

        loader.client.api.observations.get_many = boom
        result = loader.get_session_with_traces('s1', show_progress=False)
        assert result == (None, [])

    def test_unknown_session_gives_an_empty_identifiable_session(self):
        session = Session.from_langfuse(
            'nope', loader=make_loader(_session_rows()), show_progress=False
        )
        assert session.id == 'nope'
        assert len(session) == 0

    @pytest.mark.parametrize('retain', [True, False])
    def test_a_trace_that_wont_rebuild_does_not_take_the_session_down(self, retain):
        loader = make_loader(_session_rows())
        _flaky(loader, 'turn20')
        _, traces = loader.get_session_with_traces(
            's1', show_progress=False, retain_stub_on_failure=retain
        )
        ids = [t.id for t in traces]
        if retain:
            assert ids == ['other', 'turn10', 'turn20', 'turn30']
            assert not hasattr(traces[2], 'observations')
        else:
            assert ids == ['other', 'turn10', 'turn30']

    def test_get_session_traces_drops_traces_that_wont_rebuild(self):
        loader = make_loader(_session_rows())
        _flaky(loader, 'turn20')
        traces = loader.get_session_traces('s1', show_progress=False)
        assert [t.id for t in traces] == ['other', 'turn10', 'turn30']


class TestTraceFilter:
    def test_trace_name_keeps_matching_traces_and_reads_only_their_scores(self):
        loader = make_loader(_session_rows())
        _, traces = loader.get_session_with_traces(
            's1', show_progress=False, trace_name='chat-turn'
        )
        assert [t.id for t in traces] == ['turn10', 'turn20', 'turn30']
        scored = [c['trace_id'] for c in loader.client.api.scores_v3.calls]
        assert scored == ['turn10', 'turn20', 'turn30']

    def test_trace_name_with_enrich_false_filters_stubs(self):
        loader = make_loader(_session_rows())
        _, traces = loader.get_session_with_traces(
            's1', show_progress=False, enrich=False, trace_name='pipeline'
        )
        assert [t.id for t in traces] == ['other']

    def test_trace_predicate_filters(self):
        loader = make_loader(_session_rows())
        _, traces = loader.get_session_with_traces(
            's1',
            show_progress=False,
            trace_predicate=lambda s: s.id.endswith('0') and s.id != 'turn20',
        )
        assert [t.id for t in traces] == ['turn10', 'turn30']

    def test_no_match_returns_the_session_and_no_traces(self):
        loader = make_loader(_session_rows())
        session, traces = loader.get_session_with_traces(
            's1', show_progress=False, trace_name='missing'
        )
        assert session.id == 's1'
        assert traces == []


def test_fetch_session_is_a_session_with_traces():
    loader = make_loader(_session_rows())
    session = loader.fetch_session('s1')
    assert session.id == 's1'
    assert session.created_at == datetime(2026, 11, 1, 10, 5, tzinfo=UTC)
    assert len(session.traces) == 4
    assert loader.fetch_session('nope') is None
