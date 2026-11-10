"""Tests for LangfuseTraceLoader.fetch_traces and window listing."""

from __future__ import annotations

import random
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from axion._core.tracing.loaders.langfuse import LangfuseTraceLoader
from tests._core.tracing.langfuse.fakes import make_loader, row

START = datetime(2026, 11, 1, 10, tzinfo=UTC)
END = datetime(2026, 11, 1, 11, tzinfo=UTC)


def _at(minute: int, second: int = 0) -> str:
    return (START + timedelta(minutes=minute, seconds=second)).isoformat()


def _turn(tid: str, minute: int, children: int = 0, **kw: Any) -> list[Any]:
    rows = [row(tid, f'{tid}-root', _at(minute), **kw)]
    rows += [
        row(tid, f'{tid}-c{i}', _at(minute, i + 1), parent=f'{tid}-root', **kw)
        for i in range(children)
    ]
    return rows


def _list(loader: LangfuseTraceLoader, max_traces: int = 100, **kw: Any) -> Any:
    return loader.list_trace_ids(START, END, max_traces=max_traces, **kw)


# --- fetch_traces -------------------------------------------------------------------


def test_fetch_traces_name_filter_paginates_until_limit() -> None:
    rows = (
        _turn('a', 1, name='other')
        + _turn('b', 2, name='other')
        + _turn('c', 3, name='target')
        + _turn('d', 4, name='target')
    )
    traces = make_loader(rows, page_size=2).fetch_traces(
        limit=2,
        mode='absolute',
        from_timestamp=START,
        to_timestamp=END,
        name='target',
        show_progress=False,
    )
    assert sorted(t.id for t in traces) == ['c', 'd']
    assert {t.name for t in traces} == {'target'}


def test_fetch_traces_by_id_skips_unknown_ids() -> None:
    loader = make_loader(_turn('a', 1, children=2))
    traces = loader.fetch_traces(trace_ids=['a', 'missing'], show_progress=False)
    assert [t.id for t in traces] == ['a']
    assert len(traces[0].observations) == 3


def test_fetch_traces_summary_path_lists_without_per_trace_reads() -> None:
    rows = _turn('a', 1, name='target') + _turn('b', 2, name='other')
    rows += _turn('c', 3, name='target', children=2)
    loader = make_loader(rows)
    summaries = loader.fetch_traces(
        limit=5,
        mode='absolute',
        from_timestamp=START,
        to_timestamp=END,
        name='target',
        fetch_full_traces=False,
        show_progress=False,
    )
    assert [(s.id, s.name) for s in summaries] == [('c', 'target'), ('a', 'target')]
    calls = loader.client.api.observations.calls
    assert not any(c.get('trace_id') for c in calls)
    assert loader.client.api.scores_v3.calls == []


def test_summaries_are_newest_first_up_to_limit() -> None:
    rows = _turn('a', 1) + _turn('b', 2) + _turn('c', 3)
    summaries = make_loader(rows)._fetch_trace_summaries_via_list(
        limit=2, from_ts=START, to_ts=END, tags=None, name=None, environment=None
    )
    assert [s.id for s in summaries] == ['c', 'b']


# --- list_trace_ids -----------------------------------------------------------------


def test_listing_is_oldest_first_with_one_summary_per_trace() -> None:
    rows = (
        _turn('b', 20, children=5)
        + _turn('a', 10, children=3, tags=['x'])
        + _turn('c', 30)
    )
    random.Random(1).shuffle(rows)
    listing = _list(make_loader(rows))
    assert [t.id for t in listing.traces] == ['a', 'b', 'c']
    assert listing.traces[0].tags == ['x']
    assert listing.traces[0].timestamp == START + timedelta(minutes=10)
    assert (listing.end, listing.truncated) == (END, False)


@pytest.fixture
def los_angeles(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv('TZ', 'America/Los_Angeles')
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_listing_reads_a_naive_window_as_local_time(los_angeles: Any) -> None:
    """A naive bound means local time, as the SDK sends one; not UTC."""
    local_start = START.astimezone().replace(tzinfo=None)
    local_end = END.astimezone().replace(tzinfo=None)
    assert local_start.hour != START.hour  # the test runs off UTC

    rows = _turn('a', 10) + _turn('late', 70)  # 'late' is past END
    listing = make_loader(rows).list_trace_ids(local_start, local_end, max_traces=10)

    assert [t.id for t in listing.traces] == ['a']
    assert (listing.start, listing.end) == (START, END)


def test_listing_ties_break_on_id() -> None:
    listing = _list(make_loader(_turn('z', 5) + _turn('m', 5) + _turn('a', 5)))
    assert [t.id for t in listing.traces] == ['a', 'm', 'z']


def test_listing_cap_drops_the_boundary_and_cuts_end() -> None:
    rows = _turn('a', 1) + _turn('b', 2) + _turn('c', 3) + _turn('d', 3) + _turn('e', 4)
    listing = _list(make_loader(rows), max_traces=2)
    # The boundary is trace #2 (c, minute 3): a and b are kept; c and d, which share
    # its timestamp, are dropped.
    assert [t.id for t in listing.traces] == ['a', 'b']
    assert listing.end == START + timedelta(minutes=3)
    assert listing.truncated


def test_listing_cannot_advance_when_one_timestamp_holds_more_than_the_cap() -> None:
    rows = _turn('a', 1) + _turn('b', 1) + _turn('c', 1)
    listing = _list(make_loader(rows), max_traces=2)
    assert listing.traces == []
    assert listing.end == START
    assert listing.truncated


def test_listing_drops_a_trace_whose_root_is_before_the_window() -> None:
    early_root = row('early', 'early-root', (START - timedelta(minutes=5)).isoformat())
    early_child = row('early', 'early-c', _at(1), parent='early-root')
    loader = make_loader([early_root, early_child] + _turn('a', 2))
    assert [t.id for t in _list(loader).traces] == ['a']


def test_listing_drops_a_retried_trace_whose_first_attempt_is_before_the_window() -> (
    None
):
    before = (START - timedelta(minutes=5)).isoformat()
    retried = [
        row('r', 'r-first', before, parent='gone', root=True),
        row('r', 'r-retry', _at(1), parent='gone', root=True),
    ]
    loader = make_loader(retried + _turn('a', 2))
    assert [t.id for t in _list(loader).traces] == ['a']


def test_listing_filters_name_tags_and_environment() -> None:
    rows = (
        _turn('a', 1, name='chat', tags=['p', 'q'])
        + _turn('b', 2, name='chat', tags=['p'])
        + _turn('c', 3, name='other', tags=['p', 'q'])
        + _turn('d', 4, name='chat', tags=['p', 'q'], environment='preview')
    )
    listing = _list(
        make_loader(rows), name='chat', tags=['p', 'q'], environment='production'
    )
    assert [t.id for t in listing.traces] == ['a']


def test_listing_halves_a_window_over_the_page_cap_and_reads_it_whole() -> None:
    rows = [r for m in range(0, 60, 5) for r in _turn(f't{m:02d}', m, children=1)]
    loader = make_loader(rows, page_size=4)
    loader.list_max_pages = 3  # 12 rows a window
    listing = _list(loader)
    assert listing.truncated
    assert listing.end == START + timedelta(minutes=30)
    assert [t.id for t in listing.traces] == [f't{m:02d}' for m in range(0, 30, 5)]


def test_newest_first_listing_reads_older_chunks_until_the_limit() -> None:
    rows = [r for m in range(0, 60, 5) for r in _turn(f't{m:02d}', m, children=1)]
    loader = make_loader(rows, page_size=4)
    loader.list_max_pages = 3  # 12 rows a window: the hour needs several chunks
    listing = _list(loader, max_traces=10, order='desc')
    assert [t.id for t in listing.traces] == [f't{m:02d}' for m in range(55, 5, -5)]
    assert listing.truncated
    every = _list(loader, max_traces=50, order='desc')
    assert len(every.traces) == 12
    assert (every.start, every.truncated) == (START, False)


def test_newest_first_cut_through_a_tie_drops_the_tie_and_resumes_cleanly() -> None:
    rows = _turn('a', 1) + _turn('b', 2) + _turn('c', 2) + _turn('d', 3)
    loader = make_loader(rows)
    listing = _list(loader, max_traces=2, order='desc')
    # b shares c's timestamp and was cut, so c goes with it: the listing holds only
    # traces at or after its start.
    assert [t.id for t in listing.traces] == ['d']
    assert listing.start == START + timedelta(minutes=2, microseconds=1)
    assert listing.truncated
    assert all(listing.start <= t.timestamp < listing.end for t in listing.traces)
    older = loader.list_trace_ids(START, listing.start, max_traces=10, order='desc')
    assert [t.id for t in older.traces] == ['c', 'b', 'a']


def test_listing_reads_through_empty_continuation_pages() -> None:
    rows = [r for m in range(0, 60, 5) for r in _turn(f't{m:02d}', m, children=1)]
    listing = _list(make_loader(rows, page_size=3, empty_pages=True))
    assert len(listing.traces) == 12
    assert not listing.truncated
