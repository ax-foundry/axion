"""The loader rebuilds full traces from observation rows."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import Any

import pytest

from axion._core.tracing.loaders import langfuse as loader_module
from axion._core.tracing.loaders.langfuse_rows import TraceReadTruncated, parse_io
from tests._core.tracing.langfuse.fakes import make_loader, row


def test_loader_reads_no_removed_endpoint() -> None:
    """Langfuse retires the trace, session and v1 score reads; none may be called."""
    source = inspect.getsource(loader_module)
    for removed in ('api.trace.', 'api.sessions.', 'api.scores.get_many'):
        assert removed not in source, removed


def test_long_metadata_comes_back_whole_and_200_chars_is_read_again() -> None:
    long = 'x' * 900
    exactly = 'y' * 200
    loader = make_loader(
        [
            row(
                't1',
                'root',
                '2026-11-01T10:00:00Z',
                metadata={'long': long, 'exact': exactly},
            )
        ]
    )
    trace = loader.fetch_trace('t1')
    assert trace.metadata == {'long': long, 'exact': exactly}
    reads = loader.client.api.observations.calls
    assert len(reads) == 2
    assert reads[1]['expand_metadata'] == 'exact,long'


def test_short_metadata_needs_no_second_read() -> None:
    loader = make_loader(
        [row('t1', 'root', '2026-11-01T10:00:00Z', metadata={'k': 'v'})]
    )
    loader.fetch_trace('t1')
    assert len(loader.client.api.observations.calls) == 1


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        ({'a': 1}, {'a': 1}),
        ([1, 2], [1, 2]),
        ('{"a": 1}', {'a': 1}),
        ('[1, 2]', [1, 2]),
        ('plain text', 'plain text'),
        ('{"cut": ', '{"cut": '),
        (None, None),
    ],
)
def test_io_decoding(raw: Any, expected: Any) -> None:
    assert parse_io(raw) == expected


def test_root_io_is_decoded_on_trace_and_observation() -> None:
    loader = make_loader(
        [row('t1', 'root', '2026-11-01T10:00:00Z', input='{"q": "hi"}')]
    )
    trace = loader.fetch_trace('t1')
    assert trace.input == {'q': 'hi'}
    assert trace.observations[0].input == {'q': 'hi'}


def test_unknown_or_failed_trace_is_none() -> None:
    loader = make_loader([row('t1', 'root', '2026-11-01T10:00:00Z')])
    assert loader.fetch_trace('missing') is None

    def boom(**_: Any) -> Any:
        raise RuntimeError('400 bad request')

    loader.client.api.observations.get_many = boom
    assert loader.fetch_trace('t1') is None


def test_root_is_the_latest_flagged_row_and_timestamp_the_earliest() -> None:
    # A retried turn leaves two flagged roots: fields come from the latest attempt,
    # the start from the earliest.
    rows = [
        row('t1', 'child', '2026-11-01T10:00:05Z', parent='retry'),
        row(
            't1',
            'retry',
            '2026-11-01T10:00:04Z',
            parent='gone',
            root=True,
            name='final',
        ),
        row(
            't1',
            'first',
            '2026-11-01T10:00:00Z',
            parent='gone',
            root=True,
            name='first',
        ),
    ]
    trace = make_loader(rows).fetch_trace('t1')
    assert trace.name == 'final'
    assert trace.timestamp == datetime(2026, 11, 1, 10, tzinfo=UTC)


def test_root_without_the_flag_is_the_parentless_row() -> None:
    rows = [
        row('t1', 'child', '2026-11-01T10:00:01Z', parent='root', root=False),
        row('t1', 'root', '2026-11-01T10:00:00Z', root=False, name='turn'),
    ]
    trace = make_loader(rows).fetch_trace('t1')
    assert trace.name == 'turn'
    assert trace.timestamp == datetime(2026, 11, 1, 10, tzinfo=UTC)


def test_unset_trace_strings_are_none_and_unpriced_cost_is_zero() -> None:
    # The observations read returns "" for an unset string and {} for unset model
    # parameters.
    unset = row('t1', 'root', '2026-11-01T10:00:00Z')
    unset = unset.model_copy(
        update={
            'session_id': '',
            'user_id': '',
            'version': '',
            'release': '',
            'model_parameters': {},
        }
    )
    trace = make_loader([unset]).fetch_trace('t1')
    assert (trace.session_id, trace.user_id, trace.version, trace.release) == (
        None,
        None,
        None,
        None,
    )
    assert trace.total_cost == 0.0
    assert trace.observations[0].model_parameters is None


def test_unset_model_parameters_are_empty_except_on_spans_and_events() -> None:
    rows = [
        row('t1', 'root', '2026-11-01T10:00:00Z'),
        row('t1', 'gen', '2026-11-01T10:00:01Z', parent='root').model_copy(
            update={'type': 'GENERATION', 'model_parameters': {}}
        ),
        row('t1', 'tool', '2026-11-01T10:00:02Z', parent='root').model_copy(
            update={'type': 'TOOL', 'model_parameters': None}
        ),
    ]
    trace = make_loader(rows).fetch_trace('t1')
    params = {o.id: o.model_parameters for o in trace.observations}
    assert params == {'root': None, 'gen': {}, 'tool': {}}


def test_observations_are_in_start_order_with_ties_on_id() -> None:
    rows = [
        row('t1', 'late', '2026-11-01T10:00:09Z', parent='root'),
        row('t1', 'tie-b', '2026-11-01T10:00:05Z', parent='root'),
        row('t1', 'root', '2026-11-01T10:00:00Z'),
        row('t1', 'tie-a', '2026-11-01T10:00:05Z', parent='root'),
    ]
    trace = make_loader(rows).fetch_trace('t1')
    assert [o.id for o in trace.observations] == ['root', 'tie-a', 'tie-b', 'late']


def test_multi_page_reads_join() -> None:
    rows = [row('t1', 'root', '2026-11-01T10:00:00Z')] + [
        row('t1', f'c{i}', f'2026-11-01T10:00:{i:02d}Z', parent='root')
        for i in range(1, 7)
    ]
    loader = make_loader(rows, page_size=2)
    trace = loader.fetch_trace('t1')
    assert [o.id for o in trace.observations] == [r.id for r in rows]
    assert len(loader.client.api.observations.calls) == 4


def test_empty_continuation_pages_do_not_end_a_read() -> None:
    rows = [row('t1', 'root', '2026-11-01T10:00:00Z', session_id='s')] + [
        row('t1', f'c{i}', f'2026-11-01T10:00:0{i}Z', parent='root', session_id='s')
        for i in range(1, 4)
    ]
    loader = make_loader(rows, page_size=1, empty_pages=True)
    trace = loader.fetch_trace('t1')
    assert [o.id for o in trace.observations] == ['root', 'c1', 'c2', 'c3']
    _, traces = loader.get_session_with_traces('s', show_progress=False)
    assert [len(t.observations) for t in traces] == [4]


def test_session_traces_keep_child_rows_that_lack_the_session_id() -> None:
    # Only the root carries the session id; its children are found by trace id.
    rows = [
        row('t1', 'root', '2026-11-01T10:00:00Z', session_id='s'),
        row('t1', 'tool', '2026-11-01T10:00:01Z', parent='root').model_copy(
            update={'type': 'TOOL'}
        ),
        row('other', 'x', '2026-11-01T10:00:02Z'),
    ]
    _, traces = make_loader(rows).get_session_with_traces('s', show_progress=False)
    assert [(t.id, [o.id for o in t.observations]) for t in traces] == [
        ('t1', ['root', 'tool'])
    ]


def test_a_trace_without_its_root_has_no_input_or_output() -> None:
    # The root span was never recorded: only children whose parent is missing.
    rows = [
        row('t1', 'a', '2026-11-01T10:00:00Z', parent='gone', root=False, input='{}'),
        row('t1', 'b', '2026-11-01T10:00:01Z', parent='gone', root=False),
    ]
    trace = make_loader(rows).fetch_trace('t1')
    assert (trace.input, trace.output, trace.name) == (None, None, 'turn')
    assert trace.observations[0].input == {}


def test_a_root_under_a_remote_parent_keeps_the_trace_input_and_output() -> None:
    # The root span's parent lives in the calling service, so it is neither flagged
    # nor parentless; it carries the trace's name.
    rows = [
        row(
            't1', 'root', '2026-11-01T10:00:00Z', parent='remote', root=False
        ).model_copy(update={'name': 'turn', 'input': '{"q": "hi"}', 'output': 'done'}),
        row('t1', 'tool', '2026-11-01T10:00:01Z', parent='root', root=False),
    ]
    trace = make_loader(rows).fetch_trace('t1')
    assert (trace.input, trace.output) == ({'q': 'hi'}, 'done')


def test_rows_without_a_trace_name_take_the_root_name() -> None:
    # Older rows carry no trace name; the root under a remote parent names the trace,
    # and a session read filtered by that name keeps it.
    rows = [
        row(
            't1',
            'root',
            '2026-11-01T10:00:00Z',
            parent='remote',
            root=False,
            session_id='s',
            input='{"q": "hi"}',
        ).model_copy(update={'name': 'workflow', 'trace_name': None}),
        row('t1', 'step', '2026-11-01T10:00:01Z', parent='root', root=False).model_copy(
            update={'trace_name': None}
        ),
    ]
    loader = make_loader(rows)
    trace = loader.fetch_trace('t1')
    assert (trace.name, trace.input) == ('workflow', {'q': 'hi'})
    _, traces = loader.get_session_with_traces(
        's', show_progress=False, trace_name='workflow'
    )
    assert [t.id for t in traces] == ['t1']


def test_trace_and_session_reads_past_the_cap_raise() -> None:
    rows = [
        row('t1', f'o{i}', '2026-11-01T10:00:00Z', session_id='s') for i in range(5)
    ]
    loader = make_loader(rows, page_size=2)
    loader.max_pages = 2
    with pytest.raises(TraceReadTruncated):
        loader.fetch_trace('t1')
    with pytest.raises(TraceReadTruncated):
        loader.get_session_with_traces('s')
