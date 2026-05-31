from datetime import datetime
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from axion._core.tracing.loaders.langfuse import LangfuseTraceLoader


class FakeObservationsApi:
    def __init__(self, pages: dict[Optional[str], Any]):
        self.pages = pages
        self.calls = []

    def get_many(self, **kwargs):
        self.calls.append(kwargs)
        return self.pages[kwargs.get('cursor')]


class FakeTraceApi:
    def __init__(self, traces: dict[str, Any]):
        self.traces = traces
        self.calls = []

    def get(self, trace_id: str):
        self.calls.append(trace_id)
        return self.traces[trace_id]


def _response(trace_ids, cursor=None):
    return SimpleNamespace(
        data=[SimpleNamespace(trace_id=trace_id) for trace_id in trace_ids],
        meta=SimpleNamespace(cursor=cursor),
    )


def _loader(observation_pages, traces) -> LangfuseTraceLoader:
    loader = LangfuseTraceLoader.__new__(LangfuseTraceLoader)
    loader.client = SimpleNamespace(
        api=SimpleNamespace(
            observations=FakeObservationsApi(observation_pages),
            trace=FakeTraceApi(traces),
        )
    )
    loader._client_initialized = True
    loader.request_pacing = 0.0
    loader._execute_with_retry = lambda fn, description: fn()
    return loader


def test_fetch_traces_name_filter_paginates_until_limit() -> None:
    loader = _loader(
        observation_pages={
            None: _response(['trace-a', 'trace-b'], cursor='page-2'),
            'page-2': _response(['trace-c', 'trace-d']),
        },
        traces={
            'trace-a': SimpleNamespace(id='trace-a', name='other'),
            'trace-b': SimpleNamespace(id='trace-b', name='other'),
            'trace-c': SimpleNamespace(id='trace-c', name='target'),
            'trace-d': SimpleNamespace(id='trace-d', name='target'),
        },
    )

    traces = loader.fetch_traces(
        limit=2,
        mode='absolute',
        from_timestamp=datetime(2026, 1, 1),
        name='target',
        show_progress=False,
    )

    assert [trace.id for trace in traces] == ['trace-c', 'trace-d']
    assert len(loader.client.api.observations.calls) == 2
    assert loader.client.api.trace.calls == [
        'trace-a',
        'trace-b',
        'trace-c',
        'trace-d',
    ]


def test_fetch_traces_name_filter_with_stubs_warns_and_skips_filter() -> None:
    loader = _loader(
        observation_pages={None: _response(['trace-a'])},
        traces={},
    )

    with pytest.warns(UserWarning, match='name filtering requires'):
        traces = loader.fetch_traces(
            limit=1,
            mode='absolute',
            from_timestamp=datetime(2026, 1, 1),
            name='target',
            fetch_full_traces=False,
            show_progress=False,
        )

    assert [trace.id for trace in traces] == ['trace-a']
    assert loader.client.api.trace.calls == []
