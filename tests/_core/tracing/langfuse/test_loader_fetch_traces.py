from datetime import datetime
from types import SimpleNamespace
from typing import Any, Optional

from axion._core.tracing.loaders.langfuse import LangfuseTraceLoader


class FakeObservationsApi:
    def __init__(self, pages: dict[Optional[str], Any]):
        self.pages = pages
        self.calls = []

    def get_many(self, **kwargs):
        self.calls.append(kwargs)
        return self.pages[kwargs.get('cursor')]


class FakeTraceApi:
    def __init__(self, traces: dict[str, Any], summaries: Optional[list] = None):
        self.traces = traces
        self.summaries = summaries or []
        self.calls = []
        self.list_calls = []

    def get(self, trace_id: str):
        self.calls.append(trace_id)
        return self.traces[trace_id]

    def list(self, **kwargs):
        # trace.list() is page-based; return summaries on page 1, then empty so the
        # loader's pagination loop terminates.
        self.list_calls.append(kwargs)
        data = self.summaries if kwargs.get('page', 1) == 1 else []
        return SimpleNamespace(data=data)


def _response(trace_ids, cursor=None):
    return SimpleNamespace(
        data=[SimpleNamespace(trace_id=trace_id) for trace_id in trace_ids],
        meta=SimpleNamespace(cursor=cursor),
    )


def _loader(observation_pages, traces, trace_summaries=None) -> LangfuseTraceLoader:
    loader = LangfuseTraceLoader.__new__(LangfuseTraceLoader)
    loader.client = SimpleNamespace(
        api=SimpleNamespace(
            observations=FakeObservationsApi(observation_pages),
            trace=FakeTraceApi(traces, trace_summaries),
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


def test_fetch_traces_summary_path_uses_trace_list() -> None:
    # The fetch_full_traces=False (summary) path must use trace.list() — which
    # carries trace-level name/tags/metrics — instead of the observations API
    # (which does not return trace name) or per-id trace.get().
    loader = _loader(
        observation_pages={None: _response(['trace-a'])},
        traces={},
        trace_summaries=[SimpleNamespace(id='trace-c', name='target')],
    )

    traces = loader.fetch_traces(
        limit=1,
        mode='absolute',
        from_timestamp=datetime(2026, 1, 1),
        name='target',
        fetch_full_traces=False,
        show_progress=False,
    )

    # Returns trace.list() summaries carrying name, not id-only stubs.
    assert [trace.id for trace in traces] == ['trace-c']
    assert [trace.name for trace in traces] == ['target']
    # trace.list() was used with the name filter applied server-side.
    assert len(loader.client.api.trace.list_calls) == 1
    assert loader.client.api.trace.list_calls[0]['name'] == 'target'
    # No per-id trace.get() and no observations discovery on the summary path.
    assert loader.client.api.trace.calls == []
    assert loader.client.api.observations.calls == []
