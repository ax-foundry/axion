"""Tests for TraceCollection."""

from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from axion._core.tracing.collection.trace import Trace
from axion._core.tracing.collection.trace_collection import TraceCollection

# ---------------------------------------------------------------------------
# Fake trace data
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
    name: str = 'my-trace'
    input: Any = None
    output: Any = None
    observations: List[Any] = field(default_factory=list)
    latency: float = 0.5
    tags: List[str] = field(default_factory=list)


def _make_traces(n=3):
    return [
        FakeRawTrace(
            id=f't-{i}',
            name=f'trace-{i}',
            input={'query': f'question {i}'},
            output={'answer': f'response {i}'},
            observations=[
                FakeObservation(name='gen', type='GENERATION'),
            ],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# from_raw_traces factory
# ---------------------------------------------------------------------------


class TestFromRawTraces:
    def test_basic(self):
        raw = _make_traces(2)
        c = TraceCollection.from_raw_traces(raw)
        assert len(c) == 2
        assert isinstance(c[0], Trace)

    def test_with_name(self):
        c = TraceCollection.from_raw_traces(_make_traces(1), name='test')
        assert len(c) == 1


# ---------------------------------------------------------------------------
# Collection protocol
# ---------------------------------------------------------------------------


class TestCollectionProtocol:
    def test_len(self):
        c = TraceCollection(_make_traces(3))
        assert len(c) == 3

    def test_getitem(self):
        c = TraceCollection(_make_traces(3))
        assert isinstance(c[0], Trace)
        assert c[0].id == 't-0'

    def test_iter(self):
        c = TraceCollection(_make_traces(2))
        traces = list(c)
        assert len(traces) == 2
        assert all(isinstance(t, Trace) for t in traces)

    def test_repr(self):
        c = TraceCollection(_make_traces(5))
        assert 'TraceCollection' in repr(c)
        assert '5' in repr(c)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFilter:
    def test_filter_lambda(self):
        c = TraceCollection(_make_traces(5))
        filtered = c.filter(lambda t: t.id in ('t-0', 't-2'))
        assert len(filtered) == 2
        assert isinstance(filtered, TraceCollection)

    def test_filter_empty(self):
        c = TraceCollection(_make_traces(3))
        filtered = c.filter(lambda t: False)
        assert len(filtered) == 0

    def test_filter_by_attribute(self):
        c = TraceCollection(_make_traces(3))
        filtered = c.filter_by(name='trace-1')
        assert len(filtered) == 1
        assert filtered[0].name == 'trace-1'

    def test_filter_by_missing_attribute(self):
        c = TraceCollection(_make_traces(3))
        filtered = c.filter_by(nonexistent='value')
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_save_and_load(self, tmp_path):
        raw = [
            {
                'id': 't1',
                'name': 'test',
                'input': {'query': 'hello'},
                'output': {'answer': 'world'},
                'observations': [
                    {'name': 'gen', 'type': 'GENERATION'},
                ],
            },
        ]
        c = TraceCollection(raw)
        path = tmp_path / 'traces.json'
        c.save_json(path)

        loaded = TraceCollection.load_json(path)
        assert len(loaded) == 1
        assert loaded[0].id == 't1'

    def test_load_non_list_raises(self, tmp_path):
        path = tmp_path / 'bad.json'
        path.write_text('{"not": "a list"}')
        with pytest.raises(ValueError, match='JSON list'):
            TraceCollection.load_json(path)

    def test_save_creates_parent_dirs(self, tmp_path):
        c = TraceCollection([{'id': 'x', 'observations': []}])
        path = tmp_path / 'sub' / 'dir' / 'traces.json'
        c.save_json(path)
        assert path.exists()


class TestToJsonable:
    def test_primitives(self):
        assert TraceCollection._to_jsonable(42) == 42
        assert TraceCollection._to_jsonable('hello') == 'hello'
        assert TraceCollection._to_jsonable(None) is None

    def test_enum(self):
        from axion._core.tracing.collection.models import ModelUsageUnit

        assert TraceCollection._to_jsonable(ModelUsageUnit.TOKENS) == 'TOKENS'

    def test_dataclass(self):
        @dataclass
        class DC:
            x: int = 1
            y: str = 'a'

        result = TraceCollection._to_jsonable(DC())
        assert result == {'x': 1, 'y': 'a'}

    def test_nested_dict(self):
        result = TraceCollection._to_jsonable({'a': {'b': 1}})
        assert result == {'a': {'b': 1}}

    def test_list(self):
        result = TraceCollection._to_jsonable([1, 'two', None])
        assert result == [1, 'two', None]


# ---------------------------------------------------------------------------
# to_list
# ---------------------------------------------------------------------------


class TestToList:
    def test_returns_raw_objects(self):
        raw = _make_traces(2)
        c = TraceCollection(raw)
        result = c.to_list()
        assert len(result) == 2
        assert result[0] is raw[0]


# ---------------------------------------------------------------------------
# to_dataset
# ---------------------------------------------------------------------------


class TestToDataset:
    def test_default_transform(self):
        raw = [
            FakeRawTrace(
                id='t1',
                input={'query': 'what is AI?'},
                output={'answer': 'artificial intelligence'},
            ),
        ]
        c = TraceCollection(raw)
        ds = c.to_dataset(name='test-ds')
        assert ds.name == 'test-ds'
        assert len(ds) == 1
        item = ds[0]
        assert item.query == 'what is AI?'
        assert item.actual_output == 'artificial intelligence'

    def test_default_transform_with_string_io(self):
        raw = [
            FakeRawTrace(
                id='t1',
                input='direct question',
                output='direct answer',
            ),
        ]
        c = TraceCollection(raw)
        ds = c.to_dataset()
        assert len(ds) == 1
        assert ds[0].query == 'direct question'
        assert ds[0].actual_output == 'direct answer'

    def test_custom_transform(self):
        raw = [
            FakeRawTrace(
                id='t1',
                input='hello',
                output='world',
                observations=[FakeObservation(name='gen', type='GENERATION')],
            ),
        ]
        c = TraceCollection(raw)

        def custom(trace):
            return {
                'query': f'custom: {trace.input}',
                'actual_output': f'custom: {trace.output}',
            }

        ds = c.to_dataset(transform=custom)
        assert len(ds) == 1
        assert ds[0].query == 'custom: hello'

    def test_empty_traces_skipped(self):
        raw = [FakeRawTrace(id='t1', input=None, output=None)]
        c = TraceCollection(raw)
        ds = c.to_dataset()
        assert len(ds) == 0

    def test_trace_id_included(self):
        raw = [
            FakeRawTrace(
                id='t99',
                input='q',
                output='a',
            ),
        ]
        c = TraceCollection(raw)
        ds = c.to_dataset()
        assert len(ds) == 1
        assert ds[0].trace_id == 't99'

    def test_transform_returns_dataset_item(self):
        from axion.dataset import DatasetItem

        raw = [
            FakeRawTrace(
                id='t1',
                input='hello',
                output='world',
            ),
        ]
        c = TraceCollection(raw)

        def extractor(trace):
            return DatasetItem(
                query=f'extracted: {trace.input}',
                actual_output=f'extracted: {trace.output}',
                trace_id=str(trace.id),
            )

        ds = c.to_dataset(transform=extractor)
        assert len(ds) == 1
        assert ds[0].query == 'extracted: hello'
        assert ds[0].actual_output == 'extracted: world'
        assert ds[0].trace_id == 't1'

    def test_dict_traces(self):
        raw = [
            {
                'id': 'd1',
                'input': {'query': 'hello'},
                'output': {'answer': 'world'},
                'observations': [],
            },
        ]
        c = TraceCollection(raw)
        ds = c.to_dataset()
        assert len(ds) == 1
        assert ds[0].query == 'hello'


# ---------------------------------------------------------------------------
# from_langfuse with mocked loader
# ---------------------------------------------------------------------------


class TestFromLangfuse:
    def test_with_trace_ids(self):
        fake_loader = MagicMock()
        fake_loader.fetch_traces.return_value = [
            {'id': 'lf1', 'observations': [], 'input': 'q', 'output': 'a'},
        ]

        c = TraceCollection.from_langfuse(
            trace_ids=['lf1'],
            loader=fake_loader,
            show_progress=False,
        )
        assert len(c) == 1
        fake_loader.fetch_traces.assert_called_once_with(
            trace_ids=['lf1'], show_progress=False
        )

    def test_with_filters(self):
        fake_loader = MagicMock()
        fake_loader.fetch_traces.return_value = []

        c = TraceCollection.from_langfuse(
            limit=10,
            days_back=3,
            tags=['prod'],
            name='my-trace',
            loader=fake_loader,
            show_progress=False,
        )
        assert len(c) == 0
        fake_loader.fetch_traces.assert_called_once_with(
            limit=10,
            days_back=3,
            tags=['prod'],
            name='my-trace',
            show_progress=False,
        )
