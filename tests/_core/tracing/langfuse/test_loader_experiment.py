import inspect
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import format_span_id, format_trace_id

from axion._core.tracing.loaders import langfuse as loader_module
from axion._core.tracing.loaders import langfuse_experiments as experiments
from axion._core.tracing.loaders.langfuse import LangfuseTraceLoader


@dataclass
class FakeMetricScore:
    name: str
    score: float
    explanation: Optional[str] = None
    signals: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FakeDatasetItemInput:
    id: str
    query: str
    actual_output: str
    expected_output: str
    trace_id: Optional[str] = None
    observation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FakeTestResult:
    test_case: FakeDatasetItemInput
    score_results: List[FakeMetricScore]


@dataclass
class FakeEvaluationResult:
    results: List[FakeTestResult]
    run_id: str = 'run-12345678'
    evaluation_name: Optional[str] = None


class FakeRootSpan:
    """The Langfuse span wrapper around a real OTel span."""

    def __init__(self, otel_span: Any):
        context = otel_span.get_span_context()
        self.trace_id = format_trace_id(context.trace_id)
        self.id = format_span_id(context.span_id)
        self.set_trace_io_calls: List[Dict[str, Any]] = []

    def set_trace_io(self, **kwargs):
        self.set_trace_io_calls.append(kwargs)


class FakeDatasetItem:
    def __init__(self, item_id: str, expected_output: Any = None):
        self.id = item_id
        self.expected_output = expected_output


class FakeDataset:
    def __init__(self, items: List[FakeDatasetItem], dataset_id: str = 'ds-id'):
        self.id = dataset_id
        self.items = items


class FakeApi:
    """Has no dataset-run endpoints; touching one fails the item."""


class FakeLangfuseClient:
    def __init__(self, dataset: FakeDataset, project_id: Any = 'proj-1'):
        self._dataset = dataset
        self._project_id = project_id
        self._base_url = 'https://langfuse.example/'
        self.created_items: List[Dict[str, Any]] = []
        self.created_scores: List[Dict[str, Any]] = []
        self.started_observations: List[Dict[str, Any]] = []
        self.api = FakeApi()
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.tracer = provider.get_tracer('fake-langfuse')

    def _get_project_id(self):
        if isinstance(self._project_id, Exception):
            raise self._project_id
        return self._project_id

    def create_dataset(self, name: str):
        pass

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        with self.tracer.start_as_current_span(kwargs['name']) as otel_span:
            span = FakeRootSpan(otel_span)
            self.started_observations.append({**kwargs, 'span': span})
            yield span

    def create_dataset_item(self, dataset_name: str, id: str, **kwargs):
        self.created_items.append({'dataset_name': dataset_name, 'id': id, **kwargs})

    def flush(self):
        pass

    def get_dataset(self, name: str):
        return self._dataset

    def create_score(self, **kwargs):
        self.created_scores.append(kwargs)

    def spans(self) -> Dict[str, Any]:
        return {s.name: s for s in self.exporter.get_finished_spans()}


@pytest.fixture
def make_loader(monkeypatch):
    """A loader around a fake client; the `langfuse` stub is removed after the test."""
    fake_module = types.ModuleType('langfuse')
    fake_module.Langfuse = object
    monkeypatch.setitem(sys.modules, 'langfuse', fake_module)

    def make(fake_client: FakeLangfuseClient) -> LangfuseTraceLoader:
        loader = LangfuseTraceLoader.__new__(LangfuseTraceLoader)
        loader.max_retries = 0
        loader.base_delay = 0.0
        loader.max_delay = 0.0
        loader.request_pacing = 0.0
        loader.default_tags = []
        loader.client = fake_client
        loader._client_initialized = True
        return loader

    return make


def _result(item_id: str, score: float, **case: Any) -> FakeTestResult:
    return FakeTestResult(
        test_case=FakeDatasetItemInput(
            id=item_id,
            query=f'q-{item_id}',
            actual_output=f'a-{item_id}',
            expected_output=f'e-{item_id}',
            **case,
        ),
        score_results=[FakeMetricScore(name='m1', score=score)],
    )


def test_experiment_id_matches_the_langfuse_derivation():
    # Vectors computed with langfuse 5.2.0's Langfuse._create_experiment_id.
    assert (
        experiments.experiment_id_for('proj-1', 'ds-id', 'experiment-v6')
        == 'c1e4eb221c0ed610'
    )
    assert (
        experiments.experiment_id_for('cl8x2p0v20000', 'cmab1dataset', 'échec run · 1')
        == 'bd16edea99266164'
    )
    assert experiments.experiment_id_for(None, 'ds-id', 'r') is None
    assert experiments.experiment_id_for('proj-1', None, 'r') is None


def test_attribute_names_match_the_sdk():
    attributes = pytest.importorskip('langfuse._client.attributes')
    sdk = attributes.LangfuseOtelSpanAttributes
    for name in (
        'EXPERIMENT_ID',
        'EXPERIMENT_NAME',
        'EXPERIMENT_METADATA',
        'EXPERIMENT_DATASET_ID',
        'EXPERIMENT_ITEM_ID',
        'EXPERIMENT_ITEM_EXPECTED_OUTPUT',
        'EXPERIMENT_ITEM_METADATA',
        'EXPERIMENT_ITEM_ROOT_OBSERVATION_ID',
    ):
        assert getattr(experiments, name) == getattr(sdk, name), name


def test_the_loader_calls_no_dataset_run_endpoint():
    source = inspect.getsource(loader_module)
    for removed in ('dataset_run_items', 'get_dataset_run', 'DatasetRunItem'):
        assert removed not in source, removed


def test_each_item_is_a_root_span_carrying_the_experiment_attributes(make_loader):
    dataset = FakeDataset(
        [
            FakeDatasetItem('item-1', expected_output={'answer': 'curated'}),
            FakeDatasetItem('item-2'),
        ]
    )
    client = FakeLangfuseClient(dataset)
    loader = make_loader(client)
    eval_result = FakeEvaluationResult(
        results=[
            _result('item-1', 0.5, trace_id='trace-1', observation_id='obs-1'),
            _result('item-2', 0.7),
        ],
        run_id='run-abcdef01',
    )

    stats = loader.upload_experiment(
        evaluation_result=eval_result,
        dataset_name='ds',
        run_name='experiment-v6',
        run_metadata={'branch': 'main', 'attempt': 2},
        flush=False,
        tags=['production'],
    )

    assert stats['errors'] == []
    assert stats['runs_created'] == 2
    assert stats['scores_uploaded'] == 2
    assert stats['experiment_id'] == 'c1e4eb221c0ed610'
    assert stats['run_url'] == (
        'https://langfuse.example/project/proj-1/datasets/ds-id/runs/c1e4eb221c0ed610'
    )

    spans = client.spans()
    first = spans['dataset-run:experiment-v6:item-1']
    second = spans['dataset-run:experiment-v6:item-2']
    assert first.parent is None and second.parent is None
    assert first.context.trace_id != second.context.trace_id

    attrs = dict(first.attributes)
    assert attrs == {
        'langfuse.experiment.id': 'c1e4eb221c0ed610',
        'langfuse.experiment.name': 'experiment-v6',
        'langfuse.experiment.dataset.id': 'ds-id',
        'langfuse.experiment.item.id': 'item-1',
        'langfuse.experiment.item.root_observation_id': format_span_id(
            first.context.span_id
        ),
        # The dataset's stored expected output wins over the local one.
        'langfuse.experiment.item.expected_output': '{"answer": "curated"}',
        'langfuse.experiment.metadata.axion_run_id': 'run-abcdef01',
        'langfuse.experiment.metadata.branch': 'main',
        'langfuse.experiment.metadata.attempt': '2',
        'langfuse.experiment.metadata.tags': '["production"]',
        'langfuse.experiment.item.metadata.trace_id': 'trace-1',
        'langfuse.experiment.item.metadata.observation_id': 'obs-1',
    }
    second_attrs = dict(second.attributes)
    assert second_attrs['langfuse.experiment.item.expected_output'] == (
        '{"expected_output": "e-item-2"}'
    )
    assert not any('item.metadata' in key for key in second_attrs)

    for observation in client.started_observations:
        assert observation['span'].set_trace_io_calls

    # Scores go on each item's root observation.
    scores = {s['value']: s for s in client.created_scores}
    assert scores[0.5]['trace_id'] == format_trace_id(first.context.trace_id)
    assert scores[0.5]['observation_id'] == format_span_id(first.context.span_id)
    assert scores[0.7]['observation_id'] == format_span_id(second.context.span_id)


def test_uploads_under_one_run_name_share_an_experiment(make_loader):
    def upload(run_name: str) -> Dict[str, Any]:
        client = FakeLangfuseClient(FakeDataset([FakeDatasetItem('item-1')]))
        return make_loader(client).upload_experiment(
            evaluation_result=FakeEvaluationResult(results=[_result('item-1', 1.0)]),
            dataset_name='ds',
            run_name=run_name,
            flush=False,
        )

    assert upload('nightly')['experiment_id'] == upload('nightly')['experiment_id']
    assert upload('nightly')['experiment_id'] != upload('weekly')['experiment_id']


def test_items_are_trace_roots_even_inside_a_caller_span(make_loader):
    client = FakeLangfuseClient(FakeDataset([FakeDatasetItem('item-1')]))
    loader = make_loader(client)

    with client.tracer.start_as_current_span('caller'):
        loader.upload_experiment(
            evaluation_result=FakeEvaluationResult(results=[_result('item-1', 1.0)]),
            dataset_name='ds',
            run_name='r',
            flush=False,
        )
        # The caller's span is current again afterwards.
        assert otel_trace.get_current_span().name == 'caller'

    assert client.spans()['dataset-run:r:item-1'].parent is None


def test_without_a_project_id_the_experiment_gets_a_random_id(make_loader):
    client = FakeLangfuseClient(
        FakeDataset([FakeDatasetItem('item-1')]),
        project_id=RuntimeError('403'),
    )
    stats = make_loader(client).upload_experiment(
        evaluation_result=FakeEvaluationResult(results=[_result('item-1', 1.0)]),
        dataset_name='ds',
        run_name='r',
        flush=False,
    )

    assert len(stats['experiment_id']) == 16
    assert stats['run_url'] is None
    assert stats['runs_created'] == 1
    assert stats['scores_uploaded'] == 1
    assert len(stats['errors']) == 1 and 'random id' in stats['errors'][0]
    span = client.spans()['dataset-run:r:item-1']
    assert span.attributes['langfuse.experiment.id'] == stats['experiment_id']


def test_link_to_traces_also_scores_the_runtime_trace(make_loader):
    client = FakeLangfuseClient(
        FakeDataset([FakeDatasetItem('item-1'), FakeDatasetItem('item-2')])
    )
    stats = make_loader(client).upload_experiment(
        evaluation_result=FakeEvaluationResult(
            results=[
                _result('item-1', 0.5, trace_id='trace-1', observation_id='obs-1'),
                # No runtime trace: only the experiment item is scored.
                _result('item-2', 0.7),
            ]
        ),
        dataset_name='ds',
        run_name='experiment-v7',
        flush=False,
        link_to_traces=True,
    )

    assert stats['errors'] == []
    assert stats['runs_created'] == 2
    assert stats['scores_uploaded'] == 3

    spans = client.spans()
    first = spans['dataset-run:experiment-v7:item-1']
    second = spans['dataset-run:experiment-v7:item-2']
    targets = sorted(
        (s['value'], s['trace_id'], s.get('observation_id'))
        for s in client.created_scores
    )
    assert targets == sorted(
        [
            (
                0.5,
                format_trace_id(first.context.trace_id),
                format_span_id(first.context.span_id),
            ),
            (0.5, 'trace-1', 'obs-1'),
            (
                0.7,
                format_trace_id(second.context.trace_id),
                format_span_id(second.context.span_id),
            ),
        ]
    )
    assert first.attributes['langfuse.experiment.item.metadata.trace_id'] == 'trace-1'


def test_filtered_and_missing_scores_are_skipped_once(make_loader):
    client = FakeLangfuseClient(FakeDataset([FakeDatasetItem('item-1')]))
    result = _result('item-1', 0.5, trace_id='trace-1')
    result.score_results += [
        FakeMetricScore(name='m2', score=0.9),
        FakeMetricScore(name='m1', score=float('nan')),
    ]
    stats = make_loader(client).upload_experiment(
        evaluation_result=FakeEvaluationResult(results=[result]),
        dataset_name='ds',
        run_name='r',
        flush=False,
        link_to_traces=True,
        metric_names=['m1'],
    )

    assert stats['scores_uploaded'] == 2
    assert stats['scores_skipped'] == 2


def test_a_failed_item_write_is_recorded_and_the_rest_continue(make_loader):
    client = FakeLangfuseClient(
        FakeDataset([FakeDatasetItem('item-1'), FakeDatasetItem('item-2')])
    )
    real_start = client.start_as_current_observation

    @contextmanager
    def start(**kwargs):
        if kwargs['name'].endswith('item-1'):
            raise RuntimeError('exporter down')
        with real_start(**kwargs) as span:
            yield span

    client.start_as_current_observation = start
    stats = make_loader(client).upload_experiment(
        evaluation_result=FakeEvaluationResult(
            results=[_result('item-1', 0.5), _result('item-2', 0.7)]
        ),
        dataset_name='ds',
        run_name='r',
        flush=False,
    )

    assert stats['runs_created'] == 1
    assert [s['value'] for s in client.created_scores] == [0.7]
    assert len(stats['errors']) == 1 and 'item-1' in stats['errors'][0]


def test_an_upload_that_writes_no_item_returns_no_experiment(make_loader):
    empty = FakeLangfuseClient(FakeDataset([]))
    stats = make_loader(empty).upload_experiment(
        evaluation_result=FakeEvaluationResult(results=[]),
        dataset_name='ds',
        run_name='r',
        flush=False,
    )
    assert stats['runs_created'] == 0
    assert stats['experiment_id'] is None and stats['run_url'] is None

    failing = FakeLangfuseClient(FakeDataset([FakeDatasetItem('item-1')]))

    @contextmanager
    def start(**kwargs):
        raise RuntimeError('exporter down')
        yield

    failing.start_as_current_observation = start
    stats = make_loader(failing).upload_experiment(
        evaluation_result=FakeEvaluationResult(results=[_result('item-1', 1.0)]),
        dataset_name='ds',
        run_name='r',
        flush=False,
    )
    assert stats['runs_created'] == 0
    assert stats['experiment_id'] is None and stats['run_url'] is None
    assert len(stats['errors']) == 1


def test_scores_on_runtime_traces_write_no_experiment(make_loader):
    client = FakeLangfuseClient(FakeDataset([FakeDatasetItem('item-1')]))
    result = _result('item-1', 0.5, trace_id='trace-1', observation_id='obs-1')
    result.score_results = [
        FakeMetricScore(
            name='m1',
            score=0.5,
            metadata={
                'trace_id': 'metric-trace-1',
                'observation_id': 'metric-obs-1',
            },
        )
    ]

    stats = make_loader(client).upload_experiment(
        evaluation_result=FakeEvaluationResult(results=[result], run_id='run-abcdef01'),
        dataset_name='ds',
        run_name='experiment-v6',
        flush=False,
        tags=['production'],
        score_on_runtime_traces=True,
        # Takes precedence over link_to_traces.
        link_to_traces=True,
    )

    assert stats['runs_created'] == 0
    assert stats['scores_uploaded'] == 1
    assert stats['experiment_id'] is None
    assert client.started_observations == []

    assert len(client.created_scores) == 1
    score = client.created_scores[0]
    # Per-metric IDs win over test-case IDs.
    assert score['trace_id'] == 'metric-trace-1'
    assert score['observation_id'] == 'metric-obs-1'
    assert score['metadata']['dataset_name'] == 'ds'
    assert score['metadata']['run_name'] == 'experiment-v6'
