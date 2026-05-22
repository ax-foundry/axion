import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    """Stands in for both the v3 dataset_item.run() span and the v4
    `start_as_current_observation` span — same surface, just different
    method names. Tests use either depending on which path they exercise.
    """

    def __init__(self, trace_id: Optional[str] = None):
        self.update_trace_calls: List[Dict[str, Any]] = []
        self.set_trace_io_calls: List[Dict[str, Any]] = []
        self.score_trace_calls: List[Dict[str, Any]] = []
        self.trace_id = trace_id or 'fake-trace-id'

    def update_trace(self, **kwargs):
        self.update_trace_calls.append(kwargs)

    def set_trace_io(self, **kwargs):
        # v4 SDK equivalent of v3's update_trace(input=..., output=...).
        self.set_trace_io_calls.append(kwargs)

    def score_trace(self, **kwargs):
        self.score_trace_calls.append(kwargs)


class FakeDatasetItem:
    def __init__(self, item_id: str):
        self.id = item_id
        self.last_run_name = None
        self.last_run_metadata = None
        self.root_span = FakeRootSpan()

    @contextmanager
    def run(self, run_name: str, run_metadata: Optional[Dict[str, Any]] = None):
        self.last_run_name = run_name
        self.last_run_metadata = run_metadata
        yield self.root_span


class FakeDataset:
    def __init__(self, items: List[FakeDatasetItem]):
        self.items = items


class FakeDatasetRunItems:
    def __init__(self):
        self.created_run_items: List[Dict[str, Any]] = []

    def create(self, request=None, **kwargs):
        # Langfuse SDK v4 dropped the request-object signature and calls
        # `create(run_name=..., dataset_item_id=..., trace_id=..., metadata=...)`
        # directly. We accept either form so old (request=req) and new
        # (kwargs) callsites both work; tests don't have to care which one
        # axion is using internally.
        if request is not None:
            payload = {
                'run_name': getattr(request, 'run_name', None)
                or getattr(request, 'runName', None),
                'dataset_item_id': getattr(request, 'dataset_item_id', None)
                or getattr(request, 'datasetItemId', None),
                'trace_id': getattr(request, 'trace_id', None)
                or getattr(request, 'traceId', None),
                'observation_id': getattr(request, 'observation_id', None)
                or getattr(request, 'observationId', None),
                'metadata': getattr(request, 'metadata', None),
            }
        else:
            payload = {
                'run_name': kwargs.get('run_name'),
                'dataset_item_id': kwargs.get('dataset_item_id'),
                'trace_id': kwargs.get('trace_id'),
                'observation_id': kwargs.get('observation_id'),
                'metadata': kwargs.get('metadata'),
            }
        self.created_run_items.append(payload)


class FakeApi:
    def __init__(self):
        self.dataset_run_items = FakeDatasetRunItems()


class FakeLangfuseClient:
    def __init__(self, dataset: FakeDataset):
        self._dataset = dataset
        self.created_items: List[Dict[str, Any]] = []
        self.created_datasets: List[str] = []
        self.created_scores: List[Dict[str, Any]] = []
        self.started_observations: List[Dict[str, Any]] = []
        self.flush_calls = 0
        self.api = FakeApi()
        # Auto-increments so every span gets a distinct trace_id, matching
        # real-world behavior. Tests can inspect `started_observations` to
        # see which item each span belonged to.
        self._next_trace_seq = 0

    def create_dataset(self, name: str):
        self.created_datasets.append(name)

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        # v4 entry point axion now uses for the default (no link_to_traces,
        # no score_on_runtime_traces) experiment-upload path. Returns a
        # FakeRootSpan whose `trace_id` is unique per call so callers can
        # attach scores to the right trace afterwards.
        self._next_trace_seq += 1
        trace_id = f'fake-trace-{self._next_trace_seq}'
        span = FakeRootSpan(trace_id=trace_id)
        self.started_observations.append({**kwargs, 'trace_id': trace_id, 'span': span})
        yield span

    def create_dataset_item(
        self,
        dataset_name: str,
        id: str,
        input=None,
        expected_output=None,
        metadata=None,
    ):
        self.created_items.append(
            {
                'dataset_name': dataset_name,
                'id': id,
                'input': input,
                'expected_output': expected_output,
                'metadata': metadata,
            }
        )

    def flush(self):
        self.flush_calls += 1

    def get_dataset(self, name: str):
        return self._dataset

    def create_score(self, **kwargs):
        self.created_scores.append(kwargs)


class FakeCreateDatasetRunItemRequest:
    """Fake request object mimicking langfuse.api.types.CreateDatasetRunItemRequest"""

    def __init__(
        self,
        runName=None,
        datasetItemId=None,
        traceId=None,
        observationId=None,
        metadata=None,
        **kwargs,
    ):
        self.runName = runName
        self.datasetItemId = datasetItemId
        self.traceId = traceId
        self.observationId = observationId
        self.metadata = metadata


def _install_fake_langfuse_module(fake_client: FakeLangfuseClient):
    """
    LangfuseTraceLoader imports `Langfuse` from the `langfuse` package at runtime.
    Provide a minimal stub so tests do not require the real dependency.
    """
    fake_module = types.ModuleType('langfuse')

    class Langfuse:  # noqa: N801 (matches external class name)
        def __init__(self, **kwargs):
            # Loader stores this instance as `self.client`
            pass

    fake_module.Langfuse = Langfuse
    sys.modules['langfuse'] = fake_module

    # Create fake langfuse.api module with CreateDatasetRunItemRequest
    fake_api_module = types.ModuleType('langfuse.api')
    fake_api_module.CreateDatasetRunItemRequest = FakeCreateDatasetRunItemRequest
    sys.modules['langfuse.api'] = fake_api_module

    # Monkeypatching approach: we will bypass loader.__init__ client creation and
    # inject our fake client directly in the test.
    return fake_module


def test_upload_experiment_sets_input_output_and_linking_metadata(monkeypatch):
    # Dataset run will iterate over dataset.items; keep it aligned with evaluation results.
    dataset_items = [FakeDatasetItem('item-1'), FakeDatasetItem('item-2')]
    fake_dataset = FakeDataset(dataset_items)
    fake_client = FakeLangfuseClient(dataset=fake_dataset)

    _install_fake_langfuse_module(fake_client)

    # Instantiate loader without requiring real Langfuse initialization.
    loader = LangfuseTraceLoader.__new__(LangfuseTraceLoader)
    loader.max_retries = 0
    loader.base_delay = 0.0
    loader.max_delay = 0.0
    loader.request_pacing = 0.0
    loader.default_tags = []
    loader.client = fake_client
    loader._client_initialized = True

    eval_result = FakeEvaluationResult(
        results=[
            FakeTestResult(
                test_case=FakeDatasetItemInput(
                    id='item-1',
                    query='q1',
                    actual_output='a1',
                    expected_output='e1',
                    trace_id='trace-1',
                    observation_id='obs-1',
                ),
                score_results=[FakeMetricScore(name='m1', score=0.5)],
            ),
            FakeTestResult(
                test_case=FakeDatasetItemInput(
                    id='item-2',
                    query='q2',
                    actual_output='a2',
                    expected_output='e2',
                ),
                score_results=[FakeMetricScore(name='m1', score=0.7)],
            ),
        ],
        run_id='run-abcdef01',
    )

    stats = loader.upload_experiment(
        evaluation_result=eval_result,
        dataset_name='ds',
        run_name='experiment-v6',
        flush=False,
        tags=['production'],
    )

    assert stats['dataset_name'] == 'ds'
    assert stats['run_name'] == 'experiment-v6'
    assert stats['runs_created'] == 2

    # Default mode now opens one span per item via start_as_current_observation
    # and stamps input/output via set_trace_io (v4 SDK shape).
    assert len(fake_client.started_observations) == 2
    for obs in fake_client.started_observations:
        span = obs['span']
        assert span.set_trace_io_calls, 'expected set_trace_io to be called'
        call = span.set_trace_io_calls[0]
        assert 'input' in call
        assert 'output' in call

    # Each item should have produced a low-level dataset_run_items.create call
    # linking the freshly-created trace to the run.
    run_items = fake_client.api.dataset_run_items.created_run_items
    assert len(run_items) == 2
    run_items_by_id = {ri['dataset_item_id']: ri for ri in run_items}
    assert run_items_by_id['item-1']['run_name'] == 'experiment-v6'
    # Per-item metadata should include the original linking identifiers when present
    assert run_items_by_id['item-1']['metadata']['trace_id'] == 'trace-1'
    assert run_items_by_id['item-1']['metadata']['observation_id'] == 'obs-1'
    assert 'trace_id' not in run_items_by_id['item-2']['metadata']


def test_upload_experiment_scores_on_runtime_traces_no_dataset_runs(monkeypatch):
    dataset_items = [FakeDatasetItem('item-1')]
    fake_dataset = FakeDataset(dataset_items)
    fake_client = FakeLangfuseClient(dataset=fake_dataset)

    _install_fake_langfuse_module(fake_client)

    loader = LangfuseTraceLoader.__new__(LangfuseTraceLoader)
    loader.max_retries = 0
    loader.base_delay = 0.0
    loader.max_delay = 0.0
    loader.request_pacing = 0.0
    loader.default_tags = []
    loader.client = fake_client
    loader._client_initialized = True

    eval_result = FakeEvaluationResult(
        results=[
            FakeTestResult(
                test_case=FakeDatasetItemInput(
                    id='item-1',
                    query='q1',
                    actual_output='a1',
                    expected_output='e1',
                    trace_id='trace-1',
                    observation_id='obs-1',
                ),
                score_results=[
                    FakeMetricScore(
                        name='m1',
                        score=0.5,
                        metadata={
                            'trace_id': 'metric-trace-1',
                            'observation_id': 'metric-obs-1',
                        },
                    )
                ],
            ),
        ],
        run_id='run-abcdef01',
    )

    stats = loader.upload_experiment(
        evaluation_result=eval_result,
        dataset_name='ds',
        run_name='experiment-v6',
        flush=False,
        tags=['production'],
        score_on_runtime_traces=True,
    )

    assert stats['runs_created'] == 0
    assert stats['scores_uploaded'] == 1
    assert dataset_items[0].last_run_name is None, 'should not create dataset runs'

    assert len(fake_client.created_scores) == 1
    score = fake_client.created_scores[0]
    # Per-metric IDs should win over test-case IDs
    assert score['trace_id'] == 'metric-trace-1'
    assert score['observation_id'] == 'metric-obs-1'
    assert score['metadata']['dataset_name'] == 'ds'
    assert score['metadata']['run_name'] == 'experiment-v6'


def test_upload_experiment_link_to_traces_uses_low_level_api(monkeypatch):
    """Test that link_to_traces=True uses client.api.dataset_run_items.create() to link runs to existing traces."""
    dataset_items = [FakeDatasetItem('item-1'), FakeDatasetItem('item-2')]
    fake_dataset = FakeDataset(dataset_items)
    fake_client = FakeLangfuseClient(dataset=fake_dataset)

    _install_fake_langfuse_module(fake_client)

    loader = LangfuseTraceLoader.__new__(LangfuseTraceLoader)
    loader.max_retries = 0
    loader.base_delay = 0.0
    loader.max_delay = 0.0
    loader.request_pacing = 0.0
    loader.default_tags = []
    loader.client = fake_client
    loader._client_initialized = True

    eval_result = FakeEvaluationResult(
        results=[
            FakeTestResult(
                test_case=FakeDatasetItemInput(
                    id='item-1',
                    query='q1',
                    actual_output='a1',
                    expected_output='e1',
                    trace_id='trace-1',
                    observation_id='obs-1',
                ),
                score_results=[FakeMetricScore(name='m1', score=0.5)],
            ),
            # item-2 has no trace_id, should fall back to creating new trace
            FakeTestResult(
                test_case=FakeDatasetItemInput(
                    id='item-2',
                    query='q2',
                    actual_output='a2',
                    expected_output='e2',
                ),
                score_results=[FakeMetricScore(name='m2', score=0.7)],
            ),
        ],
        run_id='run-abcdef01',
    )

    stats = loader.upload_experiment(
        evaluation_result=eval_result,
        dataset_name='ds',
        run_name='experiment-v7',
        flush=False,
        tags=['production'],
        link_to_traces=True,
    )

    assert stats['dataset_name'] == 'ds'
    assert stats['run_name'] == 'experiment-v7'
    assert stats['runs_created'] == 2
    assert stats['scores_uploaded'] == 2

    # Both paths now create dataset_run_items via the low-level API. The
    # difference is what trace_id each call carries:
    #   - item-1 (link_to_traces hit): the test_case.trace_id ("trace-1")
    #   - item-2 (no trace_id, fallback): a freshly-minted trace from
    #     start_as_current_observation
    run_items = fake_client.api.dataset_run_items.created_run_items
    assert len(run_items) == 2
    by_id = {ri['dataset_item_id']: ri for ri in run_items}
    assert by_id['item-1']['run_name'] == 'experiment-v7'
    assert by_id['item-1']['trace_id'] == 'trace-1'
    assert by_id['item-1']['observation_id'] == 'obs-1'

    # item-2 fell back to creating a new trace; its run-item must point at
    # that new trace, not be None.
    assert by_id['item-2']['trace_id'] and by_id['item-2']['trace_id'] != 'trace-1'

    # Scores: item-1's m1 attaches to the existing trace; item-2's m2 attaches
    # to the freshly-created trace. Both go through create_score now.
    score_by_name = {s['name']: s for s in fake_client.created_scores}
    assert score_by_name['m1']['trace_id'] == 'trace-1'
    assert score_by_name['m1']['observation_id'] == 'obs-1'
    assert score_by_name['m1']['value'] == 0.5
    assert score_by_name['m2']['trace_id'] == by_id['item-2']['trace_id']
    assert score_by_name['m2']['value'] == 0.7

    # item-1 hit the link_to_traces branch — must NOT have opened a span.
    started_for_item_1 = [
        o for o in fake_client.started_observations if 'item-1' in (o.get('name') or '')
    ]
    assert started_for_item_1 == [], (
        'link_to_traces=True path should not open a new observation for item-1'
    )

    # item-2 fell back — must have opened exactly one observation.
    started_for_item_2 = [
        o for o in fake_client.started_observations if 'item-2' in (o.get('name') or '')
    ]
    assert len(started_for_item_2) == 1


def test_upload_experiment_link_to_traces_false_creates_new_traces(monkeypatch):
    """`link_to_traces=False` (default): create a fresh trace per item via the
    v4 `start_as_current_observation` span and link it to the run via the
    low-level API. Even when the test_case has a trace_id, the default path
    ignores it and mints a new one — that's the whole point of `False`.
    """
    dataset_items = [FakeDatasetItem('item-1')]
    fake_dataset = FakeDataset(dataset_items)
    fake_client = FakeLangfuseClient(dataset=fake_dataset)

    _install_fake_langfuse_module(fake_client)

    loader = LangfuseTraceLoader.__new__(LangfuseTraceLoader)
    loader.max_retries = 0
    loader.base_delay = 0.0
    loader.max_delay = 0.0
    loader.request_pacing = 0.0
    loader.default_tags = []
    loader.client = fake_client
    loader._client_initialized = True

    eval_result = FakeEvaluationResult(
        results=[
            FakeTestResult(
                test_case=FakeDatasetItemInput(
                    id='item-1',
                    query='q1',
                    actual_output='a1',
                    expected_output='e1',
                    trace_id='trace-1',  # Even with trace_id, should NOT use low-level API
                    observation_id='obs-1',
                ),
                score_results=[FakeMetricScore(name='m1', score=0.5)],
            ),
        ],
        run_id='run-abcdef01',
    )

    stats = loader.upload_experiment(
        evaluation_result=eval_result,
        dataset_name='ds',
        run_name='experiment-v8',
        flush=False,
        link_to_traces=False,  # Explicitly false (default behavior)
    )

    assert stats['runs_created'] == 1
    assert stats['scores_uploaded'] == 1

    # New trace was minted via start_as_current_observation, *not* reused from
    # the test_case's existing trace_id.
    assert len(fake_client.started_observations) == 1
    new_trace_id = fake_client.started_observations[0]['trace_id']
    assert new_trace_id != 'trace-1'

    # Low-level API was used to link the new trace to the dataset run.
    run_items = fake_client.api.dataset_run_items.created_run_items
    assert len(run_items) == 1
    assert run_items[0]['dataset_item_id'] == 'item-1'
    assert run_items[0]['run_name'] == 'experiment-v8'
    assert run_items[0]['trace_id'] == new_trace_id

    # Score was attached to the new trace, not the old one.
    assert len(fake_client.created_scores) == 1
    assert fake_client.created_scores[0]['trace_id'] == new_trace_id


def test_upload_experiment_score_on_runtime_traces_takes_precedence(monkeypatch):
    """Test that score_on_runtime_traces takes precedence over link_to_traces when both are True."""
    dataset_items = [FakeDatasetItem('item-1')]
    fake_dataset = FakeDataset(dataset_items)
    fake_client = FakeLangfuseClient(dataset=fake_dataset)

    _install_fake_langfuse_module(fake_client)

    loader = LangfuseTraceLoader.__new__(LangfuseTraceLoader)
    loader.max_retries = 0
    loader.base_delay = 0.0
    loader.max_delay = 0.0
    loader.request_pacing = 0.0
    loader.default_tags = []
    loader.client = fake_client
    loader._client_initialized = True

    eval_result = FakeEvaluationResult(
        results=[
            FakeTestResult(
                test_case=FakeDatasetItemInput(
                    id='item-1',
                    query='q1',
                    actual_output='a1',
                    expected_output='e1',
                    trace_id='trace-1',
                    observation_id='obs-1',
                ),
                score_results=[FakeMetricScore(name='m1', score=0.5)],
            ),
        ],
        run_id='run-abcdef01',
    )

    stats = loader.upload_experiment(
        evaluation_result=eval_result,
        dataset_name='ds',
        run_name='experiment-v9',
        flush=False,
        score_on_runtime_traces=True,  # Takes precedence
        link_to_traces=True,  # Should be ignored
    )

    # score_on_runtime_traces behavior: no runs created, scores on existing traces
    assert stats['runs_created'] == 0, (
        'score_on_runtime_traces should prevent run creation'
    )
    assert stats['scores_uploaded'] == 1

    # Should NOT have used low-level API (link_to_traces is ignored)
    assert len(fake_client.api.dataset_run_items.created_run_items) == 0

    # Should NOT have used dataset_item.run() context manager
    assert dataset_items[0].last_run_name is None

    # Scores should be created directly via create_score (score_on_runtime_traces mode)
    assert len(fake_client.created_scores) == 1
    score = fake_client.created_scores[0]
    assert score['trace_id'] == 'trace-1'
    assert 'metadata' in score  # score_on_runtime_traces adds metadata
