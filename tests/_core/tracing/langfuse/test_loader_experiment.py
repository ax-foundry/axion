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
    def __init__(self):
        self.update_trace_calls: List[Dict[str, Any]] = []
        self.score_trace_calls: List[Dict[str, Any]] = []

    def update_trace(self, **kwargs):
        self.update_trace_calls.append(kwargs)

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

    def create(self, request):
        # Extract fields from the request object (CreateDatasetRunItemRequest)
        self.created_run_items.append(
            {
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
        )


class FakeApi:
    def __init__(self):
        self.dataset_run_items = FakeDatasetRunItems()


class FakeLangfuseClient:
    def __init__(self, dataset: FakeDataset):
        self._dataset = dataset
        self.created_items: List[Dict[str, Any]] = []
        self.created_datasets: List[str] = []
        self.created_scores: List[Dict[str, Any]] = []
        self.flush_calls = 0
        self.api = FakeApi()

    def create_dataset(self, name: str):
        self.created_datasets.append(name)

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

    # Verify each dataset item run trace received input/output
    for ds_item in dataset_items:
        assert ds_item.last_run_name == 'experiment-v6'
        assert ds_item.root_span.update_trace_calls, (
            'expected update_trace to be called'
        )
        call = ds_item.root_span.update_trace_calls[0]
        assert 'input' in call
        assert 'output' in call

    # Verify per-item run_metadata includes linking identifiers when present
    assert dataset_items[0].last_run_metadata['trace_id'] == 'trace-1'
    assert dataset_items[0].last_run_metadata['observation_id'] == 'obs-1'
    assert 'trace_id' not in (dataset_items[1].last_run_metadata or {})


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

    # Verify item-1 used low-level API (link_to_traces)
    run_items = fake_client.api.dataset_run_items.created_run_items
    assert len(run_items) == 1, 'Only item-1 has trace_id, should use low-level API'

    run_item = run_items[0]
    assert run_item['run_name'] == 'experiment-v7'
    assert run_item['dataset_item_id'] == 'item-1'
    assert run_item['trace_id'] == 'trace-1'
    assert run_item['observation_id'] == 'obs-1'

    # Verify score was created directly on existing trace (not via root_span.score_trace)
    assert len(fake_client.created_scores) == 1
    score = fake_client.created_scores[0]
    assert score['trace_id'] == 'trace-1'
    assert score['observation_id'] == 'obs-1'
    assert score['name'] == 'm1'
    assert score['value'] == 0.5

    # item-1's dataset_item.run() context should NOT have been called
    assert dataset_items[0].last_run_name is None, (
        'link_to_traces should skip dataset_item.run()'
    )

    # Verify item-2 fell back to normal behavior (creating new trace via dataset_item.run)
    assert dataset_items[1].last_run_name == 'experiment-v7', (
        'item without trace_id should use fallback'
    )
    assert dataset_items[1].root_span.score_trace_calls, (
        'fallback should use root_span.score_trace'
    )


def test_upload_experiment_link_to_traces_false_uses_context_manager(monkeypatch):
    """Test that link_to_traces=False (default) uses dataset_item.run() context manager."""
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

    # Should NOT have used low-level API
    assert len(fake_client.api.dataset_run_items.created_run_items) == 0

    # Should have used dataset_item.run() context manager
    assert dataset_items[0].last_run_name == 'experiment-v8'
    assert dataset_items[0].root_span.score_trace_calls, (
        'should use root_span.score_trace'
    )

    # Scores should NOT be created directly via create_score when not using link_to_traces
    assert len(fake_client.created_scores) == 0


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
