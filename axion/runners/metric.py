import asyncio
import hashlib
import operator
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from axion._core.asyncio import SemaphoreExecutor, gather_with_progress
from axion._core.cache.manager import CacheManager
from axion._core.logging import get_logger
from axion._core.tracing import Tracer, init_tracer
from axion._core.tracing.handlers import BaseTraceHandler
from axion._core.types import TraceGranularity
from axion._core.utils import Timer
from axion.dataset import Dataset, DatasetItem, format_input
from axion.metrics.cache import AnalysisCache
from axion.metrics.signal_extractor import SignalExtractor
from axion.runners.cost import extract_cost
from axion.runners.mixin import RunnerMixin
from axion.runners.summary import BaseSummary, MetricSummary
from axion.runners.utils import input_to_dataset
from axion.schema import ErrorConfig, MetricScore, TestResult
from axion.validation import EvaluationValidation

logger = get_logger(__name__)


class BaseMetricRunner(ABC):
    """
    Abstract base class for running a single evaluation metric.
    """

    _name: str
    __name__ = 'metric_runner'
    DEFAULT_THRESHOLD = 0.5

    def __init__(
        self,
        metric: Any,
        metric_name: Optional[str] = None,
        threshold: Optional[float] = None,
        tracer: Optional[BaseTraceHandler] = None,
        **kwargs,
    ):
        self.metric = metric
        self._metric_name = metric_name or self.metric.__class__.__name__
        self.threshold = (
            threshold
            if threshold is not None
            else getattr(self.metric, 'threshold', self.DEFAULT_THRESHOLD)
        )
        self.tracer = init_tracer(
            metadata_type='llm', tool_metadata=self.get_tool_metadata(), tracer=tracer
        )

    @abstractmethod
    async def execute(
        self, evaluation_input: DatasetItem, cache: Optional[AnalysisCache] = None
    ) -> MetricScore:
        """Executes the metric for a single evaluation input."""
        pass

    def get_tool_metadata(self):
        """Builds tool metadata describing this metric runner."""
        from axion._core.metadata.schema import ToolMetadata

        return ToolMetadata(
            name=self.source,
            description=f'Runner for {self.metric_name} metric',
            owner='AI Engineering',
            version='1.0.0',
        )

    def _create_error_score(self, input_id: str, error: Exception) -> MetricScore:
        """Creates a fallback MetricScore in case of an execution error."""
        return MetricScore(
            id=input_id,
            name=self.metric_name,
            score=np.nan,
            explanation=f'Error executing metric: {str(error)}',
            source=self.source,
        )

    @staticmethod
    def _prepare_retrieved_content(content: Union[str, List, None]) -> List[str]:
        """Ensures that retrieved content is always a list of strings."""
        return [content] if isinstance(content, str) else content or []

    def _has_passed(self, score: float) -> Optional[bool]:
        """Determines if a score meets the defined threshold."""
        if score is None or np.isnan(score) or self.threshold is None:
            return None
        comparator = (
            operator.lt
            if getattr(self.metric, 'inverse_scoring_metric', False)
            else operator.ge
        )
        return comparator(score, self.threshold)

    @property
    def metric_name(self) -> str:
        """Gets the name of the metric."""
        return self._metric_name

    @property
    def source(self) -> str:
        """Gets the source name of the metric runner."""
        return self._name


class MetricRunnerFactory:
    """
    Factory to create metric runner instances based on the metric type.
    """

    _registry: ClassVar[Dict[str, Callable[..., BaseMetricRunner]]] = {}

    @classmethod
    def register(cls, metric_type: str) -> Callable:
        """Decorator to register a concrete metric runner class."""

        def decorator(runner_class: Callable[..., BaseMetricRunner]) -> Callable:
            cls._registry[metric_type.lower()] = runner_class
            return runner_class

        return decorator

    @staticmethod
    def _get_metric_type(metric: Any) -> str:
        """Infers the metric source/type based on its module path."""
        module = metric.__class__.__module__.lower()
        for source in MetricRunnerFactory._registry.keys():
            if source in module:
                return source
        return 'axion'  # Default fallback

    def create_executor(
        self,
        metric: Any,
        **kwargs,
    ) -> BaseMetricRunner:
        """
        Creates a metric executor instance based on the metric type.
        """
        from axion.eval_tree.metric import MetricNode

        metric_instance = metric.metric if isinstance(metric, MetricNode) else metric
        metric_name = (
            metric.name
            if isinstance(metric, MetricNode)
            else getattr(metric, 'name', metric.__class__.__name__)
        )

        metric_type = self._get_metric_type(metric_instance)
        runner_class = self._registry.get(metric_type)

        if not runner_class:
            raise ValueError(
                f"No runner registered for metric type '{metric_type}' for metric '{metric_name}'"
            )

        return runner_class(
            metric=metric_instance,
            metric_name=metric_name,
            **kwargs,
        )


class BaseBatchRunner(BaseMetricRunner):
    """Abstract base class for runners that support native batch processing."""

    @abstractmethod
    async def execute_batch(
        self,
        dataset: Dataset,
        runners: List['BaseMetricRunner'],
    ) -> Dict[str, List[MetricScore]]:
        """Executes a batch of metrics against a full dataset."""
        pass


@dataclass
class MetricRunner(RunnerMixin):
    """Orchestrates the evaluation of multiple metrics against a dataset."""

    metrics: List[Any]
    name: str = 'MetricRunner'
    description: str = 'Orchestrates evaluation metrics'
    max_concurrent: int = 5
    thresholds: Optional[Dict[str, float]] = None
    summary_generator: Optional[BaseSummary] = field(
        default_factory=MetricSummary, repr=False
    )
    cache_manager: Optional[CacheManager] = None
    error_config: ErrorConfig = field(default_factory=ErrorConfig)
    tracer: Optional[BaseTraceHandler] = field(default=None)
    dataset_name: Optional[str] = 'Metric Runner Dataset'

    enable_internal_caching: bool = field(
        default=True,
        repr=True,
        metadata={
            'description': 'Enables a per-item cache for metrics that share expensive internal computations.'
        },
    )
    trace_granularity: TraceGranularity = field(
        default=TraceGranularity.SEPARATE,
        repr=True,
        metadata={
            'description': 'Controls trace granularity: single (all under one trace) or separate (trace per metric execution).'
        },
    )
    flush_per_metric: bool = field(
        default=False,
        repr=True,
        metadata={
            'description': "When trace_granularity='separate', flush each metric trace immediately (slower) vs batch flush at end (faster)."
        },
    )

    _elapsed_time: Optional[float] = field(default=None, init=False, repr=False)
    _summary: Optional[Dict] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.tracer = init_tracer(
            metadata_type='base',
            tool_metadata=self.get_tool_metadata(),
            tracer=self.tracer,
        )
        self.executors: List[BaseMetricRunner] = []
        self.semaphore = SemaphoreExecutor(self.max_concurrent)
        factory = MetricRunnerFactory()

        for metric in self.metrics:
            from axion.eval_tree.metric import MetricNode

            metric_name = (
                metric.name
                if isinstance(metric, MetricNode)
                else metric.__class__.__name__
            )

            threshold = self.thresholds.get(metric_name) if self.thresholds else None
            try:
                executor = factory.create_executor(
                    metric=metric,
                    threshold=threshold,
                    tracer=self.tracer,
                )
                self.executors.append(executor)
            except ValueError as e:
                logger.warning(e)

    async def _safe_execute(
        self, coro: asyncio.Task, ignore_errors: bool
    ) -> Union[MetricScore, Exception]:
        """Wraps a coroutine to conditionally catch exceptions."""
        try:
            return await coro
        except Exception as e:
            if not ignore_errors:
                raise
            return e

    @staticmethod
    def _get_cache_key(executor: 'BaseMetricRunner', input_data: DatasetItem) -> str:
        """Generates a unique cache key for a metric and data item pair."""
        key_string = f'{executor.metric.__class__.__name__}:{input_data.id}'
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def _process_single_metric(
        self,
        executor: 'BaseMetricRunner',
        input_data: DatasetItem,
        cache: Optional[AnalysisCache] = None,
    ) -> Optional[MetricScore]:
        """Handles caching, parameter checks, and execution for one metric."""

        if self.error_config.skip_on_missing_params:
            required: Any = getattr(executor, 'required_params', set())
            if not required.issubset(input_data.keys()):
                logger.warning(
                    f"Skipping {executor.metric_name} for item '{input_data.id}' due to missing params."
                )
                return None

        cache_key = self._get_cache_key(executor, input_data)
        if self.cache_manager:
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(
                    f"Cache hit for {executor.metric_name} on item '{input_data.id}'"
                )
                return cached_result

        # ✨ Pass the injected cache down to the executor.
        result = await executor.execute(input_data, cache=cache)

        if self.cache_manager and isinstance(result, MetricScore):
            self.cache_manager.set(cache_key, result)
        return result

    @staticmethod
    def _check_valid_input(evaluation_inputs: Any):
        """Validates the type of the input data."""
        valid_types = (Dataset, list, pd.DataFrame)
        if not isinstance(evaluation_inputs, valid_types):
            raise TypeError(
                f"Invalid input type '{type(evaluation_inputs).__name__}'. "
                f'Expected one of: Dataset, List[DatasetItem], or pandas.DataFrame.'
            )

    async def execute_batch(
        self,
        evaluation_inputs: Union[Dataset, List[DatasetItem], pd.DataFrame],
        *,
        show_progress: bool = True,
    ) -> List[TestResult]:
        """Executes all configured metrics against the provided dataset.

        Trace granularity behavior:
        - SINGLE_TRACE: All metrics run under one parent trace (default)
        - SEPARATE: Each metric execution gets its own independent trace
        """
        self._check_valid_input(evaluation_inputs)
        dataset = input_to_dataset(evaluation_inputs, self.dataset_name)

        if not self.executors or not dataset:
            logger.warning(
                'No metric executors or data provided. Returning empty results.'
            )
            return []

        # Dispatch based on trace granularity
        if self.trace_granularity == TraceGranularity.SEPARATE:
            return await self._execute_batch_separate(dataset, show_progress)
        else:
            # Default to single trace (SINGLE_TRACE or any other value)
            return await self._execute_batch_single_trace(dataset, show_progress)

    async def _execute_batch_single_trace(
        self,
        dataset: Dataset,
        show_progress: bool,
    ) -> List[TestResult]:
        """Execute batch with all metrics under a single parent trace."""
        async with self.tracer.async_span('MetricRunner_Batch') as span:
            span.set_input(
                {
                    'input_count': len(dataset),
                    'metrics': [e.metric_name for e in self.executors],
                }
            )

            final_results = await self._execute_metrics_for_dataset(
                dataset, show_progress
            )

            # Capture output summary
            total_scores = sum(len(r.score_results) for r in final_results)
            passed_count = sum(
                1 for r in final_results for score in r.score_results if score.passed
            )
            span.set_output(
                {
                    'results_count': len(final_results),
                    'total_scores': total_scores,
                    'passed_count': passed_count,
                    'success_rate': passed_count / total_scores if total_scores else 0,
                }
            )

            return final_results

    async def _execute_single_metric_task(
        self,
        executor: 'BaseMetricRunner',
        item: DatasetItem,
        cache_to_pass: Optional[AnalysisCache],
        flush_per_metric: bool,
    ) -> tuple[str, Union[MetricScore, Exception]]:
        """Execute a single metric for one item with its own trace.

        Args:
            executor: The metric runner to execute
            item: The dataset item to evaluate
            cache_to_pass: Optional shared cache for metrics that support it
            flush_per_metric: Whether to flush trace after each execution

        Returns:
            Tuple of (item_id, MetricScore or Exception)
        """
        # Create a fresh tracer for this metric execution (new trace_id)
        # Pass auto_flush in constructor to ensure it's set correctly
        metric_tracer = Tracer().create(
            metadata_type='llm',
            tool_metadata=executor.get_tool_metadata(),
            auto_flush=flush_per_metric,
        )

        # Use async_span which sets the tracer in context via set_current_tracer().
        # The @trace decorator and runner execute() methods will pick up
        # the context tracer, avoiding race conditions with shared executors.
        async with metric_tracer.async_span(
            f'metric_{executor.metric_name}_item_{item.id}'
        ) as span:
            span.set_input(
                {
                    'item_id': item.id,
                    'metric_name': executor.metric_name,
                }
            )

            result = await self._safe_execute(
                self.semaphore.run(
                    self._process_single_metric,
                    executor,
                    item,
                    cache=cache_to_pass,
                ),
                ignore_errors=self.error_config.ignore_errors,
            )

            if isinstance(result, MetricScore):
                # Attach trace identifiers for downstream publishing.
                # This enables publishing scores directly onto the metric's own
                # runtime trace/span (e.g., publish_as_experiment(score_on_runtime_traces=True)).
                try:
                    result.metadata = result.metadata or {}
                    result.metadata.update(
                        {
                            'trace_id': getattr(metric_tracer, 'trace_id', None),
                            'observation_id': getattr(span, 'span_id', None),
                            'metric_name': executor.metric_name,
                        }
                    )
                except Exception:
                    pass
                span.set_output(
                    {
                        'score': float(result.score)
                        if result.score is not None
                        else None,
                        'passed': result.passed,
                    }
                )
            elif isinstance(result, Exception):
                logger.error(f'Metric execution failed: {result}', exc_info=False)
                span.set_attribute('error', str(result))

            # Flush the tracer after each metric execution if requested
            if flush_per_metric and hasattr(metric_tracer, 'flush'):
                metric_tracer.flush()

            return (item.id, result)

    async def _execute_batch_separate(
        self,
        dataset: Dataset,
        show_progress: bool,
    ) -> List[TestResult]:
        """Execute batch with a separate trace per metric execution.

        Uses gather_with_progress for concurrent execution and tqdm progress bars.
        """
        single_executors = [
            exc for exc in self.executors if not isinstance(exc, BaseBatchRunner)
        ]

        # For performance, avoid forcing a network flush for every metric trace.
        # We disable auto_flush on the runner-level tracer and flush once at the end.
        if not self.flush_per_metric:
            try:
                self.tracer.auto_flush = False  # type: ignore[attr-defined]
            except Exception:
                pass

        # Pre-create per-item caches before building task list.
        # This preserves cache sharing for metrics that use shares_internal_cache.
        item_caches: dict[str, Optional[AnalysisCache]] = {}
        for item in dataset:
            item_caches[item.id] = (
                AnalysisCache() if self.enable_internal_caching else None
            )

        # Build task list for all (item, executor) pairs
        tasks = []
        for item in dataset:
            item_specific_cache = item_caches[item.id]
            for executor in single_executors:
                cache_to_pass = None
                if self.enable_internal_caching and getattr(
                    executor.metric, 'shares_internal_cache', False
                ):
                    cache_to_pass = item_specific_cache

                tasks.append(
                    self._execute_single_metric_task(
                        executor=executor,
                        item=item,
                        cache_to_pass=cache_to_pass,
                        flush_per_metric=self.flush_per_metric,
                    )
                )

        # Run all tasks concurrently with progress bar
        all_results = await gather_with_progress(
            tasks, '📊 Evaluating metrics', show_progress
        )

        # Aggregate results into results_map
        results_map = defaultdict[Any, list](list)
        for item_id, result in all_results:
            if isinstance(result, MetricScore):
                results_map[item_id].append(result)

        final_results = [
            TestResult(test_case=item, score_results=results_map.get(item.id, []))
            for item in dataset
        ]

        self._finalize_results(final_results)

        # Batch flush once at the end (fast path) so traces appear in the UI without
        # paying a per-metric flush penalty.
        if not self.flush_per_metric:
            if hasattr(self.tracer, 'async_flush'):
                try:
                    await self.tracer.async_flush()
                except Exception:
                    pass
            elif hasattr(self.tracer, 'flush'):
                try:
                    self.tracer.flush()
                except Exception:
                    pass

        return final_results

    async def _execute_metrics_for_dataset(
        self,
        dataset: Dataset,
        show_progress: bool,
    ) -> List[TestResult]:
        """Core implementation for executing metrics against a dataset."""
        with Timer() as timer:
            logger.info(
                f'Executing {len(self.executors)} metrics against {len(dataset)} data points...'
            )

            final_results = await self._execute_metrics_for_items(
                dataset, show_progress
            )

        self._elapsed_time = timer.elapsed_time
        self._finalize_results(final_results)
        return final_results

    async def _execute_metrics_for_items(
        self,
        items: Union[Dataset, List[DatasetItem]],
        show_progress: bool,
    ) -> List[TestResult]:
        """Execute all metrics for a list of items."""
        single_executors = [
            exc for exc in self.executors if not isinstance(exc, BaseBatchRunner)
        ]

        results_map = defaultdict(list)
        tasks = []

        for item in items:
            item_specific_cache = (
                AnalysisCache() if self.enable_internal_caching else None
            )

            for executor in single_executors:
                cache_to_pass = None
                if self.enable_internal_caching and getattr(
                    executor.metric, 'shares_internal_cache', False
                ):
                    cache_to_pass = item_specific_cache

                tasks.append(
                    self._safe_execute(
                        self.semaphore.run(
                            self._process_single_metric,
                            executor,
                            item,
                            cache=cache_to_pass,
                        ),
                        ignore_errors=self.error_config.ignore_errors,
                    )
                )

        all_results = await gather_with_progress(
            tasks, '📊 Evaluating metrics', show_progress
        )

        for result in all_results:
            if isinstance(result, MetricScore):
                results_map[result.id].append(result)
            elif isinstance(result, Exception):
                logger.error(f'Metric execution failed: {result}', exc_info=False)

        return [
            TestResult(test_case=item, score_results=results_map.get(item.id, []))
            for item in items
        ]

    def _finalize_results(self, final_results: List[TestResult]) -> None:
        """Finalize results by generating summary."""
        if self.summary_generator and self._elapsed_time is not None:
            self._summary = self.summary_generator.execute(
                final_results, self._elapsed_time
            )
        elif self.summary_generator:
            # Generate summary even without timing for PER_ITEM/SEPARATE modes
            self._summary = self.summary_generator.execute(final_results, 0.0)

    @classmethod
    def display(cls, show_helper_methods: bool = False, show_init_params: bool = False):
        from axion.docs.display_registry import (
            create_metric_runner_display,
            prepare_metric_runner_registry,
        )

        prepared_registry = prepare_metric_runner_registry(
            MetricRunnerFactory._registry
        )
        metric_executor_display = create_metric_runner_display(
            show_helper_methods, show_init_params
        )
        metric_executor_display.display(prepared_registry)

    @property
    def available_types(self) -> List[str]:
        """Returns a list of available (registered) metric runner types."""
        return list[str](MetricRunnerFactory._registry.keys())

    @property
    def elapsed_time(self) -> Union[float, None]:
        """Returns the total execution time for the last batch run."""
        return self._elapsed_time

    @property
    def summary(self) -> Union[Dict[str, Any], None]:
        """Returns the summary of the last batch run."""
        return self._summary


@MetricRunnerFactory.register('axion')
class AxionRunner(BaseMetricRunner):
    """Executes native metrics from Axion."""

    _name = 'axion'

    async def execute(
        self,
        input_data: Union[DatasetItem, Dict[str, Any]],
        cache: Optional[AnalysisCache] = None,
    ) -> MetricScore:
        from axion._core.tracing.context import get_current_tracer_safe

        # Use context tracer first (for async-safe tracing), fall back to self.tracer
        tracer = get_current_tracer_safe() or self.tracer

        async with tracer.async_span(f'(Axion) {self.metric_name.title()}') as span:
            span.set_attribute('metric_name', self.metric_name)
            span.set_attribute('data_id', getattr(input_data, 'id', 'unknown'))

            input_data = format_input(input_data)

            span.set_input(
                {
                    'query': getattr(input_data, 'query', None),
                    'actual_output': getattr(input_data, 'actual_output', None),
                    'expected_output': getattr(input_data, 'expected_output', None),
                    'retrieved_content': getattr(input_data, 'retrieved_content', None),
                    'additional_input': getattr(input_data, 'additional_input', None),
                    'latency': getattr(input_data, 'latency', None),
                }
            )

            try:
                result = await self.metric.execute(input_data, cache=cache)
                score = getattr(result, 'score', np.nan)

                signals = SignalExtractor.extract(
                    self.metric, getattr(result, 'signals', None)
                )

                span.set_attribute('score', float(score))
                span.set_attribute('passed', self._has_passed(score))

                span.set_output(
                    {
                        'score': float(score) if not np.isnan(score) else None,
                        'passed': self._has_passed(score),
                        'explanation': getattr(result, 'explanation', None),
                        'signals': signals,
                    }
                )

                return MetricScore(
                    id=input_data.id,
                    name=self.metric_name,
                    score=score,
                    threshold=self.threshold,
                    passed=self._has_passed(score),
                    explanation=getattr(result, 'explanation', None),
                    signals=signals,
                    metadata=getattr(result, 'metadata', None),
                    source=self.source,
                    cost_estimate=getattr(self.metric, 'cost_estimate', 0),
                )
            except Exception as e:
                logger.error(f'AXION execution failed for {self.metric_name}: {e}')
                span.set_attribute('error', str(e))
                return self._create_error_score(input_data.id, e)


@MetricRunnerFactory.register('ragas')
class RagasRunner(BaseMetricRunner):
    """Executes metrics from the Ragas library."""

    _name = 'ragas'

    async def execute(
        self,
        input_data: Union[DatasetItem, Dict[str, Any]],
        cache: Optional[AnalysisCache] = None,
    ) -> MetricScore:
        from ragas import SingleTurnSample

        from axion._core.tracing.context import get_current_tracer_safe

        # Use context tracer first (for async-safe tracing), fall back to self.tracer
        tracer = get_current_tracer_safe() or self.tracer

        async with tracer.async_span(f'(Ragas) {self.metric.name.title()}') as span:
            span.set_attribute('metric_name', self.metric_name)
            span.set_attribute('data_id', getattr(input_data, 'id', 'unknown'))

            input_data = format_input(input_data)
            EvaluationValidation.ensure_required_fields_present(input_data, self._name)

            # Reset cost tracking before evaluation
            llm = getattr(self.metric, 'llm', None)
            if llm and hasattr(llm, 'reset_cost'):
                llm.reset_cost()

            span.set_input(
                {
                    'query': getattr(input_data, 'query', None),
                    'actual_output': getattr(input_data, 'actual_output', None),
                    'expected_output': getattr(input_data, 'expected_output', None),
                    'retrieved_content': getattr(input_data, 'retrieved_content', None),
                }
            )

            try:
                async with tracer.async_span('create_sample') as sample_span:
                    sample_span.add_trace(
                        'info', 'Creating SingleTurnSample for evaluation'
                    )

                    additional = input_data.additional_input

                    sample_span.set_input(
                        {
                            'user_input': input_data.query,
                            'response': input_data.actual_output,
                            'reference': input_data.expected_output,
                            'retrieved_contexts': self._prepare_retrieved_content(
                                input_data.retrieved_content
                            ),
                            'reference_contexts': additional.get('reference_contexts'),
                        }
                    )

                    sample = SingleTurnSample(
                        user_input=input_data.query,
                        response=input_data.actual_output,
                        reference=input_data.expected_output,
                        retrieved_contexts=self._prepare_retrieved_content(
                            input_data.retrieved_content
                        ),
                        reference_contexts=additional.get('reference_contexts'),
                        multi_responses=additional.get('multi_responses'),
                        rubrics=additional.get('rubrics'),
                    )

                    sample_span.set_output(
                        {
                            'sample_type': 'SingleTurnSample',
                            'has_reference': input_data.expected_output is not None,
                            'has_retrieved_contexts': input_data.retrieved_content
                            is not None,
                        }
                    )
                score = await self.metric.single_turn_ascore(sample)
                span.set_attribute('score', float(score))

                # Extract cost using scalable cost extraction utility
                # Follows Ragas cost tracking pattern:
                # https://docs.ragas.io/en/stable/howtos/applications/_cost/
                cost = extract_cost(self.metric)

                # Capture output - the metric result
                span.set_output(
                    {
                        'score': float(score) if score is not None else None,
                        'passed': self._has_passed(score),
                    }
                )

                return MetricScore(
                    id=input_data.id,
                    name=self.metric_name,
                    score=score,
                    threshold=self.threshold,
                    passed=self._has_passed(score),
                    source=self.source,
                    cost_estimate=cost,
                )
            except Exception as e:
                logger.error(f'Ragas execution failed for {self.metric_name}: {e}')
                span.set_attribute('error', str(e))
                return self._create_error_score(input_data.id, e)


@MetricRunnerFactory.register('deepeval')
class DeepEvalRunner(BaseMetricRunner):
    """Executes metrics from the DeepEval library."""

    _name = 'deepeval'

    async def execute(
        self,
        input_data: Union[DatasetItem, Dict[str, Any]],
        cache: Optional[AnalysisCache] = None,
    ) -> MetricScore:
        from deepeval.test_case import LLMTestCase

        from axion._core.tracing.context import get_current_tracer_safe

        # Use context tracer first (for async-safe tracing), fall back to self.tracer
        tracer = get_current_tracer_safe() or self.tracer

        async with tracer.async_span(
            f'(DeepEval) {self.metric.__class__.__name__}'
        ) as span:
            span.set_attribute('metric_name', self.metric_name)
            span.set_attribute('data_id', getattr(input_data, 'id', 'unknown'))

            input_data = format_input(input_data)
            EvaluationValidation.ensure_required_fields_present(input_data, self._name)

            # Reset cost tracking before evaluation
            model = getattr(self.metric, 'model', None)
            if model and hasattr(model, 'reset_cost'):
                model.reset_cost()

            span.set_input(
                {
                    'query': getattr(input_data, 'query', None),
                    'actual_output': getattr(input_data, 'actual_output', None),
                    'expected_output': getattr(input_data, 'expected_output', None),
                    'retrieved_content': getattr(input_data, 'retrieved_content', None),
                }
            )

            try:
                async with tracer.async_span('create_test_case') as test_case_span:
                    test_case_span.add_trace(
                        'info', 'Creating LLMTestCase for evaluation'
                    )

                    additional = input_data.additional_input

                    test_case_span.set_input(
                        {
                            'input': input_data.query,
                            'actual_output': input_data.actual_output,
                            'expected_output': input_data.expected_output,
                            'retrieval_context': self._prepare_retrieved_content(
                                input_data.retrieved_content
                            ),
                            'completion_time': input_data.latency,
                            'context': additional.get('context'),
                        }
                    )

                    test_case = LLMTestCase(
                        input=input_data.query,
                        actual_output=input_data.actual_output,
                        expected_output=input_data.expected_output,
                        retrieval_context=self._prepare_retrieved_content(
                            input_data.retrieved_content
                        ),
                        completion_time=input_data.latency,
                        context=additional.get('context'),
                        tools_called=additional.get('tools_called'),
                        expected_tools=additional.get('expected_tools'),
                        token_cost=additional.get('token_cost'),
                        comments=additional.get('comments'),
                        additional_metadata=input_data.metadata,
                    )

                    test_case_span.set_output(
                        {
                            'test_case_type': 'LLMTestCase',
                            'has_expected_output': input_data.expected_output
                            is not None,
                            'has_retrieval_context': input_data.retrieved_content
                            is not None,
                            'has_completion_time': input_data.latency is not None,
                        }
                    )

                await self.metric.a_measure(test_case, _show_indicator=False)
                score = self.metric.score
                span.set_attribute('score', float(score))

                # Extract cost using scalable cost extraction utility
                cost = extract_cost(self.metric)

                span.set_output(
                    {
                        'score': float(score) if score is not None else None,
                        'passed': self._has_passed(score),
                        'explanation': getattr(self.metric, 'reason', None),
                    }
                )

                return MetricScore(
                    id=input_data.id,
                    name=self.metric_name,
                    score=score,
                    explanation=getattr(self.metric, 'reason', None),
                    threshold=self.threshold,
                    passed=self._has_passed(score),
                    source=self.source,
                    cost_estimate=cost,
                )

            except Exception as e:
                logger.error(f'DeepEval execution failed for {self.metric_name}: {e}')
                span.set_attribute('error', str(e))
                return self._create_error_score(input_data.id, e)
