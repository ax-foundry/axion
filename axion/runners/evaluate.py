import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from axion._core.asyncio import (
    SemaphoreExecutor,
    gather_with_progress,
    run_async_function,
)
from axion._core.cache.manager import CacheManager
from axion._core.cache.schema import CacheConfig
from axion._core.config import Config
from axion._core.error import InvalidConfig
from axion._core.logging import get_logger
from axion._core.metadata.schema import ToolMetadata
from axion._core.tracing import init_tracer, trace
from axion._core.tracing.handlers import BaseTraceHandler
from axion._core.types import TraceGranularity
from axion._core.utils import Timer
from axion._core.uuid import uuid7
from axion.dataset import Dataset, DatasetItem
from axion.metrics import metric_registry
from axion.runners.api import BaseAPIRunner
from axion.runners.mixin import RunnerMixin
from axion.runners.strategies import (
    BaseScoringStrategy,
    FlatScoringStrategy,
    HierarchicalScoringStrategy,
    ScoringStrategyType,
)
from axion.runners.summary import BaseSummary, MetricSummary
from axion.schema import ErrorConfig, EvaluationResult
from axion.utils import lazy_import
from axion.validation import EvaluationValidation

logger = get_logger(__name__)


@dataclass
class EvaluationConfig:
    """
    Configuration for an evaluation run.

    Attributes:
        evaluation_inputs (Union[Dataset, List[DatasetItem], pd.DataFrame]):
            The input dataset to evaluate. Can be a high-level `Dataset` object, a list of individual `DatasetItem`
            objects, or a preloaded `pandas.DataFrame`.

        scoring_config (Optional[Union[List[Any], Dict[str, Any], str]], optional):
            The scoring configuration. Can be:
            - A list of metrics for flat evaluation
            - A dictionary with 'metric' key for flat evaluation (when scoring_strategy='flat')
            - A dictionary for hierarchical (EvalTree) evaluation (with model, weights, etc.)
            - A string file path to a YAML configuration file

        scoring_metrics (List[Any]):
            A list of metric objects or callables used to score each item in the dataset.

        scoring_strategy (Optional[Union[BaseScoringStrategy, str, ScoringStrategyType]], optional):
            Defines the scoring method. Can be a pre-initialized strategy instance
            or a string/Enum alias ('flat' or 'tree'). Overrides auto-detection.

        evaluation_name (str):
            A unique name to identify the evaluation. Used in trace logging and result storage.

        task (Optional[Union[Callable, BaseAPIRunner]], optional):
            A custom function to generate predictions or transform inputs. If provided, it will be run
            before scoring to produce the model output for each dataset item.

        scoring_key_mapping (Optional[Dict[str, str]], optional):
            An optional dictionary mapping metric input names to dataset column names.
            Useful for adapting metrics to different schema formats.

        evaluation_description (Optional[str], optional):
            A human-readable description of the evaluation for documentation and trace metadata.

        evaluation_metadata (Optional[Dict[str, Any]], optional):
            Additional metadata to include in the evaluation trace (e.g., model version, data slice info, tags).

        max_concurrent (int, optional):
            Maximum number of metric evaluations to run concurrently. Default is 5.

        throttle_delay (float, optional):
            Specifies the time in seconds to pause after each individual task
            execution. This is used as a client-side throttle to help prevent
            API rate limit errors when processing a large number of items.
            Defaults to 0.0 (no delay).

        summary_generator (Optional[BaseSummary], optional):
            Optional summary generator used to produce a high-level summary after the evaluation.
            If not provided, a default `MetricSummary` is used.

        cache_config (CacheConfig, optional):
            Configuration for caching metric results to avoid recomputation. Enables both read and write caching.

        error_config (ErrorConfig, optional):
            Configuration for how errors are handled during evaluation. Allows skipping metrics or suppressing failures.

        thresholds (Optional[Dict[str, float]], optional):
            Optional threshold values for each metric. Used to flag items or datasets that fall below a given performance level.

        show_progress (bool, optional):
            Whether to show a progress bar during evaluation. Defaults to True.

        dataset_name (Optional[str], optional):
            Optional name of the dataset being evaluated. Used for display and trace logging.

        run_id (Optional[str], optional):
            An optional identifier for this specific run. Useful for repeatability and audit logging.

        trace_granularity (Union[TraceGranularity, str], optional):
            Controls trace granularity during evaluation. Accepts enum or string values:
            - 'single_trace' / 'single' / SINGLE_TRACE (default): All evaluations under one parent trace
            - 'separate' / SEPARATE: Each metric execution gets its own independent trace
    """

    evaluation_name: str
    evaluation_inputs: Union[Dataset, List[DatasetItem], pd.DataFrame]
    scoring_config: Optional[Union[List[Any], Dict[str, Any], str]] = None
    scoring_metrics: Optional[List[Any]] = None
    scoring_strategy: Optional[Union[BaseScoringStrategy, str, ScoringStrategyType]] = (
        None
    )
    task: Optional[Union[Callable, BaseAPIRunner]] = None
    scoring_key_mapping: Optional[Dict[str, str]] = None
    evaluation_description: Optional[str] = None
    evaluation_metadata: Optional[Dict[str, Any]] = None
    max_concurrent: int = 5
    throttle_delay: Optional[float] = 0.0
    summary_generator: Optional[BaseSummary] = field(default_factory=MetricSummary)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    error_config: ErrorConfig = field(default_factory=ErrorConfig)
    thresholds: Optional[Dict[str, float]] = None
    show_progress: bool = True
    dataset_name: Optional[str] = None
    run_id: Optional[str] = None
    enable_internal_caching: bool = True
    trace_granularity: Union[TraceGranularity, str] = TraceGranularity.SINGLE_TRACE
    flush_per_metric: bool = False

    def __post_init__(self):
        """Validate and finalize scoring configuration."""
        # Convert string to TraceGranularity enum if needed
        if isinstance(self.trace_granularity, str):
            self.trace_granularity = TraceGranularity.from_str(self.trace_granularity)

        # Load from YAML path if 'scoring_config' is a string path
        if isinstance(self.scoring_config, str):
            try:
                loaded_config = Config(self.scoring_config).config
                self.scoring_config = loaded_config
            except Exception as e:
                raise InvalidConfig(
                    f"Failed to load 'scoring_config' from path: {self.scoring_config}. Error: {e}"
                )

        if self.scoring_config and self.scoring_metrics:
            raise ValueError(
                "Provide either 'scoring_config' or 'scoring_metrics', not both."
            )

        # Validate string-based strategy against the Enum
        if (
            isinstance(self.scoring_strategy, str)
            and self.scoring_strategy not in ScoringStrategyType.values()
        ):
            raise ValueError(
                f"If 'scoring_strategy' is a string, it must be one of {ScoringStrategyType.values()}."
            )

        if self.scoring_strategy:
            if isinstance(self.scoring_strategy, BaseScoringStrategy):
                return
            if not self.scoring_config and not self.scoring_metrics:
                raise ValueError(
                    f"When using scoring_strategy='{self.scoring_strategy}', "
                    f"you must also provide 'scoring_config' or 'scoring_metrics'."
                )
            return

        if not self.scoring_config and not self.scoring_metrics:
            raise ValueError(
                "Must provide either 'scoring_config', 'scoring_metrics', or a 'scoring_strategy'."
            )

        # If scoring_metrics is provided, it becomes the main scoring_config.
        if self.scoring_metrics:
            self.scoring_config = self.scoring_metrics


class EvaluationRunner(RunnerMixin):
    """
    Orchestrates the execution of evaluation experiments, managing task execution,
    metric scoring, and configuration. Automatically determines and initializes
    the appropriate scoring strategy (flat or hierarchical).
    """

    name: str = 'EvaluationRunner'
    description: str = (
        'Orchestrates evaluation experiments with task execution, metric scoring, '
        'and advanced configuration options.'
    )
    tracer: Optional[BaseTraceHandler] = None
    scoring_strategy: BaseScoringStrategy

    def __init__(
        self, config: EvaluationConfig, tracer: Optional[BaseTraceHandler] = None
    ):
        self.config = config
        self.run_id = config.run_id or f'evaluation_{uuid7()}'
        self.task_semaphore = SemaphoreExecutor(config.max_concurrent)
        self.cache_manager = CacheManager(config.cache_config)

        self.tracer = init_tracer(
            metadata_type='base',
            tool_metadata=self.get_tool_metadata(),
            tracer=tracer,
        )

        self.scoring_strategy = self._initialize_scoring_strategy(config)

        if not config.task:
            EvaluationValidation.validate_evaluation_task_inputs(
                config.evaluation_inputs
            )

    def _create_metric_instance(self, name: str, cfg: Dict[str, Any]) -> Any:
        """
        Instantiates a metric from its config dictionary using registry or lazy import.
        """
        if not isinstance(cfg, dict):
            return cfg

        class_key = cfg.get('class')
        if not class_key:
            # If no class key, assume it's an instance (or invalid, but let strategy handle it)
            logger.warning(
                f"No 'class' key found for metric config '{name}'. "
                'Assuming it is a pre-instantiated metric object.'
            )
            return cfg

        kwargs = {k: v for k, v in cfg.items() if k != 'class'}
        metric_class = None

        # Try to get from registry using the 'class' value (user's new request)
        metric_class = metric_registry.get(class_key, error=False)

        if metric_class:
            try:
                return metric_class(**kwargs)
            except Exception as e:
                raise InvalidConfig(
                    f"Failed to create metric '{name}' from registry key '{class_key}': {e}"
                )

        # Try to lazy_import the 'class' value as a class path (for external libraries)
        try:
            if '.' in class_key:  # Must pass full path with '.'
                metric_class = lazy_import(class_key)
                module_path = class_key.lower()
                model_name = kwargs.pop('model_name')

                if 'deepeval' in module_path:
                    from axion.integrations.models import LiteLLMDeepEval

                    deepeval_model = LiteLLMDeepEval(model=model_name)

                    # DeepEval uses 'model' parameter
                    kwargs['model'] = deepeval_model

                elif 'ragas' in module_path:
                    from axion.integrations.models import LiteLLMRagas

                    ragas_model = LiteLLMRagas(model=model_name)

                    # Ragas uses 'llm' parameter
                    kwargs['llm'] = ragas_model

                return metric_class(**kwargs)

        except ImportError as e:
            # This is not a fatal error, just means it's not a valid class path
            logger.debug(f"Could not lazy import '{class_key}': {e}")
            pass

        # Try to get from registry using the metric name
        metric_class = metric_registry.get(name, error=False)
        if metric_class:
            try:
                return metric_class(**kwargs)
            except Exception as e:
                raise InvalidConfig(
                    f"Failed to create metric '{name}' from registry key '{name}': {e}"
                )

        raise InvalidConfig(
            f"Failed to find or create metric '{name}'. "
            f"Could not find '{class_key}' in registry or as a valid class path. \n"
            f'For internal AXION metrics see metric_registry.display() to pass the metric name. \n'
            f'For external metrics must pass full path (i.e. deepeval.metrics.AnswerRelevancyMetric) \n'
        )

    def _initialize_scoring_strategy(
        self, config: EvaluationConfig
    ) -> BaseScoringStrategy:
        """
        Selects and instantiates the appropriate scoring strategy,
        pre-instantiating metrics from config.
        """
        strategy = config.scoring_strategy
        scoring_config = config.scoring_config

        # Infer strategy if not explicitly provided
        if not strategy:
            strategy = (
                ScoringStrategyType.FLAT
                if isinstance(scoring_config, list)
                else ScoringStrategyType.TREE
                if isinstance(scoring_config, dict)
                else None
            )

        # Handle flat strategy with dict format (when explicitly set to 'flat')
        if strategy == ScoringStrategyType.FLAT and isinstance(scoring_config, dict):
            # Extract metrics from 'metric' key if present
            if 'metric' in scoring_config and isinstance(
                scoring_config['metric'], dict
            ):
                logger.info(
                    'Instantiating metrics from flat dict config for FlatScoringStrategy...'
                )
                metric_list = []
                for name, cfg in scoring_config['metric'].items():
                    metric_list.append(self._create_metric_instance(name, cfg))
                scoring_config = metric_list
            else:
                raise InvalidConfig(
                    "When using 'flat' scoring_strategy with dict config, "
                    "config must contain a 'metric' key with metric definitions."
                )
        elif strategy == ScoringStrategyType.FLAT and isinstance(scoring_config, list):
            # Instantiate metrics in a flat list
            logger.info('Instantiating metrics for FlatScoringStrategy...')
            scoring_config = [
                self._create_metric_instance(
                    # Use metric_name from config if present, else a default
                    cfg.get('metric_name', f'metric_{i}')
                    if isinstance(cfg, dict)
                    else f'metric_{i}',
                    cfg,
                )
                for i, cfg in enumerate(scoring_config)
            ]
        elif strategy == ScoringStrategyType.TREE and isinstance(scoring_config, dict):
            # Instantiate metrics in the 'metric' section of a hierarchical config
            if 'metric' in scoring_config and isinstance(
                scoring_config['metric'], dict
            ):
                logger.info('Instantiating metrics for HierarchicalScoringStrategy...')
                # Create a new dict to avoid modifying the original config in place
                new_metric_section = {}
                for name, cfg in scoring_config['metric'].items():
                    new_metric_section[name] = self._create_metric_instance(name, cfg)

                # Clone config and update the 'metric' section
                scoring_config = {**scoring_config, 'metric': new_metric_section}

        # Direct instance provided
        if isinstance(strategy, BaseScoringStrategy):
            logger.info(
                f'Using user-defined scoring strategy: {type(strategy).__name__}'
            )
            return strategy

        # Flat strategy
        if strategy == ScoringStrategyType.FLAT:
            if not isinstance(scoring_config, list):
                raise TypeError("Flat strategy requires 'scoring_config' to be a list.")
            logger.info('Using FlatScoringStrategy.')
            return FlatScoringStrategy(
                metrics=scoring_config,
                max_concurrent=config.max_concurrent,
                summary_generator=config.summary_generator,
                cache_manager=self.cache_manager,
                thresholds=config.thresholds,
                error_config=config.error_config,
                tracer=self.tracer,
                enable_internal_caching=config.enable_internal_caching,
                trace_granularity=config.trace_granularity,
                flush_per_metric=config.flush_per_metric,
            )

        # Hierarchical strategy
        if strategy == ScoringStrategyType.TREE:
            if not isinstance(scoring_config, dict):
                raise TypeError(
                    "Tree strategy requires 'scoring_config' to be a dictionary."
                )
            logger.info('Using HierarchicalScoringStrategy.')
            return HierarchicalScoringStrategy(
                config=scoring_config,
                max_concurrent=config.max_concurrent,
                summary_generator=config.summary_generator,
                enable_internal_caching=config.enable_internal_caching,
                trace_granularity=config.trace_granularity,
            )

        # Invalid case
        raise ValueError(f'Invalid or undetermined scoring strategy: {strategy}')

    @trace(name='process_task_output', capture_args=True, capture_result=True)
    def _process_task_output(self, task_output: Any, item_id: str) -> Dict[str, Any]:
        """Convert task output to a dictionary and apply key mapping for scoring."""
        if hasattr(task_output, 'model_dump'):
            output_dict = task_output.model_dump()
        elif isinstance(task_output, dict):
            output_dict = task_output
        else:
            logger.warning(
                f'Unexpected task output type: {type(task_output)}. Attempting to convert to dict.'
            )
            try:
                output_dict = dict(task_output)
            except (TypeError, ValueError):
                logger.error(
                    f"Failed to convert task output to dict for item '{item_id}'."
                )
                return {}

        if not self.config.scoring_key_mapping:
            return output_dict

        # Invert the mapping for processing
        inverted_mapping = {v: k for k, v in self.config.scoring_key_mapping.items()}
        return {inverted_mapping.get(k, k): v for k, v in output_dict.items()}

    @trace(name='run_task', capture_args=True, capture_result=True)
    async def _run_single_task(self, item: DatasetItem) -> DatasetItem:
        """Runs the generation task for a single item, with caching."""
        task_cache_key = f'task_{self.config.task.__name__}_{item.id}'
        task_output = None

        if self.cache_manager and self.config.cache_config.cache_task:
            task_output = self.cache_manager.get(task_cache_key)

        if task_output is None:
            if isinstance(self.config.task, BaseAPIRunner):
                task_output = await self.task_semaphore.run(
                    self.config.task.execute, item.query
                )
            else:
                task_output = await self.task_semaphore.run(self.config.task, item)

            if (
                self.config.throttle_delay is not None
                and self.config.throttle_delay > 0
            ):
                await asyncio.sleep(self.config.throttle_delay)
            if self.cache_manager and self.config.cache_config.cache_task:
                self.cache_manager.set(task_cache_key, task_output)

        processed_data = self._process_task_output(
            task_output=task_output, item_id=item.id
        )
        item.update(processed_data)
        return item

    @trace(name='generation_stage')
    async def _run_generation_stage(
        self,
        evaluation_inputs: List[DatasetItem],
        return_result: bool = False,
    ) -> Optional[Any]:
        """Execute the generation stage for all evaluation inputs."""

        logger.info(f'Running generation task for {len(evaluation_inputs)} inputs...')

        tasks = [self._run_single_task(item) for item in evaluation_inputs]

        results = await gather_with_progress(
            tasks=tasks,
            description='🚀 Running generation task',
            show_progress=self.config.show_progress,
        )

        return results if return_result else None

    async def execute(self) -> EvaluationResult:
        """Executes the entire evaluation and returns the final result.

        For SINGLE_TRACE mode, wraps execution in a trace span.
        For PER_ITEM and SEPARATE modes, skips the wrapper span to allow
        each item/metric to create its own independent trace.
        """
        if self.config.trace_granularity == TraceGranularity.SINGLE_TRACE:
            return await self._execute_with_trace()
        else:
            return await self._execute_impl()

    async def _execute_with_trace(self) -> EvaluationResult:
        """Execute with trace span wrapper (for SINGLE_TRACE mode)."""
        span_name = self.config.evaluation_name or 'Evaluation_execute'
        async with self.tracer.async_span(span_name) as span:
            # Capture input - evaluation configuration
            input_count = (
                len(self.config.evaluation_inputs)
                if hasattr(self.config.evaluation_inputs, '__len__')
                else 0
            )

            # Get metric names from scoring_metrics or scoring_config
            metric_names = []
            if self.config.scoring_metrics:
                metric_names = [type(m).__name__ for m in self.config.scoring_metrics]
            elif isinstance(self.config.scoring_config, list):
                metric_names = [type(m).__name__ for m in self.config.scoring_config]
            elif isinstance(self.config.scoring_config, dict):
                metric_dict = self.config.scoring_config.get('metric', {})
                if isinstance(metric_dict, dict):
                    metric_names = list(metric_dict.keys())

            span.set_input(
                {
                    'evaluation_name': self.config.evaluation_name,
                    'dataset_name': self.config.dataset_name,
                    'input_count': input_count,
                    'metrics': metric_names,
                    'max_concurrent': self.config.max_concurrent,
                }
            )

            try:
                result = await self._execute_impl()

                # Capture output - evaluation results summary
                span.set_output(
                    {
                        'run_id': result.run_id,
                        'total_items': len(result.results) if result.results else 0,
                        'metrics_evaluated': (
                            [s.name for s in result.results[0].score_results]
                            if result.results and result.results[0].score_results
                            else []
                        ),
                        'status': 'completed',
                    }
                )

                return result

            except Exception as e:
                span.set_output(
                    {
                        'status': 'failed',
                        'error_type': type(e).__name__,
                        'error_message': str(e)[:500],
                    }
                )
                raise

    async def _execute_impl(self) -> EvaluationResult:
        """Internal implementation of evaluation execution."""
        logger.info(
            f"Starting evaluation '{self.config.evaluation_name}' with run_id: {self.run_id}"
        )

        evaluation_inputs = self.to_evaluation_input(
            self.config.evaluation_inputs, self.config.dataset_name
        )
        if self.config.task:
            await self._run_generation_stage(evaluation_inputs)

        test_results = await self.scoring_strategy.execute(
            evaluation_inputs, show_progress=self.config.show_progress
        )

        final_metadata = {
            **(
                {'description': self.config.evaluation_description}
                if self.config.evaluation_description is not None
                else {}
            ),
            **(self.config.evaluation_metadata or {}),
        }

        evaluation_result = EvaluationResult(
            run_id=self.run_id,
            evaluation_name=self.config.evaluation_name,
            timestamp=Timer.current_timestamp(),
            results=test_results,
            summary=self.scoring_strategy.summary,
            metadata=final_metadata,
        )

        if self.cache_manager:
            self.cache_manager.close()

        logger.info(
            f"Evaluation '{self.config.evaluation_name}' finished successfully."
        )
        return evaluation_result

    @property
    def summary(self) -> Union[Dict[str, Any], None]:
        """
        Returns the summary from the active scoring strategy.
        For hierarchical ('tree') strategies, this provides the detailed tree summary.
        """
        return self.scoring_strategy.summary

    @property
    def tree(self) -> Any:
        """
        Returns the underlying EvalTree instance for inspection, if the 'tree'
        strategy is active. Raises an AttributeError for other strategies.
        """
        if isinstance(self.scoring_strategy, HierarchicalScoringStrategy):
            return self.scoring_strategy.tree
        raise AttributeError(
            "'tree' property is only available when using the 'tree' scoring strategy."
        )

    @classmethod
    def display(cls):
        """Display Usage Documentation"""
        from IPython.display import HTML, display

        from axion.docs.evaluate import (
            advanced_usage_template,
            basic_usage_template,
            documentation_template,
            hierarchical_usage_template,
            task_usage_template,
        )
        from axion.docs.render import create_multi_usage_modal_card

        evaluation_runner_card = create_multi_usage_modal_card(
            key='evaluation_runner',
            title='EvaluationRunner',
            description=cls.description,
            documentation_templates=[(documentation_template, '📖 Documentation')],
            usage_templates=[
                (basic_usage_template, '▶️ Basic Usage'),
                (hierarchical_usage_template, '🌳 Hierarchical Scoring'),
                (task_usage_template, '📋 Task Example'),
                (advanced_usage_template, '🎛️ Advanced Example'),
            ],
            max_width='1350px',
        )
        display(HTML(evaluation_runner_card))


async def _run_evaluation_async(
    config: 'EvaluationConfig', tracer: Any
) -> Optional[EvaluationResult]:
    """
    Execute an evaluation asynchronously with structured metadata tracking.

    This function runs an evaluation with telemetry instrumentation, capturing
    dataset characteristics, scoring configuration, runtime, and outcome status
    for observability and performance analysis.

    For PER_ITEM and SEPARATE trace granularity modes, the wrapper span is skipped
    to allow each item/metric to create its own independent trace.
    """
    # For non-SINGLE_TRACE modes, skip the wrapper span to allow separate traces
    if config.trace_granularity != TraceGranularity.SINGLE_TRACE:
        return await _run_evaluation_no_wrapper(config, tracer)

    span_name = config.evaluation_name or 'evaluation_runner'
    async with tracer.async_span(span_name) as span:
        # Capture input - the evaluation configuration
        input_count = (
            len(config.evaluation_inputs)
            if hasattr(config.evaluation_inputs, '__len__')
            else 0
        )
        span.set_input(
            {
                'evaluation_name': config.evaluation_name,
                'input_count': input_count,
                'metrics': [type(m).__name__ for m in config.scoring_metrics]
                if config.scoring_metrics
                else [],
            }
        )

        span.set_attribute('evaluation_name', config.evaluation_name)
        span.set_attribute('dataset_name', getattr(config, 'dataset_name', 'unknown'))
        span.set_attribute('input_type', type(config.evaluation_inputs).__name__)

        # Dataset characteristics
        if hasattr(config.evaluation_inputs, '__len__'):
            span.set_attribute('input_count', len(config.evaluation_inputs))

        # Scoring configuration
        scoring = config.scoring_config
        if isinstance(scoring, list):
            metrics = [type(m).__name__ for m in scoring]
        elif isinstance(scoring, dict):
            metric_dict = scoring.get('metric', {})
            metrics = [type(m).__name__ for m in metric_dict.values()]
        else:
            metrics = []

        span.set_attribute('metrics_count', len(metrics))
        span.set_attribute('metric_types', metrics)

        # General configuration
        span.set_attribute('max_concurrent', config.max_concurrent)
        span.set_attribute('show_progress', config.show_progress)
        if config.evaluation_description:
            span.set_attribute('description', config.evaluation_description[:200])

        timer = Timer()
        timer.start()

        try:
            runner = EvaluationRunner(config, tracer=tracer)
            result = await runner.execute()

            # Mark success
            timer.stop()
            span.set_attribute('execution_time', timer.elapsed_time)
            span.set_attribute('status', 'completed')

            # Add high-level summary metrics if present
            for attr in ('success_rate', 'total_evaluations'):
                if hasattr(result, attr):
                    span.set_attribute(attr, getattr(result, attr))

            # Capture output - evaluation results summary
            span.set_output(
                {
                    'total_items': len(result.results)
                    if result and result.results
                    else 0,
                    'metrics_evaluated': (
                        [s.name for s in result.results[0].score_results]
                        if result and result.results and result.results[0].score_results
                        else []
                    ),
                    'status': 'completed',
                }
            )

            return result

        except Exception as e:
            logger.error(f"Error running evaluation_runner:' {str(e)}")
            timer.stop()
            span.set_attribute('execution_time_seconds', timer.elapsed_time)
            span.set_attribute('status', 'failed')
            span.set_attribute('error.type', type(e).__name__)
            span.set_attribute('error.message', str(e)[:500])
            return None


async def _run_evaluation_no_wrapper(
    config: 'EvaluationConfig', tracer: Any
) -> Optional[EvaluationResult]:
    """
    Execute an evaluation without a wrapper span.

    Used for PER_ITEM and SEPARATE trace granularity modes where each
    item/metric creates its own independent trace.
    """
    timer = Timer()
    timer.start()

    try:
        runner = EvaluationRunner(config, tracer=tracer)
        result = await runner.execute()
        timer.stop()
        return result

    except Exception as e:
        logger.error(f'Error running evaluation_runner: {str(e)}')
        timer.stop()
        return None


def evaluation_runner(
    evaluation_inputs: Union[Dataset, List[DatasetItem], pd.DataFrame],
    evaluation_name: str,
    scoring_config: Optional[Union[List[Any], Dict[str, Any], str]] = None,
    scoring_metrics: Optional[List[Any]] = None,
    scoring_strategy: Optional[
        Union[BaseScoringStrategy, str, ScoringStrategyType]
    ] = None,
    task: Optional[Union[Callable, BaseAPIRunner]] = None,
    scoring_key_mapping: Optional[Dict[str, str]] = None,
    evaluation_description: Optional[str] = None,
    evaluation_metadata: Optional[Dict[str, Any]] = None,
    max_concurrent: int = 5,
    throttle_delay: float = 0.0,
    summary_generator: Optional[BaseSummary] = None,
    cache_config: Optional[CacheConfig] = None,
    error_config: Optional[ErrorConfig] = None,
    enable_internal_caching: bool = True,
    thresholds: Optional[Dict[str, float]] = None,
    show_progress: bool = True,
    dataset_name: Optional[str] = None,
    run_id: Optional[str] = None,
    trace_granularity: Union[TraceGranularity, str] = TraceGranularity.SINGLE_TRACE,
    flush_per_metric: bool = False,
) -> Optional[EvaluationResult]:
    """
    Synchronously runs an evaluation experiment to evaluate metrics over a given dataset,
    supporting both flat and hierarchical scoring structures.

    Args:
        evaluation_inputs (Union[Dataset, List[DatasetItem], pd.DataFrame]):
            The input dataset to evaluate.
        evaluation_name (str):
            A unique name to identify the evaluation.
        scoring_config (Optional[Union[List[Any], Dict[str, Any], str]], optional):
            The scoring configuration. Can be:
            - A list of metrics for flat evaluation
            - A dictionary with 'metric' key for flat evaluation (when scoring_strategy='flat')
            - A dictionary for hierarchical (EvalTree) evaluation (with model, weights, etc.)
            - A string file path to a YAML configuration file
        scoring_metrics (Optional[List[Any]], optional):
            An alternative, more intuitive parameter for passing a flat list of metrics.
        scoring_strategy (Optional[Union[BaseScoringStrategy, str, ScoringStrategyType]], optional):
            Defines the scoring method. Can be a pre-initialized strategy instance
            or a string/Enum alias ('flat' or 'tree'). Overrides auto-detection.
        task (Optional[Union[Callable, BaseAPIRunner]], optional):
            A custom function to generate model outputs before scoring.
        scoring_key_mapping (Optional[Dict[str, str]], optional):
            Maps metric input names to dataset column names.
        evaluation_description (Optional[str], optional):
            A human-readable description of the evaluation.
        evaluation_metadata (Optional[Dict[str, Any]], optional):
            Additional metadata to include in the evaluation trace.
        max_concurrent (int, optional):
            Maximum number of concurrent evaluations. Defaults to 5.
        throttle_delay (float, optional):
            Specifies the time in seconds to pause after each individual task
            execution. This is used as a client-side throttle to help prevent
            API rate limit errors when processing a large number of items.
            Defaults to 0.0 (no delay).
        summary_generator (Optional[BaseSummary], optional):
            A summary generator for high-level results.
        cache_config (CacheConfig, optional):
            Configuration for caching results to avoid recomputation.
        error_config (ErrorConfig, optional):
            Configuration for handling errors during evaluation.
        enable_internal_caching (bool, optional):
            Enables a per-item cache for metrics that share expensive internal
            computations. Defaults to True.
        thresholds (Optional[Dict[str, float]], optional):
            Performance thresholds for each metric.
        show_progress (bool, optional):
            Whether to show a progress bar. Defaults to True.
        dataset_name (Optional[str], optional):
            Optional name of the dataset.
        run_id (Optional[str], optional):
            An optional identifier for this specific run.
        trace_granularity (Union[TraceGranularity, str], optional):
            Controls trace granularity during evaluation. Accepts enum or string values:
            - 'single_trace' / 'single' / SINGLE_TRACE (default): All evaluations under one parent trace
            - 'separate' / SEPARATE: Each metric execution gets its own independent trace
        flush_per_metric (bool, optional):
            When trace_granularity='separate', controls whether each metric trace is flushed
            immediately (slower, but more \"live\" in the UI) vs batched (faster). Defaults to False.

    Returns:
        EvaluationResult:
            An object containing detailed metric scores, summary, and metadata.
    """
    tracer = init_tracer(
        'base',
        tool_metadata=ToolMetadata(
            name='evaluation_runner',
            description=EvaluationRunner.description,
            owner='AI Engineering',
            version='0.0.1',
        ),
    )

    config = EvaluationConfig(
        evaluation_name=evaluation_name,
        evaluation_inputs=evaluation_inputs,
        scoring_config=scoring_config,
        scoring_metrics=scoring_metrics,
        scoring_strategy=scoring_strategy,
        task=task,
        scoring_key_mapping=scoring_key_mapping,
        evaluation_description=evaluation_description,
        evaluation_metadata=evaluation_metadata,
        max_concurrent=max_concurrent,
        throttle_delay=throttle_delay,
        summary_generator=summary_generator or MetricSummary(),
        cache_config=cache_config or CacheConfig(),
        error_config=error_config or ErrorConfig(),
        enable_internal_caching=enable_internal_caching,
        thresholds=thresholds,
        show_progress=show_progress,
        dataset_name=dataset_name,
        run_id=run_id,
        trace_granularity=trace_granularity,
        flush_per_metric=flush_per_metric,
    )
    return run_async_function(_run_evaluation_async, config, tracer)
