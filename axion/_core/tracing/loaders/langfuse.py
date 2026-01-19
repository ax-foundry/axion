"""
Langfuse trace loader for Axion.

This module provides functionality to:
1. Fetch traces from Langfuse
2. Convert them to Axion Dataset format
3. Push evaluation scores back to Langfuse traces
"""

import json
import os
import random
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar

import numpy as np
from pydantic import BaseModel

from axion._core.logging import get_logger
from axion._core.tracing.loaders.base import BaseTraceLoader
from axion.metrics.schema import DEFAULT_EXPLANATION

T = TypeVar('T')

if TYPE_CHECKING:
    from axion.schema import EvaluationResult

logger = get_logger(__name__)

__all__ = ['LangfuseTraceLoader']


class LangfuseTraceLoader(BaseTraceLoader):
    """
    Production-grade loader for Langfuse traces.

    Automatically identifies Retrieval and Generation spans to enable
    granular scoring of specific observations within a trace.

    Example:
        # Basic usage
        loader = LangfuseTraceLoader()
        dataset = loader.to_dataset(name='rag-eval', limit=100)

        # Run evaluation
        result = evaluation_runner(
            evaluation_inputs=dataset,
            scoring_metrics=[Faithfulness(), AnswerRelevancy()]
        )

        # Push scores back to Langfuse
        stats = loader.push_scores_to_langfuse(result)
        print(f"Uploaded {stats['uploaded']} scores")
    """

    # Error patterns that indicate transient/retryable failures
    _TRANSIENT_ERROR_PATTERNS = (
        'timed out',
        'timeout',
        'readtimeout',
        'read operation timed out',
        '429',
        '502',
        '503',
        '504',
        'rate limit',
    )

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        max_retries: int = 5,
        base_delay: float = 0.75,
        max_delay: float = 60.0,
        request_pacing: float = 0.03,
        default_tags: Optional[List[str]] = None,
    ):
        """
        Initialize Langfuse client.

        Args:
            public_key: Langfuse public key (falls back to LANGFUSE_PUBLIC_KEY env var)
            secret_key: Langfuse secret key (falls back to LANGFUSE_SECRET_KEY env var)
            host: Langfuse host URL (falls back to LANGFUSE_BASE_URL env var)
            max_retries: Maximum retry attempts for transient failures (default: 5)
            base_delay: Base delay in seconds for exponential backoff (default: 0.75)
            max_delay: Maximum delay in seconds between retries (default: 60.0)
            request_pacing: Delay between API requests to avoid rate limits (default: 0.03)
            default_tags: Default tags to apply to all scores (falls back to
                LANGFUSE_DEFAULT_TAGS env var, comma-separated)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.request_pacing = request_pacing

        # Support LANGFUSE_DEFAULT_TAGS env var (comma-separated)
        if default_tags is None:
            env_tags = os.environ.get('LANGFUSE_DEFAULT_TAGS')
            if env_tags:
                self.default_tags = [
                    t.strip() for t in env_tags.split(',') if t.strip()
                ]
            else:
                self.default_tags = []
        else:
            self.default_tags = default_tags

        try:
            from langfuse import Langfuse

            self.client: Langfuse = Langfuse(
                public_key=public_key or os.environ.get('LANGFUSE_PUBLIC_KEY'),
                secret_key=secret_key or os.environ.get('LANGFUSE_SECRET_KEY'),
                host=host
                or os.environ.get('LANGFUSE_BASE_URL', 'https://us.cloud.langfuse.com'),
            )
            self._client_initialized = True
        except ImportError:
            raise ImportError(
                'Langfuse SDK not found. Install with: pip install langfuse'
            )
        except Exception as e:
            logger.warning(f'Failed to initialize Langfuse client: {e}')
            self.client = None
            self._client_initialized = False

    def _is_transient_error(self, error: Exception) -> bool:
        """
        Check if an error is transient and should be retried.

        Detects rate limits (429), server errors (502, 503, 504), and timeouts.
        """
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in self._TRANSIENT_ERROR_PATTERNS)

    def _execute_with_retry(
        self,
        operation: Callable[[], T],
        description: str,
    ) -> T:
        """
        Execute an operation with exponential backoff retry.

        Args:
            operation: Callable that performs the API operation
            description: Description for logging (e.g., "fetch trace abc123")

        Returns:
            Result of the operation

        Raises:
            Exception: The last exception if all retries are exhausted
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except Exception as e:
                last_exception = e

                if attempt < self.max_retries and self._is_transient_error(e):
                    # Calculate delay with exponential backoff + jitter
                    delay = min(
                        self.base_delay * (2**attempt) + random.uniform(0, 0.4),
                        self.max_delay,
                    )
                    logger.warning(
                        f'Retry {attempt + 1}/{self.max_retries} for {description} '
                        f'in {delay:.1f}s ({e})'
                    )
                    time.sleep(delay)
                    continue

                # Non-transient error or exhausted retries
                raise

        # Should not reach here, but satisfy type checker
        raise last_exception  # type: ignore[misc]

    def _format_comment(
        self, explanation: Optional[str], signals: Optional[Any]
    ) -> Optional[str]:
        """
        Format explanation and signals into a well-formatted comment for UI legibility.

        Args:
            explanation: Optional explanation text
            signals: Optional signals data (dict, BaseModel, or other serializable)

        Returns:
            Formatted comment string combining both, or None if both are empty
        """
        parts = []

        # Add explanation if present and not the default placeholder
        if explanation and explanation.strip() != DEFAULT_EXPLANATION:
            parts.append(explanation.strip())

        # Format signals if present
        if signals is not None:
            signals_str = None

            # Handle BaseModel instances
            if isinstance(signals, BaseModel):
                try:
                    signals_str = signals.model_dump_json(indent=2)
                except Exception:
                    signals_str = str(signals)

            # Handle dictionaries
            elif isinstance(signals, dict):
                try:
                    signals_str = json.dumps(signals, indent=2, default=str)
                except Exception:
                    signals_str = str(signals)

            # Handle other types
            else:
                try:
                    # Try JSON serialization first
                    signals_str = json.dumps(signals, indent=2, default=str)
                except Exception:
                    # Fallback to string representation
                    signals_str = str(signals)

            if signals_str:
                # Add separator if explanation exists
                if parts:
                    parts.append('\n\n--- Signals ---\n')
                else:
                    parts.append('Signals:\n')
                parts.append(signals_str)

        if not parts:
            return None

        return '\n'.join(parts)

    def push_scores(
        self,
        evaluation_result: 'EvaluationResult',
        observation_id_field: Optional[str] = 'observation_id',
        flush: bool = True,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Push evaluation scores back to Langfuse.

        This is the generic interface method that delegates to push_scores_to_langfuse().

        Args:
            evaluation_result: The EvaluationResult from evaluation_runner
            observation_id_field: Field name on DatasetItem containing the
                observation ID. Default: 'observation_id'
            flush: Whether to flush the client after uploading
            tags: Optional list of tags to attach to all scores as metadata

        Returns:
            Dict with counts: {'uploaded': N, 'skipped': M}
        """
        return self.push_scores_to_langfuse(
            evaluation_result=evaluation_result,
            observation_id_field=observation_id_field,
            flush=flush,
            tags=tags,
        )

    def fetch_traces(
        self,
        limit: int = 50,
        days_back: int = 7,
        tags: Optional[List[str]] = None,
        name: Optional[str] = None,
        fetch_full_traces: bool = True,
    ) -> List[Any]:
        """
        Fetch raw traces from Langfuse.

        Args:
            limit: Maximum number of traces to fetch
            days_back: Number of days to look back
            tags: Filter by specific tags
            name: Filter by trace name
            fetch_full_traces: If True (default), fetch full trace details for each
                trace via additional API calls. If False, only return trace summaries
                (faster but less data). Set to False to avoid rate limits.

        Returns:
            List of raw Langfuse trace objects (full traces or summaries)
        """
        if not self._client_initialized:
            logger.error('Langfuse client not initialized')
            return []

        from_timestamp = self._normalize_time_window(days_back)
        logger.info(
            f'Fetching up to {limit} traces from Langfuse (since {from_timestamp})...'
        )

        # Fetch trace list with retry
        try:
            traces_page = self._execute_with_retry(
                lambda: self.client.api.trace.list(
                    limit=limit,
                    from_timestamp=from_timestamp,
                    tags=tags,
                    name=name,
                ),
                description='list traces',
            )
        except Exception as e:
            logger.error(f'Failed to fetch traces from Langfuse: {e}')
            return []

        if not fetch_full_traces:
            # Return trace summaries directly
            logger.info(f'Successfully loaded {len(traces_page.data)} trace summaries')
            return list(traces_page.data)

        # Fetch full trace details for each summary
        results = []
        for trace_summary in traces_page.data:
            try:
                full_trace = self._execute_with_retry(
                    lambda tid=trace_summary.id: self.client.api.trace.get(tid),
                    description=f'fetch trace {trace_summary.id}',
                )
                results.append(full_trace)

                # Pacing to avoid rate limits
                if self.request_pacing > 0:
                    time.sleep(self.request_pacing)

            except Exception as e:
                logger.warning(f'Skipping trace {trace_summary.id}: {e}')

        logger.info(f'Successfully loaded {len(results)} full traces')
        return results

    def push_scores_to_langfuse(
        self,
        evaluation_result: 'EvaluationResult',
        observation_id_field: Optional[str] = 'observation_id',
        flush: bool = True,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Push evaluation scores back to Langfuse traces.

        Args:
            evaluation_result: The EvaluationResult from evaluation_runner
            observation_id_field: Field name on DatasetItem containing the
                observation ID. If provided and the field has a value, scores
                attach to that specific observation. If None or field is empty,
                scores attach to the trace. Default: 'observation_id'
            flush: Whether to flush the client after uploading
            tags: Optional list of tags to attach to all scores as metadata.
                Falls back to LANGFUSE_TAGS env var if not provided.

        Note:
            Environment cannot be set when pushing scores to existing traces.
            To set environment, configure it at client initialization when creating
            traces (via LANGFUSE_ENVIRONMENT or LANGFUSE_TRACING_ENVIRONMENT env vars
            or the environment parameter in LangfuseTracer).

        Returns:
            Dict with counts: {'uploaded': N, 'skipped': M}

        Example:
            loader = LangfuseTraceLoader()

            result = evaluation_runner(
                evaluation_inputs=dataset,
                scoring_metrics=[Faithfulness()]
            )

            # Attach scores with tags
            stats = loader.push_scores_to_langfuse(
                result,
                tags=['prod', 'v1.0']
            )

            # Or attach scores to traces only
            stats = loader.push_scores_to_langfuse(result, observation_id_field=None)
        """
        if not self._client_initialized:
            logger.error('Langfuse client not initialized, cannot push scores')
            return {'uploaded': 0, 'skipped': 0}

        # Fallback to environment variables if not provided
        if tags is None:
            env_tags = os.environ.get('LANGFUSE_TAGS')
            if env_tags:
                tags = [t.strip() for t in env_tags.split(',') if t.strip()]

        stats = {'uploaded': 0, 'skipped': 0}

        for test_result in evaluation_result.results:
            if not test_result.test_case:
                continue

            trace_id = test_result.test_case.trace_id
            if not trace_id:
                stats['skipped'] += len(test_result.score_results)
                continue

            # Get observation_id if field is specified
            obs_id = None
            if observation_id_field:
                obs_id = getattr(test_result.test_case, observation_id_field, None)

            for metric_score in test_result.score_results:
                if metric_score.score is None or np.isnan(metric_score.score):
                    stats['skipped'] += 1
                    continue

                try:
                    score_kwargs = {
                        'trace_id': trace_id,
                        'name': metric_score.name,
                        'value': float(metric_score.score),
                        'comment': self._format_comment(
                            metric_score.explanation, metric_score.signals
                        ),
                    }

                    if obs_id:
                        score_kwargs['observation_id'] = obs_id

                    # Merge default_tags with per-call tags
                    effective_tags = list(self.default_tags)
                    if tags:
                        effective_tags.extend(tags)
                    if effective_tags:
                        score_kwargs['metadata'] = {'tags': effective_tags}

                    # Upload score with retry
                    self._execute_with_retry(
                        lambda kwargs=score_kwargs: self.client.create_score(**kwargs),
                        description=f'upload score {metric_score.name} for trace {trace_id}',
                    )
                    stats['uploaded'] += 1

                    # Pacing to avoid rate limits
                    if self.request_pacing > 0:
                        time.sleep(self.request_pacing)

                except Exception as e:
                    logger.warning(
                        f'Failed to upload score {metric_score.name} '
                        f'for trace {trace_id}: {e}'
                    )
                    stats['skipped'] += 1

        if flush:
            try:
                self._execute_with_retry(
                    lambda: self.client.flush(),
                    description='flush Langfuse client',
                )
            except Exception as e:
                logger.warning(f'Failed to flush Langfuse client: {e}')

        logger.info(
            f'Langfuse score upload: {stats["uploaded"]} uploaded, '
            f'{stats["skipped"]} skipped'
        )
        return stats

    def _serialize_dataset_item_input(self, item: Any) -> Dict[str, Any]:
        """
        Serialize a DatasetItem into the input format for Langfuse dataset items.

        Args:
            item: The DatasetItem to serialize

        Returns:
            Dict containing the input fields for a Langfuse dataset item
        """
        input_data: Dict[str, Any] = {}

        # Core input fields
        if hasattr(item, 'query') and item.query:
            input_data['query'] = item.query

        if hasattr(item, 'retrieved_content') and item.retrieved_content:
            input_data['retrieved_content'] = item.retrieved_content

        if hasattr(item, 'additional_input') and item.additional_input:
            input_data['additional_input'] = item.additional_input

        if hasattr(item, 'acceptance_criteria') and item.acceptance_criteria:
            input_data['acceptance_criteria'] = item.acceptance_criteria

        # Handle multi-turn conversations
        if hasattr(item, 'conversation') and item.conversation:
            try:
                if hasattr(item.conversation, 'model_dump'):
                    input_data['conversation'] = item.conversation.model_dump()
                else:
                    input_data['conversation'] = str(item.conversation)
            except Exception:
                input_data['conversation'] = str(item.conversation)

        return input_data

    def _serialize_expected_output(self, item: Any) -> Optional[Dict[str, Any]]:
        """
        Serialize the expected output from a DatasetItem.

        Args:
            item: The DatasetItem to extract expected output from

        Returns:
            Dict containing expected output, or None if not present
        """
        if hasattr(item, 'expected_output') and item.expected_output:
            return {'expected_output': item.expected_output}
        return None

    def upload_experiment(
        self,
        evaluation_result: 'EvaluationResult',
        dataset_name: Optional[str] = None,
        run_name: Optional[str] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
        flush: bool = True,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Upload evaluation results to Langfuse as a dataset experiment.

        This creates a dataset (if it doesn't exist), dataset items for each test case,
        and an experiment run with scores attached. Unlike `push_scores_to_langfuse()`,
        this method does not require existing traces - it creates everything from scratch.

        Args:
            evaluation_result: The EvaluationResult from evaluation_runner
            dataset_name: Name for the Langfuse dataset. Defaults to
                evaluation_result.evaluation_name or generates one.
            run_name: Name for the experiment run. Defaults to
                "{dataset_name}-{run_id}" pattern.
            run_metadata: Optional metadata to attach to the experiment run.
            flush: Whether to flush the client after uploading. Defaults to True.
            tags: Optional list of tags to attach to all scores as metadata.

        Returns:
            Dict with statistics:
                - dataset_name: Name of the created/used dataset
                - run_name: Name of the experiment run
                - items_created: Number of dataset items created
                - runs_created: Number of experiment runs created
                - scores_uploaded: Number of scores attached
                - scores_skipped: Number of scores skipped (None/NaN values)
                - errors: List of error messages encountered

        Example:
            loader = LangfuseTraceLoader()

            result = evaluation_runner(
                evaluation_inputs=dataset,
                scoring_metrics=[Faithfulness(), AnswerRelevancy()]
            )

            # Upload as experiment
            stats = loader.upload_experiment(
                result,
                dataset_name="my-rag-eval",
                run_name="experiment-v1",
                tags=['production']
            )
            print(f"Uploaded {stats['scores_uploaded']} scores to {stats['dataset_name']}")
        """
        if not self._client_initialized:
            logger.error('Langfuse client not initialized, cannot upload experiment')
            return {
                'dataset_name': None,
                'run_name': None,
                'items_created': 0,
                'runs_created': 0,
                'scores_uploaded': 0,
                'scores_skipped': 0,
                'errors': ['Langfuse client not initialized'],
            }

        stats: Dict[str, Any] = {
            'dataset_name': None,
            'run_name': None,
            'items_created': 0,
            'runs_created': 0,
            'scores_uploaded': 0,
            'scores_skipped': 0,
            'errors': [],
        }

        # Determine dataset name
        if dataset_name is None:
            if evaluation_result.evaluation_name:
                dataset_name = evaluation_result.evaluation_name
            else:
                dataset_name = f'axion-eval-{evaluation_result.run_id[:8]}'

        # Determine run name
        if run_name is None:
            run_name = f'{dataset_name}-{evaluation_result.run_id[:8]}'

        stats['dataset_name'] = dataset_name
        stats['run_name'] = run_name

        # Merge default_tags with per-call tags
        effective_tags = list(self.default_tags)
        if tags:
            effective_tags.extend(tags)

        # Create or get dataset (create_dataset upserts if exists)
        try:
            self._execute_with_retry(
                lambda: self.client.create_dataset(name=dataset_name),
                description=f'create dataset {dataset_name}',
            )
            logger.info(f'Created/retrieved dataset: {dataset_name}')
        except Exception as e:
            error_msg = f'Failed to create dataset {dataset_name}: {e}'
            logger.error(error_msg)
            stats['errors'].append(error_msg)
            return stats

        # Build a map of item_id -> (test_result, input_data, actual_output)
        # for lookup when iterating over dataset items
        test_result_map: Dict[str, Any] = {}

        # Phase 1: Create all dataset items
        for test_result in evaluation_result.results:
            if not test_result.test_case:
                continue

            item = test_result.test_case
            item_id = getattr(item, 'id', None)
            if not item_id:
                continue

            # Serialize input and expected output
            input_data = self._serialize_dataset_item_input(item)
            expected_output = self._serialize_expected_output(item)

            # Get actual output
            actual_output = (
                getattr(item, 'actual_output', None)
                if hasattr(item, 'actual_output')
                else None
            )

            # Store for later lookup
            test_result_map[item_id] = {
                'test_result': test_result,
                'input_data': input_data,
                'actual_output': actual_output,
            }

            # Build metadata for the dataset item
            item_metadata: Dict[str, Any] = {}
            if hasattr(item, 'trace_id') and item.trace_id:
                item_metadata['trace_id'] = item.trace_id
            if hasattr(item, 'observation_id') and item.observation_id:
                item_metadata['observation_id'] = item.observation_id
            if hasattr(item, 'metadata') and item.metadata:
                item_metadata['original_metadata'] = item.metadata

            # Create dataset item
            try:
                self._execute_with_retry(
                    lambda item_id=item_id,
                    input_data=input_data,
                    expected_output=expected_output,
                    item_metadata=item_metadata: self.client.create_dataset_item(
                        dataset_name=dataset_name,
                        id=item_id,
                        input=input_data,
                        expected_output=expected_output,
                        metadata=item_metadata if item_metadata else None,
                    ),
                    description=f'create dataset item {item_id}',
                )
                stats['items_created'] += 1

                if self.request_pacing > 0:
                    time.sleep(self.request_pacing)

            except Exception as e:
                error_str = str(e).lower()
                # Handle "already exists" gracefully - item will be reused
                if 'already exists' in error_str or 'duplicate' in error_str:
                    logger.debug(f'Dataset item {item_id} already exists, reusing')
                else:
                    error_msg = f'Failed to create dataset item {item_id}: {e}'
                    logger.warning(error_msg)
                    stats['errors'].append(error_msg)
                    # Remove from map so we don't try to create a run for it
                    test_result_map.pop(item_id, None)

        # Flush to ensure all items are created before fetching
        try:
            self._execute_with_retry(
                lambda: self.client.flush(),
                description='flush after creating dataset items',
            )
        except Exception as e:
            logger.warning(f'Failed to flush after creating items: {e}')

        # Phase 2: Get dataset and create runs using item.run() context manager
        try:
            dataset = self._execute_with_retry(
                lambda: self.client.get_dataset(name=dataset_name),
                description=f'get dataset {dataset_name}',
            )
        except Exception as e:
            error_msg = f'Failed to get dataset {dataset_name}: {e}'
            logger.error(error_msg)
            stats['errors'].append(error_msg)
            return stats

        # Build run metadata
        full_run_metadata = {
            'axion_run_id': evaluation_result.run_id,
            **(run_metadata or {}),
        }
        if effective_tags:
            full_run_metadata['tags'] = effective_tags

        # Iterate over dataset items and create runs
        for dataset_item in dataset.items:
            item_id = dataset_item.id
            if item_id not in test_result_map:
                # This item wasn't part of our evaluation
                continue

            cached = test_result_map[item_id]
            test_result = cached['test_result']
            actual_output = cached['actual_output']

            try:
                # Use item.run() context manager to create the experiment run
                with dataset_item.run(
                    run_name=run_name,
                    run_metadata=full_run_metadata,
                ) as root_span:
                    # Update trace with output
                    if actual_output is not None:
                        root_span.update_trace(output=actual_output)

                    stats['runs_created'] += 1

                    # Add scores using score_trace
                    for metric_score in test_result.score_results:
                        if metric_score.score is None or np.isnan(metric_score.score):
                            stats['scores_skipped'] += 1
                            continue

                        try:
                            root_span.score_trace(
                                name=metric_score.name,
                                value=float(metric_score.score),
                                comment=self._format_comment(
                                    metric_score.explanation, metric_score.signals
                                ),
                            )
                            stats['scores_uploaded'] += 1

                        except Exception as e:
                            error_msg = (
                                f'Failed to upload score {metric_score.name} '
                                f'for item {item_id}: {e}'
                            )
                            logger.warning(error_msg)
                            stats['errors'].append(error_msg)
                            stats['scores_skipped'] += 1

                if self.request_pacing > 0:
                    time.sleep(self.request_pacing)

            except Exception as e:
                error_msg = f'Failed to create experiment run for item {item_id}: {e}'
                logger.warning(error_msg)
                stats['errors'].append(error_msg)

        # Flush client
        if flush:
            try:
                self._execute_with_retry(
                    lambda: self.client.flush(),
                    description='flush Langfuse client',
                )
            except Exception as e:
                logger.warning(f'Failed to flush Langfuse client: {e}')

        logger.info(
            f'Langfuse experiment upload complete: dataset={dataset_name}, '
            f'run={run_name}, items={stats["items_created"]}, '
            f'runs={stats["runs_created"]}, scores={stats["scores_uploaded"]} uploaded, '
            f'{stats["scores_skipped"]} skipped'
        )

        return stats
