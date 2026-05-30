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
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, TypeVar

import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

from axion._core.logging import get_logger
from axion._core.tracing.loaders.base import BaseTraceLoader
from axion.metrics.schema import DEFAULT_EXPLANATION

T = TypeVar('T')

if TYPE_CHECKING:
    from axion.dataset import Dataset, DatasetItem
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
        timeout: Optional[int] = None,
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
            timeout: Timeout in seconds for API requests. Default: 30 seconds.
                Falls back to LANGFUSE_TIMEOUT env var. The Langfuse SDK default
                of 5s is often too short for fetching large traces.
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.request_pacing = request_pacing

        # Determine timeout: param > env var > default of 30s
        if timeout is None:
            env_timeout = os.environ.get('LANGFUSE_TIMEOUT')
            if env_timeout:
                timeout = int(env_timeout)
            else:
                timeout = 30  # Higher default than Langfuse SDK's 5s
        self.timeout = timeout

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
                timeout=timeout,
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

    def _normalize_metric_names(
        self, metric_names: Optional[List[str]]
    ) -> Optional[set[str]]:
        """
        Normalize a metric name allowlist for filtering score uploads.

        Args:
            metric_names: Optional list of metric names to allow. If provided,
                only matching metric names are uploaded.

        Returns:
            A set of normalized metric names, or None if no filtering is applied.
        """
        if metric_names is None:
            return None

        normalized = {name.strip() for name in metric_names if name and name.strip()}
        return normalized

    def _should_upload_metric(
        self,
        metric_score: Any,
        metric_name_filter: Optional[set[str]],
        stats: Dict[str, int],
        skipped_key: str,
    ) -> bool:
        """
        Determine whether a metric score should be uploaded.

        Increments the provided skipped counter when a score is filtered out.
        """
        if metric_name_filter is not None and (
            metric_score.name not in metric_name_filter
        ):
            stats[skipped_key] += 1
            return False

        if metric_score.score is None or np.isnan(metric_score.score):
            stats[skipped_key] += 1
            return False

        return True

    def _flush_client(self) -> None:
        """Flush the Langfuse client with retry."""
        try:
            self._execute_with_retry(
                lambda: self.client.flush(),
                description='flush Langfuse client',
            )
        except Exception as e:
            logger.warning(f'Failed to flush Langfuse client: {e}')

    def push_scores(
        self,
        evaluation_result: 'EvaluationResult',
        observation_id_field: Optional[str] = 'observation_id',
        flush: bool = True,
        tags: Optional[List[str]] = None,
        metric_names: Optional[List[str]] = None,
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
            metric_names: Optional list of metric names to upload. If provided,
                only scores whose metric name matches are uploaded.

        Returns:
            Dict with counts: {'uploaded': N, 'skipped': M}
        """
        return self.push_scores_to_langfuse(
            evaluation_result=evaluation_result,
            observation_id_field=observation_id_field,
            flush=flush,
            tags=tags,
            metric_names=metric_names,
        )

    def fetch_traces(
        self,
        limit: int = 50,
        mode: Literal['days_back', 'absolute', 'hours_back'] = 'days_back',
        days_back: int = 7,
        hours_back: int = 24,
        from_timestamp: datetime | str | None = None,
        to_timestamp: datetime | str | None = None,
        tags: Optional[List[str]] = None,
        name: Optional[str] = None,
        trace_ids: Optional[List[str]] = None,
        fetch_full_traces: bool = True,
        show_progress: bool = True,
        **trace_list_kwargs: Any,
    ) -> List[Any]:
        """
        Fetch raw traces from Langfuse.

        Args:
            limit: Maximum number of traces to fetch
            mode: Time window mode to use for fetching traces. Options:
                'days_back', 'hours_back', or 'absolute'.
            days_back: Number of days to look back (days_back mode)
            hours_back: Number of hours to look back (hours_back mode)
            from_timestamp: Start timestamp for absolute mode (datetime or ISO string)
            to_timestamp: End timestamp for absolute mode (datetime or ISO string)
            tags: Filter by specific tags
            name: Filter by trace name
            trace_ids: Optional list of trace IDs to fetch directly. If provided,
                skips list() and fetches each trace by ID.
            fetch_full_traces: If True (default), fetch full trace details for each
                trace via additional API calls. If False, only return trace summaries
                (faster but less data). Set to False to avoid rate limits.
            show_progress: If True, display a tqdm progress bar when fetching full traces.
            **trace_list_kwargs: Additional kwargs passed to
                langfuse_client.api.trace.list(...)

        Returns:
            List of raw Langfuse trace objects (full traces or summaries)
        """
        if not self._client_initialized:
            logger.error('Langfuse client not initialized')
            return []

        def _fetch_full_traces(trace_id_list: List[str]) -> List[Any]:
            results = []
            trace_iter = trace_id_list
            if show_progress:
                trace_iter = tqdm(trace_iter, desc='Fetching traces', unit='trace')

            for trace_id in trace_iter:
                if not trace_id:
                    continue
                trace = self.fetch_trace(trace_id)
                if trace is not None:
                    results.append(trace)

            logger.info(f'Successfully loaded {len(results)} full traces')
            return results

        if trace_ids:
            return _fetch_full_traces(trace_ids)

        if mode not in {'days_back', 'hours_back', 'absolute'}:
            logger.error(
                f'Invalid mode "{mode}". Use "days_back", "hours_back", or "absolute".'
            )
            return []

        def _parse_timestamp(
            value: datetime | str | None, label: str
        ) -> datetime | None:
            if value is None:
                return None
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError as exc:
                    raise ValueError(
                        f'Invalid {label} format. Use ISO-8601, e.g. '
                        f'2026-01-01T12:34:56+00:00.'
                    ) from exc
            raise ValueError(f'Invalid {label} type: {type(value).__name__}.')

        list_kwargs = dict(trace_list_kwargs)

        for key, value in {
            'limit': limit,
            'tags': tags,
            'name': name,
            'from_timestamp': from_timestamp,
            'to_timestamp': to_timestamp,
        }.items():
            if value is not None and key in list_kwargs:
                raise ValueError(f'Duplicate value provided for "{key}".')

        if mode != 'absolute':
            if from_timestamp is not None or to_timestamp is not None:
                raise ValueError('from_timestamp/to_timestamp require mode="absolute".')
            if mode == 'days_back':
                from_ts = self._normalize_time_window(days_back)
            else:
                from_ts = datetime.now() - timedelta(hours=hours_back)
            to_ts = None
        else:
            resolved_from = _parse_timestamp(
                from_timestamp or list_kwargs.get('from_timestamp'), 'from_timestamp'
            )
            resolved_to = _parse_timestamp(
                to_timestamp or list_kwargs.get('to_timestamp'), 'to_timestamp'
            )

            if resolved_from is None:
                raise ValueError('absolute mode requires from_timestamp.')

            if resolved_to and resolved_from > resolved_to:
                raise ValueError('from_timestamp cannot be after to_timestamp.')

            from_ts = resolved_from
            to_ts = resolved_to

        list_kwargs.pop('from_timestamp', None)
        list_kwargs.pop('to_timestamp', None)

        if 'limit' not in list_kwargs:
            list_kwargs['limit'] = limit
        if tags is not None:
            list_kwargs['tags'] = tags
        if name is not None:
            list_kwargs['name'] = name

        list_kwargs['from_timestamp'] = from_ts
        if to_ts is not None:
            list_kwargs['to_timestamp'] = to_ts

        window_suffix = (
            f'between {from_ts} and {to_ts}' if to_ts else f'since {from_ts}'
        )
        logger.info(
            f'Fetching up to {list_kwargs["limit"]} traces from Langfuse '
            f'({window_suffix})...'
        )

        # Fetch trace list with retry
        try:
            traces_page = self._execute_with_retry(
                lambda: self.client.api.trace.list(**list_kwargs),
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
        return _fetch_full_traces([trace.id for trace in traces_page.data])

    def fetch_trace(self, trace_id: str, pace: bool = True) -> Optional[Any]:
        """
        Fetch a single trace by trace_id.

        Args:
            trace_id: Langfuse trace ID to fetch
            pace: Whether to apply request pacing delay after fetch

        Returns:
            The full Langfuse trace object, or None if not found/failed.
        """
        if not self._client_initialized:
            logger.error('Langfuse client not initialized')
            return None

        if not trace_id:
            logger.warning('fetch_trace called with empty trace_id')
            return None

        try:
            trace = self._execute_with_retry(
                lambda: self.client.api.trace.get(trace_id),
                description=f'fetch trace {trace_id}',
            )

            if pace and self.request_pacing > 0:
                time.sleep(self.request_pacing)

            return trace
        except Exception as e:
            logger.warning(f'Failed to fetch trace {trace_id}: {e}')
            return None

    def fetch_session(self, session_id: str, pace: bool = True) -> Optional[Any]:
        """
        Fetch a session by session_id.

        Args:
            session_id: Langfuse session ID to fetch
            pace: Whether to apply request pacing delay after fetch

        Returns:
            The Langfuse session object (contains traces list), or None if not found/failed.
        """
        if not self._client_initialized:
            logger.error('Langfuse client not initialized')
            return None

        if not session_id:
            logger.warning('fetch_session called with empty session_id')
            return None

        try:
            session = self._execute_with_retry(
                lambda: self.client.api.sessions.get(session_id),
                description=f'fetch session {session_id}',
            )

            if pace and self.request_pacing > 0:
                time.sleep(self.request_pacing)

            return session
        except Exception as e:
            logger.warning(f'Failed to fetch session {session_id}: {e}')
            return None

    def get_session_traces(
        self,
        session_id: str,
        show_progress: bool = True,
    ) -> List[Any]:
        """
        Fetch all full trace data for a session.

        Retrieves the session by ID, then fetches full trace details
        for every trace in the session. Traces whose full fetch fails are
        dropped (see :meth:`get_session_with_traces` to retain stubs instead).

        Args:
            session_id: Langfuse session ID
            show_progress: Whether to show a progress bar while fetching traces

        Returns:
            List of full Langfuse trace objects, or empty list on failure.
        """
        _, traces = self.get_session_with_traces(
            session_id, show_progress=show_progress, retain_stub_on_failure=False
        )
        return traces

    def get_session_with_traces(
        self,
        session_id: str,
        show_progress: bool = True,
        retain_stub_on_failure: bool = True,
        enrich: bool = True,
    ) -> tuple[Optional[Any], List[Any]]:
        """
        Fetch a session along with trace details for each of its traces.

        Retrieves the session by ID (preserving session-level metadata) and,
        when ``enrich`` is True, fetches full trace details for every trace.

        Args:
            session_id: Langfuse session ID
            show_progress: Whether to show a progress bar while fetching traces
            retain_stub_on_failure: When a full ``fetch_trace`` fails, keep the
                session's stub trace instead of dropping it. Stubs still carry
                trace-level input/output (so a conversation turn is preserved),
                only observation-level detail is missing. Defaults to ``True``.
            enrich: When ``True`` (default), fetch full trace details (with
                observations) via one ``fetch_trace`` per trace. When ``False``,
                skip enrichment and return the session's stub traces directly
                (a single API call). Stubs carry trace-level input/output, so the
                conversation is still reconstructable, but observation-level data
                (``by_type``/``tools``/``find_all``) will be empty.

        Returns:
            A ``(session, traces)`` tuple. ``session`` is the raw Langfuse
            session object (or ``None`` if it could not be fetched), and
            ``traces`` is the list of full (or, when not enriching / on failure,
            stub) trace objects.
        """
        session = self.fetch_session(session_id)
        if session is None:
            return None, []

        stubs = getattr(session, 'traces', None)
        if not stubs:
            logger.info(f'Session {session_id} has no traces')
            return session, []

        if not enrich:
            logger.info(
                f'Skipping enrichment for session {session_id}; '
                f'using {len(stubs)} stub traces (trace-level I/O only).'
            )
            return session, list(stubs)

        stub_by_id = {t.id: t for t in stubs if getattr(t, 'id', None)}
        trace_ids = list(stub_by_id.keys())
        logger.info(
            f'Fetching {len(trace_ids)} full traces for session {session_id}...'
        )

        results = []
        trace_iter = trace_ids
        if show_progress:
            trace_iter = tqdm(trace_iter, desc='Fetching session traces', unit='trace')

        for tid in trace_iter:
            trace = self.fetch_trace(tid)
            if trace is not None:
                results.append(trace)
            elif retain_stub_on_failure:
                logger.warning(
                    f'Full fetch failed for trace {tid}; retaining session stub.'
                )
                results.append(stub_by_id[tid])

        logger.info(
            f'Successfully loaded {len(results)} traces for session {session_id}'
        )
        return session, results

    def push_scores_to_langfuse(
        self,
        evaluation_result: 'EvaluationResult',
        observation_id_field: Optional[str] = 'observation_id',
        flush: bool = True,
        tags: Optional[List[str]] = None,
        metric_names: Optional[List[str]] = None,
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
            metric_names: Optional list of metric names to upload. If provided,
                only scores whose metric name matches are uploaded.

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
        metric_name_filter = self._normalize_metric_names(metric_names)

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
                if not self._should_upload_metric(
                    metric_score, metric_name_filter, stats, 'skipped'
                ):
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
            self._flush_client()

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
        score_on_runtime_traces: bool = False,
        link_to_traces: bool = False,
        metric_names: Optional[List[str]] = None,
        update_existing_items: bool = False,
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
            score_on_runtime_traces: If True, do NOT create per-dataset-item \"Dataset run\" traces.
                Instead, attach scores directly to existing runtime traces (via DatasetItem.trace_id
                and optional observation_id). This requires existing traces. Takes precedence over
                link_to_traces if both are True.
            link_to_traces: If True, link experiment runs to existing traces via the
                low-level API (client.api.dataset_run_items.create()) instead of creating
                new \"Dataset run\" traces. This allows experiment runs to appear linked
                to the original evaluation traces in Langfuse UI. Falls back to creating
                new traces if trace_id is not available on the test case. Ignored if
                score_on_runtime_traces is True.
            metric_names: Optional list of metric names to upload. If provided,
                only scores whose metric name matches are uploaded.
            update_existing_items: If False (default), skip the
                ``create_dataset_item`` upsert for items whose ``id`` already
                exists on the dataset. The input shape produced by
                ``_serialize_dataset_item_input`` is lossy (it re-wraps
                ``additional_input`` under a nested key) and would otherwise
                overwrite user-curated input on every push. Set True only when
                bootstrapping a brand-new dataset from evaluation results.

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
        metric_name_filter = self._normalize_metric_names(metric_names)

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

        # When `update_existing_items=False` we must skip the
        # `create_dataset_item` upsert for items that already exist — otherwise
        # the serialized-from-runtime input shape clobbers user-curated input
        # on every push. Pre-fetch the existing item ids once so the loop
        # below can branch cheaply.
        existing_item_ids: set[str] = set()
        if not update_existing_items:
            try:
                existing_dataset = self.client.get_dataset(name=dataset_name)
                for existing in getattr(existing_dataset, 'items', None) or []:
                    existing_id = getattr(existing, 'id', None)
                    if existing_id:
                        existing_item_ids.add(str(existing_id))
            except Exception as e:
                # If we can't read existing items, fall back to attempting the
                # upsert — better to risk overwriting than to fail the whole
                # run. Log so it's visible.
                logger.warning(
                    f'Could not pre-fetch dataset items for {dataset_name} '
                    f'(falling back to upsert path): {e}'
                )

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

            # Create dataset item — but only when the caller explicitly asked
            # us to bootstrap items, or when the id doesn't already exist on
            # the dataset. See `update_existing_items` arg docstring.
            should_upsert = update_existing_items or item_id not in existing_item_ids
            try:
                if should_upsert:
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

        # Mode B: score existing runtime traces, do not create new \"Dataset run\" traces.
        if score_on_runtime_traces:
            # Upload scores onto the original runtime trace/span IDs.
            # This skips dataset_item.run(...) entirely.
            for test_result in evaluation_result.results:
                if not test_result.test_case:
                    continue

                trace_id = getattr(test_result.test_case, 'trace_id', None)
                obs_id = getattr(test_result.test_case, 'observation_id', None)
                if not trace_id:
                    stats['scores_skipped'] += len(test_result.score_results)
                    continue

                for metric_score in test_result.score_results:
                    if not self._should_upload_metric(
                        metric_score, metric_name_filter, stats, 'scores_skipped'
                    ):
                        continue

                    try:
                        # Prefer per-metric trace/span IDs when available.
                        # This enables attaching the score to the metric's own runtime trace
                        # (e.g., MetricRunner trace_granularity=SEPARATE).
                        metric_meta = getattr(metric_score, 'metadata', None) or {}
                        target_trace_id = metric_meta.get('trace_id') or trace_id
                        target_observation_id = (
                            metric_meta.get('observation_id') or obs_id
                        )

                        score_kwargs: Dict[str, Any] = {
                            'trace_id': target_trace_id,
                            'name': metric_score.name,
                            'value': float(metric_score.score),
                            'comment': self._format_comment(
                                metric_score.explanation, metric_score.signals
                            ),
                            'metadata': {
                                'dataset_name': dataset_name,
                                'run_name': run_name,
                                'axion_run_id': evaluation_result.run_id,
                                'tags': effective_tags,
                                # Keep both IDs for debugging / linking
                                'test_case_trace_id': trace_id,
                                'metric_trace_id': metric_meta.get('trace_id'),
                            },
                        }
                        if target_observation_id:
                            score_kwargs['observation_id'] = target_observation_id

                        self._execute_with_retry(
                            lambda kwargs=score_kwargs: self.client.create_score(
                                **kwargs
                            ),
                            description=f'upload score {metric_score.name} for trace {target_trace_id}',
                        )
                        stats['scores_uploaded'] += 1

                        if self.request_pacing > 0:
                            time.sleep(self.request_pacing)
                    except Exception as e:
                        error_msg = (
                            f'Failed to upload score {metric_score.name} '
                            f'for trace {trace_id}: {e}'
                        )
                        logger.warning(error_msg)
                        stats['errors'].append(error_msg)
                        stats['scores_skipped'] += 1

            if flush:
                self._flush_client()

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
            input_data = cached.get('input_data')
            actual_output = cached['actual_output']

            try:
                # Include per-item linking metadata so the experiment trace can point
                # back to a runtime trace/span when available.
                item_trace_id = getattr(test_result.test_case, 'trace_id', None)
                item_observation_id = getattr(
                    test_result.test_case, 'observation_id', None
                )
                per_item_run_metadata = {
                    **full_run_metadata,
                    'dataset_item_id': item_id,
                    **({'trace_id': item_trace_id} if item_trace_id else {}),
                    **(
                        {'observation_id': item_observation_id}
                        if item_observation_id
                        else {}
                    ),
                }

                # Mode: link_to_traces - link experiment run to existing trace via low-level API
                if link_to_traces and item_trace_id:
                    # Use low-level API to create dataset run item linked to existing trace
                    # Import the request type from langfuse SDK
                    from langfuse.api import CreateDatasetRunItemRequest

                    run_item_request = CreateDatasetRunItemRequest(
                        runName=run_name,
                        datasetItemId=item_id,
                        traceId=item_trace_id,
                        observationId=item_observation_id,
                        metadata=per_item_run_metadata,
                    )
                    self._execute_with_retry(
                        lambda req=run_item_request: (
                            self.client.api.dataset_run_items.create(request=req)
                        ),
                        description=f'create linked dataset run for item {item_id}',
                    )
                    stats['runs_created'] += 1

                    # Attach scores to existing trace
                    for metric_score in test_result.score_results:
                        if not self._should_upload_metric(
                            metric_score, metric_name_filter, stats, 'scores_skipped'
                        ):
                            continue

                        try:
                            item_score_kwargs = {
                                'trace_id': item_trace_id,
                                'name': metric_score.name,
                                'value': float(metric_score.score),
                                'comment': self._format_comment(
                                    metric_score.explanation, metric_score.signals
                                ),
                            }
                            if item_observation_id:
                                item_score_kwargs['observation_id'] = (
                                    item_observation_id
                                )

                            self._execute_with_retry(
                                lambda kwargs=item_score_kwargs: self.client.create_score(
                                    **kwargs
                                ),
                                description=f'upload score {metric_score.name}',
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
                else:
                    with self.client.start_as_current_observation(
                        name=f'dataset-run:{run_name}:{item_id}',
                        as_type='evaluator',
                        input=input_data,
                        output=actual_output,
                    ) as root_span:
                        root_span.set_trace_io(
                            input=input_data,
                            output=actual_output,
                        )
                        trace_id = root_span.trace_id

                    # Linking and scores happen after the span closes so the
                    # observation is flushed and the trace_id is final.
                    try:
                        self._execute_with_retry(
                            lambda: self.client.api.dataset_run_items.create(
                                run_name=run_name,
                                dataset_item_id=item_id,
                                trace_id=trace_id,
                                metadata=per_item_run_metadata,
                            ),
                            description=f'link dataset run for item {item_id}',
                        )
                        stats['runs_created'] += 1
                    except Exception as e:
                        error_msg = (
                            f'Failed to create dataset run item for {item_id}: {e}'
                        )
                        logger.warning(error_msg)
                        stats['errors'].append(error_msg)

                    for metric_score in test_result.score_results:
                        if not self._should_upload_metric(
                            metric_score,
                            metric_name_filter,
                            stats,
                            'scores_skipped',
                        ):
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
                            self._execute_with_retry(
                                lambda kwargs=score_kwargs: self.client.create_score(
                                    **kwargs
                                ),
                                description=f'upload score {metric_score.name}',
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
            self._flush_client()

        logger.info(
            f'Langfuse experiment upload complete: dataset={dataset_name}, '
            f'run={run_name}, items={stats["items_created"]}, '
            f'runs={stats["runs_created"]}, scores={stats["scores_uploaded"]} uploaded, '
            f'{stats["scores_skipped"]} skipped'
        )

        return stats

    @staticmethod
    def _resolve_field(lf_item: Any, spec: Any) -> Any:
        """
        Resolve a field value from a Langfuse dataset item via a spec.

        spec can be:
        - callable: called with the raw Langfuse item; return value used as-is.
        - "source.key" string: looks up lf_item.<source>["key"].
        - plain string (no dot): shorthand for "input.<key>".

        Returns None when the path cannot be resolved.
        """
        if callable(spec):
            return spec(lf_item)
        if not isinstance(spec, str):
            return None
        source, _, key = spec.partition('.')
        if not key:
            container, key = getattr(lf_item, 'input', None), source
        else:
            container = getattr(lf_item, source, None)
        return container.get(key) if isinstance(container, dict) else None

    def _build_dataset_item(
        self, lf_item: Any, field_map: Dict[str, Any]
    ) -> 'DatasetItem':
        """Convert a single Langfuse dataset item to an Axion DatasetItem."""
        from axion.dataset import DatasetItem

        raw_input = getattr(lf_item, 'input', None)
        raw_expected = getattr(lf_item, 'expected_output', None)

        def get(field: str, default: Any) -> Any:
            if field in field_map:
                return self._resolve_field(lf_item, field_map[field])
            return default

        query = get(
            'query', self._extract_query(raw_input) if raw_input is not None else None
        )
        actual_output = get('actual_output', None)

        default_expected = (
            str(raw_expected['expected_output'])
            if isinstance(raw_expected, dict) and 'expected_output' in raw_expected
            else self._extract_output(raw_expected)
            if raw_expected is not None
            else None
        )
        expected_output = get('expected_output', default_expected)

        # `lf_item.expected_output` may be a wrapper dict like
        # `{"expected_output": "...", "additional_output": {...}}` because
        # Langfuse's dataset UI requires JSON in the Expected Output box and
        # does not accept a bare string. We pull `additional_output` from the
        # wrapper (preferred) or, failing that, treat any leftover dict keys
        # as additional_output — symmetric with how `additional_input` is
        # derived from leftover `lf_item.input` keys.
        if isinstance(raw_expected, dict):
            if 'additional_output' in raw_expected and isinstance(
                raw_expected['additional_output'], dict
            ):
                default_additional_output = raw_expected['additional_output']
            else:
                default_additional_output = {
                    k: v
                    for k, v in raw_expected.items()
                    if k not in {'expected_output', 'additional_output'}
                }
        else:
            default_additional_output = {}
        additional_output_val = get('additional_output', default_additional_output)
        additional_output = (
            additional_output_val if isinstance(additional_output_val, dict) else {}
        )

        default_rc = (
            raw_input['retrieved_content']
            if isinstance(raw_input, dict) and 'retrieved_content' in raw_input
            else None
        )
        rc_val = get('retrieved_content', default_rc)
        retrieved_content = (
            rc_val
            if isinstance(rc_val, list)
            else [str(rc_val)]
            if rc_val is not None
            else None
        )

        trace_id = get('trace_id', getattr(lf_item, 'source_trace_id', None))
        observation_id = get('observation_id', None)
        latency = get('latency', None)

        # Default `additional_input` to the input-dict keys we didn't already
        # promote to a typed slot. Without this, agent-specific fields living
        # in `lf_item.input` (e.g., case_id, quote_locator) are silently dropped
        # when callers go through `load_langfuse_dataset`. A field_map entry
        # overrides this default if the caller wants finer control.
        if isinstance(raw_input, dict):
            reserved = {'query', 'retrieved_content', 'expected_output'}
            default_additional = {
                k: v for k, v in raw_input.items() if k not in reserved
            }
        else:
            default_additional = {}
        additional_input_val = get('additional_input', default_additional)
        additional_input = (
            additional_input_val if isinstance(additional_input_val, dict) else {}
        )

        metadata = getattr(lf_item, 'metadata', None)
        metadata_str: Optional[str] = None
        if metadata is not None:
            try:
                metadata_str = json.dumps(metadata, default=str)
            except Exception:
                metadata_str = str(metadata)

        return DatasetItem(
            dataset_id=getattr(lf_item, 'id', None),
            query=query,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieved_content=retrieved_content,
            additional_input=additional_input,
            additional_output=additional_output,
            trace_id=trace_id,
            observation_id=observation_id,
            latency=latency,
            dataset_metadata=metadata_str,
        )

    def load_langfuse_dataset(
        self,
        dataset_name: str,
        dataset_item_name: Optional[str] = None,
        axion_dataset_name: Optional[str] = None,
        field_map: Optional[Dict[str, Any]] = None,
    ) -> 'Dataset':
        """
        Load a Langfuse dataset by name and convert its items to an Axion Dataset.

        Args:
            dataset_name: Name of the Langfuse dataset to load.
            dataset_item_name: Optional filter — only include items whose ``id``
                or ``source_trace_id`` matches this value.
            axion_dataset_name: Name for the returned Axion ``Dataset``.
                Defaults to ``dataset_name``.
            field_map: Optional mapping from Axion ``DatasetItem`` field names to
                a dot-path string or callable. Unmapped fields fall back to default
                extraction logic.

                Supported fields: ``query``, ``actual_output``, ``expected_output``,
                ``retrieved_content``, ``additional_input``, ``additional_output``,
                ``trace_id``, ``observation_id``, ``latency``.

                ``additional_input`` defaults to the keys of ``lf_item.input`` that
                weren't promoted to a typed slot (``query`` / ``retrieved_content`` /
                ``expected_output``), so agent-specific input fields are preserved
                without an explicit mapping.

                ``additional_output`` is populated when ``lf_item.expected_output``
                is a dict — either from an explicit ``additional_output`` key inside
                that dict, or from leftover keys after ``expected_output`` is
                promoted. This lets callers stash structured expectations alongside
                the primary expected string (Langfuse's dataset UI requires JSON,
                not bare strings, in the Expected Output box).

                Dot-path — ``"<source>.<key>"`` where source is ``input`` or
                ``expected_output``; a plain key (no dot) defaults to ``input``::

                    field_map={
                        "query": "input.case_id",
                        "trace_id": "input.correlation_id",
                    }

                Callable — receives the raw Langfuse item::

                    field_map={
                        "query": lambda item: (item.input or {}).get("case_id"),
                    }

        Returns:
            Axion ``Dataset`` ready to pass to ``evaluation_runner``.
        """
        from axion.dataset import Dataset

        field_map = field_map or {}

        if not self._client_initialized:
            logger.error('Langfuse client not initialized')
            return Dataset(name=axion_dataset_name or dataset_name)

        try:
            lf_dataset = self._execute_with_retry(
                lambda: self.client.get_dataset(name=dataset_name),
                description=f'get dataset {dataset_name}',
            )
        except Exception as e:
            logger.error(f'Failed to load Langfuse dataset "{dataset_name}": {e}')
            return Dataset(name=axion_dataset_name or dataset_name)

        lf_items = lf_dataset.items
        if dataset_item_name is not None:
            lf_items = [
                item
                for item in lf_items
                if dataset_item_name
                in (
                    getattr(item, 'id', None) or '',
                    getattr(item, 'source_trace_id', None) or '',
                )
            ]

        items = []
        for lf_item in lf_items:
            items.append(self._build_dataset_item(lf_item, field_map))
            if self.request_pacing > 0:
                time.sleep(self.request_pacing)

        logger.info(f'Loaded {len(items)} items from Langfuse dataset "{dataset_name}"')
        return Dataset(name=axion_dataset_name or dataset_name, items=items)
