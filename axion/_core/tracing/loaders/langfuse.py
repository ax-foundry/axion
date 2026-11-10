"""
Langfuse trace loader for Axion.

This module provides functionality to:
1. Fetch traces from Langfuse
2. Convert them to Axion Dataset format
3. Push evaluation scores back to Langfuse traces

Traces, sessions and scores are read through Langfuse's observations and scores
reads and rebuilt into the SDK's trace and session objects; the rules for that
live in :mod:`axion._core.tracing.loaders.langfuse_rows`.
"""

import json
import os
import random
import secrets
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, TypeVar

import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

from axion._core.logging import get_logger
from axion._core.tracing.collection.scores import TraceScore
from axion._core.tracing.loaders.base import BaseTraceLoader
from axion._core.tracing.loaders.langfuse_experiments import (
    EXPERIMENT_ITEM_ROOT_OBSERVATION_ID,
    experiment_id_for,
    experiment_item_attributes,
)
from axion._core.tracing.loaders.langfuse_rows import (
    ALL_TIME,
    CLOCK_SLACK,
    EPOCH,
    FULL_FIELDS,
    LIGHT_FIELDS,
    OBSERVATIONS_PAGE_LIMIT,
    SCORE_FIELDS,
    SCORE_LOOKBACK,
    SCORES_PAGE_LIMIT,
    TraceListing,
    TraceReadTruncated,
    TraceSummary,
    caller_utc,
    full_trace_payload,
    group_by_trace,
    pick_root,
    possibly_cut,
    score_from_v3,
    trace_payload,
    trace_start,
    utc,
)
from axion.metrics.schema import DEFAULT_EXPLANATION

T = TypeVar('T')

if TYPE_CHECKING:
    from axion.dataset import Dataset, DatasetItem
    from axion.schema import EvaluationResult

logger = get_logger(__name__)

__all__ = [
    'LangfuseTraceLoader',
    'TraceListing',
    'TraceReadTruncated',
    'TraceSummary',
]


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

    # Pages one trace, session or score read may take before it refuses a partial
    # result.
    max_pages = 100
    # Pages one window of a listing may take before the window is halved.
    list_max_pages = 20
    # A listing window is never halved below this.
    min_list_window = timedelta(seconds=1)

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

    @staticmethod
    def _observation_filter_str(tags: Optional[List[str]]) -> Optional[str]:
        """Build the observations API filter for trace-level tags."""
        filters = []
        for tag in tags or []:
            filters.append(
                {
                    'type': 'arrayOptions',
                    'column': 'traceTags',
                    'operator': 'any of',
                    'value': [tag],
                }
            )

        return json.dumps(filters) if filters else None

    def _fetch_trace_id_page_via_observations(
        self,
        batch_size: int,
        environment: Optional[str],
        tags: Optional[List[str]],
        from_ts: datetime,
        to_ts: Optional[datetime],
        cursor: Optional[str],
    ) -> tuple[List[str], Optional[str]]:
        """Fetch one observations page and return unique trace IDs plus cursor."""
        try:
            response = self._execute_with_retry(
                lambda: self.client.api.observations.get_many(
                    limit=batch_size,
                    cursor=cursor,
                    environment=environment,
                    from_start_time=from_ts,
                    to_start_time=to_ts,
                    filter=self._observation_filter_str(tags),
                ),
                description='list observations v2',
            )
        except Exception as e:
            logger.error(f'Failed to fetch observations from Langfuse: {e}')
            return [], None

        page_ids: List[str] = []
        page_seen: set = set()
        for obs in response.data:
            tid = obs.trace_id
            if tid and tid not in page_seen:
                page_seen.add(tid)
                page_ids.append(tid)

        # An empty page can still carry a cursor; the caller stops when it is unset.
        return page_ids, response.meta.cursor

    def _fetch_trace_ids_via_observations(
        self,
        limit: int,
        environment: Optional[str],
        tags: Optional[List[str]],
        from_ts: datetime,
        to_ts: Optional[datetime],
    ) -> List[str]:
        """
        Collect unique trace IDs via the v2 observations endpoint.

        Paginates with cursor until `limit` unique trace IDs are collected
        or the API has no more results.
        """
        seen_ids: List[str] = []
        seen_set: set = set()
        cursor: Optional[str] = None

        while len(seen_ids) < limit:
            batch_size = min(100, max(10, (limit - len(seen_ids)) * 5))
            page_ids, next_cursor = self._fetch_trace_id_page_via_observations(
                batch_size=batch_size,
                environment=environment,
                tags=tags,
                from_ts=from_ts,
                to_ts=to_ts,
                cursor=cursor,
            )

            for tid in page_ids:
                if tid and tid not in seen_set:
                    seen_set.add(tid)
                    seen_ids.append(tid)
                    if len(seen_ids) >= limit:
                        break

            if not next_cursor:
                break
            cursor = next_cursor

        return seen_ids[:limit]

    # -- paging ----------------------------------------------------------------

    def _page(
        self, call: Callable[[str | None], Any], what: str, max_pages: int
    ) -> list[Any]:
        """Every page of a cursor read, or :class:`TraceReadTruncated` past ``max_pages``."""
        rows: list[Any] = []
        cursor: str | None = None
        for _ in range(max_pages):
            response = self._execute_with_retry(
                lambda c=cursor: call(c), description=what
            )
            data = list(response.data or [])
            rows.extend(data)
            cursor = response.meta.cursor
            # A page can come back empty with a cursor still set; only a missing
            # cursor ends the read.
            if not cursor:
                return rows
        raise TraceReadTruncated(
            f'{what} needed more than {max_pages} pages; refusing a partial read'
        )

    def _observations(
        self,
        what: str,
        *,
        fields: str = FULL_FIELDS,
        max_pages: int | None = None,
        **filters: Any,
    ) -> list[Any]:
        """Observation rows matching ``filters``, with any cut metadata read whole."""
        now = datetime.now(UTC)
        window = {
            'from_start_time': now - ALL_TIME,
            'to_start_time': now + CLOCK_SLACK,
            **filters,
        }

        def read(extra: dict[str, Any]) -> list[Any]:
            return self._page(
                lambda cursor: self.client.api.observations.get_many(
                    limit=OBSERVATIONS_PAGE_LIMIT, cursor=cursor, **window, **extra
                ),
                what,
                max_pages or self.max_pages,
            )

        rows = read({'fields': fields})
        cut = sorted(
            {
                key
                for r in rows
                if isinstance(r.metadata, dict)
                for key, value in r.metadata.items()
                if possibly_cut(value)
            }
        )
        if not cut:
            return rows
        full = {
            r.id: r.metadata
            for r in read({'fields': 'core,metadata', 'expand_metadata': ','.join(cut)})
        }
        return [
            r.model_copy(update={'metadata': full[r.id]}) if r.id in full else r
            for r in rows
        ]

    def _score_rows(self, since: datetime, what: str, **filters: Any) -> list[Any]:
        """Score rows matching ``filters``, from a day before ``since`` to now."""
        to = datetime.now(UTC) + CLOCK_SLACK
        return self._page(
            lambda cursor: self.client.api.scores_v3.get_many_v3(
                fields=SCORE_FIELDS,
                limit=SCORES_PAGE_LIMIT,
                cursor=cursor,
                from_timestamp=utc(since) - SCORE_LOOKBACK,
                to_timestamp=to,
                **filters,
            ),
            what,
            self.max_pages,
        )

    def _pace(self, pace: bool) -> None:
        if pace and self.request_pacing > 0:
            time.sleep(self.request_pacing)

    # -- window listing --------------------------------------------------------

    def _window_rows(
        self,
        start: datetime,
        end: datetime,
        name: str | None,
        tags: list[str] | None,
        environment: str | None,
    ) -> list[Any]:
        conditions: list[dict[str, Any]] = [
            {
                'type': 'datetime',
                'column': 'startTime',
                'operator': '>=',
                'value': start.isoformat(),
            },
            {
                'type': 'datetime',
                'column': 'startTime',
                'operator': '<',
                'value': end.isoformat(),
            },
        ]
        if name:
            conditions.append(
                {
                    'type': 'string',
                    'column': 'traceName',
                    'operator': '=',
                    'value': name,
                }
            )
        if tags:
            conditions.append(
                {
                    'type': 'arrayOptions',
                    'column': 'tags',
                    'operator': 'all of',
                    'value': tags,
                }
            )
        if environment:
            conditions.append(
                {
                    'type': 'stringOptions',
                    'column': 'environment',
                    'operator': 'any of',
                    'value': [environment],
                }
            )
        return self._page(
            lambda cursor: self.client.api.observations.get_many(
                fields=LIGHT_FIELDS,
                limit=OBSERVATIONS_PAGE_LIMIT,
                cursor=cursor,
                from_start_time=start,
                to_start_time=end,
                filter=json.dumps(conditions),
            ),
            f'list traces {start.isoformat()}..{end.isoformat()}',
            self.list_max_pages,
        )

    def _rows_before(self, trace_ids: list[str], start: datetime) -> list[Any]:
        """
        The rows of ``trace_ids`` that start before ``start``, 100 ids a read.

        A trace straddling the window's start (an earlier root, or a retry's earlier
        flagged root) has rows there; its timestamp needs them.
        """
        rows: list[Any] = []
        for i in range(0, len(trace_ids), 100):
            batch = trace_ids[i : i + 100]
            conditions = [
                {
                    'type': 'stringOptions',
                    'column': 'traceId',
                    'operator': 'any of',
                    'value': batch,
                },
                {
                    'type': 'datetime',
                    'column': 'startTime',
                    'operator': '<',
                    'value': start.isoformat(),
                },
            ]
            rows += self._page(
                lambda cursor, c=conditions: self.client.api.observations.get_many(
                    fields=LIGHT_FIELDS,
                    limit=OBSERVATIONS_PAGE_LIMIT,
                    cursor=cursor,
                    from_start_time=start - ALL_TIME,
                    to_start_time=start,
                    filter=json.dumps(c),
                ),
                f'read rows before {start.isoformat()}',
                self.max_pages,
            )
        return rows

    def _summaries(self, rows: list[Any], start: datetime) -> list[TraceSummary]:
        in_window = group_by_trace(rows)
        earlier = group_by_trace(self._rows_before(list(in_window), start))
        summaries = []
        for trace_id, group in in_window.items():
            group = earlier.get(trace_id, []) + group
            root = pick_root(group)
            summaries.append(
                TraceSummary(
                    id=trace_id,
                    timestamp=trace_start(group) or EPOCH,
                    name=root.trace_name or root.name,
                    session_id=root.session_id,
                    tags=sorted(root.tags or []),
                )
            )
        return summaries

    def list_trace_ids(
        self,
        start: datetime,
        end: datetime,
        *,
        name: str | None = None,
        tags: list[str] | None = None,
        environment: str | None = None,
        max_traces: int,
        order: Literal['asc', 'desc'] = 'asc',
    ) -> TraceListing:
        """
        List the traces whose trace timestamp falls in ``[start, end)``.

        A window whose rows need more than ``list_max_pages`` pages is halved (keeping
        its older half for ``asc``, its newer half for ``desc``) until it is read whole.

        ``asc`` (oldest first): with more than ``max_traces``, the boundary is the
        timestamp of trace ``max_traces``; traces before it are kept, every trace at
        it is dropped, and ``end`` is cut back to it, so the next call can resume
        from ``end`` without skipping or repeating a trace. When one timestamp holds
        more traces than the cap, nothing is returned and ``end`` is ``start``.

        ``desc`` (newest first) reads newer chunks first, moving back until it holds
        more than ``max_traces`` or reaches ``start``, keeps the newest ``max_traces``
        and sets ``start`` just after the first trace it cut.

        Args:
            start: Window start (inclusive). A naive value is read as local
                time, as the SDK sends it.
            end: Window end (exclusive). A naive value is read as local time.
            name: Keep only traces with this name.
            tags: Keep only traces carrying all of these tags.
            environment: Keep only traces in this Langfuse environment.
            max_traces: Most traces to return.
            order: ``'asc'`` (oldest first) or ``'desc'`` (newest first).

        Returns:
            A :class:`TraceListing` whose ``start``/``end`` bound the window it lists
            completely and whose ``truncated`` says whether that is narrower than asked.
        """
        lo, hi = caller_utc(start), caller_utc(end)
        if order == 'desc':
            return self._list_newest(lo, hi, name, tags, environment, max_traces)

        rows, lo, hi = self._read_window(lo, hi, name, tags, environment, 'asc')
        shrunk = hi < caller_utc(end)
        traces = [s for s in self._summaries(rows, lo) if lo <= s.timestamp < hi]
        traces.sort(key=lambda s: (s.timestamp, s.id))
        if len(traces) <= max_traces:
            return TraceListing(traces, lo, hi, shrunk)

        boundary = traces[max_traces].timestamp
        kept = [s for s in traces if s.timestamp < boundary]
        if not kept:
            logger.warning(f'Listing cannot advance: {boundary} holds over the cap')
            return TraceListing([], lo, lo, True)
        logger.warning(
            f'Trace listing truncated at {max_traces} traces; window end cut to {boundary}'
        )
        return TraceListing(kept, lo, boundary, True)

    def _read_window(
        self,
        lo: datetime,
        hi: datetime,
        name: str | None,
        tags: list[str] | None,
        environment: str | None,
        keep: Literal['asc', 'desc'],
    ) -> tuple[list[Any], datetime, datetime]:
        """Rows of ``[lo, hi)``, halving it (keeping the older or newer half) to fit."""
        while True:
            try:
                return self._window_rows(lo, hi, name, tags, environment), lo, hi
            except TraceReadTruncated:
                if hi - lo <= self.min_list_window:
                    raise
                half = (hi - lo) / 2
                lo, hi = (lo, lo + half) if keep == 'asc' else (hi - half, hi)
                logger.warning(
                    f'Trace listing over its page cap; window narrowed to {lo}..{hi}'
                )

    def _list_newest(
        self,
        lo: datetime,
        hi: datetime,
        name: str | None,
        tags: list[str] | None,
        environment: str | None,
        max_traces: int,
    ) -> TraceListing:
        traces: list[TraceSummary] = []
        upper = hi
        while upper > lo and len(traces) < max_traces:
            rows, chunk_lo, _ = self._read_window(
                lo, upper, name, tags, environment, 'desc'
            )
            traces += [
                s
                for s in self._summaries(rows, chunk_lo)
                if chunk_lo <= s.timestamp < upper
            ]
            upper = chunk_lo
        traces.sort(key=lambda s: (s.timestamp, s.id), reverse=True)
        if len(traces) <= max_traces:
            return TraceListing(traces, upper, hi, upper > lo)
        # Traces tied with the first one cut go too, so the listing holds exactly
        # the traces of [start, end) and an older window resumed from start repeats
        # none of them.
        cut_at = traces[max_traces].timestamp
        kept = [s for s in traces[:max_traces] if s.timestamp > cut_at]
        start_after = cut_at + timedelta(microseconds=1)
        logger.warning(
            f'Trace listing cut at {max_traces} traces; complete from {start_after}'
        )
        return TraceListing(kept, start_after, hi, True)

    def _fetch_trace_summaries_via_list(
        self,
        limit: int,
        from_ts: datetime,
        to_ts: datetime | None,
        tags: list[str] | None,
        name: str | None,
        environment: str | None,
    ) -> list[TraceSummary]:
        """
        The newest ``limit`` trace summaries of the window.

        Serves the ``fetch_full_traces=False`` path of :meth:`fetch_traces`. A summary
        carries id, timestamp, name, session id and tags; no IO or metrics.
        """
        listing = self.list_trace_ids(
            from_ts,
            to_ts or datetime.now(UTC) + CLOCK_SLACK,
            name=name,
            tags=tags,
            environment=environment,
            max_traces=limit,
            order='desc',
        )
        if listing.truncated:
            logger.warning(
                f'Trace summaries hit limit {limit}; nothing older than '
                f'{listing.start} listed'
            )
        logger.info(f'Successfully loaded {len(listing.traces)} trace summaries')
        return list(listing.traces)

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
        environment: Optional[str] = None,
        trace_ids: Optional[List[str]] = None,
        fetch_full_traces: bool = True,
        show_progress: bool = True,
        **deprecated_kwargs: Any,
    ) -> List[Any]:
        """
        Fetch raw traces from Langfuse.

        Uses the observations endpoint to discover trace IDs, then fetches full
        traces by ID.

        Note on time window: the discovery step filters by *observation* start
        time, not by the parent trace timestamp. This is generally equivalent
        but may differ for long-running traces.

        Args:
            limit: Maximum number of traces to fetch.
            mode: Time window mode: 'days_back', 'hours_back', or 'absolute'.
            days_back: Number of days to look back (days_back mode).
            hours_back: Number of hours to look back (hours_back mode).
            from_timestamp: Start timestamp for absolute mode (datetime or ISO string).
            to_timestamp: End timestamp for absolute mode (datetime or ISO string).
            tags: Filter by trace tags (AND semantics — all tags must be present).
            name: Filter by trace name. With fetch_full_traces=False it is applied
                server-side. With fetch_full_traces=True it is
                applied as a post-filter on fetched traces, paginating through
                candidates until enough matches are found.
            environment: Filter by Langfuse environment (e.g. 'production').
            trace_ids: Fetch these specific IDs directly, bypassing discovery.
            fetch_full_traces: If True (default), return full trace objects via
                observations discovery + per-id fetch. If False, return lightweight
                :class:`TraceSummary` objects (id/timestamp/name/session/tags), the
                newest ``limit`` of the window.
            show_progress: Show a tqdm progress bar while fetching full traces.
            **deprecated_kwargs: Formerly passed to the trace list endpoint. No
                longer used; passing any value emits a deprecation warning.

        Returns:
            List of raw Langfuse trace objects (full traces), or
            :class:`TraceSummary` objects when fetch_full_traces=False.
        """
        if deprecated_kwargs:
            import warnings

            warnings.warn(
                f'fetch_traces() received unexpected keyword arguments '
                f'{list(deprecated_kwargs.keys())}. These were formerly passed to '
                f'the trace list endpoint, which has been replaced by the '
                f'observations API. '
                f'Unknown kwargs are ignored. Remove them to silence this warning.',
                DeprecationWarning,
                stacklevel=2,
            )

        if not self._client_initialized:
            logger.error('Langfuse client not initialized')
            return []

        def _fetch_full_traces(trace_id_list: List[str]) -> List[Any]:
            results = []
            trace_iter: Any = trace_id_list
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

        if mode != 'absolute':
            if from_timestamp is not None or to_timestamp is not None:
                raise ValueError('from_timestamp/to_timestamp require mode="absolute".')
            if mode == 'days_back':
                from_ts = self._normalize_time_window(days_back)
            else:
                from_ts = datetime.now() - timedelta(hours=hours_back)
            to_ts = None
        else:
            from_ts = _parse_timestamp(from_timestamp, 'from_timestamp')
            to_ts = _parse_timestamp(to_timestamp, 'to_timestamp')

            if from_ts is None:
                raise ValueError('absolute mode requires from_timestamp.')
            if to_ts and from_ts > to_ts:
                raise ValueError('from_timestamp cannot be after to_timestamp.')

        window_suffix = (
            f'between {from_ts} and {to_ts}' if to_ts else f'since {from_ts}'
        )
        logger.info(f'Fetching up to {limit} traces from Langfuse ({window_suffix})...')

        # Summary path: one windowed observations read with trace context, filtered
        # on trace name/tags server-side; no per-trace fetch.
        if not fetch_full_traces:
            return self._fetch_trace_summaries_via_list(
                limit=limit,
                from_ts=from_ts,
                to_ts=to_ts,
                tags=tags,
                name=name,
                environment=environment,
            )

        # When name filtering is requested we must post-filter full trace objects
        # because the observations API does not reliably index trace name.
        if name:
            full_traces: List[Any] = []
            seen_ids: set = set()
            cursor: Optional[str] = None

            while len(full_traces) < limit:
                page_ids, next_cursor = self._fetch_trace_id_page_via_observations(
                    batch_size=100,
                    environment=environment,
                    tags=tags,
                    from_ts=from_ts,
                    to_ts=to_ts,
                    cursor=cursor,
                )
                if not page_ids and not next_cursor:
                    break

                candidate_ids = [tid for tid in page_ids if tid not in seen_ids]
                seen_ids.update(candidate_ids)

                for trace in _fetch_full_traces(candidate_ids):
                    if getattr(trace, 'name', None) == name:
                        full_traces.append(trace)
                        if len(full_traces) >= limit:
                            break

                if not next_cursor:
                    break
                cursor = next_cursor

            logger.info(f'After name={name!r} filter: {len(full_traces)} traces')
            return full_traces[:limit]

        discovery_limit = limit
        trace_id_list = self._fetch_trace_ids_via_observations(
            limit=discovery_limit,
            environment=environment,
            tags=tags,
            from_ts=from_ts,
            to_ts=to_ts,
        )
        logger.info(f'Found {len(trace_id_list)} unique trace IDs via observations API')

        return _fetch_full_traces(trace_id_list)

    def _build_trace(
        self, trace_id: str, rows: Sequence[Any], scores: Sequence[Any]
    ) -> Any:
        from langfuse.api import TraceWithFullDetails
        from langfuse.api.core.pydantic_utilities import parse_obj_as

        return parse_obj_as(
            TraceWithFullDetails, full_trace_payload(trace_id, rows, scores)
        )

    def _trace_scores(self, trace_id: str, rows: Sequence[Any]) -> list[Any]:
        """Score rows of one trace; a failed read logs and gives none (the trace stays)."""
        try:
            return self._score_rows(
                trace_start(rows) or datetime.now(UTC) - ALL_TIME,
                f'fetch scores for trace {trace_id}',
                trace_id=trace_id,
            )
        except TraceReadTruncated:
            raise
        except Exception as e:
            logger.warning(f'Scores for trace {trace_id} could not be read: {e}')
            return []

    def fetch_trace(self, trace_id: str, pace: bool = True) -> Optional[Any]:
        """
        Fetch a single trace by trace_id, rebuilt from its observation rows.

        Args:
            trace_id: Langfuse trace ID to fetch
            pace: Whether to apply request pacing delay after fetch

        Returns:
            The full Langfuse trace object (``TraceWithFullDetails``, with its
            observations and scores), or None if not found/failed.

        Raises:
            TraceReadTruncated: The trace needs more than ``max_pages`` pages;
                a partial trace is never returned.
        """
        if not self._client_initialized:
            logger.error('Langfuse client not initialized')
            return None

        if not trace_id:
            logger.warning('fetch_trace called with empty trace_id')
            return None

        try:
            rows = self._observations(f'fetch trace {trace_id}', trace_id=trace_id)
            if not rows:
                logger.warning(f'Trace {trace_id} not found')
                return None
            trace = self._build_trace(
                trace_id, rows, self._trace_scores(trace_id, rows)
            )
        except TraceReadTruncated:
            raise
        except Exception as e:
            logger.warning(f'Failed to fetch trace {trace_id}: {e}')
            return None

        self._pace(pace)
        return trace

    def list_score_rows(
        self,
        *,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> 'list[TraceScore]':
        """
        Scores of a session or a trace, optionally only those named ``name``.

        The read's window runs from a day before the earliest trace start of the
        session or trace to now. An unknown session or trace has no scores. Unlike
        :meth:`fetch_scores_for_session` and :meth:`fetch_scores_for_trace`, errors
        propagate, so a caller can tell "the read failed" from "no scores".

        Args:
            session_id: Langfuse session ID (exactly one of this and ``trace_id``).
            trace_id: Langfuse trace ID.
            name: Keep only scores with this name.

        Returns:
            List of :class:`TraceScore` objects.
        """
        if bool(session_id) == bool(trace_id):
            raise ValueError(
                'list_score_rows needs exactly one of session_id, trace_id'
            )
        what = f'session {session_id}' if session_id else f'trace {trace_id}'
        filters: dict[str, Any] = (
            {'session_id': session_id} if session_id else {'trace_id': trace_id}
        )
        rows = self._observations(f'read {what}', fields='core,basic', **filters)
        starts = [trace_start(group) for group in group_by_trace(rows).values()]
        known = [s for s in starts if s is not None]
        if not known:
            return []
        if name:
            filters['name'] = name
        raws = self._score_rows(min(known), f'fetch scores for {what}', **filters)
        return [score_from_v3(raw) for raw in raws]

    def fetch_scores_for_session(
        self, session_id: str, pace: bool = True
    ) -> 'dict[str, list[TraceScore]]':
        """
        Fetch a session's Langfuse eval scores, keyed by trace.

        Covers scores written against the session's traces and observations, and
        session-level scores (keyed by ``''``). Returns a dict keyed by
        ``trace_id`` so callers can attach scores to each ``Trace`` without
        issuing a per-trace request. On any error the warning is logged and an
        empty dict is returned (fail-soft — a missing scores fetch must never drop
        the item from the pipeline).

        Args:
            session_id: Langfuse session ID.
            pace: Whether to apply request pacing after the read.

        Returns:
            ``{trace_id: [TraceScore, ...]}`` — empty dict on failure or no scores.
        """
        if not self._client_initialized or not session_id:
            logger.warning(f'fetch_scores_for_session skipped (session={session_id!r})')
            return {}

        try:
            scores = self.list_score_rows(session_id=session_id)
        except TraceReadTruncated:
            raise
        except Exception as e:
            logger.warning(
                f'fetch_scores_for_session failed for session {session_id}: {e}'
            )
            return {}

        self._pace(pace)
        result: dict[str, list[TraceScore]] = {}
        for score in scores:
            result.setdefault(score.trace_id or '', []).append(score)
        return result

    def fetch_scores_for_trace(
        self, trace_id: str, pace: bool = True
    ) -> 'list[TraceScore]':
        """
        Fetch Langfuse eval scores for a single trace.

        Prefer :meth:`fetch_scores_for_session` when processing a whole session
        to avoid N per-trace requests. On any error the warning is logged and an
        empty list is returned.

        Args:
            trace_id: Langfuse trace ID.
            pace: Whether to apply request pacing after the read.

        Returns:
            List of :class:`TraceScore` objects; empty list on failure or no scores.
        """
        if not self._client_initialized or not trace_id:
            logger.warning(f'fetch_scores_for_trace skipped (trace={trace_id!r})')
            return []

        try:
            scores = self.list_score_rows(trace_id=trace_id)
        except TraceReadTruncated:
            raise
        except Exception as e:
            logger.warning(f'fetch_scores_for_trace failed for trace {trace_id}: {e}')
            return []

        self._pace(pace)
        return scores

    def _session_groups(
        self, session_id: str, fields: str
    ) -> list[tuple[str, list[Any]]]:
        """
        The session's traces as ``(trace_id, rows)``, ordered by start then id.

        The session filter finds the traces; their rows are then read by trace id,
        since a child row need not carry the session id its root does.
        """
        what = f'fetch session {session_id}'
        found = self._observations(what, fields=LIGHT_FIELDS, session_id=session_id)
        trace_ids = list(dict.fromkeys(r.trace_id for r in found if r.trace_id))
        rows: list[Any] = []
        for i in range(0, len(trace_ids), 100):
            condition = {
                'type': 'stringOptions',
                'column': 'traceId',
                'operator': 'any of',
                'value': trace_ids[i : i + 100],
            }
            rows += self._observations(
                what, fields=fields, filter=json.dumps([condition])
            )
        groups = list(group_by_trace(rows).items())
        groups.sort(key=lambda g: (trace_start(g[1]) or EPOCH, g[0]))
        return groups

    @staticmethod
    def _session_payload(
        session_id: str, groups: Sequence[tuple[str, list[Any]]]
    ) -> Any:
        from langfuse.api import SessionWithTraces, Trace
        from langfuse.api.core.pydantic_utilities import parse_obj_as

        stubs = []
        for tid, rows in groups:
            stub = trace_payload(tid, rows)
            try:
                parse_obj_as(Trace, stub)
            except Exception as e:
                # One trace that won't rebuild must not take the session down.
                logger.warning(f'Dropping trace {tid} from session {session_id}: {e}')
                continue
            stubs.append(stub)
        first = pick_root(groups[0][1])
        return parse_obj_as(
            SessionWithTraces,
            {
                'id': session_id,
                'createdAt': min(trace_start(rows) or EPOCH for _, rows in groups),
                'projectId': first.project_id or '',
                'environment': first.environment or 'default',
                'traces': stubs,
            },
        )

    def fetch_session(self, session_id: str, pace: bool = True) -> Optional[Any]:
        """
        Fetch a session by session_id.

        Args:
            session_id: Langfuse session ID to fetch
            pace: Whether to apply request pacing delay after fetch

        Returns:
            The Langfuse session object (``SessionWithTraces``; its traces carry
            trace-level fields, not observations, in start order), or None if not
            found/failed.

        Raises:
            TraceReadTruncated: The session needs more than ``max_pages`` pages.
        """
        if not self._client_initialized:
            logger.error('Langfuse client not initialized')
            return None

        if not session_id:
            logger.warning('fetch_session called with empty session_id')
            return None

        try:
            groups = self._session_groups(session_id, FULL_FIELDS)
        except TraceReadTruncated:
            raise
        except Exception as e:
            logger.warning(f'Failed to fetch session {session_id}: {e}')
            return None

        self._pace(pace)
        return self._session_payload(session_id, groups) if groups else None

    def get_session_traces(
        self,
        session_id: str,
        show_progress: bool = True,
    ) -> List[Any]:
        """
        Fetch all full trace data for a session.

        Traces that fail to rebuild are dropped (see
        :meth:`get_session_with_traces` to retain stubs instead).

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
        trace_name: Optional[str] = None,
        trace_predicate: Optional[Callable[[Any], bool]] = None,
    ) -> tuple[Optional[Any], List[Any]]:
        """
        Fetch a session along with trace details for each of its traces.

        The session's observation rows are read once and grouped by trace; the
        traces come back in start order.

        Args:
            session_id: Langfuse session ID
            show_progress: Whether to show a progress bar while reading each
                trace's scores
            retain_stub_on_failure: When a full trace fails to rebuild, keep its
                stub trace instead of dropping it. Stubs still carry trace-level
                input/output (so a conversation turn is preserved), only
                observation-level detail is missing. Defaults to ``True``.
            enrich: When ``True`` (default), return full traces with their
                observations and scores (one score read per trace). When
                ``False``, return the stub traces: trace-level input/output only,
                so ``by_type``/``tools``/``find_all`` will be empty.
            trace_name: When given, keep only traces whose ``name`` equals this
                value, decided on the stubs before any score is read.
            trace_predicate: When given, keep only traces for which this
                ``(stub) -> bool`` returns ``True``, decided on the stubs.
                Combined with ``trace_name`` (both must pass).

        Returns:
            A ``(session, traces)`` tuple. ``session`` is the raw Langfuse
            session object (or ``None`` if it is unknown or could not be read),
            and ``traces`` is the list of full (or, when not enriching / on
            failure, stub) trace objects.

        Raises:
            TraceReadTruncated: The session needs more than ``max_pages`` pages.
        """
        if not self._client_initialized:
            logger.error('Langfuse client not initialized')
            return None, []

        if not session_id:
            logger.warning('get_session_with_traces called with empty session_id')
            return None, []

        try:
            groups = self._session_groups(session_id, FULL_FIELDS)
        except TraceReadTruncated:
            raise
        except Exception as e:
            logger.warning(f'Failed to fetch session {session_id}: {e}')
            return None, []

        if not groups:
            return None, []

        session = self._session_payload(session_id, groups)
        stubs = self._filter_stub_traces(session.traces, trace_name, trace_predicate)
        if not stubs:
            logger.info(f'No traces in session {session_id} matched the trace filter.')
            return session, []

        if not enrich:
            return session, list(stubs)

        stub_by_id = {s.id: s for s in stubs}
        kept = [(tid, rows) for tid, rows in groups if tid in stub_by_id]
        trace_iter: Any = kept
        if show_progress:
            trace_iter = tqdm(kept, desc='Fetching session traces', unit='trace')

        results = []
        for tid, rows in trace_iter:
            try:
                results.append(
                    self._build_trace(tid, rows, self._trace_scores(tid, rows))
                )
            except TraceReadTruncated:
                raise
            except Exception as e:
                logger.warning(f'Trace {tid} would not rebuild: {e}')
                if retain_stub_on_failure:
                    results.append(stub_by_id[tid])

        logger.info(
            f'Successfully loaded {len(results)} traces for session {session_id}'
        )
        return session, results

    @staticmethod
    def _filter_stub_traces(
        stubs: List[Any],
        trace_name: Optional[str],
        trace_predicate: Optional[Callable[[Any], bool]],
    ) -> List[Any]:
        """
        Filter session stub traces by name/predicate before enrichment.

        Applied to the cheap stubs so non-matching traces are never fetched.
        """
        if trace_name is None and trace_predicate is None:
            return list(stubs)

        kept = []
        for stub in stubs:
            if trace_name is not None and getattr(stub, 'name', None) != trace_name:
                continue
            if trace_predicate is not None and not trace_predicate(stub):
                continue
            kept.append(stub)
        return kept

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

        Each item is written as its own trace. The trace's root span carries the
        experiment attributes Langfuse reads to link it to the dataset item and the
        experiment, and the item's scores go on that root observation. The
        experiment id is derived from the project, the dataset and ``run_name``, so
        uploads under the same run name add to the same experiment.

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
            link_to_traces: If True, also attach each item's scores to its existing
                runtime trace (and observation, when set). The experiment item is
                still written as its own trace, since Langfuse cannot make an existing
                trace an experiment item; the runtime ids are recorded in the item's
                experiment metadata. Ignored if score_on_runtime_traces is True.
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
                - runs_created: Number of experiment item traces written
                - scores_uploaded: Number of scores attached
                - experiment_id: The experiment (and dataset run) id, or None
                  when no experiment was written
                - run_url: Link to the experiment run in Langfuse, or None when
                  no experiment was written or the project or dataset id could
                  not be read
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
                'experiment_id': None,
                'run_url': None,
                'errors': ['Langfuse client not initialized'],
            }

        stats: Dict[str, Any] = {
            'dataset_name': None,
            'run_name': None,
            'items_created': 0,
            'runs_created': 0,
            'scores_uploaded': 0,
            'scores_skipped': 0,
            'experiment_id': None,
            'run_url': None,
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
                'expected_output': expected_output,
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

        # Each item is written as its own trace whose root span carries the
        # experiment attributes; Langfuse groups those traces into the run.
        dataset_id = getattr(dataset, 'id', None)
        project_id = self._project_id_or_none()
        experiment_id = experiment_id_for(project_id, dataset_id, run_name)
        if experiment_id is None:
            experiment_id = secrets.token_hex(8)
            error_msg = (
                f'Could not resolve the Langfuse project or dataset id; experiment '
                f'{run_name!r} gets a random id, so a re-run under the same name '
                f'starts a new experiment'
            )
            logger.warning(error_msg)
            stats['errors'].append(error_msg)

        experiment_metadata: Dict[str, Any] = {
            'axion_run_id': evaluation_result.run_id,
            **(run_metadata or {}),
        }
        if effective_tags:
            experiment_metadata['tags'] = effective_tags

        for dataset_item in dataset.items:
            item_id = dataset_item.id
            if item_id not in test_result_map:
                # This item wasn't part of our evaluation
                continue

            cached = test_result_map[item_id]
            test_result = cached['test_result']
            item_trace_id = getattr(test_result.test_case, 'trace_id', None)
            item_observation_id = getattr(test_result.test_case, 'observation_id', None)
            # The dataset's own expected output is what the experiment compares
            # against; it can differ from the local one when items are not upserted.
            expected_output = getattr(dataset_item, 'expected_output', None)
            if expected_output is None:
                expected_output = cached['expected_output']
            item_metadata = {
                **({'trace_id': item_trace_id} if item_trace_id else {}),
                **(
                    {'observation_id': item_observation_id}
                    if item_observation_id
                    else {}
                ),
            }
            scores = [
                metric_score
                for metric_score in test_result.score_results
                if self._should_upload_metric(
                    metric_score, metric_name_filter, stats, 'scores_skipped'
                )
            ]

            try:
                trace_id, root_observation_id = self._write_experiment_item(
                    name=f'dataset-run:{run_name}:{item_id}',
                    input_data=cached.get('input_data'),
                    output=cached['actual_output'],
                    attributes=experiment_item_attributes(
                        experiment_id=experiment_id,
                        run_name=run_name,
                        dataset_id=dataset_id,
                        item_id=item_id,
                        expected_output=expected_output,
                        experiment_metadata=experiment_metadata,
                        item_metadata=item_metadata,
                    ),
                )
                stats['runs_created'] += 1
            except Exception as e:
                error_msg = f'Failed to create experiment run for item {item_id}: {e}'
                logger.warning(error_msg)
                stats['errors'].append(error_msg)
                continue

            # Scores go on the item's root observation, after its span has closed.
            targets = [(trace_id, root_observation_id)]
            if link_to_traces and item_trace_id:
                targets.append((item_trace_id, item_observation_id))
            for target_trace_id, target_observation_id in targets:
                self._upload_item_scores(
                    scores,
                    trace_id=target_trace_id,
                    observation_id=target_observation_id,
                    item_id=item_id,
                    stats=stats,
                )

            if self.request_pacing > 0:
                time.sleep(self.request_pacing)

        # The experiment exists only once an item trace is written, so an upload
        # that wrote none returns no id and no link.
        if stats['runs_created']:
            stats['experiment_id'] = experiment_id
            stats['run_url'] = self._experiment_run_url(
                project_id, dataset_id, experiment_id
            )

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

    def _project_id_or_none(self) -> Optional[str]:
        """The Langfuse project id for these keys, or None when it can't be read."""
        try:
            project_id = self.client._get_project_id()
        except Exception as e:
            logger.warning(f'Could not read the Langfuse project id: {e}')
            return None
        return str(project_id) if project_id else None

    def _experiment_run_url(
        self,
        project_id: Optional[str],
        dataset_id: Optional[str],
        experiment_id: str,
    ) -> Optional[str]:
        """Link to the experiment's dataset run; its run id is the experiment id."""
        base = getattr(self.client, '_base_url', None)
        if not (isinstance(base, str) and base and project_id and dataset_id):
            return None
        base = base.rstrip('/')
        return f'{base}/project/{project_id}/datasets/{dataset_id}/runs/{experiment_id}'

    def _write_experiment_item(
        self,
        *,
        name: str,
        input_data: Any,
        output: Any,
        attributes: Dict[str, str],
    ) -> tuple[str, str]:
        """Write one experiment item as its own trace; return (trace id, root id).

        The span is started under an empty context so it is a trace root even
        when the caller has a span open; Langfuse needs one trace per item.
        """
        from opentelemetry import context as otel_context
        from opentelemetry import trace as otel_trace

        token = otel_context.attach(otel_context.Context())
        try:
            with self.client.start_as_current_observation(
                name=name,
                as_type='evaluator',
                input=input_data,
                output=output,
            ) as root_span:
                root_span.set_trace_io(input=input_data, output=output)
                otel_trace.get_current_span().set_attributes(
                    {
                        **attributes,
                        EXPERIMENT_ITEM_ROOT_OBSERVATION_ID: root_span.id,
                    }
                )
                return root_span.trace_id, root_span.id
        finally:
            otel_context.detach(token)

    def _upload_item_scores(
        self,
        scores: List[Any],
        *,
        trace_id: str,
        observation_id: Optional[str],
        item_id: str,
        stats: Dict[str, Any],
    ) -> None:
        """Attach an item's scores to one trace (and observation, when given)."""
        for metric_score in scores:
            score_kwargs: Dict[str, Any] = {
                'trace_id': trace_id,
                'name': metric_score.name,
                'value': float(metric_score.score),
                'comment': self._format_comment(
                    metric_score.explanation, metric_score.signals
                ),
            }
            if observation_id:
                score_kwargs['observation_id'] = observation_id
            try:
                self._execute_with_retry(
                    lambda kwargs=score_kwargs: self.client.create_score(**kwargs),
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
