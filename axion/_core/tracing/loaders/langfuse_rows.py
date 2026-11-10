"""
Rebuild Langfuse traces, sessions and scores from observation and score rows.

Langfuse's observations read returns one row per observation and has no endpoint
for a whole trace or a session's traces. The functions here turn those rows back
into the SDK's ``TraceWithFullDetails`` / ``SessionWithTraces`` payloads, and the
scores read's rows into :class:`TraceScore`, so ``Trace``, ``Session`` and every
consumer read the same attribute names as before.

Row rules:

- A trace's fields come from its root row, picked in code (:func:`pick_root`).
  Reads never filter on ``isRootObservation``: it is false for a root whose parent
  id names an observation that does not exist.
- A trace's timestamp is :func:`trace_start`: the earliest flagged root (a retried
  turn leaves one per attempt), else the picked root's start.
- Input and output come back as raw strings; :func:`parse_io` decodes them.
- A trace's observations are in start order (:func:`chronological`); the read's own
  row order is not stable.

Metadata keeps the observations read's shape: an object-valued entry comes back as
dotted keys (``conversation.messages`` rather than ``conversation: {messages}``), and
trace-level keys may repeat on child rows. Flat and nested writers return the same rows,
so the nested shape cannot be rebuilt exactly.

The read cuts every top-level metadata value to 200 characters with no marker, so a
string of 200 or more characters counts as possibly cut (:func:`possibly_cut`) and
the loader reads those keys again with ``expand_metadata``.
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from axion._core.tracing.collection.scores import TraceScore

__all__ = [
    'TraceListing',
    'TraceReadTruncated',
    'TraceSummary',
    'parse_io',
    'pick_root',
    'score_from_v3',
    'trace_start',
]

FULL_FIELDS = 'core,basic,time,io,metadata,model,usage,prompt,metrics,trace_context'
LIGHT_FIELDS = 'core,basic,trace_context'
SCORE_FIELDS = 'core,subject,details'
OBSERVATIONS_PAGE_LIMIT = 1000
SCORES_PAGE_LIMIT = 100
METADATA_CUT_CHARS = 200
# The scores read needs a window. A score is written after its trace, sometimes days
# later, and a backfilled or skewed timestamp can sit a little before it.
SCORE_LOOKBACK = timedelta(days=1)
CLOCK_SLACK = timedelta(minutes=5)
# Reads by trace or session id still take a start-time window; this one is
# unbounded in practice.
ALL_TIME = timedelta(days=3650)

EPOCH = datetime.min.replace(tzinfo=UTC)
_TRACE_SCORE_TYPES = frozenset({'NUMERIC', 'CATEGORICAL', 'BOOLEAN', 'TEXT'})
# These types carry no usage or cost on a trace payload. The observations read
# reports the usage a root SPAN was given, which would count a turn's tokens twice.
_NO_USAGE_TYPES = frozenset({'SPAN', 'EVENT'})


class TraceReadTruncated(RuntimeError):
    """A read needed more pages than the loader allows; its result would be partial."""


@dataclass(frozen=True)
class TraceSummary:
    """A trace in a window listing: no IO and no observations; fetch it by id for those."""

    id: str
    timestamp: datetime
    name: str | None
    session_id: str | None
    tags: list[str]


@dataclass(frozen=True)
class TraceListing:
    """The traces of ``[start, end)``, complete for that window (which may be cut back)."""

    traces: list[TraceSummary]
    start: datetime
    end: datetime
    truncated: bool


# --- Rows -----------------------------------------------------------------------


def utc(moment: datetime) -> datetime:
    """Aware UTC; a naive value is read as UTC, as the API reads it."""
    return (
        moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
    )


def caller_utc(moment: datetime) -> datetime:
    """A caller's datetime as aware UTC; a naive value is read as local time.

    The SDK sends a naive datetime in the runtime's local timezone, so a window
    built from ``datetime.now()`` means the same instants here as it does there.
    """
    return moment.astimezone(UTC)


def _start(row: Any) -> datetime:
    value = getattr(row, 'start_time', None)
    return utc(value) if isinstance(value, datetime) else EPOCH


def pick_root(rows: Sequence[Any]) -> Any:
    """The row that stands for its trace.

    The row flagged ``is_root_observation`` wins, the latest start when a retry left
    several; failing that, a row with no parent; failing that, a row whose parent is
    not among ``rows``. When no row qualifies, the earliest stands in.

    Among flagged rows, one with no parent beats one under a remote parent. A trace
    written both as a trace record and as a span whose parent lives in a calling
    service has two flagged rows: the trace record (``t-<trace id>``, no parent)
    holds the trace's metadata and tags, and the span holds only its own.
    """
    flagged = [r for r in rows if getattr(r, 'is_root_observation', None)]
    if flagged:
        own = [r for r in flagged if not getattr(r, 'parent_observation_id', None)]
        return max(own or flagged, key=_start)
    parentless = [r for r in rows if not getattr(r, 'parent_observation_id', None)]
    known = {getattr(r, 'id', None) for r in rows}
    orphaned = [
        r for r in rows if getattr(r, 'parent_observation_id', None) not in known
    ]
    return min(parentless or orphaned or list(rows), key=_start)


def is_true_root(row: Any) -> bool:
    """
    Whether ``row`` is its trace's own root rather than a stand-in for a missing one.

    A root is flagged, has no parent, or carries the trace's name (or the trace has
    none). The last covers a root whose parent span lives in a calling service and so
    was never recorded here.
    """
    trace_name = getattr(row, 'trace_name', None)
    return bool(
        getattr(row, 'is_root_observation', None)
        or not getattr(row, 'parent_observation_id', None)
        or not trace_name
        or row.name == trace_name
    )


def trace_start(rows: Sequence[Any]) -> datetime | None:
    """When the trace began: the earliest flagged root, else the picked root's start."""
    flagged = [r for r in rows if getattr(r, 'is_root_observation', None)]
    pool = flagged or [pick_root(rows)]
    starts = [
        _start(r) for r in pool if isinstance(getattr(r, 'start_time', None), datetime)
    ]
    return min(starts) if starts else None


def parse_io(value: Any) -> Any:
    """Decode an input or output returned as a raw JSON string; keep anything else."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def possibly_cut(value: Any) -> bool:
    """Whether a metadata value may have been cut to the read's 200-character limit."""
    return isinstance(value, str) and len(value) >= METADATA_CUT_CHARS


def _sum_prefixed(details: dict[str, Any] | None, prefix: str) -> Any:
    values = [v for k, v in (details or {}).items() if k.startswith(prefix) and v]
    return sum(values) if values else None


def _legacy_usage(details: dict[str, int] | None) -> dict[str, Any]:
    """The legacy ``usage`` block: input and output sum their prefixed usage keys."""
    details = details or {}
    inp = _sum_prefixed(details, 'input') or 0
    out = _sum_prefixed(details, 'output') or 0
    return {
        'unit': 'TOKENS',
        'input': inp,
        'output': out,
        'total': details.get('total', inp + out),
    }


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _text(value: str | None) -> str | None:
    """The observations read returns ``""`` for an unset string; payloads use null."""
    return value or None


def observation_payload(row: Any) -> dict[str, Any]:
    """One observation row as an ``ObservationsView`` payload."""
    priced = row.type not in _NO_USAGE_TYPES
    usage = (row.usage_details or {}) if priced else {}
    costs = (row.cost_details or {}) if priced else {}
    return {
        'id': row.id,
        'traceId': row.trace_id,
        'type': row.type,
        'name': row.name,
        'startTime': row.start_time,
        'endTime': row.end_time,
        'completionStartTime': row.completion_start_time,
        'model': _text(row.model),
        'modelParameters': row.model_parameters or ({} if priced else None),
        'input': parse_io(row.input),
        'output': parse_io(row.output),
        'version': _text(row.version),
        'metadata': row.metadata,
        'usage': _legacy_usage(usage),
        'level': row.level or 'DEFAULT',
        'statusMessage': _text(row.status_message),
        'parentObservationId': row.parent_observation_id,
        'promptId': _text(row.prompt_id),
        'promptName': _text(row.prompt_name),
        'promptVersion': row.prompt_version,
        'usageDetails': usage,
        'costDetails': costs,
        'environment': row.environment or 'default',
        'modelId': _text(row.model_id),
        'inputPrice': _float(row.input_price),
        'outputPrice': _float(row.output_price),
        'totalPrice': _float(row.total_price),
        'calculatedInputCost': _sum_prefixed(costs, 'input'),
        'calculatedOutputCost': _sum_prefixed(costs, 'output'),
        'calculatedTotalCost': row.total_cost if priced else None,
        'latency': row.latency,
        'timeToFirstToken': row.time_to_first_token,
    }


def trace_payload(trace_id: str, rows: Sequence[Any]) -> dict[str, Any]:
    """
    The trace-level fields of ``trace_id``, from its root row (no observations).

    Input and output come only from a true root (:func:`is_true_root`): a trace whose
    root was never recorded has no input or output, not an orphaned child's. Rows
    written without a trace name take the root's own name, as the trace read did.
    """
    root = pick_root(rows)
    own = is_true_root(root)
    return {
        'id': trace_id,
        'timestamp': trace_start(rows),
        'name': _text(root.trace_name) or _text(root.name),
        'input': parse_io(root.input) if own else None,
        'output': parse_io(root.output) if own else None,
        'sessionId': _text(root.session_id),
        'release': _text(root.release),
        'version': _text(root.version),
        'userId': _text(root.user_id),
        'metadata': root.metadata,
        'tags': sorted(root.tags or []),
        'public': bool(root.public),
        'environment': root.environment or 'default',
    }


def chronological(rows: Iterable[Any]) -> list[Any]:
    """Rows by start time, ties broken on id, so observation order is stable."""
    return sorted(rows, key=lambda r: (r.start_time is None, r.start_time or 0, r.id))


def full_trace_payload(
    trace_id: str, rows: Sequence[Any], scores: Sequence[Any]
) -> dict[str, Any]:
    """A ``TraceWithFullDetails`` payload: trace fields, observations and scores."""
    payload = trace_payload(trace_id, rows)
    root = pick_root(rows)
    costs = [r.total_cost for r in rows if r.total_cost is not None]
    payload.update(
        htmlPath=f'/project/{root.project_id}/traces/{trace_id}',
        latency=root.latency,
        totalCost=float(sum(costs)),
        observations=[observation_payload(r) for r in chronological(rows)],
        scores=[
            trace_score_payload(s, trace_id)
            for s in scores
            if s.data_type in _TRACE_SCORE_TYPES
        ],
    )
    return payload


def group_by_trace(rows: Iterable[Any]) -> dict[str, list[Any]]:
    """Rows by trace id, in first-seen order, each trace keeping the read's row order."""
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        if row.trace_id:
            grouped.setdefault(row.trace_id, []).append(row)
    return grouped


# --- Scores ---------------------------------------------------------------------


def _subject_ids(raw: Any) -> tuple[str | None, str | None]:
    subject = getattr(raw, 'subject', None)
    kind = getattr(subject, 'kind', None)
    if kind == 'trace':
        return subject.id, None
    if kind == 'observation':
        return getattr(subject, 'trace_id', None), subject.id
    return None, None


def _value(raw: Any) -> tuple[float, str | None]:
    """The ``(value, string_value)`` pair :class:`TraceScore` holds."""
    if raw.data_type == 'NUMERIC':
        return float(raw.value), None
    if raw.data_type == 'BOOLEAN':
        return (1.0 if raw.value else 0.0), ('True' if raw.value else 'False')
    return 0.0, str(raw.value)


def score_from_v3(raw: Any) -> TraceScore:
    """One scores-read row as a :class:`TraceScore`, its ids taken from its subject."""
    trace_id, observation_id = _subject_ids(raw)
    value, string_value = _value(raw)
    return TraceScore(
        name=raw.name,
        value=value,
        data_type=str(raw.data_type),
        trace_id=trace_id,
        observation_id=observation_id,
        string_value=string_value,
        comment=getattr(raw, 'comment', None),
        source=str(raw.source) if getattr(raw, 'source', None) is not None else None,
        timestamp=getattr(raw, 'timestamp', None),
    )


def trace_score_payload(raw: Any, trace_id: str) -> dict[str, Any]:
    """One scores-read row as an entry of a trace payload's ``scores``."""
    _, observation_id = _subject_ids(raw)
    value, string_value = _value(raw)
    payload = {
        'dataType': raw.data_type,
        'value': value,
        'id': raw.id,
        'traceId': trace_id,
        'name': raw.name,
        'source': raw.source,
        'observationId': observation_id,
        'timestamp': raw.timestamp,
        'createdAt': raw.created_at,
        'updatedAt': raw.updated_at,
        'authorUserId': getattr(raw, 'author_user_id', None),
        'comment': getattr(raw, 'comment', None),
        'metadata': getattr(raw, 'metadata', None),
        'configId': getattr(raw, 'config_id', None),
        'queueId': getattr(raw, 'queue_id', None),
        'environment': raw.environment,
    }
    if raw.data_type == 'TEXT':
        payload.pop('value')
    if string_value is not None:
        payload['stringValue'] = string_value
    return payload
