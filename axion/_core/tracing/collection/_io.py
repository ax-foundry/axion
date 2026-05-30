"""Shared, pure helpers for extracting input/output and timestamps from raw
trace/observation/session payloads.

These live here (rather than on a collection class) so that ``trace.py``,
``trace_collection.py``, ``session.py``, and ``session_collection.py`` can all
reuse them without importing each other or reaching into private statics.

Symbols imported by other modules are public (no leading underscore); helpers
used only within this module stay underscore-prefixed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

# Keys searched (in priority order) when pulling a query/output string out of a
# structured payload. Kept module-level so callers can reuse the exact same
# precedence the trace collection uses.
QUERY_KEYS = ('query', 'question', 'input', 'message', 'prompt', 'user_input', 'text')
OUTPUT_KEYS = ('output', 'response', 'answer', 'result', 'content', 'text', 'message')

# UTC-aware sentinel for missing/invalid timestamps. Using an aware value keeps
# every comparison in a single timezone policy so sorts never mix naive/aware.
_TS_SENTINEL = datetime.min.replace(tzinfo=timezone.utc)

# Timestamp-like keys, snake_case first then camelCase, so a dict/object using
# either Langfuse convention (created_at vs createdAt) sorts correctly.
_TS_KEYS = (
    'timestamp',
    'created_at',
    'createdAt',
    'start_time',
    'startTime',
)


def safe_json_load(data: Any) -> Any:
    """Best-effort JSON decode: parse a JSON string, else return as-is."""
    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data
    return data


def _extract_by_keys(payload: Any, keys: tuple[str, ...]) -> str:
    """Extract text from a payload by prioritized keys, with safe fallbacks."""
    text, _ = extract_text_with_match(payload, keys)
    return text


def extract_text_with_match(payload: Any, keys: tuple[str, ...]) -> tuple[str, bool]:
    """
    Extract text from a payload, reporting whether a real key matched.

    Returns ``(text, matched)`` where *matched* is ``True`` only when the value
    came from a string payload or a recognised key in a dict. When a dict has no
    matching key we fall back to ``json.dumps(dict)`` and report ``matched=False``
    so callers can distinguish "found a real message field" from "dumped a blob".
    """
    data = safe_json_load(payload)
    if isinstance(data, str):
        return data, True
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                return str(data[key]), True
        return json.dumps(data), False
    return str(data), False


def extract_query(input_data: Any) -> str:
    return _extract_by_keys(input_data, QUERY_KEYS)


def extract_output(output_data: Any) -> str:
    return _extract_by_keys(output_data, OUTPUT_KEYS)


def coerce_ts(value: Any) -> datetime:
    """
    Normalize a timestamp value to a UTC-aware ``datetime``.

    - ``datetime``: naive values are assumed UTC; aware values are converted.
    - ISO 8601 strings: parsed (``Z`` handled), then normalized as above.
    - Anything missing/unparseable: the UTC-aware sentinel ``_TS_SENTINEL``.

    Never returns a naive datetime, so callers can sort freely.
    """
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            return _TS_SENTINEL
        return _as_utc(parsed)
    return _TS_SENTINEL


def _as_utc(dt: datetime) -> datetime:
    """Assume-UTC for naive datetimes; convert aware datetimes to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _get_timestamp_value(src: dict) -> Any:
    """Return the first present timestamp-like key from a dict payload."""
    for k in _TS_KEYS:
        if k in src:
            return src[k]
    return None


def extract_trace_io(trace_obj: Any) -> tuple[Any, Any, Any, Any]:
    """
    Extract raw ``(input, output, id, timestamp)`` from supported shapes.

    Handles SmartAccess-backed views (``TraceView``/``ObservationsView`` expose a
    ``_data`` dict), plain dicts, and generic SDK objects with attributes.
    """
    # Local import to avoid a circular import at module load time.
    from axion._core.tracing.collection.models import ObservationsView, TraceView

    if isinstance(trace_obj, (TraceView, ObservationsView)):
        data = trace_obj._data
        return (
            data.get('input'),
            data.get('output'),
            data.get('id'),
            _get_timestamp_value(data),
        )

    if isinstance(trace_obj, dict):
        return (
            trace_obj.get('input'),
            trace_obj.get('output'),
            trace_obj.get('id'),
            _get_timestamp_value(trace_obj),
        )

    ts = None
    for k in _TS_KEYS:
        ts = getattr(trace_obj, k, None)
        if ts is not None:
            break

    return (
        getattr(trace_obj, 'input', None),
        getattr(trace_obj, 'output', None),
        getattr(trace_obj, 'id', None),
        ts,
    )
